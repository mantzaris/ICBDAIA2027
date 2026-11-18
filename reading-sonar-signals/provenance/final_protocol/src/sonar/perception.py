"""Small CUDA scene-classification baselines on weak recording labels.

P1 has one LOFAR branch; P2 adds an independent DEMON branch. They are not
localization models and do not assign a vessel label to individual components.
All numerical features, model initialization, training and metrics run on CUDA.
"""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from .signal import require_cuda, lofar, demon


FEATURE_CONFIG = {
    "shape": [2, 128, 128],
    "lofar_window_s": 1.0,
    "lofar_hop_s": 0.25,
    "lofar_band_hz": [0.0, 1800.0],
    "demon_carrier_band_hz": [2000.0, 5000.0],
    "demon_window_s": 4.0,
    "demon_hop_s": 0.5,
    "demon_modulation_band_hz": [0.5, 200.0],
    "interpolation": "bilinear_align_corners_false",
    "power_floor": 1e-20,
    "normalization": "branch_global_mean_std_fitted_to_training_scenes_only",
}


def waveform_features(x: torch.Tensor, sample_rate: float,
                      offset_s: float = 0.0) -> torch.Tensor:
    """Return unnormalized CUDA [LOFAR, DEMON, 128, 128] log-PSD images.

    Resizing changes the CNN input grid only. Physical measurements use the
    original SpectralView grids and are not made from these classifier inputs.
    Missing bandwidth raises an error rather than being silently zero-filled.
    """
    require_cuda(x)
    acoustic = lofar(x, sample_rate, window_s=1., hop_s=.25, offset_s=offset_s)
    modulation = demon(x, sample_rate, (2000., 5000.), window_s=4.,
                       hop_s=.5, offset_s=offset_s)
    return torch.cat([
        F.interpolate(v.log_power()[None, None], size=(128, 128),
                      mode="bilinear", align_corners=False)
        for v in (acoustic, modulation)
    ], dim=1).squeeze(0)


def training_normalization(features: torch.Tensor,
                           train_indices: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    require_cuda(features); require_cuda(train_indices)
    if train_indices.numel() < 1:
        raise ValueError("Training normalization requires training samples")
    subset = features[train_indices]
    mean = subset.mean(dim=(0, 2, 3), keepdim=True)
    std = subset.std(dim=(0, 2, 3), unbiased=False, keepdim=True).clamp_min(1e-6)
    return mean, std


class CompactSceneCNN(nn.Module):
    """Independent three-convolution encoders and one shared classifier head.

    P2 contains more parameters because it has two encoders and a larger fusion
    projection. This is reported explicitly, not described as parameter-matched.
    """
    def __init__(self, classes: int = 4, dual_view: bool = False,
                 device: torch.device | str = "cuda:0"):
        super().__init__()
        device = torch.device(device)
        if device.type != "cuda":
            raise ValueError("Model initialization must be on CUDA")
        require_cuda()
        self.dual_view = dual_view
        self.classes = classes

        def branch():
            return nn.Sequential(
                nn.Conv2d(1, 16, 5, stride=2, padding=2, device=device),
                nn.GroupNorm(4, 16, device=device), nn.GELU(),
                nn.Conv2d(16, 32, 3, stride=2, padding=1, device=device),
                nn.GroupNorm(8, 32, device=device), nn.GELU(),
                nn.Conv2d(32, 64, 3, stride=2, padding=1, device=device),
                nn.GroupNorm(8, 64, device=device), nn.GELU(),
                # Feature inputs are fixed at 128 square, giving 16 square
                # after the three strided convolutions. Fixed pooling avoids
                # adaptive-pooling backward nondeterminism on some CUDA builds.
                nn.AvgPool2d(4), nn.Flatten(),
            )

        self.lofar_encoder = branch()
        self.demon_encoder = branch() if dual_view else None
        self.classifier = nn.Sequential(
            nn.Linear(1024 * (2 if dual_view else 1), 128, device=device),
            nn.GELU(), nn.Dropout(.15), nn.Linear(128, classes, device=device),
        )
        assert_model_cuda(self)

    def forward(self, views: torch.Tensor) -> torch.Tensor:
        require_cuda(views)
        if views.ndim != 4 or views.shape[1] != 2 or views.shape[-2:] != (128, 128):
            raise ValueError("Expected [batch, two views, 128 frequency, 128 time]")
        a = self.lofar_encoder(views[:, :1])
        if self.dual_view:
            a = torch.cat([a, self.demon_encoder(views[:, 1:2])], dim=1)
        return self.classifier(a)


def assert_model_cuda(model: nn.Module) -> None:
    for name, tensor in list(model.named_parameters()) + list(model.named_buffers()):
        if tensor.device.type != "cuda":
            raise RuntimeError(f"CPU model parameter/buffer found: {name}")


def confusion_counts(labels: torch.Tensor, predictions: torch.Tensor,
                     num_classes: int) -> torch.Tensor:
    require_cuda(labels); require_cuda(predictions)
    if labels.shape != predictions.shape:
        raise ValueError("Prediction/label shapes differ")
    if ((labels < 0) | (labels >= num_classes) |
        (predictions < 0) | (predictions >= num_classes)).any():
        raise ValueError("Class index outside the frozen class vocabulary")
    return torch.bincount(labels.long() * num_classes + predictions.long(),
                          minlength=num_classes ** 2).reshape(num_classes, num_classes)


def metric_tensors(confusion: torch.Tensor) -> dict[str, torch.Tensor]:
    """Metrics over the last two axes, supporting bootstrap batches on CUDA."""
    require_cuda(confusion)
    counts = confusion.float()
    tp = counts.diagonal(dim1=-2, dim2=-1)
    support = counts.sum(dim=-1)
    predicted = counts.sum(dim=-2)
    precision = tp / predicted.clamp_min(1)
    recall = tp / support.clamp_min(1)
    f1 = 2 * tp / (support + predicted).clamp_min(1)
    return {
        "accuracy": tp.sum(dim=-1) / counts.sum(dim=(-2, -1)).clamp_min(1),
        "macro_f1": f1.mean(dim=-1),
        "macro_f1_all_frozen_classes": f1.mean(dim=-1),
        "macro_f1_present_reference_classes": (f1 * (support > 0)).sum(dim=-1) /
                                             (support > 0).sum(dim=-1).clamp_min(1),
        "balanced_accuracy": (recall * (support > 0)).sum(dim=-1) /
                             (support > 0).sum(dim=-1).clamp_min(1),
        "per_class_precision": precision,
        "per_class_recall": recall,
        "per_class_f1": f1,
        "per_class_support": support,
    }


def grouped_confusions(labels: torch.Tensor, predictions: torch.Tensor,
                       group_indices: torch.Tensor, n_groups: int,
                       num_classes: int) -> torch.Tensor:
    require_cuda(labels); require_cuda(predictions); require_cuda(group_indices)
    flat = group_indices * num_classes ** 2 + labels * num_classes + predictions
    return torch.bincount(flat, minlength=n_groups * num_classes ** 2).reshape(
        n_groups, num_classes, num_classes)


def bootstrap_methods(method_group_confusions: dict[str, torch.Tensor],
                      draws: int = 2000, seed: int = 89731) -> dict:
    """Paired recording-group bootstrap, conditional on matched training seeds.

    Input values have [training_seed, group, class, class]. A common sampled
    group list is used for every method and seed. The statistic averages each
    seed's metric; seed variability is also reported separately by the caller.
    """
    if not method_group_confusions:
        raise ValueError("No bootstrap input")
    first = next(iter(method_group_confusions.values()))
    device = require_cuda(first)
    n_groups = first.shape[1]
    if n_groups < 2:
        return {"status": "insufficient_independent_groups", "n_groups": n_groups}
    generator = torch.Generator(device=device).manual_seed(seed)
    sampled = torch.randint(n_groups, (draws, n_groups), generator=generator, device=device)
    metrics = {}
    result = {
        "status": "complete", "draws": draws, "seed": seed, "n_groups": n_groups,
        "resampling_unit": "original_recording_group",
        "uncertainty": "recording-group sampling conditional on the matched trained seeds; seeds are not resampled",
        "statistic": "mean of per-seed metrics on pooled sampled recording groups",
        "device": str(device), "methods": {}, "paired_P2_minus_P1": {},
    }
    quantiles = torch.tensor([.025, .975], device=device)
    for name, counts in method_group_confusions.items():
        require_cuda(counts)
        if counts.shape != first.shape:
            raise ValueError("Methods must use the same seeds, groups and classes")
        # [seed, draw, sampled_group, C, C] -> [seed, draw, C, C]
        totals = counts[:, sampled].sum(dim=2)
        stat = metric_tensors(totals)
        point = metric_tensors(counts.sum(dim=1))
        metrics[name] = {}
        result["methods"][name] = {}
        for metric in ("accuracy", "macro_f1", "macro_f1_all_frozen_classes", "macro_f1_present_reference_classes", "balanced_accuracy"):
            values = stat[metric].mean(dim=0)
            metrics[name][metric] = values
            result["methods"][name][metric] = {
                "point": point[metric].mean().item(),
                "ci95": torch.quantile(values, quantiles).tolist(),
            }
    if "P1" in metrics and "P2" in metrics:
        for metric in ("accuracy", "macro_f1", "macro_f1_all_frozen_classes", "macro_f1_present_reference_classes", "balanced_accuracy"):
            difference = metrics["P2"][metric] - metrics["P1"][metric]
            point = (metric_tensors(method_group_confusions["P2"].sum(dim=1))[metric] -
                     metric_tensors(method_group_confusions["P1"].sum(dim=1))[metric]).mean()
            result["paired_P2_minus_P1"][metric] = {
                "point": point.item(), "ci95": torch.quantile(difference, quantiles).tolist(),
            }
    return result
