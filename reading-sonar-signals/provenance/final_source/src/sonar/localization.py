"""CUDA dense morphology baselines with separate acoustic/modulation coordinates.

The networks learn from exact synthetic masks. Pixel probabilities are uncalibrated
scores. Thresholded connected regions are converted to measurements on the native
GPU PSD grid; resizing never changes the stated measurement resolution.
"""
from __future__ import annotations
import hashlib
import torch
from torch import nn
from torch.nn import functional as F
from .signal import require_cuda, lofar, demon, bandpass, SpectralView

ACOUSTIC_TYPES = ("tone", "chirp", "transient", "broadband")
CARRIERS = ((500., 2000.), (2000., 5000.))
GRID = (128, 128)
MAX_COMPONENTS = 6
CONFIG = dict(grid=list(GRID), acoustic_types=list(ACOUSTIC_TYPES), carrier_bands_hz=CARRIERS,
              max_output_components=MAX_COMPONENTS,
              acoustic_window_s=1., acoustic_hop_s=.25, modulation_window_s=4.,
              modulation_hop_s=.5, target_ridge_halfwidth_grid_pixels=1.5,
              component_decoder="threshold_connected_regions_then_native_PSD_ridge_measurement",
              training_truth="exact_auxiliary_synthetic_only", real_annotations_used_for_training=False)


def auxiliary_broadband_scene(seed: int, split: str, snr_db: float, sample_rate=16000, duration_s=30.):
    """Independent auxiliary foreground burst, absent from the dense benchmark.

    All sampling and waveform arithmetic execute on CUDA. Exact labels describe
    the declared foreground passband/support; stationary background and incidental
    envelope structure are not labelled as additional foreground events.
    """
    if split not in ("train","validation"):
        raise ValueError("Auxiliary broadband generator is development-only")
    device=require_cuda()
    generator_seed={"train":31000000,"validation":32000000}[split]+int(seed)
    generator=torch.Generator(device=device).manual_seed(generator_seed)
    parameters=torch.rand(4,device=device,generator=generator)
    low=(200+800*parameters[0]).item(); high=low+(400+400*parameters[1]).item()
    begin=(3+6*parameters[2]).item(); end=begin+(4+8*parameters[3]).item()
    t=torch.arange(round(sample_rate*duration_s),device=device)/sample_rate
    envelope=torch.minimum(((t-begin)/.1).clamp(0,1),((end-t)/.1).clamp(0,1))
    foreground=bandpass(torch.randn(t.numel(),device=device,generator=generator),sample_rate,(low,high))
    foreground=foreground/foreground.square().mean().sqrt().clamp_min(1e-10)*envelope
    background=torch.randn(t.numel(),device=device,generator=generator)
    background*=foreground.square().mean().sqrt()/background.square().mean().sqrt()*10**(-snr_db/20)
    waveform=foreground+background
    waveform/=torch.maximum(waveform.abs().max(),waveform.new_tensor(1.))/.95
    identity=hashlib.sha256(f"auxiliary-broadband-v1:{split}:{seed}".encode()).hexdigest()[:16]
    truth=dict(scene_id=f"aux-{identity}",generator_seed=generator_seed,kind="auxiliary_broadband",split=split,
        snr_db=snr_db,sample_rate_hz=sample_rate,duration_s=duration_s,modulation_exhaustive=False,
        annotation_origin="exact_auxiliary_foreground_passband_and_support",generator_version="auxiliary_broadband_v1",
        known_label_scope="declared broadband foreground; background and incidental envelope structure excluded",
        events=[dict(id="gold:0",type="broadband",frequency_domain="acoustic",time_s=[begin,end],
                     frequency_hz=[low,high],source_identity="not_defined")])
    return waveform,truth


def localizer_features(x, sample_rate, offset_s=0.):
    require_cuda(x)
    views = [lofar(x, sample_rate, window_s=1., hop_s=.25, offset_s=offset_s)]
    views.extend(demon(x, sample_rate, band, window_s=4., hop_s=.5, offset_s=offset_s) for band in CARRIERS)
    features = torch.cat([F.interpolate(v.log_power()[None, None], size=GRID,
                                       mode="bilinear", align_corners=True) for v in views], dim=1)[0]
    return features, views


def _track_frequency(event, time):
    if event.get("frequency_track"):
        points = torch.tensor(event["frequency_track"], device=time.device)
        if points.shape[0] == 1:
            return points[0, 1].expand_as(time)
        right = torch.searchsorted(points[:, 0].contiguous(), time).clamp(1, points.shape[0] - 1)
        left = right - 1
        frac = ((time - points[left, 0]) / (points[right, 0] - points[left, 0]).clamp_min(1e-8)).clamp(0, 1)
        return points[left, 1] + frac * (points[right, 1] - points[left, 1])
    return torch.tensor(event["frequency_hz"], device=time.device).mean().expand_as(time)


def dense_targets(views: list[SpectralView], events: list[dict], modulation_exhaustive: bool = True):
    device = views[0].psd.device
    targets = torch.zeros((6, *GRID), device=device)
    valid = torch.ones_like(targets)
    if not modulation_exhaustive:
        valid[4:] = 0
    for channel in range(6):
        view_index = 0 if channel < 4 else channel - 3
        view = views[view_index]
        f = torch.linspace(view.frequencies_hz[0], view.frequencies_hz[-1], GRID[0], device=device)
        t = torch.linspace(view.times_s[0], view.times_s[-1], GRID[1], device=device)
        kind = ACOUSTIC_TYPES[channel] if channel < 4 else "modulation"
        for event in events:
            if event["type"] != kind:
                continue
            if kind == "modulation":
                band = torch.tensor([event.get("carrier_band_hz", [0., 0.]), CARRIERS[channel - 4]], device=device)
                overlap = (band[:, 1].min() - band[:, 0].max()).clamp_min(0)
                if overlap / (band[:, 1] - band[:, 0]).min().clamp_min(1e-8) < .5:
                    continue
            temporal = (t >= event["time_s"][0]) & (t <= event["time_s"][1])
            if kind in ("tone", "chirp", "modulation"):
                width = 1.5 * (f[1] - f[0])
                mask = (f[:, None] - _track_frequency(event, t)[None]).abs() <= width
            else:
                mask = ((f >= event["frequency_hz"][0]) & (f <= event["frequency_hz"][1]))[:, None].expand(-1, GRID[1])
            mask = mask & temporal[None]
            if event.get("evaluation_required") is False:
                valid[channel][mask] = 0
            else:
                targets[channel][mask] = 1
    return targets, valid


class DenseMorphologyCNN(nn.Module):
    def __init__(self, dual_view=False):
        super().__init__()
        require_cuda()
        self.dual_view = dual_view
        def encoder():
            return nn.Sequential(nn.Conv2d(1, 16, 5, padding=2, device="cuda"), nn.GroupNorm(4, 16, device="cuda"), nn.GELU(),
                                 nn.Conv2d(16, 24, 3, padding=1, device="cuda"), nn.GroupNorm(4, 24, device="cuda"), nn.GELU(),
                                 nn.Conv2d(24, 24, 3, padding=1, device="cuda"), nn.GroupNorm(4, 24, device="cuda"), nn.GELU())
        self.acoustic_encoder = encoder()
        self.acoustic_head = nn.Conv2d(24, 4, 1, device="cuda")
        if dual_view:
            self.modulation_encoder = encoder()
            self.modulation_head = nn.Conv2d(24, 1, 1, device="cuda")
            self.modulation_context = nn.Linear(48, 24, device="cuda")

    def forward(self, x):
        require_cuda(x)
        acoustic = self.acoustic_encoder(x[:, :1])
        if not self.dual_view:
            return self.acoustic_head(acoustic)
        modulation = [self.modulation_encoder(x[:, i:i + 1]) for i in (1, 2)]
        # Global context permits fusion without equating acoustic and modulation
        # pixel coordinates. The two carrier-band output grids remain distinct.
        context = torch.cat([value.mean(dim=(-1, -2)) for value in modulation], dim=1)
        acoustic = acoustic + self.modulation_context(context)[:, :, None, None]
        return torch.cat([self.acoustic_head(acoustic)] + [self.modulation_head(value) for value in modulation], dim=1)


def segmentation_loss(logits, target, valid, positive_weight):
    channels = logits.shape[1]
    target, valid = target[:, :channels], valid[:, :channels]
    bce = F.binary_cross_entropy_with_logits(logits, target, reduction="none",
                                             pos_weight=positive_weight[:channels][None, :, None, None])
    bce = (bce * valid).sum() / valid.sum().clamp_min(1)
    probabilities = logits.sigmoid() * valid
    intersection = (probabilities * target).sum(dim=(-1, -2))
    denominator = probabilities.sum(dim=(-1, -2)) + (target * valid).sum(dim=(-1, -2))
    dice = 1 - (2 * intersection + 1) / (denominator + 1)
    return bce + .5 * dice.mean()


def _connected_regions(active: torch.Tensor):
    """GPU eight-neighbor label propagation on the bounded 128-square grid."""
    if not active.any():
        return []
    labels = torch.arange(1, active.numel() + 1, device=active.device).reshape_as(active).float() * active
    for _ in range(sum(active.shape)):
        updated = F.max_pool2d(labels[None, None], 3, stride=1, padding=1)[0, 0] * active
        if torch.equal(updated, labels):
            break
        labels = updated
    ids, counts = torch.unique(labels[active], return_counts=True)
    ids = ids[counts >= 3]
    return [labels == label for label in ids]


@torch.inference_mode()
def decode_components(probabilities: torch.Tensor, views: list[SpectralView], threshold: float = .5,
                      max_components: int = MAX_COMPONENTS) -> list[dict]:
    require_cuda(probabilities)
    events = []
    for channel in range(probabilities.shape[0]):
        view = views[0 if channel < 4 else channel - 3]
        kind = ACOUSTIC_TYPES[channel] if channel < 4 else "modulation"
        for region in _connected_regions(probabilities[channel] >= threshold):
            native = F.interpolate(region.float()[None, None], size=view.psd.shape,
                                   mode="nearest")[0, 0].bool()
            columns = native.any(dim=0)
            rows = native.any(dim=1)
            if not columns.any() or not rows.any():
                continue
            times = view.times_s[columns]
            frequencies = view.frequencies_hz[rows]
            half_window = view.metadata["window_s"] / 2
            start = torch.maximum(times[0] - half_window, times.new_tensor(view.metadata["time_origin_s"]))
            stop = torch.minimum(times[-1] + half_window, times.new_tensor(
                view.metadata["time_origin_s"] + view.metadata["source_duration_s"]))
            event = dict(id=f"predicted:{len(events)}", type=kind,
                         frequency_domain="acoustic" if channel < 4 else "modulation",
                         time_s=[start.item(), stop.item()],
                         frequency_hz=[frequencies[0].item(), frequencies[-1].item()],
                         bin_spacing_hz=view.metadata["bin_spacing_hz"],
                         boundary_uncertainty_s=half_window,
                         source_identity="unknown", evidence_ids=[],
                         localizer_score=probabilities[channel][region].mean().item(),
                         score_kind="uncalibrated_sigmoid_region_mean")
            if kind in ("tone", "chirp", "modulation"):
                # Physical frequency estimates are from the original PSD, not
                # from pixel interpolation or a hidden simulator trajectory.
                selected = torch.where(native[:, columns], view.psd[:, columns], -torch.ones_like(view.psd[:, columns]))
                ridge = view.frequencies_hz[selected.argmax(dim=0)]
                event["frequency_track"] = torch.stack((times, ridge), dim=1).tolist()
                event["frequency_hz"] = [ridge.min().item(), ridge.max().item()]
                event["center_frequency_hz"] = ridge.median().item()
                centered = times - times.mean()
                event["slope_hz_per_s"] = ((ridge - ridge.mean()) * centered).sum().div(centered.square().sum().clamp_min(1e-8)).item()
            if channel >= 4:
                event["carrier_band_hz"] = list(CARRIERS[channel - 4])
            events.append(event)
    # This cap is a frozen output budget, not source count or confidence calibration.
    if len(events) > max_components:
        scores = torch.tensor([e["localizer_score"] for e in events], device=probabilities.device)
        events = [events[i] for i in scores.topk(max_components).indices.tolist()]
    return events
