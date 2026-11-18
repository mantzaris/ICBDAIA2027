"""CUDA localized-event metrics and independent-group paired uncertainty.

No real-data correctness can be computed without independently supplied labels.
Schema validity and internal verifier pass rate are deliberately separate metrics.
"""
from __future__ import annotations
from typing import Sequence
import torch
from .signal import require_cuda


def _prf(counts: torch.Tensor) -> torch.Tensor:
    tp, fp, fn = counts.unbind(-1)
    precision = torch.where(tp + fp > 0, tp / (tp + fp), torch.full_like(tp, float("nan")))
    recall = torch.where(tp + fn > 0, tp / (tp + fn), torch.full_like(tp, float("nan")))
    f1 = torch.where(2 * tp + fp + fn > 0, 2 * tp / (2 * tp + fp + fn), torch.zeros_like(tp))
    return torch.stack((precision, recall, f1), dim=-1)


def _interpolate_track(event: dict, t: torch.Tensor) -> torch.Tensor:
    if event.get("frequency_track"):
        track = torch.tensor(event["frequency_track"], device=t.device)
        # Repeated times are not accepted as independent ridge support.
        if track.shape[0] == 1:
            return torch.full_like(t, track[0, 1])
        right = torch.searchsorted(track[:, 0].contiguous(), t).clamp(1, track.shape[0] - 1)
        left = right - 1
        fraction = (t - track[left, 0]) / (track[right, 0] - track[left, 0]).clamp_min(1e-8)
        return track[left, 1] + fraction.clamp(0, 1) * (track[right, 1] - track[left, 1])
    center = torch.tensor(event["frequency_hz"], device=t.device).mean()
    return center.expand_as(t)


def event_metrics(predicted: list[dict], gold: list[dict], time_iou_threshold: float = .3,
                  box_iou_threshold: float = .3, frequency_tolerance_hz: float = 2.,
                  exhaustive: bool = True) -> dict:
    """One-to-one greedy descending-quality matching, frozen before evaluation.

    Narrowband events need matching morphology/domain, temporal IoU, and mean
    track error <= max(two analysis bins, supplied annotation uncertainty,
    frequency_tolerance_hz). Broadband events use time-frequency box IoU.
    Duplicates remain false positives. Unannotated real regions are not negatives:
    exhaustive=False returns TP/FN only and undefined precision/F1.
    """
    device = require_cuda()
    optional = [g for g in gold if g.get("evaluation_required") is False]
    if optional:
        required = [g for g in gold if g.get("evaluation_required") is not False]
        primary = event_metrics(predicted, required, time_iou_threshold, box_iou_threshold,
                                frequency_tolerance_hz, exhaustive)
        matched = {m["prediction_index"] for m in primary["matches"]}
        remaining_indices = [i for i in range(len(predicted)) if i not in matched]
        optional_plain = [{k: v for k, v in g.items() if k != "evaluation_required"} for g in optional]
        extra = event_metrics([predicted[i] for i in remaining_indices], optional_plain,
                              time_iou_threshold, box_iou_threshold, frequency_tolerance_hz, exhaustive=True)
        ignored = [remaining_indices[m["prediction_index"]] for m in extra["matches"]]
        if exhaustive:
            counts = torch.tensor([primary["tp"], primary["fp"] - len(ignored), primary["fn"]],
                                  device=device, dtype=torch.float32)
            score = _prf(counts)
            primary.update(fp=int(counts[1].item()),
                           precision=score[0].item() if torch.isfinite(score[0]) else None,
                           recall=score[1].item() if torch.isfinite(score[1]) else None,
                           f1=score[2].item())
        primary.update(ignored_predictions=ignored, optional_known_structure=len(optional),
                       optional_rule="Known optional structure gives neither TP credit nor FP/FN penalty")
        return primary
    n, m = len(predicted), len(gold)
    if n == 0 or m == 0:
        counts = torch.tensor([0, n, m], device=device, dtype=torch.float32)
        scores = _prf(counts)
        result = dict(tp=0, fp=n if exhaustive else None, fn=m,
                      precision=None if n == 0 or not exhaustive else 0.,
                      recall=None if m == 0 else 0., f1=scores[2].item() if exhaustive else None,
                      matches=[], matching_algorithm="greedy_descending_quality", exhaustive=exhaustive)
        return result
    p_times = torch.tensor([e["time_s"] for e in predicted], device=device)
    g_times = torch.tensor([e["time_s"] for e in gold], device=device)
    lower = torch.maximum(p_times[:, None, 0], g_times[None, :, 0])
    upper = torch.minimum(p_times[:, None, 1], g_times[None, :, 1])
    intersection = (upper - lower).clamp_min(0)
    union = torch.maximum(p_times[:, None, 1], g_times[None, :, 1]) - torch.minimum(p_times[:, None, 0], g_times[None, :, 0])
    time_iou = intersection / union.clamp_min(1e-8)
    quality = torch.full((n, m), -1., device=device)
    errors = torch.full_like(quality, float("nan"))
    for i, p in enumerate(predicted):
        for j, g in enumerate(gold):
            if p.get("type") != g.get("type") or p.get("frequency_domain", "acoustic") != g.get("frequency_domain", "acoustic"):
                continue
            if p.get("frequency_domain") == "modulation":
                if not p.get("carrier_band_hz") or not g.get("carrier_band_hz"):
                    continue
                bands = torch.tensor([p["carrier_band_hz"], g["carrier_band_hz"]], device=device)
                overlap = (bands[:, 1].min() - bands[:, 0].max()).clamp_min(0)
                if overlap / (bands[:, 1] - bands[:, 0]).min().clamp_min(1e-8) < .5:
                    continue
            if p["type"] in ("tone", "chirp", "modulation"):
                times = lower[i, j] + torch.linspace(0, 1, 33, device=device) * intersection[i, j]
                error = (_interpolate_track(p, times) - _interpolate_track(g, times)).abs().mean()
                tolerance = max(frequency_tolerance_hz, 2 * p.get("bin_spacing_hz", 1),
                                g.get("frequency_uncertainty_hz", 0))
                errors[i, j] = error
                valid = (time_iou[i, j] >= time_iou_threshold) & (error <= tolerance)
                quality[i, j] = torch.where(valid, time_iou[i, j] + .1 * (1 - error / tolerance), -1.)
            else:
                pf, gf = torch.tensor(p["frequency_hz"], device=device), torch.tensor(g["frequency_hz"], device=device)
                f_intersection = (torch.minimum(pf[1], gf[1]) - torch.maximum(pf[0], gf[0])).clamp_min(0)
                area_i = intersection[i, j] * f_intersection
                area_p = (p_times[i, 1] - p_times[i, 0]) * (pf[1] - pf[0])
                area_g = (g_times[j, 1] - g_times[j, 0]) * (gf[1] - gf[0])
                iou = area_i / (area_p + area_g - area_i).clamp_min(1e-8)
                quality[i, j] = torch.where(iou >= box_iou_threshold, iou, -1.)
    matches = []
    remaining = quality.clone()
    for _ in range(min(n, m)):
        best = remaining.argmax()
        i, j = torch.div(best, m, rounding_mode="floor"), best % m
        if remaining[i, j] < 0:
            break
        matches.append(dict(prediction_index=i.item(), gold_index=j.item(),
                            quality=quality[i, j].item(), time_iou=time_iou[i, j].item(),
                            frequency_error_hz=errors[i, j].item() if torch.isfinite(errors[i, j]) else None))
        remaining[i, :] = -1
        remaining[:, j] = -1
    tp = torch.as_tensor(len(matches), device=device, dtype=torch.float32)
    counts = torch.stack((tp, n - tp, m - tp))
    scores = _prf(counts)
    return dict(tp=int(tp.item()), fp=int(counts[1].item()) if exhaustive else None,
                fn=int(counts[2].item()), precision=scores[0].item() if exhaustive else None,
                recall=scores[1].item(), f1=scores[2].item() if exhaustive else None,
                matches=matches, matching_algorithm="greedy_descending_quality", exhaustive=exhaustive,
                time_iou_threshold=time_iou_threshold, box_iou_threshold=box_iou_threshold,
                minimum_frequency_tolerance_hz=frequency_tolerance_hz)


def aggregate_event_metrics(rows: Sequence[dict]) -> dict:
    device = require_cuda()
    if any(r.get("fp") is None for r in rows):
        raise ValueError("Precision/F1 require exhaustive or independently verified negative regions")
    counts = torch.tensor([[r["tp"], r["fp"], r["fn"]] for r in rows], device=device, dtype=torch.float32)
    total = counts.sum(dim=0) if len(rows) else torch.zeros(3, device=device)
    score = _prf(total)
    return dict(tp=int(total[0].item()), fp=int(total[1].item()), fn=int(total[2].item()),
                precision=score[0].item() if torch.isfinite(score[0]) else None,
                recall=score[1].item() if torch.isfinite(score[1]) else None,
                f1=score[2].item(), scenes=len(rows))


def _group_counts(rows: Sequence[dict], group_ids: Sequence[str]) -> tuple[torch.Tensor, list[str]]:
    device = require_cuda()
    if len(rows) != len(group_ids) or not rows:
        raise ValueError("Require one nonempty row per declared parent group ID")
    if any(r.get("fp") is None for r in rows):
        raise ValueError("Grouped F1 requires verified negatives")
    groups = sorted(set(group_ids))
    mapping = {g: i for i, g in enumerate(groups)}
    indices = torch.tensor([mapping[g] for g in group_ids], device=device)
    values = torch.tensor([[r["tp"], r["fp"], r["fn"]] for r in rows], device=device, dtype=torch.float64)
    counts = torch.zeros((len(groups), 3), device=device, dtype=torch.float64)
    counts.scatter_add_(0, indices[:, None].expand(-1, 3), values)
    return counts, groups


def grouped_bootstrap(rows_a: Sequence[dict], rows_b: Sequence[dict], group_ids: Sequence[str],
                      repetitions: int = 2000, seed: int = 601) -> dict:
    a, groups = _group_counts(rows_a, group_ids)
    b, groups_b = _group_counts(rows_b, group_ids)
    if groups != groups_b:
        raise ValueError("Methods must use identical independent-group ordering")
    generator = torch.Generator(device=a.device).manual_seed(seed)
    indices = torch.randint(len(groups), (repetitions, len(groups)), generator=generator, device=a.device)
    boot_a, boot_b = a[indices].sum(dim=1), b[indices].sum(dim=1)
    differences = _prf(boot_a)[:, 2] - _prf(boot_b)[:, 2]
    quantiles = torch.quantile(differences, torch.tensor([.025, .975], device=a.device, dtype=a.dtype))
    point = _prf(a.sum(dim=0))[2] - _prf(b.sum(dim=0))[2]
    return dict(metric="micro_event_F1_A_minus_B", estimate=point.item(),
                confidence_interval_95=quantiles.tolist(), independent_groups=len(groups),
                bootstrap_repetitions=repetitions, seed=seed, device=str(a.device),
                resampling_unit="declared_independent_parent_group",
                uncertainty="recording_sampling_only; no independent model-seed variability inferred")


def paired_group_permutation(rows_a, rows_b, group_ids, repetitions=4096, seed=602):
    a, groups = _group_counts(rows_a, group_ids)
    b, _ = _group_counts(rows_b, group_ids)
    generator = torch.Generator(device=a.device).manual_seed(seed)
    swaps = torch.randint(2, (repetitions, len(groups), 1), generator=generator, device=a.device).bool()
    perm_a = torch.where(swaps, a[None], b[None]).sum(dim=1)
    perm_b = torch.where(swaps, b[None], a[None]).sum(dim=1)
    null = _prf(perm_a)[:, 2] - _prf(perm_b)[:, 2]
    actual = _prf(a.sum(dim=0))[2] - _prf(b.sum(dim=0))[2]
    p = (1 + (null.abs() >= actual.abs() - 1e-12).sum()) / (repetitions + 1)
    return dict(effect_f1=actual.item(), p_value=p.item(), groups=len(groups),
                test="two_sided_paired_group_label_swap", repetitions=repetitions, seed=seed)


def holm(p_values: list[float]) -> list[float]:
    device = require_cuda()
    p = torch.tensor(p_values, device=device)
    order = p.argsort()
    adjusted = (p[order] * torch.arange(len(p), 0, -1, device=device)).clamp(max=1.)
    adjusted = torch.cummax(adjusted, dim=0).values
    result = torch.empty_like(p)
    result[order] = adjusted
    return result.tolist()


def temporal_relation_metrics(predicted, gold, matches, threshold=.1) -> dict:
    """Temporal overlap edges after event matching, without source identity."""
    device = require_cuda()
    def adjacency(events):
        if not events:
            return torch.zeros((0, 0), device=device, dtype=torch.bool)
        intervals = torch.tensor([e["time_s"] for e in events], device=device)
        duration = intervals[:, 1] - intervals[:, 0]
        overlap = (torch.minimum(intervals[:, None, 1], intervals[None, :, 1]) -
                   torch.maximum(intervals[:, None, 0], intervals[None, :, 0])).clamp_min(0)
        ratio = overlap / torch.minimum(duration[:, None], duration[None, :]).clamp_min(1e-8)
        return torch.triu(ratio > threshold, diagonal=1)
    pa, ga = adjacency(predicted), adjacency(gold)
    tp = torch.zeros((), device=device)
    mapping = torch.full((len(predicted),), -1, device=device, dtype=torch.long)
    for match in matches:
        mapping[match["prediction_index"]] = match["gold_index"]
    pairs = torch.nonzero(pa)
    if pairs.numel():
        translated = mapping[pairs]
        valid = (translated >= 0).all(dim=1)
        translated = translated[valid].sort(dim=1).values
        if translated.numel():
            tp = ga[translated[:, 0], translated[:, 1]].float().sum()
    counts = torch.stack((tp, pa.sum() - tp, ga.sum() - tp))
    scores = _prf(counts)
    return dict(tp=int(counts[0].item()), fp=int(counts[1].item()), fn=int(counts[2].item()),
                f1=scores[2].item(), relation="overlaps_time", threshold=threshold)


def calibration(probabilities: torch.Tensor, labels: torch.Tensor, bins: int = 10):
    require_cuda(probabilities); require_cuda(labels)
    if probabilities.shape != labels.shape or probabilities.numel() == 0:
        raise ValueError("Nonempty aligned probabilities and independent correctness labels required")
    if ((probabilities < 0) | (probabilities > 1)).any():
        raise ValueError("Probabilities must lie in [0,1]")
    p, y = probabilities.flatten().float(), labels.flatten().float()
    assignment = (p * bins).long().clamp(max=bins - 1)
    counts = torch.zeros(bins, device=p.device)
    sum_p, sum_y = counts.clone(), counts.clone()
    counts.scatter_add_(0, assignment, torch.ones_like(p))
    sum_p.scatter_add_(0, assignment, p); sum_y.scatter_add_(0, assignment, y)
    confidence = sum_p / counts.clamp_min(1)
    accuracy = sum_y / counts.clamp_min(1)
    ece = ((confidence - accuracy).abs() * counts).sum() / p.numel()
    order = p.argsort(descending=True)
    coverage = torch.arange(1, p.numel() + 1, device=p.device) / p.numel()
    risk = (1 - y[order]).cumsum(dim=0) / torch.arange(1, p.numel() + 1, device=p.device)
    return dict(brier=(p - y).square().mean().item(), ece=ece.item(),
                counts=counts.int().tolist(), confidence=confidence.tolist(), accuracy=accuracy.tolist(),
                risk=risk.tolist(), coverage=coverage.tolist())
