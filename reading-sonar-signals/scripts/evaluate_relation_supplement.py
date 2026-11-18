#!/usr/bin/env python3
"""Secondary exact-synthetic relation metrics; never changes frozen node scores.

Run the common evaluator first, then:
  PYTHONPATH=src python scripts/evaluate_relation_supplement.py --run results/main

Saved common-evaluator matches are authoritative. Metadata, IDs, JSON and hashes
are administrative CPU work; geometry, edge counts and uncertainty use CUDA.
This script does not assess natural-language claims or real physical sources.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
from itertools import combinations
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import torch
from sonar.signal import require_cuda
from sonar.metrics import _prf, aggregate_event_metrics
import evaluate_benchmark as common

CONFIG = ROOT / "configs/relation_supplement.json"
NARROW_TYPES = {"tone", "chirp", "modulation"}
SUPPORT_WIDTH_HZ = 2.0
TEMPORAL_THRESHOLD = 0.1
CARRIER_THRESHOLD = 0.5
METRIC_KEYS = ("reported_temporal_relations", "geometric_time_frequency_relations")


def _tensor(value):
    return torch.tensor(value, device=require_cuda(), dtype=torch.float64)


def temporal_adjacency(events):
    """The original frozen temporal criterion, without morphology restrictions."""
    n = len(events)
    if not n:
        return torch.zeros((0, 0), device=require_cuda(), dtype=torch.bool)
    times = _tensor([event["time_s"] for event in events])
    overlap = (torch.minimum(times[:, None, 1], times[None, :, 1]) -
               torch.maximum(times[:, None, 0], times[None, :, 0])).clamp_min(0)
    durations = times[:, 1] - times[:, 0]
    ratio = overlap / torch.minimum(durations[:, None], durations[None, :]).clamp_min(1e-8)
    return torch.triu(ratio > TEMPORAL_THRESHOLD, diagonal=1)


def _center(event, times):
    """Float64 piecewise-linear interpolation with constant endpoint extension."""
    if not event.get("frequency_track"):
        return _tensor(event["frequency_hz"]).mean().expand_as(times)
    track = _tensor(event["frequency_track"])
    if track.shape[0] == 1:
        return track[0, 1].expand_as(times)
    right = torch.searchsorted(track[:, 0].contiguous(), times).clamp(1, len(track) - 1)
    left = right - 1
    fraction = ((times - track[left, 0]) / (track[right, 0] - track[left, 0])).clamp(0, 1)
    return track[left, 1] + fraction * (track[right, 1] - track[left, 1])


def _support(event, times):
    if event["type"] in NARROW_TYPES:
        center = _center(event, times)
        return center - SUPPORT_WIDTH_HZ / 2, center + SUPPORT_WIDTH_HZ / 2
    band = _tensor(event["frequency_hz"])
    return band[0].expand_as(times), band[1].expand_as(times)


def tf_overlap_duration(first, second):
    """Exact duration with positive-width intersecting frequency support.

    The support bounds are linear between the union of both tracks' knots.
    Solving two strict linear inequalities on every segment catches crossings
    between knots; isolated boundary contacts contribute zero duration.
    """
    zero = _tensor(0.)
    domain = first.get("frequency_domain", "acoustic")
    if domain != second.get("frequency_domain", "acoustic"):
        return zero
    if domain == "modulation":
        if not first.get("carrier_band_hz") or not second.get("carrier_band_hz"):
            return zero
        bands = _tensor([first["carrier_band_hz"], second["carrier_band_hz"]])
        widths = bands[:, 1] - bands[:, 0]
        overlap = (bands[:, 1].min() - bands[:, 0].max()).clamp_min(0)
        if not bool((widths > 0).all() & (overlap / widths.min().clamp_min(1e-8) >= CARRIER_THRESHOLD)):
            return zero
    time = _tensor([first["time_s"], second["time_s"]])
    lower, upper = time[:, 0].max(), time[:, 1].min()
    if not bool(upper > lower):
        return zero
    knots = [torch.stack((lower, upper))]
    for event in (first, second):
        if event["type"] not in NARROW_TYPES:
            band = _tensor(event["frequency_hz"])
            if not bool(band[1] > band[0]):
                return zero
        elif event.get("frequency_track"):
            track_times = _tensor(event["frequency_track"])[:, 0]
            knots.append(track_times[(track_times > lower) & (track_times < upper)])
    knots = torch.unique(torch.cat(knots), sorted=True)
    a_low, a_high = _support(first, knots)
    b_low, b_high = _support(second, knots)
    margins = torch.stack((b_high - a_low, a_high - b_low), dim=1)
    initial, slope = margins[:-1], margins[1:] - margins[:-1]
    safe_slope = torch.where(slope != 0, slope, torch.ones_like(slope))
    roots = -initial / safe_slope
    starts = torch.where(slope > 0, roots, torch.zeros_like(roots)).amax(dim=1).clamp_min(0)
    ends = torch.where(slope < 0, roots, torch.ones_like(roots)).amin(dim=1).clamp_max(1)
    impossible = ((slope == 0) & (initial <= 0)).any(dim=1)
    fraction = torch.where(impossible, torch.zeros_like(starts), (ends - starts).clamp_min(0))
    return (fraction * (knots[1:] - knots[:-1])).sum()


def tf_adjacency(events):
    result = torch.zeros((len(events), len(events)), device=require_cuda(), dtype=torch.bool)
    durations = torch.zeros_like(result, dtype=torch.float64)
    for i, j in combinations(range(len(events)), 2):
        duration = tf_overlap_duration(events[i], events[j])
        durations[i, j] = duration
        result[i, j] = duration > 0
    return result, durations


def _node_mapping(prediction_count, gold_count, matching):
    matches = matching.get("matches", [])
    if any(type(m.get(k)) is not int for m in matches for k in ("prediction_index", "gold_index")):
        raise ValueError("Frozen match indices must be integers")
    p = [m["prediction_index"] for m in matches]
    g = [m["gold_index"] for m in matches]
    ignored = matching.get("ignored_predictions", [])
    if len(set(p)) != len(p) or len(set(g)) != len(g) or set(p) & set(ignored):
        raise ValueError("Frozen matches are not one-to-one or include optional ignored nodes")
    if any(i < 0 or i >= prediction_count for i in p + ignored) or any(i < 0 or i >= gold_count for i in g):
        raise ValueError("Frozen match indices are outside the reconstructed scoring scope")
    mapping = torch.full((prediction_count,), -1, device=require_cuda(), dtype=torch.long)
    if p:
        mapping[torch.tensor(p, device=mapping.device)] = torch.tensor(g, device=mapping.device)
    return mapping


def _score_pairs(pairs, gold_adjacency, mapping, invalid_claims=0):
    """Repeated true edges count once as TP; extra copies remain FP."""
    pairs = torch.as_tensor(pairs, device=mapping.device, dtype=torch.long).reshape(-1, 2)
    n_gold = gold_adjacency.shape[0]
    mapped = mapping[pairs] if pairs.numel() else pairs.clone()
    valid = (mapped >= 0).all(dim=1) & (mapped[:, 0] != mapped[:, 1])
    mapped = mapped.sort(dim=1).values
    hits = torch.zeros(n_gold * n_gold, device=mapping.device, dtype=torch.long)
    if bool(valid.any()):
        flat = mapped[valid, 0] * n_gold + mapped[valid, 1]
        hits.scatter_add_(0, flat, torch.ones_like(flat))
    correct = (hits > 0) & gold_adjacency.flatten()
    tp = correct.sum()
    counts = torch.stack((tp, tp.new_tensor(pairs.shape[0] + invalid_claims) - tp,
                          gold_adjacency.sum() - tp)).to(torch.float64)
    scores = _prf(counts)
    result = dict(zip(("tp", "fp", "fn"), [int(v) for v in counts.tolist()]))
    result.update(precision=scores[0].item() if bool(torch.isfinite(scores[0])) else None,
                  recall=scores[1].item() if bool(torch.isfinite(scores[1])) else None,
                  f1=scores[2].item(), gold_edges=int(gold_adjacency.sum().item()),
                  scored_predicted_edges=pairs.shape[0] + invalid_claims,
                  invalid_relation_claims=invalid_claims)
    return result


def report_id_states(report, reference, predictions, invalid, scoped, matching):
    """IDs stay the actual report IDs; duplicate IDs are deliberately ambiguous."""
    components = report.get("components", []) if isinstance(report, dict) else []
    if not isinstance(components, list):
        components = []
    invalid_by_index = {entry["index"]: entry for entry in invalid if type(entry.get("index")) is int}
    eligible_invalid = common.score_scope([], invalid, report, reference)[1]
    eligible_invalid_indices = {entry["index"] for entry in eligible_invalid}
    scoped_positions = {id(event): i for i, event in enumerate(scoped)}
    optional = set(matching.get("ignored_predictions", []))
    by_id = defaultdict(list)
    valid_iterator = iter(predictions)
    for original_index, component in enumerate(components):
        if original_index in invalid_by_index:
            state = {"status": "invalid" if original_index in eligible_invalid_indices else "unscored"}
        else:
            event = next(valid_iterator)
            position = scoped_positions.get(id(event))
            state = {"status": "unscored" if position is None or position in optional else "scored",
                     "prediction_index": position}
        if isinstance(component, dict) and isinstance(component.get("id"), str):
            by_id[component["id"]].append(state)
    return {key: values[0] if len(values) == 1 else {"status": "ambiguous"}
            for key, values in by_id.items()}


def explicit_temporal_scores(report, gold, mapping, id_states):
    """Only explicitly emitted overlaps_time relations are predictions."""
    relations = report.get("relations", []) if isinstance(report, dict) else []
    if not isinstance(relations, list):
        relations = []
    pairs, trace = [], []
    invalid_claims, ignored, other = 0, 0, 0
    for row_index, relation in enumerate(relations):
        if not isinstance(relation, dict) or relation.get("type") != "overlaps_time":
            other += 1
            continue
        ids = relation.get("component_ids")
        if not isinstance(ids, list) or len(ids) < 2 or any(not isinstance(x, str) for x in ids):
            invalid_claims += 1
            trace.append(dict(relation_index=row_index, status="false_positive_malformed_relation"))
            continue
        for a, b in combinations(ids, 2):
            first, second = id_states.get(a, {"status": "dangling"}), id_states.get(b, {"status": "dangling"})
            states = {first["status"], second["status"]}
            entry = dict(relation_index=row_index, component_ids=[a, b])
            if a == b or states & {"dangling", "ambiguous", "invalid"}:
                invalid_claims += 1
                entry["status"] = "false_positive_invalid_endpoint"
                entry["endpoint_states"] = [first["status"], second["status"]]
            elif "unscored" in states:
                ignored += 1
                entry["status"] = "unscored_reference_scope_or_optional_structure"
            else:
                pairs.append([first["prediction_index"], second["prediction_index"]])
                entry.update(status="scored_using_frozen_node_matches", prediction_indices=pairs[-1])
            trace.append(entry)
    result = _score_pairs(pairs, temporal_adjacency(gold), mapping, invalid_claims)
    result.update(ignored_out_of_scope_edges=ignored, other_or_untyped_relations=other,
                  edge_trace=trace, interpretation="Explicit reported temporal edges; no free-text or source-association correctness claim.")
    return result


def geometric_tf_scores(predictions, gold, mapping, ignored_predictions=()):
    predicted_edges, durations = tf_adjacency(predictions)
    if ignored_predictions:
        indices = torch.tensor(list(ignored_predictions), device=predicted_edges.device)
        predicted_edges[indices, :] = False
        predicted_edges[:, indices] = False
    gold_edges, gold_durations = tf_adjacency(gold)
    pairs = predicted_edges.nonzero()
    result = _score_pairs(pairs, gold_edges, mapping)
    result.update(prediction_index_pairs=pairs.tolist(),
                  positive_overlap_durations_s=durations[predicted_edges].tolist(),
                  gold_index_pairs=gold_edges.nonzero().tolist(),
                  gold_positive_overlap_durations_s=gold_durations[gold_edges].tolist(),
                  interpretation="Geometry-induced TF support overlap, not explicit relation or natural-language reasoning accuracy.")
    return result


def assess_episode(scene, run, frozen_row):
    reference = common.resolve_reference(scene, run, {})
    if reference is None or reference["kind"] != "exact_synthetic_component_truth":
        raise ValueError(f"Missing exact sealed truth for {scene['scene_id']}")
    episode_path = run / scene["scene_id"] / frozen_row["method"] / "episode.json"
    if frozen_row.get("episode_sha256"):
        if not episode_path.exists() or common.digest(episode_path) != frozen_row["episode_sha256"]:
            raise ValueError("Episode differs from the frozen common evaluation: " + str(episode_path))
        episode = common.read_json(episode_path)
    else:
        if episode_path.exists():
            raise ValueError("A newly created episode requires a new common evaluation first")
        episode = {}
    report = episode.get("report")
    predictions, invalid, _ = common.report_events(report)
    scoped, scoped_invalid, excluded = common.score_scope(predictions, invalid, report, reference)
    scope = frozen_row["scoring_scope"]
    if ([e.get("id") for e in scoped] != scope["scored_prediction_event_ids"] or
            len(scoped_invalid) != scope["eligible_invalid_predictions"] or
            excluded != scope["excluded_predictions"] or
            reference["modulation_exhaustive"] != scope["modulation_exhaustive"]):
        raise ValueError("Reconstructed scope differs from saved common evaluation")
    gold = [e for e in reference["events"] if e.get("evaluation_required") is not False]
    if len(gold) != frozen_row["reference_event_count"]:
        raise ValueError("Exact reference event count differs from common evaluation")
    matching = frozen_row["event_metrics"]
    if matching is None:
        raise ValueError("Exact synthetic episode has no saved common node matching")
    mapping = _node_mapping(len(scoped), len(gold), matching)
    id_states = report_id_states(report, reference, predictions, invalid, scoped, matching)
    return dict(scene_id=scene["scene_id"], method=frozen_row["method"],
                stratum=scene["stratum"], domain=frozen_row["domain"],
                independent_group=frozen_row["independent_group"],
                execution_status=frozen_row["execution_status"],
                episode_sha256=frozen_row["episode_sha256"], exact_truth_sha256=reference["sha256"],
                reference_kind=reference["kind"], node_matching="saved_common_evaluator_matches_unmodified",
                frozen_node_matches=matching["matches"],
                ignored_optional_prediction_indices=matching.get("ignored_predictions", []),
                invalid_geometry_nodes_without_inferred_TF_edges=len(scoped_invalid),
                excluded_node_predictions=excluded, report_id_states=id_states,
                reported_temporal_relations=explicit_temporal_scores(report, gold, mapping, id_states),
                geometric_time_frequency_relations=geometric_tf_scores(scoped, gold, mapping,
                    matching.get("ignored_predictions", [])))


def verify_protocol(config_path):
    config = common.read_json(config_path)
    for filename, expected in config["sha256"].items():
        if common.digest(ROOT / filename) != expected:
            raise ValueError("Supplement frozen dependency changed: " + filename)
    if (config["geometry"]["narrowband_total_support_width_hz"] != SUPPORT_WIDTH_HZ or
            config["geometry"]["modulation_carrier_overlap_min_fraction_of_smaller"] != CARRIER_THRESHOLD or
            config["reported_temporal"]["minimum_overlap_fraction_of_shorter_strict"] != TEMPORAL_THRESHOLD):
        raise ValueError("Supplement source definitions and configuration differ")
    return config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--common-evaluation", type=Path)
    parser.add_argument("--config", type=Path, default=CONFIG)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    protocol = verify_protocol(args.config)
    require_cuda()
    run = args.run.resolve()
    if run.name not in protocol["eligible_run_names"]:
        raise ValueError("Run is not part of the predeclared secondary relation supplement")
    config = common.read_json(run / "run.json")
    common_dir = args.common_evaluation or run / "evaluation"
    frozen_summary = common.read_json(common_dir / "summary.json")
    manifest = Path(config["manifest"])
    if not manifest.is_absolute():
        manifest = ROOT / manifest
    if (common.digest(run / "run.json") != frozen_summary["run_configuration_sha256"] or
            common.digest(manifest) != config["manifest_sha256"] or
            config["manifest_sha256"] != frozen_summary["manifest_sha256"] or
            common.digest(ROOT / "src/sonar/metrics.py") != frozen_summary["metrics_code_sha256"] or
            common.digest(ROOT / "scripts/evaluate_benchmark.py") != frozen_summary["evaluator_code_sha256"]):
        raise ValueError("Common evaluation provenance differs from current immutable inputs")
    records = common.read_jsonl(manifest)[:config["scene_count"]]
    if len(records) != config["scene_count"] or len({r["scene_id"] for r in records}) != len(records):
        raise ValueError("Manifest scene count or identity mismatch")
    manifest_relative = manifest.relative_to(ROOT).as_posix()
    if manifest_relative not in protocol["eligible_manifests"]:
        raise ValueError("Manifest is not part of the predeclared secondary supplement")
    frozen_rows = common.read_jsonl(common_dir / "per_episode.jsonl")
    by_key = {(row["scene_id"], row["method"]): row for row in frozen_rows}
    expected = {(row["scene_id"], method) for row in records for method in config["methods"]}
    if len(by_key) != len(frozen_rows) or set(by_key) != expected:
        raise ValueError("Common per-episode rows differ from expected manifest/method grid")
    rows = [assess_episode(scene, run, by_key[(scene["scene_id"], method)])
            for scene in records if scene["stratum"].startswith("synthetic")
            for method in config["methods"]]
    results = {}
    for stratum in sorted({r["stratum"] for r in rows}):
        results[stratum] = {}
        for domain in ["all"] + sorted({r["domain"] for r in rows if r["stratum"] == stratum}):
            results[stratum][domain] = {}
            for method in config["methods"]:
                subset = [row for row in rows if row["stratum"] == stratum and row["method"] == method
                          and (domain == "all" or row["domain"] == domain)]
                if not subset:
                    continue
                result = {key: aggregate_event_metrics([r[key] for r in subset]) for key in METRIC_KEYS}
                result["group_sampling_intervals"] = {
                    key: common.group_metric_intervals(subset, key, protocol["bootstrap_repetitions"])
                    for key in METRIC_KEYS}
                result["zero_gold_edge_episodes"] = {
                    key: int(torch.tensor([r[key]["gold_edges"] == 0 for r in subset], device="cuda").sum().item())
                    for key in METRIC_KEYS}
                results[stratum][domain][method] = result
    output = args.output or run / "relation_supplement"
    if output.exists() and any(output.iterdir()):
        raise ValueError("Refusing to overwrite an existing relation supplement; use a new output directory")
    output.mkdir(parents=True, exist_ok=True)
    summary = dict(evaluated_at=datetime.now(timezone.utc).isoformat(),
        status="secondary_exact_synthetic_assessment", numerical_execution="cuda_only",
        device=torch.cuda.get_device_name(), run=str(run),
        config_sha256=common.digest(args.config), source_sha256=common.digest(Path(__file__)),
        run_configuration_sha256=common.digest(run / "run.json"), manifest_sha256=common.digest(manifest),
        common_summary_sha256=common.digest(common_dir / "summary.json"),
        common_per_episode_sha256=common.digest(common_dir / "per_episode.jsonl"),
        common_evaluation_status=frozen_summary["status"], exact_synthetic_episodes=len(rows),
        excluded_nonexact_episodes=len(expected) - len(rows),
        unqualified_real_metrics="unavailable", natural_language_reasoning_accuracy="not_scored",
        primary_scores_modified=False, results=results, definitions=protocol,
        execution_status_counts=dict(Counter(r["execution_status"] for r in rows)))
    common.write_json(output / "summary.json", summary)
    import json
    (output / "per_episode.jsonl").write_text("".join(json.dumps(row, allow_nan=False) + "\n" for row in rows))
    print(json.dumps({"output": str(output), "exact_synthetic_episodes": len(rows),
                      "primary_scores_modified": False}), flush=True)


if __name__ == "__main__":
    main()
