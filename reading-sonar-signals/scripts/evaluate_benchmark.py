#!/usr/bin/env python3
"""CUDA evaluation with separate exact-truth and AI-reference-agreement strata.

Examples:
  python scripts/evaluate_benchmark.py --run results/development --select-fixed \
    --references results/ai_annotations/consensus.jsonl
  python scripts/evaluate_benchmark.py --run results/main \
    --references results/ai_annotations/consensus.jsonl \
    --fixed-selection results/development/evaluation/fixed_selection.json

JSON parsing, hashing, schema shape checks and provenance connected components
are administrative CPU work. All numerical matching, metrics, uncertainty,
multiple-testing adjustment, and runtime/token aggregation execute on CUDA.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import torch
from sonar.signal import require_cuda
from sonar.metrics import (event_metrics, aggregate_event_metrics,
                           temporal_relation_metrics, grouped_bootstrap,
                           paired_group_permutation, holm)

MORPHOLOGIES = {"tone", "chirp", "transient", "broadband", "modulation"}


def read_json(path):
    return json.loads(Path(path).read_text())


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def reference_index(path):
    if path is None:
        return {}
    path = Path(path)
    entries = read_jsonl(path) if path.is_file() else [read_json(p) for p in sorted(path.glob("*.json"))]
    result = {}
    for item in entries:
        if item["scene_id"] in result:
            raise ValueError("Duplicate AI-reference scene ID")
        result[item["scene_id"]] = item
    return result


def parent_nodes(record):
    """Do not use a mixture's pair ID as an independent recording group."""
    if record.get("parents"):
        nodes = set()
        for parent in record["parents"]:
            nodes.update(parent_nodes(parent) if isinstance(parent, dict) else {"parent:" + str(parent)})
        return nodes
    group = record.get("group_id") or record.get("parent_group") or record.get("source_id")
    if group is None:
        raise ValueError(f"No independent-parent provenance for {record.get('scene_id')}")
    return {str(group)}


def connected_groups(records):
    """Union shared parents within each data stratum, not across headline strata.

    Synthetic mixtures must not unnecessarily merge independent real recordings
    in the untouched-real analysis; their own connected mixture groups do merge.
    """
    result, descriptions = {}, {}
    for stratum in sorted({row["stratum"] for row in records}):
        subset = [row for row in records if row["stratum"] == stratum]
        parents = {}
        def find(node):
            parents.setdefault(node, node)
            while parents[node] != node:
                parents[node] = parents[parents[node]]
                node = parents[node]
            return node
        for row in subset:
            nodes = sorted(parent_nodes(row))
            for node in nodes[1:]:
                a, b = find(nodes[0]), find(node)
                if a != b:
                    parents[max(a, b)] = min(a, b)
            find(nodes[0])
        members = {}
        for node in sorted(parents):
            members.setdefault(find(node), []).append(node)
        for row in subset:
            group_members = members[find(sorted(parent_nodes(row))[0])]
            group = stratum + ":cluster:" + hashlib.sha256("\n".join(group_members).encode()).hexdigest()[:16]
            result[row["scene_id"]] = group
            descriptions[group] = group_members
    return result, descriptions


def domain_name(record):
    if record.get("domain"):
        return record["domain"]
    if record.get("parents"):
        return "+".join(sorted({domain_name(p) for p in record["parents"] if isinstance(p, dict)})) + "_mixture"
    return "controlled"


def validate_events(values, ai_reference=False):
    """Validate geometry on CUDA; retain invalid entries as scored failures."""
    valid, invalid = [], []
    if not isinstance(values, list):
        return [], [{"index": None, "reason": "components/events is not a list"}]
    for index, value in enumerate(values):
        reason = None
        if not isinstance(value, dict):
            reason = "event is not an object"
        elif value.get("type") not in MORPHOLOGIES:
            reason = "outside frozen component morphology ontology"
        else:
            try:
                interval = torch.tensor(value["time_s"], device="cuda", dtype=torch.float64)
                frequency = torch.tensor(value["frequency_hz"], device="cuda", dtype=torch.float64)
                if interval.shape != (2,) or frequency.shape != (2,):
                    reason = "time/frequency bounds require two numbers"
                elif not bool(torch.isfinite(interval).all() & torch.isfinite(frequency).all()):
                    reason = "nonfinite geometry"
                elif not bool((interval[1] > interval[0]) & (frequency[1] >= frequency[0]) & (frequency[0] >= 0)):
                    reason = "unordered or negative-frequency bounds"
                domain = value.get("frequency_domain", "acoustic")
                if domain not in {"acoustic", "modulation"}:
                    reason = "unknown frequency domain"
                if value["type"] == "modulation" and domain != "modulation":
                    reason = "modulation event assigned to acoustic-frequency domain"
                if value["type"] != "modulation" and domain != "acoustic":
                    reason = "acoustic morphology assigned to modulation-frequency domain"
                if domain == "modulation":
                    band = torch.tensor(value["carrier_band_hz"], device="cuda", dtype=torch.float64)
                    if band.shape != (2,) or not bool(torch.isfinite(band).all() & (band[1] > band[0]) & (band[0] > 0)):
                        reason = "invalid or missing modulation carrier band"
                if value.get("frequency_track"):
                    track = torch.tensor(value["frequency_track"], device="cuda", dtype=torch.float64)
                    if track.ndim != 2 or track.shape[1] != 2 or not bool(torch.isfinite(track).all()):
                        reason = "invalid frequency track"
                    elif track.shape[0] > 1 and not bool((track[1:, 0] > track[:-1, 0]).all()):
                        reason = "frequency track times are not strictly increasing"
                if value["type"] == "chirp" and len(value.get("frequency_track", [])) < 2:
                    reason = "chirp requires an ordered frequency track"
                if ai_reference:
                    uncertainty = torch.tensor(value.get("frequency_uncertainty_hz", 0.), device="cuda")
                    if uncertainty.ndim or not bool(torch.isfinite(uncertainty) & (uncertainty >= 0)):
                        reason = "invalid AI frequency uncertainty"
            except (KeyError, TypeError, ValueError, RuntimeError, IndexError) as error:
                reason = "unusable event fields: " + str(error)[:160]
        if reason:
            invalid.append({"index": index, "reason": reason})
        else:
            valid.append(dict(value))
    return valid, invalid


def report_events(report):
    if not isinstance(report, dict):
        return [], [], False
    valid, invalid = validate_events(report.get("components", []))
    for event in valid:
        event["model_declared_bin_spacing_hz"] = event.get("bin_spacing_hz")
        # Report text cannot expand its own acceptance tolerance. The shared
        # assessment grid is 1 Hz; the declared 2-Hz minimum already dominates
        # the 0.25-Hz DEMON grid and longer-window acoustic grids.
        event["bin_spacing_hz"] = 1.
    schema = (isinstance(report.get("components"), list) and
              isinstance(report.get("relations"), list) and
              isinstance(report.get("claims"), list) and
              isinstance(report.get("unresolved_questions"), list) and
              isinstance(report.get("abstained"), bool) and not invalid)
    return valid, invalid, schema


def count_scores(tp, fp, fn):
    return aggregate_event_metrics([{"tp": tp, "fp": fp, "fn": fn}])


def score_events(predictions, invalid, gold, frequency_tolerance, box_iou_threshold=.3):
    predictions = [dict(event, bin_spacing_hz=1.) for event in predictions]
    measured = event_metrics(predictions, gold, frequency_tolerance_hz=frequency_tolerance,
                             box_iou_threshold=box_iou_threshold, exhaustive=True)
    # A malformed emitted component is an unsuccessful prediction, not an omission.
    if invalid:
        updated = count_scores(measured["tp"], measured["fp"] + len(invalid), measured["fn"])
        measured.update({key: updated[key] for key in ("tp", "fp", "fn", "precision", "recall", "f1")})
    measured["invalid_component_predictions"] = len(invalid)
    return measured


def duplicates(predictions, tolerance):
    """Diagnostic duplicate candidates: same morphology, >=.8 temporal IoU."""
    repeated = []
    representatives = []
    for index, event in enumerate(predictions):
        found = None
        for prior in representatives:
            match = event_metrics([event], [predictions[prior]], time_iou_threshold=.8,
                                  box_iou_threshold=.8, frequency_tolerance_hz=tolerance)
            if match["tp"]:
                found = prior; break
        if found is None:
            representatives.append(index)
        else:
            repeated.append([found, index])
    return {"duplicate_candidate_count": len(repeated), "pairs": repeated,
            "definition": "same morphology/domain and metric-compatible geometry at 0.8 overlap"}


def relation_scores(predictions, gold, matching):
    # Optional exact beat harmonics earn no TP, FP, FN, or relation penalties.
    required = [event for event in gold if event.get("evaluation_required") is not False]
    ignored = set(matching.get("ignored_predictions", []))
    retained = [i for i in range(len(predictions)) if i not in ignored]
    remap = {old: new for new, old in enumerate(retained)}
    matches = [dict(match, prediction_index=remap[match["prediction_index"]])
               for match in matching["matches"]]
    result = temporal_relation_metrics([predictions[i] for i in retained], required, matches)
    result["meaning"] = "geometric temporal-overlap relations induced by component intervals, not free-text relation correctness"
    return result


def event_in_reference_scope(event, reference):
    """Eligibility limits agreement targets, never certifies negative truth."""
    allowed = reference.get("eligible_types")
    domains = reference.get("eligible_frequency_domains")
    if (allowed is not None and not allowed) or domains == []:
        return False
    if not isinstance(event, dict):
        return True  # Unscopable malformed output remains a schema/prediction failure.
    kind = event.get("type")
    if allowed is not None and kind in MORPHOLOGIES and kind not in allowed:
        return False
    domain = event.get("frequency_domain", "modulation" if kind == "modulation" else "acoustic")
    if domains is not None and domain in {"acoustic", "modulation"} and domain not in domains:
        return False
    observed=reference.get('observed_view_scope',{})
    frequency_scope=observed.get(domain+'_frequency_hz')
    interval=event.get('frequency_hz')
    if frequency_scope is not None and isinstance(interval,list) and len(interval)==2 and all(type(v) in (int,float) for v in interval):
        bounds=torch.tensor([interval,frequency_scope],device='cuda',dtype=torch.float64)
        if bool(torch.isfinite(bounds).all()) and bool(bounds[0,1]>=bounds[0,0]):
            if not bool((bounds[0,0]>=bounds[1,0]) & (bounds[0,1]<=bounds[1,1])):return False
    bands=reference.get('eligible_carrier_bands_hz')
    if (domain=='modulation' or kind=='modulation') and bands is not None:
        if not bands:return False
        band=event.get('carrier_band_hz')
        if isinstance(band,list) and len(band)==2 and all(type(v) in (int,float) for v in band):
            geometry=torch.tensor([band]+bands,device='cuda',dtype=torch.float64)
            if bool(torch.isfinite(geometry).all()) and bool(geometry[0,1]>geometry[0,0]):
                overlap=(torch.minimum(geometry[0,1],geometry[1:,1])-torch.maximum(geometry[0,0],geometry[1:,0])).clamp_min(0)
                denominator=torch.minimum(geometry[0,1]-geometry[0,0],geometry[1:,1]-geometry[1:,0]).clamp_min(1e-8)
                if not bool((overlap/denominator>=.5).any()):return False
        # An unscopable malformed carrier remains a validation failure.
    return True


def resolve_reference(scene, run, ai_references, ai_max_uncertainty=8.):
    truth_path = run / "sealed_truth" / (scene["scene_id"] + ".json")
    truth = read_json(truth_path) if truth_path.exists() else None
    if scene["stratum"].startswith("synthetic"):
        if not truth or "events" not in truth:
            return None
        events, invalid = validate_events(truth["events"])
        if invalid:
            raise ValueError(f"Invalid exact synthetic reference for {scene['scene_id']}: {invalid}")
        modulation_exhaustive = truth.get("modulation_exhaustive", truth.get("kind") in {"am_noise", "beating", "mixed", "noise", "silence"})
        excluded_modulation_gold = len([e for e in events if e.get("frequency_domain") == "modulation"])
        if not modulation_exhaustive:
            events = [e for e in events if e.get("frequency_domain") != "modulation"]
        return {"kind": "exact_synthetic_component_truth", "events": events,
                "valid_for_primary": True, "reference_status": "exact",
                "modulation_exhaustive": modulation_exhaustive,
                "modulation_evaluation_scope": truth.get("modulation_evaluation_scope", "Frozen pre-test evaluator scope: modulation is exhaustive only in am_noise, beating, mixed, noise and silence."),
                "excluded_modulation_gold": excluded_modulation_gold if not modulation_exhaustive else 0,
                "independently_exhaustive": True, "physical_real_world_truth": False,
                "path": str(truth_path), "sha256": digest(truth_path), "invalid_reference_events": []}
    reference = ai_references.get(scene["scene_id"])
    if reference is None:
        return None
    annotation = reference.get("annotation", reference)
    status = annotation.get("reference_status", annotation.get("annotation_status", "unknown"))
    if status in {"pending", "invalid", "failed"}:
        return None
    allowed = annotation.get("eligible_types")
    domains = annotation.get("eligible_frequency_domains")
    carrier_scope=annotation.get('eligible_carrier_bands_hz')
    if allowed is not None and (not isinstance(allowed, list) or not set(allowed).issubset(MORPHOLOGIES)):
        raise ValueError("Reference eligible_types must be an explicit list from the frozen ontology")
    if domains is not None and (not isinstance(domains, list) or not set(domains).issubset({"acoustic", "modulation"})):
        raise ValueError("Invalid eligible_frequency_domains reference scope")
    raw_events = annotation.get("events", [])
    scoped_events = [event for event in raw_events if event_in_reference_scope(event, annotation)] if isinstance(raw_events, list) else raw_events
    events, invalid = validate_events(scoped_events, ai_reference=True)
    for event in events:
        original = event.get("frequency_uncertainty_hz", 0.)
        event["declared_ai_frequency_uncertainty_hz"] = original
        event["frequency_uncertainty_hz"] = torch.tensor(original, device="cuda").clamp(max=ai_max_uncertainty).item()
    ontology_scope = [kind for kind in MORPHOLOGIES if (allowed is None or kind in allowed)
                      and (domains is None or ('modulation' if kind == 'modulation' else 'acoustic') in domains)
                      and not (kind=='modulation' and carrier_scope==[])]
    scope_available = bool(ontology_scope) and status not in {"unavailable", "no_eligible_types", "no_eligible_classes"}
    return {"kind": annotation.get("reference_kind", "ai_selected_checkpoint_agreement_proxy" if allowed is not None else "ai_consensus_event_agreement_proxy"), "events": events,
            "valid_for_primary": status == "complete" and not invalid and scope_available,
            "scorable": scope_available, "eligible_types": allowed,
            "eligible_frequency_domains": domains,
            "eligible_carrier_bands_hz": carrier_scope,
            "observed_view_scope": annotation.get('observed_view_scope',{}),
            "excluded_reference_event_count": len(raw_events) - len(scoped_events) if isinstance(raw_events, list) else None,
            "selected_models_by_type": annotation.get("selected_models_by_type"),
            "eligibility_protocol_sha256": annotation.get("eligibility_protocol_sha256"),
            "reference_status": status,
            "independently_exhaustive": False, "physical_real_world_truth": False,
            "reference_record_sha256": hashlib.sha256(json.dumps(reference, sort_keys=True).encode()).hexdigest(),
            "disputed_regions": annotation.get("disputed_regions", []),
            "disputed_events": annotation.get("disputed_events", {}),
            "ai_interannotator_agreement": annotation.get("ai_interannotator_agreement"),
            "invalid_reference_events": invalid}


def score_scope(predictions, invalid, report, reference):
    def in_scope(event):
        if not event_in_reference_scope(event, reference):
            return False
        return reference.get("modulation_exhaustive", True) or not isinstance(event, dict) or (
            event.get("frequency_domain") != "modulation" and event.get("type") != "modulation")
    retained = [e for e in predictions if in_scope(e)]
    retained_invalid = []
    components = report.get("components", []) if isinstance(report, dict) else []
    for entry in invalid:
        index = entry.get("index")
        original = components[index] if isinstance(components, list) and index is not None and index < len(components) else None
        if not in_scope(original):
            continue
        retained_invalid.append(entry)
    return retained, retained_invalid, len(predictions) + len(invalid) - len(retained) - len(retained_invalid)


def evaluate_one(scene, method, run, reference, cluster, ai_tolerance):
    folder = run / scene["scene_id"] / method
    episode_path, failure_path = folder / "episode.json", folder / "failure.json"
    episode = read_json(episode_path) if episode_path.exists() else {}
    failure = read_json(failure_path) if failure_path.exists() and not episode else None
    status = "completed" if episode else "failed" if failure else "missing"
    report = episode.get("report")
    raw = episode.get("raw_report")
    predicted, invalid, schema = report_events(report)
    raw_predicted, raw_invalid, raw_schema = report_events(raw)
    report_schema_failed = not (schema and raw_schema and episode.get("schema_valid", raw_schema))
    actions = episode.get("actions", [])
    row = {
        "scene_id": scene["scene_id"], "method": method,
        "stratum": scene["stratum"], "domain": domain_name(scene),
        "independent_group": cluster, "execution_status": status,
        "episode_path": str(episode_path) if episode else None,
        "episode_sha256": digest(episode_path) if episode else None,
        "failure": failure, "schema_failed": report_schema_failed,
        "reported_component_count": len(predicted) + len(invalid),
        "valid_component_count": len(predicted), "invalid_components": invalid,
        "raw_valid_component_count": len(raw_predicted), "raw_invalid_components": raw_invalid,
        "empty_report": not bool(predicted),
        "declared_abstention": bool(report.get("abstained", False)) if isinstance(report, dict) else False,
        "source_identity_assertions": len([e for e in predicted if e.get("source_identity", "unknown") not in (None, "unknown", "not_defined")]),
        "reference_kind": reference["kind"] if reference else "unavailable",
        "reference_status": reference["reference_status"] if reference else "unavailable",
        "reference_valid_for_primary": reference["valid_for_primary"] if reference else False,
        "reference_eligible_types": reference.get("eligible_types") if reference else None,
        "reference_eligible_frequency_domains": reference.get("eligible_frequency_domains") if reference else None,
        "reference_eligible_carrier_bands_hz": reference.get("eligible_carrier_bands_hz") if reference else None,
        "reference_event_count": len([g for g in reference["events"] if g.get("evaluation_required") is not False]) if reference else None,
        "event_metrics": None, "raw_event_metrics": None, "geometric_overlap_metrics": None,
        "event_metrics_regional_iou05": None,
        "timing": {key: episode.get(key, failure.get(key, 0.) if failure else 0.)
                   for key in ("wall_seconds", "preprocess_gpu_seconds", "model_gpu_seconds", "tool_gpu_seconds", "input_tokens", "output_tokens")},
        "actions_attempted": len(actions),
        "actions_invalid_or_failed": len([a for a in actions if a.get("status") != "completed"]),
        "generation_failures": len([g for g in episode.get("generations", []) if g.get("parsed") is None]),
    }
    tolerance = 2. if scene["stratum"].startswith("synthetic") else ai_tolerance
    row["duplicate_diagnostic"] = duplicates(predicted, tolerance)
    if reference is not None:
        scoped, scoped_invalid, excluded = score_scope(predicted, invalid, report, reference)
        raw_scoped, raw_scoped_invalid, raw_excluded = score_scope(raw_predicted, raw_invalid, raw, reference)
        row["scoring_scope"] = {"modulation_exhaustive": reference.get("modulation_exhaustive", False),
                                "eligible_types": reference.get("eligible_types"),
                                "eligible_frequency_domains": reference.get("eligible_frequency_domains"),
                                "eligible_carrier_bands_hz": reference.get("eligible_carrier_bands_hz"),
                                "observed_view_scope": reference.get('observed_view_scope',{}),
                                "excluded_predictions": excluded,
                                "excluded_raw_predictions": raw_excluded,
                                "eligible_valid_predictions": len(scoped),
                                "eligible_invalid_predictions": len(scoped_invalid),
                                "scored_prediction_event_ids": [e.get("id") for e in scoped],
                                "reference_scorable": reference.get("scorable", True),
                                "excluded_modulation_gold": reference.get("excluded_modulation_gold", 0),
                                "rule": reference.get("modulation_evaluation_scope", "Only reference-eligible morphology/domain is scored. Other predictions are unscored, not false positives; eligible real scores remain AI agreement, not exhaustive truth.")}
        if not reference.get("scorable", True):
            row["reference_unavailable_reason"] = "No eligible AI-reference morphology/domain; primary agreement cannot be computed."
            return row
        row["event_metrics"] = score_events(scoped, scoped_invalid, reference["events"], tolerance)
        row["event_metrics_regional_iou05"] = score_events(scoped, scoped_invalid, reference["events"], tolerance, box_iou_threshold=.5)
        row["raw_event_metrics"] = score_events(raw_scoped, raw_scoped_invalid, reference["events"], tolerance)
        row["geometric_overlap_metrics"] = relation_scores(scoped, reference["events"], row["event_metrics"])
        if reference["kind"].startswith("ai_"):
            row["negative_region_interpretation"] = "Unmatched predictions are disagreement with an incomplete AI reference, not established physical false positives."
            row["reference_only_recall"] = event_metrics(scoped, reference["events"],
                frequency_tolerance_hz=tolerance, exhaustive=False)["recall"]
            row["disputed_region_count"] = len(reference["disputed_regions"])
            row["invalid_reference_events"] = reference["invalid_reference_events"]
    return row


def group_metric_intervals(rows, metric_key, repetitions=2000):
    """Individual-method recording-group intervals; no pseudo model replicates."""
    groups = sorted({r["independent_group"] for r in rows})
    if len(groups) < 2:
        return {"status": "insufficient_independent_groups", "independent_groups": len(groups)}
    device = require_cuda()
    index = torch.tensor([groups.index(r["independent_group"]) for r in rows], device=device)
    values = torch.tensor([[r[metric_key][k] for k in ("tp", "fp", "fn")] for r in rows], device=device, dtype=torch.float64)
    counts = torch.zeros((len(groups), 3), device=device, dtype=torch.float64)
    counts.scatter_add_(0, index[:, None].expand(-1, 3), values)
    generator = torch.Generator(device=device).manual_seed(603)
    sample = torch.randint(len(groups), (repetitions, len(groups)), device=device, generator=generator)
    tp, fp, fn = counts[sample].sum(dim=1).unbind(dim=-1)
    estimates = {
        "precision": torch.where(tp + fp > 0, tp / (tp + fp), torch.full_like(tp, float("nan"))),
        "recall": torch.where(tp + fn > 0, tp / (tp + fn), torch.full_like(tp, float("nan"))),
        "f1": torch.where(2 * tp + fp + fn > 0, 2 * tp / (2 * tp + fp + fn), torch.zeros_like(tp)),
    }
    result = {"status": "complete", "independent_groups": len(groups), "repetitions": repetitions,
              "uncertainty": "recording/provenance-group sampling only; one frozen model configuration",
              "seed": 603, "metrics": {}}
    quantile = torch.tensor([.025, .975], device=device, dtype=torch.float64)
    for name, values in estimates.items():
        valid = torch.isfinite(values)
        result["metrics"][name] = {
            "ci95": torch.quantile(values[valid], quantile).tolist() if bool(valid.any()) else None,
            "defined_resample_fraction": valid.float().mean().item(),
        }
    return result


def summarize(rows, repetitions=2000):
    device = require_cuda()
    scored = [row for row in rows if row["event_metrics"] is not None and row["reference_valid_for_primary"]]
    partial_scored = [row for row in rows if row["event_metrics"] is not None and not row["reference_valid_for_primary"]]
    result = {"expected_episodes": len(rows), "reference_scored_episodes": len(scored),
              "reference_missing_episodes": len(rows) - len(scored),
              "partial_reference_diagnostic_episodes": len(partial_scored),
              "partial_reference_diagnostic_metrics": aggregate_event_metrics([row["event_metrics"] for row in partial_scored]) if partial_scored else None,
              "independent_groups": len({row["independent_group"] for row in rows}),
              "reference_kinds": sorted({row["reference_kind"] for row in scored}),
              "reference_statuses": sorted({row["reference_status"] for row in rows}),
              "reference_scopes": [json.loads(value) for value in sorted({json.dumps({
                  "eligible_types": row.get("reference_eligible_types"),
                  "eligible_frequency_domains": row.get("reference_eligible_frequency_domains"),
                  "eligible_carrier_bands_hz": row.get("reference_eligible_carrier_bands_hz")}, sort_keys=True) for row in rows})],
              "event_metrics": aggregate_event_metrics([row["event_metrics"] for row in scored]) if scored else None,
              "event_metrics_regional_iou05": aggregate_event_metrics([row["event_metrics_regional_iou05"] for row in scored]) if scored else None,
              "secondary_iou_interpretation": "Regional broadband/transient boxes use IoU0.5; narrowband temporal/track criteria stay at their primary values. Secondary diagnostic, not a replacement primary endpoint.",
              "raw_event_metrics": aggregate_event_metrics([row["raw_event_metrics"] for row in scored]) if scored else None,
              "geometric_overlap_metrics": aggregate_event_metrics([row["geometric_overlap_metrics"] for row in scored]) if scored else None,
              "free_text_claim_correctness": "not_scored",
              "real_physical_correctness": "not_established"}
    counter_names = ["schema_failed", "empty_report", "declared_abstention", "source_identity_assertions",
                     "reported_component_count", "valid_component_count", "actions_attempted",
                     "actions_invalid_or_failed", "generation_failures"]
    counter = torch.tensor([[float(row[name]) for name in counter_names] for row in rows], device=device)
    totals = counter.sum(dim=0)
    result["totals"] = {name: int(value) for name, value in zip(counter_names, totals.tolist())}
    statuses = torch.tensor([[row["execution_status"] == s for s in ("completed", "failed", "missing")]
                             for row in rows], device=device).sum(dim=0)
    result["execution_status"] = dict(zip(("completed", "failed", "missing"), statuses.tolist()))
    result["report_emission_coverage"] = (1 - counter[:, counter_names.index("empty_report")].mean()).item()
    result["duplicate_candidates"] = int(torch.tensor([row["duplicate_diagnostic"]["duplicate_candidate_count"] for row in rows], device=device).sum().item())
    timing_names = ["wall_seconds", "preprocess_gpu_seconds", "model_gpu_seconds", "tool_gpu_seconds", "input_tokens", "output_tokens"]
    timings = torch.tensor([[row["timing"][key] or 0. for key in timing_names] for row in rows], device=device, dtype=torch.float64)
    result["compute"] = {name: {"total": timings[:, i].sum().item(), "mean_per_expected_episode": timings[:, i].mean().item(),
                               "median": timings[:, i].median().item(), "p95": torch.quantile(timings[:, i], .95).item()}
                         for i, name in enumerate(timing_names)}
    gpu = timings[:, 1:4].sum(dim=1)
    result["compute"]["cuda_event_seconds"] = {"total": gpu.sum().item(), "mean_per_expected_episode": gpu.mean().item()}
    result["compute"]["timing_note"] = "CUDA event elapsed spans are not occupancy-derived active time; failed/missing unrecorded work is zero-known-time, not evidence of zero actual cost."
    if scored:
        matched=[match for row in scored for match in row['event_metrics'].get('matches',[])]
        frequency_errors=[match['frequency_error_hz'] for match in matched if match.get('frequency_error_hz') is not None]
        result['matched_geometry_diagnostics']={
            'matched_components':len(matched), 'frequency_matched_components':len(frequency_errors),
            'mean_frequency_error_hz':torch.tensor(frequency_errors,device=device).mean().item() if frequency_errors else None,
            'median_frequency_error_hz':torch.tensor(frequency_errors,device=device).median().item() if frequency_errors else None,
            'mean_temporal_iou':torch.tensor([match['time_iou'] for match in matched],device=device).mean().item() if matched else None,
            'interpretation':'Conditional on accepted one-to-one matches; report with coverage, not as all-event accuracy.'}
        tp = torch.tensor([row["event_metrics"]["tp"] for row in scored], device=device, dtype=torch.float64).sum()
        fn = torch.tensor([row["event_metrics"]["fn"] for row in scored], device=device, dtype=torch.float64).sum()
        result["reference_event_coverage"] = (tp / (tp + fn)).item() if bool(tp + fn > 0) else None
        result["event_group_intervals"] = group_metric_intervals(scored, "event_metrics", repetitions)
        result["overlap_group_intervals"] = group_metric_intervals(scored, "geometric_overlap_metrics", repetitions)
    else:
        result['matched_geometry_diagnostics']=None
        result["reference_event_coverage"] = None
    return result


def grouped_error_test(raw_rows, verified_rows, group_ids, repetitions, seed=604):
    """H4 proxy: reduction in unmatched-component fraction, with coverage shown."""
    device = require_cuda(); groups = sorted(set(group_ids))
    if len(groups) < 2:
        return {"status": "insufficient_independent_groups"}
    index = torch.tensor([groups.index(g) for g in group_ids], device=device)
    def grouped(rows):
        values = torch.tensor([[r["tp"], r["fp"], r["fn"]] for r in rows], device=device, dtype=torch.float64)
        counts = torch.zeros((len(groups), 3), device=device, dtype=torch.float64)
        counts.scatter_add_(0, index[:, None].expand(-1, 3), values)
        return counts
    raw, verified = grouped(raw_rows), grouped(verified_rows)
    def error(counts):
        denominator = counts[..., 0] + counts[..., 1]
        return torch.where(denominator > 0, counts[..., 1] / denominator, torch.full_like(denominator, float("nan")))
    point = error(raw.sum(dim=0)) - error(verified.sum(dim=0))
    if not bool(torch.isfinite(point)):
        return {"status": "undefined_error_rate_due_to_no_emitted_components",
                "coverage_warning": "All-abstention precision is undefined, not perfect."}
    generator = torch.Generator(device=device).manual_seed(seed)
    draws = torch.randint(len(groups), (repetitions, len(groups)), generator=generator, device=device)
    values = error(raw[draws].sum(dim=1)) - error(verified[draws].sum(dim=1))
    valid = torch.isfinite(values)
    fraction = valid.float().mean().item()
    interval = torch.quantile(values[valid], torch.tensor([.025, .975], device=device, dtype=torch.float64)).tolist() if bool(valid.any()) else None
    swaps = torch.randint(2, (repetitions, len(groups), 1), generator=generator, device=device).bool()
    null = error(torch.where(swaps, raw[None], verified[None]).sum(dim=1)) - error(torch.where(swaps, verified[None], raw[None]).sum(dim=1))
    null_valid = torch.isfinite(null)
    p = ((1 + (null[null_valid].abs() >= point.abs() - 1e-12).sum()) / (1 + null_valid.sum())).item()
    stable = fraction >= .95 and bool(null_valid.float().mean() >= .95)
    return {"status": "complete" if stable else "unstable_due_to_empty_report_resamples",
            "metric": "raw_F_minus_verified_F_unmatched_component_fraction",
            "meaning": "component reference-agreement error proxy, not free-text supported-claim precision",
            "estimate": point.item(), "confidence_interval_95": interval,
            "p_value": p if stable else None, "independent_groups": len(groups),
            "bootstrap_valid_fraction": fraction, "repetitions": repetitions, "seed": seed,
            "test": "two_sided_paired_group_label_swap", "device": str(device)}


def primary_comparisons(rows, selection, repetitions):
    # Real recordings alone determine the primary family. No synthetic pooling.
    real = [row for row in rows if row["stratum"] == "real" and row["event_metrics"] is not None
            and row["reference_valid_for_primary"]]
    result = {"endpoint_interpretation": "agreement with blinded AI reference labels, not independently validated physical correctness",
              "H1": {"status": "fixed_comparator_not_selected_on_validation"},
              "H4": {"status": "no_F_episodes" if real else "no_eligible_real_reference_scores"},
              "family": ["H1 adaptive inspection", "H4 verifier component-agreement proxy"]}
    result["excluded_reference_ineligible_scene_ids"] = sorted({r["scene_id"] for r in rows if r["stratum"] == "real" and not r["reference_valid_for_primary"]})
    full = {r["scene_id"]: r for r in real if r["method"] == "F"}
    if full:
        ordered = [full[key] for key in sorted(full)]
        result["H4"] = grouped_error_test([r["raw_event_metrics"] for r in ordered],
                                          [r["event_metrics"] for r in ordered],
                                          [r["independent_group"] for r in ordered], repetitions)
        result["H4"]["verified_reference_coverage"] = aggregate_event_metrics([r["event_metrics"] for r in ordered])["recall"]
        raw_counts = aggregate_event_metrics([r["raw_event_metrics"] for r in ordered])
        result["H4"]["raw_reference_coverage"] = raw_counts["recall"]
    if selection and selection.get("selected_method") in {"G3", "G4"}:
        name = selection["selected_method"]
        comparison = {r["scene_id"]: r for r in real if r["method"] == name}
        shared = sorted(set(full) & set(comparison))
        if shared and set(full) == set(comparison):
            a, b = [full[k] for k in shared], [comparison[k] for k in shared]
            groups = [r["independent_group"] for r in a]
            if len(set(groups)) >= 2:
                aa, bb = [r["event_metrics"] for r in a], [r["event_metrics"] for r in b]
                result["H1"] = {"status": "complete", "comparison": "F_minus_" + name,
                    "bootstrap": grouped_bootstrap(aa, bb, groups, repetitions=repetitions),
                    **paired_group_permutation(aa, bb, groups, repetitions=repetitions),
                    "fixed_selection_provenance": selection}
            else:
                result["H1"] = {"status": "insufficient_independent_groups"}
        else:
            result["H1"] = {"status": "missing_or_unpaired_methods", "F_scenes": len(full), "comparator_scenes": len(comparison)}
    elif selection:
        result["H1"] = {"status": selection.get("status", "comparator_unavailable"), "selection_provenance": selection}
    values = [result[key].get("p_value") for key in ("H1", "H4")]
    adjusted = holm([p if p is not None else 1. for p in values])
    for key, p, value in zip(("H1", "H4"), values, adjusted):
        result[key]["holm_adjusted_p_value"] = value if p is not None else None
    result["holm_missing_test_policy"] = "Unavailable primary tests occupy a prespecified family slot with p=1; no test family is shrunk."
    return result


def select_fixed(rows, records, run, output, reference_path=None):
    if not all(row["split"] in ("validation", "val") for row in records):
        raise ValueError("Fixed comparator selection requires validation-only manifest")
    candidates = {}
    unavailable = []
    for method in ("G3", "G4"):
        subset = [r for r in rows if r["method"] == method and r["stratum"] == "real" and r["reference_valid_for_primary"]]
        if not subset:
            unavailable.append(method + ": no eligible primary real-reference scores")
            continue
        if any(r["execution_status"] == "missing" for r in subset):
            unavailable.append(method + ": missing validation episodes")
            continue
        candidates[method] = aggregate_event_metrics([r["event_metrics"] for r in subset])
        counts = torch.tensor([candidates[method]["tp"], candidates[method]["fn"]], device="cuda").sum()
        if not bool(counts > 0):
            unavailable.append(method + ": no eligible reference events in validation")
    selected = None
    if not unavailable:
        values = torch.tensor([candidates[m]["f1"] for m in ("G3", "G4")], device="cuda")
        selected = ("G3", "G4")[values.argmax().item()]
    result = {"status": "selected" if selected else "primary_comparator_unavailable", "unavailable_reasons": unavailable,
              "selected_method": selected, "source": "validation_only_real_AI_reference_event_agreement",
              "test_labels_used": False, "tie_rule": "G3", "candidates": candidates,
              "reference_source_sha256": digest(reference_path) if reference_path and Path(reference_path).is_file() else None,
              "validation_run": str(run), "run_configuration_sha256": digest(run / "run.json"),
              "validation_scene_ids": sorted({r["scene_id"] for r in rows if r["stratum"] == "real"}),
              "reference_ineligible_validation_scene_ids": sorted({r["scene_id"] for r in rows if r["stratum"] == "real" and not r["reference_valid_for_primary"]})}
    path = output / "fixed_selection.json"
    if path.exists() and read_json(path) != result:
        raise RuntimeError("Frozen fixed selection differs; do not overwrite after test exposure")
    write_json(path, result)
    return result


def self_test():
    """Meaningful CUDA smoke cases; never consumes or creates study outputs."""
    checks = []
    records = [
        {"scene_id": "rA", "stratum": "real", "group_id": "A"},
        {"scene_id": "rB", "stratum": "real", "group_id": "B"},
        {"scene_id": "mAB", "stratum": "semi_synthetic", "parents": [{"group_id": "A"}, {"group_id": "B"}]},
        {"scene_id": "mBC", "stratum": "semi_synthetic", "parents": [{"group_id": "B"}, {"group_id": "C"}]},
    ]
    grouped, _ = connected_groups(records)
    assert grouped["mAB"] == grouped["mBC"]
    assert grouped["rA"] != grouped["rB"]
    checks.append("shared-parent mixture connected groups without merging untouched-real groups")
    tone = {"id": "p1", "type": "tone", "frequency_domain": "acoustic", "time_s": [2., 8.],
            "frequency_hz": [100., 100.], "frequency_track": [[2., 100.], [8., 100.]],
            "source_identity": "unknown", "evidence_ids": []}
    result = score_events([tone, dict(tone, id="p2")], [], [tone], 2.)
    assert (result["tp"], result["fp"], result["fn"]) == (1, 1, 0)
    assert duplicates([tone, dict(tone, id="p2")], 2.)["duplicate_candidate_count"] == 1
    checks.append("duplicate predictions cannot match the same event twice")
    region=dict(tone,type='broadband',time_s=[0.,10.],frequency_hz=[100.,200.],frequency_track=[])
    partial=dict(region,time_s=[0.,4.])
    assert score_events([partial],[],[region],2.,box_iou_threshold=.3)['tp']==1
    assert score_events([partial],[],[region],2.,box_iou_threshold=.5)['tp']==0
    checks.append('secondary regional IoU0.5 is stricter than the declared primary IoU0.3')
    assert report_events({"nested": {"components": []}})[2] is False
    assert report_events({"components": [], "relations": [], "claims": [],
                          "unresolved_questions": [], "abstained": True})[2] is True
    valid, invalid = validate_events([dict(tone, time_s=[8., 2.])])
    assert not valid and len(invalid) == 1
    result = score_events([], invalid, [tone], 2.)
    assert (result["fp"], result["fn"]) == (1, 1)
    checks.append("wrong roots and invalid components remain explicit failures")
    modulation = dict(tone, id="mod", type="modulation", frequency_domain="modulation",
                      frequency_hz=[12., 12.], frequency_track=[[2., 12.], [8., 12.]], carrier_band_hz=[500., 2000.])
    scoped, bad, excluded = score_scope([tone, modulation], [], {"components": [tone, modulation]},
                                       {"modulation_exhaustive": False})
    assert scoped == [tone] and not bad and excluded == 1
    checks.append("unenumerated synthetic beating excluded without credit or penalty")
    scoped, bad, excluded = score_scope([tone, modulation], [], {"components": [tone, modulation]},
                                       {"eligible_types": ["tone"], "eligible_frequency_domains": ["acoustic"]})
    assert scoped == [tone] and not bad and excluded == 1
    scoped, bad, excluded = score_scope([tone, modulation], [], {"components": [tone, modulation]},
                                       {"eligible_types": [], "eligible_frequency_domains": []})
    assert not scoped and not bad and excluded == 2
    checks.append("ineligible AI-reference morphology is unscored; empty eligibility cannot create negatives")
    assert not event_in_reference_scope(None, {"eligible_frequency_domains": []})
    import tempfile
    with tempfile.TemporaryDirectory() as temporary:
        contradictory = resolve_reference({"scene_id": "fixture", "stratum": "real"}, Path(temporary),
            {"fixture": {"events": [], "reference_status": "complete", "eligible_types": ["tone"],
                         "eligible_frequency_domains": ["modulation"]}})
    assert not contradictory["scorable"] and not contradictory["valid_for_primary"]
    checks.append("empty morphology/domain intersection is unavailable, including malformed events in an empty domain scope")
    carrier_scope={'eligible_types':['modulation'],'eligible_carrier_bands_hz':[[500.,2000.],[2000.,5000.]]}
    assert event_in_reference_scope(modulation,carrier_scope)
    assert not event_in_reference_scope(dict(modulation,carrier_band_hz=[5000.,8000.]),carrier_scope)
    assert event_in_reference_scope(dict(modulation,carrier_band_hz=None),carrier_scope)
    checks.append('unobserved modulation carrier bands are unscored, not negative AI-reference labels')
    observed_scope={'observed_view_scope':{'acoustic_frequency_hz':[0,1800]}}
    assert event_in_reference_scope(tone,observed_scope)
    assert not event_in_reference_scope(dict(tone,frequency_hz=[2000.,2000.]),observed_scope)
    checks.append('components outside observed AI-reference frequency bounds are unscored')
    optional = dict(modulation, evaluation_required=False)
    scored = score_events([tone, modulation], [], [tone, optional], 2.)
    assert (scored["tp"], scored["fp"], scored["fn"]) == (1, 0, 0)
    relations = relation_scores([tone, modulation], [tone, optional], scored)
    assert (relations["tp"], relations["fp"], relations["fn"]) == (0, 0, 0)
    checks.append("optional beat harmonics excluded from event and relation penalties")
    raw = [{"tp": 1, "fp": 1, "fn": 0}] * 2
    verified = [{"tp": 1, "fp": 0, "fn": 0}] * 2
    error = grouped_error_test(raw, verified, ["a", "b"], 100)
    assert abs(error["estimate"] - .5) < 1e-8
    empty = grouped_error_test(raw, [{"tp": 0, "fp": 0, "fn": 1}] * 2, ["a", "b"], 100)
    assert empty["status"] == "undefined_error_rate_due_to_no_emitted_components"
    checks.append("H4 error proxy and all-abstention undefined precision")
    adjusted = holm([.01, .04])
    assert torch.allclose(torch.tensor(adjusted, device="cuda"), torch.tensor([.02, .04], device="cuda"))
    checks.append("two-comparison Holm correction on CUDA")
    print(json.dumps({"status": "self_test_passed", "checks": checks,
                      "device": torch.cuda.get_device_name(), "numeric_device": "cuda"}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path)
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--references", type=Path)
    parser.add_argument("--fixed-selection", type=Path)
    parser.add_argument("--select-fixed", action="store_true")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--ai-frequency-tolerance-hz", type=float, default=4.)
    parser.add_argument("--ai-max-frequency-uncertainty-hz", type=float, default=8.)
    args = parser.parse_args()
    require_cuda()
    if args.self_test:
        self_test()
        return
    if args.run is None:
        parser.error("--run is required unless --self-test is selected")
    if args.bootstrap < 100:
        raise ValueError("Use at least 100 uncertainty draws")
    run = args.run.resolve(); config = read_json(run / "run.json")
    manifest = args.manifest or Path(config["manifest"])
    records = read_jsonl(manifest)
    if len(records) > config["scene_count"]:
        records = records[:config["scene_count"]]
    if len(records) != config["scene_count"]:
        raise ValueError("Manifest scene count differs from run configuration")
    if digest(manifest) != config["manifest_sha256"]:
        raise ValueError("Manifest hash differs from the frozen run configuration")
    if len({row["scene_id"] for row in records}) != len(records):
        raise ValueError("Duplicate manifest scene IDs")
    output = args.output or run / "evaluation"
    output.mkdir(parents=True, exist_ok=True)
    ai_references = reference_index(args.references)
    clusters, cluster_members = connected_groups(records)
    references = {scene["scene_id"]: resolve_reference(scene, run, ai_references, args.ai_max_frequency_uncertainty_hz) for scene in records}
    rows = [evaluate_one(scene, method, run, references[scene["scene_id"]],
                         clusters[scene["scene_id"]], args.ai_frequency_tolerance_hz)
            for scene in records for method in config["methods"]]
    selection = read_json(args.fixed_selection) if args.fixed_selection else None
    if selection and (selection.get("test_labels_used") is not False or not (
            selection.get("selected_method") in {"G3", "G4"} or (
                selection.get("selected_method") is None and selection.get("status") == "primary_comparator_unavailable"))):
        raise ValueError("Invalid validation fixed-selection record")
    if selection and selection.get("reference_source_sha256") and (
            not args.references or not args.references.is_file() or digest(args.references) != selection["reference_source_sha256"]):
        raise ValueError("Test reference file differs from validation comparator-selection references")
    if selection and set(selection.get("validation_scene_ids", [])) & {r["scene_id"] for r in records}:
        raise ValueError("Validation comparator-selection scenes overlap this evaluation manifest")
    if args.select_fixed:
        selection = select_fixed(rows, records, run, output, args.references)
    results = {}
    for stratum in sorted({r["stratum"] for r in rows}):
        results[stratum] = {}
        domains = ["all"] + sorted({r["domain"] for r in rows if r["stratum"] == stratum})
        for domain in domains:
            results[stratum][domain] = {}
            for method in config["methods"]:
                subset = [r for r in rows if r["stratum"] == stratum and r["method"] == method
                          and (domain == "all" or r["domain"] == domain)]
                results[stratum][domain][method] = summarize(subset, args.bootstrap)
    failures = [{k: r[k] for k in ("scene_id", "method", "execution_status", "schema_failed", "failure", "invalid_components", "episode_path")}
                for r in rows if r["execution_status"] != "completed" or r["schema_failed"]]
    # Evaluate annotation consensus on the exact synthetic scenes where both exist.
    annotation_controls = []
    for scene in records:
        reference = references[scene["scene_id"]]
        if scene["stratum"] == "synthetic" and reference and scene["scene_id"] in ai_references:
            ai = ai_references[scene["scene_id"]]
            annotation = ai.get("annotation", ai)
            if annotation.get("eligible_types") == [] or annotation.get("eligible_frequency_domains") == []:
                continue
            raw_events = annotation.get("events", [])
            predicted, invalid = validate_events(raw_events, ai_reference=True)
            predicted, invalid, excluded_ai = score_scope(predicted, invalid, {"components": raw_events}, annotation)
            predicted, invalid, excluded = score_scope(predicted, invalid, {"components": raw_events}, reference)
            eligible_gold = [event for event in reference["events"] if event_in_reference_scope(event, annotation)]
            annotation_controls.append({"scene_id": scene["scene_id"],
                "metrics": score_events(predicted, invalid, eligible_gold, 2.),
                "excluded_modulation_predictions": excluded,
                "excluded_ineligible_predictions": excluded_ai,
                "eligible_types": annotation.get("eligible_types"),
                "interpretation": "AI annotation accuracy on exact controlled synthetic truth only"})
    missing = [r for r in rows if r["execution_status"] == "missing"]
    comparisons = primary_comparisons(rows, selection, args.bootstrap)
    if missing:
        comparisons["inference_status"] = "provisional_incomplete_run; missing episodes scored as empty reports"
    summary = {
        "evaluated_at": datetime.now(timezone.utc).isoformat(),
        "status": "incomplete_missing_episodes" if missing else "complete_evaluation",
        "run": str(run), "run_configuration_sha256": digest(run / "run.json"),
        "manifest_sha256": digest(manifest), "metrics_code_sha256": digest(ROOT / "src/sonar/metrics.py"),
        "evaluator_code_sha256": digest(Path(__file__)),
        "reference_source": str(args.references) if args.references else None,
        "reference_source_sha256": digest(args.references) if args.references and args.references.is_file() else None,
        "device": torch.cuda.get_device_name(), "numerical_execution": "cuda_only",
        "methods": config["methods"], "results": results, "primary_comparisons": comparisons,
        "matching": {"algorithm": "greedy_descending_quality_one_to_one", "time_iou": .3, "box_iou": .3,
                     "exact_synthetic_minimum_frequency_tolerance_hz": 2.,
                     "ai_reference_minimum_frequency_tolerance_hz": args.ai_frequency_tolerance_hz,
                     "maximum_uncalibrated_ai_reference_frequency_uncertainty_hz": args.ai_max_frequency_uncertainty_hz,
                     "AI_reference_eligibility": "Per-record eligible morphology/domain scopes filter both predictions and gold. Unqualified types receive neither credit nor penalty; empty scope produces unavailable metrics and cannot select a primary comparator.",
                     "ridge_tolerance": "max(minimum tolerance, 2Hz from the shared 1Hz assessment grid, reference frequency uncertainty); model-declared bins cannot relax acceptance",
                     "optional_synthetic_events": "Known optional beat-envelope harmonics give neither TP credit nor FP/FN penalty.",
                     "modulation_scope": "Synthetic modulation is scored only for am_noise, beating, mixed, noise and silence. For truth.modulation_exhaustive=false, modulation predictions and references are excluded without credit or penalty.",
                     "time_uncertainty": "Temporal IoU rule is fixed; stated boundary uncertainty is retained in annotations, not silently used to enlarge boxes."},
        "grouping": {"rule": "connected components of shared parent provenance within each stratum",
                     "cluster_members": cluster_members},
        "annotation_calibration_on_exact_synthetic": {
            "scenes": annotation_controls,
            "aggregate": aggregate_event_metrics([r["metrics"] for r in annotation_controls]) if annotation_controls else None},
        "interpretation": [
            "Real and real-mixture scores measure agreement with AI references, not independently validated physical correctness.",
            "The distinct annotation checkpoints share Qwen-family language backbones; blind passes are not statistically independent and correlated errors remain possible.",
            "Unmatched predictions in disputed or unreviewed real regions are disagreements with the reference set, not certified false positives.",
            "Exact synthetic metrics apply only to the declared generator and cannot validate real source identity.",
            "Schema/model failures remain in all applicable denominators as unsuccessful or empty reports.",
            "Data strata are never pooled into a headline accuracy value.",
            "No free-text claim fidelity or calibrated probability score is inferred from the verifier.",
            "Repeated frozen deterministic generations do not establish independent model-seed variability.",
        ],
    }
    write_json(output / "summary.json", summary)
    write_json(output / "failed_runs.json", {"failures": failures, "missing_reference_scene_ids": [sid for sid, ref in references.items() if ref is None],
        "ineligible_reference_scene_ids": [sid for sid, ref in references.items() if ref and not ref.get("valid_for_primary")]})
    with (output / "per_episode.jsonl").open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row, allow_nan=False) + "\n")
    cells, claims = [], []
    for stratum, domains in results.items():
        for domain, methods in domains.items():
            for method, value in methods.items():
                for metric in ("precision", "recall", "f1"):
                    cell = {"stratum": stratum, "domain": domain, "method": method, "metric": metric,
                            "value": value["event_metrics"][metric] if value["event_metrics"] else None,
                            "source": "summary.json", "json_pointer": f"/results/{stratum}/{domain}/{method}/event_metrics/{metric}",
                            "reference_scored_episodes": value["reference_scored_episodes"],
                            "interpretation": "exact synthetic component score" if stratum.startswith("synthetic") else "AI-reference event-agreement proxy"}
                    cells.append(cell)
                    if domain == "all":
                        claims.append(dict(cell, claim_id=f"{stratum}_{method}_{metric}",
                                           unsupported_extensions=["real physical source correctness", "free-text claim truth"]))
    write_json(output / "table_cells.json", cells)
    write_json(output / "claim_to_evidence.json", claims)
    print(json.dumps({"status": summary["status"], "output": str(output),
                      "episodes": len(rows), "failures": len(failures)}))


if __name__ == "__main__":
    main()
