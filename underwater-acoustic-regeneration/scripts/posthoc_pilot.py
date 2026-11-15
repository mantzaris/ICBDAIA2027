"""Descriptive development diagnostics, without changing any frozen decision."""
from __future__ import annotations

import gzip
import json
from pathlib import Path

import numpy as np

from acoustic_regeneration.data.preprocessing import crop_source
from acoustic_regeneration.evaluation.statistics import paired_bootstrap
from acoustic_regeneration.registry import append_record, file_hash

ROOT = Path(__file__).resolve().parents[1]


def main():
    directory = ROOT / "results/pilot_pools_clipped_seed17"
    summary = json.loads((directory / "pilot_summary.json").read_text())
    assert summary["stage"] == "development_only_pilot" and not summary["actual_test_opened"]
    cases = [case for case in map(json.loads, (directory / "cases.jsonl").read_text().splitlines()) if case["pilot_role"] == "dev_test"]
    decisions = list(map(json.loads, (directory / "dry_test_decisions.jsonl").read_text().splitlines()))
    with gzip.open(ROOT / "manifests/pilot_sources.jsonl.gz", "rt") as stream:
        sources = {row["original_file_id"]: row for row in map(json.loads, stream)}
    by_id = {case["case_id"]: case for case in cases}
    curves = []
    for policy in ("full", "no_probes", "duration_only", "dispersion_only"):
        rows = sorted((row for row in decisions if row["policy"] == policy), key=lambda row: (row["score"], row["case_id"]))
        deltas = np.array([by_id[row["case_id"]]["candidate_metrics"][row["selected_index"]]["Q"] -
                           by_id[row["case_id"]]["fallback_metrics"]["Q"] for row in rows])
        for count in range(1, len(rows) + 1):
            curves.append({"policy": policy, "accepted_count_for_ranking_diagnostic": count, "coverage": count / len(rows),
                           "conditional_harm": float(np.mean(deltas[:count] > 0.02)),
                           "all_case_harm": float(np.sum(deltas[:count] > 0.02) / len(rows)),
                           "Q_difference_vs_B": float(np.sum(deltas[:count]) / len(rows)),
                           "maximum_score": rows[count - 1]["score"], "calibrated_policy": False})
    (directory / "ranking_curves.json").write_text(json.dumps({"scope": "descriptive development ranking only; accepts positive scores counterfactually; never a recalibrated policy",
               "tie_break": "score then case ID", "rows": curves}, indent=2) + "\n")
    full = {row["case_id"]: row for row in decisions if row["policy"] == "full"}
    if any(row["accepted"] for row in full.values()):
        raise ValueError("This no-go report expects the frozen all-fallback policy")
    floor = json.loads((ROOT / "checkpoints/pilot_conditional_seed17/config.json").read_text())["rms_floor"]
    secondary = []
    for case in cases:
        reference = crop_source(np.load(ROOT / sources[case["source_id"]]["processed_path"], allow_pickle=False))
        target = (reference - case["normalization_mean"]) / case["normalization_scale"]
        path = ROOT / case["pool_path"]
        if file_hash(path) != case["pool_sha256"]:
            raise ValueError("Pool checksum changed")
        with np.load(path, allow_pickle=False) as pool:
            chosen = full[case["case_id"]]["selected_index"]
            estimates = {"B": pool["fallback_gap"], "C1": pool["candidates_gap"][0],
                         "C2": pool["candidates_gap"][case["medoid"]], "C3": pool["candidates_gap"][chosen]}
            estimates["P"] = estimates["C3"] if full[case["case_id"]]["accepted"] else estimates["B"]
            a, b = case["gap_start"], case["gap_end"]
            for method, estimate in estimates.items():
                error = float(np.sum((estimate - target[a:b]) ** 2))
                secondary.append({"case_id": case["case_id"], "group_id": case["group_id"], "method": method,
                                  "full_window_nmse_secondary": error / max(float(np.dot(target, target)),
                                                                           len(target) * (floor / case["normalization_scale"]) ** 2),
                                  "gap_nmse": case["fallback_metrics"]["gap_nmse"] if method in {"B", "P"} else
                                     case["candidate_metrics"][{"C1": 0, "C2": case["medoid"], "C3": chosen}[method]]["gap_nmse"]})
    (directory / "secondary_metrics.json").write_text(json.dumps(secondary, indent=2) + "\n")
    baseline = [case["fallback_metrics"]["Q"] for case in cases]
    selected = [case["candidate_metrics"][full[case["case_id"]]["selected_index"]]["Q"] for case in cases]
    dates = [sources[case["source_id"]]["date"] for case in cases]
    sensitivity = paired_bootstrap(dates, baseline, selected)
    sensitivity.update(scope="date-balanced development sensitivity only; distinct estimand from vessel/source-balanced mean",
                       actual_test_opened=False, unique_dates=len(set(dates)))
    (directory / "date_sensitivity.json").write_text(json.dumps(sensitivity, indent=2) + "\n")
    report = {"status": "completed", "run_id": "pilot_posthoc_descriptive", "stage": "development_only",
              "input_summary_sha256": file_hash(directory / "pilot_summary.json"), "decision_policy_changed": False,
              "script_sha256": file_hash(Path(__file__)),
              "ranking_curves_calibrated": False, "dry_test_cases": len(cases), "unique_dates": len(set(dates)),
              "output_path": str(directory.relative_to(ROOT))}
    append_record(ROOT / "results/local_run_registry.jsonl", report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
