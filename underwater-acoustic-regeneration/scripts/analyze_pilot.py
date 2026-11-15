"""Freeze diagnostics, calibrate on development groups, then assess the dry test."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np

from acoustic_regeneration.diagnostics.calibration import CalibrationCase, calibrate, freeze_thresholds
from acoustic_regeneration.diagnostics.predictor import fit_predictor, predict_deltas
from acoustic_regeneration.evaluation.statistics import paired_bootstrap
from acoustic_regeneration.registry import append_record, file_hash

ROOT = Path(__file__).resolve().parents[1]
POLICIES = ("full", "no_probes", "duration_only", "dispersion_only")


def features(case):
    path = ROOT / case["pool_path"]
    if file_hash(path) != case["pool_sha256"]:
        raise ValueError("Pool checksum mismatch")
    with np.load(path, allow_pickle=False) as pool:
        return pool["features"].copy()


def selected(predictors, name, matrix):
    selector = predictors["full"] if name in {"duration_only", "dispersion_only"} else predictors[name]
    index = int(np.argmin(predict_deltas(selector, matrix)))
    score = float(predict_deltas(predictors[name], matrix[index:index + 1])[0])
    return index, score


def load_cases(directory, role):
    path = directory / "cases.jsonl"
    result = [case for case in map(json.loads, path.read_text().splitlines()) if case["pilot_role"] == role]
    if len({case["group_id"] for case in result}) != len(result):
        raise ValueError("A dry calibration/test role may contain only one case per group")
    expected_partition = "D" if role == "D" else "V"
    if not result or any(case["source_partition"] != expected_partition for case in result):
        raise ValueError("Pilot roles must never draw from actual C/test")
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", default="pilot_pools_seed17")
    parser.add_argument("--stage", choices=["fit", "calibrate", "evaluate"], required=True)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--git-commit", required=True)
    args = parser.parse_args()
    directory = ROOT / "results" / args.run_id
    model_path = ROOT / "results/pools" / args.run_id / "predictors.joblib"
    metadata_path = directory / "frozen_predictors.json"
    if args.stage == "fit":
        if (directory / "calibration.json").exists():
            raise ValueError("Do not refit predictors after opening calibration")
        training, validation = load_cases(directory, "D"), load_cases(directory, "V")
        if len(training) != 200 or len(validation) != 64:
            raise ValueError("Complete all 200 D and 64 tuning V pools before fitting")
        if {case["group_id"] for case in training} & {case["group_id"] for case in validation}:
            raise ValueError("Diagnostic/validation groups overlap")
        matrices = [features(case) for case in training]
        targets = [np.array([item["Q"] for item in case["candidate_metrics"]]) - case["fallback_metrics"]["Q"] for case in training]
        all_features = np.concatenate(matrices)
        all_targets = np.concatenate(targets)
        groups = [case["group_id"] for case in training for _ in range(4)]
        predictors = {}
        for name in ("full", "no_probes"):
            predictors[name] = fit_predictor(all_features, all_targets, groups, ["D"] * len(groups), args.seed, name)
        # Gate-only controls retain the full method's candidate selection. Otherwise
        # identical duration features would arbitrarily choose candidate zero and
        # confound the gate comparison with a different candidate selector.
        chosen = [int(np.argmin(predict_deltas(predictors["full"], matrix))) for matrix in matrices]
        selected_features = np.stack([matrix[index] for matrix, index in zip(matrices, chosen)])
        selected_targets = np.array([target[index] for target, index in zip(targets, chosen)])
        for name in ("duration_only", "dispersion_only"):
            predictors[name] = fit_predictor(selected_features, selected_targets,
                                              [case["group_id"] for case in training], ["D"] * len(training), args.seed, name)
        thresholds = {}
        for name in POLICIES:
            scores = [selected(predictors, name, features(case))[1] for case in validation]
            thresholds[name] = freeze_thresholds(scores)
        joblib.dump(predictors, model_path)
        opportunity = [min(item["Q"] for item in case["candidate_metrics"]) <= case["fallback_metrics"]["Q"] - 0.02
                       for case in validation]
        metadata = {"git_commit": args.git_commit, "predictor_path": str(model_path.relative_to(ROOT)),
                    "predictor_sha256": file_hash(model_path), "training_groups": len(training),
                    "validation_groups": len(validation), "thresholds": thresholds,
                    "gate_only_controls_share_full_selector": True,
                    "oracle_opportunity_fraction_on_tuning_V": float(np.mean(opportunity)),
                    "oracle_opportunity_cases": int(sum(opportunity)),
                    "oracle_feasibility_minimum": 0.20, "oracle_feasibility_passed": bool(np.mean(opportunity) >= 0.20),
                    "scientific_test_results": False}
        metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")
        print(json.dumps(metadata, indent=2))
        return
    metadata = json.loads(metadata_path.read_text())
    if file_hash(model_path) != metadata["predictor_sha256"]:
        raise ValueError("Frozen predictor artifact changed")
    predictors = joblib.load(model_path)  # Trusted artifact created by the fit stage.
    if args.stage == "calibrate":
        cases = load_cases(directory, "dev_C")
        if len(cases) != 64:
            raise ValueError("Dry calibration needs its 64 reserved V groups")
        results = {}
        for name in POLICIES:
            calibration_cases = []
            for case in cases:
                index, score = selected(predictors, name, features(case))
                calibration_cases.append(CalibrationCase(case["group_id"], score,
                    case["candidate_metrics"][index]["Q"], case["fallback_metrics"]["Q"]))
            results[name] = calibrate(calibration_cases, metadata["thresholds"][name]).to_dict()
        output = {"stage": "development_only_dry_calibration", "real_C_opened": False,
                  "predictor_sha256": metadata["predictor_sha256"], "policies": results,
                  "git_commit": args.git_commit}
        (directory / "calibration.json").write_text(json.dumps(output, indent=2) + "\n")
        print(json.dumps(output, indent=2))
        return
    calibration = json.loads((directory / "calibration.json").read_text())
    if calibration["predictor_sha256"] != metadata["predictor_sha256"]:
        raise ValueError("Predictor changed after calibration")
    cases = load_cases(directory, "dev_test")
    if len(cases) != 64:
        raise ValueError("Dry test requires its 64 reserved V groups")
    baseline = np.array([case["fallback_metrics"]["Q"] for case in cases])
    groups = [case["group_id"] for case in cases]
    summaries = {}
    def acoustic_summary(measurements):
        return {"mean_gap_nmse": float(np.mean([item["gap_nmse"] for item in measurements])),
                "mean_gap_spectral_convergence": float(np.mean([item["gap_spectral_convergence"] for item in measurements])),
                "gap_nmse_p95": float(np.quantile([item["gap_nmse"] for item in measurements], 0.95)),
                "gap_nmse_p99": float(np.quantile([item["gap_nmse"] for item in measurements], 0.99)),
                "maximum_observed_drift": max(item["observed_max_drift"] for item in measurements)}
    full_choices = [selected(predictors, "full", features(case))[0] for case in cases]
    always_accept = np.array([case["candidate_metrics"][index]["Q"] for case, index in zip(cases, full_choices)])
    summaries["B"] = {"mean_Q": float(baseline.mean()), "generative_coverage": 0.0,
                       **acoustic_summary([case["fallback_metrics"] for case in cases])}
    for name, choices in (("C1", [0] * len(cases)), ("C2", [case["medoid"] for case in cases]), ("C3", full_choices)):
        values = np.array([case["candidate_metrics"][index]["Q"] for case, index in zip(cases, choices)])
        summaries[name] = {"mean_Q": float(values.mean()), "generative_coverage": 1.0,
                           "all_case_harm_rate": float(np.mean(values - baseline > 0.02)),
                           **acoustic_summary([case["candidate_metrics"][index] for case, index in zip(cases, choices)])}
    outputs = []
    for name in POLICIES:
        chosen = [selected(predictors, name, features(case)) for case in cases]
        threshold = calibration["policies"][name]["threshold"]
        accepted = np.array([threshold is not None and np.isfinite(score) and score <= threshold for _, score in chosen])
        candidate_q = np.array([case["candidate_metrics"][index]["Q"] for case, (index, _) in zip(cases, chosen)])
        output_q = np.where(accepted, candidate_q, baseline)
        harmful = (candidate_q - baseline > 0.02) & accepted
        summaries[name] = {"mean_Q": float(output_q.mean()), "generative_coverage": float(accepted.mean()),
                           "accepted_cases": int(accepted.sum()), "harmful_accepted_cases": int(harmful.sum()),
                           "conditional_accepted_harm": float(harmful.sum() / accepted.sum()) if accepted.any() else None,
                           "all_case_harm_rate": float(harmful.mean()), "threshold": threshold,
                           **acoustic_summary([case["candidate_metrics"][index] if accept else case["fallback_metrics"]
                                               for case, (index, _), accept in zip(cases, chosen, accepted)]),
                           "Q_difference_vs_B": paired_bootstrap(groups, output_q, baseline),
                           "Q_difference_vs_C3": paired_bootstrap(groups, output_q, always_accept)}
        for case, (index, score), accept, q, harm in zip(cases, chosen, accepted, output_q, harmful):
            outputs.append({"case_id": case["case_id"], "group_id": case["group_id"], "policy": name,
                            "selected_index": index, "score": score, "accepted": bool(accept),
                            "output_Q": float(q), "harmful": bool(harm), "role": "dev_test"})
    (directory / "dry_test_decisions.jsonl").write_text("".join(json.dumps(row) + "\n" for row in outputs))
    summary = {"stage": "development_only_pilot", "actual_test_opened": False, "actual_calibration_opened": False,
               "groups": len(cases), "policies": summaries,
               "oracle_opportunity_fraction_on_tuning_V": metadata["oracle_opportunity_fraction_on_tuning_V"],
               "oracle_feasibility_passed": metadata["oracle_feasibility_passed"],
               "all_preregistered_scientific_success_criteria": "untested on real held-out C/test",
               "unrun_controls": ["C4", "C5", "C6", "utility classifier", "external archive", "three training seeds"],
               "git_commit": args.git_commit}
    (directory / "pilot_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    append_record(ROOT / "results/run_registry.jsonl", {"run_id": args.run_id + "_analysis", "status": "completed",
                  "git_commit": args.git_commit, "predictor_sha256": metadata["predictor_sha256"],
                  "output_path": str((directory / "pilot_summary.json").relative_to(ROOT))})
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
