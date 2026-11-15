"""Frozen-policy synthetic identifiability and development-only shift tests."""
from __future__ import annotations

import argparse
import gzip
import json
import time
from collections import defaultdict
from pathlib import Path

import joblib
import numpy as np
import torch

from acoustic_regeneration.data.observation import Observation, normalize_observation
from acoustic_regeneration.data.preprocessing import crop_source
from acoustic_regeneration.data.synthetic import signal_cases, stress_masks
from acoustic_regeneration.diagnostics.calibration import CalibrationResult
from acoustic_regeneration.diagnostics.policy import decide, generate_batched_pool
from acoustic_regeneration.diagnostics.predictor import predict_deltas
from acoustic_regeneration.evaluation.metrics import MetricFloors, gap_metrics
from acoustic_regeneration.registry import append_record, file_hash, remaining_seconds
from acoustic_regeneration.restoration.classical import bidirectional_ar
from acoustic_regeneration.restoration.learned import deterministic_restore, load_ema_model

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", default="pilot_pools_clipped_seed17")
    parser.add_argument("--realizations", type=int, default=100)
    parser.add_argument("--git-commit", required=True)
    args = parser.parse_args()
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    directory = ROOT / "results" / args.run_id
    metadata = json.loads((directory / "frozen_predictors.json").read_text())
    pool_config = json.loads((directory / "pool_config.json").read_text())
    calibration = json.loads((directory / "calibration.json").read_text())["policies"]["full"]
    frozen = CalibrationResult(calibration["threshold"], calibration["groups"], calibration["multiplicity"],
                               calibration["family_error"], calibration["harm_limit"], ())
    for path, expected in (("checkpoints/pilot_conditional_seed17/best.pt", pool_config["conditional_sha256"]),
                           ("checkpoints/pilot_deterministic_seed17/best.pt", pool_config["deterministic_sha256"])):
        if file_hash(ROOT / path) != expected:
            raise ValueError("Frozen generator/fallback checkpoint changed")
    model, config, _ = load_ema_model(ROOT / "checkpoints/pilot_conditional_seed17/best.pt")
    deterministic, _, _ = load_ema_model(ROOT / "checkpoints/pilot_deterministic_seed17/best.pt")
    predictor_path = ROOT / metadata["predictor_path"]
    if file_hash(predictor_path) != metadata["predictor_sha256"]:
        raise ValueError("Frozen predictor changed")
    predictor = joblib.load(predictor_path)["full"]
    chosen_fallback = json.loads((directory / "frozen_fallback.json").read_text())["selected_fallback"]
    fallback = ((lambda obs: deterministic_restore(deterministic, obs)) if chosen_fallback == "deterministic"
                else (lambda obs: bidirectional_ar(obs, int(chosen_fallback.split("_")[1]))))
    floor = config["rms_floor"]
    output = ROOT / "results" / (args.run_id + "_diagnostics")
    output.mkdir(exist_ok=True)
    arrays = ROOT / "results/pools" / (args.run_id + "_diagnostics")
    arrays.mkdir(parents=True, exist_ok=True)
    if (output / "cases.jsonl").exists():
        raise ValueError("Preserve prior diagnostic outcomes; use a new run or implement a verified resume")
    deadline = json.loads((ROOT / "configs/run_authorization.json").read_text())["stop_jobs_by_utc"]
    start = time.monotonic()
    run_id = args.run_id + "_diagnostics"
    append_record(ROOT / "results/run_registry.jsonl", {"run_id": run_id, "status": "running", "git_commit": args.git_commit,
                  "predictor_sha256": metadata["predictor_sha256"], "realizations_per_mechanism": args.realizations,
                  "stage": "synthetic_and_V_shift", "actual_test_opened": False})
    all_records, pairs = [], []

    def evaluate(name, reference, mask, seed, group, domain):
        if remaining_seconds(deadline) < 180:
            raise TimeoutError("Authorized time margin reached")
        observation = Observation(reference, mask)
        seeds = tuple(seed * 10 + index for index in range(6))
        torch.cuda.synchronize()
        before = time.perf_counter()
        pool = generate_batched_pool(observation, model, fallback, floor, seeds[:4], seeds[4:],
                                     pool_config["steps_per_draw"], pool_config.get("clean_prediction_clip_abs"),
                                     pool_config.get("initial_timestep"))
        if pool is None:
            normalized, transform = normalize_observation(observation, floor)
            baseline = normalized.project(fallback(normalized))
            estimates = {"B": baseline, "P": baseline}
            accepted, calls, reason = False, 0, "insufficient_single_gap_probe_context"
        else:
            transform = pool.normalization
            selection = int(np.argmin(predict_deltas(predictor, pool.features)))
            result = decide(observation, pool, predictor, frozen)
            estimates = {"B": pool.fallback, "C1": pool.candidates[0], "C3": pool.candidates[selection],
                         "P": pool.candidates[selection] if result["accepted"] else pool.fallback}
            accepted, calls, reason = result["accepted"], pool.denoiser_calls, None
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - before
        target = transform.normalize_reference_for_scoring(reference)
        measurements = {key: gap_metrics(value, target, observation.gaps, MetricFloors(floor / transform.scale))
                        for key, value in estimates.items()}
        restored = {key: transform.restore(value, observation) for key, value in estimates.items()}
        if any(not np.array_equal(value[mask], reference[mask]) for value in restored.values()):
            raise ValueError("Observed data consistency failure")
        case_id = name + "_" + str(seed)
        path = arrays / (case_id + ".npz")
        values = {key + "_gap": value[~mask] for key, value in restored.items()}
        values.update(reference_gap=reference[~mask], imputed_indices=np.flatnonzero(~mask))
        if seed % 100 == 15:
            values.update(reference=reference, observation=observation.values, observed_mask=mask,
                          **{key + "_full": value for key, value in restored.items()})
        np.savez_compressed(path, **values)
        record = {"case_id": case_id, "mechanism": name, "seed": seed, "group_id": group, "domain": domain,
                  "metrics": measurements, "accepted": bool(accepted), "denoiser_calls": calls,
                  "fallback_reason": reason, "pipeline_seconds": elapsed,
                  "artifact_path": str(path.relative_to(ROOT)), "artifact_sha256": file_hash(path)}
        append_record(output / "cases.jsonl", record)
        all_records.append(record)
        return restored

    status = "running"
    try:
        for index in range(args.realizations):
            seed = 20261115 + index
            signals, event = signal_cases(seed)
            paired = {}
            for mechanism, (reference, mask) in signals.items():
                result = evaluate(mechanism, reference, mask, seed, "synthetic_" + str(seed), "synthetic")
                if mechanism.startswith("hidden_event"):
                    paired[mechanism] = result
            absent, present = paired["hidden_event_absent"], paired["hidden_event_present"]
            for method in absent:
                difference = float(np.max(np.abs(absent[method] - present[method])))
                if difference != 0:
                    raise ValueError("Same-input fixed-seed hidden targets changed inference")
                base = signals["hidden_event_absent"][0]
                coefficient = float(np.dot(absent[method] - base, event) / np.dot(event, event))
                detected = coefficient >= 0.5
                pairs.append({"seed": seed, "method": method, "maximum_output_difference": difference,
                              "estimated_event_coefficient": coefficient, "false_event_on_absent": bool(detected),
                              "missed_event_on_present": bool(not detected)})
            if (index + 1) % 10 == 0:
                print("synthetic realizations", index + 1, "/", args.realizations, flush=True)
        with gzip.open(ROOT / "manifests/pilot_sources.jsonl.gz", "rt") as stream:
            sources = [row for row in map(json.loads, stream) if row["partition"] == "V"][:20]
        for index, row in enumerate(sources):
            reference = crop_source(np.load(ROOT / row["processed_path"], allow_pickle=False))
            for name, mask in stress_masks().items():
                evaluate(name, reference, mask, 20262115 + index, row["group_id"], "V_shift")
            mask = np.ones(len(reference), dtype=bool)
            mask[14384:18384] = False
            gain = np.exp(np.linspace(np.log(0.25), np.log(4.0), len(reference)))
            from scipy.signal import butter, sosfiltfilt
            lowpass = sosfiltfilt(butter(4, 400, fs=8000, output="sos"), reference)
            mix = np.linspace(0, 1, len(reference))
            shifted = gain * ((1 - mix) * reference + mix * lowpass)
            evaluate("smooth_level_spectrum_shift", shifted, mask, 20263115 + index, row["group_id"], "V_derived_shift")
        status = "completed"
    except TimeoutError:
        status = "budget_stopped"
    except Exception as error:
        status = "failed"
        append_record(output / "failures.jsonl", {"error": repr(error), "completed_cases": len(all_records)})
        raise
    finally:
        grouped = defaultdict(list)
        for row in all_records:
            grouped[row["mechanism"]].append(row)
        summary = {"status": status, "completed_cases": len(all_records), "elapsed_seconds": time.monotonic() - start,
                   "actual_test_opened": False, "no_adaptation": True, "git_commit": args.git_commit,
                   "mechanisms": {key: {"cases": len(value), "coverage": float(np.mean([row["accepted"] for row in value])),
                                         "mean_P_Q": float(np.mean([row["metrics"]["P"]["Q"] for row in value])),
                                         "mean_B_Q": float(np.mean([row["metrics"]["B"]["Q"] for row in value]))}
                                  for key, value in grouped.items()},
                   "paired_hidden_target_tests": len(pairs), "maximum_paired_output_difference": max([row["maximum_output_difference"] for row in pairs], default=None),
                   "interpretation": "Identical observations cannot identify the hidden event. Synthetic and shifted cases are outside the calibration guarantee; these are development diagnostics, not main held-out evidence."}
        (output / "hidden_event_pairs.json").write_text(json.dumps(pairs, indent=2) + "\n")
        (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
        append_record(ROOT / "results/run_registry.jsonl", dict(summary, run_id=run_id, output_path=str((output / "summary.json").relative_to(ROOT))))
        print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
