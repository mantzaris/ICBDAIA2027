"""Prespecified 32/64-step comparison on the same 64 tuning V cases only."""
from __future__ import annotations

import argparse
import gzip
import json
import time
from pathlib import Path

import numpy as np
import torch

from make_pools import ROOT, case_observation
from acoustic_regeneration.diagnostics.policy import generate_batched_pool
from acoustic_regeneration.evaluation.metrics import MetricFloors, gap_metrics
from acoustic_regeneration.registry import append_record, file_hash, remaining_seconds
from acoustic_regeneration.restoration.classical import bidirectional_ar
from acoustic_regeneration.restoration.learned import deterministic_restore, load_ema_model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", default="pilot_pools_seed17")
    parser.add_argument("--conditional", type=Path, required=True)
    parser.add_argument("--deterministic", type=Path, required=True)
    parser.add_argument("--git-commit", required=True)
    args = parser.parse_args()
    torch.set_num_threads(4)
    directory = ROOT / "results" / args.run_id
    config = json.loads((directory / "pool_config.json").read_text())
    if config["steps_per_draw"] != 32:
        raise ValueError("Comparison starts from the 32-step pools")
    if file_hash(args.conditional) != config["conditional_sha256"] or file_hash(args.deterministic) != config["deterministic_sha256"]:
        raise ValueError("Models changed after baseline pools")
    model, training_config, _ = load_ema_model(args.conditional)
    deterministic, _, _ = load_ema_model(args.deterministic)
    selection = json.loads((directory / "frozen_fallback.json").read_text())["selected_fallback"]
    fallback = ((lambda observation: deterministic_restore(deterministic, observation)) if selection == "deterministic"
                else (lambda observation: bidirectional_ar(observation, int(selection.split("_")[1]))))
    with gzip.open(ROOT / "manifests/pilot_sources.jsonl.gz", "rt") as stream:
        sources = {row["original_file_id"]: row for row in map(json.loads, stream)}
    cases = [row for row in map(json.loads, (directory / "cases.jsonl").read_text().splitlines()) if row["pilot_role"] == "V"]
    if len(cases) != 64 or any(row["source_partition"] != "V" for row in cases):
        raise ValueError("Exactly the frozen 64 tuning V cases are required")
    deadline = json.loads((ROOT / "configs/run_authorization.json").read_text())["stop_jobs_by_utc"]
    output_path = directory / "step_comparison_cases.jsonl"
    results = list(map(json.loads, output_path.read_text().splitlines())) if output_path.exists() else []
    existing = {row["source_id"] for row in results}
    start = time.monotonic()
    run_id = args.run_id + "_compare_steps"
    append_record(ROOT / "results/run_registry.jsonl", {"run_id": run_id, "status": "running", "git_commit": args.git_commit,
                  "config_hash": file_hash(directory / "pool_config.json"), "model_hash": file_hash(args.conditional),
                  "stage": "development_only_pilot", "planned_cases": 64})
    for case in cases:
        if case["source_id"] in existing:
            continue
        if remaining_seconds(deadline) < 180:
            raise RuntimeError("Budget margin reached; comparison incomplete")
        reference, observation, seed = case_observation(sources[case["source_id"]], config["mask_seed"])
        torch.cuda.synchronize()
        before = time.perf_counter()
        pool = generate_batched_pool(observation, model, fallback, training_config["rms_floor"],
                                     tuple(case["candidate_seeds"]), tuple(case["probe_seeds"]), 64)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - before
        target = pool.normalization.normalize_reference_for_scoring(reference)
        floors = MetricFloors(training_config["rms_floor"] / pool.normalization.scale)
        measurements = [gap_metrics(candidate, target, observation.gaps, floors) for candidate in pool.candidates]
        result = {"source_id": case["source_id"], "group_id": case["group_id"], "candidate_metrics_64": measurements,
                  "best_Q_32": min(row["Q"] for row in case["candidate_metrics"]),
                  "best_Q_64": min(row["Q"] for row in measurements), "fallback_Q": case["fallback_metrics"]["Q"],
                  "seconds_32": case["pipeline_seconds"], "seconds_64": elapsed, "denoiser_calls_64": pool.denoiser_calls}
        append_record(output_path, result)
        results.append(result)
        if len(results) % 10 == 0:
            print("step comparison", len(results), "/ 64", flush=True)
    gain = float(np.mean([row["best_Q_32"] - row["best_Q_64"] for row in results]))
    opportunity = {str(steps): float(np.mean([row["best_Q_" + str(steps)] <= row["fallback_Q"] - 0.02 for row in results]))
                   for steps in (32, 64)}
    # Frozen before either result was available. A 0.01 average oracle-Q gain
    # plus a feasible pool is required to pay double for every matched method.
    chosen = 64 if gain >= 0.01 and opportunity["64"] >= 0.20 else 32
    summary = {"groups": len(results), "source_partition": "V", "mean_best_Q_gain_from_64": gain,
               "oracle_opportunity": opportunity, "selected_steps": chosen,
               "selection_rule": "choose 64 only if mean oracle-Q improves by >=0.01 and its opportunity fraction is >=0.20",
               "mean_pipeline_seconds": {str(steps): float(np.mean([row["seconds_" + str(steps)] for row in results])) for steps in (32, 64)},
               "git_commit": args.git_commit, "scientific_test_results": False}
    (directory / "step_comparison.json").write_text(json.dumps(summary, indent=2) + "\n")
    append_record(ROOT / "results/run_registry.jsonl", {"run_id": run_id, "status": "completed", "completed_cases": len(results),
                  "elapsed_seconds": time.monotonic() - start, "git_commit": args.git_commit,
                  "output_path": str((directory / "step_comparison.json").relative_to(ROOT))})
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
