"""Development-only pilot pools with a V-frozen fallback and reusable candidates.

The actual C/test partitions cannot be passed to this pilot entry point. References
are scored only after the observation-only generation call returns.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from acoustic_regeneration.data.observation import Observation, normalize_observation
from acoustic_regeneration.data.preprocessing import crop_source, draw_mask
from acoustic_regeneration.diagnostics.features import spectral_distances
from acoustic_regeneration.diagnostics.policy import generate_batched_pool
from acoustic_regeneration.evaluation.metrics import MetricFloors, gap_metrics
from acoustic_regeneration.registry import append_record, file_hash, remaining_seconds
from acoustic_regeneration.restoration.classical import bidirectional_ar, linear_interpolation
from acoustic_regeneration.restoration.learned import deterministic_restore, load_ema_model

ROOT = Path(__file__).resolve().parents[1]


def case_observation(row: dict, seed: int):
    reference = crop_source(np.load(ROOT / row["processed_path"], allow_pickle=False))
    case_seed = int(hashlib.sha256((str(seed) + ":" + row["original_file_id"]).encode()).hexdigest()[:8], 16)
    mask = draw_mask(len(reference), np.random.default_rng(case_seed))
    return reference, Observation(reference, mask), case_seed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--conditional", type=Path, required=True)
    parser.add_argument("--deterministic", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--mask-seed", type=int, default=20261117)
    parser.add_argument("--steps", type=int, default=32)
    parser.add_argument("--sampler-config", type=Path)
    parser.add_argument("--run-id", default="pilot_pools_seed17")
    parser.add_argument("--phase", choices=["fit", "calibration", "test"], default="fit")
    parser.add_argument("--git-commit", required=True)
    args = parser.parse_args()
    sampler_config = json.loads(args.sampler_config.read_text()) if args.sampler_config else {}
    clean_clip = sampler_config.get("clean_prediction_clip_abs")
    initial_timestep = sampler_config.get("initial_timestep")
    torch.set_num_threads(4)
    model, config, model_metadata = load_ema_model(args.conditional)
    deterministic, deterministic_config, deterministic_metadata = load_ema_model(args.deterministic)
    if config["phase"] != "development_only_pilot" or deterministic_config["phase"] != "development_only_pilot":
        raise ValueError("This runner is for development pilot checkpoints only")
    floor = config["rms_floor"]
    manifest = ROOT / "manifests/pilot_sources.jsonl.gz"
    if any(item["manifest_hash"] != file_hash(manifest) for item in (model_metadata, deterministic_metadata)):
        raise ValueError("Pilot manifest changed since training")
    with gzip.open(manifest, "rt") as stream:
        rows = list(map(json.loads, stream))
    by_partition = defaultdict(list)
    seen = set()
    for row in rows:
        if row["partition"] not in {"D", "V"} or row["group_id"] in seen:
            continue
        seen.add(row["group_id"])
        by_partition[row["partition"]].append(row)
    if len(by_partition["V"]) < 192:
        raise ValueError("Pilot dry flow needs 192 distinct V groups; do not inflate with descendants")
    roles = [(row, "D") for row in by_partition["D"][:200]]
    roles += [(row, "V" if index < 64 else ("dev_C" if index < 128 else "dev_test"))
              for index, row in enumerate(by_partition["V"][:192])]
    # Roles and seeds are fixed before generating any diagnostic outcomes.
    case_manifest = [{"source_id": row["original_file_id"], "group_id": row["group_id"],
                      "source_partition": row["partition"], "pilot_role": role,
                      "mask_seed": args.mask_seed} for row, role in roles]
    output = ROOT / "results" / args.run_id
    output.mkdir(parents=True, exist_ok=True)
    frozen_config = {"conditional_sha256": file_hash(args.conditional),
                     "deterministic_sha256": file_hash(args.deterministic),
                     "manifest_sha256": file_hash(manifest), "seed": args.seed,
                     "mask_seed": args.mask_seed, "steps_per_draw": args.steps,
                     "clean_prediction_clip_abs": clean_clip,
                     "initial_timestep": initial_timestep,
                     "sampler_config_sha256": file_hash(args.sampler_config) if args.sampler_config else None,
                     "rms_floor": floor, "roles": {"D": 200, "V": 64, "dev_C": 64, "dev_test": 64}}
    config_path = output / "pool_config.json"
    if config_path.exists() and json.loads(config_path.read_text()) != frozen_config:
        raise ValueError("Frozen pool configuration changed; use a separate run ID")
    if not config_path.exists():
        config_path.write_text(json.dumps(frozen_config, indent=2) + "\n")
    if args.phase == "calibration" and not (output / "frozen_predictors.json").is_file():
        raise ValueError("Freeze predictors and V thresholds before generating dev_C outcomes")
    if args.phase == "test" and not (output / "calibration.json").is_file():
        raise ValueError("Freeze dev_C calibration before generating dev_test outcomes")
    case_path = output / "case_manifest.json"
    if case_path.exists() and json.loads(case_path.read_text()) != case_manifest:
        raise ValueError("Frozen case manifest changed")
    if not case_path.exists():
        case_path.write_text(json.dumps(case_manifest, indent=2) + "\n")
    deadline = json.loads((ROOT / "configs/run_authorization.json").read_text())["stop_jobs_by_utc"]
    registry = ROOT / "results/run_registry.jsonl"
    registry_id = args.run_id + "_" + args.phase
    start = time.monotonic()
    append_record(registry, {"run_id": registry_id, "status": "running", "git_commit": args.git_commit,
                            "seed": args.seed, "manifest_hash": file_hash(manifest),
                            "case_manifest_hash": file_hash(case_path), "model_hash": file_hash(args.conditional),
                            "stage": "development_only_pilot", "steps_per_draw": args.steps,
                            "command": sys.argv, "config_hash": file_hash(config_path),
                            "hardware": torch.cuda.get_device_name()})
    methods = {"linear": linear_interpolation,
               **{"ar_" + str(order): (lambda observation, order=order: bidirectional_ar(observation, order))
                  for order in (16, 32, 64, 128)},
               "deterministic": lambda observation: deterministic_restore(deterministic, observation)}
    selection_path = output / "frozen_fallback.json"
    if selection_path.exists():
        selection = json.loads(selection_path.read_text())
    else:
        if args.phase != "fit":
            raise ValueError("Fallback must first be frozen on V")
        baseline_scores = defaultdict(list)
        for row in by_partition["V"][:64]:
            reference, original, case_seed = case_observation(row, args.mask_seed)
            observation, transform = normalize_observation(original, floor)
            target = transform.normalize_reference_for_scoring(reference)
            floors = MetricFloors(floor / transform.scale)
            for name, method in methods.items():
                estimate = observation.project(method(observation))
                metrics = gap_metrics(estimate, target, observation.gaps, floors)
                baseline_scores[name].append(metrics["Q"])
                append_record(output / "baseline_validation.jsonl", {"source_id": row["original_file_id"],
                              "group_id": row["group_id"], "method": name, "case_seed": case_seed, **metrics})
        means = {name: float(np.mean(values)) for name, values in baseline_scores.items()}
        fallback_name = min((name for name in methods if name != "linear"), key=lambda name: (means[name], name))
        selection = {"selected_fallback": fallback_name, "mean_Q_on_64_V_groups": means,
                     "selected_before_D_and_dev_C_pools": True, "rms_floor": floor}
        selection_path.write_text(json.dumps(selection, indent=2) + "\n")
    fallback = methods[selection["selected_fallback"]]
    print(json.dumps(selection), flush=True)
    pool_directory = ROOT / "results/pools" / args.run_id
    pool_directory.mkdir(parents=True, exist_ok=True)
    torch.cuda.reset_peak_memory_stats()
    phase_roles = {"fit": {"D", "V"}, "calibration": {"dev_C"}, "test": {"dev_test"}}[args.phase]
    roles = [(row, role) for row, role in roles if role in phase_roles]
    existing = {}
    if (output / "cases.jsonl").exists():
        existing = {row["source_id"]: row for row in map(json.loads, (output / "cases.jsonl").read_text().splitlines())}
    status, completed, generated = "running", 0, 0
    try:
        for index, (row, role) in enumerate(roles):
            if row["original_file_id"] in existing:
                cached = existing[row["original_file_id"]]
                if file_hash(ROOT / cached["pool_path"]) != cached["pool_sha256"]:
                    raise ValueError("Cached pool hash changed")
                completed += 1
                continue
            if remaining_seconds(deadline) < 180:
                status = "budget_stopped"
                break
            reference, original, case_seed = case_observation(row, args.mask_seed)
            seeds = tuple(case_seed * 10 + number for number in range(6))
            torch.cuda.synchronize()
            case_start = time.perf_counter()
            pool = generate_batched_pool(original, model, fallback, floor, seeds[:4], seeds[4:], args.steps,
                                         clean_clip, initial_timestep)
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - case_start
            if pool is None:
                raise RuntimeError("Main-distribution case unexpectedly has insufficient probe context")
            target = pool.normalization.normalize_reference_for_scoring(reference)
            floors = MetricFloors(floor / pool.normalization.scale)
            candidate_metrics = [gap_metrics(candidate, target, original.gaps, floors) for candidate in pool.candidates]
            fallback_metrics = gap_metrics(pool.fallback, target, original.gaps, floors)
            zero_metrics = gap_metrics(pool.normalized_observation.values, target, original.gaps, floors)
            linear_metrics = gap_metrics(linear_interpolation(pool.normalized_observation), target, original.gaps, floors)
            distances = spectral_distances(pool.candidates, original.gaps[0], floors.rms)
            medoid = int(np.argmin(distances.sum(axis=1)))
            a, b = original.gaps[0]
            case_id = row["original_file_id"] + "_" + str((b - a) * 1000 // 8000)
            pool_path = pool_directory / (case_id + ".npz")
            # Cache missing samples only. Observations are reconstructed from the
            # hashed source and hard-projected; no lossy observed values are saved.
            np.savez_compressed(pool_path, candidates_gap=pool.candidates[:, a:b],
                                fallback_gap=pool.fallback[a:b], features=pool.features,
                                probe_deltas=np.array(pool.probe_deltas))
            record = {"run_id": args.run_id, "case_id": case_id, "source_id": row["original_file_id"],
                      "group_id": row["group_id"], "source_partition": row["partition"], "pilot_role": role,
                      "gap_start": a, "gap_end": b, "gap_ms": (b - a) * 1000 / 8000,
                      "candidate_seeds": seeds[:4], "probe_seeds": seeds[4:],
                      "normalization_mean": pool.normalization.mean, "normalization_scale": pool.normalization.scale,
                      "candidate_metrics": candidate_metrics, "fallback_metrics": fallback_metrics,
                      "zero_metrics": zero_metrics, "linear_metrics": linear_metrics, "medoid": medoid,
                      "denoiser_calls": pool.denoiser_calls, "pipeline_seconds": elapsed,
                      "pool_path": str(pool_path.relative_to(ROOT)), "pool_sha256": file_hash(pool_path)}
            append_record(output / "cases.jsonl", record)
            completed += 1
            generated += 1
            if completed == 1 or completed % 10 == 0:
                print("pool", completed, "/", len(roles), "role", role, "seconds", round(elapsed, 3), flush=True)
        if status == "running":
            status = "completed"
    except Exception as error:
        status = "failed"
        append_record(output / "failures.jsonl", {"error": repr(error), "completed_cases": completed})
        raise
    finally:
        append_record(registry, {"run_id": registry_id, "status": status, "completed_cases": completed,
                                "planned_cases": len(roles), "elapsed_seconds": time.monotonic() - start,
                                "peak_vram_bytes": torch.cuda.max_memory_allocated(),
                                "forward_evaluations_in_this_invocation": generated * 6 * args.steps,
                                "new_cases": generated, "cached_cases": completed - generated,
                                "output_path": str(output.relative_to(ROOT)), "git_commit": args.git_commit})


if __name__ == "__main__":
    main()
