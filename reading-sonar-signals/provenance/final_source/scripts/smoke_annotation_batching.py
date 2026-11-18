#!/usr/bin/env python3
"""Replay completed calibration evidence to compare native, batch1 and batch4.

No waveform, view, prompt, truth label, or demonstration is generated or changed.
Main-study and held-out audit records are rejected before loading any model.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time

from sonar.reference_protocol import object_hash, select_controls


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_fixture(calibration, model_size, manifest, policy):
    records = [json.loads(line) for line in Path(manifest).read_text().splitlines() if line.strip()]
    controls = select_controls(records, policy, "calibration")
    name = "internvl" + model_size
    root = Path(calibration)
    summary = json.loads((root / (name + "_summary.json")).read_text())
    if summary.get("calibration_only") is not True or summary.get("scenes") != len(controls):
        raise RuntimeError("Smoke requires a completed calibration-only summary")
    if summary.get("control_records_sha256") != object_hash(controls):
        raise RuntimeError("Calibration summary does not identify the exact declared controls")
    if summary.get("control_seeds") != [r["seed"] for r in controls]:
        raise RuntimeError("Calibration summary seed identity mismatch")
    fixture_path = root / name / (controls[0]["scene_id"] + ".json")
    fixture = json.loads(fixture_path.read_text())
    if fixture.get("scene_id") != controls[0]["scene_id"] or fixture.get("input_record_sha256") != object_hash(controls[0]):
        raise RuntimeError("Fixture is not the first declared calibration control")
    if fixture.get("protocol_identity") != summary.get("protocol_identity"):
        raise RuntimeError("Fixture and completed calibration protocol identities differ")
    expected_views = ["lofar_1s", "lofar_4s", "demon_500_2000", "demon_2000_5000"]
    passes = fixture["generation"]["passes"]
    if [p["view"] for p in passes] != expected_views:
        raise RuntimeError("Require the four original fixed-view calibration passes")
    requests, baselines = [], []
    for row in passes:
        if digest(row["image"]) != row["image_sha256"]:
            raise RuntimeError("Calibration target image hash changed")
        generation = row["generation"]
        examples = generation["training_examples"]
        for example in examples:
            if digest(example["image"]) != example["image_sha256"]:
                raise RuntimeError("Calibration demonstration image hash changed")
            if (example.get("provenance", {}).get("role") != "exact_separate_synthetic_training_demonstration"
                    or example["provenance"].get("target_truth_used") is not False):
                raise RuntimeError("Require the saved separate synthetic demonstrations")
        requests.append(dict(prompt=row["prompt"], images=[row["image"]],
                             max_new_tokens=generation["max_new_tokens"], examples=examples))
        baselines.append(generation)
    return requests, baselines, {
        "scene_id": fixture["scene_id"], "control_role": "annotation_calibration",
        "fixture_path": str(fixture_path), "fixture_sha256": digest(fixture_path),
        "summary_sha256": digest(root / (name + "_summary.json")),
        "requests_sha256": object_hash(requests), "views": expected_views,
        "model_id": fixture["generation"]["model_id"],
        "model_revision": fixture["generation"]["model_revision"],
        "protocol_identity": fixture["protocol_identity"],
        "main_or_audit_data_used": False,
    }


def comparison(first, second):
    if len(first) != len(second):
        raise ValueError("Cannot compare different numbers of annotation requests")
    fields = ["raw_text", "input_tokens", "output_tokens", "max_new_tokens", "at_generation_limit"]
    return [{"index": i, "exact": all(a.get(key) == b.get(key) for key in fields),
             "changed_fields": [key for key in fields if a.get(key) != b.get(key)],
             "parsed_equal": a.get("parsed") == b.get("parsed")}
            for i, (a, b) in enumerate(zip(first, second))]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-size", choices=["4b", "8b"], required=True)
    parser.add_argument("--calibration", default="results/ai_reference_calibration")
    parser.add_argument("--manifest", default="data/manifests/annotation_controls.jsonl")
    parser.add_argument("--policy", default="configs/ai_reference_eligibility.json")
    parser.add_argument("--output", required=True)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--rerun-serial", action="store_true", help="Also replay native AnnotationVLM serial calls")
    parser.add_argument("--dry-run", action="store_true", help="Validate fixture provenance without model/GPU imports")
    args = parser.parse_args()
    if not 2 <= args.batch_size <= 4:
        parser.error("Comparison batch size must be between 2 and 4")
    output = Path(args.output)
    if output.exists():
        raise RuntimeError("Refusing to overwrite an earlier batching smoke report")
    policy = json.loads(Path(args.policy).read_text())
    requests, recorded_serial, fixture = load_fixture(args.calibration, args.model_size, args.manifest, policy)
    config_path = "configs/ai_annotations.json" if args.model_size == "4b" else "configs/ai_annotations_internvl8b_pilot.json"
    config = json.loads(Path(config_path).read_text())
    spec = config["models"]["internvl"]
    if spec["model_id"] != fixture["model_id"] or spec["revision"] != fixture["model_revision"]:
        raise RuntimeError("Model configuration differs from saved calibration fixture")
    if digest(config_path) != fixture["protocol_identity"]["config_sha256"]:
        raise RuntimeError("Model configuration hash differs from saved calibration")
    if args.dry_run:
        print(json.dumps(dict(validated=True, requests=len(requests), fixture=fixture), indent=2))
        return
    from sonar.annotators import AnnotationVLM
    from sonar.annotation_batching import BatchedAnnotationVLM
    backend = AnnotationVLM(spec, config)
    native = [backend.generate(**request) for request in requests] if args.rerun_serial else recorded_serial
    wall = time.perf_counter()
    with BatchedAnnotationVLM(backend, max_batch_size=1, max_wait_ms=0) as adapter:
        single = adapter.generate_batch(requests)
        single_ledger = list(adapter.batch_ledger)
    single_wall = time.perf_counter() - wall
    single_comparison = comparison(native, single)
    report = dict(fixture=fixture, batch_size=args.batch_size,
                  execution_adapter_sha256=digest("src/sonar/annotation_batching.py"),
                  json_protocol_sha256=digest("src/sonar/json_protocol.py"),
                  smoke_script_sha256=digest(__file__),
                  baseline="native_serial_rerun" if args.rerun_serial else "recorded_calibration_serial",
                  recorded_serial=recorded_serial, serial=native, batch1=single,
                  batch1_comparison=single_comparison, batch1_ledger=single_ledger,
                  batch1_exact=all(row["exact"] for row in single_comparison),
                  batch1_wall_seconds=single_wall, performance_is_not_acoustic_accuracy=True,
                  deployment_requires_explicit_execution_amendment=True)
    if report["batch1_exact"]:
        wall = time.perf_counter()
        with BatchedAnnotationVLM(backend, max_batch_size=args.batch_size, max_wait_ms=80) as adapter:
            batch = adapter.generate_batch(requests)
            ledger = list(adapter.batch_ledger)
        batch_wall = time.perf_counter() - wall
        compared = comparison(native, batch)
        single_gpu = sum(row["batch_gpu_seconds"] for row in single_ledger)
        batch_gpu = sum(row["batch_gpu_seconds"] for row in ledger)
        report.update(batch=batch, batch_comparison=compared, batch_ledger=ledger,
                      batch_exact=all(row["exact"] for row in compared),
                      batch_wall_seconds=batch_wall,
                      batch1_over_batch_gpu_speedup=single_gpu / batch_gpu if batch_gpu else None,
                      batch1_over_batch_wall_speedup=single_wall / batch_wall if batch_wall else None,
                      batch_timing_sums_once_per_ledger_entry=True)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps({key: report.get(key) for key in ["batch1_exact", "batch_exact", "batch1_over_batch_gpu_speedup", "batch1_over_batch_wall_speedup"]}))
    if not report["batch1_exact"]:
        raise SystemExit("Batch1 differs from the native baseline; multi-row smoke was not run")
    if not report.get("batch_exact"):
        raise SystemExit("Multi-row outputs differ; report retained and deployment is not approved")


if __name__ == "__main__":
    main()
