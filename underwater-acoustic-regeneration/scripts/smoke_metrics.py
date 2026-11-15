"""Audit processed hashes and exercise every acoustic score on 200 G crops."""
from __future__ import annotations

import gzip
import json
import time
from pathlib import Path

import numpy as np

from acoustic_regeneration.data.observation import Observation, normalize_observation
from acoustic_regeneration.data.preprocessing import crop_source, draw_mask
from acoustic_regeneration.evaluation.metrics import MetricFloors, gap_metrics
from acoustic_regeneration.registry import append_record, file_hash
from acoustic_regeneration.restoration.classical import bidirectional_ar, linear_interpolation

ROOT = Path(__file__).resolve().parents[1]


def main():
    start = time.monotonic()
    with gzip.open(ROOT / "manifests/pilot_sources.jsonl.gz", "rt") as stream:
        rows = list(map(json.loads, stream))
    for row in rows:
        if file_hash(ROOT / row["processed_path"]) != row["processed_sha256"]:
            raise ValueError("Processed audio hash mismatch")
    training = [row for row in rows if row["partition"] == "G"]
    rms = [np.std(np.load(ROOT / row["processed_path"], allow_pickle=False).astype(float)) for row in training]
    floor = max(float(np.quantile(rms, 0.01)) * 0.001, 1e-12)
    methods = {"zero": lambda item: item.values, "linear": linear_interpolation,
               **{"ar_" + str(order): (lambda item, order=order: bidirectional_ar(item, order)) for order in (16, 32, 64, 128)}}
    output = ROOT / "results/metric_smoke.jsonl"
    if output.exists():
        raise ValueError("Preserve completed smoke results; do not append a second copy")
    maximum_drift = 0.0
    for index, row in enumerate(training[:200]):
        reference = crop_source(np.load(ROOT / row["processed_path"], allow_pickle=False))
        mask = draw_mask(len(reference), np.random.default_rng(20261115 + index), two_gap_probability=0.2)
        original = Observation(reference, mask)
        observation, transform = normalize_observation(original, floor)
        target = transform.normalize_reference_for_scoring(reference)
        for name, method in methods.items():
            estimate = observation.project(method(observation))
            metrics = gap_metrics(estimate, target, original.gaps, MetricFloors(floor / transform.scale))
            if not np.isfinite(list(metrics.values())).all():
                raise FloatingPointError("Nonfinite smoke score")
            restored = transform.restore(estimate, original)
            if not np.array_equal(restored[mask], reference[mask]):
                raise ValueError("Observed samples changed")
            maximum_drift = max(maximum_drift, metrics["observed_max_drift"])
            append_record(output, {"source_id": row["original_file_id"], "group_id": row["group_id"],
                          "partition": "G", "method": name, **metrics})
        if (index + 1) % 50 == 0:
            print("metric smoke", index + 1, "/ 200", flush=True)
    summary = {"processed_source_hashes_verified": len(rows), "G_crops": 200, "methods": list(methods),
               "scored_outputs": 200 * len(methods), "all_metrics_finite": True,
               "max_normalized_observation_drift": maximum_drift, "original_observations_bitwise_preserved": True,
               "elapsed_seconds": time.monotonic() - start, "scientific_efficacy_results": False,
               "manifest_sha256": file_hash(ROOT / "manifests/pilot_sources.jsonl.gz")}
    (ROOT / "reports/metric_smoke.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
