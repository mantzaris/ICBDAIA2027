"""Secondary morphology accounting from saved, unchanged exact-truth matches.

No rematching, new inference, real-reference qualification, or primary score
replacement occurs. Arithmetic and aggregation require CUDA.
"""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import torch
import evaluate_benchmark as evaluator
from sonar.metrics import aggregate_event_metrics
from sonar.signal import require_cuda

KINDS = ["tone", "chirp", "transient", "broadband", "modulation", "unknown_malformed"]


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def rows(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def category(event):
    value = event.get("type") if isinstance(event, dict) else None
    return value if value in KINDS[:-1] else "unknown_malformed"


def partition(predicted, invalid_originals, required_gold, measured):
    """Keep frozen one-to-one matches and optional-structure exclusions."""
    device = require_cuda()
    matched_predictions = {match["prediction_index"] for match in measured["matches"]}
    matched_gold = {match["gold_index"] for match in measured["matches"]}
    ignored = set(measured.get("ignored_predictions", []))
    if matched_predictions & ignored:
        raise ValueError("Matched prediction cannot also be ignored")
    for match in measured["matches"]:
        if category(predicted[match["prediction_index"]]) != category(required_gold[match["gold_index"]]):
            raise ValueError("Frozen match crosses morphology categories")
    indicators = []
    for index, event in enumerate(predicted):
        if index not in ignored:
            indicators.append((KINDS.index(category(event)), int(index in matched_predictions), int(index not in matched_predictions), 0))
    indicators.extend((KINDS.index(category(event)), 0, 1, 0) for event in invalid_originals)
    indicators.extend((KINDS.index(category(event)), 0, 0, int(index not in matched_gold))
                      for index, event in enumerate(required_gold))
    counts = torch.zeros((len(KINDS), 3), device=device, dtype=torch.int64)
    if indicators:
        values = torch.tensor(indicators, device=device, dtype=torch.int64)
        counts.scatter_add_(0, values[:, :1].expand(-1, 3), values[:, 1:])
    expected = torch.tensor([measured[key] for key in ("tp", "fp", "fn")], device=device)
    if not torch.equal(counts.sum(0), expected):
        raise ValueError("Morphology counts do not reconstruct the frozen event counts")
    return {kind: dict(zip(("tp", "fp", "fn"), values)) for kind, values in zip(KINDS, counts.tolist())}


def self_test():
    tone = {"type": "tone"}; chirp = {"type": "chirp"}; modulation = {"type": "modulation"}
    measured = {"tp": 1, "fp": 3, "fn": 1,
                "matches": [{"prediction_index": 0, "gold_index": 0}], "ignored_predictions": [2]}
    value = partition([tone, chirp, modulation], [chirp, None], [tone, modulation], measured)
    assert value["tone"] == {"tp": 1, "fp": 0, "fn": 0}
    assert value["chirp"] == {"tp": 0, "fp": 2, "fn": 0}
    assert value["modulation"] == {"tp": 0, "fp": 0, "fn": 1}
    assert value["unknown_malformed"]["fp"] == 1
    assert partition([], [], [], {"tp": 0, "fp": 0, "fn": 0, "matches": []})["tone"]["tp"] == 0
    print(json.dumps({"status": "passed", "device": "cuda", "checks": ["saved matches", "optional ignored", "typed and unknown malformed", "empty support", "counts reconstruct"]}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    require_cuda()
    protocol_path = Path("configs/morphology_supplement.json")
    protocol = read(protocol_path)
    for path, expected in protocol["sha256"].items():
        if digest(path) != expected:
            raise RuntimeError("Frozen supplemental dependency changed: " + path)
    if args.self_test:
        return self_test()
    if args.run is None:
        parser.error("--run is required")
    run = args.run.resolve()
    configuration = read(run / "run.json")
    manifest_path = Path(configuration["manifest"])
    if digest(manifest_path) != configuration["manifest_sha256"]:
        raise RuntimeError("Run manifest changed")
    scenes = {record["scene_id"]: record for record in rows(manifest_path)}
    evaluation = run / "evaluation/per_episode.jsonl"
    primary_summary = read(run / "evaluation/summary.json")
    for key, path in (("run_configuration_sha256", run / "run.json"),
                      ("manifest_sha256", manifest_path),
                      ("evaluator_code_sha256", Path("scripts/evaluate_benchmark.py")),
                      ("metrics_code_sha256", Path("src/sonar/metrics.py"))):
        if primary_summary[key] != digest(path):
            raise RuntimeError("Primary evaluation provenance differs: " + key)
    results = []
    for row in rows(evaluation):
        if row["reference_kind"] != "exact_synthetic_component_truth" or row["event_metrics"] is None:
            continue
        episode_path = run / row["scene_id"] / row["method"] / "episode.json" if row.get("episode_path") else None
        if episode_path and digest(episode_path) != row["episode_sha256"]:
            raise RuntimeError("Episode changed after primary evaluation")
        episode = read(episode_path) if episode_path else {}
        report = episode.get("report")
        reference = evaluator.resolve_reference(scenes[row["scene_id"]], run, {}, 8.)
        predicted, invalid, _ = evaluator.report_events(report)
        scoped, scoped_invalid, _ = evaluator.score_scope(predicted, invalid, report, reference)
        if [event.get("id") for event in scoped] != row["scoring_scope"]["scored_prediction_event_ids"]:
            raise RuntimeError("Saved prediction order no longer agrees")
        components = report.get("components", []) if isinstance(report, dict) else []
        malformed = [components[item["index"]] if isinstance(components, list) and item.get("index") is not None else None
                     for item in scoped_invalid]
        gold = [event for event in reference["events"] if event.get("evaluation_required") is not False]
        counts = partition(scoped, malformed, gold, row["event_metrics"])
        results.append({"scene_id": row["scene_id"], "method": row["method"], "stratum": row["stratum"],
                        "domain": row["domain"], "episode_sha256": row["episode_sha256"],
                        "truth_sha256": reference["sha256"], "counts": counts})
    summary = {}
    for stratum in sorted({row["stratum"] for row in results}):
        summary[stratum] = {}
        for method in sorted({row["method"] for row in results if row["stratum"] == stratum}):
            subset = [row for row in results if row["stratum"] == stratum and row["method"] == method]
            summary[stratum][method] = {}
            for kind in KINDS:
                measured = aggregate_event_metrics([row["counts"][kind] for row in subset])
                support = torch.tensor([measured["tp"], measured["fn"]], device="cuda").sum().item()
                summary[stratum][method][kind] = dict(measured, reference_support=int(support),
                    support_status="has_positive_reference_support" if support else "no_positive_reference_support")
    output = run / "morphology_supplement"
    output.mkdir(exist_ok=True)
    (output / "per_episode.jsonl").write_text("".join(json.dumps(row, allow_nan=False) + "\n" for row in results))
    (output / "summary.json").write_text(json.dumps({"created_at": datetime.now(timezone.utc).isoformat(),
        "status": "complete", "device": "cuda", "protocol_sha256": digest(protocol_path),
        "frozen_evaluation_sha256": digest(evaluation), "results": summary,
        "interpretation": "Secondary exact synthetic per-morphology accounting using unchanged primary matches. No real correctness, rematching, confirmatory tests, or positive-class performance claim where support is zero."}, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"status": "complete", "episodes": len(results), "output": str(output)}))


if __name__ == "__main__":
    main()
