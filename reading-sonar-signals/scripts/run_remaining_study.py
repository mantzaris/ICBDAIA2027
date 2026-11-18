"""Operational queue for frozen study phases and separately declared summaries.

No inference/scoring choices are changed here. This queue never stops the pod;
review, verified local preservation, and explicit lifecycle control follow it.
"""
from __future__ import annotations
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time


def main():
    root = Path(__file__).resolve().parents[1]
    os.chdir(root)
    os.environ["PYTHONPATH"] = str(root / "src")
    if not Path("results/ai_annotations/COMPLETED.json").is_file():
        raise RuntimeError("Final AI-reference assembly is required")
    smoke = json.loads(Path("reports/viewer_http_cuda_smoke.json").read_text())
    if smoke["status"] != "passed" or not smoke["server_stopped"]:
        raise RuntimeError("Exploratory viewer GPU smoke is unfinished")
    phases = [
        ("original_through_ablations", ["scripts/run_study_phases.py", "--start", "freeze_cases", "--stop-after", "ablations_evaluate"]),
        ("synthetic_ablation_supplement", ["scripts/run_synthetic_ablations.py"]),
        ("original_budgets_and_interventions", ["scripts/run_study_phases.py", "--start", "budget2", "--stop-after", "output_sensitivity"]),
    ]
    for run in ["main", "localization/predictions", "synthetic_ablations", "budget2", "budget4", "robustness", "sensitivity"]:
        phases.append(("morphology_" + run.replace("/", "_"), ["scripts/summarize_morphology.py", "--run", "results/" + run]))
    for run in ["main", "synthetic_ablations", "budget2", "budget4", "robustness", "sensitivity"]:
        phases.append(("relations_" + run, ["scripts/evaluate_relation_supplement.py", "--run", "results/" + run]))
    phases.append(("figures_and_viewer", ["scripts/make_figures.py"]))
    directory = Path("reports/remaining_study")
    directory.mkdir(parents=True, exist_ok=True)
    ledger = directory / "ledger.jsonl"
    completed = {row["phase"] for row in [json.loads(line) for line in ledger.read_text().splitlines()] if row["returncode"] == 0} if ledger.exists() else set()
    for name, command in phases:
        if name in completed:
            print(json.dumps({"event": "skip_completed_phase", "phase": name}), flush=True)
            continue
        if shutil.disk_usage(root).free < 1_000_000_000:
            raise RuntimeError("Less than 1GB free; preserve artifacts before continuing")
        started = time.time()
        log = directory / (name + "." + str(int(started)) + ".log")
        record = {"phase": name, "argv": [sys.executable] + command,
                  "started_at_unix": started, "log": str(log)}
        (directory / "active.json").write_text(json.dumps(record, indent=2) + "\n")
        print(json.dumps(dict(event="start", **record)), flush=True)
        with log.open("w") as stream:
            result = subprocess.run(record["argv"], stdout=stream, stderr=subprocess.STDOUT)
        record.update(finished_at_unix=time.time(), returncode=result.returncode,
                      log_sha256=hashlib.sha256(log.read_bytes()).hexdigest())
        with ledger.open("a") as stream:
            stream.write(json.dumps(record) + "\n")
        print(json.dumps(dict(event="finished", **record)), flush=True)
        if result.returncode:
            raise RuntimeError("Study phase failed: " + name + "; see " + str(log))
    (directory / "COMPLETED.json").write_text(json.dumps({
        "status": "all_queued_experiments_and_exports_finished",
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "requires_before_shutdown": ["result and figure review", "verified local artifact preservation"],
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "phases": [name for name, _ in phases]}, indent=2) + "\n")
    print("REMAINING_STUDY_QUEUE_FINISHED", flush=True)


if __name__ == "__main__":
    main()
