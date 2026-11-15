#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
research_commit="$1"
export PYTHONPATH=src
export OMP_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export MKL_NUM_THREADS=4
export CUBLAS_WORKSPACE_CONFIG=:4096:8
conditional=checkpoints/pilot_conditional_seed17/best.pt
deterministic=checkpoints/pilot_deterministic_seed17/best.pt
# Require completed stages, not merely intermediate checkpoint files.
.venv/bin/python -c 'import json; from pathlib import Path; rows=list(map(json.loads,Path("results/run_registry.jsonl").read_text().splitlines())); states={r["run_id"]:r["status"] for r in rows}; assert all(states.get(k)=="completed" for k in ("pilot_conditional_seed17","pilot_deterministic_seed17")),states'
.venv/bin/python -m pytest -q > logs/gpu_evaluation_tests.log
pool_run=pilot_pools_clipped_seed17
.venv/bin/python scripts/make_pools.py --sampler-config preregistration/sampler_amendment.json --conditional "$conditional" --deterministic "$deterministic" --phase fit --run-id "$pool_run" --git-commit "$research_commit"
.venv/bin/python scripts/compare_steps.py --conditional "$conditional" --deterministic "$deterministic" --run-id "$pool_run" --git-commit "$research_commit"
selected_steps=$(.venv/bin/python -c 'import json; print(json.load(open("results/pilot_pools_clipped_seed17/step_comparison.json"))["selected_steps"])')
if [ "$selected_steps" = 64 ]; then
    pool_run=pilot_pools_clipped_seed17_steps64
    .venv/bin/python scripts/make_pools.py --sampler-config preregistration/sampler_amendment.json --conditional "$conditional" --deterministic "$deterministic" --phase fit --steps 64 --run-id "$pool_run" --git-commit "$research_commit"
fi
.venv/bin/python scripts/analyze_pilot.py --stage fit --run-id "$pool_run" --git-commit "$research_commit"
.venv/bin/python scripts/make_pools.py --sampler-config preregistration/sampler_amendment.json --conditional "$conditional" --deterministic "$deterministic" --phase calibration --steps "$selected_steps" --run-id "$pool_run" --git-commit "$research_commit"
.venv/bin/python scripts/analyze_pilot.py --stage calibrate --run-id "$pool_run" --git-commit "$research_commit"
.venv/bin/python scripts/make_pools.py --sampler-config preregistration/sampler_amendment.json --conditional "$conditional" --deterministic "$deterministic" --phase test --steps "$selected_steps" --run-id "$pool_run" --git-commit "$research_commit"
.venv/bin/python scripts/analyze_pilot.py --stage evaluate --run-id "$pool_run" --git-commit "$research_commit"
