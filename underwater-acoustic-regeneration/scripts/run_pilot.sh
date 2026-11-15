#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
research_commit="$1"
export PYTHONPATH=src
export OMP_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export MKL_NUM_THREADS=4
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export RESEARCH_GIT_COMMIT="$research_commit"
mkdir -p logs reports results
.venv/bin/python scripts/capture_environment.py
.venv/bin/python -m pytest -q > logs/gpu_tests.log
.venv/bin/python scripts/resume_check.py
.venv/bin/python scripts/train.py --kind conditional --width 32 --layers 12 --overfit 16 --updates 200 --validation-every 200 --run-id pilot_smoke_seed17 --git-commit "$research_commit"
.venv/bin/python scripts/train.py --kind conditional --updates 5000 --run-id pilot_conditional_seed17 --git-commit "$research_commit"
.venv/bin/python scripts/train.py --kind deterministic --updates 2500 --run-id pilot_deterministic_seed17 --git-commit "$research_commit"
