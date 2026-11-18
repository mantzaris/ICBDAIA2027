#!/usr/bin/env bash
set -euo pipefail
# Raw acquired data already exists locally; public base-model weights are pinned
# by revision and hash. Keep every generated result and trained checkpoint.
protocol_filters=()
if [[ "${1:-}" == "--protocol" ]]; then
  # Use after all writers finish. These are the frozen configurations/snapshots
  # that actually generated the artifacts, and must accompany the final receipt.
  protocol_filters=(--include='configs/***' --include='provenance/***'
    --include='src/***' --include='scripts/***' --include='tests/***'
    --include='README*' --include='requirements*.txt' --include='pyproject.toml'
    --include='.gitignore' --include='underwater_acoustic_agent_research_plan.md')
fi
rsync -azH --checksum --exclude='.venv/' --exclude='models/' --exclude='data/raw/' \
  --exclude='__pycache__/' --exclude='.pytest_cache/' \
  --include='reports/***' --include='results/***' --include='checkpoints/***' \
  --include='data/***' \
  --include='annotations/***' \
  "${protocol_filters[@]}" \
  --include='figures/***' --include='viewer/***' --exclude='*' \
  -e 'ssh -S /tmp/runpod-ssh.YZ5pS8/session -p 45012 -o BatchMode=yes -o StrictHostKeyChecking=yes -o UserKnownHostsFile=/tmp/runpod-ssh.YZ5pS8/known_hosts' \
  root@195.26.233.13:/workspace/reading-sonar-signals/ ./
