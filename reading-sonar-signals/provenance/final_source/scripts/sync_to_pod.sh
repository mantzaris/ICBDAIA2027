#!/usr/bin/env bash
set -euo pipefail
rsync -az --exclude='.venv/' --exclude='models/' --exclude='results/' \
  --exclude='reports/' --exclude='manuscript/' --exclude='data/raw/' \
  --exclude='data/annotation_views*/' --exclude='data/annotation_control_truth/' \
  --exclude='figures/' --exclude='viewer/' --exclude='__pycache__/' \
  --exclude='.pytest_cache/' \
  --exclude='runpod_archive/' \
  -e 'ssh -S /tmp/runpod-ssh.YZ5pS8/session -p 45012 -o BatchMode=yes -o StrictHostKeyChecking=yes -o UserKnownHostsFile=/tmp/runpod-ssh.YZ5pS8/known_hosts' \
  ./ root@195.26.233.13:/workspace/reading-sonar-signals/
