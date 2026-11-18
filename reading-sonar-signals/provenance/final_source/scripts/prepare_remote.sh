#!/usr/bin/env bash
set -euo pipefail
mkdir -p /workspace/reading-sonar-signals
cd /workspace/reading-sonar-signals
python -m venv --system-site-packages .venv
.venv/bin/python -m pip install --disable-pip-version-check -r requirements.gpu.txt
.venv/bin/python -m pip freeze > reports/environment.lock.txt
.venv/bin/python - <<'PY'
import json, platform, subprocess, time
from pathlib import Path
import torch
from huggingface_hub import HfApi, snapshot_download
if not torch.cuda.is_available():
    raise RuntimeError('CUDA is mandatory; no CPU fallback.')
repo = 'Qwen/Qwen3-VL-8B-Instruct'
info = HfApi().model_info(repo)
revision = info.sha
model = {'model_id': repo, 'revision': revision, 'retrieved_at_unix': time.time(),
         'license': (info.card_data or {}).get('license'), 'torch': torch.__version__,
         'cuda': torch.version.cuda, 'python': platform.python_version(),
         'gpu': torch.cuda.get_device_name(0),
         'total_vram_bytes': torch.cuda.get_device_properties(0).total_memory,
         'container': 'runpod/pytorch:1.4.1-cu1281-torch280-ubuntu2404'}
Path('reports/model_revision.json').write_text(json.dumps(model, indent=2)+'\n')
print(json.dumps(model, indent=2), flush=True)
snapshot_download(repo_id=repo, revision=revision, local_dir='models/qwen3-vl-8b',
                  allow_patterns=['*.json','*.safetensors','*.txt','*.jinja','*.model','LICENSE*','README.md'])
print('MODEL_DOWNLOAD_COMPLETE', flush=True)
PY
