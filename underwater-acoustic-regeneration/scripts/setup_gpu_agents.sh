#!/usr/bin/env bash
set -euo pipefail
cd /workspace/underwater-acoustic-regeneration
export HF_HOME="$PWD/.cache/huggingface"
export PIP_CACHE_DIR="$PWD/.cache/pip"
export TMPDIR="$PWD/.cache/tmp"
mkdir -p "$HF_HOME" "$PIP_CACHE_DIR" "$TMPDIR" logs provenance .local/models
python3 -m venv --system-site-packages .venv-agents
.venv-agents/bin/python -m pip install 'transformers==4.57.3' 'accelerate==1.12.0' 'huggingface_hub==0.36.0' 'tokenizers==0.22.1'
.venv-agents/bin/python - <<'PY'
import hashlib, json, platform, subprocess
from datetime import datetime, timezone
from pathlib import Path
import torch, transformers, huggingface_hub, accelerate, tokenizers
from huggingface_hub import snapshot_download
root=Path.cwd()
revision='b968826d9c46dd6066d109eabc6255188de91218'
path=snapshot_download('Qwen/Qwen3-8B',revision=revision,local_dir=root/'.local/models/Qwen3-8B',
                       allow_patterns=['*.json','*.safetensors','*.txt','*.model','*.jinja','LICENSE','README.md'])
files={}
for p in sorted(Path(path).rglob('*')):
    if p.is_file() and '.cache' not in p.relative_to(path).parts:
        h=hashlib.sha256()
        with p.open('rb') as f:
            for chunk in iter(lambda:f.read(8*1024*1024),b''): h.update(chunk)
        files[str(p.relative_to(path))]=h.hexdigest()
record={'recorded_utc':datetime.now(timezone.utc).isoformat(),'model':'Qwen/Qwen3-8B',
        'revision':revision,'model_files_sha256':files,'license':'Apache-2.0',
        'python':platform.python_version(),'torch':torch.__version__,'cuda':torch.version.cuda,
        'transformers':transformers.__version__,'huggingface_hub':huggingface_hub.__version__,
        'accelerate':accelerate.__version__,'tokenizers':tokenizers.__version__,
        'gpu':torch.cuda.get_device_name(0),'compute_capability':torch.cuda.get_device_capability(0),
        'gpu_memory_bytes':torch.cuda.get_device_properties(0).total_memory,
        'driver':subprocess.check_output(['nvidia-smi','--query-gpu=driver_version','--format=csv,noheader'],text=True).strip(),
        'inference_device':'cuda:0','weights_dtype':'bfloat16','CPU_offload':False}
(root/'provenance/gpu_agent_environment.json').write_text(json.dumps(record,indent=2)+'\n')
(root/'provenance/gpu_agent_environment.lock').write_text(subprocess.check_output(['.venv-agents/bin/python','-m','pip','freeze'],text=True))
print(json.dumps({k:v for k,v in record.items() if k!='model_files_sha256'},indent=2))
PY
