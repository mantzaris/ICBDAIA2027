#!/usr/bin/env python3
"""Download only pinned local reference models; no inference or remote code."""
import hashlib
import json
import time
import argparse
from pathlib import Path
from huggingface_hub import snapshot_download

root=Path(__file__).resolve().parents[1]
parser=argparse.ArgumentParser();parser.add_argument('--config',default='configs/ai_annotations.json');parser.add_argument('--report',default='reports/annotation_model_revisions.json')
args=parser.parse_args()
config=json.loads((root/args.config).read_text())
records=[]
for name,spec in config["models"].items():
    started=time.time()
    snapshot_download(repo_id=spec["model_id"],revision=spec["revision"],
                      local_dir=str(root/spec["model_path"]),
                      allow_patterns=["*.json","*.safetensors","*.txt","*.jinja","*.model","LICENSE*","README.md"])
    directory=root/spec["model_path"]
    files=[]
    for path in sorted(directory.glob("*")):
        if path.is_file():
            digest=hashlib.sha256()
            with path.open("rb") as stream:
                for block in iter(lambda:stream.read(8*1024*1024),b""):digest.update(block)
            files.append(dict(name=path.name,bytes=path.stat().st_size,sha256=digest.hexdigest()))
    records.append(dict(name=name,**spec,download_wall_seconds=time.time()-started,files=files))
    print("DOWNLOADED",name,flush=True)
(root/"reports").mkdir(exist_ok=True)
(root/args.report).write_text(json.dumps(records,indent=2))
