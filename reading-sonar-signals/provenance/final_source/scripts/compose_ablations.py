"""Assemble a common-cohort ablation run without repeating unchanged inference.

F and G4 are reused byte-for-byte from main; no_verifier exposes the same F raw
report before numerical verification. Its inference cost is inherited for fair
method accounting, while additional inference cost is explicitly zero.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import time


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, value):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    payload=json.dumps(value,indent=2)+'\n'
    if path.exists() and path.read_text()!=payload:
        raise RuntimeError(f'Refusing to overwrite a different composed artifact: {path}')
    path.write_text(payload)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--main',type=Path,default=Path('results/main'))
    parser.add_argument('--inference',type=Path,default=Path('results/ablation_inference'))
    parser.add_argument('--manifest',type=Path,default=Path('data/manifests/ablation_test.jsonl'))
    parser.add_argument('--output',type=Path,default=Path('results/ablations'))
    args=parser.parse_args()
    records=[json.loads(line) for line in args.manifest.read_text().splitlines() if line.strip()]
    main_config=read(args.main/'run.json'); inference_config=read(args.inference/'run.json')
    if sha(args.manifest)!=inference_config['manifest_sha256']:
        raise ValueError('Ablation cohort differs from completed inference cohort')
    if not all((root/'COMPLETED.json').exists() for root in [args.main,args.inference]):
        raise ValueError('Both source inference phases must finish before composition')
    if main_config.get('source_sha256')!=inference_config.get('source_sha256'):
        raise ValueError('Baseline and ablation inference used different frozen behavior')
    methods=['F','G4']+inference_config['methods']+['no_verifier']
    if len(set(methods))!=len(methods): raise ValueError('Duplicate composed method')
    provenance=[]
    for row in records:
        for method in methods:
            source_method='F' if method=='no_verifier' else method
            source_root=args.main if method in ['F','G4','no_verifier'] else args.inference
            source=source_root/row['scene_id']/source_method/'episode.json'
            if not source.exists():
                failure=source.with_name('failure.json')
                if not failure.exists(): raise ValueError(f'Missing episode/failure: {source}')
                value=read(failure); value['method']=method
                save(args.output/row['scene_id']/method/'failure.json',value)
                provenance.append(dict(scene_id=row['scene_id'],method=method,source=str(failure),sha256=sha(failure),status='failed'))
                continue
            value=read(source)
            if method=='no_verifier':
                value['method']=method
                value['report']=value.get('raw_report') if value.get('schema_valid') else None
                value['verification_applied']=False
                value['derivation']={'source_episode':str(source),'source_sha256':sha(source),
                    'intervention':'expose identical F generation before measurement verification',
                    'additional_inference_calls':0,'additional_gpu_seconds':0,
                    'timing_meaning':'Inherited F inference cost; no numerical verification savings estimated.'}
            save(args.output/row['scene_id']/method/'episode.json',value)
            provenance.append(dict(scene_id=row['scene_id'],method=method,source=str(source),sha256=sha(source),status='complete'))
    config=dict(inference_config,manifest=str(args.manifest),manifest_sha256=sha(args.manifest),
        scene_count=len(records),methods=methods,composition=True,
        composition_sources={str(root/'run.json'):sha(root/'run.json') for root in [args.main,args.inference]},
        timing_note='Logical method costs inherited from original episodes. Reused F/G4 and no-verifier reports add zero model calls; do not sum composed costs as extra billed GPU time.')
    save(args.output/'run.json',config)
    save(args.output/'composition_provenance.json',provenance)
    marker=args.output/'COMPLETED.json'
    if not marker.exists(): save(marker,dict(composed_at_unix=time.time(),episodes=len(provenance),additional_inference_calls=0))
    print(json.dumps({'composed_episodes':len(provenance),'methods':methods,'output':str(args.output)}))


if __name__=='__main__': main()
