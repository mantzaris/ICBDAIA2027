"""Pre-main supplemental ablations on the entire existing synthetic stratum.

The original80-real ablation cohort and all inference/scoring code stay frozen.
This supplement was added after failed AI-reference qualification, before main
generative test inference. It supplies controlled mechanism evidence only.
"""
from __future__ import annotations
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    root=Path(__file__).resolve().parents[1];os.chdir(root)
    os.environ['PYTHONPATH']=str(root/'src')
    config=json.loads(Path('configs/synthetic_ablation_amendment.json').read_text())
    for path,expected in config['sha256'].items():
        if sha(path)!=expected:raise RuntimeError('Supplement frozen input changed: '+path)
    base=json.loads(Path('configs/final_protocol_freeze.json').read_text())
    for path,expected in base['sha256'].items():
        if sha(path)!=expected:raise RuntimeError('Original protocol changed: '+path)
    manifest='data/manifests/synthetic_ablation_test.jsonl'
    methods='no_demon,no_graph,no_quantitative,forced_exclusivity,fixed_resolution,random'
    logs=Path('reports/synthetic_ablation_phases');logs.mkdir(exist_ok=True)
    def run(name,arguments):
        started=time.time();log=logs/f'{name}.{int(started)}.log'
        with log.open('w') as stream:
            result=subprocess.run([sys.executable]+arguments,stdout=stream,stderr=subprocess.STDOUT)
        record=dict(phase=name,argv=arguments,started_at_unix=started,finished_at_unix=time.time(),
                    returncode=result.returncode,log=str(log),log_sha256=sha(log))
        with (logs/'ledger.jsonl').open('a') as stream:stream.write(json.dumps(record)+'\n')
        print(json.dumps(record),flush=True)
        if result.returncode:raise RuntimeError('Supplement phase failed: '+name)
    run('inference',['scripts/run_benchmark.py','--manifest',manifest,'--output','results/synthetic_ablation_inference',
        '--methods',methods,'--budget','6','--workers','16','--batch-size','8','--batch-wait-ms','2000'])
    run('composition',['scripts/compose_ablations.py','--main','results/main','--inference','results/synthetic_ablation_inference',
        '--manifest',manifest,'--output','results/synthetic_ablations'])
    inventory=[]
    for line in Path(manifest).read_text().splitlines():
        record=json.loads(line);sid=record['scene_id']
        source=Path('results/main/sealed_truth')/(sid+'.json')
        if json.loads(source.read_text()).get('scene_id')!=sid:raise RuntimeError('Exact-truth scene identity mismatch')
        destination=Path('results/synthetic_ablations/sealed_truth')/source.name
        destination.parent.mkdir(exist_ok=True)
        if destination.exists() and sha(destination)!=sha(source):raise RuntimeError('Existing exact truth differs')
        shutil.copy2(source,destination)
        inventory.append(dict(scene_id=sid,source=str(source),destination=str(destination),sha256=sha(source)))
    Path('results/synthetic_ablations/exact_truth_provenance.json').write_text(json.dumps(inventory,indent=2)+'\n')
    run('evaluation',['scripts/evaluate_benchmark.py','--run','results/synthetic_ablations',
        '--references','results/ai_annotations/primary_reference.jsonl'])
    print('SYNTHETIC_ABLATION_SUPPLEMENT_FINISHED',flush=True)


if __name__=='__main__':main()
