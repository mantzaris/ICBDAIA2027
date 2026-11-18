"""Sequential, fail-fast GPU study execution after explicit protocol freeze.

Launch only after AI-reference assembly and the final development compatibility
smoke. This script neither changes protocol nor stops the pod; retrieval and
verified local preservation are required before the separate stop command.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time


def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--workers',type=int,default=16)
    parser.add_argument('--batch-size',type=int,default=8)
    parser.add_argument('--batch-wait-ms',type=int,default=2000)
    parser.add_argument('--start',default='self_test')
    parser.add_argument('--stop-after')
    args=parser.parse_args()
    root=Path(__file__).resolve().parents[1]; os.chdir(root)
    os.environ['PYTHONPATH']=str(root/'src')
    reference='results/ai_annotations/primary_reference.jsonl'
    selection='results/development/evaluation/fixed_selection.json'
    for required in ['configs/ai_reference_freeze.json','configs/final_protocol_freeze.json']:
        if not Path(required).is_file(): raise RuntimeError(f'Unfinished prerequisite: {required}')
    freeze=json.loads(Path('configs/final_protocol_freeze.json').read_text())
    for path,expected in freeze['sha256'].items():
        if sha(path)!=expected: raise RuntimeError(f'Frozen input changed: {path}')
    def infer(manifest,output,methods,budget=6):
        return ['scripts/run_benchmark.py','--manifest',f'data/manifests/{manifest}.jsonl','--output',f'results/{output}',
                '--methods',methods,'--budget',str(budget),'--workers',str(args.workers),
                '--batch-size',str(args.batch_size),'--batch-wait-ms',str(args.batch_wait_ms)]
    def evaluate(output,*extras):
        return ['scripts/evaluate_benchmark.py','--run',f'results/{output}','--references',reference,*extras]
    phases=[
      ('self_test',['scripts/evaluate_benchmark.py','--self-test']),
      ('perception',['scripts/train_perception.py']),
      ('localization',['scripts/train_localization.py']),
      ('freeze_cases',['scripts/make_figures.py','--freeze-cases-only']),
      ('localization_evaluate',evaluate('localization/predictions')),
      ('development',infer('development','development','G3,G4')),
      ('development_evaluate',evaluate('development','--select-fixed')),
      ('main',infer('main_test','main','G1,G2,G3,G4,F')),
      ('main_evaluate',evaluate('main','--fixed-selection',selection)),
      ('diagnostic_internvl4b',['scripts/evaluate_benchmark.py','--run','results/main','--references','results/ai_annotations/diagnostic_internvl4b.jsonl','--output','results/main/diagnostic_internvl4b']),
      ('diagnostic_internvl8b',['scripts/evaluate_benchmark.py','--run','results/main','--references','results/ai_annotations/diagnostic_internvl8b.jsonl','--output','results/main/diagnostic_internvl8b']),
      ('ablation_inference',infer('ablation_test','ablation_inference','no_demon,no_graph,no_quantitative,forced_exclusivity,fixed_resolution,random')),
      ('compose_ablations',['scripts/compose_ablations.py']),
      ('ablations_evaluate',evaluate('ablations','--fixed-selection',selection)),
      ('budget2',infer('budget_test','budget2','F,G3,G4',2)),
      ('budget2_evaluate',evaluate('budget2','--fixed-selection',selection)),
      ('budget4',infer('budget_test','budget4','F,G3,G4',4)),
      ('budget4_evaluate',evaluate('budget4','--fixed-selection',selection)),
      ('robustness',infer('robustness_test','robustness','F,G4')),
      ('robustness_evaluate',evaluate('robustness','--fixed-selection',selection)),
      ('sensitivity',infer('sensitivity_test','sensitivity','F,G4')),
      ('sensitivity_evaluate',evaluate('sensitivity','--fixed-selection',selection)),
      ('output_sensitivity',['scripts/evaluate_sensitivity.py']),
      ('figures',['scripts/make_figures.py']),
    ]
    names=[name for name,_ in phases]
    if args.start not in names or args.stop_after and args.stop_after not in names:
        raise ValueError('Unknown phase name')
    log_dir=Path('reports/study_phases'); log_dir.mkdir(parents=True,exist_ok=True)
    ledger=log_dir/'ledger.jsonl'
    for name,command in phases[names.index(args.start):]:
        if reference in command or name=='freeze_cases':
            if not Path(reference).is_file() or not Path('results/ai_annotations/COMPLETED.json').is_file():
                raise RuntimeError('Final dual AI annotations/audit and assembly must finish before '+name)
        if shutil.disk_usage(root).free<1_000_000_000: raise RuntimeError('Less than 1GB free; preserve artifacts before further execution')
        started=time.time(); log=log_dir/f'{name}.{int(started)}.log'
        record={'phase':name,'started_at_unix':started,'argv':[sys.executable]+command,'log':str(log)}
        (log_dir/'active.json').write_text(json.dumps(record,indent=2)+'\n')
        print(json.dumps(dict(event='phase_start',**record)),flush=True)
        with log.open('w') as handle:
            completed=subprocess.run([sys.executable]+command,stdout=handle,stderr=subprocess.STDOUT)
        record.update(finished_at_unix=time.time(),returncode=completed.returncode,log_sha256=sha(log))
        with ledger.open('a') as handle: handle.write(json.dumps(record)+'\n')
        print(json.dumps(dict(event='phase_end',**record)),flush=True)
        if completed.returncode: raise RuntimeError(f'Phase {name} failed; retained log {log}')
        if name==args.stop_after: break
    (log_dir/'invocation_complete.json').write_text(json.dumps({'finished_at_unix':time.time(),'start':args.start,'stop_after':args.stop_after})+'\n')


if __name__=='__main__': main()
