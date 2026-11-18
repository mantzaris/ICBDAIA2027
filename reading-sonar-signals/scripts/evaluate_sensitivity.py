"""Measure output changes under probes; real changes are not causal correctness."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import torch
from sonar.metrics import event_metrics,aggregate_event_metrics
from evaluate_benchmark import report_events


def read(path):return json.loads(Path(path).read_text())
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def tensor(values):return torch.tensor(values,device='cuda',dtype=torch.float64)


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--run',type=Path,default=Path('results/sensitivity'))
    ap.add_argument('--main',type=Path,default=Path('results/main'))
    args=ap.parse_args()
    config=read(args.run/'run.json');manifest=Path(config['manifest'])
    if sha(manifest)!=config['manifest_sha256']:raise ValueError('Probe manifest changed')
    if config['source_sha256']!=read(args.main/'run.json')['source_sha256']:
        raise ValueError('Sensitivity and original episodes use different behavior')
    records=[json.loads(line) for line in manifest.read_text().splitlines() if line.strip()]
    rows=[]
    for record in records:
        for method in config['methods']:
            baseline=args.main/record['base_scene_id']/method/'episode.json'
            path=args.run/record['scene_id']/method/'episode.json'
            row=dict(scene_id=record['scene_id'],base_scene_id=record['base_scene_id'],method=method,
                     condition=record['sensitivity']['name'],stratum=record['stratum'],group_id=record['group_id'],
                     original_path=str(baseline),probe_path=str(path))
            if not baseline.exists() or not path.exists():
                rows.append(dict(row,status='missing_or_failed'));continue
            original,probe=read(baseline),read(path)
            a,ai,avalid=report_events(original.get('report'));b,bi,bvalid=report_events(probe.get('report'))
            row.update(status='complete',original_sha256=sha(baseline),probe_sha256=sha(path),
                original_schema_valid=bool(avalid and original.get('schema_valid')),
                probe_schema_valid=bool(bvalid and probe.get('schema_valid')),
                original_component_count=len(a)+len(ai),probe_component_count=len(b)+len(bi),
                both_empty=not a and not b,component_count_change=len(b)+len(bi)-len(a)-len(ai),
                output_agreement=None)
            if record['sensitivity']['name']!='duration' and (a or b):
                row['output_agreement']=event_metrics(b,a,frequency_tolerance_hz=2.)
            rows.append(row)
    summaries=[]
    for method,condition in sorted({(row['method'],row['condition']) for row in rows}):
        selected=[row for row in rows if row['method']==method and row['condition']==condition]
        complete=[row for row in selected if row['status']=='complete']
        metrics=[row['output_agreement'] for row in complete if row['output_agreement'] is not None]
        summaries.append(dict(method=method,condition=condition,expected_episodes=len(selected),complete_episodes=len(complete),
            schema_valid_fraction=tensor([r['probe_schema_valid'] for r in complete]).mean().item() if complete else None,
            both_empty_fraction=tensor([r['both_empty'] for r in complete]).mean().item() if complete else None,
            mean_component_count_change=tensor([r['component_count_change'] for r in complete]).mean().item() if complete else None,
            output_agreement=aggregate_event_metrics(metrics) if metrics else None,
            agreement_scored_episodes=len(metrics)))
    result=dict(interpretation='Agreement between transformed-input and original-input reports measures output stability, not physical correctness. Empty/empty pairs are separate; changed duration is evaluated against clipped exact truth by the benchmark evaluator.',
        real_attenuation_scope='Prespecified 200–400Hz target attenuation and matched 1000–1200Hz sham, both10–20s. Band selection uses no test outcomes and may miss actual components. No causal source attribution follows.',
        display_mismatch_scope='Initial DEMON image comes from another exact synthetic scene; numerical evidence and subsequent inspection remain target-based. This is a deliberately incoherent interface test.',
        input_manifest_sha256=sha(manifest),run_sha256=sha(args.run/'run.json'),rows=rows,summaries=summaries)
    output=args.run/'evaluation/output_sensitivity.json';output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps({'output':str(output),'rows':len(rows),'conditions':len(summaries)}))


if __name__=='__main__':main()
