#!/usr/bin/env python3
"""Execution-only smoke on exact training demonstration images; no target labels."""
import argparse
import gc
import json
from pathlib import Path
import torch
from sonar.annotators import AnnotationVLM,file_hash
from sonar.annotation_batching import AnnotationBatchAdapter
from run_reference_labels import demonstration,VIEWS
from smoke_annotation_batching import comparison


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--model-size',choices=['4b','8b'],required=True);ap.add_argument('--batch-size',choices=[2,4],type=int,default=4);args=ap.parse_args()
    root=Path('results/ai_reference_training_execution_smoke')/('internvl'+args.model_size+('_batch2' if args.batch_size==2 else ''))
    root.mkdir(parents=True,exist_ok=True);destination=root/'smoke.json'
    if destination.exists():raise RuntimeError('Refuse to overwrite execution evidence')
    config_path='configs/ai_annotations.json' if args.model_size=='4b' else 'configs/ai_annotations_internvl8b_pilot.json'
    config=json.loads(Path(config_path).read_text());spec=config['models']['internvl']
    requests=[]
    for tag,domain,window,band in VIEWS:
        examples=demonstration(domain,window,band,root/'training_examples'/tag)
        target=examples[-1]
        requests.append(dict(prompt=target['prompt'],images=[target['image']],max_new_tokens=768,examples=examples))
    model=AnnotationVLM(spec,config)
    report=dict(purpose='software execution equivalence only; no acoustic qualification or accuracy inference',
                inputs='exact training demonstration images reused as target fixtures; no calibration/audit/main targets',
                model_id=spec['model_id'],revision=spec['revision'],config_sha256=file_hash(config_path),
                adapter_sha256=file_hash('src/sonar/annotation_batching.py'),parser_sha256=file_hash('src/sonar/json_protocol.py'),
                requests=requests,batch1_exact=False,batch4_runtime_success=False,compared_batch_size=args.batch_size)
    try:
        native=[model.generate(**r) for r in requests];report['native_serial']=native
        with AnnotationBatchAdapter(model,max_batch_size=1,max_wait_ms=0) as adapter:
            one=adapter.generate_batch(requests);report['batch1_ledger']=list(adapter.batch_ledger)
        compared=comparison(native,one);report.update(batch1=one,batch1_comparison=compared,batch1_exact=all(x['exact'] for x in compared))
        if report['batch1_exact']:
            with AnnotationBatchAdapter(model,max_batch_size=args.batch_size,max_wait_ms=80) as adapter:
                four=adapter.generate_batch(requests);report['batch4_ledger']=list(adapter.batch_ledger)
            compared=comparison(native,four)
            report.update(batch4=four,batch4_comparison=compared,batch4_exact=all(x['exact'] for x in compared),
                batch4_runtime_success=True,batch4_parsed_equal=sum(x['parsed_equal'] for x in compared))
            report['batch4_gpu_seconds']=sum(x['batch_gpu_seconds'] for x in report['batch4_ledger'])
            report['native_gpu_seconds']=sum(x['gpu_seconds'] for x in native)
    except Exception as exc:
        report['runtime_failure']=dict(type=type(exc).__name__,message=str(exc))
    destination.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(json.dumps({k:report.get(k) for k in ['model_id','batch1_exact','batch4_runtime_success','batch4_exact','batch4_parsed_equal','native_gpu_seconds','batch4_gpu_seconds','runtime_failure']},indent=2),flush=True)
    del model;gc.collect();torch.cuda.empty_cache()


if __name__=='__main__':main()
