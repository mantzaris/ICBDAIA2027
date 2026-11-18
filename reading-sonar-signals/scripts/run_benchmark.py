"""Resumable CUDA episodes with immutable provenance and explicit failure records."""
from __future__ import annotations
import argparse
import hashlib
import json
import time
import traceback
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import torch
from sonar.data import read_manifest, opaque_id
from sonar.input_interventions import load_experimental_waveform, wrap_display_inputs
from sonar.evidence import AcousticTools
from sonar.controller import run_episode
from sonar.vlm import LocalVLM
from sonar.batching import BatchedLocalVLM, GPU_LOCK
from sonar.constrained_decoding import grammar_provenance
from sonar.prompting import PROMPT_PROTOCOL


def save_json(path, value):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value,indent=2)+'\n')
    temporary.replace(path)


def freeze_or_resume(destination, configuration):
    """Never combine completed episodes from different source/protocol states."""
    path=Path(destination)/'run.json'
    if path.exists():
        original=json.loads(path.read_text())
        new={k:v for k,v in configuration.items() if k!='started_at_unix'}
        old={k:v for k,v in original.items() if k!='started_at_unix'}
        differences=sorted(k for k in set(new)|set(old) if new.get(k)!=old.get(k))
        if differences:
            raise ValueError('Existing run has incompatible immutable configuration: '+', '.join(differences)+'. Use a fresh output directory.')
        with (Path(destination)/'resume_ledger.jsonl').open('a') as handle:
            handle.write(json.dumps({'resumed_at_unix':configuration['started_at_unix'],
                                     'original_started_at_unix':original['started_at_unix']})+'\n')
        return original
    if any(Path(destination).rglob('episode.json')):
        raise ValueError('Existing episodes have no frozen run.json; use a fresh output directory')
    save_json(path,configuration)
    return configuration


def episode_is_complete(path, scene_id, method):
    """An interrupted JSON write is preserved and rerun under the same protocol."""
    path=Path(path)
    if not path.exists():
        return False
    try:
        value=json.loads(path.read_text())
    except (json.JSONDecodeError,UnicodeDecodeError):
        path.replace(path.with_name(f'episode.incomplete.{time.time_ns()}.json'))
        return False
    if not isinstance(value,dict) or value.get('scene_id')!=scene_id or value.get('method')!=method:
        raise ValueError('Existing episode identity does not match its frozen scene/method path')
    return True


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--manifest',required=True)
    parser.add_argument('--output',required=True)
    parser.add_argument('--methods',default='G1,G2,G3,G4,F')
    parser.add_argument('--budget',type=int,default=6)
    parser.add_argument('--model',default='models/qwen3-vl-8b')
    parser.add_argument('--limit',type=int)
    parser.add_argument('--workers',type=int,default=1)
    parser.add_argument('--batch-size',type=int,default=4)
    parser.add_argument('--batch-wait-ms',type=int,default=2000)
    args=parser.parse_args()
    if args.workers<1 or args.batch_size<1 or args.budget<0 or args.batch_wait_ms<0 or (args.limit is not None and args.limit<1):
        raise ValueError('Positive worker/batch/limit values and a nonnegative action budget are required')
    methods=args.methods.split(',')
    if len(methods)!=len(set(methods)) or not set(methods)<=set(['G1','G2','G3','G4','F','no_demon','no_graph','no_verifier','no_quantitative','forced_exclusivity','fixed_resolution','random']):
        raise ValueError('Methods must be distinct supported controller names')
    destination=Path(args.output); destination.mkdir(parents=True,exist_ok=True)
    manifest_bytes=Path(args.manifest).read_bytes()
    records=read_manifest(args.manifest)
    if args.limit is not None: records=records[:args.limit]
    if len({r['scene_id'] for r in records})!=len(records):
        raise ValueError('Scene IDs must be unique within a run')
    configuration={'manifest':args.manifest,'manifest_sha256':hashlib.sha256(manifest_bytes).hexdigest(),
                   'methods':args.methods.split(','),'budget':args.budget,'scene_count':len(records),
                   'model':args.model,'started_at_unix':time.time(),
                   'numerical_device':'cuda:0','decoding':'greedy',
                   'reference_annotations_visible_to_model':False}
    configuration['workers']=args.workers
    configuration['max_batch_size']=args.batch_size if args.workers>1 else 1
    configuration['batch_wait_ms']=args.batch_wait_ms if args.workers>1 else 0
    configuration['structured_decoding']=grammar_provenance()
    configuration['prompt_serialization']=PROMPT_PROTOCOL
    configuration['timing_note']='Batch GPU elapsed amortized equally over requests; serial latency is measured separately in pilot.' if args.workers>1 else 'Serial episode latency; CUDA events include host launch gaps.'
    all_source_hashes={str(path):hashlib.sha256(path.read_bytes()).hexdigest()
                       for path in sorted(Path('src/sonar').glob('*.py'))}
    behavior_modules=['signal','evidence','synthetic','metrics','data','vlm','batching','controller','rendering',
                      'json_protocol','constrained_decoding','prompting','input_interventions']
    configuration['source_sha256']={f'src/sonar/{name}.py':all_source_hashes[f'src/sonar/{name}.py']
                                    for name in behavior_modules}
    configuration['source_sha256']['scripts/run_benchmark.py']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    configuration['model_revision']=json.loads(Path('reports/model_revision.json').read_text())
    configuration=freeze_or_resume(destination,configuration)
    save_json(destination/'source_snapshots'/f'{time.time_ns()}.json',
              {'recorded_at_unix':time.time(),'source_sha256':all_source_hashes})
    torch.set_num_threads(4)
    backend=LocalVLM(args.model)
    model=BatchedLocalVLM(backend,max_batch_size=args.batch_size,max_wait_ms=args.batch_wait_ms) if args.workers>1 else backend
    (destination/'model_device_audit.json').write_text(json.dumps(model.device_audit,indent=2)+'\n')
    old_batch_path=destination/'batch_ledger.json'
    previous_batches=json.loads(old_batch_path.read_text()) if old_batch_path.exists() else []
    def process(record,method,number):
        scene_id=record['scene_id']
        episode_dir=destination/scene_id/method
        started=time.time()
        tools=None; waveform=None; ledger=None
        try:
            if episode_is_complete(episode_dir/'episode.json',scene_id,method):
                print(f'SKIP completed {scene_id} {method}',flush=True); return None
            print(f'START {number+1}/{len(records)} {scene_id} {method}',flush=True)
            with GPU_LOCK:
                waveform,sample_rate,origin,truth=load_experimental_waveform(record)
                if truth:
                    saved_truth=dict(truth)
                    if saved_truth.get('scene_id') not in [None,scene_id]:
                        saved_truth['generator_scene_id']=saved_truth['scene_id']
                    saved_truth['scene_id']=scene_id
                    save_json(destination/'sealed_truth'/f'{scene_id}.json',saved_truth)
                tools=AcousticTools(waveform,sample_rate,scene_id,offset_s=origin)
            setup_seconds=time.time()-started
            episode_model=wrap_display_inputs(record,model,episode_dir)
            result=run_episode(tools,episode_model,method,episode_dir,budget=args.budget,
                               render_cmap=record.get('intervention',{}).get('colormap','magma'))
            ledger={'scene_id':scene_id,'method':method,'status':'complete','wall_seconds':result['wall_seconds'],
                    'schema_valid':result['schema_valid'],'output_tokens':result['output_tokens'],
                    'gpu_seconds':result['model_gpu_seconds']+result['tool_gpu_seconds']+result['preprocess_gpu_seconds'],
                    'setup_wall_seconds':setup_seconds,'total_worker_wall_seconds':time.time()-started}
        except Exception as error:
            episode_dir.mkdir(parents=True,exist_ok=True)
            ledger={'scene_id':scene_id,'method':method,'status':'failed','error':str(error),
                    'traceback':traceback.format_exc(),'wall_seconds':time.time()-started}
            failure_path=episode_dir/'failure.json'
            if failure_path.exists():
                failure_path.replace(episode_dir/f'failure.previous.{time.time_ns()}.json')
            save_json(failure_path,ledger)
        finally:
            try:
                with GPU_LOCK:
                    tools=None; waveform=None
                    torch.cuda.empty_cache()
            except Exception as cleanup_error:
                if ledger is not None:
                    ledger['cleanup_error']=str(cleanup_error)
        return ledger
    completed=[]
    try:
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures=[pool.submit(process,record,method,number) for number,record in enumerate(records)
                     for method in args.methods.split(',')]
            for future in as_completed(futures):
                ledger=future.result()
                if ledger is None: continue
                completed.append(ledger)
                with (destination/'ledger.jsonl').open('a') as stream:
                    stream.write(json.dumps(ledger)+'\n')
                print(json.dumps(ledger),flush=True)
                if hasattr(model,'batch_ledger'):
                    save_json(destination/'batch_ledger.json',previous_batches+list(model.batch_ledger))
    finally:
        if isinstance(model,BatchedLocalVLM):
            model.close()
            save_json(destination/'batch_ledger.json',previous_batches+list(model.batch_ledger))
    (destination/'COMPLETED.json').write_text(json.dumps({'finished_at_unix':time.time(),
        'new_completed_episodes':len(completed),'execution_failures':sum(r['status']=='failed' for r in completed),
        **configuration},indent=2)+'\n')
    print('BENCHMARK_FINISHED',flush=True)


if __name__=='__main__': main()
