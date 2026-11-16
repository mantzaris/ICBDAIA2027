"""Replay measurements, audit prompts, and summarize every frozen GPU-agent case."""
import hashlib
import json
import math
from pathlib import Path

import numpy as np

from acoustic_regeneration.agents.benchmark_tools import BenchmarkTools, solve_rules, score_report

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'results/gpu_agent_benchmark'
ARMS=['rules','direct','reasoning','self_review','two_agents']


def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,d): p.write_text(json.dumps(d,indent=2,allow_nan=False)+'\n')


def assert_close(actual,expected,path=''):
    if isinstance(expected,dict):
        assert isinstance(actual,dict) and actual.keys()==expected.keys(),path
        for k in expected: assert_close(actual[k],expected[k],path+'/'+k)
    elif isinstance(expected,(list,tuple)):
        assert isinstance(actual,(list,tuple)) and len(actual)==len(expected),path
        for i,(a,b) in enumerate(zip(actual,expected)): assert_close(a,b,path+'/'+str(i))
    elif isinstance(expected,float):
        assert math.isclose(actual,expected,rel_tol=1e-9,abs_tol=1e-5),(path,actual,expected)
    else: assert actual==expected,(path,actual,expected)


def wilson(k,n):
    z=1.959963984540054;p=k/n
    mid=(p+z*z/(2*n))/(1+z*z/n)
    half=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/(1+z*z/n)
    return [mid-half,mid+half]


def compare(a,b,rng):
    difference=np.asarray(a,dtype=float)-np.asarray(b,dtype=float)
    samples=rng.integers(0,len(a),size=(10000,len(a)))
    boot=difference[samples].mean(axis=1)
    wins=int(np.sum(difference>0));losses=int(np.sum(difference<0));n=wins+losses
    p=min(1.,2*sum(math.comb(n,k) for k in range(min(wins,losses)+1))/2**n) if n else 1.
    return {'difference':float(difference.mean()),'paired_bootstrap_95':np.quantile(boot,[.025,.975]).tolist(),
            'wins':wins,'losses':losses,'ties':len(a)-n,'mcnemar_exact_p':p}


def main():
    manifest=json.loads((BASE/'inputs/manifest.json').read_text())
    cases=[c for c in manifest['cases'] if c['split']=='evaluation']
    completion=json.loads((BASE/'evaluation/completion.json').read_text())
    assert completion['cases']==32 and completion['episodes']==160
    definition=json.loads((BASE/'evaluation/run_definition.json').read_text())
    for p,h in definition['artifact_sha256'].items(): assert sha(ROOT/p)==h,p
    grouped={arm:[] for arm in ARMS}; records=[]; hashes={}; replay_count=0; prompt_count=0; gpu_turns=0
    for case in cases:
        cid=case['case_id']; task=json.loads((BASE/'inputs'/f'{cid}.json').read_text())
        with np.load(BASE/'inputs'/f'{cid}.npz',allow_pickle=False) as d:
            audio,mask=d['audio'],d['observed']
        expected=solve_rules(BenchmarkTools(audio,mask,task))
        assert_close(expected,case['expected_report'],cid+'/expected')
        for arm in ARMS:
            path=BASE/'evaluation'/f'{cid}__{arm}.json'
            summary=json.loads(path.read_text()); events_path=path.with_suffix('.jsonl')
            assert sha(events_path)==summary['events_sha256'],events_path
            tools=BenchmarkTools(audio,mask,task)
            event_tools=[]; model_tokens=0; thinking_tokens=0; action_caps=0
            for event in map(json.loads,events_path.read_text().splitlines()):
                if event['kind']=='tool_result':
                    result=tools.call(event['name'],event['arguments'])
                    saved={k:event[k] for k in ['evidence_id','name','arguments','result']}
                    assert_close(result,saved,cid+'/'+arm+'/'+event['evidence_id'])
                    event_tools.append(saved);replay_count+=1
                if event['kind']=='model_response':
                    assert event['model_device']=='cuda:0'
                    model_tokens+=event['generated_tokens'];thinking_tokens+=event['reasoning_tokens']
                    action_caps+=int(event['action_hit_token_cap']);gpu_turns+=1
                    messages=event['request_messages'];prompt_count+=1
                    text=json.dumps(messages)
                    assert case['source_id'] not in text and case['group_id'] not in text
                    assert 'expected_report' not in text and 'reference_tool_trace' not in text
                    # Every separately supplied tool result must be a return already executed.
                    for message in messages:
                        if message['role']=='tool':
                            item=json.loads(message['content'])
                            assert item in event_tools,(cid,arm,'future or fabricated tool return in prompt')
                    if arm=='direct': assert event['reasoning_tokens']==0
            assert_close(event_tools,summary['tool_trace'],cid+'/'+arm+'/trace')
            assert model_tokens==summary['generated_tokens']
            checks=score_report(summary['final_report'],tools,expected)
            assert checks==summary['checks'],(cid,arm,checks,summary['checks'])
            assert tools.input_hash()==tools.initial_hash and summary['input_unchanged']
            row={**summary,'reasoning_tokens':thinking_tokens,'action_cap_hits':action_caps}
            grouped[arm].append(row);records.append(row)
            hashes[str(path.relative_to(ROOT))]=sha(path)
            hashes[str(events_path.relative_to(ROOT))]=sha(events_path)
    metrics={}
    for arm,rows in grouped.items():
        n=len(rows);success=sum(r['checks']['strict_success'] for r in rows)
        metrics[arm]={'n':n,'successes':success,'success_rate':success/n,'wilson_95':wilson(success,n),
            'setting_correct':sum(r['checks']['setting_correct'] for r in rows),
            'cue_correct':sum(r['checks']['cue_correct'] for r in rows),
            'numeric_correct':sum(r['checks']['numeric_correct'] for r in rows),
            'valid_reports':sum(r['checks']['valid_report'] for r in rows),
            'verified_feasible_cases':sum(r['checks']['verification_complete'] for r,c in zip(rows,cases) if c['expected_report']['decision']=='configure'),
            'feasible_cases':sum(c['expected_report']['decision']=='configure' for c in cases),
            'unsupported_report_cases':sum((r['checks']['unsupported_claims'] or 0)>0 for r in rows),
            'identity_and_gap_correct':sum(r['checks']['identity_and_gap_correct'] for r in rows),
            'failed_or_incomplete_episodes':sum(r['status']!='completed' for r in rows),
            'mean_generated_tokens':float(np.mean([r['generated_tokens'] for r in rows])),
            'mean_reasoning_tokens':float(np.mean([r['reasoning_tokens'] for r in rows])),
            'median_elapsed_seconds':float(np.median([r['elapsed_seconds'] for r in rows])),
            'total_inference_seconds':float(sum(r['inference_seconds'] for r in rows)),
            'tool_calls':sum(r['tool_calls'] for r in rows),'tool_errors':sum(r['tool_errors'] for r in rows),
            'forced_reasoning_closures':sum(r['forced_reasoning_closures'] for r in rows),
            'action_cap_hits':sum(r['action_cap_hits'] for r in rows),
            'families':{family:{'n':sum(r['family']==family for r in rows),
                               'successes':sum(r['checks']['strict_success'] for r in rows if r['family']==family)}
                        for family in sorted({r['family'] for r in rows})}}
    rng=np.random.default_rng(20261116);paired={}
    for a,b in [('reasoning','direct'),('self_review','reasoning'),('two_agents','self_review')]:
        paired[a+'_minus_'+b]=compare([r['checks']['strict_success'] for r in grouped[a]],
                                     [r['checks']['strict_success'] for r in grouped[b]],rng)
    ordered=sorted(paired,key=lambda k:paired[k]['mcnemar_exact_p']);previous=0.
    for rank,key in enumerate(ordered):
        corrected=max(previous,min(1.,(len(ordered)-rank)*paired[key]['mcnemar_exact_p']))
        paired[key]['holm_p']=corrected;previous=corrected
    summary={'status':'completed','cases':32,'episodes':160,'neural_episodes':128,
             'source_groups':len({c['group_id'] for c in cases}),'source_dates':len({c['date'] for c in cases}),
             'reference_labels':{k:sum(c['expected_report']['cue_label']==k for c in cases) for k in ['stable','changed','insufficient']},
             'metrics':metrics,'paired_comparisons':paired,'gpu_completion':completion,
             'scope':'One model, one decoding seed, bounded reasoning, task-defined acoustic cue labels, reused V recordings; no object classification or hardware trial',
             'numeric_replay_tolerance':1e-5,'unsupported_claim_metric':'Structured numeric/citation/verification claims; invalid reports counted separately; no free-text semantic score',
             'equal_budget_note':'Equal ceilings, not equal actual consumption; deterministic rules encode the task policy and provide a transparent ceiling'}
    write(BASE/'summary.json',summary)
    validation={'status':'passed','tool_returns_replayed':replay_count,'model_prompts_audited':prompt_count,
                'CUDA_model_turns_verified':gpu_turns,'source_groups':32,'source_dates':32,
                'hidden_references_or_dataset_classes_sent_to_models':False,
                'all_input_hashes_preserved':True,'all_scores_recomputed':True,'artifact_sha256':hashes,
                'analysis_script_sha256':sha(Path(__file__)),'summary_sha256':sha(BASE/'summary.json'),
                'limitations':['Task-rule ground truth is not an independent object-identification annotation',
                               'Forced reasoning closure and any incomplete outputs remain included',
                               'The reviewer uses the same model weights and runs sequentially on one GPU']}
    write(ROOT/'reports/gpu_agent_validation.json',validation)
    print(json.dumps({'metrics':metrics,'paired':paired},indent=2))


if __name__=='__main__': main()
