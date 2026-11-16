"""Replay and score the separate, group/date-disjoint configuration-gate follow-up."""
import hashlib
import json
from pathlib import Path
import numpy as np
from acoustic_regeneration.agents.benchmark_tools import BenchmarkTools, qualifies, score_report, solve_rules
from acoustic_regeneration.agents.guarded_tools import GuardedBenchmarkTools
from analyze_gpu_agent_benchmark import assert_close, compare, wilson

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'results/gpu_agent_guard'


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    completion=json.loads((BASE/'evaluation/completion.json').read_text())
    assert completion['cases']==16 and completion['episodes']==32
    for name,expected in completion['artifact_sha256'].items():assert sha(ROOT/name)==expected,name
    cases=json.loads((BASE/'inputs/manifest.json').read_text())['cases']
    primary=json.loads((ROOT/'results/gpu_agent_benchmark/inputs/manifest.json').read_text())['cases']
    assert not ({c['group_id'] for c in cases}&{c['group_id'] for c in primary})
    assert not ({c['date'] for c in cases}&{c['date'] for c in primary})
    grouped={arm:[] for arm in ['reasoning','reasoning_guarded']};hashes={};replayed=0;turns=0
    for case in cases:
        cid=case['case_id'];task=json.loads((BASE/'inputs'/f'{cid}.json').read_text())
        with np.load(BASE/'inputs'/f'{cid}.npz',allow_pickle=False) as d:audio,mask=d['audio'],d['observed']
        expected=solve_rules(BenchmarkTools(audio,mask,task));assert_close(expected,case['expected_report'])
        measurements={e['arguments']['window_ms']:e['result'] for e in case['reference_tool_trace'] if e['name']=='analyze_window'}
        for arm in grouped:
            path=BASE/'evaluation'/f'{cid}__{arm}.json';summary=json.loads(path.read_text())
            events_path=path.with_suffix('.jsonl');assert sha(events_path)==summary['events_sha256']
            tools=(GuardedBenchmarkTools if arm=='reasoning_guarded' else BenchmarkTools)(audio,mask,task)
            applied_bad=0;rejected=0;event_tools=[];tokens=0
            for event in map(json.loads,events_path.read_text().splitlines()):
                if event['kind']=='tool_result':
                    result=tools.call(event['name'],event['arguments'])
                    saved={k:event[k] for k in ['evidence_id','name','arguments','result']}
                    assert_close(result,saved);event_tools.append(saved);replayed+=1
                    if event['name']=='configure_display':
                        if 'error' not in event['result']:
                            applied_bad+=int(not qualifies(measurements[event['arguments']['window_ms']],task['constraints']))
                        else:rejected+=int('Configuration rejected:' in event['result']['error'])
                elif event['kind']=='model_response':
                    assert event['model_device']=='cuda:0';turns+=1;tokens+=event['generated_tokens']
                    prompt=json.dumps(event['request_messages'])
                    assert case['source_id'] not in prompt and 'expected_report' not in prompt
                    for message in event['request_messages']:
                        if message['role']=='tool':assert json.loads(message['content']) in event_tools
            assert_close(event_tools,summary['tool_trace'])
            assert tokens==summary['generated_tokens']
            checks=score_report(summary['final_report'],tools,expected);assert checks==summary['checks']
            assert tools.input_hash()==tools.initial_hash
            if arm=='reasoning_guarded':assert applied_bad==0
            grouped[arm].append({**summary,'constraint_violating_configurations':applied_bad,'rejected_infeasible_changes':rejected})
            for p in [path,events_path]:hashes[str(p.relative_to(ROOT))]=sha(p)
    metrics={}
    for arm,rows in grouped.items():
        successes=sum(r['checks']['strict_success'] for r in rows)
        metrics[arm]={'n':16,'successes':successes,'success_rate':successes/16,'wilson_95':wilson(successes,16),
            'setting_correct':sum(r['checks']['setting_correct'] for r in rows),
            'cue_correct':sum(r['checks']['cue_correct'] for r in rows),
            'cases_with_infeasible_applied_change':sum(r['constraint_violating_configurations']>0 for r in rows),
            'infeasible_applied_changes':sum(r['constraint_violating_configurations'] for r in rows),
            'rejected_infeasible_changes':sum(r['rejected_infeasible_changes'] for r in rows),
            'mean_generated_tokens':float(np.mean([r['generated_tokens'] for r in rows])),
            'median_elapsed_seconds':float(np.median([r['elapsed_seconds'] for r in rows])),
            'total_inference_seconds':sum(r['inference_seconds'] for r in rows),
            'failed_or_incomplete':sum(r['status']!='completed' for r in rows),
            'unsupported_report_cases':sum((r['checks']['unsupported_claims'] or 0)>0 for r in rows)}
    paired=compare([r['checks']['strict_success'] for r in grouped['reasoning_guarded']],
                   [r['checks']['strict_success'] for r in grouped['reasoning']],np.random.default_rng(20261117))
    output={'status':'completed','cases':16,'episodes':32,'source_groups':16,'source_dates':16,
            'primary_groups_and_dates_disjoint':True,'metrics':metrics,'paired_guard_minus_unguarded':paired,
            'completion':completion,'scope':'Separate intervention motivated by initial primary traces; new agent groups within V, same model/prompt/budgets; no physical hardware',
            'interpretation':'Blocking infeasible settings follows from the gate definition; improved end-to-end completion requires the paired outcome comparison'}
    (BASE/'summary.json').write_text(json.dumps(output,indent=2)+'\n')
    validation={'status':'passed','cases':16,'episodes':32,'tool_returns_replayed':replayed,
                'CUDA_turns_verified':turns,'group_and_date_disjoint_from_primary':True,
                'all_primary_scores_recomputed':True,'artifact_sha256':hashes,
                'analysis_sha256':sha(Path(__file__)),'summary_sha256':sha(BASE/'summary.json')}
    (ROOT/'reports/gpu_agent_guard_validation.json').write_text(json.dumps(validation,indent=2)+'\n')
    print(json.dumps({'metrics':metrics,'paired':paired},indent=2))


if __name__=='__main__':main()
