"""Descriptive review-transition and configuration-error audit; primary scores stay fixed."""
import hashlib
import json
from pathlib import Path

import numpy as np

from acoustic_regeneration.agents.benchmark_tools import BenchmarkTools, qualifies, score_report
from run_gpu_agent_benchmark import parse_action

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'results/gpu_agent_benchmark'


def main():
    cases=[c for c in json.loads((BASE/'inputs/manifest.json').read_text())['cases'] if c['split']=='evaluation']
    rows=[]
    for case in cases:
        cid=case['case_id'];task=json.loads((BASE/'inputs'/f'{cid}.json').read_text())
        with np.load(BASE/'inputs'/f'{cid}.npz',allow_pickle=False) as d:audio,mask=d['audio'],d['observed']
        measurements={e['arguments']['window_ms']:e['result'] for e in case['reference_tool_trace'] if e['name']=='analyze_window'}
        for arm in ['direct','reasoning','self_review','two_agents']:
            summary=json.loads((BASE/'evaluation'/f'{cid}__{arm}.json').read_text())
            tools=BenchmarkTools(audio,mask,task);draft=None;before_review=None;review_started=False
            bad_changes=[];new_review_tools=[]
            for event in map(json.loads,(BASE/'evaluation'/f'{cid}__{arm}.jsonl').read_text().splitlines()):
                if event['kind']=='tool_result':
                    tools.call(event['name'],event['arguments'])
                    if review_started:new_review_tools.append(event['name'])
                    if event['name']=='configure_display' and 'error' not in event['result']:
                        w=event['arguments']['window_ms']
                        if not qualifies(measurements[w],task['constraints']):bad_changes.append(event['evidence_id'])
                elif event['kind']=='model_response':
                    try:
                        _,candidate=parse_action(event['text'])
                        if candidate is not None:draft=candidate
                    except (ValueError,TypeError,KeyError):pass
                elif event['kind']=='review_start':
                    before_review={'draft_present':draft is not None,'checks':score_report(draft,tools,case['expected_report'])}
                    review_started=True
            final=summary['checks']['strict_success']
            transition=None
            if before_review is not None:
                initial=before_review['checks']['strict_success']
                transition=('kept_correct' if initial and final else 'regressed' if initial else 'rescued' if final else 'still_incorrect')
            rows.append({'case_id':cid,'arm':arm,'family':case['family'],'final_success':final,
                         'before_review':before_review,'review_transition':transition,
                         'review_tool_calls':new_review_tools,'constraint_violating_configurations':bad_changes})
    metrics={}
    for arm in ['direct','reasoning','self_review','two_agents']:
        subset=[r for r in rows if r['arm']==arm]
        metrics[arm]={'cases':len(subset),
            'cases_with_constraint_violating_configuration':sum(bool(r['constraint_violating_configurations']) for r in subset),
            'total_constraint_violating_configurations':sum(len(r['constraint_violating_configurations']) for r in subset),
            'review_transitions':{name:sum(r['review_transition']==name for r in subset) for name in ['kept_correct','regressed','rescued','still_incorrect']},
            'initial_draft_missing':sum(r['before_review'] is not None and not r['before_review']['draft_present'] for r in subset)}
    record={'scope':'Descriptive secondary audit specified during execution; not an additional confirmatory hypothesis or a changed primary score',
            'configuration_definition':'Applied window violates numeric constraints or has fewer than three observed frames in either requested region; no real hardware connected',
            'review_definition':'Score the proposed draft against the tool state at review handoff, then compare with its final score; completing an absent draft is identified separately',
            'metrics':metrics,'rows':rows,'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            'summary_sha256':hashlib.sha256((BASE/'summary.json').read_bytes()).hexdigest()}
    (ROOT/'reports/gpu_agent_failure_audit.json').write_text(json.dumps(record,indent=2)+'\n')
    print(json.dumps(metrics,indent=2))


if __name__=='__main__':main()
