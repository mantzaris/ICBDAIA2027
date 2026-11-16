"""Descriptive review-transition and configuration-error audit; primary scores stay fixed."""
import hashlib
import json
from pathlib import Path

import numpy as np

from acoustic_regeneration.agents.benchmark_tools import BenchmarkTools, qualifies, score_report
from acoustic_regeneration.agents.guarded_tools import GuardedBenchmarkTools
from run_gpu_agent_benchmark import parse_action

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'results/gpu_agent_benchmark'


def audit(base, arms):
    cases=[c for c in json.loads((base/'inputs/manifest.json').read_text())['cases'] if c.get('split','evaluation')=='evaluation']
    rows=[]
    for case in cases:
        cid=case['case_id'];task=json.loads((base/'inputs'/f'{cid}.json').read_text())
        with np.load(base/'inputs'/f'{cid}.npz',allow_pickle=False) as d:audio,mask=d['audio'],d['observed']
        measurements={e['arguments']['window_ms']:e['result'] for e in case['reference_tool_trace'] if e['name']=='analyze_window'}
        for arm in arms:
            summary=json.loads((base/'evaluation'/f'{cid}__{arm}.json').read_text())
            tools=(GuardedBenchmarkTools if arm=='reasoning_guarded' else BenchmarkTools)(audio,mask,task)
            draft=None;before_review=None;review_started=False
            bad_changes=[];new_review_tools=[]
            for event in map(json.loads,(base/'evaluation'/f'{cid}__{arm}.jsonl').read_text().splitlines()):
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
            report=summary['final_report']
            successful=[event for event in tools.events if 'error' not in event['result']]
            configured=[i for i,event in enumerate(successful) if event['name']=='configure_display']
            verified=[i for i,event in enumerate(successful) if event['name']=='verify_display']
            actual_verification=bool(configured and verified and verified[-1]>configured[-1])
            all_ids={event['evidence_id'] for event in tools.events}
            error_ids={event['evidence_id'] for event in tools.events if 'error' in event['result']}
            cited=report['evidence_ids'] if summary['checks']['valid_report'] else []
            cited_errors=sorted(set(cited)&error_ids)
            nonexistent=sorted(set(cited)-all_ids)
            failed_components=[key for key,value in summary['checks'].items() if value is False and key!='strict_success']
            only_error_citation=(bool(cited_errors) and not nonexistent and failed_components==['evidence_supported']
                and summary['checks']['unsupported_claims']==1 and summary['input_unchanged']
                and report['verified'] is (case['expected_report']['decision']=='configure'))
            report_setting_correct=isinstance(report,dict) and report.get('decision')==case['expected_report']['decision'] and report.get('window_ms')==case['expected_report']['window_ms']
            stated_applied_mismatch=isinstance(report,dict) and report.get('decision')=='configure' and report.get('window_ms')!=tools.window_ms
            transition=None
            if before_review is not None:
                initial=before_review['checks']['strict_success']
                transition=('kept_correct' if initial and final else 'regressed' if initial else 'rescued' if final else 'still_incorrect')
            rows.append({'case_id':cid,'arm':arm,'family':case['family'],'final_success':final,
                         'expected_decision':case['expected_report']['decision'],
                         'post_configuration_verification_executed':actual_verification,
                         'final_checks':summary['checks'],
                         'cited_actual_error_ids':cited_errors,'cited_nonexistent_ids':nonexistent,
                         'only_evidence_component_failed_with_actual_error_citation':only_error_citation,
                         'cue_correct':summary['checks']['cue_correct'],
                         'reference_cue_label':case['expected_report']['cue_label'],
                         'reported_cue_label':report.get('cue_label') if summary['checks']['valid_report'] else 'invalid',
                         'report_setting_correct':report_setting_correct,'final_display_window_ms':tools.window_ms,
                         'stated_applied_setting_mismatch':stated_applied_mismatch,
                         'before_review':before_review,'review_transition':transition,
                         'review_tool_calls':new_review_tools,'constraint_violating_configurations':bad_changes})
    metrics={}
    for arm in arms:
        subset=[r for r in rows if r['arm']==arm]
        labels=['stable','changed','insufficient']
        confusion={expected:{predicted:0 for predicted in labels+['invalid_or_other']} for expected in labels}
        for row in subset:
            prediction=row['reported_cue_label']
            if prediction not in labels:prediction='invalid_or_other'
            confusion[row['reference_cue_label']][prediction]+=1
        assert sum(confusion[label][label] for label in labels)==sum(row['cue_correct'] for row in subset)
        metrics[arm]={'cases':len(subset),
            'feasible_cases':sum(row['expected_decision']=='configure' for row in subset),
            'feasible_cases_with_executed_post_configuration_verification':sum(row['expected_decision']=='configure' and row['post_configuration_verification_executed'] for row in subset),
            'reports_citing_actual_errors':sum(bool(row['cited_actual_error_ids']) for row in subset),
            'reports_citing_nonexistent_ids':sum(bool(row['cited_nonexistent_ids']) for row in subset),
            'only_evidence_component_failed_with_actual_error_citation':sum(row['only_evidence_component_failed_with_actual_error_citation'] for row in subset),
            'final_component_pass_counts':{key:sum(row['final_checks'][key] for row in subset)
                for key,value in subset[0]['final_checks'].items() if isinstance(value,bool)},
            'cue_confusion_counts_reference_then_reported':confusion,
            'report_setting_correct':sum(r['report_setting_correct'] for r in subset),
            'stated_applied_setting_mismatch':sum(r['stated_applied_setting_mismatch'] for r in subset),
            'cases_with_constraint_violating_configuration':sum(bool(r['constraint_violating_configurations']) for r in subset),
            'total_constraint_violating_configurations':sum(len(r['constraint_violating_configurations']) for r in subset),
            'review_transitions':{name:sum(r['review_transition']==name for r in subset) for name in ['kept_correct','regressed','rescued','still_incorrect']},
            'initial_draft_missing':sum(r['before_review'] is not None and not r['before_review']['draft_present'] for r in subset),
            'rescued_existing_draft':sum(r['review_transition']=='rescued' and r['before_review']['draft_present'] for r in subset),
            'completed_missing_draft':sum(r['review_transition']=='rescued' and not r['before_review']['draft_present'] for r in subset)}
    return {'metrics':metrics,'rows':rows}


def main():
    primary=audit(BASE,['direct','reasoning','self_review','two_agents'])
    guard=audit(ROOT/'results/gpu_agent_guard',['reasoning','reasoning_guarded'])
    record={'scope':'Descriptive secondary audit specified during execution; not an additional confirmatory hypothesis or a changed primary score',
            'configuration_definition':'Applied window violates numeric constraints or has fewer than three observed frames in either requested region; no real hardware connected',
            'review_definition':'Score the proposed draft against the tool state at review handoff, then compare with its final score; completing an absent draft is identified separately',
            'setting_mismatch_definition':'A configure report states a window different from the final executed virtual-display state; this audit was motivated by the first primary-case trace and does not change frozen primary scores',
            'verification_definition':'Actual post-configuration verification is read directly from successful executed tool returns, independently of final-report validity; feasible-task counts exclude refusal cases.',
            'citation_definition':'The frozen score accepts only successful tool returns. Citing a real error is distinguished from inventing an ID. A separate flag identifies otherwise passing reports that fail only the evidence component while citing actual errors; no scores are relaxed or replaced.',
            'citation_audit_timing':'Added after inspection of completed gate cases 03 and 04; descriptive diagnosis, not a prespecified endpoint or revised score.',
            'component_count_note':'Component pass counts are nonexclusive; an invalid report fails every component under the frozen score.',
            **primary,'guard_followup':guard,'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            'source_summary_sha256':{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in
                [BASE/'summary.json',ROOT/'results/gpu_agent_guard/summary.json']}}
    (ROOT/'reports/gpu_agent_failure_audit.json').write_text(json.dumps(record,indent=2)+'\n')
    print(json.dumps({'primary':primary['metrics'],'guard_followup':guard['metrics']},indent=2))


if __name__=='__main__':main()
