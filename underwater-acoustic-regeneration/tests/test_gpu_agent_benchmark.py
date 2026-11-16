import copy
import numpy as np

from acoustic_regeneration.agents.benchmark_tools import BenchmarkTools, solve_rules, score_report


def inputs(spacing=8,support=160):
    t=np.arange(32768)/8000
    audio=np.sin(2*np.pi*125*t)+0.01*np.cos(2*np.pi*331*t)
    mask=np.ones(32768,dtype=bool); mask[14000:18000]=False
    task={'constraints':{'max_bin_spacing_hz':spacing,'max_time_support_ms':support}}
    return audio,mask,task


def test_hidden_audio_cannot_change_measurements():
    audio,mask,task=inputs(); changed=audio.copy(); changed[~mask]=np.nan
    a,b=BenchmarkTools(audio,mask,task),BenchmarkTools(changed,mask,task)
    assert a.input_hash()==b.input_hash()
    for window in (32,64,128,256): assert a.analyze_window(window)==b.analyze_window(window)


def test_known_tone_shortest_feasible_setting_and_verified_report():
    tools=BenchmarkTools(*inputs()); report=solve_rules(tools)
    assert report['window_ms']==128 and report['cue_label']=='stable'
    assert report['peak_before_hz']==report['peak_after_hz']==125
    assert score_report(report,tools,report)['strict_success']


def test_conflicting_constraints_decline_without_configuration():
    tools=BenchmarkTools(*inputs(6,128)); report=solve_rules(tools)
    assert report['decision']=='decline' and report['window_ms'] is None
    assert not any(e['name']=='configure_display' for e in tools.events)
    assert score_report(report,tools,report)['strict_success']


def test_fragmentation_prevents_long_window_despite_enough_total_samples():
    audio,mask,task=inputs(16,128)
    for start in range(0,len(mask),512): mask[start:start+128]=False
    tools=BenchmarkTools(audio,mask,task); report=solve_rules(tools)
    assert mask.mean()>.5 and report['decision']=='decline'
    assert tools.analyze_window(64)['regions']['before']['fully_observed_frames']==0


def test_claimed_verification_and_invented_citations_fail():
    tools=BenchmarkTools(*inputs()); report=solve_rules(tools); expected=copy.deepcopy(report)
    tools.events=[e for e in tools.events if e['name']!='verify_display']
    score=score_report(report,tools,expected)
    assert not score['strict_success'] and score['unsupported_claims']>=1
    report['evidence_ids']=['invented']; assert not score_report(report,tools,expected)['evidence_supported']


def test_tool_boundary_and_missing_report():
    tools=BenchmarkTools(*inputs())
    assert 'error' in tools.call('configure_display',{'window_ms':128})['result']
    assert 'error' in tools.call('shell',{'command':'anything'})['result']
    assert 'error' in tools.call('analyze_window',{'window_ms':True})['result']
    assert score_report(None,tools,{})['strict_success'] is False
