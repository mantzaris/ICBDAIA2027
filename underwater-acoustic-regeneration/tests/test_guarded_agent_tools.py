import numpy as np
from acoustic_regeneration.agents.guarded_tools import GuardedBenchmarkTools


def make_tools():
    audio=np.sin(2*np.pi*125*np.arange(32768)/8000)
    mask=np.ones(32768,dtype=bool);mask[14000:18000]=False
    return GuardedBenchmarkTools(audio,mask,{'constraints':{'max_bin_spacing_hz':8,'max_time_support_ms':160}})


def test_gate_rejects_measured_but_infeasible_change_without_moving_display():
    tools=make_tools();tools.call('analyze_window',{'window_ms':64})
    result=tools.call('configure_display',{'window_ms':64})
    assert 'error' in result['result'] and tools.window_ms==32
    assert tools.input_hash()==tools.initial_hash


def test_gate_permits_feasible_change_but_does_not_verify_or_label_for_agent():
    tools=make_tools();tools.call('analyze_window',{'window_ms':128})
    result=tools.call('configure_display',{'window_ms':128})
    assert result['result']['current_window_ms']==128
    assert not any(e['name']=='verify_display' for e in tools.events)
    assert 'cue_label' not in str(result) and 'expected' not in str(result)
