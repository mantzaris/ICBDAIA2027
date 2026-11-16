"""Replay recorded acoustic tool calls and plot the first completed local LLM episode."""
import copy
import hashlib
import json
import shutil
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import numpy as np

from acoustic_regeneration.agents.analysis_tools import AcousticTools

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / 'results/cpu_agent_demo_run02'
CONTINUATION = ROOT / 'results/cpu_agent_demo_audited'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def main():
    summary = read(RUN / 'summary.json')
    continued = read(CONTINUATION / 'summary.json')
    assert summary['status'] == 'completed', 'Retain incomplete evidence; do not manufacture a completed episode'
    assert continued['status'] == 'completed' and all(continued['checks'].values()), continued
    assert summary['all_residency_samples_zero_vram'] and continued['all_residency_samples_zero_vram']
    for record in (summary, continued):
        for path, expected in record['artifact_sha256'].items():
            assert sha(ROOT / path) == expected, path
    phases = [[json.loads(line) for line in (directory / 'events.jsonl').read_text().splitlines()]
              for directory in (RUN, CONTINUATION)]
    events = [row for phase in phases for row in phase]
    requests = [e for e in events if e['kind'] == 'model_request']
    responses = [e for e in events if e['kind'] == 'model_response']
    calls = [e for e in events if e['kind'] == 'tool_result']
    assert len(requests) == len(responses) == summary['model_turns'] + continued['additional_model_turns']
    assert len(calls) == summary['tool_calls'] + continued['additional_tool_calls']
    inputs = read(ROOT / 'results/cpu_agent_demo/task.json')
    messages = inputs['messages'].copy()
    with np.load(ROOT / 'results/cpu_agent_demo/observation.npz', allow_pickle=False) as arrays:
        functions = AcousticTools(arrays['audio'], arrays['observed'], int(arrays['sample_rate']))
    observations = {}
    for index, phase in enumerate(phases):
        if index:
            messages.append({'role': 'user', 'content': read(ROOT / 'preregistration/cpu_agent_audit_continuation.json')['feedback']})
        phase_requests = [e for e in phase if e['kind'] == 'model_request']
        phase_responses = [e for e in phase if e['kind'] == 'model_response']
        for request, response in zip(phase_requests, phase_responses):
            assert request['request']['messages'] == messages
            assert request['request']['options']['num_gpu'] == 0
            assert request['request']['tools'] == inputs['tools']
            message = response['response']['message']
            messages.append(message)
            executed = [e for e in phase if e['kind'] == 'tool_result' and e['turn'] == response['turn']]
            selected = message.get('tool_calls', [])
            assert len(executed) == len(selected)
            for selected_call, actual in zip(selected, executed):
                name = selected_call['function']['name']
                arguments = selected_call['function']['arguments']
                assert name == actual['tool'] and arguments == actual['arguments']
                try:
                    calculated = functions.call(name, arguments)
                except (ValueError, TypeError) as error:
                    calculated = {'error': str(error)}
                result = {'evidence_id': actual['evidence_id'], **calculated}
                assert result == actual['result'], actual['evidence_id']
                observations[actual['evidence_id']] = result
                messages.append({'role': 'tool', 'tool_name': name, 'content': json.dumps(result, ensure_ascii=False)})
    assert continued['final_response'] == responses[-1]['response']['message']['content']
    assert functions.window_ms == continued['final_display_window_ms']
    assert not summary['checks']['both_windows_analyzed'] and not summary['checks']['verified_after_change']
    for row in [e for e in events if e['kind'] == 'model_residency']:
        model = next(m for m in row['response']['models'] if m['name'] == summary['model'])
        assert model['size_vram'] == 0 and model['digest'] == summary['model_digest']
    factual = read(ROOT / 'reports/cpu_agent_response_audit.json')
    assert factual['original_summary_sha256'] == sha(RUN / 'summary.json')
    assert factual['continued_summary_sha256'] == sha(CONTINUATION / 'summary.json')
    assert factual['final_response_sha256'] == hashlib.sha256(continued['final_response'].encode()).hexdigest()
    verified = next(c for c in reversed(calls) if c['tool'] == 'verify_display' and 'error' not in c['result'])
    measurement = verified['result']['measurement']
    coarse = next(c for c in calls if c['tool'] == 'analyze_spectrum' and c['arguments']['window_ms'] == 32)
    card = {'kind': 'Tool-derived operator evidence card; not model-authored prose',
            'selected_window_ms': measurement['window_ms'], 'bin_spacing_hz': measurement['bin_spacing_hz'],
            'observed_peak_bin_hz': measurement['observed_peak_bin_hz'],
            'comparison_window_ms': coarse['result']['window_ms'],
            'comparison_bin_spacing_hz': coarse['result']['bin_spacing_hz'],
            'time_support_increase_factor': measurement['time_support_ms'] / coarse['result']['time_support_ms'],
            'identity_status': 'unconfirmed', 'identity_evidence_available': measurement['source_identity_evidence_available'],
            'gap_policy': 'Missing samples remain marked; every frame touching them is excluded',
            'audio_unchanged': verified['result']['audio_unchanged'],
            'physical_hardware_changed': verified['result']['physical_hardware_changed'],
            'evidence_ids': [coarse['evidence_id'], verified['evidence_id']],
            'free_text_model_report_verified': False,
            'scope': 'Observed feature and analysis settings only; no identification or continuity claim inside the gap'}
    card_path = ROOT / 'reports/cpu_agent_operator_card.json'
    card_path.write_text(json.dumps(card, indent=2) + '\n')

    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 7, 'axes.titlesize': 7,
                         'axes.labelsize': 7, 'xtick.labelsize': 6, 'ytick.labelsize': 6,
                         'pdf.fonttype': 42, 'svg.hashsalt': 'cpu-agent-demonstration-v1'})
    fig = plt.figure(figsize=(4.7, 3.0))
    flow = fig.add_axes([.02, .76, .96, .21])
    flow.set(xlim=(0, 1), ylim=(0, 1))
    flow.axis('off')
    cards = [('Initial agent draft', '128 ms applied;\ncomparison claimed'),
             ('Execution audit', '32 ms comparison missing;\nverification missing'),
             ('Agent continuation', 'Measures 32 ms;\nverifies selected display')]
    for i, (title, detail) in enumerate(cards):
        x = .006 + i * .337
        face, edge = [('#EAF4F8', '#277C94'), ('#FFF2DB', '#BA7922'), ('#EAF5EE', '#33825C')][i]
        flow.add_patch(FancyBboxPatch((x, .05), .31, .75, boxstyle='round,pad=0.005,rounding_size=0.04',
                                     edgecolor=edge, facecolor=face, linewidth=.8))
        flow.text(x + .155, .61, title, ha='center', va='center', fontsize=6.5, weight='bold', color='#23333D')
        flow.text(x + .155, .28, detail, ha='center', va='center', fontsize=6.0, color='#23333D')
        if i < 2:
            flow.add_patch(FancyArrowPatch((x + .312, .45), (x + .336, .45), arrowstyle='-|>', mutation_scale=6,
                                          linewidth=.7, color='#64747C'))
    fig.text(.5, .985, 'CPU agent: check actions before trusting the report', ha='center', va='top', fontsize=7.2, weight='bold')
    axes = [fig.add_axes([.10, .29, .39, .39]), fig.add_axes([.59, .29, .39, .39])]
    data = [functions.spectrum_arrays(w) for w in (32, 128)]
    peak = max(np.nanmax(row[2]) for row in data)
    cmap = copy.copy(plt.get_cmap('magma'))
    cmap.set_bad('#DDDDDD')
    for axis, window, (times, frequency, power, _) in zip(axes, (32, 128), data):
        values = 10 * np.log10(np.maximum(power / peak, 1e-12))
        axis.pcolormesh(times, frequency, np.ma.masked_invalid(values), vmin=-60, vmax=0,
                        cmap=cmap, shading='nearest', rasterized=True)
        axis.set(xlim=(0, 4.096), ylim=(0, 600), yticks=[0, 200, 400, 600], xlabel='Time (s)',
                 title='%d ms: %.2f Hz bins' % (window, 1000 / window))
        if window == 32:
            axis.set_ylabel('Frequency (Hz)')
        for a, b in functions.inspect_recording()['missing_intervals_seconds']:
            axis.text((a + b) / 2, 500, 'Gap', ha='center', fontsize=6, color='#444444')
    measured = functions.analyze_spectrum(functions.window_ms)['observed_peak_bin_hz']
    axes[1].annotate('%.2f Hz' % measured, xy=(3.0, measured), xytext=(2.7, 400), ha='center', fontsize=6,
                     color='white', arrowprops={'arrowstyle': '->', 'color': 'white', 'lw': .6})
    fig.text(.5, .14, 'Tool evidence: %.3f Hz bin | Source identity: unconfirmed' % measured, ha='center', fontsize=6.5)
    fig.text(.5, .075, 'Action checks completed; model prose still needs review. Hardware unchanged.', ha='center', fontsize=6.0)
    fig.text(.5, .015, 'Llama 3.1 8B · %d turns · %d tool calls · %.1f min CPU | One guided case' %
             (len(responses), len(calls), continued['total_episode_seconds_excluding_researcher_audit_time'] / 60), ha='center', fontsize=5.8)
    for extension in ('pdf', 'png', 'svg'):
        path = ROOT / ('figures/agent_episode.' + extension)
        metadata = {'CreationDate': None, 'ModDate': None} if extension == 'pdf' else ({'Date': None} if extension == 'svg' else {})
        fig.savefig(path, dpi=450, bbox_inches='tight', pad_inches=.04, metadata=metadata)
        if extension == 'svg':
            path.write_text('\n'.join(line.rstrip() for line in path.read_text().splitlines()) + '\n')
        if extension == 'pdf':
            shutil.copyfile(path, ROOT / 'manuscript' / path.name)
    plt.close(fig)
    paths = ['results/cpu_agent_demo_run02/summary.json', 'results/cpu_agent_demo_run02/events.jsonl',
             'results/cpu_agent_demo_audited/summary.json', 'results/cpu_agent_demo_audited/events.jsonl',
             'preregistration/cpu_agent_audit_continuation.json',
             'results/cpu_agent_demo/task.json', 'results/cpu_agent_demo/observation.npz',
             'reports/cpu_agent_response_audit.json', 'src/acoustic_regeneration/agents/analysis_tools.py',
             'reports/cpu_agent_operator_card.json',
             'scripts/build_agent_figure.py', 'figures/agent_episode.pdf']
    report = {'status': 'passed', 'tool_returns_replayed_exactly': len(calls),
              'model_context_and_tool_results_chain_verified': True, 'CPU_residency_verified': True,
              'model_turns': len(responses), 'observed_peak_bin_hz': measured,
              'final_display_window_ms': functions.window_ms, 'initial_checks': summary['checks'],
              'continued_checks': continued['checks'], 'unsupported_original_comparison_claim_retained': True,
              'artifact_sha256': {path: sha(ROOT / path) for path in paths},
              'scope': 'Integrity and factual review of a single guided episode; no identification accuracy or general reasoning benchmark'}
    (ROOT / 'reports/cpu_agent_validation.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
