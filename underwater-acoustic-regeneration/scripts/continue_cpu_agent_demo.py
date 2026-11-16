"""One post hoc execution-audit continuation; preserve the original agent draft."""
import hashlib
import json
import subprocess
import time
from pathlib import Path

import numpy as np

from acoustic_regeneration.agents.analysis_tools import AcousticTools
import run_cpu_agent_demo as base

ROOT = base.ROOT
OLD = ROOT / 'results/cpu_agent_demo_run02'
OUT = ROOT / 'results/cpu_agent_demo_audited'


def main():
    protocol = base.read('preregistration/cpu_agent_audit_continuation.json')
    original = base.read('results/cpu_agent_demo_run02/summary.json')
    assert base.sha(OLD / 'summary.json') == protocol['original_summary_sha256']
    task = base.read('results/cpu_agent_demo/task.json')
    with np.load(ROOT / 'results/cpu_agent_demo/observation.npz', allow_pickle=False) as arrays:
        functions = AcousticTools(arrays['audio'], arrays['observed'], int(arrays['sample_rate']))
    events = [json.loads(line) for line in (OLD / 'events.jsonl').read_text().splitlines()]
    messages = task['messages'].copy()
    all_calls = []
    for event in events:
        if event['kind'] == 'model_response':
            messages.append(event['response']['message'])
        elif event['kind'] == 'tool_result':
            calculated = functions.call(event['tool'], event['arguments'])
            assert {'evidence_id': event['evidence_id'], **calculated} == event['result']
            messages.append({'role': 'tool', 'tool_name': event['tool'], 'content': json.dumps(event['result'], ensure_ascii=False)})
            all_calls.append(event)
    # The feedback identifies missing recorded actions; it supplies no new acoustic measurement.
    feedback = protocol['feedback']
    messages.append({'role': 'user', 'content': feedback})
    OUT.mkdir(parents=True, exist_ok=True)
    git_commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
    paths = ['scripts/continue_cpu_agent_demo.py', 'preregistration/cpu_agent_audit_continuation.json',
             'src/acoustic_regeneration/agents/analysis_tools.py', 'scripts/run_cpu_agent_demo.py']
    hashes = {p: base.sha(ROOT / p) for p in paths}
    for path, expected in hashes.items():
        data = subprocess.check_output(['git', 'show', git_commit + ':underwater-acoustic-regeneration/' + path], cwd=ROOT)
        assert hashlib.sha256(data).hexdigest() == expected
    stream = (OUT / 'events.jsonl').open('x')
    start = time.perf_counter()
    def record(kind, **values):
        row = {'kind': kind, 'recorded_utc': base.now(), 'elapsed_seconds': time.perf_counter() - start, **values}
        stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + '\n')
        stream.flush()
        return row
    record('start', git_commit=git_commit, artifact_sha256=hashes,
           original_summary_sha256=protocol['original_summary_sha256'], feedback=feedback)
    turns, added, residency, final = [], [], [], None
    status, failure = 'maximum_turns_reached', None
    try:
        for turn in range(1, protocol['maximum_additional_model_turns'] + 1):
            remaining = protocol['maximum_total_episode_seconds'] - original['elapsed_seconds'] - (time.perf_counter() - start)
            if remaining <= 0:
                status = 'time_limit_reached'
                break
            request = {'model': original['model'], 'messages': messages, 'tools': task['tools'], 'stream': False,
                       'options': task['inference_options'], 'keep_alive': '60s'}
            record('model_request', turn=turn, request=request)
            before = time.perf_counter()
            response = base.api('/api/chat', request, timeout=min(300, remaining))
            turns.append({'turn': turn, 'wall_seconds': time.perf_counter() - before,
                          **{k: response.get(k) for k in ['eval_count', 'prompt_eval_count', 'total_duration', 'load_duration']}})
            record('model_response', turn=turn, response=response)
            active = base.api('/api/ps')
            residency.append(active)
            record('model_residency', turn=turn, response=active)
            model = next(row for row in active['models'] if row['name'] == original['model'])
            assert model['size_vram'] == 0 and model['digest'] == original['model_digest']
            message = response['message']
            messages.append(message)
            calls = message.get('tool_calls', [])
            print('Continuation turn', turn, 'calls', [c['function']['name'] for c in calls], flush=True)
            if not calls:
                final = message.get('content', '')
                status = 'completed'
                break
            for call in calls:
                if len(added) >= protocol['maximum_additional_tool_calls']:
                    raise RuntimeError('Additional tool budget exceeded')
                name, args = call['function']['name'], call['function']['arguments']
                identifier = 'tool_%02d' % (len(all_calls) + 1)
                try:
                    result = functions.call(name, args)
                except (TypeError, ValueError) as error:
                    result = {'error': str(error)}
                result = {'evidence_id': identifier, **result}
                row = record('tool_result', turn=turn, tool=name, arguments=args,
                             evidence_id=identifier, result=result)
                added.append(row)
                all_calls.append(row)
                messages.append({'role': 'tool', 'tool_name': name, 'content': json.dumps(result, ensure_ascii=False)})
    except Exception as error:
        status, failure = 'execution_failed', type(error).__name__ + ': ' + str(error)
        record('failure', error=failure)
    elapsed = time.perf_counter() - start
    record('end', status=status, final_response=final, elapsed_total_seconds=elapsed)
    stream.close()
    measures = {c['arguments'].get('window_ms') for c in all_calls if c['tool'] == 'analyze_spectrum' and 'error' not in c['result']}
    checks = {'both_windows_analyzed': measures == {32, 128},
              'verification_after_original_configuration': any(c['tool'] == 'verify_display' and 'error' not in c['result'] for c in added),
              'target_bin_spacing_met': 1000 / functions.window_ms <= 10,
              'audio_preserved': functions.audio_hash() == functions.initial_sha256}
    summary = {'status': status, 'failure': failure, 'elapsed_seconds': elapsed,
               'total_episode_seconds_excluding_researcher_audit_time': original['elapsed_seconds'] + elapsed,
               'original': 'results/cpu_agent_demo_run02', 'original_summary_sha256': base.sha(OLD / 'summary.json'),
               'git_commit': git_commit, 'model': original['model'], 'model_digest': original['model_digest'],
               'additional_model_turns': len(turns), 'additional_tool_calls': len(added), 'turns': turns,
               'final_response': final, 'final_display_window_ms': functions.window_ms, 'checks': checks,
               'all_residency_samples_zero_vram': bool(residency) and all(all(m['size_vram'] == 0 for m in r['models']
                   if m['name'] == original['model']) for r in residency),
               'artifact_sha256': {**hashes, 'results/cpu_agent_demo_audited/events.jsonl': base.sha(OUT / 'events.jsonl')},
               'scope': protocol['scope'], 'final_response_factual_audit': 'Pending separate audit'}
    base.write(OUT / 'summary.json', summary)
    with (ROOT / 'results/local_run_registry.jsonl').open('a') as registry:
        registry.write(json.dumps({'run_id': 'cpu_agent_demo_audit_continuation', 'status': status,
                       'recorded_utc': base.now(), 'git_commit': git_commit, 'elapsed_seconds': elapsed,
                       'model_turns': len(turns), 'tool_calls': len(added), 'GPU_requested': False,
                       'summary_sha256': base.sha(OUT / 'summary.json'), 'output_path': str(OUT.relative_to(ROOT)),
                       'scope': protocol['scope']}) + '\n')
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
