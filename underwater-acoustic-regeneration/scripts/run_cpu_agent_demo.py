"""Prepare and execute one native tool-calling episode against an existing local LLM.

Preparation does not run model inference. All requests, responses, tool returns,
failures, input provenance, and CPU-residency observations are retained.
"""
import argparse
import gzip
import hashlib
import json
import platform
import subprocess
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from acoustic_regeneration.agents.analysis_tools import AcousticTools, tool_schemas

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'results/cpu_agent_demo'
API = 'http://127.0.0.1:11434'
SYSTEM = ('You are an acoustic analysis assistant using a bounded set of real analysis tools. '
          'Tool results are your only acoustic evidence. Call the tools to complete the task; '
          'do not merely describe hypothetical calls. Inspect the recording, compare the initial '
          'display setting with an allowed alternative, apply a justified setting, and verify it. '
          'A tool request is not a measurement until its result is returned. '
          'Never invent measurements or infer an object class from a spectral peak. '
          'Only a virtual analysis display can be changed; acquisition hardware is not connected. '
          'When finished, give a concise operator report of at most 120 words, citing tool evidence IDs. '
          'Include the measured peak bin, selected window and bin spacing, timing tradeoff, '
          'missing-data limitation, and whether object identity is established. '
          'Give a brief decision justification, not a detailed internal reasoning transcript.')
TASK = ('Review clip_001 for an operator. Assess the observed low-frequency component in the '
        '20–600 Hz band and choose an analysis display setting with frequency-bin spacing '
        'at most 10 Hz. Compare the existing setting with an available alternative, apply '
        'your choice to the virtual display, remeasure it, and report what can and cannot be concluded.')


def now():
    return datetime.now(timezone.utc).isoformat()


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads((ROOT / path).read_text())


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n')


def api(path, payload=None, timeout=300):
    request = urllib.request.Request(API + path, data=None if payload is None else json.dumps(payload).encode(),
                                     headers={'Content-Type': 'application/json'})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.load(response)


def preparation():
    OUT.mkdir(parents=True, exist_ok=True)
    if (OUT / 'events.jsonl').exists():
        raise RuntimeError('An episode already exists; preserve it instead of overwriting')
    protocol = read('preregistration/cpu_agent_demo.json')
    case = next(row for row in map(json.loads, (ROOT / 'results/pilot_pools_clipped_seed17/cases.jsonl').read_text().splitlines())
                if row['case_id'] == protocol['case_id'])
    assert case['source_partition'] == 'V' and case['pilot_role'] == 'dev_test'
    with gzip.open(ROOT / 'manifests/pilot_sources.jsonl.gz', 'rt') as stream:
        source = next(row for row in map(json.loads, stream) if row['original_file_id'] == case['source_id'])
    source_path = ROOT / source['processed_path']
    assert sha(source_path) == source['processed_sha256']
    reference = np.load(source_path, allow_pickle=False)[:32768].astype(np.float64)
    observed = np.ones(len(reference), dtype=bool)
    observed[case['gap_start']:case['gap_end']] = False
    normalized = (reference - case['normalization_mean']) / case['normalization_scale']
    available = np.where(observed, normalized, 0.0)
    np.savez_compressed(OUT / 'observation.npz', audio=available, observed=observed, sample_rate=np.array(8000))
    models = api('/api/tags')['models']
    model = next(row for row in models if row['name'] == protocol['model'])
    assert model['digest'] == protocol['model_digest']
    version = api('/api/version')
    assert version['version'] == protocol['ollama_version']
    info = api('/api/show', {'name': protocol['model']})
    write(OUT / 'model_environment.json', {'recorded_utc': now(), 'server': version, 'model': model,
          'details': info['details'], 'parameters': info['parameters'],
          'chat_template': info['template'], 'model_info': info['model_info'],
          'python': platform.python_version(), 'platform': platform.platform(),
          'dependency_note': 'Uses an already installed local model; no model download, API key, or remote GPU'})
    write(OUT / 'task.json', {'messages': [{'role': 'system', 'content': SYSTEM}, {'role': 'user', 'content': TASK}],
                            'tools': tool_schemas(), 'inference_options': protocol['inference_options']})
    write(OUT / 'input_provenance.json', {'case_id': case['case_id'], 'source_partition': 'V',
          'source_id': case['source_id'], 'source_array_path': source['processed_path'],
          'source_array_sha256': source['processed_sha256'],
          'observation_artifact_sha256': sha(OUT / 'observation.npz'),
          'normalized_observed_audio_sha256': hashlib.sha256(available.astype('<f8').tobytes()).hexdigest(),
          'gap_start': case['gap_start'], 'gap_end': case['gap_end'],
          'normalization_mean': case['normalization_mean'], 'normalization_scale': case['normalization_scale'],
          'model_saw_dataset_label': False, 'model_saw_reference_gap': False,
          'model_saw_reconstruction_or_reference_scores': False,
          'selection': protocol['selection']})
    print('Prepared one observed-only clip, model identity, task, and tool schemas; no inference performed.')


def run():
    protocol = read('preregistration/cpu_agent_demo.json')
    task = read('results/cpu_agent_demo/task.json')
    provenance = read('results/cpu_agent_demo/input_provenance.json')
    assert sha(OUT / 'observation.npz') == provenance['observation_artifact_sha256']
    assert task['inference_options'] == protocol['inference_options']
    assert task['tools'] == tool_schemas()
    with np.load(OUT / 'observation.npz', allow_pickle=False) as arrays:
        functions = AcousticTools(arrays['audio'], arrays['observed'], int(arrays['sample_rate']))
    assert functions.audio_hash() == provenance['normalized_observed_audio_sha256']
    code = ['scripts/run_cpu_agent_demo.py', 'src/acoustic_regeneration/agents/analysis_tools.py',
            'preregistration/cpu_agent_demo.json', 'results/cpu_agent_demo/task.json']
    hashes = {p: sha(ROOT / p) for p in code}
    git_commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
    # Require the executable experiment definition to be committed before outcomes.
    for path, expected in hashes.items():
        committed = subprocess.check_output(['git', 'show', git_commit + ':underwater-acoustic-regeneration/' + path], cwd=ROOT)
        assert hashlib.sha256(committed).hexdigest() == expected, path
    start = time.perf_counter()
    tool_events, turns, residency = [], [], []
    status, final_response, failure = 'maximum_turns_reached', None, None
    event_file = (OUT / 'events.jsonl').open('x')
    def event(kind, **values):
        item = {'recorded_utc': now(), 'elapsed_seconds': time.perf_counter() - start, 'kind': kind, **values}
        event_file.write(json.dumps(item, ensure_ascii=False, allow_nan=False) + '\n')
        event_file.flush()
        return item
    event('start', git_commit=git_commit, artifact_sha256=hashes, model=protocol['model'],
          options=task['inference_options'], scope=protocol['scope'])
    messages = list(task['messages'])
    try:
        for turn in range(1, protocol['maximum_model_turns'] + 1):
            if time.perf_counter() - start >= protocol['maximum_episode_seconds']:
                status = 'time_limit_reached'
                break
            request = {'model': protocol['model'], 'messages': messages, 'tools': task['tools'],
                       'stream': False, 'options': task['inference_options'], 'keep_alive': '60s'}
            event('model_request', turn=turn, request=request)
            before = time.perf_counter()
            response = api('/api/chat', request, timeout=min(300, protocol['maximum_episode_seconds'] - (before - start)))
            turns.append({'turn': turn, 'wall_seconds': time.perf_counter() - before,
                          **{key: response.get(key) for key in ['total_duration', 'load_duration', 'prompt_eval_count',
                              'prompt_eval_duration', 'eval_count', 'eval_duration']}})
            event('model_response', turn=turn, response=response)
            active = api('/api/ps')
            residency.append(active)
            event('model_residency', turn=turn, response=active)
            current = next(row for row in active['models'] if row.get('name', row.get('model')) == protocol['model'])
            assert current['digest'] == protocol['model_digest'] and current['size_vram'] == 0, 'CPU-only model execution required'
            message = response['message']
            messages.append(message)
            calls = message.get('tool_calls', [])
            print('Model turn', turn, 'tool calls:', [call['function']['name'] for call in calls], flush=True)
            if not calls:
                final_response = message.get('content', '')
                status = 'completed'
                break
            for call in calls:
                if len(tool_events) >= protocol['maximum_tool_calls']:
                    raise RuntimeError('Tool-call limit reached')
                function = call['function']
                name, arguments = function['name'], function['arguments']
                identifier = 'tool_%02d' % (len(tool_events) + 1)
                before = time.perf_counter()
                try:
                    result = functions.call(name, arguments)
                except (ValueError, TypeError) as error:
                    result = {'error': str(error)}
                result = {'evidence_id': identifier, **result}
                item = event('tool_result', turn=turn, evidence_id=identifier, tool=name,
                             arguments=arguments, result=result, wall_seconds=time.perf_counter() - before)
                tool_events.append(item)
                messages.append({'role': 'tool', 'tool_name': name, 'content': json.dumps(result, ensure_ascii=False)})
    except Exception as error:
        failure = type(error).__name__ + ': ' + str(error)
        status = 'execution_failed'
        event('failure', error=failure)
    finally:
        elapsed = time.perf_counter() - start
        event('end', status=status, final_response=final_response, elapsed_total_seconds=elapsed)
        event_file.close()
    configured = [r for r in tool_events if r['tool'] == 'configure_display' and 'error' not in r['result']]
    verified = [r for r in tool_events if r['tool'] == 'verify_display' and 'error' not in r['result']]
    measured = {r['arguments'].get('window_ms') for r in tool_events if r['tool'] == 'analyze_spectrum' and 'error' not in r['result']}
    summary = {'status': status, 'failure': failure, 'elapsed_seconds': elapsed, 'git_commit': git_commit,
               'model': protocol['model'], 'model_digest': protocol['model_digest'], 'model_turns': len(turns),
               'tool_calls': len(tool_events), 'tool_errors': sum('error' in r['result'] for r in tool_events),
               'turns': turns, 'final_response': final_response, 'final_display_window_ms': functions.window_ms,
               'all_residency_samples_zero_vram': bool(residency) and all(all(row['size_vram'] == 0 for row in sample['models']
                   if row.get('name', row.get('model')) == protocol['model']) for sample in residency),
               'checks': {'inspected': any(r['tool'] == 'inspect_recording' for r in tool_events),
                          'both_windows_analyzed': measured == {32, 128},
                          'display_changed_with_evidence': any(r['result']['previous_window_ms'] != r['result']['current_window_ms'] for r in configured),
                          'verified_after_change': bool(configured and verified and verified[-1]['elapsed_seconds'] > configured[-1]['elapsed_seconds']),
                          'target_bin_spacing_met': 1000 / functions.window_ms <= protocol['task_target_bin_spacing_hz'],
                          'audio_preserved': functions.audio_hash() == functions.initial_sha256},
               'artifact_sha256': {**hashes, 'results/cpu_agent_demo/events.jsonl': sha(OUT / 'events.jsonl'),
                                  'results/cpu_agent_demo/observation.npz': sha(OUT / 'observation.npz')},
               'scope': protocol['scope'], 'final_response_factual_audit': 'Pending separate audit; no success claim from tool checks alone'}
    write(OUT / 'summary.json', summary)
    write(OUT / 'display_state.json', {'window_ms': functions.window_ms, 'physical_hardware_changed': False})
    with (ROOT / 'results/local_run_registry.jsonl').open('a') as stream:
        stream.write(json.dumps({'run_id': 'cpu_agent_demo_seed17', 'status': status, 'recorded_utc': now(),
                    'git_commit': git_commit, 'elapsed_seconds': elapsed, 'model_turns': len(turns),
                    'tool_calls': len(tool_events), 'GPU_requested': False,
                    'summary_sha256': sha(OUT / 'summary.json'), 'output_path': 'results/cpu_agent_demo',
                    'scope': 'Additional local CPU demonstration after the bounded GPU study; no remote pod access'}) + '\n')
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    if status == 'execution_failed':
        raise SystemExit(1)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--prepare', action='store_true')
    args = parser.parse_args()
    preparation() if args.prepare else run()
