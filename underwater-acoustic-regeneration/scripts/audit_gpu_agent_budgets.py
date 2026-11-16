"""Audit recorded resource limits and truncation without running a language model."""
import hashlib
import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def audit(directory, expected_episodes, overrides=None):
    protocol = json.loads((ROOT / 'preregistration/gpu_agent_benchmark.json').read_text())
    limits = dict(protocol['budgets'])
    limits.update(overrides or {})
    completion = json.loads((directory / 'completion.json').read_text())
    assert completion['episodes'] == expected_episodes
    paths = sorted(directory.glob('*__*.json'))
    assert len(paths) == expected_episodes
    arms = {}
    hashes = {}
    for path in paths:
        summary = json.loads(path.read_text())
        event_path = path.with_suffix('.jsonl')
        assert sha(event_path) == summary['events_sha256']
        events = [json.loads(line) for line in event_path.read_text().splitlines()]
        responses = [e for e in events if e['kind'] == 'model_response']
        returned = [e for e in events if e['kind'] == 'tool_result']
        assert len(responses) == summary['model_turns'] <= limits['max_model_turns']
        assert len(returned) == summary['tool_calls'] <= limits['max_tool_calls']
        assert sum(e['generated_tokens'] for e in responses) == summary['generated_tokens'] <= limits['max_generated_tokens']
        assert sum(e['forced_reasoning_close'] for e in responses) == summary['forced_reasoning_closures']
        for event in responses:
            assert event['model_device'] == 'cuda:0'
            assert event['reasoning_tokens'] <= limits['max_thinking_tokens_per_turn']
            assert event['action_tokens'] <= limits['max_action_tokens_per_turn']
            assert event['generated_tokens'] == event['reasoning_tokens'] + event['action_tokens']
            assert event['prompt_tokens'] <= limits['max_context_tokens']
            if summary['arm'] == 'direct':
                assert event['reasoning_tokens'] == 0 and not event['forced_reasoning_close']
        item = arms.setdefault(summary['arm'], {
            'episodes': 0, 'statuses': Counter(), 'errors': Counter(), 'model_turns': 0,
            'thinking_turns': 0, 'forced_thinking_closures': 0, 'action_cap_hits': 0,
            'forced_closure_token_counts': Counter(),
            'episodes_with_forced_thinking_closure': 0, 'episodes_with_action_cap_hit': 0,
            'maximum_context_tokens': 0, 'maximum_generated_tokens': 0,
            'maximum_episode_seconds': 0., 'total_generated_tokens': 0,
        })
        item['episodes'] += 1
        item['statuses'][summary['status']] += 1
        if summary['error']:
            item['errors'][summary['error']] += 1
        item['model_turns'] += len(responses)
        item['thinking_turns'] += sum(e['reasoning_tokens'] > 0 for e in responses)
        item['forced_thinking_closures'] += summary['forced_reasoning_closures']
        item['forced_closure_token_counts'].update(e['reasoning_tokens'] for e in responses if e['forced_reasoning_close'])
        item['action_cap_hits'] += sum(e['action_hit_token_cap'] for e in responses)
        item['episodes_with_forced_thinking_closure'] += int(summary['forced_reasoning_closures'] > 0)
        item['episodes_with_action_cap_hit'] += int(any(e['action_hit_token_cap'] for e in responses))
        item['maximum_context_tokens'] = max(item['maximum_context_tokens'], max((e['prompt_tokens'] for e in responses), default=0))
        item['maximum_generated_tokens'] = max(item['maximum_generated_tokens'], summary['generated_tokens'])
        item['maximum_episode_seconds'] = max(item['maximum_episode_seconds'], summary['elapsed_seconds'])
        item['total_generated_tokens'] += summary['generated_tokens']
        hashes[str(path.relative_to(ROOT))] = sha(path)
        hashes[str(event_path.relative_to(ROOT))] = sha(event_path)
    return {'episodes': len(paths), 'limits': limits, 'arms': arms, 'artifact_sha256': hashes}


def main():
    primary = audit(ROOT / 'results/gpu_agent_benchmark/evaluation', 160)
    followup = audit(ROOT / 'results/gpu_agent_guard/evaluation', 32)
    report = {
        'status': 'passed', 'primary': primary, 'guard_followup': followup,
        'scope': 'Descriptive resource and truncation audit; frozen accuracy scores unchanged',
        'time_limit_note': 'The episode deadline is checked between model calls; the external job deadline is enforced separately.',
        'token_note': 'Counts are generated tokens, including reasoning; externally inserted closing markers are not model-generated tokens.',
        'analysis_sha256': sha(Path(__file__)),
    }
    budget_directory = ROOT / 'results/gpu_agent_budget/evaluation'
    if (budget_directory / 'completion.json').exists():
        budget_protocol = json.loads((ROOT / 'preregistration/gpu_agent_budget_followup.json').read_text())
        overrides = {key: budget_protocol['intervention'][key]
                     for key in ('max_thinking_tokens_per_turn', 'max_generated_tokens')}
        report['larger_budget_followup'] = audit(budget_directory, 16, overrides)
        report['larger_budget_scope'] = 'Sixteen new episodes only; their original controls are already counted in the primary audit.'
    (ROOT / 'reports/gpu_agent_budget_audit.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({name: value['arms'] for name, value in report.items()
                      if isinstance(value, dict) and 'arms' in value}, indent=2))


if __name__ == '__main__':
    main()
