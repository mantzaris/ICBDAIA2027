"""Replay the predeclared larger-budget condition against its original paired outputs."""
import hashlib
import json
from pathlib import Path

import numpy as np

from acoustic_regeneration.agents.benchmark_tools import BenchmarkTools, score_report, solve_rules
from analyze_gpu_agent_benchmark import assert_close, compare, wilson

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'results/gpu_agent_budget'
PRIMARY = ROOT / 'results/gpu_agent_benchmark'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    completion = json.loads((BASE/'evaluation/completion.json').read_text())
    assert completion['cases'] == completion['episodes'] == 16
    for name, expected in completion['artifact_sha256'].items():
        assert sha(ROOT/name) == expected, name
    cases = [c for c in json.loads((PRIMARY/'inputs/manifest.json').read_text())['cases']
             if c['split'] == 'evaluation'][:16]
    groups = {'reasoning': [], 'reasoning_long': []}
    artifacts = {}
    replayed = 0
    cuda_turns = 0
    for case in cases:
        cid = case['case_id']
        task = json.loads((PRIMARY/'inputs'/f'{cid}.json').read_text())
        with np.load(PRIMARY/'inputs'/f'{cid}.npz', allow_pickle=False) as arrays:
            audio, mask = arrays['audio'], arrays['observed']
        expected = solve_rules(BenchmarkTools(audio, mask, task))
        assert_close(expected, case['expected_report'])
        for arm in groups:
            directory = PRIMARY if arm == 'reasoning' else BASE
            path = directory/'evaluation'/f'{cid}__{arm}.json'
            row = json.loads(path.read_text())
            event_path = path.with_suffix('.jsonl')
            assert sha(event_path) == row['events_sha256']
            tools = BenchmarkTools(audio, mask, task)
            actual_tools = []
            tokens = 0
            turns = 0
            thinking_cap = 512 if arm == 'reasoning' else 2048
            total_cap = 6144 if arm == 'reasoning' else 12288
            for event in map(json.loads, event_path.read_text().splitlines()):
                if event['kind'] == 'tool_result':
                    result = tools.call(event['name'], event['arguments'])
                    saved = {k: event[k] for k in ('evidence_id', 'name', 'arguments', 'result')}
                    assert_close(result, saved)
                    actual_tools.append(saved)
                    replayed += 1
                elif event['kind'] == 'model_response':
                    assert event['model_device'] == 'cuda:0'
                    assert event['reasoning_tokens'] <= thinking_cap
                    assert event['action_tokens'] <= 512 and event['prompt_tokens'] <= 16384
                    tokens += event['generated_tokens']
                    turns += 1
                    cuda_turns += 1
                    prompt = json.dumps(event['request_messages'])
                    assert case['source_id'] not in prompt and case['group_id'] not in prompt
                    assert 'expected_report' not in prompt and 'reference_tool_trace' not in prompt
                    for message in event['request_messages']:
                        if message['role'] == 'tool':
                            assert json.loads(message['content']) in actual_tools
            assert_close(actual_tools, row['tool_trace'])
            assert tokens == row['generated_tokens'] <= total_cap
            assert turns == row['model_turns'] <= 8 and len(actual_tools) <= 16
            assert score_report(row['final_report'], tools, expected) == row['checks']
            assert tools.input_hash() == tools.initial_hash
            report = row['final_report']
            mismatch = isinstance(report, dict) and report.get('decision') == 'configure' and report.get('window_ms') != tools.window_ms
            groups[arm].append({**row, 'stated_applied_setting_mismatch': mismatch})
            for p in (path, event_path):
                artifacts[str(p.relative_to(ROOT))] = sha(p)
    metrics = {}
    for arm, rows in groups.items():
        successes = sum(r['checks']['strict_success'] for r in rows)
        turns = sum(r['model_turns'] for r in rows)
        closures = sum(r['forced_reasoning_closures'] for r in rows)
        metrics[arm] = {
            'n': 16, 'successes': successes, 'success_rate': successes/16,
            'wilson_95': wilson(successes, 16),
            'setting_correct': sum(r['checks']['setting_correct'] for r in rows),
            'cue_correct': sum(r['checks']['cue_correct'] for r in rows),
            'stated_applied_setting_mismatch': sum(r['stated_applied_setting_mismatch'] for r in rows),
            'unsupported_report_cases': sum((r['checks']['unsupported_claims'] or 0) > 0 for r in rows),
            'model_turns': turns, 'forced_thinking_closures': closures,
            'forced_closure_fraction': closures/turns,
            'mean_generated_tokens': float(np.mean([r['generated_tokens'] for r in rows])),
            'median_elapsed_seconds': float(np.median([r['elapsed_seconds'] for r in rows])),
            'total_inference_seconds': sum(r['inference_seconds'] for r in rows),
            'failed_or_incomplete': sum(r['status'] != 'completed' for r in rows),
        }
    paired = compare([r['checks']['strict_success'] for r in groups['reasoning_long']],
                     [r['checks']['strict_success'] for r in groups['reasoning']], np.random.default_rng(20261118))
    output = {
        'status': 'completed', 'cases': 16, 'new_neural_episodes': 16,
        'reused_primary_control_episodes': 16, 'case_ids': [c['case_id'] for c in cases],
        'metrics': metrics, 'paired_long_minus_original': paired, 'completion': completion,
        'scope': 'Exploratory budget sensitivity on the first sixteen frozen primary tasks; not new independent recordings. Both per-turn and total reasoning-token caps change. All other model/task settings and scoring remain fixed.',
    }
    (BASE/'summary.json').write_text(json.dumps(output, indent=2)+'\n')
    validation = {
        'status': 'passed', 'cases': 16, 'new_episodes': 16, 'paired_primary_controls': 16,
        'tool_returns_replayed': replayed, 'CUDA_turns_verified': cuda_turns,
        'all_scores_recomputed': True, 'artifact_sha256': artifacts,
        'analysis_sha256': sha(Path(__file__)), 'summary_sha256': sha(BASE/'summary.json'),
    }
    (ROOT/'reports/gpu_agent_budget_validation.json').write_text(json.dumps(validation, indent=2)+'\n')
    print(json.dumps({'metrics': metrics, 'paired': paired}, indent=2))


if __name__ == '__main__':
    main()
