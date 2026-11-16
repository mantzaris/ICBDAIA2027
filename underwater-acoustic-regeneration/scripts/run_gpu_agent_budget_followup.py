"""One larger reasoning budget on the first sixteen frozen primary tasks."""
import copy
import json
from datetime import datetime, timezone
from pathlib import Path

import run_gpu_agent_benchmark as runner

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'results/gpu_agent_budget'


def main():
    assert (ROOT / 'results/gpu_agent_guard/evaluation/completion.json').exists()
    primary = json.loads((ROOT / 'preregistration/gpu_agent_benchmark.json').read_text())
    follow = json.loads((ROOT / 'preregistration/gpu_agent_budget_followup.json').read_text())
    auth = json.loads((ROOT / 'configs/gpu_agent_authorization.json').read_text())
    deadline = datetime.fromisoformat(auth['stop_jobs_by_utc'])
    assert (deadline-datetime.now(timezone.utc)).total_seconds() >= 10800
    freeze = json.loads((ROOT / 'preregistration/gpu_agent_budget_freeze.json').read_text())
    for name, expected in freeze['artifact_sha256'].items():
        assert runner.sha(ROOT/name) == expected, name
    cases = [c for c in json.loads((runner.INPUT/'manifest.json').read_text())['cases']
             if c['split'] == 'evaluation'][:follow['cases']]
    assert len(cases) == 16
    protocol = copy.deepcopy(primary)
    for name in ('max_thinking_tokens_per_turn', 'max_generated_tokens'):
        protocol['budgets'][name] = follow['intervention'][name]
    model = runner.GPUModel(protocol)
    model.torch.cuda.reset_peak_memory_stats()
    out = BASE / 'evaluation'
    out.mkdir(parents=True, exist_ok=True)
    assert not (out/'run_definition.json').exists(), 'Retain partial outputs; do not silently rerun'
    runner.write(out/'run_definition.json', {
        'started_utc': runner.now(), 'artifact_sha256': freeze['artifact_sha256'],
        'model_environment_sha256': runner.sha(ROOT/'provenance/gpu_agent_environment.json'),
        'effective_budgets': protocol['budgets'], 'cases': [c['case_id'] for c in cases]})
    for i, case in enumerate(cases, 1):
        if datetime.now(timezone.utc) >= deadline:
            raise RuntimeError('Authorized deadline reached')
        for suffix, key in [('.npz', 'input_sha256'), ('.json', 'task_sha256')]:
            assert runner.sha(runner.INPUT/(case['case_id']+suffix)) == case[key]
        runner.run_episode(model, case, 'reasoning_long', protocol, out)
        runner.write(out/'progress.json', {
            'updated_utc': runner.now(), 'completed_episodes': i, 'planned_episodes': 16,
            'last_case': case['case_id'], 'last_arm': 'reasoning_long'})
    runner.write(out/'completion.json', {
        'completed_utc': runner.now(), 'cases': 16, 'episodes': 16, 'model_device': 'cuda:0',
        'peak_allocated_bytes': model.torch.cuda.max_memory_allocated(),
        'artifact_sha256': freeze['artifact_sha256']})
    assert all(runner.sha(ROOT/p) == h for p, h in freeze['artifact_sha256'].items())


if __name__ == '__main__':
    main()
