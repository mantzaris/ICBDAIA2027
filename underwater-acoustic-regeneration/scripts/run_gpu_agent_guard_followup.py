"""Paired GPU follow-up; imports the unchanged primary runner and switches only its tool class."""
import json
from datetime import datetime,timezone
from pathlib import Path

import run_gpu_agent_benchmark as runner
from acoustic_regeneration.agents.benchmark_tools import BenchmarkTools
from acoustic_regeneration.agents.guarded_tools import GuardedBenchmarkTools

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'results/gpu_agent_guard'


def main():
    assert (ROOT/'results/gpu_agent_benchmark/evaluation/completion.json').exists()
    primary=json.loads((ROOT/'preregistration/gpu_agent_benchmark.json').read_text())
    follow=json.loads((ROOT/'preregistration/gpu_agent_guard_followup.json').read_text())
    auth=json.loads((ROOT/'configs/gpu_agent_authorization.json').read_text())
    deadline=datetime.fromisoformat(auth['stop_jobs_by_utc'])
    assert (deadline-datetime.now(timezone.utc)).total_seconds()>=7200,'Insufficient authorized time for follow-up'
    freeze=json.loads((ROOT/'preregistration/gpu_agent_guard_freeze.json').read_text())
    for name,expected in freeze['artifact_sha256'].items():assert runner.sha(ROOT/name)==expected,name
    manifest=json.loads((BASE/'inputs/manifest.json').read_text())
    assert manifest['protocol_sha256']==runner.sha(ROOT/'preregistration/gpu_agent_guard_followup.json')
    runner.INPUT=BASE/'inputs'
    for case in manifest['cases']:
        assert runner.sha(runner.INPUT/(case['case_id']+'.npz'))==case['input_sha256']
        assert runner.sha(runner.INPUT/(case['case_id']+'.json'))==case['task_sha256']
    model=runner.GPUModel(primary);model.torch.cuda.reset_peak_memory_stats()
    out=BASE/'evaluation';out.mkdir(parents=True,exist_ok=True)
    definition={'started_utc':runner.now(),'artifact_sha256':freeze['artifact_sha256'],
                'model_environment_sha256':runner.sha(ROOT/'provenance/gpu_agent_environment.json')}
    assert not (out/'run_definition.json').exists(),'Follow-up already started; retain partial outputs'
    runner.write(out/'run_definition.json',definition);summaries=[]
    for index,case in enumerate(manifest['cases']):
        arms=follow['arms'] if index%2==0 else list(reversed(follow['arms']))
        for arm in arms:
            if datetime.now(timezone.utc)>=deadline:raise RuntimeError('Authorized deadline reached')
            runner.BenchmarkTools=GuardedBenchmarkTools if arm=='reasoning_guarded' else BenchmarkTools
            summaries.append(runner.run_episode(model,case,arm,primary,out))
            runner.write(out/'progress.json',{'updated_utc':runner.now(),'completed_episodes':len(summaries),
                'planned_episodes':32,'last_case':case['case_id'],'last_arm':arm})
    runner.write(out/'completion.json',{'completed_utc':runner.now(),'cases':16,'episodes':32,
        'model_device':'cuda:0','peak_allocated_bytes':model.torch.cuda.max_memory_allocated(),
        'artifact_sha256':freeze['artifact_sha256']})
    assert all(runner.sha(ROOT/p)==h for p,h in freeze['artifact_sha256'].items())


if __name__=='__main__':main()
