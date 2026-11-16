"""Remote launch with a process deadline; leaves the pod running after jobs finish."""
import hashlib
import json
import os
import subprocess
from datetime import datetime,timezone
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]


def main():
    os.chdir(ROOT)
    assert (ROOT/'results/gpu_agent_benchmark/smoke/completion.json').exists()
    freeze=json.loads((ROOT/'preregistration/gpu_agent_evaluation_freeze.json').read_text())
    for path,expected in freeze['artifact_sha256'].items():
        assert hashlib.sha256((ROOT/path).read_bytes()).hexdigest()==expected,path
    authorization=json.loads((ROOT/'configs/gpu_agent_authorization.json').read_text())
    deadline=datetime.fromisoformat(authorization['stop_jobs_by_utc'])
    seconds=int((deadline-datetime.now(timezone.utc)).total_seconds())
    assert seconds>600
    marker=ROOT/'results/gpu_agent_benchmark/evaluation_launch.json'
    assert not marker.exists(),'A launch is already recorded; do not duplicate it'
    env=os.environ.copy();env['PYTHONPATH']=str(ROOT/'src');env['HF_HOME']=str(ROOT/'.cache/huggingface')
    log=(ROOT/'logs/gpu_agent_evaluation.log').open('w')
    command=['timeout','--signal=TERM','--kill-after=30s',str(seconds),
             str(ROOT/'.venv-agents/bin/python'),'scripts/run_gpu_agent_benchmark.py','--phase','evaluation']
    process=subprocess.Popen(command,cwd=ROOT,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,
                             start_new_session=True,close_fds=True)
    record={'launched_utc':datetime.now(timezone.utc).isoformat(),'pid':process.pid,'command':command,
            'deadline_utc':deadline.isoformat(),'freeze_sha256':hashlib.sha256((ROOT/'preregistration/gpu_agent_evaluation_freeze.json').read_bytes()).hexdigest(),
            'pod_shutdown_requested':False,'local_language_model_inference':False}
    marker.write_text(json.dumps(record,indent=2)+'\n')
    print(json.dumps(record,indent=2))


if __name__=='__main__':main()
