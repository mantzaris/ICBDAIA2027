"""Run the frozen follow-up after the primary process exits, within the same authorization."""
import json
import os
import subprocess
import time
from datetime import datetime,timezone
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]


def alive(pid):
    try:return Path(f'/proc/{pid}/stat').read_text().split()[2]!='Z'
    except FileNotFoundError:return False


def main():
    os.chdir(ROOT)
    auth=json.loads((ROOT/'configs/gpu_agent_authorization.json').read_text())
    deadline=datetime.fromisoformat(auth['stop_jobs_by_utc'])
    launch=json.loads((ROOT/'results/gpu_agent_benchmark/evaluation_launch.json').read_text())
    completion=ROOT/'results/gpu_agent_benchmark/evaluation/completion.json'
    record_path=ROOT/'results/gpu_agent_guard/queue.json'
    record={'queued_utc':datetime.now(timezone.utc).isoformat(),'status':'waiting_for_primary',
            'primary_pid':launch['pid'],'deadline_utc':deadline.isoformat()}
    record_path.write_text(json.dumps(record,indent=2)+'\n')
    while alive(launch['pid']):
        if (deadline-datetime.now(timezone.utc)).total_seconds()<7200:
            record['status']='not_started_insufficient_time';record_path.write_text(json.dumps(record,indent=2)+'\n');return
        time.sleep(30)
    if not completion.exists():
        record['status']='not_started_primary_incomplete';record_path.write_text(json.dumps(record,indent=2)+'\n');return
    seconds=int((deadline-datetime.now(timezone.utc)).total_seconds())
    if seconds<7200:
        record['status']='not_started_insufficient_time';record_path.write_text(json.dumps(record,indent=2)+'\n');return
    env=os.environ.copy();env['PYTHONPATH']=str(ROOT/'src');env['HF_HOME']=str(ROOT/'.cache/huggingface')
    command=['timeout','--signal=TERM','--kill-after=30s',str(seconds),
             str(ROOT/'.venv-agents/bin/python'),'scripts/run_gpu_agent_guard_followup.py']
    record.update(status='running',started_utc=datetime.now(timezone.utc).isoformat(),command=command)
    record_path.write_text(json.dumps(record,indent=2)+'\n')
    with (ROOT/'logs/gpu_agent_guard.log').open('w') as log:
        result=subprocess.run(command,cwd=ROOT,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT)
    record.update(status='completed' if result.returncode==0 else 'failed',exit_code=result.returncode,
                  finished_utc=datetime.now(timezone.utc).isoformat())
    record_path.write_text(json.dumps(record,indent=2)+'\n')


if __name__=='__main__':main()
