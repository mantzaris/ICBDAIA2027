"""Queue the frozen budget check after the configuration-gate experiment."""
import json
import os
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    auth = json.loads((ROOT/'configs/gpu_agent_authorization.json').read_text())
    deadline = datetime.fromisoformat(auth['stop_jobs_by_utc'])
    target = ROOT/'results/gpu_agent_budget/queue.json'
    target.parent.mkdir(parents=True, exist_ok=True)
    assert not target.exists(), 'Budget check is already queued'
    record = {'queued_utc': datetime.now(timezone.utc).isoformat(),
              'status': 'waiting_for_guard_followup', 'deadline_utc': deadline.isoformat()}
    target.write_text(json.dumps(record, indent=2)+'\n')
    while True:
        seconds = int((deadline-datetime.now(timezone.utc)).total_seconds())
        if seconds < 10800:
            record['status'] = 'not_started_insufficient_time'
            target.write_text(json.dumps(record, indent=2)+'\n')
            return
        previous = json.loads((ROOT/'results/gpu_agent_guard/queue.json').read_text())
        if previous['status'] == 'completed':
            break
        if previous['status'] not in ('waiting_for_primary', 'running'):
            record['status'] = 'not_started_prior_phase_incomplete'
            target.write_text(json.dumps(record, indent=2)+'\n')
            return
        time.sleep(30)
    env = os.environ.copy()
    env['PYTHONPATH'] = str(ROOT/'src')
    env['HF_HOME'] = str(ROOT/'.cache/huggingface')
    command = ['timeout', '--signal=TERM', '--kill-after=30s', str(seconds),
               str(ROOT/'.venv-agents/bin/python'), 'scripts/run_gpu_agent_budget_followup.py']
    monitoring = ['nvidia-smi', '--query-gpu=timestamp,name,utilization.gpu,memory.used,memory.total,power.draw',
                  '--format=csv', '--loop=60']
    record.update(status='running', started_utc=datetime.now(timezone.utc).isoformat(),
                  command=command, utilization_command=monitoring)
    target.write_text(json.dumps(record, indent=2)+'\n')
    with (ROOT/'logs/gpu_agent_budget.log').open('w') as log, (ROOT/'logs/gpu_agent_budget_utilization.csv').open('w') as resource_log:
        monitor = subprocess.Popen(monitoring, stdin=subprocess.DEVNULL, stdout=resource_log, stderr=log)
        try:
            result = subprocess.run(command, cwd=ROOT, env=env, stdin=subprocess.DEVNULL,
                                    stdout=log, stderr=subprocess.STDOUT)
        finally:
            monitor.terminate()
            monitor.wait(timeout=10)
    record.update(status='completed' if result.returncode == 0 else 'failed',
                  exit_code=result.returncode, finished_utc=datetime.now(timezone.utc).isoformat())
    target.write_text(json.dumps(record, indent=2)+'\n')


if __name__ == '__main__':
    main()
