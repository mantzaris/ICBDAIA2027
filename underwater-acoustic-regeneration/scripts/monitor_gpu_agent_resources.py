"""Record minute-spaced GPU snapshots until the queued experiments finish."""
import csv
import json
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    auth = json.loads((ROOT / 'configs/gpu_agent_authorization.json').read_text())
    deadline = datetime.fromisoformat(auth['stop_jobs_by_utc'])
    log = ROOT / 'logs/gpu_agent_utilization.jsonl'
    # This monitor starts during the primary evaluation. Do not imply coverage
    # of setup, smoke checks, or the earlier part of the primary run.
    with log.open('x') as stream:
        while True:
            now = datetime.now(timezone.utc)
            queue = json.loads((ROOT / 'results/gpu_agent_guard/queue.json').read_text())
            output = subprocess.check_output([
                'nvidia-smi', '--query-gpu=name,utilization.gpu,memory.used,memory.total,power.draw',
                '--format=csv,noheader,nounits'], text=True)
            values = next(csv.reader(output.splitlines()))
            item = {'utc': now.isoformat(), 'gpu': values[0].strip(),
                    'utilization_percent': float(values[1]), 'memory_used_mib': float(values[2]),
                    'memory_total_mib': float(values[3]), 'power_watts': float(values[4]),
                    'followup_queue_status': queue['status'], 'sampling_interval_seconds': 60}
            stream.write(json.dumps(item) + '\n')
            stream.flush()
            if now >= deadline or queue['status'] not in ('waiting_for_primary', 'running'):
                return
            time.sleep(min(60, max(0, (deadline-now).total_seconds())))


if __name__ == '__main__':
    main()
