"""Back up only the new intervention, preserving local manuscript and audit edits."""
import hashlib
import json
import shlex
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    config = json.loads((ROOT / '.local/connection.json').read_text())
    ssh = ['ssh', '-i', config['key_path'], '-p', str(config['port']), '-o', 'IdentitiesOnly=yes',
           '-o', 'BatchMode=yes', '-o', 'StrictHostKeyChecking=yes', '-o', 'UserKnownHostsFile=' + config['known_hosts_path']]
    host = config['user'] + '@' + config['host']
    project = config['project_directory']
    paths = ['checkpoints/pilot_velocity_seed17', 'results/velocity_followup', 'results/pools/velocity_followup',
             'logs/pilot_velocity_seed17.jsonl', 'logs/pilot_velocity_seed17.log',
             'logs/velocity_followup_evaluation.log', 'logs/gpu_utilization.jsonl', 'results/run_registry.jsonl',
             'reports/velocity_numerics_terminal.json', 'reports/velocity_numerics_amended.json',
             'reports/velocity_training_match.json']
    code = 'import json; from pathlib import Path; d=json.loads(Path("results/velocity_followup/summary.json").read_text()); assert d["status"]=="completed"; print(d["groups"])'
    command = 'cd ' + shlex.quote(project) + ' && .venv/bin/python -c ' + shlex.quote(code)
    subprocess.run(ssh + [host, command], check=True)
    old_registry = (ROOT / 'results/run_registry.jsonl').read_bytes()
    old_gpu_log = (ROOT / 'logs/gpu_utilization.jsonl').read_bytes()
    for path in paths:
        subprocess.run(['rsync', '-a', '--relative', '--partial', '-e', shlex.join(ssh),
                        host + ':' + project + '/./' + path, str(ROOT) + '/'], check=True)
    assert (ROOT / 'results/run_registry.jsonl').read_bytes().startswith(old_registry)
    assert (ROOT / 'logs/gpu_utilization.jsonl').read_bytes().startswith(old_gpu_log)
    files = []
    for path in paths:
        local = ROOT / path
        files.extend([str(p.relative_to(ROOT)) for p in local.rglob('*') if p.is_file()] if local.is_dir() else [path])
    remote_code = 'import hashlib,json; from pathlib import Path; paths=' + repr(files) + '; print(json.dumps({p:hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in paths}))'
    command = 'cd ' + shlex.quote(project) + ' && .venv/bin/python -c ' + shlex.quote(remote_code)
    expected = json.loads(subprocess.check_output(ssh + [host, command], text=True))
    actual = {path: sha(ROOT / path) for path in files}
    assert actual == expected, 'Remote files changed during backup or a transfer hash differs'
    (ROOT / 'manifests/verified_velocity_backup.json').write_text(json.dumps(actual, indent=2) + '\n')
    checkpoint_path = ROOT / 'manifests/verified_checkpoint_backup.json'
    checkpoints = json.loads(checkpoint_path.read_text())
    checkpoints.update({path: value for path, value in actual.items() if path.endswith('.pt')})
    checkpoint_path.write_text(json.dumps(checkpoints, indent=2) + '\n')
    print('Verified', len(actual), 'new or extended artifact files against the RunPod copies.')


if __name__ == '__main__':
    main()
