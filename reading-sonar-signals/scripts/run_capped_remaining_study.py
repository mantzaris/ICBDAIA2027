"""User-authorized runtime amendment; frozen inference and metrics are unchanged."""
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    root = Path(__file__).resolve().parents[1]
    os.chdir(root)
    os.environ['PYTHONPATH'] = str(root / 'src')
    amendment = json.loads(Path('configs/ablation_user_cap.json').read_text())
    cap = amendment['capped']
    if amendment['status'] != 'user_authorized_runtime_cap' or cap['episodes'] > amendment['max_new_ablation_episodes']:
        raise RuntimeError('Invalid user cap')
    freeze = json.loads(Path('configs/final_protocol_freeze.json').read_text())
    for path, expected in freeze['sha256'].items():
        if sha(path) != expected:
            raise RuntimeError('Frozen input changed: ' + path)
    if sha(cap['manifest']) != cap['manifest_sha256'] or not Path(cap['reuse_provenance']).is_file():
        raise RuntimeError('Capped preparation has not been verified')
    reference = 'results/ai_annotations/primary_reference.jsonl'
    selection = 'results/development/evaluation/fixed_selection.json'
    phases = [
        ('capped_ablation_inference', ['scripts/run_benchmark.py', '--manifest', cap['manifest'],
         '--output', cap['run'], '--methods', ','.join(cap['methods']), '--budget', '6',
         '--workers', '16', '--batch-size', '8', '--batch-wait-ms', '2000']),
        ('capped_ablation_composition', ['scripts/compose_ablations.py', '--inference', cap['run'],
         '--manifest', cap['manifest'], '--output', cap['composed_run']]),
        ('capped_ablation_evaluation', ['scripts/evaluate_benchmark.py', '--run', cap['composed_run'],
         '--references', reference, '--fixed-selection', selection]),
        ('synthetic_reuse_only', ['scripts/run_synthetic_reuse_only.py']),
        ('original_budgets_and_interventions', ['scripts/run_study_phases.py', '--start', 'budget2', '--stop-after', 'output_sensitivity']),
    ]
    for run in ['main', 'localization/predictions', 'synthetic_ablations', 'budget2', 'budget4', 'robustness', 'sensitivity']:
        phases.append(('morphology_' + run.replace('/', '_'), ['scripts/summarize_morphology.py', '--run', 'results/' + run]))
    for run in ['main', 'synthetic_ablations', 'budget2', 'budget4', 'robustness', 'sensitivity']:
        phases.append(('relations_' + run, ['scripts/evaluate_relation_supplement.py', '--run', 'results/' + run]))
    phases.append(('figures_and_viewer', ['scripts/make_figures.py']))
    directory = Path('reports/remaining_study')
    directory.mkdir(parents=True, exist_ok=True)
    ledger = directory / 'ledger.jsonl'
    completed = {r['phase'] for r in (json.loads(line) for line in ledger.read_text().splitlines()) if r['returncode'] == 0} if ledger.exists() else set()
    for name, command in phases:
        if name in completed:
            print(json.dumps({'event': 'skip_completed_phase', 'phase': name}), flush=True)
            continue
        if shutil.disk_usage(root).free < 1_000_000_000:
            raise RuntimeError('Less than 1GB free')
        started = time.time()
        log = directory / (name + '.' + str(int(started)) + '.log')
        record = {'phase': name, 'argv': [sys.executable] + command, 'started_at_unix': started,
                  'log': str(log), 'amendment_sha256': sha('configs/ablation_user_cap.json')}
        (directory / 'active.json').write_text(json.dumps(record, indent=2) + '\n')
        print(json.dumps(dict(event='start', **record)), flush=True)
        with log.open('w') as stream:
            result = subprocess.run(record['argv'], stdout=stream, stderr=subprocess.STDOUT)
        record.update(finished_at_unix=time.time(), returncode=result.returncode, log_sha256=sha(log))
        with ledger.open('a') as stream:
            stream.write(json.dumps(record) + '\n')
        print(json.dumps(dict(event='finished', **record)), flush=True)
        if result.returncode:
            raise RuntimeError('Study phase failed: ' + name + '; see ' + str(log))
    (directory / 'COMPLETED.json').write_text(json.dumps({
        'status': 'all_queued_experiments_and_exports_finished',
        'completed_at': datetime.now(timezone.utc).isoformat(),
        'requires_before_shutdown': ['result and figure review', 'verified local artifact preservation'],
        'script_sha256': sha(__file__), 'amendment_sha256': sha('configs/ablation_user_cap.json'),
        'phases': [name for name, _ in phases]}, indent=2) + '\n')
    print('CAPPED_REMAINING_STUDY_QUEUE_FINISHED', flush=True)


if __name__ == '__main__':
    main()
