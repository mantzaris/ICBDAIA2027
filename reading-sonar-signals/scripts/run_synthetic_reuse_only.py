"""Compose existing synthetic F/G4 reports under the user cap; zero new inference."""
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


def save_once(path, value):
    path = Path(path)
    text = json.dumps(value, indent=2) + '\n'
    if path.exists() and json.loads(path.read_text()) != value:
        raise RuntimeError('Refusing to overwrite different artifact: ' + str(path))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)


def main():
    root = Path(__file__).resolve().parents[1]
    os.chdir(root)
    os.environ['PYTHONPATH'] = str(root / 'src')
    amendment_path = Path('configs/ablation_user_cap.json')
    amendment = json.loads(amendment_path.read_text())
    reuse = amendment['synthetic_reuse']
    if reuse['new_inference_episodes'] != 0 or reuse['methods'] != ['F', 'G4', 'no_verifier']:
        raise RuntimeError('Unexpected synthetic reuse amendment')
    for path, expected in json.loads(Path('configs/final_protocol_freeze.json').read_text())['sha256'].items():
        if sha(path) != expected:
            raise RuntimeError('Frozen input changed: ' + path)
    manifest = Path(reuse['manifest'])
    if sha(manifest) != reuse['manifest_sha256']:
        raise RuntimeError('Synthetic manifest changed')
    scenes = [json.loads(line) for line in manifest.read_text().splitlines() if line.strip()]
    if len(scenes) != reuse['scene_count']:
        raise RuntimeError('Synthetic scene count differs')
    source = Path(reuse['source_run'])
    if not (source / 'COMPLETED.json').is_file():
        raise RuntimeError('Main inference is unfinished')
    stub = Path(reuse['stub_run'])
    original = json.loads((source / 'run.json').read_text())
    derivation = {'kind': 'reuse_only_no_inference', 'additional_inference_calls': 0,
                  'source_run_configuration': str(source / 'run.json'),
                  'source_run_configuration_sha256': sha(source / 'run.json'),
                  'source_completion_sha256': sha(source / 'COMPLETED.json'),
                  'amendment_sha256': sha(amendment_path), 'administrative_script_sha256': sha(__file__)}
    config = dict(original, manifest=str(manifest), manifest_sha256=sha(manifest),
                  scene_count=len(scenes), methods=[], started_at_unix=time.time(), derivation=derivation)
    if (stub / 'run.json').exists():
        config['started_at_unix'] = json.loads((stub / 'run.json').read_text())['started_at_unix']
    save_once(stub / 'run.json', config)
    proof = dict(status='complete_reuse_only', additional_inference_calls=0, new_inference_episodes=0,
                 source_run=str(source), source_run_configuration_sha256=sha(source / 'run.json'),
                 source_completion_sha256=sha(source / 'COMPLETED.json'), amendment_sha256=sha(amendment_path),
                 administrative_script_sha256=sha(__file__), scene_count=len(scenes),
                 composed_methods=reuse['methods'], composed_episodes=reuse['composed_episodes'],
                 omitted_methods=reuse['omitted_methods'])
    save_once(reuse['proof'], proof)
    marker = dict(config, new_completed_episodes=0, execution_failures=0,
                  finished_at_unix=time.time(), additional_inference_calls=0)
    if (stub / 'COMPLETED.json').exists():
        marker['finished_at_unix'] = json.loads((stub / 'COMPLETED.json').read_text())['finished_at_unix']
    save_once(stub / 'COMPLETED.json', marker)
    subprocess.run([sys.executable, 'scripts/compose_ablations.py', '--main', str(source),
                    '--inference', str(stub), '--manifest', str(manifest), '--output', reuse['composed_run']], check=True)
    inventory = []
    for scene in scenes:
        sid = scene['scene_id']
        truth = source / 'sealed_truth' / (sid + '.json')
        if json.loads(truth.read_text()).get('scene_id') != sid:
            raise RuntimeError('Exact truth identity mismatch')
        destination = Path(reuse['composed_run']) / 'sealed_truth' / truth.name
        destination.parent.mkdir(exist_ok=True)
        if destination.exists() and sha(destination) != sha(truth):
            raise RuntimeError('Existing exact truth differs')
        shutil.copy2(truth, destination)
        inventory.append(dict(scene_id=sid, source=str(truth), destination=str(destination), sha256=sha(truth)))
    save_once(Path(reuse['composed_run']) / 'exact_truth_provenance.json', inventory)
    subprocess.run([sys.executable, 'scripts/evaluate_benchmark.py', '--run', reuse['composed_run'],
                    '--references', 'results/ai_annotations/primary_reference.jsonl'], check=True)
    print('SYNTHETIC_REUSE_ONLY_FINISHED', flush=True)


if __name__ == '__main__':
    main()
