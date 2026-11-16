"""Evaluate the frozen, single velocity intervention on existing development cases.

No old diagnostic gate is applied to the new generator. Actual C/test recordings
cannot enter this runner. References are only used after generation for scoring.
"""
from __future__ import annotations

import argparse
import gzip
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch

from acoustic_regeneration.data.observation import Observation, normalize_observation
from acoustic_regeneration.data.preprocessing import crop_source
from acoustic_regeneration.diagnostics.features import spectral_distances
from acoustic_regeneration.evaluation.metrics import MetricFloors, gap_metrics
from acoustic_regeneration.registry import append_record, file_hash, remaining_seconds
from acoustic_regeneration.restoration.diffusion import ddim_batch
from acoustic_regeneration.restoration.learned import load_ema_model

ROOT = Path(__file__).resolve().parents[1]
OLD = ROOT / 'results/pilot_pools_clipped_seed17'
OUT = ROOT / 'results/velocity_followup'


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def paired_interval(values, seed=20261116):
    values = np.asarray(values, dtype=np.float64)
    rng = np.random.default_rng(seed)
    draws = values[rng.integers(0, len(values), size=(2000, len(values)))].mean(axis=1)
    return {'mean': float(values.mean()), 'lower_95': float(np.quantile(draws, .025)),
            'upper_95': float(np.quantile(draws, .975)), 'groups': len(values),
            'scope': 'Exploratory paired group bootstrap on reused development cases'}


def aggregate(records):
    summary = {}
    for role in ['V', 'dev_C', 'dev_test']:
        selected = [row for row in records if row['pilot_role'] == role]
        if len(selected) != 64:
            raise ValueError('Every frozen V role requires exactly 64 completed groups')
        methods = {}
        for name in ['epsilon_amended', 'velocity_terminal', 'velocity_amended']:
            groups = [row['methods'][name] for row in selected]
            methods[name] = {
                'first_draw_mean_Q': float(np.mean([g['candidate_metrics'][0]['Q'] for g in groups])),
                'first_draw_mean_gap_nmse': float(np.mean([g['candidate_metrics'][0]['gap_nmse'] for g in groups])),
                'first_draw_mean_gap_SC': float(np.mean([g['candidate_metrics'][0]['gap_spectral_convergence'] for g in groups])),
                'medoid_mean_Q': float(np.mean([g['candidate_metrics'][g['medoid']]['Q'] for g in groups])),
                'mean_Q_over_four_draws_per_group': float(np.mean([np.mean([m['Q'] for m in g['candidate_metrics']]) for g in groups])),
                'oracle_opportunity_groups': sum(min(m['Q'] for m in row['methods'][name]['candidate_metrics']) <= row['baseline_metrics']['Q'] - .02 for row in selected),
                'first_draw_worse_than_baseline_groups': sum(row['methods'][name]['candidate_metrics'][0]['Q'] > row['baseline_metrics']['Q'] + .02 for row in selected),
                'maximum_observed_drift': max(m['observed_max_drift'] for g in groups for m in g['candidate_metrics']),
            }
        if role == 'V':
            terminal = [row['epsilon_terminal_metrics'] for row in selected]
            methods['epsilon_terminal'] = {key: float(np.mean([r[metric] for r in terminal])) for key, metric in
                [('first_draw_mean_Q', 'Q'), ('first_draw_mean_gap_nmse', 'gap_nmse'), ('first_draw_mean_gap_SC', 'gap_spectral_convergence')]}
        delta = {}
        for comparator in ['epsilon_amended', 'velocity_amended']:
            delta['velocity_terminal_minus_' + comparator] = paired_interval([
                row['methods']['velocity_terminal']['candidate_metrics'][0]['Q'] -
                row['methods'][comparator]['candidate_metrics'][0]['Q'] for row in selected])
        delta['velocity_terminal_minus_baseline'] = paired_interval([
            row['methods']['velocity_terminal']['candidate_metrics'][0]['Q'] - row['baseline_metrics']['Q'] for row in selected])
        summary[role] = {'groups': len(selected), 'baseline_mean_Q': float(np.mean([row['baseline_metrics']['Q'] for row in selected])),
                         'methods': methods, 'paired_first_draw_Q': delta}
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--git-commit', required=True)
    args = parser.parse_args()
    torch.set_num_threads(4)
    protocol_path = ROOT / 'preregistration/velocity_followup.json'
    protocol = json.loads(protocol_path.read_text())
    for path, expected in protocol['source_hashes'].items():
        assert file_hash(ROOT / path) == expected, path
    original_cases = {row['case_id']: row for row in map(json.loads, (OLD / 'cases.jsonl').read_text().splitlines())}
    with gzip.open(ROOT / 'manifests/pilot_sources.jsonl.gz', 'rt') as stream:
        sources = {row['original_file_id']: row for row in map(json.loads, stream)}
    checkpoint = ROOT / 'checkpoints/pilot_velocity_seed17/latest.pt'
    velocity, config, metadata = load_ema_model(checkpoint)
    checkpoint_payload = torch.load(checkpoint, map_location='cpu', weights_only=False)
    assert checkpoint_payload['update'] == 5000
    del checkpoint_payload
    assert config['prediction_type'] == 'velocity'
    epsilon_path = ROOT / 'checkpoints/pilot_conditional_seed17/best.pt'
    epsilon, _, _ = load_ema_model(epsilon_path)
    assert file_hash(epsilon_path) == json.loads((OLD / 'pool_config.json').read_text())['conditional_sha256']
    deadline = json.loads((ROOT / 'configs/run_authorization.json').read_text())['stop_jobs_by_utc']
    OUT.mkdir(parents=True, exist_ok=True)
    paths = ROOT / 'results/pools/velocity_followup'
    paths.mkdir(parents=True, exist_ok=True)
    identity = {'protocol_sha256': file_hash(protocol_path), 'velocity_checkpoint_sha256': file_hash(checkpoint),
                'epsilon_checkpoint_sha256': file_hash(epsilon_path), 'script_sha256': file_hash(Path(__file__)),
                'diagnostic_script_sha256': file_hash(ROOT / 'scripts/diagnose_sampler.py'),
                'git_commit': args.git_commit, 'checkpoint_update': 5000,
                'source_manifest_sha256': metadata['manifest_hash']}
    config_path = OUT / 'config.json'
    if config_path.exists():
        assert json.loads(config_path.read_text()) == identity, 'Do not mix changed follow-up configurations'
    save(config_path, identity)
    start, calls, records = time.monotonic(), 0, []
    cases_path = OUT / 'cases.jsonl'
    if cases_path.exists():
        records = list(map(json.loads, cases_path.read_text().splitlines()))
        for row in records:
            assert file_hash(ROOT / row['artifact_path']) == row['artifact_sha256']
    completed = {row['case_id'] for row in records}
    registry = ROOT / 'results/run_registry.jsonl'
    run_id = 'velocity_followup_evaluation'
    append_record(registry, {'run_id': run_id, 'status': 'running', 'command': sys.argv,
                            **identity, 'hardware': torch.cuda.get_device_name(), 'planned_cases': 192})
    torch.cuda.reset_peak_memory_stats()
    status = 'running'
    try:
        for label, sampler_path in [('terminal', None), ('amended', ROOT / 'preregistration/sampler_amendment.json')]:
            output = 'reports/velocity_numerics_' + label + '.json'
            if (ROOT / output).exists():
                old = json.loads((ROOT / output).read_text())
                assert old['checkpoint_sha256'] == identity['velocity_checkpoint_sha256']
                continue
            command = [sys.executable, 'scripts/diagnose_sampler.py', '--checkpoint', str(checkpoint), '--output', output]
            if sampler_path:
                command += ['--sampler-config', str(sampler_path)]
            with (ROOT / 'logs' / ('velocity_numerics_' + label + '.log')).open('w') as log:
                subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)
            calls += 4 * 2 * (8 + 32)
        for frozen in protocol['cases']:
            if frozen['case_id'] in completed:
                continue
            if remaining_seconds(deadline) < 180:
                status = 'budget_stopped'
                break
            old = original_cases[frozen['case_id']]
            source = sources[frozen['source_id']]
            assert old['source_partition'] == source['partition'] == frozen['source_partition'] == 'V'
            for key in frozen:
                assert old[key] == frozen[key], key
            source_path = ROOT / source['processed_path']
            reference = crop_source(np.load(source_path, allow_pickle=False))
            mask = np.ones(len(reference), dtype=bool)
            a, b = old['gap_start'], old['gap_end']
            mask[a:b] = False
            original = Observation(reference, mask)
            observation, transform = normalize_observation(original, config['rms_floor'])
            assert abs(transform.mean - old['normalization_mean']) < 1e-12
            assert abs(transform.scale - old['normalization_scale']) < 1e-12
            seeds = old['candidate_seeds']
            generated = {}
            elapsed = {}
            for method, endpoint, clip in [('velocity_terminal', 999, None), ('velocity_amended', 934, 5.896913617434104)]:
                torch.cuda.synchronize()
                case_start = time.perf_counter()
                draws = ddim_batch(velocity, [observation] * 4, seeds, 32, initial_timestep=endpoint, clean_clip=clip)
                torch.cuda.synchronize()
                elapsed[method] = time.perf_counter() - case_start
                calls += sum(draw.denoiser_calls for draw in draws)
                generated[method] = np.stack([draw.waveform for draw in draws])
            terminal = None
            if old['pilot_role'] == 'V':
                draw = ddim_batch(epsilon, [observation], seeds[:1], 32)[0]
                calls += draw.denoiser_calls
                terminal = draw.waveform
            # Scoring sees the target only after all observation-only calls return.
            target = transform.normalize_reference_for_scoring(reference)
            floors = MetricFloors(config['rms_floor'] / transform.scale)
            assert file_hash(ROOT / old['pool_path']) == old['pool_sha256']
            with np.load(ROOT / old['pool_path'], allow_pickle=False) as pool:
                cached = np.repeat(observation.values[None], 4, axis=0)
                cached[:, a:b] = pool['candidates_gap']
            generated['epsilon_amended'] = cached
            methods, arrays = {}, {}
            for name, candidates in generated.items():
                metrics = [gap_metrics(candidate, target, original.gaps, floors) for candidate in candidates]
                distances = spectral_distances(candidates, original.gaps[0], floors.rms)
                medoid = int(np.argmin(distances.sum(axis=1)))
                if name == 'epsilon_amended':
                    assert medoid == old['medoid']
                    for left, right in zip(metrics, old['candidate_metrics']):
                        for key in ['Q', 'gap_nmse', 'gap_spectral_convergence']:
                            assert abs(left[key] - right[key]) < 1e-10, key
                methods[name] = {'candidate_metrics': metrics, 'medoid': medoid,
                                 'inference_seconds': elapsed.get(name), 'diffusion_calls': 128}
                arrays[name + '_gap'] = candidates[:, a:b]
            if terminal is not None:
                arrays['epsilon_terminal_gap'] = terminal[a:b]
            path = paths / (old['case_id'] + '.npz')
            np.savez_compressed(path, **arrays)
            record = {key: old[key] for key in ['case_id', 'source_id', 'group_id', 'source_partition', 'pilot_role', 'gap_start', 'gap_end', 'gap_ms', 'candidate_seeds']}
            record.update(methods=methods, baseline_metrics=old['fallback_metrics'],
                          epsilon_terminal_metrics=gap_metrics(terminal, target, original.gaps, floors) if terminal is not None else None,
                          source_array_sha256=file_hash(source_path), old_pool_sha256=old['pool_sha256'],
                          artifact_path=str(path.relative_to(ROOT)), artifact_sha256=file_hash(path))
            append_record(cases_path, record)
            records.append(record)
            if len(records) == 1 or len(records) % 16 == 0:
                print(json.dumps({'completed': len(records), 'planned': 192, 'role': old['pilot_role'],
                                  'elapsed_seconds': time.monotonic() - start}), flush=True)
        if status == 'running':
            summary = {'status': 'completed', **identity, 'groups': len(records), 'roles': aggregate(records),
                       'scope': 'Targeted post hoc intervention on reused V cases; actual C/test outcomes remain unopened',
                       'old_release_gate_reused': False, 'new_release_gate_calibrated': False,
                       'classification_or_equipment_control_evaluated': False,
                       'elapsed_seconds': time.monotonic() - start}
            save(OUT / 'summary.json', summary)
            status = 'completed'
            print(json.dumps(summary, indent=2), flush=True)
    except Exception as error:
        status = 'failed'
        append_record(OUT / 'failures.jsonl', {'error': repr(error), 'completed_cases': len(records)})
        raise
    finally:
        append_record(registry, {'run_id': run_id, 'status': status, **identity,
                                'completed_cases': len(records), 'planned_cases': 192,
                                'forward_evaluations_in_this_invocation': calls,
                                'elapsed_seconds': time.monotonic() - start,
                                'peak_vram_bytes': torch.cuda.max_memory_allocated(),
                                'output_path': str(OUT.relative_to(ROOT))})


if __name__ == '__main__':
    main()
