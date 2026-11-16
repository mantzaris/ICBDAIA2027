"""Plot saved correction evidence and write independently checked summaries."""
import gzip
import hashlib
import json
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from acoustic_regeneration.evaluation.metrics import MetricFloors, gap_metrics

ROOT = Path(__file__).resolve().parents[1]


def read(path):
    return json.loads((ROOT / path).read_text())


def digest(path):
    return hashlib.sha256((ROOT / path).read_bytes()).hexdigest()


def main():
    summary = read('results/velocity_followup/summary.json')
    assert summary['status'] == 'completed' and summary['groups'] == 192
    records = [json.loads(line) for line in (ROOT / 'results/velocity_followup/cases.jsonl').read_text().splitlines()]
    assert len(records) == len({r['case_id'] for r in records}) == 192
    old = {r['case_id']: r for r in map(json.loads, (ROOT / 'results/pilot_pools_clipped_seed17/cases.jsonl').read_text().splitlines())}
    with gzip.open(ROOT / 'manifests/pilot_sources.jsonl.gz', 'rt') as stream:
        sources = {r['original_file_id']: r for r in map(json.loads, stream)}
    numerical_paths = ['reports/sampler_numerics.json', 'reports/sampler_numerics_bounded.json',
                       'reports/sampler_numerics_truncated.json', 'reports/velocity_numerics_terminal.json',
                       'reports/velocity_numerics_amended.json']
    numerical = [read(path) for path in numerical_paths]
    for index, report in enumerate(numerical):
        expected = summary['epsilon_checkpoint_sha256'] if index < 3 else summary['velocity_checkpoint_sha256']
        assert report['checkpoint_sha256'] == expected
    assert numerical[3]['initial_timestep'] == 999 and numerical[3]['clean_prediction_clip_abs'] is None
    assert numerical[4]['initial_timestep'] == 934
    identities = [{(r['source_id'], r['precision']) for r in report['cases']} for report in numerical]
    assert all(keys == identities[0] for keys in identities)
    numerical_summary = {}
    for label, report in zip(['epsilon_terminal', 'epsilon_clipped', 'epsilon_amended', 'velocity_terminal', 'velocity_amended'], numerical):
        numerical_summary[label] = {}
        for precision in ['fp32', 'bf16']:
            cases = [r for r in report['cases'] if r['precision'] == precision]
            assert len(cases) == 4
            row = {key: float(np.mean([r['metrics'][key] for r in cases])) for key in ['gap_nmse', 'gap_spectral_convergence', 'Q']}
            row['maximum_observed_drift'] = max(r['metrics']['observed_max_drift'] for r in cases)
            row['terminal_teacher_forced_clean_MSE'] = float(np.mean([r['teacher_forced'][0]['x0_mse'] for r in cases]))
            numerical_summary[label][precision] = row
        # Check the derived local error-propagation equations against measured G errors.
        for case in report['cases']:
            for point in case['teacher_forced']:
                target_error = point.get('prediction_mse', point.get('epsilon_mse'))
                multiplier = (1 - point['alpha_bar']) if label.startswith('velocity') else (1 - point['alpha_bar']) / point['alpha_bar']
                assert np.isclose(point['x0_mse'], multiplier * target_error, rtol=1e-3, atol=1e-7)
    rms_floor = read('results/pilot_pools_clipped_seed17/pool_config.json')['rms_floor']
    verified_draws = 0
    for row in records:
        assert row['source_partition'] == 'V'
        assert digest(row['artifact_path']) == row['artifact_sha256']
        assert row['old_pool_sha256'] == old[row['case_id']]['pool_sha256']
        assert row['candidate_seeds'] == old[row['case_id']]['candidate_seeds']
        source_path = sources[row['source_id']]['processed_path']
        assert digest(source_path) == row['source_array_sha256']
        reference = np.load(ROOT / source_path, allow_pickle=False)[:32768].astype(np.float64)
        previous = old[row['case_id']]
        target = (reference - previous['normalization_mean']) / previous['normalization_scale']
        floors = MetricFloors(rms_floor / previous['normalization_scale'])
        a, b = row['gap_start'], row['gap_end']
        with np.load(ROOT / row['artifact_path'], allow_pickle=False) as arrays:
            for name in ['velocity_terminal', 'velocity_amended']:
                for index, stored in enumerate(row['methods'][name]['candidate_metrics']):
                    reconstructed = target.copy()
                    reconstructed[a:b] = arrays[name + '_gap'][index]
                    calculated = gap_metrics(reconstructed, target, [(a, b)], floors)
                    for key in ['Q', 'gap_nmse', 'gap_spectral_convergence']:
                        assert np.isclose(calculated[key], stored[key], rtol=0, atol=1e-10), (row['case_id'], name, key)
                    verified_draws += 1
        for method in row['methods'].values():
            assert len(method['candidate_metrics']) == 4
            assert all(m['observed_max_drift'] == 0 for m in method['candidate_metrics'])
        for actual, expected in zip(row['methods']['epsilon_amended']['candidate_metrics'], old[row['case_id']]['candidate_metrics']):
            for key in ['Q', 'gap_nmse', 'gap_spectral_convergence']:
                assert np.isclose(actual[key], expected[key], rtol=0, atol=1e-10)
    # Recompute headline aggregates independently of the remote summary writer.
    date_sensitivity = {}
    for role in ['V', 'dev_C', 'dev_test']:
        cases = [r for r in records if r['pilot_role'] == role]
        assert len(cases) == 64
        for method in ['epsilon_amended', 'velocity_terminal', 'velocity_amended']:
            metrics = [r['methods'][method]['candidate_metrics'][0] for r in cases]
            values = summary['roles'][role]['methods'][method]
            for source, key in [('Q', 'first_draw_mean_Q'), ('gap_nmse', 'first_draw_mean_gap_nmse'), ('gap_spectral_convergence', 'first_draw_mean_gap_SC')]:
                assert np.isclose(np.mean([m[source] for m in metrics]), values[key], atol=1e-12, rtol=0)
            opportunity = sum(min(m['Q'] for m in r['methods'][method]['candidate_metrics']) <= r['baseline_metrics']['Q'] - .02 for r in cases)
            assert opportunity == values['oracle_opportunity_groups']
        by_date = {}
        for row in cases:
            date = sources[row['source_id']]['date']
            delta = row['methods']['velocity_terminal']['candidate_metrics'][0]['Q'] - row['methods']['epsilon_amended']['candidate_metrics'][0]['Q']
            by_date.setdefault(date, []).append(delta)
        date_means = np.array([np.mean(values) for _, values in sorted(by_date.items())])
        rng = np.random.default_rng(20261116)
        bootstrap = date_means[rng.integers(0, len(date_means), size=(2000, len(date_means)))].mean(axis=1)
        date_sensitivity[role] = {'unique_dates': len(date_means), 'mean_delta_Q': float(date_means.mean()),
                                 'lower_95': float(np.quantile(bootstrap, .025)), 'upper_95': float(np.quantile(bootstrap, .975)),
                                 'estimand': 'Equal weight per date; velocity terminal minus original amended epsilon, fixed first draw'}
    sensitivity_path = ROOT / 'reports/velocity_followup_date_sensitivity.json'
    sensitivity_path.write_text(json.dumps({'roles': date_sensitivity, 'scope': 'Exploratory date-balanced sensitivity; a different estimand from the recording-group mean'}, indent=2) + '\n')
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 7, 'axes.titlesize': 7.2,
                         'axes.labelsize': 7, 'xtick.labelsize': 6, 'ytick.labelsize': 6,
                         'pdf.fonttype': 42, 'svg.hashsalt': 'velocity-followup-v1'})
    fig, axes = plt.subplots(1, 2, figsize=(4.7, 2.1))
    colors = {'epsilon': '#C45237', 'velocity': '#277C94'}
    for label, report in [('epsilon', numerical[0]), ('velocity', numerical[3])]:
        for precision in ['fp32', 'bf16']:
            cases = [r for r in report['cases'] if r['precision'] == precision]
            times = np.array([p['time'] for p in cases[0]['teacher_forced']])
            errors = np.mean([[p['x0_mse'] for p in r['teacher_forced']] for r in cases], axis=0)
            order = np.argsort(times)
            axes[0].semilogy(times[order], errors[order], color=colors[label],
                             linestyle='-' if precision == 'fp32' else '--',
                             marker='o' if precision == 'fp32' else None, markersize=2.3, linewidth=.9,
                             label=('Noise target' if label == 'epsilon' else 'Velocity target') if precision == 'fp32' else None)
    axes[0].set(xlabel='Diffusion step (more noise →)', ylabel='Clean-signal MSE',
                title='A  Isolate conversion error: 4 G sources', xticks=[0, 500, 999])
    axes[0].legend(frameon=False, fontsize=6, loc='upper left')
    axes[0].grid(alpha=.18, linewidth=.4)
    cases = [r for r in records if r['pilot_role'] == 'V']
    before = np.array([r['methods']['epsilon_amended']['candidate_metrics'][0]['Q'] for r in cases])
    after = np.array([r['methods']['velocity_terminal']['candidate_metrics'][0]['Q'] for r in cases])
    axes[1].plot([0, 1], [0, 1], color='#777777', linestyle='--', linewidth=.7)
    axes[1].scatter(before, after, s=10, color=colors['velocity'], alpha=.7, linewidths=.25, edgecolors='white')
    axes[1].set(xlabel='Original amended sampler: error Q', ylabel='Velocity correction: error Q',
                title='B  Check sound repair: 64 V groups', xlim=(0, 1), ylim=(0, 1), xticks=[0, .5, 1], yticks=[0, .5, 1])
    axes[1].text(.04, .94, 'Below line: lower error', transform=axes[1].transAxes, fontsize=6, va='top')
    axes[1].grid(alpha=.18, linewidth=.4)
    fig.subplots_adjust(left=.12, right=.98, top=.87, bottom=.22, wspace=.5)
    for extension in ['pdf', 'png', 'svg']:
        path = ROOT / ('figures/velocity_correction.' + extension)
        metadata = {'CreationDate': None, 'ModDate': None} if extension == 'pdf' else ({'Date': None} if extension == 'svg' else {})
        fig.savefig(path, dpi=450, bbox_inches='tight', pad_inches=.04, metadata=metadata)
        if extension == 'svg':
            path.write_text('\n'.join(line.rstrip() for line in path.read_text().splitlines()) + '\n')
        if extension == 'pdf':
            shutil.copyfile(path, ROOT / 'manuscript' / path.name)
    plt.close(fig)
    report = {'status': 'passed', 'script_sha256': digest('scripts/plot_velocity_followup.py'),
              'source_files': {p: digest(p) for p in numerical_paths + ['results/velocity_followup/summary.json', 'results/velocity_followup/cases.jsonl']},
              'numerical_controls': numerical_summary, 'development_groups': 192, 'verified_gap_artifacts': 192,
              'old_candidate_metrics_recomputed_and_matched': True, 'local_error_amplification_identity_verified': True,
              'new_velocity_draw_gap_metrics_recomputed': verified_draws,
              'primary_plot_groups': 64, 'primary_plot_role': 'V', 'plot_draw': 'fixed first candidate seed',
              'date_sensitivity_sha256': digest('reports/velocity_followup_date_sensitivity.json'),
              'figure_sha256': digest('figures/velocity_correction.pdf'),
              'scope': 'Numerical diagnosis on four G sources and exploratory paired evaluation on reused V; no classification, release-policy calibration, or hardware-control claim'}
    save_path = ROOT / 'reports/velocity_followup_validation.json'
    save_path.write_text(json.dumps(report, indent=2) + '\n')
    with (ROOT / 'results/local_run_registry.jsonl').open('a') as stream:
        stream.write(json.dumps({'run_id': 'velocity_followup_presentation', 'status': 'completed',
            'recorded_utc': datetime.now(timezone.utc).isoformat(),
            'git_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
            'script_sha256': report['script_sha256'], 'report_sha256': digest('reports/velocity_followup_validation.json'),
            'output_paths': ['figures/velocity_correction.pdf', 'reports/velocity_followup_validation.json'],
            'scope': 'Validation and plotting of saved results; no new inference'}) + '\n')
    print(json.dumps({'verified_groups': 192, 'numerical_controls': numerical_summary}, indent=2))


if __name__ == '__main__':
    main()
