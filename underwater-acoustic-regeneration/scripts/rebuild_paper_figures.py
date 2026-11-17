"""Redraw the three current paper figures using only committed evidence.

This renders saved measurements; the original full waveform/training audits stay
in their existing scripts. No data download, model loading, or inference occurs.
"""
import argparse
import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def read(name):
    return json.loads((ROOT / name).read_text())


def acoustic_figure():
    import numpy as np
    import build_acoustic_examples as acoustic
    from export_paper_plot_inputs import array_hash

    manifest = read('figures/data/acoustic_examples.json')
    path = ROOT / 'figures/data' / manifest['array_file']
    assert hashlib.sha256(path.read_bytes()).hexdigest() == manifest['array_sha256']
    for field in ('original_renderer', 'selection_manifest'):
        assert acoustic.digest(ROOT / manifest[field]) == manifest[field + '_sha256']
    examples = []
    with np.load(path, allow_pickle=False) as arrays:
        for case in manifest['cases']:
            evidence = case['evidence']
            tag = evidence['tag']
            start, end = evidence['gap_start_sample'], evidence['gap_end_sample_exclusive']
            reference = arrays[tag + '_reference']
            signals = {name: reference.copy() for name in case['signal_sha256_float64_le']}
            signals['masked'][start:end] = 0
            for name in ('fallback_B_equals_P', 'selected_C3', 'velocity_first'):
                signals[name][start:end] = arrays[tag + '_' + name + '_gap']
            for name, expected in case['signal_sha256_float64_le'].items():
                assert array_hash(signals[name]) == expected, (tag, name)
            examples.append((evidence, signals))
    # Exactly the original spectrogram rendering function and floating-point samples.
    acoustic.comparisons(examples)


def velocity_figure():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np

    # Rendering matches plot_velocity_followup.py. That original script also
    # re-audits 192 archived waveform cases; this path uses its saved measurements.
    numerical = [read(name) for name in ('reports/sampler_numerics.json',
                                        'reports/velocity_numerics_terminal.json')]
    records = [json.loads(line) for line in
               (ROOT / 'results/velocity_followup/cases.jsonl').read_text().splitlines()]
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 7, 'axes.titlesize': 7.2,
                         'axes.labelsize': 7, 'xtick.labelsize': 6, 'ytick.labelsize': 6,
                         'pdf.fonttype': 42, 'svg.hashsalt': 'velocity-followup-v1'})
    fig, axes = plt.subplots(1, 2, figsize=(4.7, 2.1))
    colors = {'epsilon': '#C45237', 'velocity': '#277C94'}
    for label, report in zip(('epsilon', 'velocity'), numerical):
        for precision in ('fp32', 'bf16'):
            cases = [r for r in report['cases'] if r['precision'] == precision]
            assert len(cases) == 4
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
    assert len(cases) == 64
    before = np.array([r['methods']['epsilon_amended']['candidate_metrics'][0]['Q'] for r in cases])
    after = np.array([r['methods']['velocity_terminal']['candidate_metrics'][0]['Q'] for r in cases])
    axes[1].plot([0, 1], [0, 1], color='#777777', linestyle='--', linewidth=.7)
    axes[1].scatter(before, after, s=10, color=colors['velocity'], alpha=.7, linewidths=.25, edgecolors='white')
    axes[1].set(xlabel='Original amended sampler: error Q', ylabel='Velocity correction: error Q',
                title='B  Check sound repair: 64 V groups', xlim=(0, 1), ylim=(0, 1), xticks=[0, .5, 1], yticks=[0, .5, 1])
    axes[1].text(.04, .94, 'Below line: lower error', transform=axes[1].transAxes, fontsize=6, va='top')
    axes[1].grid(alpha=.18, linewidth=.4)
    fig.subplots_adjust(left=.12, right=.98, top=.87, bottom=.22, wspace=.5)
    for extension in ('pdf', 'png', 'svg'):
        path = ROOT / ('figures/velocity_correction.' + extension)
        metadata = {'CreationDate': None, 'ModDate': None} if extension == 'pdf' else ({'Date': None} if extension == 'svg' else {})
        fig.savefig(path, dpi=450, bbox_inches='tight', pad_inches=.04, metadata=metadata)
        if extension == 'svg':
            path.write_text('\n'.join(line.rstrip() for line in path.read_text().splitlines()) + '\n')
        if extension == 'pdf':
            shutil.copyfile(path, ROOT / 'manuscript' / path.name)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--figure', choices=('all', 'acoustic', 'velocity', 'agents'), default='all')
    args = parser.parse_args()
    # Separate processes preserve each original renderer's Matplotlib defaults.
    if args.figure == 'all':
        for name in ('acoustic', 'velocity', 'agents'):
            subprocess.run([sys.executable, str(Path(__file__).resolve()), '--figure', name],
                           cwd=ROOT, check=True)
    elif args.figure == 'acoustic':
        acoustic_figure()
    elif args.figure == 'velocity':
        velocity_figure()
    else:
        import plot_gpu_agent_benchmark
        plot_gpu_agent_benchmark.main()
    print('Rebuilt', args.figure, 'figure(s) from committed evidence; no inference.')


if __name__ == '__main__':
    main()
