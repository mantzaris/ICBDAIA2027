"""Build anecdotal hydrophone figures and listening clips from frozen V outputs.

Run with PYTHONPATH=src python3 scripts/build_acoustic_examples.py.
No model inference, held-out C/test access, or subjective listening scores.
"""
from __future__ import annotations

import gzip
import copy
import hashlib
import json
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch, Rectangle
import numpy as np
from scipy.io import wavfile

from acoustic_regeneration.evaluation.metrics import MetricFloors, gap_metrics

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / 'results/pilot_pools_clipped_seed17'
OUT = ROOT / 'figures'
AUDIO = ROOT / 'examples/acoustics'
RATE = 8000
VARIANTS = ('reference', 'masked', 'fallback_B_equals_P', 'selected_C3')
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 7,
                     'axes.titlesize': 7, 'axes.labelsize': 7,
                     'xtick.labelsize': 6, 'ytick.labelsize': 6,
                     'legend.fontsize': 6, 'pdf.fonttype': 42,
                     'ps.fonttype': 42, 'svg.hashsalt': 'acoustic-case-studies-v1',
                     'axes.spines.top': False, 'axes.spines.right': False})


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(fig, name):
    OUT.mkdir(exist_ok=True)
    for extension in ('pdf', 'svg', 'png'):
        path = OUT / f'{name}.{extension}'
        metadata = {'CreationDate': None, 'ModDate': None} if extension == 'pdf' else (
            {'Date': None} if extension == 'svg' else None)
        fig.savefig(path, dpi=450, bbox_inches='tight', pad_inches=.04, metadata=metadata)
        if extension == 'svg':
            path.write_text('\n'.join(line.rstrip() for line in path.read_text().splitlines()) + '\n')
        elif extension == 'pdf' and name in {'operator_configuration', 'operator_decisions', 'acoustic_case_studies'}:
            shutil.copyfile(path, ROOT / 'manuscript' / path.name)
    plt.close(fig)


def stft_power(signal, window=256, hop=32):
    starts = np.arange(0, len(signal) - window + 1, hop)
    frames = np.stack([signal[start:start + window] for start in starts])
    power = np.abs(np.fft.rfft(frames * np.hanning(window + 1)[:-1], axis=-1)) ** 2 / window
    return (starts + window / 2) / RATE, np.fft.rfftfreq(window, 1 / RATE), power.T


def observed_spectrum(case, target):
    """128-ms windows wholly outside the artificial gap; no joined segments."""
    window = 1024
    starts = np.arange(0, len(target) - window + 1, 256)
    starts = starts[(starts + window <= case['gap_start']) | (starts >= case['gap_end'])]
    frames = np.stack([target[start:start + window] for start in starts])
    frames -= frames.mean(axis=1, keepdims=True)
    power = np.mean(np.abs(np.fft.rfft(frames * np.hanning(window + 1)[:-1], axis=-1)) ** 2, axis=0)
    frequencies = np.fft.rfftfreq(window, 1 / RATE)
    band = frequencies >= 20
    flatness = np.exp(np.log(np.maximum(power[band], 1e-20)).mean()) / power[band].mean()
    peak = frequencies[band][np.argmax(power[band])]
    return float(flatness), float(peak)


def load_examples():
    cases = {row['case_id']: row for row in map(json.loads, (RUN / 'cases.jsonl').read_text().splitlines())
             if row['pilot_role'] == 'dev_test'}
    decisions = {row['case_id']: row for row in map(json.loads, (RUN / 'dry_test_decisions.jsonl').read_text().splitlines())
                 if row['policy'] == 'full'}
    assert cases.keys() == decisions.keys() and len(cases) == 64
    with gzip.open(ROOT / 'manifests/pilot_sources.jsonl.gz', 'rt') as stream:
        sources = {row['original_file_id']: row for row in map(json.loads, stream)}
    targets, flatness, peaks = {}, {}, {}
    for identifier, case in cases.items():
        source = sources[case['source_id']]
        assert source['partition'] == case['source_partition'] == 'V'
        path = ROOT / source['processed_path']
        assert digest(path) == source['processed_sha256']
        target = (np.load(path, allow_pickle=False)[:32768].astype(float) - case['normalization_mean']) / case['normalization_scale']
        targets[identifier] = target
        if case['gap_ms'] == 500:
            flatness[identifier], peaks[identifier] = observed_spectrum(case, target)
    delta = {identifier: case['candidate_metrics'][decisions[identifier]['selected_index']]['Q'] - case['fallback_metrics']['Q']
             for identifier, case in cases.items()}
    assert len(flatness) == 17
    ordering = sorted(cases, key=lambda identifier: (delta[identifier], identifier))
    chosen = [ordering[0], ordering[-1], min(flatness, key=lambda identifier: (flatness[identifier], identifier))]
    assert len(set(chosen)) == 3
    rules = ['Minimum C3-minus-B Q among 64 dry-test V cases (post hoc)',
             'Maximum C3-minus-B Q among 64 dry-test V cases (post hoc)',
             'Minimum observed-context spectral flatness among the 17 dry-test V cases with 500-ms gaps (post hoc)']
    floor = json.loads((RUN / 'pool_config.json').read_text())['rms_floor']
    examples = []
    for tag, identifier, rule in zip('ABC', chosen, rules):
        case, decision, target = cases[identifier], decisions[identifier], targets[identifier]
        source = sources[case['source_id']]
        assert not decision['accepted']
        path = ROOT / case['pool_path']
        assert digest(path) == case['pool_sha256']
        start, end = case['gap_start'], case['gap_end']
        signals = {variant: target.copy() for variant in VARIANTS}
        signals['masked'][start:end] = 0
        with np.load(path, allow_pickle=False) as pool:
            signals['fallback_B_equals_P'][start:end] = pool['fallback_gap']
            signals['selected_C3'][start:end] = pool['candidates_gap'][decision['selected_index']]
        observed = np.ones(len(target), dtype=bool)
        observed[start:end] = False
        metrics = {}
        for variant, saved in [('fallback_B_equals_P', case['fallback_metrics']),
                               ('selected_C3', case['candidate_metrics'][decision['selected_index']])]:
            assert np.array_equal(signals[variant][observed], target[observed])
            metrics[variant] = gap_metrics(signals[variant], target, [(start, end)], MetricFloors(floor / case['normalization_scale']))
            for name in ('Q', 'gap_nmse', 'gap_spectral_convergence'):
                assert np.isclose(metrics[variant][name], saved[name], rtol=1e-9, atol=1e-10), (identifier, name)
        reference_rms = np.sqrt(np.mean(target[start:end] ** 2))
        levels = {variant: float(20 * np.log10(np.sqrt(np.mean(signals[variant][start:end] ** 2)) / reference_rms))
                  for variant in ('fallback_B_equals_P', 'selected_C3')}
        evidence = {'tag': tag, 'case_id': identifier, 'source_id': case['source_id'],
                    'source_class': source['class'], 'source_partition': 'V', 'pilot_role': 'dev_test',
                    'timestamp_interval_utc': source['timestamp_interval'], 'hydrophone': source['hydrophone'],
                    'original_archive_member': source['archive_member'], 'selection_rule': rule,
                    'source_array_sha256': source['processed_sha256'], 'pool_sha256': case['pool_sha256'],
                    'normalization_mean': case['normalization_mean'], 'normalization_scale': case['normalization_scale'],
                    'gap_start_sample': start, 'gap_end_sample_exclusive': end,
                    'gap_seconds': [start / RATE, end / RATE], 'gap_ms': case['gap_ms'],
                    'selected_index': decision['selected_index'], 'policy_accepted': False,
                    'C3_minus_B_Q': delta[identifier], 'recomputed_metrics': metrics,
                    'gap_rms_db_relative_to_reference': levels,
                    'observed_context_flatness': flatness.get(identifier),
                    'observed_context_peak_hz': peaks.get(identifier)}
        examples.append((evidence, signals))
    return examples


def overview(example):
    evidence, signals = example
    target = signals['reference']
    a, b = evidence['gap_seconds']
    fig, axes = plt.subplots(1, 2, figsize=(4.7, 1.5), gridspec_kw={'width_ratios': [1, 1.13]})
    waveform, spectrum = axes
    time = np.arange(len(target)) / RATE
    waveform.plot(time, target, color='#78838b', linewidth=.35, label='Complete reference')
    known = target.copy()
    known[evidence['gap_start_sample']:evidence['gap_end_sample_exclusive']] = np.nan
    waveform.plot(time, known, color='#146D88', linewidth=.35)
    waveform.axvspan(a, b, facecolor='#E8C66A', alpha=.45)
    waveform.text((a + b) / 2, .97, 'Missing\n0.5 s', transform=waveform.get_xaxis_transform(),
                  ha='center', va='top', fontsize=6, bbox={'facecolor': 'white', 'alpha': .8, 'edgecolor': 'none', 'pad': .5})
    waveform.set(xlim=(0, 4.096), xlabel='Time (s)', ylabel='Relative amplitude', title='Recorded sound with a test gap')
    times, frequencies, power = stft_power(target, window=1024, hop=128)
    db = 10 * np.log10(np.maximum(power / power.max(), 1e-12))
    spectrum.pcolormesh(times, frequencies, db, vmin=-60, vmax=0, cmap='magma', shading='nearest', rasterized=True)
    for boundary in (a, b):
        spectrum.axvline(boundary, color='white', linestyle='--', linewidth=.6)
    spectrum.annotate('Low tone near 78 Hz', xy=(3.1, 78), xytext=(2.55, 450),
                      color='white', fontsize=6, ha='center',
                      arrowprops={'arrowstyle': '->', 'color': 'white', 'lw': .7})
    spectrum.set(xlim=(0, 4.096), ylim=(0, 600), yticks=[0, 200, 400, 600],
                 xlabel='Time (s)', ylabel='Frequency (Hz)', title='Frequency view: brighter = stronger')
    fig.tight_layout(pad=.5, w_pad=.8)
    save(fig, 'acoustic_context')


def comparisons(examples):
    fig, axes = plt.subplots(3, 4, figsize=(4.7, 3.6))
    titles = ['Complete audio\nTest reference', 'Available audio\nGap erased',
              'Baseline fill\nUsed by rule', 'AI proposal\nDeclined by rule']
    for row, (evidence, signals) in enumerate(examples):
        a, b = evidence['gap_seconds']
        left, right = a - .12, b + .12
        all_spectra = {variant: stft_power(signal) for variant, signal in signals.items()}
        reference_times, _, reference_power = all_spectra['reference']
        reference_peak = reference_power[:, (reference_times >= left) & (reference_times <= right)].max()
        for col, variant in enumerate(VARIANTS):
            ax = axes[row, col]
            times, frequencies, power = all_spectra[variant]
            db = 10 * np.log10(np.maximum(power / reference_peak, 1e-12))
            mesh = ax.pcolormesh((times - a) * 1000, frequencies / 1000, db,
                                 vmin=-60, vmax=6, cmap='magma', shading='nearest', rasterized=True)
            for boundary in (0, (b - a) * 1000):
                ax.axvline(boundary, color='white', linestyle='--', linewidth=.55)
            ax.set(xlim=(-120, (b - a) * 1000 + 120), ylim=(0, 1 if row == 2 else 4),
                   xticks=[0, 500] if row == 2 else [-100, 0, 100])
            if col == 0:
                ax.set_ylabel('Frequency (kHz)')
            else:
                ax.tick_params(labelleft=False)
            if row == 0:
                ax.set_title(titles[col], pad=17)
        descriptions = ['Small score gain, changed frequency pattern',
                        'Extra energy inside the gap', 'Low tone weakens across the gap']
        label = f"{evidence['tag']}  {evidence['source_class'].capitalize()} | {evidence['gap_ms']:.0f} ms | {descriptions[row]}"
        axes[row, 0].text(0, 1.07, label, transform=axes[row, 0].transAxes, fontsize=6.6, ha='left')
    fig.subplots_adjust(left=.105, right=.97, bottom=.17, top=.88, wspace=.14, hspace=.65)
    fig.text(.54, .085, 'Time from gap start (ms); dashed lines mark the missing input interval', ha='center', fontsize=6.3)
    cax = fig.add_axes([.27, .027, .43, .021])
    bar = fig.colorbar(mesh, cax=cax, orientation='horizontal', ticks=[-60, -40, -20, 0])
    bar.ax.tick_params(labelsize=6, pad=1, length=2)
    fig.text(.72, .028, 'dB / ref. peak', fontsize=6, va='center')
    save(fig, 'acoustic_case_studies')


def configuration_views(example):
    """Demonstrate display settings on measured samples only, with gap frames excluded."""
    evidence, signals = example
    available = signals['masked']
    gap_start, gap_end = evidence['gap_start_sample'], evidence['gap_end_sample_exclusive']
    settings, spectra = [], []
    for window, hop in [(256, 32), (1024, 128)]:
        times, frequencies, power = stft_power(available, window, hop)
        starts = np.arange(0, len(available) - window + 1, hop)
        touches_gap = (starts < gap_end) & (starts + window > gap_start)
        # One-sided periodogram density; color is relative to a shared observed peak.
        hann_energy = np.sum(np.hanning(window + 1)[:-1] ** 2)
        power *= window / (RATE * hann_energy)
        power[1:-1] *= 2
        power[:, touches_gap] = np.nan
        assert np.isfinite(power[:, ~touches_gap]).all()
        spectra.append((times, frequencies, power))
        settings.append({'window_samples': window, 'window_ms': window / RATE * 1000,
                         'hop_samples': hop, 'hop_ms': hop / RATE * 1000,
                         'bin_spacing_hz': RATE / window,
                         'frames_touching_missing_samples_excluded': int(touches_gap.sum()),
                         'display_frequency_hz': [0, 600]})
    peak = max(np.nanmax(power) for _, _, power in spectra)
    fig, axes = plt.subplots(1, 2, figsize=(4.7, 2.05))
    cmap = copy.copy(plt.get_cmap('magma'))
    cmap.set_bad('#DDDDDD')
    for index, (ax, (times, frequencies, power), setting) in enumerate(zip(axes, spectra, settings)):
        db = 10 * np.log10(np.maximum(power / peak, 1e-12))
        ax.pcolormesh(times, frequencies, np.ma.masked_invalid(db), vmin=-60, vmax=0,
                      cmap=cmap, shading='nearest', rasterized=True)
        ax.set(xlim=(0, 4.096), ylim=(0, 600), yticks=[0, 200, 400, 600],
               xlabel='Time (s)', ylabel='Frequency (Hz)' if index == 0 else '')
        ax.set_title(f"{setting['window_ms']:.0f} ms window: {setting['bin_spacing_hz']:.2f} Hz bins", fontsize=7)
        ax.text((gap_start + gap_end) / (2 * RATE), 500, 'Gap', ha='center', fontsize=6, color='#333333')
        if index == 1:
            ax.annotate('Band near 78 Hz', xy=(3.05, 78), xytext=(2.85, 400),
                        color='white', fontsize=6, ha='center',
                        arrowprops={'arrowstyle': '->', 'color': 'white', 'lw': .7})
    fig.subplots_adjust(left=.10, right=.98, top=.89, bottom=.38, wspace=.25)
    fig.text(.5, .19, 'Recorded cue: low-frequency band. Dataset annotation: tanker.', ha='center', fontsize=6.6)
    fig.text(.5, .11, 'Object identity was not predicted; these are display settings, not hardware commands.', ha='center', fontsize=6.1)
    fig.text(.5, .015, 'Proposed equipment loop: observe → label / unknown → recommend → approve / configure → remeasure', ha='center', fontsize=5.7)
    save(fig, 'operator_configuration')
    report = {'scope': 'Offline analysis-setting demonstration on one existing V recording; no physical equipment control or identification model',
              'script_sha256': digest(Path(__file__)), 'case_id': evidence['case_id'],
              'source_array_sha256': evidence['source_array_sha256'],
              'dataset_annotation': evidence['source_class'], 'annotation_is_model_prediction': False,
              'observed_feature': {'description': 'dominant observed-context component',
                                   'frequency_hz': evidence['observed_context_peak_hz']},
              'input': 'Available observed-normalized samples; the artificial gap is zeroed before analysis',
              'sample_rate_hz': RATE, 'settings': settings,
              'display_power': 'One-sided periodic-Hann periodogram density; -60 to 0 dB relative to the common peak over fully observed frames',
              'missing_frame_policy': 'Every analysis frame touching the gap is excluded from both views',
              'identification_evaluated': False, 'hardware_configuration_evaluated': False,
              'agent_workflow_status': 'Proposed only', 'new_inference_calls': 0}
    (OUT / 'operator_configuration.json').write_text(json.dumps(report, indent=2) + '\n')


def operator_decisions():
    """Show the same 64 frozen cases under always-use and calibrated policies."""
    cases = {row['case_id']: row for row in map(json.loads, (RUN / 'cases.jsonl').read_text().splitlines())
             if row['pilot_role'] == 'dev_test'}
    decisions = {row['case_id']: row for row in map(json.loads, (RUN / 'dry_test_decisions.jsonl').read_text().splitlines())
                 if row['policy'] == 'full'}
    calibration = json.loads((RUN / 'calibration.json').read_text())['policies']['full']
    assert cases.keys() == decisions.keys() and len(cases) == 64
    assert calibration['threshold'] is None
    rows = []
    for identifier in sorted(cases):
        case, decision = cases[identifier], decisions[identifier]
        assert case['source_partition'] == 'V' and not decision['accepted']
        baseline_q = case['fallback_metrics']['Q']
        proposal_q = case['candidate_metrics'][decision['selected_index']]['Q']
        rows.append({'case_id': identifier, 'baseline_Q': baseline_q, 'selected_proposal_Q': proposal_q,
                     'worse_than_baseline': bool(proposal_q > baseline_q + .02),
                     'checked_policy_uses_AI': bool(decision['accepted'])})
    worse = sum(row['worse_than_baseline'] for row in rows)
    assert worse == 52
    colors = {'worse': '#C45237', 'other': '#D8D8D8', 'baseline': '#277C94'}
    fig, axes = plt.subplots(1, 2, figsize=(4.7, 2.25))
    for panel, ax in enumerate(axes):
        ax.set(xlim=(-.15, 8.15), ylim=(-.15, 8.15), aspect='equal')
        ax.axis('off')
        for index, row in enumerate(rows):
            x, y = index % 8, 7 - index // 8
            kind = ('worse' if row['worse_than_baseline'] else 'other') if panel == 0 else 'baseline'
            ax.add_patch(Rectangle((x + .06, y + .06), .86, .86, facecolor=colors[kind], edgecolor='white', linewidth=.35))
            if kind == 'worse':
                ax.plot([x + .3, x + .68], [y + .3, y + .68], color='white', linewidth=.6)
                ax.plot([x + .3, x + .68], [y + .68, y + .3], color='white', linewidth=.6)
            elif kind == 'baseline':
                ax.plot([x + .28, x + .7], [y + .49, y + .49], color='white', linewidth=.65)
        ax.set_title(['Always use the AI proposal', 'Apply the release rule'][panel], fontsize=7.5, pad=4)
        ax.text(.5, -.06, ['52 of 64 fills worse than baseline', '0 proposals used; 64 baseline fills'][panel],
                transform=ax.transAxes, ha='center', va='top', fontsize=7,
                color=colors['worse' if panel == 0 else 'baseline'], fontweight='bold')
    fig.subplots_adjust(left=.04, right=.98, top=.86, bottom=.28, wspace=.16)
    handles = [Patch(facecolor=colors[key], label=label) for key, label in
               [('worse', 'Worse reconstruction'), ('other', 'Within tolerance'), ('baseline', 'Baseline fill')]]
    fig.legend(handles=handles, loc='lower center', bbox_to_anchor=(.5, .04), ncol=3, frameon=False,
               fontsize=6.1, handlelength=.9, columnspacing=1)
    fig.text(.5, .015, 'Each square is one recording. Baseline fill is estimated audio too.', ha='center', fontsize=6.3)
    save(fig, 'operator_decisions')
    report = {'scope': 'Retrospective display of 64 existing dry-test V groups; no operator trial or new inference',
              'script_sha256': digest(Path(__file__)), 'cases': 64, 'always_use_AI': 64,
              'always_use_worse_than_baseline': worse, 'checked_policy_AI_used': 0,
              'checked_policy_baseline_used': 64, 'harm_margin_Q': .02,
              'calibrated_threshold': calibration['threshold'],
              'threshold_interpretation': 'No candidate threshold passed dry calibration; the release gate is disabled',
              'baseline_is_still_estimated': True, 'tile_order': 'case_id ascending, row-major from top left; same order in both panels',
              'source_files': {str(path.relative_to(ROOT)): digest(path) for path in
                               [RUN / 'cases.jsonl', RUN / 'dry_test_decisions.jsonl', RUN / 'calibration.json']},
              'rows': rows}
    (OUT / 'operator_decisions.json').write_text(json.dumps(report, indent=2) + '\n')


def export_audio(examples):
    AUDIO.mkdir(parents=True, exist_ok=True)
    for evidence, signals in examples:
        gain = .8 / max(float(np.max(np.abs(signal))) for signal in signals.values())
        observed = np.ones(32768, dtype=bool)
        observed[evidence['gap_start_sample']:evidence['gap_end_sample_exclusive']] = False
        reference_pcm = np.rint(signals['reference'] * gain * 32767).astype('<i2')
        evidence['audio_gain_after_observed_normalization'] = gain
        evidence['audio_files'] = {}
        for variant, signal in signals.items():
            assert np.max(np.abs(signal * gain)) <= .800001
            pcm = np.rint(signal * gain * 32767).astype('<i2')
            path = AUDIO / f"{evidence['tag']}_{evidence['source_class']}_{variant}.wav"
            wavfile.write(path, RATE, pcm)
            rate, reread = wavfile.read(path)
            assert rate == RATE and np.array_equal(pcm, reread)
            assert np.array_equal(reread[observed], reference_pcm[observed])
            if variant == 'masked':
                assert not reread[~observed].any()
            evidence['audio_files'][variant] = {'path': str(path.relative_to(ROOT)), 'sha256': digest(path)}


def main():
    examples = load_examples()
    overview(examples[2])
    comparisons(examples)
    configuration_views(examples[2])
    operator_decisions()
    export_audio(examples)
    report = {'scope': 'Post hoc anecdotes from three existing dry-test V cases; no new inference or held-out outcomes',
              'source_run': 'results/pilot_pools_clipped_seed17', 'script_sha256': digest(Path(__file__)),
              'sample_rate': RATE, 'samples_per_clip': 32768, 'audio_format': '16-bit PCM WAV',
              'audio_scaling': 'Observed mean/RMS normalization, then one shared gain per case across all four variants; maximum peak 0.8; no independent loudness normalization, fading, or time stretching',
              'amplitude_interpretation': 'Relative signal units, not calibrated Pa or dB re 1 micropascal',
              'comparison_stft': {'window': 'periodic Hann', 'samples': 256, 'hop_samples': 32,
                                  'center_padding': False, 'frame_centers': True,
                                  'display_db': [-60, 6], 'reference': 'Peak reference STFT power in the displayed time interval, shared across each row',
                                  'display_khz_by_row': [4, 4, 1]},
              'overview_stft': {'window': 'periodic Hann', 'samples': 1024, 'hop_samples': 128,
                                'display_hz': [0, 600], 'display_db': [-60, 0]},
              'selection_flatness': {'window_samples': 1024, 'hop_samples': 256, 'band_hz': [20, 4000],
                                     'per_frame_mean_removed': True, 'fully_observed_windows_only': True,
                                     'eligible_500ms_cases': 17},
              'verification': {'source_and_pool_hashes': True, 'saved_metrics_recomputed': True,
                               'observed_samples_preserved_before_and_after_pcm_export': True,
                               'masked_pcm_gap_is_zero': True, 'no_audio_clipping': True},
              'not_claimed': ['Representative examples', 'Auditory quality scores', 'Physical source attribution of individual tones', 'Acoustic pressure calibration'],
              'cases': [evidence for evidence, _ in examples]}
    manifest = OUT / 'acoustic_example_selection.json'
    manifest.write_text(json.dumps(report, indent=2) + '\n')
    record = {'run_id': 'acoustic_example_presentation', 'status': 'completed',
              'recorded_utc': datetime.now(timezone.utc).isoformat(), 'command': sys.argv,
              'git_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
              'working_tree_dirty': bool(subprocess.check_output(['git', 'status', '--porcelain', '--', '.'], cwd=ROOT, text=True).strip()),
              'script_sha256': report['script_sha256'],
              'source_identity_note': 'Base HEAD plus exact script hash; presentation derived from existing frozen V outputs',
              'unique_existing_cases': 3, 'audio_files': 12, 'new_model_inference_calls': 0,
              'actual_C_test_opened': False, 'output_manifest': str(manifest.relative_to(ROOT)),
              'output_manifest_sha256': digest(manifest),
              'operator_plot_manifest': 'figures/operator_decisions.json',
              'operator_plot_manifest_sha256': digest(OUT / 'operator_decisions.json'),
              'configuration_plot_manifest': 'figures/operator_configuration.json',
              'configuration_plot_manifest_sha256': digest(OUT / 'operator_configuration.json'),
              'scope': 'Post hoc acoustic illustration and PCM export, not an additional independent experiment'}
    with (ROOT / 'results/local_run_registry.jsonl').open('a') as stream:
        stream.write(json.dumps(record, sort_keys=True) + '\n')
    print(json.dumps({'cases': [{key: evidence[key] for key in ('tag', 'source_class', 'gap_ms', 'C3_minus_B_Q', 'observed_context_peak_hz', 'gap_rms_db_relative_to_reference')} for evidence, _ in examples],
                      'figures': ['acoustic_context', 'acoustic_case_studies', 'operator_configuration', 'operator_decisions'], 'audio_clips': 12}, indent=2))


if __name__ == '__main__':
    main()
