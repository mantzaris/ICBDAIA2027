"""Build anecdotal hydrophone figures and listening clips from frozen V outputs.

Run with PYTHONPATH=src python3 scripts/build_acoustic_examples.py.
No model inference, held-out C/test access, or subjective listening scores.
"""
from __future__ import annotations

import gzip
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
        elif extension == 'pdf':
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
    waveform.text((a + b) / 2, .97, '500 ms\ngap', transform=waveform.get_xaxis_transform(),
                  ha='center', va='top', fontsize=6, bbox={'facecolor': 'white', 'alpha': .8, 'edgecolor': 'none', 'pad': .5})
    waveform.set(xlim=(0, 4.096), xlabel='Time (s)', ylabel='Normalized amplitude', title='Tanker-labeled hydrophone audio')
    times, frequencies, power = stft_power(target, window=1024, hop=128)
    db = 10 * np.log10(np.maximum(power / power.max(), 1e-12))
    spectrum.pcolormesh(times, frequencies, db, vmin=-60, vmax=0, cmap='magma', shading='nearest', rasterized=True)
    for boundary in (a, b):
        spectrum.axvline(boundary, color='white', linestyle='--', linewidth=.6)
    spectrum.set(xlim=(0, 4.096), ylim=(0, 600), yticks=[0, 200, 400, 600],
                 xlabel='Time (s)', ylabel='Frequency (Hz)', title='Reference: low-frequency detail')
    fig.tight_layout(pad=.5, w_pad=.8)
    save(fig, 'acoustic_context')


def comparisons(examples):
    fig, axes = plt.subplots(3, 4, figsize=(4.7, 3.6))
    titles = ['Reference', 'Gap erased', 'Fallback B = P', 'Selected C3']
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
        label = f"{evidence['tag']}  {evidence['source_class'].capitalize()} | {evidence['gap_ms']:.0f} ms | C3 − B: ΔQ = {evidence['C3_minus_B_Q']:+.3f}"
        axes[row, 0].text(0, 1.07, label, transform=axes[row, 0].transAxes, fontsize=6.6, ha='left')
    fig.subplots_adjust(left=.105, right=.97, bottom=.17, top=.88, wspace=.14, hspace=.65)
    fig.text(.54, .085, 'Time from gap start (ms); dashed lines delimit missing samples', ha='center', fontsize=6.5)
    cax = fig.add_axes([.27, .027, .43, .021])
    bar = fig.colorbar(mesh, cax=cax, orientation='horizontal', ticks=[-60, -40, -20, 0])
    bar.ax.tick_params(labelsize=6, pad=1, length=2)
    fig.text(.72, .028, 'dB / ref. peak', fontsize=6, va='center')
    save(fig, 'acoustic_case_studies')


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
              'scope': 'Post hoc acoustic illustration and PCM export, not an additional independent experiment'}
    with (ROOT / 'results/local_run_registry.jsonl').open('a') as stream:
        stream.write(json.dumps(record, sort_keys=True) + '\n')
    print(json.dumps({'cases': [{key: evidence[key] for key in ('tag', 'source_class', 'gap_ms', 'C3_minus_B_Q', 'observed_context_peak_hz', 'gap_rms_db_relative_to_reference')} for evidence, _ in examples],
                      'figures': ['acoustic_context', 'acoustic_case_studies'], 'audio_clips': 12}, indent=2))


if __name__ == '__main__':
    main()
