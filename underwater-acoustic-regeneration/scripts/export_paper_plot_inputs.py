"""Export only the three frozen acoustic examples needed to redraw the paper.

This maintenance command needs the local archived waveform/pool files. Normal
figure rebuilding uses the resulting committed files and needs no large archive.
"""
import hashlib
import json
from pathlib import Path

import numpy as np

import build_acoustic_examples as acoustic

ROOT = Path(__file__).resolve().parents[1]


def array_hash(array):
    return hashlib.sha256(np.asarray(array, dtype='<f8').tobytes()).hexdigest()


def main():
    # Reuses the original selection and gap-metric verification, with no inference.
    examples = acoustic.load_examples()
    recorded = json.loads((ROOT / 'figures/acoustic_example_selection.json').read_text())
    assert [e['case_id'] for e, _ in examples] == [e['case_id'] for e in recorded['cases']]
    destination = ROOT / 'figures/data'
    destination.mkdir(parents=True, exist_ok=True)
    arrays, cases = {}, []
    for evidence, signals in examples:
        tag = evidence['tag']
        start, end = evidence['gap_start_sample'], evidence['gap_end_sample_exclusive']
        arrays[tag + '_reference'] = signals['reference']
        for name in ('fallback_B_equals_P', 'selected_C3', 'velocity_first'):
            arrays[tag + '_' + name + '_gap'] = signals[name][start:end]
        cases.append({'evidence': evidence,
                      'signal_sha256_float64_le': {name: array_hash(signal)
                                                  for name, signal in signals.items()}})
    path = destination / 'acoustic_examples.npz'
    np.savez_compressed(path, **arrays)
    manifest = {'scope': 'Lossless plotting inputs for the three previously selected V examples; no new data or inference',
                'array_file': path.name, 'array_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
                'original_renderer': 'scripts/build_acoustic_examples.py',
                'original_renderer_sha256': acoustic.digest(ROOT / 'scripts/build_acoustic_examples.py'),
                'selection_manifest': 'figures/acoustic_example_selection.json',
                'selection_manifest_sha256': acoustic.digest(ROOT / 'figures/acoustic_example_selection.json'),
                'data_attribution': 'provenance/DATA_ATTRIBUTION.md', 'cases': cases}
    (destination / 'acoustic_examples.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(f'Exported three lossless examples: {path.stat().st_size} bytes; no inference.')


if __name__ == '__main__':
    main()
