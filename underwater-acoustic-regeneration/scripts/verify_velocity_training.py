"""Compare completed checkpoint states for the matched training intervention."""
import json
from pathlib import Path

import numpy as np
import torch

from acoustic_regeneration.registry import file_hash

ROOT = Path(__file__).resolve().parents[1]


def same(left, right):
    if isinstance(left, torch.Tensor):
        return isinstance(right, torch.Tensor) and torch.equal(left, right)
    if isinstance(left, np.ndarray):
        return isinstance(right, np.ndarray) and np.array_equal(left, right)
    if isinstance(left, dict):
        return left.keys() == right.keys() and all(same(value, right[key]) for key, value in left.items())
    if isinstance(left, (list, tuple)):
        return len(left) == len(right) and all(same(a, b) for a, b in zip(left, right))
    return left == right


def main():
    old = ROOT / 'checkpoints/pilot_conditional_seed17/latest.pt'
    new = ROOT / 'checkpoints/pilot_velocity_seed17/latest.pt'
    checkpoints = [torch.load(path, map_location='cpu', weights_only=False) for path in [old, new]]
    configs = [json.loads((path.parent / 'config.json').read_text()) for path in [old, new]]
    a, b = checkpoints
    assert a['update'] == b['update'] == 5000
    assert a['metadata']['manifest_hash'] == b['metadata']['manifest_hash']
    keys = ['seed', 'batch_size', 'microbatch', 'layers', 'width', 'overfit', 'validation_every', 'validation_cases', 'rms_floor']
    assert all(configs[0][key] == configs[1][key] for key in keys)
    assert configs[0].get('prediction_type', 'epsilon') == 'epsilon'
    assert configs[1]['prediction_type'] == 'velocity'
    assert a['model'].keys() == b['model'].keys()
    assert all(a['model'][key].shape == b['model'][key].shape for key in a['model'])
    assert same(a['sampler']['batch_sampler'], b['sampler']['batch_sampler'])
    assert same(a['rng'], b['rng'])
    assert same(a['optimizer']['param_groups'], b['optimizer']['param_groups'])
    report = {'status': 'passed', 'updates_each': 5000, 'source_manifest_sha256': a['metadata']['manifest_hash'],
              'matching_configuration_fields': keys, 'model_parameter_shapes_match': True,
              'optimizer_hyperparameters_match': True, 'ending_batch_sampler_state_matches': True,
              'ending_python_numpy_torch_cuda_RNG_states_match': True,
              'changed': 'conditional prediction target and its loss weighting; same training random streams and update budget',
              'scope': 'A controlled one-seed comparison; matching final random states is not a multi-seed reproducibility study',
              'script_sha256': file_hash(Path(__file__)),
              'checkpoints': {str(path.relative_to(ROOT)): file_hash(path) for path in [old, new]}}
    (ROOT / 'reports/velocity_training_match.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
