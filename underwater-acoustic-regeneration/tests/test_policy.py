import numpy as np

from acoustic_regeneration.data.observation import Observation
from acoustic_regeneration.diagnostics.features import FEATURE_NAMES
from acoustic_regeneration.diagnostics.policy import generate_pool
from acoustic_regeneration.restoration.classical import linear_interpolation


def test_entire_inference_pool_is_invariant_to_hidden_reference():
    original = np.sin(np.arange(32768) * 0.1)
    mask = np.ones(32768, dtype=bool)
    mask[14000:14800] = False
    alternative = original.copy()
    alternative[~mask] = np.arange((~mask).sum()) * 10000
    calls = []
    def generate(observation, seed):
        calls.append(seed)
        rng = np.random.default_rng(seed)
        return observation.project(rng.normal(size=len(observation.values)))
    kwargs = dict(generator=generate, fallback=linear_interpolation, rms_floor=1e-5,
                  candidate_seeds=(1, 2, 3, 4), probe_seeds=(5, 6))
    first = generate_pool(Observation(original, mask), **kwargs)
    second = generate_pool(Observation(alternative, mask), **kwargs)
    np.testing.assert_array_equal(first.candidates, second.candidates)
    np.testing.assert_array_equal(first.features, second.features)
    assert first.probe_deltas == second.probe_deltas
    assert first.features.shape == (4, len(FEATURE_NAMES))
    assert len(calls) == 12
    assert first.denoiser_calls == 192
