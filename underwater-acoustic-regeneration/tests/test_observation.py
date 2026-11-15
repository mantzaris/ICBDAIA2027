import numpy as np
import pytest

from acoustic_regeneration.data.observation import Observation, normalize_observation
from acoustic_regeneration.data.preprocessing import crop_source, draw_mask, resample_source
from acoustic_regeneration.diagnostics.probes import make_probes, score_probes
from acoustic_regeneration.restoration.classical import bidirectional_ar, linear_interpolation


def example():
    values = np.sin(np.arange(32768) * 2 * np.pi * 230 / 8000)
    mask = np.ones(len(values), dtype=bool)
    mask[14000:16000] = False
    return values, mask


def test_hidden_reference_and_nan_are_removed_before_normalization():
    values, mask = example()
    altered = values.copy()
    altered[~mask] = np.nan
    first, a = normalize_observation(Observation(values, mask), 1e-5)
    second, b = normalize_observation(Observation(altered, mask), 1e-5)
    np.testing.assert_array_equal(first.values, second.values)
    assert a == b
    assert np.all(first.values[~mask] == 0)


def test_projection_restores_original_observations_bit_for_bit():
    values, mask = example()
    observation = Observation(values, mask)
    normalized, transform = normalize_observation(observation, 1e-5)
    restored = transform.restore(np.ones(len(values), dtype=np.float32) * 100, observation)
    np.testing.assert_array_equal(restored[mask], values[mask])
    assert not observation.values.flags.writeable
    assert not observation.observed.flags.writeable


def test_probe_values_cannot_affect_their_own_input_or_normalization():
    values, mask = example()
    original = Observation(values, mask)
    probe = make_probes(original)[0]
    values[probe.start:probe.end] = 1e6
    modified = make_probes(Observation(values, mask))[0]
    first, transform_a = normalize_observation(probe.observation, 1e-5)
    second, transform_b = normalize_observation(modified.observation, 1e-5)
    np.testing.assert_array_equal(first.values, second.values)
    assert transform_a == transform_b
    assert np.all(~first.observed[14000:16000])


def test_probe_scorer_passes_only_masked_observations():
    values, mask = example()
    seen = []
    def generator(observation, seed):
        assert isinstance(observation, Observation)
        assert len(observation.gaps) == 2
        assert np.all(observation.values[~observation.observed] == 0)
        seen.append(seed)
        return linear_interpolation(observation)
    scores = score_probes(Observation(values, mask), generator, linear_interpolation, 1e-5, (17, 29))
    assert scores == (0.0, 0.0)
    assert seen == [17, 29]


def test_insufficient_context_does_not_invent_probes():
    values, mask = example()
    mask[:] = True
    mask[5:2005] = False
    assert make_probes(Observation(values, mask)) is None


def test_invalid_observation_masks_are_rejected():
    with pytest.raises(ValueError, match="binary"):
        Observation([1, 2, 3], [1, 2, 0])


def test_classical_methods_preserve_context_and_ignore_hidden_event():
    values, mask = example()
    alternative = values.copy()
    alternative[~mask] += 100
    for method in (linear_interpolation, bidirectional_ar):
        result = method(Observation(values, mask))
        np.testing.assert_array_equal(result, method(Observation(alternative, mask)))
        np.testing.assert_array_equal(result[mask], values[mask])
        assert np.isfinite(result).all()


def test_resampling_precedes_simulated_mask_and_rejects_native_gaps():
    source = np.sin(np.arange(160000) * 2 * np.pi * 1000 / 32000)
    prepared = resample_source(source, 32000)
    assert len(prepared) == 40000
    cropped = crop_source(prepared)
    mask = draw_mask(len(cropped), np.random.default_rng(17))
    observed = Observation(cropped, mask)
    np.testing.assert_array_equal(observed.values[mask], prepared[:32768][mask])
    native_mask = np.ones(len(source), dtype=bool)
    native_mask[10000:12000] = False
    with pytest.raises(ValueError, match="filter-support"):
        resample_source(source, 32000, native_observed=native_mask)


def test_mask_draw_is_reproducible_and_has_context():
    left = draw_mask(32768, np.random.default_rng(43))
    right = draw_mask(32768, np.random.default_rng(43))
    np.testing.assert_array_equal(left, right)
    assert left[:6000].all() and left[-6000:].all()
    assert (~left).sum() in {400, 800, 2000, 4000}
