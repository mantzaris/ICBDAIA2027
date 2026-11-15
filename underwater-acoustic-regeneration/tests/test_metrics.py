import numpy as np
import pytest

from acoustic_regeneration.evaluation.metrics import MetricFloors, gap_metrics, magnitude_stft
from acoustic_regeneration.evaluation.statistics import paired_bootstrap, holm_adjust


def test_gap_scores_are_not_diluted_by_observed_samples():
    reference = np.ones(32768)
    prediction = reference.copy()
    prediction[15000:15400] = 0
    result = gap_metrics(prediction, reference, [(15000, 15400)], MetricFloors(1e-6))
    assert result["gap_nmse"] == pytest.approx(1)
    assert result["gap_spectral_convergence"] == pytest.approx(1)
    assert result["Q"] == pytest.approx(0.375)
    assert result["observed_max_drift"] == 0
    assert np.mean((prediction - reference) ** 2) < result["gap_nmse"] / 50


def test_silent_truth_has_finite_scores_without_target_fitted_floor():
    reference = np.zeros(1000)
    prediction = np.ones(1000)
    result = gap_metrics(prediction, reference, [(300, 700)], MetricFloors(0.001))
    assert result["Q"] == 1
    assert result["quiet_reference"]
    assert np.isfinite(list(result.values())).all()


def test_stft_is_periodic_hann_normalized_and_unpadded():
    output = magnitude_stft(np.ones(400), 64)
    assert output.shape == (22, 33)
    assert output[0, 0] == pytest.approx(4)


def test_bootstrap_groups_descendants_before_resampling():
    first = paired_bootstrap(["a", "b"], [1, 3], [0, 0])
    repeated = paired_bootstrap(["a"] * 100 + ["b"], [1] * 100 + [3], [0] * 101)
    assert first == repeated
    np.testing.assert_allclose(holm_adjust([0.01, 0.04]), [0.02, 0.04])
