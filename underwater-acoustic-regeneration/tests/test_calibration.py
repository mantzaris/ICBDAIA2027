import numpy as np
import pytest

from acoustic_regeneration.diagnostics.calibration import CalibrationCase, calibrate, freeze_thresholds, harm_upper_bound


def cases(count, harmful=False):
    return [CalibrationCase(str(i), -0.2, 0.5 if harmful else 0.1, 0.2) for i in range(count)]


def test_zero_error_bound_matches_closed_form_and_multiplicity():
    assert harm_upper_bound(0, 100, 0.05, 9) == pytest.approx(1 - (0.05 / 9) ** (1 / 100))
    assert harm_upper_bound(0, 100, 0.05, 9) > harm_upper_bound(0, 100, 0.05, 1)
    assert harm_upper_bound(0, 0, 0.05, 9) == 1
    assert harm_upper_bound(100, 100, 0.05, 9) == 1


def test_no_certified_threshold_returns_all_fallback():
    result = calibrate(cases(10), [-0.3, -0.2, 0])
    assert result.threshold is None
    assert not result.accepts(-100)
    assert calibrate(cases(200, True), [-0.2]).threshold is None


def test_ties_choose_stricter_threshold_and_invalid_inputs_reject():
    result = calibrate(cases(200), [-0.2, -0.1, 0])
    assert result.threshold == -0.2
    assert result.accepts(-0.2)
    assert not result.accepts(-0.3, valid=False)
    assert not result.accepts(np.nan)


def test_calibration_rejects_repeated_groups_and_postive_threshold():
    with pytest.raises(ValueError, match="one case"):
        calibrate(cases(10) * 2, [-0.2])
    with pytest.raises(ValueError, match="nonpositive"):
        calibrate(cases(10), [0.1])


def test_quantiles_are_validation_only_nonpositive_and_unique():
    thresholds = freeze_thresholds([-1, -0.5, 0.2, 0.4])
    assert all(value <= 0 for value in thresholds)
    assert len(thresholds) <= 9
    assert len(set(thresholds)) == len(thresholds)
    assert freeze_thresholds([float("nan")]) == ()
