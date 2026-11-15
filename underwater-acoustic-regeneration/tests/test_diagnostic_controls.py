import importlib.util
from pathlib import Path

import numpy as np
import pytest
import torch

from acoustic_regeneration.data.observation import Observation
from acoustic_regeneration.diagnostics.policy import generate_batched_pool, selective_restore
from acoustic_regeneration.diagnostics.calibration import calibrate
from acoustic_regeneration.diagnostics.predictor import fit_predictor
from acoustic_regeneration.models.diffwave import ModelConfig, WaveformModel
from acoustic_regeneration.restoration.classical import linear_interpolation


def test_batched_probe_pipeline_cannot_distinguish_hidden_events():
    torch.manual_seed(17)
    model = WaveformModel(ModelConfig(4, 2, 2))
    waveform = np.sin(np.arange(32768) * 0.13)
    mask = np.ones(32768, dtype=bool)
    mask[14000:14800] = False
    altered = waveform.copy()
    altered[~mask] = np.nan
    options = dict(model=model, fallback=linear_interpolation, rms_floor=1e-5,
                   candidate_seeds=(1, 2, 3, 4), probe_seeds=(5, 6), steps_per_draw=4)
    original = generate_batched_pool(Observation(waveform, mask), **options)
    hidden_event = generate_batched_pool(Observation(altered, mask), **options)
    np.testing.assert_array_equal(original.candidates, hidden_event.candidates)
    np.testing.assert_array_equal(original.features, hidden_event.features)
    assert original.denoiser_calls == 24


def test_diagnostic_fitting_refuses_calibration_or_test_targets():
    for partition in ("G", "V", "C", "test"):
        with pytest.raises(ValueError, match="only on finite D"):
            fit_predictor(np.zeros((40, 18)), np.zeros(40), [str(i) for i in range(40)], [partition] * 40, 17)


def test_duration_gate_uses_full_selector_not_candidate_zero():
    path = Path(__file__).resolve().parents[1] / "scripts/analyze_pilot.py"
    spec = importlib.util.spec_from_file_location("pilot_analysis_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    class Predictor:
        def predict(self, matrix):
            return matrix[:, 0]
    predictors = {"full": {"model": Predictor(), "indices": [1]},
                  "duration_only": {"model": Predictor(), "indices": [0]}}
    matrix = np.zeros((4, 18))
    matrix[:, 0] = 0.5
    matrix[:, 1] = [0.1, 0.05, -0.2, 0.3]
    index, score = module.selected(predictors, "duration_only", matrix)
    assert index == 2
    assert score == 0.5


def test_invalid_generation_returns_fallback_and_counts_failed_compute():
    class InvalidModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.dummy = torch.nn.Parameter(torch.tensor(0.0))
        def forward(self, waveform, *args):
            return torch.full_like(waveform, float("nan"))
    waveform = np.sin(np.arange(32768) * 0.13)
    mask = np.ones(32768, dtype=bool)
    mask[14000:14800] = False
    observation = Observation(waveform, mask)
    result = selective_restore(observation, InvalidModel(), linear_interpolation, {}, calibrate([], [0]),
                               1e-5, (1, 2, 3, 4), (5, 6), steps_per_draw=4)
    assert not result["accepted"]
    assert result["fallback_reason"] == "nonfinite_generation"
    assert result["denoiser_calls"] == 24
    np.testing.assert_allclose(result["waveform"], linear_interpolation(observation), atol=1e-14)
    np.testing.assert_array_equal(result["imputed_mask"], ~mask)
