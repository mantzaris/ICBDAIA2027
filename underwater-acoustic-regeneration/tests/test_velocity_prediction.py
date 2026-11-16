"""Check reconstruction algebra, terminal stability, and inference boundaries."""
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from acoustic_regeneration.data.observation import Observation
from acoustic_regeneration.restoration.diffusion import (
    cosine_schedule, ddim, ddim_batch, diffusion_target, masked_noise_loss,
    prediction_components,
)


@pytest.mark.parametrize("alpha", [0.0, 2.428766965e-9, 0.01, 0.5, 1.0])
def test_velocity_recovers_signal_and_noise_even_at_zero_signal_scale(alpha):
    reference = torch.tensor([[0.3, -1.2, 2.1]], dtype=torch.float64)
    noise = torch.tensor([[-0.7, 0.2, 1.4]], dtype=torch.float64)
    schedule = torch.tensor([alpha], dtype=torch.float64)
    state = schedule.sqrt() * reference + (1 - schedule).sqrt() * noise
    velocity = diffusion_target(reference, noise, torch.tensor([0]), schedule, "velocity")
    clean, recovered_noise = prediction_components(state, velocity, schedule[0], "velocity")
    torch.testing.assert_close(clean, reference, atol=1e-12, rtol=1e-12)
    torch.testing.assert_close(recovered_noise, noise, atol=1e-12, rtol=1e-12)


def test_terminal_error_amplification_is_removed_by_velocity_parameterization():
    alpha = cosine_schedule()[-1].double()
    clean = torch.tensor([[0.3, -1.2]], dtype=torch.float64)
    noise = torch.tensor([[-0.7, 0.2]], dtype=torch.float64)
    state = alpha.sqrt() * clean + (1 - alpha).sqrt() * noise
    error = torch.full_like(noise, 0.01)
    velocity = alpha.sqrt() * noise - (1 - alpha).sqrt() * clean
    epsilon_result, _ = prediction_components(state, noise + error, alpha, "epsilon")
    velocity_result, _ = prediction_components(state, velocity + error, alpha, "velocity")
    epsilon_gain = float((epsilon_result - clean).square().mean() / error.square().mean())
    velocity_gain = float((velocity_result - clean).square().mean() / error.square().mean())
    assert epsilon_gain == pytest.approx(float((1 - alpha) / alpha))
    assert epsilon_gain > 1e8
    assert velocity_gain == pytest.approx(float(1 - alpha))
    assert velocity_gain <= 1.0


def test_velocity_loss_has_no_gradient_on_observed_samples():
    prediction = torch.ones(1, 16, requires_grad=True)
    mask = torch.ones_like(prediction, dtype=torch.bool)
    mask[:, 5:9] = False
    loss = masked_noise_loss(prediction, torch.zeros_like(prediction), mask)
    loss.backward()
    assert torch.all(prediction.grad[mask] == 0)
    assert torch.all(prediction.grad[~mask] != 0)


def test_velocity_single_and_batched_sampler_recover_oracle_and_preserve_observations():
    class CleanOracle(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.dummy = torch.nn.Parameter(torch.tensor(0.0))
            self.config = SimpleNamespace(prediction_type="velocity")

        def forward(self, state, time, *args):
            alpha = cosine_schedule()[time].unsqueeze(-1)
            # A known synthetic target for this test, never a production input.
            return (alpha.sqrt() * state - 0.25) / (1 - alpha).sqrt()

    model = CleanOracle()
    signal = 3 * np.sin(np.arange(1024) * 0.1)
    mask = np.ones(1024, dtype=bool)
    mask[300:700] = False
    alternative = signal.copy()
    alternative[~mask] = np.nan
    single = ddim(model, Observation(signal, mask), 17, steps=8)
    batch = ddim_batch(model, [Observation(signal, mask), Observation(alternative, mask)], [17, 17], steps=8)
    for draw in [single] + batch:
        np.testing.assert_allclose(draw.waveform[~mask], 0.25, atol=1e-6)
        np.testing.assert_array_equal(draw.waveform[mask], signal[mask])
        assert draw.denoiser_calls == 8
    np.testing.assert_array_equal(batch[0].waveform, batch[1].waveform)
    np.testing.assert_array_equal(single.waveform, batch[0].waveform)
