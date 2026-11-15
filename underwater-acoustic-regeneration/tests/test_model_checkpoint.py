import copy
import random

import numpy as np
import pytest
import torch

from acoustic_regeneration.checkpoint import load_checkpoint, save_checkpoint
from acoustic_regeneration.data.observation import Observation
from acoustic_regeneration.models.diffwave import ModelConfig, WaveformModel
from acoustic_regeneration.restoration.diffusion import cosine_schedule, ddim, ddim_batch, masked_noise_loss, noisy_training_input


def test_architecture_receptive_field_and_projection_training_contract():
    assert ModelConfig().receptive_field_samples == 24571
    reference = torch.randn(2, 512)
    mask = torch.ones_like(reference, dtype=torch.bool)
    mask[:, 100:400] = False
    noise = torch.randn_like(reference)
    state, observed = noisy_training_input(reference, mask, torch.tensor([1, 99]), noise, cosine_schedule())
    assert torch.equal(state[mask], reference[mask])
    assert torch.all(observed[~mask] == 0)
    predicted = noise.clone()
    predicted[mask] += 1000
    assert masked_noise_loss(predicted, noise, mask) == 0


def test_fixed_seed_hidden_event_produces_identical_ddim_output():
    torch.manual_seed(17)
    model = WaveformModel(ModelConfig(4, 2, 2, 16))
    values = np.sin(np.arange(1024) * 0.1)
    mask = np.ones(1024, dtype=bool)
    mask[300:700] = False
    alternative = values.copy()
    alternative[~mask] += 99
    left = ddim(model, Observation(values, mask), 29, steps=4, training_steps=16)
    right = ddim(model, Observation(alternative, mask), 29, steps=4, training_steps=16)
    np.testing.assert_array_equal(left.waveform, right.waveform)
    np.testing.assert_array_equal(left.waveform[mask], values[mask])
    assert left.denoiser_calls == 4


def test_resume_reproduces_next_random_batch_and_optimizer_update(tmp_path):
    torch.manual_seed(43)
    random.seed(43)
    np.random.seed(43)
    model = torch.nn.Linear(3, 1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    metadata = {"git_commit": "test-fixture", "config_hash": "test-fixture", "manifest_hash": "test-fixture"}
    rng = np.random.default_rng(29)
    permutation = rng.permutation(16)

    def update():
        batch = torch.randn(4, 3) + float(np.random.random()) + random.random()
        optimizer.zero_grad()
        model(batch).square().mean().backward()
        optimizer.step()
        return batch, copy.deepcopy(model.state_dict())

    update()
    path = tmp_path / "resume.pt"
    sampler = {"permutation": permutation, "position": 4, "generator_state": rng.bit_generator.state}
    save_checkpoint(path, model, optimizer, update=1, sampler_state=sampler, metadata=metadata, ema=model.state_dict())
    expected_batch, expected_parameters = update()
    restored = load_checkpoint(path, model, optimizer, expected_metadata=metadata)
    actual_batch, actual_parameters = update()
    assert torch.equal(expected_batch, actual_batch)
    assert all(torch.equal(expected_parameters[key], actual_parameters[key]) for key in expected_parameters)
    np.testing.assert_array_equal(restored["sampler"]["permutation"][4:8], permutation[4:8])
    assert restored["sampler"]["position"] == 4
    with pytest.raises(ValueError, match="metadata"):
        load_checkpoint(path, model, optimizer, expected_metadata=dict(metadata, manifest_hash="changed"))


def test_batched_sampling_keeps_each_seed_and_observation_separate():
    torch.manual_seed(17)
    model = WaveformModel(ModelConfig(4, 2, 2, 16))
    mask = np.ones(1024, dtype=bool)
    mask[300:700] = False
    observations = [Observation(np.sin(np.arange(1024) * value), mask) for value in (0.1, 0.2)]
    first = ddim_batch(model, observations, [17, 29], steps=4, training_steps=16)
    second = ddim_batch(model, observations, [17, 29], steps=4, training_steps=16)
    for index in range(2):
        np.testing.assert_array_equal(first[index].waveform, second[index].waveform)
        np.testing.assert_array_equal(first[index].waveform[mask], observations[index].values[mask])
        assert first[index].denoiser_calls == 4


def test_frozen_amplitude_bound_controls_bad_denoising_without_clipping_observations():
    torch.manual_seed(17)
    model = WaveformModel(ModelConfig(4, 2, 2, 1000))
    values = 20 * np.sin(np.arange(1024) * 0.1)
    mask = np.ones(1024, dtype=bool)
    mask[300:700] = False
    observation = Observation(values, mask)
    raw = ddim(model, observation, 17, steps=4)
    bounded = ddim_batch(model, [observation], [17], steps=4, clean_clip=5.0)[0]
    assert np.max(np.abs(raw.waveform[~mask])) > 5.0
    assert np.max(np.abs(bounded.waveform[~mask])) <= 5.0
    np.testing.assert_array_equal(bounded.waveform[mask], values[mask])
    assert bounded.denoiser_calls == 4
