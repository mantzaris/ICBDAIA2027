"""Missing-coordinate diffusion and eta=0 DDIM with exact observation projection."""
from __future__ import annotations

from dataclasses import dataclass
from contextlib import nullcontext
from typing import Optional, Sequence

import numpy as np
import torch

from acoustic_regeneration.data.observation import Observation


def cosine_schedule(steps: int = 1000, offset: float = 0.008) -> torch.Tensor:
    if steps < 2:
        raise ValueError("At least two diffusion levels are required")
    grid = torch.linspace(0, steps, steps + 1, dtype=torch.float64) / steps
    cumulative = torch.cos((grid + offset) / (1 + offset) * np.pi / 2) ** 2
    cumulative = cumulative / cumulative[0]
    betas = (1 - cumulative[1:] / cumulative[:-1]).clamp(max=0.999)
    return torch.cumprod(1 - betas, dim=0).float()


def noisy_training_input(reference: torch.Tensor, mask: torch.Tensor, time: torch.Tensor,
                         noise: torch.Tensor, alpha_bar: torch.Tensor):
    if reference.shape != mask.shape or noise.shape != reference.shape or reference.ndim != 2:
        raise ValueError("Reference, binary mask, and noise must have shape [batch, samples]")
    if not torch.all((mask == 0) | (mask == 1)):
        raise ValueError("Mask must be binary")
    observed = torch.where(mask.bool(), reference, torch.zeros_like(reference))
    cumulative = alpha_bar.to(reference.device)[time].unsqueeze(-1)
    noised = cumulative.sqrt() * reference + (1 - cumulative).sqrt() * noise
    return torch.where(mask.bool(), observed, noised), observed


def masked_noise_loss(predicted: torch.Tensor, noise: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    hidden = ~mask.bool()
    if not hidden.any():
        raise ValueError("Noise loss requires missing samples")
    return (predicted.float()[hidden] - noise.float()[hidden]).square().mean()


@dataclass(frozen=True)
class Draw:
    waveform: np.ndarray
    denoiser_calls: int
    seed: int


@torch.no_grad()
def ddim(model: torch.nn.Module, observation: Observation, seed: int,
         steps: int = 32, training_steps: int = 1000, clean_clip: Optional[float] = None,
         initial_timestep: Optional[int] = None) -> Draw:
    initial_timestep = training_steps - 1 if initial_timestep is None else initial_timestep
    if not 0 <= initial_timestep < training_steps or not 1 <= steps <= initial_timestep + 1:
        raise ValueError("DDIM step count must fit the training schedule")
    if clean_clip is not None and (not np.isfinite(clean_clip) or clean_clip <= 0):
        raise ValueError("The G-frozen amplitude bound must be finite and positive")
    if not observation.gaps:
        return Draw(observation.values.copy(), 0, seed)
    device = next(model.parameters()).device
    values = torch.tensor(observation.values.copy(), dtype=torch.float32, device=device).unsqueeze(0)
    mask = torch.tensor(observation.observed.copy(), device=device).unsqueeze(0)
    random = torch.Generator(device=device).manual_seed(seed)
    state = torch.where(mask, values, torch.randn(values.shape, generator=random, device=device))
    schedule = cosine_schedule(training_steps).to(device)
    times = np.linspace(initial_timestep, 0, steps, dtype=int).tolist()
    previous_training = model.training
    model.eval()
    try:
        for index, time in enumerate(times):
            context = torch.autocast(device_type="cuda", dtype=torch.bfloat16) if device.type == "cuda" else nullcontext()
            with context:
                prediction = model(state, torch.tensor([time], device=device), values, mask)
            clean = (state - (1 - schedule[time]).sqrt() * prediction) / schedule[time].sqrt()
            if clean_clip is not None:
                clean = clean.clamp(-clean_clip, clean_clip)
                prediction = (state - schedule[time].sqrt() * clean) / (1 - schedule[time]).sqrt()
            next_time = times[index + 1] if index + 1 < len(times) else -1
            if next_time < 0:
                state = clean
            else:
                alpha_next = schedule[next_time]
                state = alpha_next.sqrt() * clean + (1 - alpha_next).sqrt() * prediction
            state = torch.where(mask, values, state)
    finally:
        model.train(previous_training)
    result = observation.project(state.squeeze(0).cpu().numpy())
    return Draw(result, len(times), seed)


@torch.no_grad()
def ddim_batch(model: torch.nn.Module, observations: Sequence[Observation], seeds: Sequence[int],
               steps: int = 32, training_steps: int = 1000, clean_clip: Optional[float] = None,
               initial_timestep: Optional[int] = None) -> list:
    """Batch independent draws while retaining a separate initial RNG per draw.

    A six-example batch still counts as six denoiser evaluations per reverse
    level; batching does not change the matched scientific computation budget.
    """
    initial_timestep = training_steps - 1 if initial_timestep is None else initial_timestep
    if (not observations or len(observations) != len(seeds) or not 0 <= initial_timestep < training_steps
            or not 1 <= steps <= initial_timestep + 1):
        raise ValueError("Invalid batched sampler configuration")
    if clean_clip is not None and (not np.isfinite(clean_clip) or clean_clip <= 0):
        raise ValueError("The G-frozen amplitude bound must be finite and positive")
    if len({len(item.values) for item in observations}) != 1:
        raise ValueError("Batched observations must have equal lengths")
    device = next(model.parameters()).device
    values = torch.tensor(np.stack([item.values for item in observations]), dtype=torch.float32, device=device)
    mask = torch.tensor(np.stack([item.observed for item in observations]), device=device)
    initial = torch.stack([torch.randn(values.shape[1:], generator=torch.Generator(device=device).manual_seed(seed), device=device)
                           for seed in seeds])
    state = torch.where(mask, values, initial)
    schedule = cosine_schedule(training_steps).to(device)
    times = np.linspace(initial_timestep, 0, steps, dtype=int).tolist()
    previous_training = model.training
    model.eval()
    try:
        for index, time in enumerate(times):
            context = torch.autocast(device_type="cuda", dtype=torch.bfloat16) if device.type == "cuda" else nullcontext()
            with context:
                predicted = model(state, torch.full((len(seeds),), time, device=device), values, mask)
            clean = (state - (1 - schedule[time]).sqrt() * predicted) / schedule[time].sqrt()
            if clean_clip is not None:
                clean = clean.clamp(-clean_clip, clean_clip)
                predicted = (state - schedule[time].sqrt() * clean) / (1 - schedule[time]).sqrt()
            following = times[index + 1] if index + 1 < len(times) else -1
            state = clean if following < 0 else schedule[following].sqrt() * clean + (1 - schedule[following]).sqrt() * predicted
            state = torch.where(mask, values, state)
    finally:
        model.train(previous_training)
    return [Draw(observation.project(state[index].float().cpu().numpy()), steps, seed)
            for index, (observation, seed) in enumerate(zip(observations, seeds))]
