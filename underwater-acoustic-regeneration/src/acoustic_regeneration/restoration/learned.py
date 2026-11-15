"""Trusted project checkpoint loading and deterministic observation-only inference."""
from __future__ import annotations

import inspect
import json
from pathlib import Path

import torch

from acoustic_regeneration.data.observation import Observation
from acoustic_regeneration.models.diffwave import ModelConfig, WaveformModel


def load_ema_model(checkpoint: Path, device: str = "cuda"):
    checkpoint = Path(checkpoint)
    config = json.loads((checkpoint.parent / "config.json").read_text())
    kwargs = {"map_location": "cpu"}
    if "weights_only" in inspect.signature(torch.load).parameters:
        kwargs["weights_only"] = False
    payload = torch.load(checkpoint, **kwargs)
    model = WaveformModel(ModelConfig(**config["model"]))
    model.load_state_dict(payload["ema"])
    model.to(device).eval().requires_grad_(False)
    return model, config, payload["metadata"]


@torch.no_grad()
def deterministic_restore(model: WaveformModel, observation: Observation):
    device = next(model.parameters()).device
    values = torch.tensor(observation.values.copy(), dtype=torch.float32, device=device).unsqueeze(0)
    mask = torch.tensor(observation.observed.copy(), device=device).unsqueeze(0)
    times = torch.zeros(1, dtype=torch.long, device=device)
    if device.type == "cuda":
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            result = model(values, times, values, mask)
    else:
        result = model(values, times, values, mask)
    return observation.project(result.squeeze(0).float().cpu().numpy())
