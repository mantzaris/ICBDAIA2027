# Copyright 2020 LMNT, Inc. All Rights Reserved.
# Licensed under the Apache License, Version 2.0.
# https://www.apache.org/licenses/LICENSE-2.0
# Distributed on an AS IS basis, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND.
# See provenance/licenses/diffwave-Apache-2.0.txt for the full license.
#
# Modified for this study: sample-aligned waveform/mask conditioning replaces
# the mel upsampler; typed configuration and an explicit deterministic mode added.
# Upstream: lmnt-com/diffwave@0594106093b8d8c444de8bd8cd26482f653c569f.
from __future__ import annotations

from dataclasses import dataclass
from math import sqrt
from typing import Optional

import torch
from torch import nn
from torch.nn import functional as F


@dataclass(frozen=True)
class ModelConfig:
    residual_channels: int = 64
    residual_layers: int = 36
    dilation_cycle_length: int = 12
    diffusion_steps: int = 1000
    mode: str = "conditional"

    def __post_init__(self) -> None:
        if min(self.residual_channels, self.residual_layers, self.dilation_cycle_length, self.diffusion_steps) <= 0:
            raise ValueError("Model dimensions must be positive")
        if self.mode not in {"conditional", "unconditional", "deterministic"}:
            raise ValueError("Unknown model mode")

    @property
    def receptive_field_samples(self) -> int:
        return 1 + 2 * sum(2 ** (layer % self.dilation_cycle_length) for layer in range(self.residual_layers))


def _conv(input_channels: int, output_channels: int, kernel: int, **kwargs) -> nn.Conv1d:
    layer = nn.Conv1d(input_channels, output_channels, kernel, **kwargs)
    nn.init.kaiming_normal_(layer.weight)
    return layer


class DiffusionEmbedding(nn.Module):
    def __init__(self, steps: int):
        super().__init__()
        phases = torch.arange(steps).unsqueeze(1) * 10.0 ** (torch.arange(64).unsqueeze(0) * 4.0 / 63.0)
        self.register_buffer("table", torch.cat([torch.sin(phases), torch.cos(phases)], dim=1), persistent=False)
        self.projection1 = nn.Linear(128, 512)
        self.projection2 = nn.Linear(512, 512)

    def forward(self, time: torch.Tensor) -> torch.Tensor:
        return F.silu(self.projection2(F.silu(self.projection1(self.table[time.long()]))))


class ResidualBlock(nn.Module):
    def __init__(self, channels: int, dilation: int, conditional: bool):
        super().__init__()
        self.dilated = _conv(channels, channels * 2, 3, padding=dilation, dilation=dilation)
        self.time_projection = nn.Linear(512, channels)
        self.condition_projection = _conv(2, channels * 2, 1) if conditional else None
        self.output_projection = _conv(channels, channels * 2, 1)

    def forward(self, state: torch.Tensor, time: torch.Tensor, condition: Optional[torch.Tensor]):
        hidden = self.dilated(state + self.time_projection(time).unsqueeze(-1))
        if self.condition_projection is not None:
            hidden = hidden + self.condition_projection(condition)
        gate, values = hidden.chunk(2, dim=1)
        hidden = torch.sigmoid(gate) * torch.tanh(values)
        residual, skip = self.output_projection(hidden).chunk(2, dim=1)
        return (state + residual) / sqrt(2), skip


class WaveformModel(nn.Module):
    def __init__(self, config: ModelConfig = ModelConfig()):
        super().__init__()
        self.config = config
        channels = config.residual_channels
        self.input_projection = _conv(1, channels, 1)
        self.embedding = DiffusionEmbedding(config.diffusion_steps)
        self.blocks = nn.ModuleList(ResidualBlock(channels, 2 ** (index % config.dilation_cycle_length),
                                                 config.mode != "unconditional")
                                    for index in range(config.residual_layers))
        self.skip_projection = _conv(channels, channels, 1)
        self.output_projection = _conv(channels, 1, 1)
        nn.init.zeros_(self.output_projection.weight)

    def forward(self, waveform: torch.Tensor, time: torch.Tensor,
                observed: Optional[torch.Tensor] = None, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        if waveform.ndim != 2:
            raise ValueError("Waveform shape must be [batch, samples]")
        if self.config.mode == "unconditional":
            if observed is not None or mask is not None:
                raise ValueError("The unconditional prior must not receive observations")
            condition = None
        else:
            if observed is None or mask is None or observed.shape != waveform.shape or mask.shape != waveform.shape:
                raise ValueError("Sample-aligned observations and mask are required")
            observed = torch.where(mask.bool(), observed, torch.zeros_like(observed))
            condition = torch.stack([observed, mask.to(observed.dtype)], dim=1)
        if self.config.mode == "deterministic":
            waveform = observed
            time = torch.zeros(waveform.shape[0], dtype=torch.long, device=waveform.device)
        hidden = F.relu(self.input_projection(waveform.unsqueeze(1)))
        embedding = self.embedding(time)
        skips = None
        for block in self.blocks:
            hidden, skip = block(hidden, embedding, condition)
            skips = skip if skips is None else skips + skip
        hidden = skips / sqrt(len(self.blocks))
        return self.output_projection(F.relu(self.skip_projection(hidden))).squeeze(1)

    @property
    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())
