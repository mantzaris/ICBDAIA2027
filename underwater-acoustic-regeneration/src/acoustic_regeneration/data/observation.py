"""Inference boundary: unavailable samples are erased, never merely flagged."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np


@dataclass(frozen=True)
class Observation:
    values: np.ndarray
    observed: np.ndarray
    sample_rate: int = 8000

    def __post_init__(self) -> None:
        values = np.asarray(self.values, dtype=np.float64)
        mask = np.asarray(self.observed)
        if values.ndim != 1 or mask.shape != values.shape:
            raise ValueError("Expected matching one-dimensional waveform and mask")
        if not np.isin(mask, [0, 1]).all():
            raise ValueError("Observed mask must be binary")
        mask = mask.astype(bool, copy=True)
        if not mask.any() or self.sample_rate <= 0:
            raise ValueError("At least one observed sample and a positive sample rate are required")
        if not np.isfinite(values[mask]).all():
            raise ValueError("Observed samples must be finite")
        # Multiplication by zero does not remove NaN, infinity, or hidden secrets.
        values = np.where(mask, values, 0.0)
        values.setflags(write=False)
        mask.setflags(write=False)
        object.__setattr__(self, "values", values)
        object.__setattr__(self, "observed", mask)

    @property
    def gaps(self) -> Tuple[Tuple[int, int], ...]:
        edges = np.diff(np.r_[False, ~self.observed, False].astype(np.int8))
        return tuple(zip(np.flatnonzero(edges == 1).tolist(), np.flatnonzero(edges == -1).tolist()))

    def hide(self, start: int, end: int) -> Observation:
        if not 0 <= start < end <= len(self.values):
            raise ValueError("Invalid interval")
        mask = self.observed.copy()
        mask[start:end] = False
        return Observation(self.values, mask, self.sample_rate)

    def project(self, proposed: np.ndarray) -> np.ndarray:
        proposed = np.asarray(proposed, dtype=np.float64)
        if proposed.shape != self.values.shape:
            raise ValueError("Proposal shape does not match observation")
        if not np.isfinite(proposed[~self.observed]).all():
            raise ValueError("Nonfinite completion")
        return np.where(self.observed, self.values, proposed)


@dataclass(frozen=True)
class Normalization:
    mean: float
    scale: float

    def normalize_reference_for_scoring(self, reference: np.ndarray) -> np.ndarray:
        """Evaluation/training only; the transform was fitted without the target."""
        return (np.asarray(reference, dtype=np.float64) - self.mean) / self.scale

    def restore(self, normalized: np.ndarray, original: Observation) -> np.ndarray:
        return original.project(np.asarray(normalized) * self.scale + self.mean)


def normalize_observation(observation: Observation, rms_floor: float) -> Tuple[Observation, Normalization]:
    if not np.isfinite(rms_floor) or rms_floor <= 0:
        raise ValueError("Supply a positive RMS floor frozen from G")
    samples = observation.values[observation.observed]
    mean = float(samples.mean())
    scale = max(float(np.sqrt(np.mean((samples - mean) ** 2))), rms_floor)
    transform = Normalization(mean, scale)
    normalized = np.where(observation.observed, (observation.values - mean) / scale, 0.0)
    return Observation(normalized, observation.observed, observation.sample_rate), transform
