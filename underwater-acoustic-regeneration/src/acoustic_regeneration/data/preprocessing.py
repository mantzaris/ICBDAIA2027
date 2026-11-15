"""Resample complete source recordings before creating artificial gaps."""
from __future__ import annotations

from math import gcd
from typing import Optional

import numpy as np
from scipy.signal import resample_poly


def resample_source(source: np.ndarray, native_rate: int, target_rate: int = 8000,
                    native_observed: Optional[np.ndarray] = None) -> np.ndarray:
    if native_observed is not None and not np.asarray(native_observed, dtype=bool).all():
        raise ValueError("Naturally missing native audio requires filter-support mask propagation; unsupported here")
    source = np.asarray(source, dtype=np.float64)
    if source.ndim != 1 or not np.isfinite(source).all():
        raise ValueError("A complete, finite mono source is required")
    if native_rate <= 0 or target_rate <= 0:
        raise ValueError("Sample rates must be positive")
    divisor = gcd(native_rate, target_rate)
    return resample_poly(source, target_rate // divisor, native_rate // divisor,
                         window=("kaiser", 5.0), padtype="constant")


def crop_source(source: np.ndarray, start: int = 0, samples: int = 32768) -> np.ndarray:
    if start < 0 or samples <= 0 or start + samples > len(source):
        raise ValueError("Crop must fit entirely inside the resampled source")
    return np.asarray(source[start:start + samples], dtype=np.float64).copy()


def draw_mask(samples: int, rng: np.random.Generator, sample_rate: int = 8000,
              gap_ms: tuple = (50, 100, 250, 500), two_gap_probability: float = 0.0,
              minimum_context_ms: int = 750) -> np.ndarray:
    if not 0 <= two_gap_probability <= 1 or not gap_ms or min(gap_ms) <= 0:
        raise ValueError("Invalid mask distribution")
    margin = round(minimum_context_ms * sample_rate / 1000)
    duration = round(int(rng.choice(gap_ms)) * sample_rate / 1000)
    if samples < 2 * margin + duration:
        raise ValueError("Insufficient context for the requested gap")
    start = int(rng.integers(margin, samples - margin - duration + 1))
    mask = np.ones(samples, dtype=bool)
    mask[start:start + duration] = False
    if rng.random() < two_gap_probability:
        second_duration = round(int(rng.choice(gap_ms)) * sample_rate / 1000)
        guard = round(0.064 * sample_rate)
        starts = np.arange(margin, samples - margin - second_duration + 1)
        valid = starts[(starts + second_duration + guard <= start) | (starts >= start + duration + guard)]
        if len(valid):
            second_start = int(rng.choice(valid))
            mask[second_start:second_start + second_duration] = False
    return mask
