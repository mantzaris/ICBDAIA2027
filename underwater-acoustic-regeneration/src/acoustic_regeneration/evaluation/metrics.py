"""Gap-only scores; no centering, padding, or full-window dilution."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence, Tuple

import numpy as np


@dataclass(frozen=True)
class MetricFloors:
    """RMS floor from G, expressed in the current observation's normalized units.

    Energy floor is n*rms**2. Spectral floors use the expected Frobenius norm
    of RMS-floor white noise under our explicitly normalized periodic-Hann STFT.
    This fixes units and window-size scaling without inspecting hidden energy.
    """
    rms: float

    def __post_init__(self) -> None:
        if not np.isfinite(self.rms) or self.rms <= 0:
            raise ValueError("RMS floor must be positive and frozen from G")


def magnitude_stft(values: np.ndarray, window: int) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 1 or len(values) < window or window < 4:
        raise ValueError("STFT needs an unpadded interval at least one window long")
    frames = np.lib.stride_tricks.sliding_window_view(values, window)[::window // 4]
    hann = np.hanning(window + 1)[:-1]
    return np.abs(np.fft.rfft(frames * hann, axis=-1)) / np.sqrt(window)


def gap_metrics(prediction: np.ndarray, reference: np.ndarray,
                gaps: Sequence[Tuple[int, int]], floors: MetricFloors) -> dict:
    prediction, reference = np.asarray(prediction, dtype=float), np.asarray(reference, dtype=float)
    if prediction.ndim != 1 or prediction.shape != reference.shape or not gaps:
        raise ValueError("Matching vectors and explicit gap intervals are required")
    occupied = np.zeros(len(reference), dtype=bool)
    for start, end in gaps:
        if not 0 <= start < end <= len(reference) or end - start < 256 or occupied[start:end].any():
            raise ValueError("Gap intervals must be disjoint, in bounds, and at least 256 samples")
        occupied[start:end] = True
    truth, estimate = reference[occupied], prediction[occupied]
    if not np.isfinite(truth).all() or not np.isfinite(estimate).all():
        raise ValueError("Nonfinite target or prediction")
    nmse = float(np.sum((estimate - truth) ** 2) / max(float(truth @ truth), len(truth) * floors.rms ** 2))
    convergence, log_distance = [], []
    for size in (64, 128, 256):
        target = np.concatenate([magnitude_stft(reference[a:b], size) for a, b in gaps])
        predicted = np.concatenate([magnitude_stft(prediction[a:b], size) for a, b in gaps])
        hann_energy = float(np.sum(np.hanning(size + 1)[:-1] ** 2))
        bin_floor = floors.rms * np.sqrt(hann_energy / size)
        norm_floor = bin_floor * np.sqrt(target.size)
        convergence.append(float(np.linalg.norm(predicted - target) / max(np.linalg.norm(target), norm_floor)))
        differences = 20 * np.log10(np.maximum(predicted, bin_floor) / np.maximum(target, bin_floor))
        log_distance.append(float(np.sqrt(np.mean(differences ** 2))))
    spectral = float(np.mean(convergence))
    observed = ~occupied
    drift = float(np.max(np.abs(prediction[observed] - reference[observed]))) if observed.any() else 0.0
    boundary_amplitude, boundary_derivative = [], []
    for start, end in gaps:
        for index in (start, end):
            if 1 <= index < len(prediction):
                boundary_amplitude.append(abs(float(prediction[index] - prediction[index - 1])))
            if 2 <= index < len(prediction):
                boundary_derivative.append(abs(float(prediction[index] - 2 * prediction[index - 1] + prediction[index - 2])))
    return {
        "gap_nmse": nmse,
        "gap_spectral_convergence": spectral,
        "gap_log_spectral_distance_db": float(np.mean(log_distance)),
        "Q": 0.5 * min(nmse / 4, 1) + 0.5 * min(spectral / 2, 1),
        "observed_max_drift": drift,
        "boundary_amplitude_max": max(boundary_amplitude, default=0.0),
        "boundary_derivative_max": max(boundary_derivative, default=0.0),
        "gap_absolute_error_p95": float(np.quantile(np.abs(estimate - truth), 0.95)),
        "gap_absolute_error_p99": float(np.quantile(np.abs(estimate - truth), 0.99)),
        "gap_samples": int(occupied.sum()),
        "quiet_reference": bool(float(truth @ truth) <= len(truth) * floors.rms ** 2),
    }
