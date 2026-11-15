"""Scalar diagnostics whose API has no reference waveform or label parameter."""
from __future__ import annotations

import numpy as np

from acoustic_regeneration.data.observation import Observation
from acoustic_regeneration.evaluation.metrics import magnitude_stft

FEATURE_NAMES = (
    "gap_seconds", "left_context_seconds", "right_context_seconds", "left_right_log_psd_distance",
    "observed_rms", "observed_spectral_flatness", "boundary_amplitude", "boundary_derivative",
    "gap_to_observed_energy", "fallback_waveform_distance", "fallback_spectral_distance",
    "pool_dispersion_mean", "pool_dispersion_max", "distance_to_medoid",
    "left_probe_delta", "right_probe_delta", "left_probe_missing", "right_probe_missing",
)


def spectral_distances(candidates: np.ndarray, gap: tuple, floor: float) -> np.ndarray:
    start, end = gap
    spectra = [np.concatenate([magnitude_stft(candidate[start:end], size).ravel()
                               for size in (64, 128, 256)]) for candidate in candidates]
    distances = np.zeros((len(candidates), len(candidates)))
    for i in range(len(candidates)):
        for j in range(i):
            denominator = max((np.linalg.norm(spectra[i]) + np.linalg.norm(spectra[j])) / 2, floor)
            distances[i, j] = distances[j, i] = np.linalg.norm(spectra[i] - spectra[j]) / denominator
    return distances


def candidate_features(observation: Observation, candidates: np.ndarray, fallback: np.ndarray,
                       probe_deltas: tuple, rms_floor: float) -> np.ndarray:
    if len(observation.gaps) != 1 or len(candidates) < 1 or candidates.shape[1:] != observation.values.shape:
        raise ValueError("Features require a single real gap and a finite candidate pool")
    if not np.isfinite(candidates).all() or not np.isfinite(fallback).all() or rms_floor <= 0:
        raise ValueError("Invalid pool or floor")
    start, end = observation.gaps[0]
    width = min(4000, start, len(observation.values) - end)
    if width < 256:
        raise ValueError("Insufficient observed context for diagnostics")
    left, right = observation.values[start - width:start], observation.values[end:end + width]
    context = np.r_[left, right]
    rms = max(float(np.sqrt(np.mean(context ** 2))), rms_floor)
    left_psd = np.mean(magnitude_stft(left, 256) ** 2, axis=0)
    right_psd = np.mean(magnitude_stft(right, 256) ** 2, axis=0)
    psd = (left_psd + right_psd) / 2
    log_distance = float(np.sqrt(np.mean((np.log(np.maximum(left_psd, rms_floor ** 2)) -
                                         np.log(np.maximum(right_psd, rms_floor ** 2))) ** 2)))
    flatness = float(np.exp(np.mean(np.log(np.maximum(psd, rms_floor ** 2)))) / max(psd.mean(), rms_floor ** 2))
    distances = spectral_distances(candidates, (start, end), rms_floor)
    medoid = int(np.argmin(distances.sum(axis=1)))
    off_diagonal = distances[~np.eye(len(candidates), dtype=bool)]
    dispersion_mean = float(off_diagonal.mean()) if len(off_diagonal) else 0.0
    dispersion_max = float(off_diagonal.max()) if len(off_diagonal) else 0.0
    result = []
    for index, candidate in enumerate(candidates):
        jumps = [abs(candidate[start] - observation.values[start - 1]),
                 abs(observation.values[end] - candidate[end - 1])]
        derivatives = [abs((candidate[start + 1] - candidate[start]) - (left[-1] - left[-2])),
                       abs((right[1] - right[0]) - (candidate[end - 1] - candidate[end - 2]))]
        fallback_distance = spectral_distances(np.stack([candidate, fallback]), (start, end), rms_floor)[0, 1]
        probe_values = [float(value) if np.isfinite(value) else np.nan for value in probe_deltas]
        result.append([
            (end - start) / observation.sample_rate, start / observation.sample_rate,
            (len(observation.values) - end) / observation.sample_rate, log_distance, rms, flatness,
            max(jumps) / rms, max(derivatives) / rms,
            float(np.mean(candidate[start:end] ** 2)) / rms ** 2,
            float(np.sqrt(np.mean((candidate[start:end] - fallback[start:end]) ** 2))) / rms,
            fallback_distance, dispersion_mean, dispersion_max, distances[index, medoid],
            *probe_values, *[float(not np.isfinite(value)) for value in probe_values],
        ])
    return np.asarray(result, dtype=float)
