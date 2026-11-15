"""Fixed-pool selection plus a calibrated gate; no target-dependent inference."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional, Tuple

import numpy as np

from acoustic_regeneration.data.observation import Normalization, Observation, normalize_observation
from acoustic_regeneration.diagnostics.calibration import CalibrationResult
from acoustic_regeneration.diagnostics.features import candidate_features
from acoustic_regeneration.diagnostics.predictor import predict_deltas
from acoustic_regeneration.diagnostics.probes import make_probes, score_probes


@dataclass(frozen=True)
class CandidatePool:
    normalized_observation: Observation
    normalization: Normalization
    candidates: np.ndarray
    fallback: np.ndarray
    features: np.ndarray
    probe_deltas: Tuple[float, float]
    candidate_seeds: tuple
    probe_seeds: tuple
    denoiser_calls: int


def generate_pool(observation: Observation, generator: Callable, fallback: Callable,
                  rms_floor: float, candidate_seeds: tuple, probe_seeds: tuple,
                  steps_per_draw: int = 32) -> Optional[CandidatePool]:
    if len(candidate_seeds) != 4 or len(probe_seeds) != 2 or len(set(candidate_seeds + probe_seeds)) != 6:
        raise ValueError("Four candidate and two probe seeds must be distinct")
    if make_probes(observation) is None:
        return None
    normalized, normalization = normalize_observation(observation, rms_floor)
    baseline = normalized.project(fallback(normalized))
    candidates = np.stack([normalized.project(generator(normalized, seed)) for seed in candidate_seeds])
    deltas = score_probes(observation, generator, fallback, rms_floor, probe_seeds)
    features = candidate_features(normalized, candidates, baseline, deltas, rms_floor / normalization.scale)
    return CandidatePool(normalized, normalization, candidates, baseline, features, deltas,
                         candidate_seeds, probe_seeds, 6 * steps_per_draw)


def decide(observation: Observation, pool: CandidatePool, predictor: dict,
           calibration: CalibrationResult, always_accept: bool = False) -> dict:
    scores = predict_deltas(predictor, pool.features)
    valid = bool(np.isfinite(scores).all() and np.isfinite(pool.candidates).all())
    index = int(np.argmin(scores)) if valid else 0
    accepted = bool(valid and (always_accept or calibration.accepts(float(scores[index]))))
    normalized = pool.candidates[index] if accepted else pool.fallback
    repaired = pool.normalization.restore(normalized, observation)
    return {"waveform": repaired, "imputed_mask": ~observation.observed.copy(), "accepted": accepted,
            "selected_candidate": index, "candidate_seed": pool.candidate_seeds[index],
            "score": float(scores[index]), "probe_deltas": pool.probe_deltas,
            "denoiser_calls": pool.denoiser_calls, "valid": valid}
