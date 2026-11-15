"""Fixed-pool selection plus a calibrated gate; no target-dependent inference."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional, Tuple

import numpy as np

from acoustic_regeneration.data.observation import NonFiniteCompletionError, Normalization, Observation, normalize_observation
from acoustic_regeneration.diagnostics.calibration import CalibrationResult
from acoustic_regeneration.diagnostics.features import candidate_features
from acoustic_regeneration.diagnostics.predictor import predict_deltas
from acoustic_regeneration.diagnostics.probes import make_probes, score_probes
from acoustic_regeneration.evaluation.metrics import MetricFloors, gap_metrics


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
            "score": float(scores[index]) if valid else None, "probe_deltas": pool.probe_deltas,
            "denoiser_calls": pool.denoiser_calls, "valid": valid}


def generate_batched_pool(observation: Observation, model, fallback: Callable,
                          rms_floor: float, candidate_seeds: tuple, probe_seeds: tuple,
                          steps_per_draw: int = 32) -> Optional[CandidatePool]:
    from acoustic_regeneration.restoration.diffusion import ddim_batch
    if len(candidate_seeds) != 4 or len(probe_seeds) != 2 or len(set(candidate_seeds + probe_seeds)) != 6:
        raise ValueError("Four candidate and two probe seeds must be distinct")
    probes = make_probes(observation)
    if probes is None:
        return None
    normalized, normalization = normalize_observation(observation, rms_floor)
    normalized_probes = [normalize_observation(probe.observation, rms_floor) for probe in probes]
    inputs = [normalized] * 4 + [item[0] for item in normalized_probes]
    draws = ddim_batch(model, inputs, candidate_seeds + probe_seeds, steps_per_draw)
    candidates = np.stack([draw.waveform for draw in draws[:4]])
    baseline = normalized.project(fallback(normalized))
    deltas = []
    for probe, (probe_observation, transform), draw in zip(probes, normalized_probes, draws[4:]):
        probe_baseline = probe_observation.project(fallback(probe_observation))
        reference = transform.normalize_reference_for_scoring(observation.values)
        floors = MetricFloors(rms_floor / transform.scale)
        interval = [(probe.start, probe.end)]
        deltas.append(gap_metrics(draw.waveform, reference, interval, floors)["Q"] -
                      gap_metrics(probe_baseline, reference, interval, floors)["Q"])
    features = candidate_features(normalized, candidates, baseline, tuple(deltas), rms_floor / normalization.scale)
    return CandidatePool(normalized, normalization, candidates, baseline, features, tuple(deltas), candidate_seeds,
                         probe_seeds, sum(draw.denoiser_calls for draw in draws))


def selective_restore(observation: Observation, model, fallback: Callable, predictor: dict,
                      calibration: CalibrationResult, rms_floor: float, candidate_seeds: tuple,
                      probe_seeds: tuple, steps_per_draw: int = 32) -> dict:
    """Total primary policy for missing probes or nonfinite generative output.

    Model hooks count actual per-example denoiser evaluations, including failed
    draws. Validity failures retain the fixed fallback and the imputation mask.
    """
    calls = 0
    def count_calls(module, inputs):
        nonlocal calls
        calls += len(inputs[0])
    handle = model.register_forward_pre_hook(count_calls)
    reason = None
    try:
        pool = generate_batched_pool(observation, model, fallback, rms_floor,
                                     candidate_seeds, probe_seeds, steps_per_draw)
        if pool is None:
            reason = "insufficient_probe_context"
        else:
            result = decide(observation, pool, predictor, calibration)
            result["denoiser_calls"] = calls
            return result
    except NonFiniteCompletionError:
        reason = "nonfinite_generation"
    finally:
        handle.remove()
    normalized, transform = normalize_observation(observation, rms_floor)
    estimate = normalized.project(fallback(normalized))
    return {"waveform": transform.restore(estimate, observation), "imputed_mask": ~observation.observed.copy(),
            "accepted": False, "selected_candidate": None, "score": None, "probe_deltas": None,
            "denoiser_calls": calls, "valid": False, "fallback_reason": reason}
