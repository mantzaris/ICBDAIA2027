from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional, Tuple

import numpy as np

from acoustic_regeneration.data.observation import Observation, normalize_observation
from acoustic_regeneration.evaluation.metrics import MetricFloors, gap_metrics


@dataclass(frozen=True)
class Probe:
    start: int
    end: int
    observation: Observation


def make_probes(observation: Observation, guard_ms: int = 64) -> Optional[Tuple[Probe, Probe]]:
    if len(observation.gaps) != 1:
        return None
    start, end = observation.gaps[0]
    duration = end - start
    guard = round(guard_ms * observation.sample_rate / 1000)
    intervals = ((start - guard - duration, start - guard), (end + guard, end + guard + duration))
    if any(a < 0 or b > len(observation.values) or not observation.observed[a:b].all() for a, b in intervals):
        return None
    return tuple(Probe(a, b, observation.hide(a, b)) for a, b in intervals)


def score_probes(observation: Observation, generator: Callable, fallback: Callable,
                 rms_floor: float, seeds: Tuple[int, int]) -> Optional[Tuple[float, float]]:
    probes = make_probes(observation)
    if probes is None:
        return None
    differences = []
    for probe, seed in zip(probes, seeds):
        # Re-mask first, then recompute mean, RMS, model input, and features.
        normalized, transform = normalize_observation(probe.observation, rms_floor)
        generated = normalized.project(generator(normalized, seed))
        baseline = normalized.project(fallback(normalized))
        # Only the originally OBSERVED probe is scored. The original gap remains
        # unavailable even in this evaluation target (zero there is never scored).
        reference = transform.normalize_reference_for_scoring(observation.values)
        floors = MetricFloors(rms_floor / transform.scale)
        interval = [(probe.start, probe.end)]
        differences.append(gap_metrics(generated, reference, interval, floors)["Q"] -
                           gap_metrics(baseline, reference, interval, floors)["Q"])
    return tuple(differences)
