"""Conservative interpolation from observed values only."""
from __future__ import annotations

import numpy as np

from acoustic_regeneration.data.observation import Observation


def linear_interpolation(observation: Observation) -> np.ndarray:
    indices = np.arange(len(observation.values))
    result = np.interp(indices, indices[observation.observed], observation.values[observation.observed])
    return observation.project(result)


def burg_coefficients(values: np.ndarray, order: int) -> np.ndarray:
    """Burg reflection recursion; clipping maintains a stable AR polynomial."""
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 1 or len(values) <= order + 1 or order < 1:
        raise ValueError("Insufficient context for the AR order")
    forward, backward = values[1:].copy(), values[:-1].copy()
    coefficients = np.ones(1)
    for _ in range(order):
        denominator = float(forward @ forward + backward @ backward)
        if denominator <= np.finfo(float).eps * max(float(values @ values), 1e-30):
            break
        reflection = np.clip(-2 * float(backward @ forward) / denominator, -0.999, 0.999)
        coefficients = np.r_[coefficients, 0] + reflection * np.r_[0, coefficients[::-1]]
        next_forward = forward + reflection * backward
        backward = (backward + reflection * forward)[:-1]
        forward = next_forward[1:]
    return coefficients


def _extrapolate(context: np.ndarray, samples: int, order: int) -> np.ndarray:
    mean = float(context.mean())
    centered = context - mean
    coefficients = burg_coefficients(centered, order)
    # Feed the context through the prediction filter to initialize its state.
    # A direct recurrence avoids treating measured context as new innovations.
    history = centered.tolist()
    output = []
    limit = 20 * max(float(np.sqrt(np.mean(centered ** 2))), 1e-12)
    for _ in range(samples):
        count = len(coefficients) - 1
        value = -float(np.dot(coefficients[1:], history[-count:][::-1])) if count else 0.0
        if not np.isfinite(value) or abs(value) > limit:
            raise FloatingPointError("Unstable or excessive AR extrapolation")
        history.append(value)
        output.append(value + mean)
    return np.asarray(output)


def bidirectional_ar(observation: Observation, order: int = 32,
                     context_seconds: float = 0.5) -> np.ndarray:
    result = linear_interpolation(observation)
    width = round(context_seconds * observation.sample_rate)
    for start, end in observation.gaps:
        # Stop context at another gap; never use its zero values or an imputation.
        left_start = start
        while left_start > max(0, start - width) and observation.observed[left_start - 1]:
            left_start -= 1
        right_end = end
        while right_end < min(len(result), end + width) and observation.observed[right_end]:
            right_end += 1
        left = observation.values[left_start:start]
        right = observation.values[end:right_end]
        if min(len(left), len(right)) <= order + 1:
            continue
        try:
            forward = _extrapolate(left, end - start, order)
            backward = _extrapolate(right[::-1], end - start, order)[::-1]
            blend = np.arange(1, end - start + 1) / (end - start + 1)
            result[start:end] = (1 - blend) * forward + blend * backward
        except (FloatingPointError, np.linalg.LinAlgError, ValueError):
            continue
    return observation.project(result)
