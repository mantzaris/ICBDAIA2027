"""Paired analysis with dependent descendants aggregated before resampling."""
from __future__ import annotations

from collections import defaultdict
from typing import Sequence

import numpy as np


def group_differences(group_ids: Sequence[str], proposed: Sequence[float], comparator: Sequence[float]) -> np.ndarray:
    if not len(group_ids) or len(group_ids) != len(proposed) or len(proposed) != len(comparator):
        raise ValueError("Nonempty paired measurements are required")
    grouped = defaultdict(list)
    for group, left, right in zip(group_ids, proposed, comparator):
        if not group or not np.isfinite([left, right]).all():
            raise ValueError("Invalid group or nonfinite measurement")
        grouped[group].append(left - right)
    return np.array([np.mean(grouped[key]) for key in sorted(grouped)])


def paired_bootstrap(group_ids: Sequence[str], proposed: Sequence[float], comparator: Sequence[float],
                     seed: int = 17, draws: int = 2000) -> dict:
    differences = group_differences(group_ids, proposed, comparator)
    if len(differences) < 2 or draws < 2:
        raise ValueError("At least two groups and bootstrap draws are required")
    rng = np.random.default_rng(seed)
    bootstrap = np.array([rng.choice(differences, len(differences), replace=True).mean() for _ in range(draws)])
    return {"mean_difference": float(differences.mean()), "ci95": np.quantile(bootstrap, [0.025, 0.975]).tolist(),
            "one_sided_upper95": float(np.quantile(bootstrap, 0.95)),
            "one_sided_lower95": float(np.quantile(bootstrap, 0.05)), "groups": len(differences), "draws": draws}


def paired_randomization(group_ids: Sequence[str], proposed: Sequence[float], comparator: Sequence[float],
                         seed: int = 17, draws: int = 10000) -> float:
    differences = group_differences(group_ids, proposed, comparator)
    rng = np.random.default_rng(seed)
    observed = abs(float(differences.mean()))
    exceedances = sum(abs(float((differences * rng.choice([-1, 1], len(differences))).mean())) >= observed
                      for _ in range(draws))
    return (exceedances + 1) / (draws + 1)


def holm_adjust(p_values: Sequence[float]) -> np.ndarray:
    values = np.asarray(p_values, dtype=float)
    if values.ndim != 1 or not np.isfinite(values).all() or np.any((values < 0) | (values > 1)):
        raise ValueError("P-values must be finite probabilities")
    order = np.argsort(values)
    adjusted = np.empty_like(values)
    adjusted[order] = np.minimum(1, np.maximum.accumulate(values[order] * np.arange(len(values), 0, -1)))
    return adjusted
