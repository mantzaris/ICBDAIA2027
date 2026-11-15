"""Simultaneous binomial selective-harm calibration on independent groups."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Optional, Sequence, Tuple

import numpy as np
from scipy.stats import beta


@dataclass(frozen=True)
class CalibrationCase:
    group_id: str
    score: float
    candidate_q: float
    fallback_q: float
    valid: bool = True


@dataclass(frozen=True)
class ThresholdResult:
    threshold: float
    accepted: int
    harmful: int
    harm_upper_bound: float


@dataclass(frozen=True)
class CalibrationResult:
    threshold: Optional[float]
    groups: int
    multiplicity: int
    family_error: float
    harm_limit: float
    rows: Tuple[ThresholdResult, ...]

    def accepts(self, score: float, valid: bool = True) -> bool:
        return bool(valid and self.threshold is not None and np.isfinite(score) and score <= self.threshold)

    def to_dict(self) -> dict:
        return asdict(self)


def freeze_thresholds(validation_scores: Sequence[float]) -> Tuple[float, ...]:
    scores = np.asarray(validation_scores, dtype=float)
    scores = scores[np.isfinite(scores)]
    if not len(scores):
        return ()
    quantiles = (0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.80, 1.00)
    return tuple(np.unique(np.minimum(np.quantile(scores, quantiles), 0.0)).tolist())


def harm_upper_bound(errors: int, accepted: int, family_error: float, multiplicity: int) -> float:
    if not 0 <= errors <= accepted or multiplicity < 1 or not 0 < family_error < 1:
        raise ValueError("Invalid binomial counts or multiplicity")
    if accepted == 0 or errors == accepted:
        return 1.0
    return float(beta.ppf(1 - family_error / multiplicity, errors + 1, accepted - errors))


def calibrate(cases: Sequence[CalibrationCase], frozen_thresholds: Sequence[float],
              family_error: float = 0.05, harm_limit: float = 0.10,
              harm_margin: float = 0.02) -> CalibrationResult:
    ids = [case.group_id for case in cases]
    if any(not item for item in ids) or len(set(ids)) != len(ids):
        raise ValueError("Calibration requires exactly one case per independent group")
    if not 0 < family_error < 1 or not 0 < harm_limit < 1 or harm_margin < 0:
        raise ValueError("Invalid risk parameters")
    thresholds = tuple(sorted(set(float(value) for value in frozen_thresholds)))
    if len(thresholds) > 9 or any(not np.isfinite(value) or value > 0 for value in thresholds):
        raise ValueError("At most nine finite, nonpositive V-frozen thresholds are allowed")
    for case in cases:
        if not np.isfinite([case.candidate_q, case.fallback_q]).all() or not (
                0 <= case.candidate_q <= 1 and 0 <= case.fallback_q <= 1):
            raise ValueError("Q values must be finite and within [0, 1]")
    rows = []
    for threshold in thresholds:
        accepted = [case for case in cases if case.valid and np.isfinite(case.score) and case.score <= threshold]
        harmful = sum(case.candidate_q - case.fallback_q > harm_margin for case in accepted)
        rows.append(ThresholdResult(threshold, len(accepted), harmful,
                                    harm_upper_bound(harmful, len(accepted), family_error, len(thresholds))))
    eligible = [row for row in rows if row.harm_upper_bound <= harm_limit]
    best = min(eligible, key=lambda row: (-row.accepted, row.threshold)) if eligible else None
    return CalibrationResult(best.threshold if best else None, len(cases), len(thresholds),
                             family_error, harm_limit, tuple(rows))
