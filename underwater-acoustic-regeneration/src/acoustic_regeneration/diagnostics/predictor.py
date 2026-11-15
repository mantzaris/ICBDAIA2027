from __future__ import annotations

from collections import Counter
from typing import Sequence

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor

from acoustic_regeneration.diagnostics.features import FEATURE_NAMES


def feature_indices(ablation: str = "full") -> list:
    if ablation == "full":
        return list(range(len(FEATURE_NAMES)))
    if ablation == "no_probes":
        return [i for i, name in enumerate(FEATURE_NAMES) if "probe" not in name]
    if ablation == "duration_only":
        return [FEATURE_NAMES.index("gap_seconds")]
    if ablation == "dispersion_only":
        return [i for i, name in enumerate(FEATURE_NAMES) if "dispersion" in name]
    raise ValueError("Unknown ablation")


def fit_predictor(features: np.ndarray, deltas: np.ndarray, group_ids: Sequence[str],
                  partitions: Sequence[str], seed: int, ablation: str = "full") -> dict:
    if len(features) != len(deltas) or len(group_ids) != len(deltas) or len(partitions) != len(deltas):
        raise ValueError("Diagnostic arrays must align")
    if not len(deltas) or set(partitions) != {"D"} or not np.isfinite(deltas).all():
        raise ValueError("Fit diagnostics only on finite D targets")
    counts = Counter(group_ids)
    weights = np.array([1 / counts[group] for group in group_ids])
    weights *= len(weights) / weights.sum()
    indices = feature_indices(ablation)
    model = HistGradientBoostingRegressor(max_iter=200, max_leaf_nodes=15, learning_rate=0.05,
                                         l2_regularization=1, min_samples_leaf=20, early_stopping=False,
                                         random_state=seed)
    model.fit(features[:, indices], deltas, sample_weight=weights)
    return {"model": model, "indices": indices, "ablation": ablation, "seed": seed,
            "training_groups": sorted(counts), "feature_names": [FEATURE_NAMES[i] for i in indices]}


def predict_deltas(predictor: dict, features: np.ndarray) -> np.ndarray:
    return predictor["model"].predict(features[:, predictor["indices"]])
