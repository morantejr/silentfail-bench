"""Rank-normalization and uncertainty statistics (train-fit only)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class RankNormFit:
    """Percentile rank mapping fit on training raw scores per critic."""

    critic_id: str
    sorted_train: np.ndarray

    def transform(self, raw: float) -> float:
        if self.sorted_train.size == 0:
            return 0.5
        # Empirical CDF rank in [0, 1]
        return float(np.searchsorted(self.sorted_train, raw, side="right") / len(self.sorted_train))


def fit_rank_norms(train_raw: dict[str, list[float]]) -> dict[str, RankNormFit]:
    fits: dict[str, RankNormFit] = {}
    for cid, vals in train_raw.items():
        arr = np.sort(np.asarray(vals, dtype=float))
        fits[cid] = RankNormFit(critic_id=cid, sorted_train=arr)
    return fits


def uncertainty_std(rank_norm_scores: dict[str, float]) -> float:
    vals = np.asarray(list(rank_norm_scores.values()), dtype=float)
    if vals.size < 2:
        return 0.0
    return float(np.std(vals, ddof=1))


def apply_panel_u(
    raw_scores: dict[str, float],
    fits: dict[str, RankNormFit],
) -> tuple[dict[str, float], float]:
    rn = {cid: fits[cid].transform(raw) for cid, raw in raw_scores.items() if cid in fits}
    return rn, uncertainty_std(rn)
