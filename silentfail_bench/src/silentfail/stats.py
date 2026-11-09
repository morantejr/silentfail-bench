"""Calibrated inference for the paired held-out dAUROC estimand.

Defect #4 (2026-09-22). The previous `pilot.simulate_power` defined "power" as
P(dAUROC_hat > 0) in the test sample. That statistic is ~0.5 under the true
null, so the function reported 0.39-0.44 "power" when u carried no signal at
all -- it measured the sign of noise, not detection. Measured before the fix:

    delta=0, failure_rate=0.65:  n=100 -> 0.390, n=200 -> 0.395, n=400 -> 0.435

A second bug deflated it in the other direction: sims skipped by the
`if too few fails: continue` guard stayed in the `wins / n_sims` denominator.

Replacement: AUROC(M_c+u) and AUROC(M_c) are computed on the SAME held-out
rows, so the comparison is paired and the two AUROCs are correlated. Inference
is a cluster bootstrap that resamples GROUPS (repo, or problem_id in the
local-exec profile), because rows inside a group are not independent. The
fitted models are held fixed while the test set is resampled, which is exactly
the estimand: the held-out dAUROC of the models trained on the train split.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np
from numpy.typing import NDArray


def auroc_fast(y_fail: NDArray, scores: NDArray) -> float:
    """Mann-Whitney AUROC with mid-rank tie handling. Vectorized.

    Matches combine.auroc but O(n log n) instead of O(n_pos * n_neg), which is
    what makes bootstrap-inside-simulation tractable.
    """
    y = np.asarray(y_fail).astype(int)
    s = np.asarray(scores, dtype=float)
    n_pos = int((y == 1).sum())
    n_neg = int((y == 0).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(s, kind="mergesort")
    s_sorted = s[order]
    ranks = np.empty(len(s), dtype=float)
    i = 0
    while i < len(s_sorted):
        j = i
        while j + 1 < len(s_sorted) and s_sorted[j + 1] == s_sorted[i]:
            j += 1
        ranks[order[i : j + 1]] = 0.5 * (i + j) + 1.0  # mid-rank, 1-based
        i = j + 1
    return float((ranks[y == 1].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


@dataclass
class DeltaTest:
    delta: float
    auroc_a: float
    auroc_b: float
    ci_lo: float
    ci_hi: float
    p_one_sided: float
    p_two_sided: float
    n_boot_valid: int
    n_groups: int
    alpha: float = 0.05
    boot_deltas: NDArray = field(default=None, repr=False)

    @property
    def significant_one_sided(self) -> bool:
        return self.p_one_sided < self.alpha


def cluster_bootstrap_delta_auroc(
    y_fail: NDArray,
    scores_base: NDArray,
    scores_aug: NDArray,
    groups: Sequence,
    *,
    n_boot: int = 2000,
    alpha: float = 0.05,
    seed: int = 0,
) -> DeltaTest:
    """Paired dAUROC = AUROC(aug) - AUROC(base), with group-clustered CI.

    groups: cluster label per row (repo / problem_id). Resampled with
    replacement; all rows of a drawn group come along.
    """
    y = np.asarray(y_fail).astype(int)
    sb = np.asarray(scores_base, float)
    sa = np.asarray(scores_aug, float)
    g = np.asarray(groups)
    uniq = np.unique(g)
    idx_by_group = {k: np.flatnonzero(g == k) for k in uniq}

    a_base = auroc_fast(y, sb)
    a_aug = auroc_fast(y, sa)
    point = a_aug - a_base

    rng = np.random.default_rng(seed)
    deltas = np.empty(n_boot)
    deltas[:] = np.nan
    for b in range(n_boot):
        drawn = rng.choice(uniq, size=len(uniq), replace=True)
        idx = np.concatenate([idx_by_group[k] for k in drawn])
        yb = y[idx]
        if yb.sum() == 0 or yb.sum() == len(yb):
            continue                      # degenerate resample: excluded, not counted
        deltas[b] = auroc_fast(yb, sa[idx]) - auroc_fast(yb, sb[idx])

    valid = deltas[~np.isnan(deltas)]
    if valid.size < 50:
        return DeltaTest(point, a_base, a_aug, float("nan"), float("nan"),
                         float("nan"), float("nan"), int(valid.size), len(uniq), alpha)

    ci_lo, ci_hi = np.percentile(valid, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    # Bootstrap p: mass on the wrong side of the null.
    p_one = float((valid <= 0).mean())
    p_two = float(2 * min((valid <= 0).mean(), (valid >= 0).mean()))
    return DeltaTest(point, a_base, a_aug, float(ci_lo), float(ci_hi),
                     p_one, min(1.0, p_two), int(valid.size), len(uniq), alpha, valid)


def simulate_power(
    failure_rate: float,
    true_u_effect: float,
    n_groups: int,
    rows_per_group: int = 1,
    *,
    n_sims: int = 200,
    n_boot: int = 400,
    alpha: float = 0.05,
    beta_c: float = -0.8,
    seed: int = 0,
) -> dict:
    """Rejection rate of the one-sided clustered dAUROC test.

    `true_u_effect` = 0 makes this a CALIBRATION check: the returned
    `rejection_rate` must come out near `alpha`, not near 0.5.

    Skipped (degenerate) sims are reported separately and are NOT in the
    denominator -- that was the second half of defect #4.
    """
    from silentfail.combine import delta_auroc, fit_models

    rng = np.random.default_rng(seed)
    n = n_groups * rows_per_group
    group_ids = np.repeat(np.arange(n_groups), rows_per_group)
    base_logit = np.log(max(1e-6, failure_rate) / max(1e-6, 1 - failure_rate))

    rejections = 0
    valid = 0
    skipped = 0
    deltas = []
    for s in range(n_sims):
        c = rng.normal(0, 1, n)
        u = 0.3 * c + rng.normal(0, 1, n)
        logit = base_logit + beta_c * c + true_u_effect * u
        p_fail = 1 / (1 + np.exp(-logit))
        g = (rng.random(n) > p_fail).astype(int)          # g=1 success
        ctrl = np.column_stack([np.abs(c) * 1000, rng.integers(1, 20, n)])

        # Group-level split, matching the real design.
        tr_groups = rng.random(n_groups) < 0.7
        tr = tr_groups[group_ids]
        te = ~tr
        y_te = (g[te] == 0).astype(int)
        if tr.sum() < 20 or te.sum() < 10 or y_te.sum() < 3 or (1 - y_te).sum() < 3:
            skipped += 1
            continue
        if (g[tr] == 0).sum() < 3 or (g[tr] == 1).sum() < 3:
            skipped += 1
            continue

        models = fit_models(c[tr], u[tr], ctrl[tr], g[tr])
        from silentfail.combine import _design_matrix
        X_c, _ = _design_matrix(c[te], u[te], ctrl[te], True, False)
        X_cu, _ = _design_matrix(c[te], u[te], ctrl[te], True, True)
        s_base = models["M_c"].predict_proba_fail(X_c)
        s_aug = models["M_c+u"].predict_proba_fail(X_cu)

        t = cluster_bootstrap_delta_auroc(
            y_te, s_base, s_aug, group_ids[te], n_boot=n_boot, alpha=alpha, seed=s
        )
        if not np.isfinite(t.p_one_sided):
            skipped += 1
            continue
        valid += 1
        deltas.append(t.delta)
        if t.p_one_sided < alpha:
            rejections += 1

    return {
        "failure_rate": failure_rate,
        "true_u_effect": true_u_effect,
        "n_groups": n_groups,
        "rows_per_group": rows_per_group,
        "n_trajectories": n,
        "alpha": alpha,
        "n_sims_requested": n_sims,
        "n_sims_valid": valid,
        "n_sims_skipped": skipped,
        "rejection_rate": (rejections / valid) if valid else float("nan"),
        "mean_delta_auroc": float(np.mean(deltas)) if deltas else float("nan"),
        "is_calibration_run": true_u_effect == 0.0,
    }


def required_groups_for_power(
    failure_rate: float,
    true_u_effect: float,
    rows_per_group: int = 1,
    target_power: float = 0.8,
    group_grid: Sequence[int] | None = None,
    *,
    n_sims: int = 150,
    n_boot: int = 300,
) -> dict:
    grid = list(group_grid or [40, 80, 160, 320, 640])
    results = []
    chosen = None
    for ng in grid:
        sim = simulate_power(failure_rate, true_u_effect, ng, rows_per_group,
                             n_sims=n_sims, n_boot=n_boot, seed=ng)
        results.append(sim)
        if chosen is None and sim["rejection_rate"] >= target_power:
            chosen = ng
    return {
        "target_power": target_power,
        "chosen_n_groups": chosen,
        "chosen_n_trajectories": (chosen * rows_per_group) if chosen else None,
        "grid": results,
    }


# ---------------------------------------------------------------------------
# Budgeted-escalation endpoints
#
# Global AUROC spends its power on the middle of the score distribution, which
# no deployment acts on. The decision an escalation policy actually makes is
# "audit the riskiest k% of selected trajectories", so precision@k and partial
# AUROC at low FPR are both more decision-relevant AND potentially better
# powered at fixed n. Whether they ARE better powered is an empirical question
# -- see tools/endpoint_power_comparison.py.
# ---------------------------------------------------------------------------

def precision_at_k(y_fail: NDArray, scores: NDArray, k_frac: float = 0.10) -> float:
    """Fraction of the top-k_frac highest-risk trajectories that really failed."""
    y = np.asarray(y_fail).astype(int)
    s = np.asarray(scores, float)
    n = len(y)
    k = max(1, int(round(k_frac * n)))
    top = np.argsort(-s, kind="mergesort")[:k]
    return float(y[top].mean())


def lift_at_k(y_fail: NDArray, scores: NDArray, k_frac: float = 0.10) -> float:
    """precision@k divided by the base failure rate. 1.0 = no better than random."""
    base = float(np.asarray(y_fail).astype(int).mean())
    if base <= 0:
        return float("nan")
    return precision_at_k(y_fail, scores, k_frac) / base


def partial_auroc(y_fail: NDArray, scores: NDArray, max_fpr: float = 0.20) -> float:
    """McClish-standardized partial AUROC over FPR in [0, max_fpr].

    0.5 = chance within that region, 1.0 = perfect, so it is read on the same
    scale as AUROC.
    """
    y = np.asarray(y_fail).astype(int)
    s = np.asarray(scores, float)
    if y.sum() == 0 or (1 - y).sum() == 0:
        return float("nan")
    order = np.argsort(-s, kind="mergesort")
    y_sorted = y[order]
    tp = np.cumsum(y_sorted)
    fp = np.cumsum(1 - y_sorted)
    tpr = tp / max(1, y.sum())
    fpr = fp / max(1, (1 - y).sum())
    tpr = np.concatenate([[0.0], tpr])
    fpr = np.concatenate([[0.0], fpr])
    keep = fpr <= max_fpr
    if keep.sum() < 2:
        return float("nan")
    x, yv = fpr[keep], tpr[keep]
    if x[-1] < max_fpr:                       # extend to the boundary
        x = np.concatenate([x, [max_fpr]])
        yv = np.concatenate([yv, [yv[-1]]])
    area = float(np.trapz(yv, x))
    # McClish standardization onto [0.5, 1]
    min_area = 0.5 * max_fpr**2
    max_area = max_fpr
    return 0.5 * (1 + (area - min_area) / (max_area - min_area))


ENDPOINTS = {
    "auroc": lambda y, s: auroc_fast(y, s),
    "precision_at_10pct": lambda y, s: precision_at_k(y, s, 0.10),
    "precision_at_20pct": lambda y, s: precision_at_k(y, s, 0.20),
    "partial_auroc_fpr20": lambda y, s: partial_auroc(y, s, 0.20),
}


def cluster_bootstrap_delta(
    y_fail: NDArray,
    scores_base: NDArray,
    scores_aug: NDArray,
    groups: Sequence,
    *,
    endpoint: str = "auroc",
    n_boot: int = 2000,
    alpha: float = 0.05,
    seed: int = 0,
) -> DeltaTest:
    """Paired delta for ANY endpoint in ENDPOINTS, with group-clustered CI."""
    fn = ENDPOINTS[endpoint]
    y = np.asarray(y_fail).astype(int)
    sb, sa = np.asarray(scores_base, float), np.asarray(scores_aug, float)
    g = np.asarray(groups)
    uniq = np.unique(g)
    idx_by_group = {k: np.flatnonzero(g == k) for k in uniq}

    a_base, a_aug = fn(y, sb), fn(y, sa)
    rng = np.random.default_rng(seed)
    deltas = np.full(n_boot, np.nan)
    for b in range(n_boot):
        drawn = rng.choice(uniq, size=len(uniq), replace=True)
        idx = np.concatenate([idx_by_group[k] for k in drawn])
        yb = y[idx]
        if yb.sum() == 0 or yb.sum() == len(yb):
            continue
        d = fn(yb, sa[idx]) - fn(yb, sb[idx])
        if np.isfinite(d):
            deltas[b] = d

    valid = deltas[~np.isnan(deltas)]
    if valid.size < 50:
        return DeltaTest(a_aug - a_base, a_base, a_aug, float("nan"), float("nan"),
                         float("nan"), float("nan"), int(valid.size), len(uniq), alpha)
    ci_lo, ci_hi = np.percentile(valid, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    p_one = float((valid <= 0).mean())
    p_two = float(2 * min((valid <= 0).mean(), (valid >= 0).mean()))
    return DeltaTest(a_aug - a_base, a_base, a_aug, float(ci_lo), float(ci_hi),
                     p_one, min(1.0, p_two), int(valid.size), len(uniq), alpha, valid)
