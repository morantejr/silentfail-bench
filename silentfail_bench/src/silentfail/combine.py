"""Frozen logistic combining models M_c, M_u, M_{c+u}."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


CONTROL_NAMES = ("length_chars", "step_count", "mean_agent_entropy")


def _sigmoid(z: NDArray[np.floating]) -> NDArray[np.floating]:
    return 1.0 / (1.0 + np.exp(-np.clip(z, -50, 50)))


@dataclass
class LogisticModel:
    """Binary logistic: P(g=0 | features). Positive class = failure.

    Carries the standardization fit on TRAIN ONLY. Raw controls such as
    length_chars run to 1e3-1e5; feeding them unscaled to IRLS overflowed the
    Hessian (observed: divide-by-zero / overflow in matmul). Standardizing also
    means test rows never contribute to the feature scaling.
    """

    feature_names: list[str]
    coef: NDArray[np.floating]
    intercept: float
    mu: NDArray[np.floating] | None = None
    sigma: NDArray[np.floating] | None = None

    def _scale(self, X: NDArray[np.floating]) -> NDArray[np.floating]:
        if self.mu is None or self.sigma is None:
            return X
        return (X - self.mu) / self.sigma

    def predict_proba_fail(self, X: NDArray[np.floating]) -> NDArray[np.floating]:
        z = self._scale(X) @ self.coef + self.intercept
        return _sigmoid(z)


def _fit_standardizer(X: NDArray[np.floating]):
    mu = X.mean(axis=0)
    sigma = X.std(axis=0)
    sigma = np.where(sigma < 1e-12, 1.0, sigma)   # constant columns -> no-op
    return mu, sigma


def _design_matrix(
    c: NDArray[np.floating] | None,
    u: NDArray[np.floating] | None,
    controls: NDArray[np.floating],
    include_c: bool,
    include_u: bool,
) -> tuple[NDArray[np.floating], list[str]]:
    cols: list[NDArray[np.floating]] = []
    names: list[str] = []
    if include_c:
        assert c is not None
        cols.append(c.reshape(-1, 1))
        names.append("c")
    if include_u:
        assert u is not None
        cols.append(u.reshape(-1, 1))
        names.append("u")
    cols.append(controls)
    names.extend(CONTROL_NAMES)
    return np.hstack(cols), names


def fit_logistic_irls(
    X: NDArray[np.floating],
    y_fail: NDArray[np.floating],
    max_iter: int = 50,
    l2: float = 1e-2,
) -> tuple[NDArray[np.floating], float]:
    """L2-regularized IRLS with a divergence guard; no sklearn dependency.

    Under (quasi-)separation -- likely at pilot n -- plain IRLS drives the
    intercept to +/-inf, giving inf/nan in the design matmul and a NaN AUROC.
    Guards: ridge on the intercept too, step-halving on non-finite or
    non-improving steps, and a final finiteness fallback.
    """
    n, p = X.shape
    X_aug = np.hstack([np.ones((n, 1)), X])
    beta = np.zeros(p + 1)
    reg_diag = np.full(p + 1, l2)
    reg_diag[0] = 1e-6                      # near-free intercept, but not free

    def nll(b):
        z = np.clip(X_aug @ b, -50, 50)
        ph = _sigmoid(z)
        eps = 1e-12
        return -np.sum(y_fail * np.log(ph + eps) + (1 - y_fail) * np.log(1 - ph + eps)) \
               + 0.5 * np.sum(reg_diag * b * b)

    cur = nll(beta)
    for _ in range(max_iter):
        z = np.clip(X_aug @ beta, -50, 50)
        p_hat = _sigmoid(z)
        W = p_hat * (1.0 - p_hat) + 1e-6
        grad = X_aug.T @ (y_fail - p_hat) - reg_diag * beta
        H = X_aug.T @ (X_aug * W[:, None]) + np.diag(reg_diag)
        try:
            delta = np.linalg.solve(H, grad)
        except np.linalg.LinAlgError:
            delta = np.linalg.lstsq(H, grad, rcond=None)[0]
        if not np.all(np.isfinite(delta)):
            break
        # Step-halving: accept only a finite, non-worsening step.
        step = 1.0
        for _ in range(12):
            cand = beta + step * delta
            if np.all(np.isfinite(cand)):
                val = nll(cand)
                if np.isfinite(val) and val <= cur + 1e-9:
                    beta, cur = cand, val
                    break
            step *= 0.5
        else:
            break
        if np.linalg.norm(step * delta) < 1e-8:
            break

    if not np.all(np.isfinite(beta)):
        beta = np.zeros(p + 1)
    return beta[1:], float(beta[0])


def fit_models(
    c: NDArray[np.floating],
    u: NDArray[np.floating],
    controls: NDArray[np.floating],
    g: NDArray[np.integer],
) -> dict[str, LogisticModel]:
    """Fit M_c, M_u, M_{c+u}. y = 1 indicates failure (g==0)."""
    y_fail = (g == 0).astype(float)
    out: dict[str, LogisticModel] = {}
    for name, inc_c, inc_u in (
        ("M_c", True, False),
        ("M_u", False, True),
        ("M_c+u", True, True),
    ):
        X, names = _design_matrix(c, u, controls, inc_c, inc_u)
        mu, sigma = _fit_standardizer(X)
        coef, intercept = fit_logistic_irls((X - mu) / sigma, y_fail)
        out[name] = LogisticModel(
            feature_names=names, coef=coef, intercept=intercept, mu=mu, sigma=sigma
        )
    return out


def auroc(y_fail: NDArray[np.floating], scores: NDArray[np.floating]) -> float:
    """Mann-Whitney AUROC; positive class = failure."""
    y = np.asarray(y_fail).astype(int)
    s = np.asarray(scores, dtype=float)
    pos = s[y == 1]
    neg = s[y == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    # Handle ties via mid-rank
    correct = 0.0
    for p in pos:
        correct += np.sum(neg < p) + 0.5 * np.sum(neg == p)
    return float(correct / (len(pos) * len(neg)))


def delta_auroc(
    models: dict[str, LogisticModel],
    c: NDArray[np.floating],
    u: NDArray[np.floating],
    controls: NDArray[np.floating],
    g: NDArray[np.integer],
) -> dict[str, float]:
    y_fail = (g == 0).astype(float)
    results: dict[str, float] = {}
    for name, inc_c, inc_u in (
        ("M_c", True, False),
        ("M_u", False, True),
        ("M_c+u", True, True),
    ):
        X, _ = _design_matrix(c, u, controls, inc_c, inc_u)
        scores = models[name].predict_proba_fail(X)
        results[f"AUROC_{name}"] = auroc(y_fail, scores)
    results["delta_AUROC"] = results["AUROC_M_c+u"] - results["AUROC_M_c"]
    return results
