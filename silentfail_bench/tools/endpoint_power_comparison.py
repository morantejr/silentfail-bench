#!/usr/bin/env python3
"""Does a budgeted-escalation endpoint beat global AUROC at fixed n?

Runs every endpoint through the SAME simulated data and the SAME clustered
test, so null rejection and power are directly comparable. Reports both --
an endpoint is only better if it gains power WITHOUT inflating the null.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from silentfail.combine import _design_matrix, fit_models          # noqa: E402
from silentfail.stats import ENDPOINTS, cluster_bootstrap_delta    # noqa: E402


def trial(rng, n, u_effect, failure_rate=0.65, alpha=0.05, n_boot=250):
    grp = np.arange(n)
    c = rng.normal(0, 1, n)
    u = 0.3 * c + rng.normal(0, 1, n)
    logit = np.log(failure_rate / (1 - failure_rate)) - 0.8 * c + u_effect * u
    g = (rng.random(n) > 1 / (1 + np.exp(-logit))).astype(int)
    ctrl = np.column_stack([np.abs(c) * 1000, rng.integers(1, 20, n)])
    tr = rng.random(n) < 0.7
    te = ~tr
    y = (g[te] == 0).astype(int)
    if y.sum() < 5 or (1 - y).sum() < 5 or (g[tr] == 0).sum() < 3 or (g[tr] == 1).sum() < 3:
        return None
    m = fit_models(c[tr], u[tr], ctrl[tr], g[tr])
    Xc, _ = _design_matrix(c[te], u[te], ctrl[te], True, False)
    Xcu, _ = _design_matrix(c[te], u[te], ctrl[te], True, True)
    sb = m["M_c"].predict_proba_fail(Xc)
    sa = m["M_c+u"].predict_proba_fail(Xcu)
    out = {}
    for name in ENDPOINTS:
        t = cluster_bootstrap_delta(y, sb, sa, grp[te], endpoint=name,
                                    n_boot=n_boot, alpha=alpha, seed=0)
        out[name] = (t.p_one_sided < alpha) if np.isfinite(t.p_one_sided) else None
    return out


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 400
    n_sims = int(sys.argv[2]) if len(sys.argv) > 2 else 120
    print(f"n={n}, n_sims={n_sims}, alpha=0.05, groups=1 row each\n")
    print(f"{'endpoint':22s} {'NULL rejection':>15s} {'POWER u=0.5':>13s} {'POWER u=1.0':>13s}")
    print("-" * 66)
    res = {name: {} for name in ENDPOINTS}
    for eff in (0.0, 0.5, 1.0):
        rng = np.random.default_rng(1234)
        acc = {name: [] for name in ENDPOINTS}
        for _ in range(n_sims):
            r = trial(rng, n, eff)
            if r is None:
                continue
            for k, v in r.items():
                if v is not None:
                    acc[k].append(v)
        for k in ENDPOINTS:
            res[k][eff] = float(np.mean(acc[k])) if acc[k] else float("nan")
    for k in ENDPOINTS:
        print(f"{k:22s} {res[k][0.0]:15.3f} {res[k][0.5]:13.3f} {res[k][1.0]:13.3f}")
    print("\nAn endpoint wins only if power rises WITHOUT the null column rising.")


if __name__ == "__main__":
    main()
