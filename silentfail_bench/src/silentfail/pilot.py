"""Pilot tables, go/no-go, and simulation-based power helpers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from silentfail.combine import CONTROL_NAMES, delta_auroc, fit_models
from silentfail.identifiers import assign_split


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def pilot_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    usable = [r for r in rows if not r.get("excluded")]
    n = len(usable)
    fails = sum(1 for r in usable if r.get("g") == 0)
    rate = fails / n if n else float("nan")
    # Wilson-ish normal approx CI
    if n:
        se = (rate * (1 - rate) / n) ** 0.5
        ci = (max(0.0, rate - 1.96 * se), min(1.0, rate + 1.96 * se))
    else:
        ci = (float("nan"), float("nan"))

    c = np.array([r["c"] for r in usable], dtype=float)
    u = np.array([r["u"] for r in usable], dtype=float)
    corr = float(np.corrcoef(c, u)[0, 1]) if n > 2 else float("nan")

    summary: dict[str, Any] = {
        "n_trajectories_logged": len(rows),
        "n_usable": n,
        "n_excluded": len(rows) - n,
        "n_failures_g0": fails,
        "failure_rate": rate,
        "failure_rate_ci95": ci,
        "c_mean": float(np.mean(c)) if n else None,
        "c_std": float(np.std(c)) if n else None,
        "u_mean": float(np.mean(u)) if n else None,
        "u_std": float(np.std(u)) if n else None,
        "c_u_correlation": corr,
    }

    if n >= 20 and fails >= 5 and (n - fails) >= 5:
        # Assign split if needed
        for r in usable:
            if r.get("split") == "pilot_unassigned":
                # builtin hash() is per-process salted -> non-reproducible splits
                r["split"] = assign_split(r["repo"])
        train = [r for r in usable if r["split"] == "train"]
        test = [r for r in usable if r["split"] == "test"]
        if len(train) < 10 or len(test) < 5:
            train, test = usable[: int(0.7 * n)], usable[int(0.7 * n) :]

        def pack(rs: list[dict[str, Any]]):
            c_ = np.array([r["c"] for r in rs], float)
            u_ = np.array([r["u"] for r in rs], float)
            g_ = np.array([r["g"] for r in rs], int)
            ctrl = np.array(
                [[r["length_chars"], r["step_count"], r.get("mean_agent_entropy") or 0.0] for r in rs],
                float,
            )
            return c_, u_, ctrl, g_

        c_tr, u_tr, ctrl_tr, g_tr = pack(train)
        models = fit_models(c_tr, u_tr, ctrl_tr, g_tr)
        c_te, u_te, ctrl_te, g_te = pack(test)
        metrics = delta_auroc(models, c_te, u_te, ctrl_te, g_te)
        summary["heldout_metrics"] = metrics
        summary["n_train"] = len(train)
        summary["n_test"] = len(test)
        summary["M_c_saturates"] = bool(metrics.get("AUROC_M_c", 0) > 0.90)
        summary["delta_auroc"] = metrics.get("delta_AUROC")
    else:
        summary["heldout_metrics"] = None
        summary["note"] = "Insufficient usable/failure counts for held-out logistic metrics"

    return summary


def go_no_go(
    summary: dict[str, Any],
    *,
    min_failures: int,
    max_mc_auroc: float = 0.90,
) -> dict[str, Any]:
    fails = int(summary.get("n_failures_g0") or 0)
    metrics = summary.get("heldout_metrics") or {}
    mc = metrics.get("AUROC_M_c")
    reasons = []
    ok_fail = fails >= min_failures
    if not ok_fail:
        reasons.append(f"failures {fails} < min_failures {min_failures}")
    ok_head = True
    if mc is not None and mc > max_mc_auroc:
        ok_head = False
        reasons.append(f"AUROC(M_c)={mc:.3f} > {max_mc_auroc} (no headroom)")
    decision = "GO" if ok_fail and ok_head else "NO_GO"
    return {
        "decision": decision,
        "min_failures_required": min_failures,
        "failures_observed": fails,
        "AUROC_M_c": mc,
        "reasons": reasons,
    }


def simulate_power(
    failure_rate: float,
    delta: float,
    n_traj: int,
    n_sims: int = 500,
    seed: int = 0,
) -> dict[str, Any]:
    """
    Rough power sim: generate latent failure probs from c and optional u effect.
    Not a substitute for full clustered design — Stage-0 planning tool only.
    """
    rng = np.random.default_rng(seed)
    wins = 0
    for _ in range(n_sims):
        # Simulate c ~ N(0,1), u correlated weakly
        c = rng.normal(0, 1, n_traj)
        u = 0.3 * c + rng.normal(0, 1, n_traj)
        # Failure logit driven by -c and +delta*u
        logit = np.log(failure_rate / max(1e-6, 1 - failure_rate)) - 0.8 * c + delta * 4 * u
        p = 1 / (1 + np.exp(-logit))
        g = (rng.random(n_traj) > p).astype(int)  # g=1 success
        # If too few fails, skip
        if ((g == 0).sum() < 5) or ((g == 1).sum() < 5):
            continue
        ctrl = np.column_stack([np.abs(c) * 100, rng.integers(1, 20, n_traj), rng.random(n_traj)])
        split = int(0.7 * n_traj)
        models = fit_models(c[:split], u[:split], ctrl[:split], g[:split])
        metrics = delta_auroc(models, c[split:], u[split:], ctrl[split:], g[split:])
        d = metrics["delta_AUROC"]
        if d == d and d > 0:  # not nan
            # Treat detection as delta CI proxy: delta > 0 in sample
            wins += 1
    return {
        "failure_rate": failure_rate,
        "assumed_delta": delta,
        "n_traj": n_traj,
        "n_sims": n_sims,
        "power_delta_gt_0": wins / n_sims,
    }


def required_n_for_power(
    failure_rate: float,
    delta: float,
    target_power: float = 0.8,
    n_grid: list[int] | None = None,
) -> dict[str, Any]:
    n_grid = n_grid or [100, 200, 400, 800, 1200, 2000]
    results = []
    chosen = None
    for n in n_grid:
        sim = simulate_power(failure_rate, delta, n, n_sims=300, seed=n)
        results.append(sim)
        if chosen is None and sim["power_delta_gt_0"] >= target_power:
            chosen = n
    return {"target_power": target_power, "chosen_n": chosen, "grid": results}
