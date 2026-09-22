#!/usr/bin/env python3
"""Go/no-go report for a profile-A scored JSONL, using the calibrated test."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from silentfail.combine import _design_matrix, fit_models  # noqa: E402
from silentfail.stats import cluster_bootstrap_delta_auroc, required_groups_for_power  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("jsonl", type=Path)
    ap.add_argument("--min-failures", type=int, default=12)
    ap.add_argument("--max-mc-auroc", type=float, default=0.90)
    ap.add_argument("--out", type=Path, default=None)
    a = ap.parse_args()

    rows = [json.loads(l) for l in a.jsonl.read_text().splitlines() if l.strip()]
    usable = [r for r in rows if not r["excluded"]]
    n = len(usable)
    fails = sum(1 for r in usable if r["g"] == 0)
    rate = fails / n if n else float("nan")
    se = (rate * (1 - rate) / n) ** 0.5 if n else float("nan")

    rep = {
        "scored_log": str(a.jsonl),
        "profile": "A_local_exec",
        "critic_type": "llm_judge",
        "is_true_reward_model": False,
        "gold": "executed HumanEval+ held-out tests",
        "n_logged": len(rows),
        "n_usable": n,
        "n_excluded": len(rows) - n,
        "n_failures_g0": fails,
        "failure_rate": rate,
        "failure_rate_ci95": [max(0, rate - 1.96 * se), min(1, rate + 1.96 * se)] if n else None,
    }

    if n:
        c = np.array([r["c"] for r in usable], float)
        u = np.array([r["u"] for r in usable], float)
        vis = np.array([r.get("visible_pass", False) for r in usable], bool)
        rep["c_mean"], rep["c_std"] = float(c.mean()), float(c.std())
        rep["u_mean"], rep["u_std"] = float(u.mean()), float(u.std())
        rep["c_u_correlation"] = float(np.corrcoef(c, u)[0, 1]) if n > 2 else None
        rep["visible_pass_rate"] = float(vis.mean())
        # The silent-failure cell: visible tests green, hidden gold red.
        g = np.array([r["g"] for r in usable], int)
        rep["silent_failure_count"] = int(((g == 0) & vis).sum())
        rep["silent_failure_rate_among_visible_pass"] = (
            float(((g == 0) & vis).sum() / vis.sum()) if vis.sum() else None
        )

    tr = [r for r in usable if r["split"] == "train"]
    te = [r for r in usable if r["split"] == "test"]
    rep["n_train"], rep["n_test"] = len(tr), len(te)

    def pack(rs):
        return (np.array([r["c"] for r in rs], float),
                np.array([r["u"] for r in rs], float),
                np.array([[r["length_chars"], r["step_count"]] for r in rs], float),
                np.array([r["g"] for r in rs], int),
                [r["repo"] for r in rs])

    testable = (len(tr) >= 10 and len(te) >= 5
                and 2 <= sum(1 for r in te if r["g"] == 0) and 2 <= sum(1 for r in te if r["g"] == 1)
                and 2 <= sum(1 for r in tr if r["g"] == 0) and 2 <= sum(1 for r in tr if r["g"] == 1))
    if testable:
        c1, u1, x1, g1, _ = pack(tr)
        c2, u2, x2, g2, grp2 = pack(te)
        models = fit_models(c1, u1, x1, g1)
        Xc, _ = _design_matrix(c2, u2, x2, True, False)
        Xcu, _ = _design_matrix(c2, u2, x2, True, True)
        y = (g2 == 0).astype(int)
        t = cluster_bootstrap_delta_auroc(
            y, models["M_c"].predict_proba_fail(Xc),
            models["M_c+u"].predict_proba_fail(Xcu), grp2, n_boot=4000, seed=1)
        rep["heldout"] = {
            "AUROC_M_c": t.auroc_a, "AUROC_M_c+u": t.auroc_b, "delta_AUROC": t.delta,
            "delta_CI95": [t.ci_lo, t.ci_hi], "p_one_sided": t.p_one_sided,
            "n_test_groups": t.n_groups,
            "NOTE": "smoke test only; n=40 has no power (see power_projection)",
        }
    else:
        rep["heldout"] = None
        rep["heldout_note"] = "too few usable rows / one class absent in a split"

    reasons = []
    if fails < a.min_failures:
        reasons.append(f"selected-set failures {fails} < {a.min_failures} required")
    mc = (rep.get("heldout") or {}).get("AUROC_M_c")
    if mc is not None and mc > a.max_mc_auroc:
        reasons.append(f"AUROC(M_c)={mc:.3f} > {a.max_mc_auroc}: no headroom for u")
    rep["go_no_go"] = {"decision": "GO" if not reasons else "NO_GO", "reasons": reasons}

    if n and 0.02 < rate < 0.98:
        rep["power_projection"] = {
            "observed_failure_rate": rate,
            "note": ("Required n at 80% power, from the OBSERVED failure rate, using the "
                     "calibrated clustered test (conservative: actual alpha ~0.01 vs nominal 0.05)."),
            "for_delta_auroc_0.045": required_groups_for_power(rate, 0.5, 1, 0.8,
                                        [240, 480, 960, 1920], n_sims=80, n_boot=200)["chosen_n_trajectories"],
            "for_delta_auroc_0.095": required_groups_for_power(rate, 0.75, 1, 0.8,
                                        [240, 480, 960, 1920], n_sims=80, n_boot=200)["chosen_n_trajectories"],
        }

    txt = json.dumps(rep, indent=2, default=str)
    print(txt)
    out = a.out or a.jsonl.with_name("pilot_report.json")
    out.write_text(txt)
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
