#!/usr/bin/env python3
"""Amendment 4a: does u (or c) add anything beyond structure + public difficulty?

Leave-one-repo-out logistic models; out-of-fold probabilities pooled; AUROC.
In-sample numbers are deliberately not computed. See PREREGISTRATION Amendment 4.
"""
from __future__ import annotations

import argparse, json, sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from silentfail.combine import _fit_standardizer, fit_logistic_irls  # noqa: E402
from silentfail.stats import auroc_fast, cluster_bootstrap_delta  # noqa: E402

PUBLIC_RUNS = ["20260217_mini-v2.0.0_gpt-5-mini", "20260217_mini-v2.0.0_gemini-3-flash-high",
               "20260217_mini-v2.0.0_deepseek-3-2-high", "20260217_mini-v2.0.0_claude-4-5-sonnet-high",
               "20260217_mini-v2.0.0_gpt-5-2-high", "20260217_mini-v2.0.0_claude-4-6-opus"]
STRUCT = ["patch_added_lines", "step_count", "patch_chars", "patch_files_touched"]


def patch_feats(diff: str) -> tuple[int, int, int]:
    lines = diff.splitlines()
    added = sum(1 for l in lines if l.startswith("+") and not l.startswith("+++"))
    files = sum(1 for l in lines if l.startswith("diff --git"))
    return added, len(diff), files


def loro_oof(X: np.ndarray, y: np.ndarray, groups: np.ndarray) -> tuple[np.ndarray, list[str]]:
    """Out-of-fold P(fail) by leave-one-repo-out. Folds whose training set has <2 of either class are skipped."""
    p = np.full(len(y), np.nan); skipped = []
    for g in np.unique(groups):
        te = groups == g; tr = ~te
        if y[tr].sum() < 2 or (1 - y[tr]).sum() < 2:
            skipped.append(str(g)); continue
        mu, sg = _fit_standardizer(X[tr])
        coef, b = fit_logistic_irls((X[tr] - mu) / sg, y[tr].astype(float))
        z = ((X[te] - mu) / sg) @ coef + b
        p[te] = 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))
    return p, skipped


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", type=Path, default=ROOT / "data/swebench_pilot50/pilot_free_scored.jsonl")
    ap.add_argument("--public", type=Path, default=ROOT / "data/public_results")
    ap.add_argument("--out", type=Path, default=ROOT / "data/swebench_pilot50/prereg_readout_amendment4a.json")
    ap.add_argument("--exclude-run", nargs="*", default=[], help="SENSITIVITY (post-hoc): drop public runs, e.g. the gpt-5-mini run that shares our agent model")
    ap.add_argument("--no-difficulty", action="store_true", help="SENSITIVITY (post-hoc): structure-only base")
    a = ap.parse_args()
    rows = [json.loads(l) for l in a.rows.read_text().splitlines() if l.strip()]
    pop = [r for r in rows if not r["excluded"] and r.get("visible_pass")]
    y = np.array([1 if r["g"] == 0 else 0 for r in pop]); grp = np.array([r["repo"] for r in pop])
    # public difficulty: fraction of six independent runs that resolved the instance
    used_runs = [d for d in PUBLIC_RUNS if not any(x in d for x in a.exclude_run)]
    runs = [json.loads((a.public / f"{d}.json").read_text()) for d in used_runs]
    def solve_rate(iid: str) -> float:
        v = [str(rn[iid]["resolved"]).lower() == "true" for rn in runs if iid in rn]
        return float(np.mean(v)) if v else float("nan")
    diff = np.array([solve_rate(r["instance_id"]) for r in pop])
    assert not np.isnan(diff).any(), "instance missing from public results"
    pf = np.array([patch_feats(r["final_solution"] or "") for r in pop], float)
    feats = {"patch_added_lines": pf[:, 0], "patch_chars": pf[:, 1], "patch_files_touched": pf[:, 2],
             "step_count": np.array([r["step_count"] for r in pop], float),
             "difficulty_public_solve_rate": diff,
             "c": np.array([r["c"] for r in pop], float), "u": np.array([r["u"] for r in pop], float)}
    base = STRUCT if a.no_difficulty else STRUCT + ["difficulty_public_solve_rate"]
    models = {"M_base": base, "M_base+c": base + ["c"], "M_base+u": base + ["u"], "M_base+c+u": base + ["c", "u"]}
    variant = "PRE-REGISTERED" if not (a.exclude_run or a.no_difficulty) else f"SENSITIVITY (post-hoc): exclude_run={a.exclude_run} no_difficulty={a.no_difficulty}"
    rep = {"analysis": f"Amendment 4a [{variant}]: leave-one-repo-out out-of-fold AUROC; in-sample not computed", "public_runs_used": used_runs,
           "population_n": len(pop), "silent_failures": int(y.sum()), "n_repos": int(len(np.unique(grp))),
           "univariate_AUROC": {k: round(auroc_fast(y, v if k != "c" else -v), 3) for k, v in feats.items()},
           "oof": {}, "deltas": {}}
    oof = {}
    for name, cols in models.items():
        X = np.column_stack([feats[c] for c in cols])
        p, skipped = loro_oof(X, y, grp); oof[name] = p
        ok = ~np.isnan(p)
        rep["oof"][name] = {"features": cols, "AUROC_oof": round(auroc_fast(y[ok], p[ok]), 3), "rows_scored": int(ok.sum()), "folds_skipped": skipped}
    ok = np.all([~np.isnan(oof[m]) for m in models], axis=0)
    for aug, pred in (("M_base+u", "A4a-P1"), ("M_base+c", "A4a-P2"), ("M_base+c+u", "extra")):
        t = cluster_bootstrap_delta(y[ok], oof["M_base"][ok], oof[aug][ok], grp[ok], endpoint="auroc", n_boot=4000, seed=4)
        d = round(t.delta, 3)
        rep["deltas"][f"{aug} - M_base"] = {"prediction": pred, "delta_AUROC_oof": d, "CI95_repo_clustered": [round(t.ci_lo, 3), round(t.ci_hi, 3)],
                                             "reading": "conditional signal (>=0.05)" if d >= 0.05 else ("no incremental value (<=0.02)" if d <= 0.02 else "between thresholds")}
    rep["note"] = "CI is a cluster bootstrap over pooled out-of-fold predictions; models are not refit per draw."
    a.out.write_text(json.dumps(rep, indent=1)); print(json.dumps(rep, indent=1))


if __name__ == "__main__":
    main()
