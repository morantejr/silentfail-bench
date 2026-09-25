#!/usr/bin/env python3
"""Two checks the reviews asked for:
  1. VERIFY: are the stored u values (Table 1's primary row) computed with train-only rank norms,
     i.e. post rank-normalization-leak fix? Recompute from raw audit scores and compare.
  2. ROBUSTIFY: is the registered endpoint's bound an artifact of the LORO logistic? Recompute
     delta AUROC(M_c+u - M_c) four ways: pooled-OOF logistic (current), fold-averaged AUROC,
     a FIXED unit-weight combination of standardized scores (no fitted weights), and 5-fold
     repo-clustered CV instead of leave-one-repo-out.
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT / "src"))
from silentfail.combine import _fit_standardizer, fit_logistic_irls  # noqa: E402
from silentfail.scoring import fit_rank_norms, apply_panel_u  # noqa: E402
from silentfail.stats import auroc_fast, cluster_bootstrap_delta  # noqa: E402

def load(path, tag):
    rows = [json.loads(l) for l in Path(path).read_text().splitlines() if l.strip()]
    for r in rows: r["_bench"] = tag; r["_grp"] = f"{tag}:{r['repo']}"
    return rows

# ---------- 1. leak verification ----------
def verify(rows, tag):
    usable = [r for r in rows if not r["excluded"]]
    members = list(usable[0]["audit_scores"].keys())
    train_all = {m: [r["audit_scores"][m] for r in usable] for m in members}                      # LEAKY: all rows
    train_only = {m: [r["audit_scores"][m] for r in usable if r["split"] == "train"] for m in members}  # correct
    u_leaky = {r["instance_id"]: apply_panel_u(r["audit_scores"], fit_rank_norms(train_all))[1] for r in usable}
    u_train = {r["instance_id"]: apply_panel_u(r["audit_scores"], fit_rank_norms(train_only))[1] for r in usable}
    stored = {r["instance_id"]: r["u"] for r in usable}
    d_train = max(abs(stored[i] - u_train[i]) for i in stored); d_leaky = max(abs(stored[i] - u_leaky[i]) for i in stored)
    pop = [r for r in usable if r["visible_pass"]]; y = np.array([1 if r["g"] == 0 else 0 for r in pop])
    return {"benchmark": tag, "n_usable": len(usable), "n_train_rows_used_for_norms": len(train_only[members[0]]),
            "max_abs_diff_vs_train_only_recompute": round(d_train, 6), "max_abs_diff_vs_leaky_recompute": round(d_leaky, 6),
            "stored_matches_train_only": bool(d_train < 1e-9), "AUROC_u_stored": round(auroc_fast(y, np.array([stored[r["instance_id"]] for r in pop])), 3),
            "AUROC_u_if_leaky": round(auroc_fast(y, np.array([u_leaky[r["instance_id"]] for r in pop])), 3)}

# ---------- 2. endpoint robustness ----------
def folds_loro(grp): return [(grp == g) for g in np.unique(grp)]
def folds_k(grp, k=5, seed=0):
    u = np.unique(grp); rng = np.random.default_rng(seed); u = u[rng.permutation(len(u))]
    return [np.isin(grp, u[i::k]) for i in range(k)]

def oof(X, y, grp, folds):
    p = np.full(len(y), np.nan)
    for te in folds:
        tr = ~te
        if y[tr].sum() < 2 or (1 - y[tr]).sum() < 2: continue
        mu, sg = _fit_standardizer(X[tr]); coef, b = fit_logistic_irls((X[tr] - mu) / sg, y[tr].astype(float))
        p[te] = 1 / (1 + np.exp(-np.clip(((X[te] - mu) / sg) @ coef + b, -30, 30)))
    return p

def fixed_combo(X, y, grp, folds):
    """No fitted weights: z(c-part) summed with z(u-part), standardizer from the training fold only."""
    p = np.full(len(y), np.nan)
    for te in folds:
        tr = ~te
        mu, sg = _fit_standardizer(X[tr]); p[te] = ((X[te] - mu) / sg).sum(axis=1)
    return p

def fold_avg_delta(Xc, Xcu, y, grp, folds):
    """AUROC computed inside each fold and averaged by fold size: avoids cross-fold calibration scrambling."""
    num = den = 0.0
    for te in folds:
        tr = ~te
        if y[tr].sum() < 2 or (1 - y[tr]).sum() < 2 or not (0 < y[te].sum() < te.sum()): continue
        pc, pcu = oof(Xc, y, grp, [te]), oof(Xcu, y, grp, [te])
        w = te.sum(); num += w * (auroc_fast(y[te], pcu[te]) - auroc_fast(y[te], pc[te])); den += w
    return round(num / den, 3) if den else float("nan")

def endpoint(pop, label):
    y = np.array([1 if r["g"] == 0 else 0 for r in pop]); grp = np.array([r["_grp"] for r in pop])
    c = np.array([r["c"] for r in pop], float); u = np.array([r["u"] for r in pop], float)
    Xc, Xcu = c[:, None], np.column_stack([c, u])
    out = {"label": label, "n": len(y), "positives": int(y.sum()), "raw_AUROC_negc": round(auroc_fast(y, -c), 3), "raw_AUROC_u": round(auroc_fast(y, u), 3)}
    for name, folds in (("LORO", folds_loro(grp)), ("5fold_clustered", folds_k(grp))):
        pc, pcu = oof(Xc, y, grp, folds), oof(Xcu, y, grp, folds); ok = ~np.isnan(pc) & ~np.isnan(pcu)
        t = cluster_bootstrap_delta(y[ok], pc[ok], pcu[ok], grp[ok], endpoint="auroc", n_boot=4000, seed=11)
        out[f"logistic_{name}"] = {"AUROC_Mc": round(auroc_fast(y[ok], pc[ok]), 3), "AUROC_Mcu": round(auroc_fast(y[ok], pcu[ok]), 3),
                                   "delta": round(t.delta, 3), "CI95": [round(t.ci_lo, 3), round(t.ci_hi, 3)]}
    # fixed unit-weight combination: no fitted coefficients at all
    pc = fixed_combo((-c)[:, None], y, grp, folds_loro(grp)); pcu = fixed_combo(np.column_stack([-c, u]), y, grp, folds_loro(grp))
    t = cluster_bootstrap_delta(y, pc, pcu, grp, endpoint="auroc", n_boot=4000, seed=12)
    out["fixed_unit_weights_no_fitting"] = {"AUROC_negc": round(auroc_fast(y, pc), 3), "AUROC_negc_plus_u": round(auroc_fast(y, pcu), 3),
                                            "delta": round(t.delta, 3), "CI95": [round(t.ci_lo, 3), round(t.ci_hi, 3)]}
    out["fold_averaged_delta_LORO"] = fold_avg_delta(Xc, Xcu, y, grp, folds_loro(grp))
    return out

V = load(ROOT / "data/swebench_pilot50/pilot_free_scored.jsonl", "verified")
M = load(ROOT / "data/replication_multilingual/replication_scored.jsonl", "multilingual")
popV = [r for r in V if not r["excluded"] and r["visible_pass"]]; popM = [r for r in M if not r["excluded"] and r["visible_pass"]]
rep = {"leak_verification": [verify(V, "verified"), verify(M, "multilingual")],
       "endpoint_robustness": [endpoint(popV, "verified"), endpoint(popM, "multilingual"), endpoint(popV + popM, "pooled")]}
(ROOT / "data/verify_and_robustify.json").write_text(json.dumps(rep, indent=1)); print(json.dumps(rep, indent=1))

# ---------- addendum (review request): interval and fold count for the within-fold estimator ----------
def fold_avg_with_ci(pop, label, n_boot=2000, seed=13):
    y = np.array([1 if r["g"] == 0 else 0 for r in pop]); grp = np.array([r["_grp"] for r in pop])
    c = np.array([r["c"] for r in pop], float); u = np.array([r["u"] for r in pop], float)
    Xc, Xcu = c[:, None], np.column_stack([c, u])
    def stat(idx):
        yy, gg, xc, xcu = y[idx], grp[idx], Xc[idx], Xcu[idx]
        num = den = 0.0; used = 0
        for g in np.unique(gg):
            te = gg == g; tr = ~te
            if yy[tr].sum() < 2 or (1 - yy[tr]).sum() < 2 or not (0 < yy[te].sum() < te.sum()): continue
            pc, pcu = oof(xc, yy, gg, [te]), oof(xcu, yy, gg, [te])
            if np.isnan(pc[te]).any() or np.isnan(pcu[te]).any(): continue
            w = te.sum(); num += w * (auroc_fast(yy[te], pcu[te]) - auroc_fast(yy[te], pc[te])); den += w; used += 1
        return (num / den if den else np.nan), used, den
    point, used, covered = stat(np.arange(len(y)))
    uniq = np.unique(grp); idx_by = {g: np.flatnonzero(grp == g) for g in uniq}; rng = np.random.default_rng(seed); vals = []
    for _ in range(n_boot):
        drawn = rng.choice(uniq, size=len(uniq), replace=True)
        idx = np.concatenate([idx_by[g] for g in drawn])
        v, _, _ = stat(idx)
        if v == v: vals.append(v)
    return {"label": label, "delta": round(float(point), 3), "CI95": [round(float(np.percentile(vals, 2.5)), 3), round(float(np.percentile(vals, 97.5)), 3)] if vals else None,
            "folds_contributing": used, "n_repos_total": int(len(uniq)), "rows_covered": int(covered), "rows_total": len(y)}

if __name__ == "__main__":
    add = [fold_avg_with_ci(popV, "verified"), fold_avg_with_ci(popM, "multilingual"), fold_avg_with_ci(popV + popM, "pooled")]
    rep["fold_averaged_with_ci"] = add
    (ROOT / "data/verify_and_robustify.json").write_text(json.dumps(rep, indent=1))
    print(json.dumps(add, indent=1))
