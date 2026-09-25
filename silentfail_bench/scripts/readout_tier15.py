#!/usr/bin/env python3
"""Tier-1.5 readouts requested by external review ($0, existing scores):
  A. the proposal's confirmatory endpoint at last: LORO out-of-fold AUROC of M_{c+u} vs M_c, pooled and per benchmark
  B. review-budget operating points (recall / precision at 5, 10, 20% flagged) for u, -c, lines added, best judge
  C. alternative disagreement statistics on the same panel scores (range, MAD, max-min, entropy of rank norms)
  D. the widest upper interval in the configuration sweep (simultaneous bound)
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT / "src"))
from silentfail.combine import _fit_standardizer, fit_logistic_irls  # noqa: E402
from silentfail.stats import auroc_fast, cluster_bootstrap_delta  # noqa: E402

def load(path, tag):
    rows = [json.loads(l) for l in Path(path).read_text().splitlines() if l.strip()]
    pop = [r for r in rows if not r["excluded"] and r["visible_pass"]]
    for r in pop: r["_bench"] = tag; r["_grp"] = f"{tag}:{r['repo']}"
    return pop

def added(diff): return sum(1 for l in diff.splitlines() if l.startswith("+") and not l.startswith("+++"))

def loro(X, y, grp):
    p = np.full(len(y), np.nan)
    for g in np.unique(grp):
        te = grp == g; tr = ~te
        if y[tr].sum() < 2 or (1 - y[tr]).sum() < 2: continue
        mu, sg = _fit_standardizer(X[tr]); coef, b = fit_logistic_irls((X[tr] - mu) / sg, y[tr].astype(float))
        p[te] = 1 / (1 + np.exp(-np.clip(((X[te] - mu) / sg) @ coef + b, -30, 30)))
    return p

def endpoint(pop, label):
    y = np.array([1 if r["g"] == 0 else 0 for r in pop]); grp = np.array([r["_grp"] for r in pop])
    c = np.array([r["c"] for r in pop], float); u = np.array([r["u"] for r in pop], float)
    pc = loro(c[:, None], y, grp); pcu = loro(np.column_stack([c, u]), y, grp); ok = ~np.isnan(pc) & ~np.isnan(pcu)
    t = cluster_bootstrap_delta(y[ok], pc[ok], pcu[ok], grp[ok], endpoint="auroc", n_boot=4000, seed=11)
    return {"label": label, "n": int(ok.sum()), "positives": int(y[ok].sum()), "AUROC_Mc_oof": round(auroc_fast(y[ok], pc[ok]), 3), "AUROC_Mcu_oof": round(auroc_fast(y[ok], pcu[ok]), 3),
            "delta_AUROC_oof": round(t.delta, 3), "CI95_repo_clustered": [round(t.ci_lo, 3), round(t.ci_hi, 3)]}

def budgets(pop, label, judges=None):
    y = np.array([1 if r["g"] == 0 else 0 for r in pop]); n = len(y); out = {}
    sig = {"u": np.array([r["u"] for r in pop]), "-c": -np.array([r["c"] for r in pop]), "lines_added": np.array([added(r["final_solution"]) for r in pop], float)}
    if judges: sig["best_judge_J1"] = np.array([judges.get(r["instance_id"], np.nan) for r in pop]); sig["best_judge_J1"] = np.where(np.isnan(sig["best_judge_J1"]), np.nanmedian(sig["best_judge_J1"]), sig["best_judge_J1"])
    for name, s in sig.items():
        order = np.argsort(-s); row = {}
        for b in (0.05, 0.10, 0.20):
            k = max(1, int(round(b * n))); top = order[:k]; tp = int(y[top].sum())
            row[f"{int(b*100)}%"] = {"flagged": k, "failures_caught": tp, "recall": round(tp / y.sum(), 3), "precision": round(tp / k, 3)}
        out[name] = row
    out["_prevalence"] = round(float(y.mean()), 3); out["_n"] = n; return {label: out}

def alt_u(pop, label):
    y = np.array([1 if r["g"] == 0 else 0 for r in pop]); grp = [r["_grp"] for r in pop]; out = {}
    R = [np.array(list(r["audit_scores_rank_norm"].values()), float) for r in pop]
    stats = {"std_registered": lambda v: v.std(ddof=1), "range": lambda v: v.max() - v.min(), "mad": lambda v: np.mean(np.abs(v - v.mean())),
             "entropy_of_normalized": lambda v: -np.sum((v + 1e-9) / (v.sum() + 4e-9) * np.log((v + 1e-9) / (v.sum() + 4e-9))), "mean_rank_norm": lambda v: v.mean(), "min_rank_norm": lambda v: v.min()}
    for k, f in stats.items():
        s = np.array([f(v) for v in R]); out[k] = round(auroc_fast(y, s if k not in ("mean_rank_norm", "min_rank_norm") else -s), 3)
    return {label: out}

def judges_map(cache, cond="J1", judge="gpt-5"):
    m = {}
    for l in Path(cache).read_text().splitlines():
        if l.strip():
            d = json.loads(l)
            if d["cond"] == cond and d["judge"] == judge and d.get("p_resolved") is not None: m[d["instance_id"]] = 100 - d["p_resolved"]
    return m

V = load(ROOT / "data/swebench_pilot50/pilot_free_scored.jsonl", "verified"); M = load(ROOT / "data/replication_multilingual/replication_scored.jsonl", "multilingual")
jv = judges_map(ROOT / "data/swebench_pilot50/llm_judges/judgments.jsonl"); jm = judges_map(ROOT / "data/replication_multilingual/llm_judges/judgments.jsonl")
rep = {"A_confirmatory_endpoint_LORO": [endpoint(V, "verified"), endpoint(M, "multilingual"), endpoint(V + M, "pooled")],
       "B_operating_points": {**budgets(V, "verified", jv), **budgets(M, "multilingual", jm)},
       "C_alternative_u_statistics": {**alt_u(V, "verified"), **alt_u(M, "multilingual")},
       "D_simultaneous_bound": {"primary_protocol_M3_upper": {"verified": 0.219, "multilingual": 0.217}, "widest_upper_in_sweep": {"row": "code-trained critic selecting; 5-RM panel (Verified)", "upper": 0.452},
                                "note": "the +0.22 bound is per-protocol; the widest interval in the Table 1 sweep reaches +0.45"},
       "note": "A is the proposal's confirmatory endpoint computed with leave-one-repository-out in place of the 70/30 split that never reached 10 test positives; declared post-hoc substitution of the CV scheme, same estimand."}
out = ROOT / "data/readout_tier15.json"; out.write_text(json.dumps(rep, indent=1)); print(json.dumps(rep, indent=1))
