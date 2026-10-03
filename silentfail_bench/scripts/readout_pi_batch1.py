#!/usr/bin/env python3
"""Three analyses a PI would ask for, all from data already on disk.

E2  the proposal's own mechanism experiment: at matched panel quality, does
    lower error correlation buy more residual detection value?
LC  is it disagreement specifically, or the panel entirely? Feed the four raw
    panel scores to a learned combiner instead of only their spread.
CD  are the four candidates actually different from each other? If not, that
    explains mechanically why selection concentrated nothing (Amendment 4f).
"""
from __future__ import annotations
import json, sys, itertools
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT / "src"))
from silentfail.combine import _fit_standardizer, fit_logistic_irls  # noqa: E402
from silentfail.scoring import fit_rank_norms, apply_panel_u  # noqa: E402
from silentfail.stats import auroc_fast  # noqa: E402

BENCH = {"verified": ("data/swebench_pilot50/pilot_free_scored.jsonl", "data/swebench_pilot50/l4/out/gpu_scores_16k",
                      "data/swebench_pilot50/rescoring_bundle.jsonl"),
         "multilingual": ("data/replication_multilingual/replication_scored.jsonl", "data/replication_multilingual/modal/prefix4k",
                          "data/replication_multilingual/rescoring_bundle.jsonl")}

def load(tag):
    rowf, scored, bundlef = BENCH[tag]
    rows = {r["instance_id"]: r for r in map(json.loads, (ROOT/rowf).read_text().splitlines()) if r}
    S = {f.stem.replace("scores_gpu_", ""): json.loads(f.read_text())["scores"] for f in (ROOT/scored).glob("scores_gpu_*.json")}
    cands = {}
    for c in map(json.loads, (ROOT/bundlef).read_text().splitlines()):
        cands.setdefault(c["instance_id"], {})[c["candidate"]] = c
    pop = [i for i, r in rows.items() if not r["excluded"] and r.get("visible_pass")]
    return rows, S, cands, pop

def loro_oof(X, y, grp):
    p = np.full(len(y), np.nan)
    for g in np.unique(grp):
        te = grp == g; tr = ~te
        if y[tr].sum() < 2 or (1-y[tr]).sum() < 2: continue
        mu, sg = _fit_standardizer(X[tr]); co, b = fit_logistic_irls((X[tr]-mu)/sg, y[tr].astype(float))
        p[te] = 1/(1+np.exp(-np.clip(((X[te]-mu)/sg)@co+b, -30, 30)))
    return p

def fixed_oof(X, y, grp):
    p = np.full(len(y), np.nan)
    for g in np.unique(grp):
        te = grp == g; mu, sg = _fit_standardizer(X[~te]); p[te] = ((X[te]-mu)/sg).sum(axis=1)
    return p

rep = {}
for tag in BENCH:
    rows, S, cands, pop = load(tag)
    key = lambda i: f"{i}#{rows[i]['selected_candidate_index']}"
    y = np.array([1 if rows[i]["g"] == 0 else 0 for i in pop]); grp = np.array([rows[i]["repo"] for i in pop])
    critics = sorted(S)
    M = {m: np.array([S[m].get(key(i), {}).get("score", np.nan) for i in pop], float) for m in critics}
    M = {m: np.where(np.isnan(v), np.nanmedian(v), v) for m, v in M.items()}
    train = np.array([rows[i]["split"] == "train" for i in pop])
    out = {"n": len(pop), "failures": int(y.sum()), "critics": critics}

    # ---------- E2 ----------
    rn = {}
    for m in critics:
        fit = fit_rank_norms({m: list(M[m][train])})[m]
        rn[m] = np.array([fit.transform(v) for v in M[m]])
    err = {m: rn[m] - (1 - y) for m in critics}          # predicted-pass minus actual-pass
    panels = []
    for combo in itertools.combinations(critics, 4):
        q = float(np.mean([auroc_fast(y, -M[m]) for m in combo]))
        ec = float(np.mean([np.corrcoef(err[a], err[b])[0, 1] for a, b in itertools.combinations(combo, 2)]))
        fits = fit_rank_norms({m: list(M[m][train]) for m in combo})
        u = np.array([apply_panel_u({m: M[m][k] for m in combo}, fits)[1] for k in range(len(pop))])
        panels.append({"members": list(combo), "quality": round(q, 3), "error_corr": round(ec, 3), "AUROC_u": round(auroc_fast(y, u), 3)})
    qs = np.array([p["quality"] for p in panels]); ecs = np.array([p["error_corr"] for p in panels]); us = np.array([p["AUROC_u"] for p in panels])
    # partial correlation of AUROC(u) with error correlation, controlling for quality
    def resid(a, b):
        A = np.column_stack([np.ones(len(b)), b]); return a - A @ np.linalg.lstsq(A, a, rcond=None)[0]
    pc = float(np.corrcoef(resid(us, qs), resid(ecs, qs))[0, 1])
    out["E2_diversity"] = {"n_panels": len(panels), "error_corr_range": [round(float(ecs.min()), 3), round(float(ecs.max()), 3)],
        "quality_range": [round(float(qs.min()), 3), round(float(qs.max()), 3)],
        "AUROC_u_range": [round(float(us.min()), 3), round(float(us.max()), 3)],
        "corr_AUROCu_vs_error_corr": round(float(np.corrcoef(us, ecs)[0, 1]), 3),
        "partial_corr_controlling_quality": round(pc, 3),
        "proposal_predicted": "negative (lower error correlation -> higher AUROC(u))",
        "best_panel": max(panels, key=lambda p: p["AUROC_u"]), "worst_panel": min(panels, key=lambda p: p["AUROC_u"])}

    # ---------- LC: learned combiner ----------
    reg = ["ray2333_grm_gemma2_2b_rewardmodel_ft", "ray2333_grm_llama3_2_3b_rewardmodel_ft",
           "skywork_skywork_reward_v2_llama_3_2_1b", "skywork_skywork_reward_v2_qwen3_1_7b"]
    reg = [m for m in reg if m in M]
    X4 = np.column_stack([M[m] for m in reg])
    u_reg = np.array([r["u"] for r in (rows[i] for i in pop)])
    c = np.array([rows[i]["c"] for i in pop], float)
    # DEFECTS #23 (3 Oct 2026): the fixed-weight sum of RAW reward-model scores (and of c) is higher-is-better, i.e. a
    # PASS score, but it was scored as a failure detector without negation, so the published fixed_weights_oof for the
    # raw-panel and c rows are 1 - AUROC. The logistic rows are sign-free (the fit learns the direction). The published
    # key is kept, unchanged, for traceability; fixed_weights_oof_detect is the same combination in the detection
    # direction (reward-model columns and c negated; u is already a detection score) and is what the paper now reports.
    lc = {}
    for name, X, Xd in (("u alone (published)", u_reg[:, None], u_reg[:, None]), ("4 raw panel scores", X4, -X4),
                        ("c alone", c[:, None], -c[:, None]), ("c + u", np.column_stack([c, u_reg]), np.column_stack([-c, u_reg])),
                        ("c + 4 raw panel scores", np.column_stack([c, X4]), np.column_stack([-c, -X4]))):
        p_l, p_f, p_d = loro_oof(X, y, grp), fixed_oof(X, y, grp), fixed_oof(Xd, y, grp)
        ok = ~np.isnan(p_l)
        lc[name] = {"logistic_oof": round(auroc_fast(y[ok], p_l[ok]), 3), "fixed_weights_oof": round(auroc_fast(y, p_f), 3),
                    "fixed_weights_oof_detect": round(auroc_fast(y, p_d), 3)}
    lc["_note"] = ("fixed_weights_oof sums standardized columns as given (raw reward-model scores and c are pass-direction; "
                   "DEFECTS #23); fixed_weights_oof_detect negates them so every row is a failure detector. Use _detect.")
    out["LC_learned_combiner"] = lc

    # ---------- CD: candidate diversity ----------
    def toks(s): return set(l.strip() for l in (s or "").splitlines() if l.startswith(("+", "-")) and not l.startswith(("+++", "---")))
    jac, spread = [], []
    for i in pop:
        ps = [toks(cands[i][k]["final_solution"]) for k in sorted(cands.get(i, {})) if cands[i][k]["final_solution"].strip()]
        if len(ps) >= 2:
            sims = [len(a & b)/max(len(a | b), 1) for a, b in itertools.combinations(ps, 2)]
            jac.append(float(np.mean(sims)))
        cs = [v for v in rows[i]["candidate_scores"] if v == v and abs(v) != float("inf")]
        if len(cs) >= 2: spread.append(float(np.std(cs, ddof=1)))
    out["CD_candidate_diversity"] = {
        "instances_with_2plus_patches": len(jac),
        "mean_pairwise_patch_overlap": round(float(np.mean(jac)), 3),
        "median_pairwise_patch_overlap": round(float(np.median(jac)), 3),
        "frac_instances_with_near_identical_patches_overlap_gt_0.9": round(float(np.mean(np.array(jac) > 0.9)), 3),
        "mean_selector_score_spread": round(float(np.mean(spread)), 3),
        "reading": "overlap 1.0 = the candidates are the same patch; 0.0 = no shared changed lines"}
    rep[tag] = out

(ROOT/"data/readout_pi_batch1.json").write_text(json.dumps(rep, indent=1))
for tag, o in rep.items():
    print(f"\n=== {tag}: n={o['n']}, failures={o['failures']} ===")
    e = o["E2_diversity"]
    print(f"  E2  {e['n_panels']} four-critic panels | error corr {e['error_corr_range']} | AUROC(u) {e['AUROC_u_range']}")
    print(f"      corr(AUROC_u, error_corr) = {e['corr_AUROCu_vs_error_corr']:+.3f}; controlling for quality {e['partial_corr_controlling_quality']:+.3f}")
    print(f"      proposal predicted NEGATIVE (less correlated errors -> better u)")
    print(f"  LC  learned combiner vs disagreement (out-of-fold):")
    for k, v in o["LC_learned_combiner"].items():
        if k.startswith("_"): continue
        print(f"      {k:26s} logistic {v['logistic_oof']:.3f}   fixed-weights (as published, pass-direction for raw/c) {v['fixed_weights_oof']:.3f}   fixed-weights detect {v['fixed_weights_oof_detect']:.3f}")
    d = o["CD_candidate_diversity"]
    print(f"  CD  mean pairwise patch overlap {d['mean_pairwise_patch_overlap']:.3f} (median {d['median_pairwise_patch_overlap']:.3f});"
          f" near-identical in {d['frac_instances_with_near_identical_patches_overlap_gt_0.9']:.0%} of instances")
