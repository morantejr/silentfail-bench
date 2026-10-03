#!/usr/bin/env python3
"""Critic-demeaned disagreement u_perp on the committed rows. Post-hoc analysis, 3 October 2026.

u_perp_i = sd_j( r_ji - alpha_hat_j ),  alpha_hat_j = mean of critic j over split=="train" rows only.

Matrices (stated before any number was seen):
  Verified      primary 16k pass (l4/out/gpu_scores_16k, 7 critics); like-for-like = the rows' own
                audit_scores (the September MPS pass, 4 critics), which is what published u was computed on.
  Multilingual  primary 16k pass (modal/full16k, 7 critics); like-for-like = modal/prefix4k, which is what
                published u and the batch-1 E2/LC analysis used.
Panels: the registered four, and the zoo of seven (where the matrix has seven).
Population: the batch-1 committed population (not excluded and visible_pass): Verified n=34, Multilingual n=77.
Estimators, folds, bootstrap sizes and seeds are imported from / copied verbatim from the project's own
readout_amendment7.py (endpoint) and verify_and_robustify.py (within-fold size-weighted). No DeLong.
Writes for_claude/algorithmic_to_7/results.json deterministically.
"""
from __future__ import annotations
import hashlib, itertools, json, sys
from pathlib import Path
import numpy as np

ROOT = Path("/Users/jmorante/Downloads/xtail_spec/silentfail_bench")
OUT = Path(__file__).resolve().parent / "results.json"
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "scripts"))
from silentfail.combine import _fit_standardizer, fit_logistic_irls          # noqa: E402
from silentfail.scoring import fit_rank_norms, apply_panel_u                 # noqa: E402
from silentfail.stats import auroc_fast, cluster_bootstrap_delta             # noqa: E402
from readout_amendment7 import boot, folds_loro, folds_k, oof, fixed_combo   # noqa: E402  (the paper's estimators)

PANEL = ["ray2333_grm_gemma2_2b_rewardmodel_ft", "ray2333_grm_llama3_2_3b_rewardmodel_ft",
         "skywork_skywork_reward_v2_llama_3_2_1b", "skywork_skywork_reward_v2_qwen3_1_7b"]
AUDIT_TO_PANEL = {"verified": {"audit_0": PANEL[0], "audit_1": PANEL[1], "audit_2": PANEL[2], "audit_3": PANEL[3]},
                  "multilingual": {"grm_gemma2_2b": PANEL[0], "grm_llama3_2_3b": PANEL[1],
                                   "skywork_llama3_2_1b": PANEL[2], "skywork_qwen3_1_7b": PANEL[3]}}
BENCH = {
    "verified": {"rows": "data/swebench_pilot50/pilot_free_scored.jsonl",
                 "matrices": {"16k_primary": "data/swebench_pilot50/l4/out/gpu_scores_16k", "rows_audit_scores_like_for_like": None}},
    "multilingual": {"rows": "data/replication_multilingual/replication_scored.jsonl",
                     "matrices": {"16k_primary": "data/replication_multilingual/modal/full16k", "prefix4k_like_for_like": "data/replication_multilingual/modal/prefix4k"}},
}
SEED_ENDPOINT, SEED_FIXED, N_BOOT = 21, 22, 4000            # readout_amendment7.endpoint
SEED_FOLDAVG, N_BOOT_FOLDAVG = 13, 2000                      # verify_and_robustify.fold_avg_with_ci


def r3(x): return None if x is None or (isinstance(x, float) and not np.isfinite(x)) else round(float(x), 3)


def load_rows(tag):
    rows = {}
    for l in (ROOT / BENCH[tag]["rows"]).read_text().splitlines():
        if l.strip(): r = json.loads(l); rows[r["instance_id"]] = r
    pop = [i for i in rows if not rows[i]["excluded"] and rows[i].get("visible_pass")]
    return rows, pop


def matrix_from_dir(rows, pop, d):
    S = {f.stem.replace("scores_gpu_", ""): json.loads(f.read_text())["scores"] for f in sorted((ROOT / d).glob("scores_gpu_*.json"))}
    crit = sorted(S); key = lambda i: f"{i}#{rows[i]['selected_candidate_index']}"
    R = np.array([[S[m].get(key(i), {}).get("score", np.nan) for m in crit] for i in pop], float)
    return crit, R


def matrix_from_audit(rows, pop, tag):
    names = sorted(rows[pop[0]]["audit_scores"]); crit = [AUDIT_TO_PANEL[tag][k] for k in names]
    R = np.array([[rows[i]["audit_scores"][k] for k in names] for i in pop], float)
    order = np.argsort(crit); return [crit[o] for o in order], R[:, order]


def ss_decomposition(R):
    """Two-way additive sum of squares on the analysed matrix (descriptive; all analysed rows)."""
    n, J = R.shape; gm = R.mean(); cm = R.mean(axis=0); tm = R.mean(axis=1)
    ss_c = n * ((cm - gm) ** 2).sum(); ss_t = J * ((tm - gm) ** 2).sum()
    resid = R - cm[None, :] - tm[:, None] + gm; ss_r = (resid ** 2).sum(); tot = ((R - gm) ** 2).sum()
    ms_t, ms_r = ss_t / (n - 1), ss_r / ((n - 1) * (J - 1))
    icc_c1 = (ms_t - ms_r) / (ms_t + (J - 1) * ms_r)        # consistency ICC, Shrout-Fleiss ICC(3,1): critic shifts ignored
    return {"n": n, "J": J, "SS_critic": r3(ss_c), "SS_trajectory": r3(ss_t), "SS_residual": r3(ss_r), "SS_total": r3(tot),
            "share_critic": r3(ss_c / tot), "share_trajectory": r3(ss_t / tot), "share_residual": r3(ss_r / tot),
            "df": {"critic": J - 1, "trajectory": n - 1, "residual": (n - 1) * (J - 1)},
            "MS_trajectory": r3(ms_t), "MS_residual": r3(ms_r), "ICC_C1_consistency": r3(icc_c1),
            "note": "SS_trajectory carries q_i + c_i (true quality plus common-mode error), which the matrix cannot separate; "
                    "ICC(C,1) is the trajectory-level share after removing critic location, i.e. the relative-error view (Brennan)."}, resid


def fold_avg_delta(y, grp, Xc, Xcu, n_boot=N_BOOT_FOLDAVG, seed=SEED_FOLDAVG):
    """verify_and_robustify.fold_avg_with_ci, verbatim logic: LORO folds, per-fold AUROC delta weighted by fold size, repo bootstrap."""
    def stat(idx):
        yy, gg, xc, xcu = y[idx], grp[idx], Xc[idx], Xcu[idx]
        num = den = 0.0; used = 0
        for g in np.unique(gg):
            te = gg == g; tr = ~te
            if yy[tr].sum() < 2 or (1 - yy[tr]).sum() < 2 or not (0 < yy[te].sum() < te.sum()): continue
            pc, pcu = oof(xc, yy, [te]), oof(xcu, yy, [te])
            if np.isnan(pc[te]).any() or np.isnan(pcu[te]).any(): continue
            w = te.sum(); num += w * (auroc_fast(yy[te], pcu[te]) - auroc_fast(yy[te], pc[te])); den += w; used += 1
        return (num / den if den else np.nan), used
    point, used = stat(np.arange(len(y)))
    uniq = np.unique(grp); idx_by = {g: np.flatnonzero(grp == g) for g in uniq}; rng = np.random.default_rng(seed); vals = []
    for _ in range(n_boot):
        drawn = rng.choice(uniq, size=len(uniq), replace=True); v, _ = stat(np.concatenate([idx_by[g] for g in drawn]))
        if v == v: vals.append(v)
    return {"delta": r3(point), "CI95": [r3(np.percentile(vals, 2.5)), r3(np.percentile(vals, 97.5))] if vals else None, "folds_contributing": used}


def endpoint_four(y, cdet, feat, grp):
    """readout_amendment7.endpoint's three estimators + the within-fold size-weighted one; adding `feat` to the detection score."""
    Xc, Xcu = cdet[:, None], np.column_stack([cdet, feat]); est = {}
    for name, folds in (("logistic_LORO", folds_loro(grp)), ("logistic_5fold", folds_k(grp))):
        pc, pcu = oof(Xc, y, folds), oof(Xcu, y, folds); ok = ~np.isnan(pc) & ~np.isnan(pcu)
        t = cluster_bootstrap_delta(y[ok], pc[ok], pcu[ok], grp[ok], endpoint="auroc", n_boot=N_BOOT, seed=SEED_ENDPOINT)
        est[name] = {"baseline_AUROC": r3(auroc_fast(y[ok], pc[ok])), "delta": r3(t.delta), "CI95": [r3(t.ci_lo), r3(t.ci_hi)], "rows_used": int(ok.sum())}
    est["within_fold_size_weighted"] = fold_avg_delta(y, grp, Xc, Xcu)
    pc, pcu = fixed_combo(Xc, y, folds_loro(grp)), fixed_combo(Xcu, y, folds_loro(grp))
    t = cluster_bootstrap_delta(y, pc, pcu, grp, endpoint="auroc", n_boot=N_BOOT, seed=SEED_FIXED)
    est["fixed_unit_weights_PRIMARY"] = {"baseline_AUROC": r3(auroc_fast(y, pc)), "delta": r3(t.delta), "CI95": [r3(t.ci_lo), r3(t.ci_hi)]}
    return est


def refit_bootstrap_fixed_weights(y, cdet, R, train, grp, n_boot=N_BOOT, seed=SEED_FIXED):
    """Sensitivity: repo-clustered bootstrap that REFITS the critic means on the draw's training rows, recomputes
    u_perp for the draw, and takes the fixed-unit-weights LORO delta within the draw. Same seed and size as the
    primary estimator so the two intervals differ only by the refit."""
    uniq = np.unique(grp); idx_by = {g: np.flatnonzero(grp == g) for g in uniq}; rng = np.random.default_rng(seed); vals = []
    for _ in range(n_boot):
        drawn = rng.choice(uniq, size=len(uniq), replace=True); idx = np.concatenate([idx_by[g] for g in drawn])
        yb, gb, cb, Rb, tb = y[idx], grp[idx], cdet[idx], R[idx], train[idx]
        if yb.sum() == 0 or yb.sum() == len(yb) or tb.sum() < 2: continue
        alpha = Rb[tb].mean(axis=0); ub = (Rb - alpha).std(axis=1, ddof=1)
        pc, pcu = fixed_combo(cb[:, None], yb, folds_loro(gb)), fixed_combo(np.column_stack([cb, ub]), yb, folds_loro(gb))
        d = auroc_fast(yb, pcu) - auroc_fast(yb, pc)
        if np.isfinite(d): vals.append(d)
    return {"CI95_refit": [r3(np.percentile(vals, 2.5)), r3(np.percentile(vals, 97.5))], "draws_valid": len(vals)}


def error_corr_ranges(y, R, crit, train):
    """readout_pi_batch1 E2 definition: err_j = rank_norm_j(score) - (1 - y), rank norms fit on train rows; mean pairwise
    correlation over every four-member subset. Computed on the raw scores and on the critic-demeaned scores."""
    def rank_norm(M):
        rn = {}
        for k, m in enumerate(crit):
            fit = fit_rank_norms({m: list(M[train, k])})[m]; rn[m] = np.array([fit.transform(v) for v in M[:, k]])
        return rn
    def ranges_err(yy, rn):
        err = {m: rn[m] - (1 - yy) for m in crit}; ecs = []
        for combo in itertools.combinations(crit, min(4, len(crit))):
            ecs.append(float(np.mean([np.corrcoef(err[a], err[b])[0, 1] for a, b in itertools.combinations(combo, 2)])))
        return np.array(ecs)
    def ranges(M): return ranges_err(y, rank_norm(M))
    alpha = R[train].mean(axis=0)
    before, after = ranges(R), ranges(R - alpha[None, :])
    # Uninformative-critic floor: err_j = rn_j - (1 - y) shares the label term across critics. If the rank scores were
    # independent of y and of each other, corr(err_a, err_b) = Var(y) / sqrt((Var(rn_a)+Var(y)) (Var(rn_b)+Var(y))).
    rn = rank_norm(R)
    vy = float(np.var(y, ddof=1)); vr = {m: float(np.var(rn[m], ddof=1)) for m in crit}
    floors = []
    for combo in itertools.combinations(crit, min(4, len(crit))):
        floors.append(float(np.mean([vy / np.sqrt((vr[a] + vy) * (vr[b] + vy)) for a, b in itertools.combinations(combo, 2)])))
    floors = np.array(floors)
    # Label permutation (Cursor review, 3 Oct): removes label dependence but keeps between-critic rank dependence.
    # It is NOT the independence floor; the gap between the two is dependence among the rank scores themselves.
    rng = np.random.default_rng(0); pmin, pmax = [], []
    for _ in range(200):
        ep = ranges_err(rng.permutation(y), rn); pmin.append(ep.min()); pmax.append(ep.max())
    return {"n_subsets": int(len(before)), "before_demeaning": [r3(before.min()), r3(before.max())],
            "after_demeaning": [r3(after.min()), r3(after.max())], "max_abs_change": float(np.abs(before - after).max()),
            "uninformative_critic_floor": [r3(floors.min()), r3(floors.max())],
            "label_permutation_range_mean": [r3(np.mean(pmin)), r3(np.mean(pmax))],
            "floor_note": "independence floor: the E2 error correlation if every critic's rank score were independent of the label and of "
                          "the other critics. A permutation of y does not reproduce it, because between-critic rank correlation remains; "
                          "the upper edge of the observed band survives permutation.",
            "why": "the E2 error uses per-critic empirical-CDF ranks fit on train rows; a per-critic shift leaves those ranks "
                   "unchanged, so critic demeaning cannot move this correlation. Pearson pairwise correlation of raw columns is "
                   "shift-invariant per column for the same reason."}


def raw_pairwise_corr(R, resid):
    J = R.shape[1]; pairs = list(itertools.combinations(range(J), 2))
    c_raw = np.array([np.corrcoef(R[:, a], R[:, b])[0, 1] for a, b in pairs])
    c_res = np.array([np.corrcoef(resid[:, a], resid[:, b])[0, 1] for a, b in pairs])
    return {"raw_scores_pairwise_corr_range": [r3(c_raw.min()), r3(c_raw.max())], "raw_scores_pairwise_corr_mean": r3(c_raw.mean()),
            "double_centered_residual_pairwise_corr_range": [r3(c_res.min()), r3(c_res.max())], "double_centered_residual_pairwise_corr_mean": r3(c_res.mean()),
            "note": "residual correlations are mechanically negative on average (sum-to-zero across critics, expectation -1/(J-1)); "
                    "they are a diagnostic of what remains after removing both main effects, not an estimate of shared error."}


def analyse(tag, mname, crit, R, rows, pop):
    y = np.array([1 if rows[i]["g"] == 0 else 0 for i in pop]); grp = np.array([rows[i]["repo"] for i in pop])
    train = np.array([rows[i]["split"] == "train" for i in pop]); cdet = -np.array([rows[i]["c"] for i in pop], float)
    u_pub = np.array([rows[i]["u"] for i in pop], float)
    out = {"matrix": mname, "critics": crit, "J": len(crit), "n": int(len(pop)), "failures": int(y.sum()), "repositories": int(len(np.unique(grp))),
           "train_rows": int(train.sum()), "missing_cells": int(np.isnan(R).sum()), "missing_rule": "abort if any cell is missing (none were)"}
    if np.isnan(R).any(): raise SystemExit(f"{tag}/{mname}: {int(np.isnan(R).sum())} missing cells")
    panels = {"panel4": [crit.index(m) for m in PANEL if m in crit]}
    if len(crit) == 7: panels["zoo7"] = list(range(7))
    out["AUROC_c_detection"] = r3(auroc_fast(y, cdet)); out["AUROC_u_published"] = r3(auroc_fast(y, u_pub))
    out["sign_check"] = "detection AUROC of -c above 0.5" if auroc_fast(y, cdet) > 0.5 else "WARNING: detection AUROC of -c below 0.5"
    out["panels"] = {}
    for pname, cols in panels.items():
        if len(cols) < 2: continue
        Rp = R[:, cols]; cp = [crit[k] for k in cols]
        alpha = Rp[train].mean(axis=0)
        u_perp = (Rp - alpha[None, :]).std(axis=1, ddof=1)
        fits = fit_rank_norms({m: list(Rp[train, k]) for k, m in enumerate(cp)})
        u_rec = np.array([apply_panel_u({m: Rp[i, k] for k, m in enumerate(cp)}, fits)[1] for i in range(len(pop))])
        ss, resid = ss_decomposition(Rp)
        # identity checks
        alpha_all = Rp.mean(axis=0); tm = Rp.mean(axis=1)
        dc_train = Rp - alpha[None, :] - tm[:, None] + Rp.mean()        # double-centred with the SAME (train-fit) critic means
        dc_all = Rp - alpha_all[None, :] - tm[:, None] + Rp.mean()      # textbook double-centring with all-row critic means
        ident = {"max_abs_diff_sd(dc_train_fit) - u_perp": float(np.abs(dc_train.std(axis=1, ddof=1) - u_perp).max()),
                 "max_abs_diff_sd(dc_all_rows) - u_perp": float(np.abs(dc_all.std(axis=1, ddof=1) - u_perp).max()),
                 "max_abs_diff_sd(dc_all_rows) - sd(critic_demeaned_all_rows)": float(np.abs(dc_all.std(axis=1, ddof=1) - (Rp - alpha_all).std(axis=1, ddof=1)).max()),
                 "reading": "spread after double-centring equals spread after critic demeaning whenever the same critic means are used "
                            "(the trajectory mean is constant across critics); the second line differs only because all-row and train-only "
                            "critic means differ, which is the train/held-out rule, not a bug"}
        # rank scale: the published u is sd of per-critic ECDF ranks fit on train. Each critic's train-mean rank is the
        # same constant (n_train+1)/(2 n_train), so critic-demeaning the rank matrix cannot change its within-row sd:
        # published u IS critic-demeaned disagreement on the rank scale. Verified numerically here.
        RN = np.column_stack([np.array([fits[m].transform(v) for v in Rp[:, k]]) for k, m in enumerate(cp)])
        alpha_rn = RN[train].mean(axis=0); u_perp_rank = (RN - alpha_rn[None, :]).std(axis=1, ddof=1)
        ss_rank, _ = ss_decomposition(RN)
        ident["max_abs_diff_u_perp_on_rank_scale - u_recomputed"] = float(np.abs(u_perp_rank - u_rec).max())
        ident["train_mean_rank_per_critic"] = [r3(a) for a in alpha_rn]
        p = {"members": cp, "alpha_hat_train": [r3(a) for a in alpha], "ss": ss, "ss_rank_normalized": ss_rank, "identity": ident,
             "AUROC_u_recomputed_rank_norm_sd": r3(auroc_fast(y, u_rec)), "AUROC_u_recomputed_CI": boot(y, u_rec, grp),
             "AUROC_u_perp": r3(auroc_fast(y, u_perp)), "AUROC_u_perp_CI": boot(y, u_perp, grp),
             "corr_u_perp_vs_u_recomputed": r3(np.corrcoef(u_perp, u_rec)[0, 1]),
             "corr_u_perp_vs_u_published": r3(np.corrcoef(u_perp, u_pub)[0, 1]),
             "endpoint_add_u_perp_to_c": endpoint_four(y, cdet, u_perp, grp),
             "endpoint_add_u_recomputed_to_c": endpoint_four(y, cdet, u_rec, grp),
             "refit_bootstrap_fixed_unit_weights": refit_bootstrap_fixed_weights(y, cdet, Rp, train, grp),
             "pairwise": raw_pairwise_corr(Rp, resid)}
        if pname == "panel4":
            p["endpoint_add_u_published_to_c"] = endpoint_four(y, cdet, u_pub, grp)
        if len(cp) == 7 or pname == "panel4":
            p["E2_error_corr"] = error_corr_ranges(y, Rp, cp, train)
        out["panels"][pname] = p
    return out


def main():
    rep = {"analysis": "critic-demeaned disagreement u_perp, post hoc, 3 October 2026",
           "conventions": {"positive_class": "failure, y = 1[g == 0]", "selector": "-c (reward-model c is higher-is-better)",
                           "critic_means": "fit on split == 'train' rows only, applied to all committed rows",
                           "u_perp": "sd over critics (ddof=1, as uncertainty_std) of r_ji - alpha_hat_j",
                           "estimators": "readout_amendment7.endpoint (logistic LORO, logistic 5-fold seed 0, fixed unit weights; "
                                         "repo-clustered bootstrap n=4000 seeds 21/22) + verify_and_robustify.fold_avg_with_ci (n=2000 seed 13); "
                                         "AUROC CIs via readout_amendment7.boot (n=4000 seed 9)"},
           "benchmarks": {}}
    for tag in ("verified", "multilingual"):
        rows, pop = load_rows(tag); rep["benchmarks"][tag] = {}
        for mname, d in BENCH[tag]["matrices"].items():
            crit, R = (matrix_from_audit(rows, pop, tag) if d is None else matrix_from_dir(rows, pop, d))
            rep["benchmarks"][tag][mname] = analyse(tag, mname, crit, R, rows, pop)
    txt = json.dumps(rep, indent=1, sort_keys=True) + "\n"; OUT.write_text(txt)
    print("wrote", OUT, "sha256", hashlib.sha256(txt.encode()).hexdigest()[:16])
    for tag, ms in rep["benchmarks"].items():
        for mname, o in ms.items():
            print(f"\n== {tag} / {mname}: n={o['n']} failures={o['failures']} repos={o['repositories']} J={o['J']} | AUROC(-c)={o['AUROC_c_detection']} AUROC(u_pub)={o['AUROC_u_published']}")
            for pname, p in o["panels"].items():
                ss = p["ss"]; e = p["endpoint_add_u_perp_to_c"]
                print(f"  {pname}: SS shares critic {ss['share_critic']} / trajectory {ss['share_trajectory']} / residual {ss['share_residual']} | ICC(C,1) {ss['ICC_C1_consistency']}")
                print(f"         AUROC u_rec {p['AUROC_u_recomputed_rank_norm_sd']} {p['AUROC_u_recomputed_CI']} | u_perp {p['AUROC_u_perp']} {p['AUROC_u_perp_CI']} | corr(u_perp,u_rec) {p['corr_u_perp_vs_u_recomputed']}")
                print(f"         +u_perp: LORO {e['logistic_LORO']['delta']} {e['logistic_LORO']['CI95']} | 5fold {e['logistic_5fold']['delta']} {e['logistic_5fold']['CI95']} | within-fold {e['within_fold_size_weighted']['delta']} {e['within_fold_size_weighted']['CI95']} | fixed {e['fixed_unit_weights_PRIMARY']['delta']} {e['fixed_unit_weights_PRIMARY']['CI95']} refit {p['refit_bootstrap_fixed_unit_weights']['CI95_refit']}")
                print(f"         identity: |sd(dc_train) - u_perp| = {p['identity']['max_abs_diff_sd(dc_train_fit) - u_perp']:.2e}")
                if "E2_error_corr" in p: ec = p["E2_error_corr"]; print(f"         E2 error corr before {ec['before_demeaning']} after {ec['after_demeaning']} (max change {ec['max_abs_change']:.1e})")


if __name__ == "__main__":
    main()
