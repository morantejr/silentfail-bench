#!/usr/bin/env python3
"""Amendment 5's committed-set analysis, with the population defect fixed, and
the endpoint computed by the same four estimators the paper's main table uses.

TWO DEFECTS THIS FIXES.

1. Population. prereg_readout_amendment5.py selects a NEW committed candidate with
   gpt-5, then filters the population with `rows[i]["visible_pass"]` -- the
   agent-declared-complete flag of the candidate the 4B critic committed, which is
   a DIFFERENT trajectory. The registered population is "committed,
   agent-declared-complete trajectories", so for this arm the flag must come from
   the gpt-5-committed candidate. The bundle carries it per candidate. On Verified
   the two readings coincide exactly; on Multilingual the published reading dropped
   16 instances whose gpt-5-committed trajectory was complete, non-empty and graded,
   purely because the 4B's pick had been empty.

2. Provenance. The `endpoint` block in prereg_readout_amendment5.json, including
   the "+0.029" the paper quotes and the "BASELINE DEGENERATE" annotations, is not
   produced by any committed script. This script regenerates it.

Reports both populations side by side so the correction is auditable, and both
sign conventions for the baseline, because `c` here is already a DETECTION score
(c = 100 - p_resolved, higher = worse) and negating it again inverts the baseline.
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT / "src"))
from silentfail.combine import _fit_standardizer, fit_logistic_irls  # noqa: E402
from silentfail.scoring import fit_rank_norms, apply_panel_u  # noqa: E402
from silentfail.stats import auroc_fast, cluster_bootstrap_delta  # noqa: E402

PANEL = ["ray2333_grm_gemma2_2b_rewardmodel_ft", "ray2333_grm_llama3_2_3b_rewardmodel_ft",
         "skywork_skywork_reward_v2_llama_3_2_1b", "skywork_skywork_reward_v2_qwen3_1_7b"]
S_P2_BAR, S_P3_U_BAR, S_P3_DELTA_BAR = 0.10, 0.60, 0.05


def boot(y, s, grp, n=4000, seed=9):
    g = np.asarray(grp); u = np.unique(g); idx = {k: np.flatnonzero(g == k) for k in u}
    rng = np.random.default_rng(seed); v = []
    for _ in range(n):
        ii = np.concatenate([idx[k] for k in rng.choice(u, size=len(u), replace=True)])
        if 0 < y[ii].sum() < len(ii): v.append(auroc_fast(y[ii], s[ii]))
    return [round(float(np.percentile(v, 2.5)), 3), round(float(np.percentile(v, 97.5)), 3)] if v else None


def folds_loro(grp): return [(grp == g) for g in np.unique(grp)]
def folds_k(grp, k=5, seed=0):
    u = np.unique(grp); rng = np.random.default_rng(seed); u = u[rng.permutation(len(u))]
    return [np.isin(grp, u[i::k]) for i in range(k)]

def oof(X, y, folds):
    p = np.full(len(y), np.nan)
    for te in folds:
        tr = ~te
        if y[tr].sum() < 2 or (1 - y[tr]).sum() < 2: continue
        mu, sg = _fit_standardizer(X[tr]); coef, b = fit_logistic_irls((X[tr] - mu) / sg, y[tr].astype(float))
        p[te] = 1 / (1 + np.exp(-np.clip(((X[te] - mu) / sg) @ coef + b, -30, 30)))
    return p

def fixed_combo(X, y, folds):
    p = np.full(len(y), np.nan)
    for te in folds:
        mu, sg = _fit_standardizer(X[~te]); p[te] = ((X[te] - mu) / sg).sum(axis=1)
    return p


def analyse(bench, rows_f, bundle_f, judge_cache, panel_dir, extras, picks_f, per_candidate):
    rows = {r["instance_id"]: r for r in map(json.loads, Path(rows_f).read_text().splitlines()) if r}
    bl = [json.loads(l) for l in Path(bundle_f).read_text().splitlines() if l.strip()]
    cands = {(c["instance_id"], c["candidate"]): c for c in bl}
    J = {}
    for d in map(json.loads, Path(judge_cache).read_text().splitlines()):
        if d.get("p_resolved") is not None: J[(d["instance_id"], d["candidate"])] = d["p_resolved"]
    gold = {}
    for i, r in rows.items():
        if not r["excluded"]: gold[(i, r["selected_candidate_index"])] = r["g"]
    for f in extras:
        if Path(f).exists():
            for d in map(json.loads, Path(f).read_text().splitlines()):
                if d.get("g") in (0, 1): gold[(d["instance_id"], d["candidate"])] = d["g"]
    picks = {d["instance_id"]: d["candidate"] for d in map(json.loads, Path(picks_f).read_text().splitlines())}
    rr = json.loads((Path(picks_f).parent.parent / "gold_modal" / f"random_arm_{bench}" / "report.json").read_text())
    rand_gold = {}
    for i in rr["resolved_ids"]:
        if i in picks: rand_gold[i] = 1; gold[(i, picks[i])] = 1
    for i in rr["unresolved_ids"]:
        if i in picks: rand_gold[i] = 0; gold[(i, picks[i])] = 0

    newc = {}
    for i in rows:
        av = [(c, J[(i, c)]) for c in range(4) if (i, c) in J and (cands.get((i, c), {}).get("final_solution") or "").strip()]
        if av: newc[i] = max(av, key=lambda x: x[1])[0]

    eligible = (lambda i: cands[(i, newc[i])].get("visible_pass")) if per_candidate else (lambda i: rows[i].get("visible_pass"))
    pop = [i for i in newc if (i, newc[i]) in gold and eligible(i)]

    # pool AUROC: every graded candidate with a judge score; no eligibility filter (unchanged by this fix)
    keys = [k for k in gold if k in J]
    yp = np.array([1 - gold[k] for k in keys]); sp = np.array([100 - J[k] for k in keys], float)
    gp = [rows[k[0]]["repo"] for k in keys]
    pool_auroc = round(auroc_fast(yp, sp), 3)

    S = {m: json.load(open(Path(panel_dir) / f"scores_gpu_{m}.json"))["scores"] for m in PANEL}
    y = np.array([1 - gold[(i, newc[i])] for i in pop]); grp = np.array([rows[i]["repo"] for i in pop])
    c = np.array([100 - J[(i, newc[i])] for i in pop], float)      # already a DETECTION score
    tr = [i for i in pop if rows[i]["split"] == "train"]
    fits = fit_rank_norms({m: [S[m][f"{i}#{newc[i]}"]["score"] for i in tr if f"{i}#{newc[i]}" in S[m]] for m in PANEL})
    u = np.array([apply_panel_u({m: S[m].get(f"{i}#{newc[i]}", {}).get("score", np.nan) for m in PANEL}, fits)[1] for i in pop])

    o = {"benchmark": bench, "population_rule": "per-candidate visible_pass (corrected)" if per_candidate else "row-level visible_pass (as published)",
         "n": len(pop), "failures": int(y.sum()), "repositories": int(len(set(grp))),
         "pool_AUROC_detect": pool_auroc,
         "committed_AUROC_detect": round(auroc_fast(y, c), 3),
         "AUROC_u": round(auroc_fast(y, u), 3), "AUROC_u_CI": boot(y, u, grp)}
    o["S_P2_gap"] = round(pool_auroc - o["committed_AUROC_detect"], 3)
    o["S_P2_passes"] = bool(o["S_P2_gap"] > S_P2_BAR)
    o["S_P3_cond1_u_above_0.60"] = bool(o["AUROC_u"] > S_P3_U_BAR)

    Xc, Xcu = c[:, None], np.column_stack([c, u])
    est = {}
    for name, folds in (("logistic_LORO", folds_loro(grp)), ("logistic_5fold", folds_k(grp))):
        pc, pcu = oof(Xc, y, folds), oof(Xcu, y, folds); ok = ~np.isnan(pc) & ~np.isnan(pcu)
        t = cluster_bootstrap_delta(y[ok], pc[ok], pcu[ok], grp[ok], endpoint="auroc", n_boot=4000, seed=5)
        est[name] = {"baseline_AUROC": round(auroc_fast(y[ok], pc[ok]), 3), "delta": round(t.delta, 3),
                     "CI95": [round(t.ci_lo, 3), round(t.ci_hi, 3)]}
    pc, pcu = fixed_combo(Xc, y, folds_loro(grp)), fixed_combo(Xcu, y, folds_loro(grp))
    t = cluster_bootstrap_delta(y, pc, pcu, grp, endpoint="auroc", n_boot=4000, seed=12)
    est["fixed_unit_weights"] = {"baseline_AUROC": round(auroc_fast(y, pc), 3), "delta": round(t.delta, 3),
                                "CI95": [round(t.ci_lo, 3), round(t.ci_hi, 3)]}
    for k, v in est.items():
        v["baseline_degenerate"] = bool(v["baseline_AUROC"] < o["committed_AUROC_detect"] - 0.10)
    o["estimators"] = est
    prim = est["fixed_unit_weights"]
    o["S_P3_cond2_delta_above_0.05_CI_excludes_0"] = bool(prim["delta"] > S_P3_DELTA_BAR and prim["CI95"][0] > 0)
    o["S_P3_passes"] = bool(o["S_P3_cond1_u_above_0.60"] and o["S_P3_cond2_delta_above_0.05_CI_excludes_0"])
    o["upper_bound_vs_margin"] = {"upper": prim["CI95"][1], "margin": S_P3_DELTA_BAR,
                                  "null_established": bool(prim["CI95"][1] < S_P3_DELTA_BAR)}
    # S-P1's second condition, paired against the random arm on this population
    pairs = [i for i in pop if i in rand_gold]
    if len(pairs) >= 10:
        fc = np.array([1 - gold[(i, newc[i])] for i in pairs], float); fr = np.array([1 - rand_gold[i] for i in pairs], float)
        gg = np.array([rows[i]["repo"] for i in pairs]); d = fc - fr
        uq = np.unique(gg); idx = {k: np.flatnonzero(gg == k) for k in uq}; rng = np.random.default_rng(17)
        vs = [float(np.mean(d[np.concatenate([idx[k] for k in rng.choice(uq, size=len(uq), replace=True)])])) for _ in range(4000)]
        o["S_P1b"] = {"pairs": len(pairs), "committed_rate": round(float(fc.mean()), 3), "random_rate": round(float(fr.mean()), 3),
                      "difference": round(float(d.mean()), 3),
                      "CI95": [round(float(np.percentile(vs, 2.5)), 3), round(float(np.percentile(vs, 97.5)), 3)],
                      "reverse_discordant": int(((fc == 1) & (fr == 0)).sum()), "passes": bool(d.mean() < 0)}
    return o


ARGS = {"verified": (ROOT/"data/swebench_pilot50/pilot_free_scored.jsonl", ROOT/"data/swebench_pilot50/rescoring_bundle.jsonl",
                     ROOT/"data/selector_axis/judge_verified.jsonl", ROOT/"data/swebench_pilot50/l4/out/gpu_scores_16k",
                     [ROOT/"data/swebench_pilot50/l4/out/gpu_scores_16k/gold_extra_merged.jsonl", ROOT/"data/selector_axis/gold_extra_verified.jsonl"],
                     ROOT/"data/random_arm/verified_picks.jsonl"),
        "multilingual": (ROOT/"data/replication_multilingual/replication_scored.jsonl", ROOT/"data/replication_multilingual/rescoring_bundle.jsonl",
                     ROOT/"data/selector_axis/judge_multilingual.jsonl", ROOT/"data/replication_multilingual/modal/prefix4k",
                     [ROOT/"data/selector_axis/gold_extra_multilingual.jsonl"], ROOT/"data/random_arm/multilingual_picks.jsonl")}
rep = {"analysis": "Amendment 5 committed-set readout with the population defect fixed; supersedes the endpoint block in prereg_readout_amendment5.json.",
       "defect": "the published readout filtered the population by the 4B-committed candidate's visible_pass, not the gpt-5-committed candidate's.",
       "note": "c is a detection score (100 - p_resolved); the baseline is c itself, not -c.",
       "as_published": {}, "corrected": {}}
for bench, a in ARGS.items():
    rep["as_published"][bench] = analyse(bench, *a, per_candidate=False)
    rep["corrected"][bench] = analyse(bench, *a, per_candidate=True)
(ROOT / "data/selector_axis/readout_amendment5_corrected.json").write_text(json.dumps(rep, indent=2) + "\n")

print("=== Amendment 5, published population vs corrected population ===")
for bench in ARGS:
    print(f"\n--- {bench} ---")
    for k in ("as_published", "corrected"):
        o = rep[k][bench]
        print(f"  {k:13s} n={o['n']:3d} fail={o['failures']:2d} repos={o['repositories']:2d} "
              f"pool={o['pool_AUROC_detect']:.3f} committed={o['committed_AUROC_detect']:.3f} u={o['AUROC_u']:.3f} {o['AUROC_u_CI']}")
        e = o["estimators"]
        print(f"                LORO {e['logistic_LORO']['delta']:+.3f} (base {e['logistic_LORO']['baseline_AUROC']:.3f}"
              f"{' DEGENERATE' if e['logistic_LORO']['baseline_degenerate'] else ''})  "
              f"5fold {e['logistic_5fold']['delta']:+.3f}  unfitted {e['fixed_unit_weights']['delta']:+.3f} {e['fixed_unit_weights']['CI95']}")
        print(f"                S-P2 gap {o['S_P2_gap']:+.3f} -> {'PASS' if o['S_P2_passes'] else 'FAIL'} (bar 0.10) | "
              f"S-P3 cond1 u>0.60 {'MET' if o['S_P3_cond1_u_above_0.60'] else 'NOT MET'} | S-P3 {'PASS' if o['S_P3_passes'] else 'FAIL'}")
        if "S_P1b" in o:
            s = o["S_P1b"]
            print(f"                S-P1b {s['committed_rate']:.3f} vs {s['random_rate']:.3f} = {s['difference']:+.3f} {s['CI95']} "
                  f"({s['pairs']} pairs, {s['reverse_discordant']} reverse-discordant) -> {'PASS' if s['passes'] else 'FAIL'}")
print("\nwrote data/selector_axis/readout_amendment5_corrected.json")
