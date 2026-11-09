#!/usr/bin/env python3
"""Amendment 3 readout: u_max / u_mean / u_last from windowed panel scores, vs the pre-registered bars."""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT / "src"))
from silentfail.scoring import fit_rank_norms  # noqa: E402
from silentfail.stats import auroc_fast, cluster_bootstrap_delta  # noqa: E402
PANEL = ["ray2333_grm_gemma2_2b_rewardmodel_ft", "ray2333_grm_llama3_2_3b_rewardmodel_ft", "skywork_skywork_reward_v2_llama_3_2_1b", "skywork_skywork_reward_v2_qwen3_1_7b"]
NAMED = "skywork_skywork_reward_v2_llama_3_1_8b"

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--windows", type=Path, required=True)
    ap.add_argument("--primary-rows", type=Path, default=ROOT / "data/swebench_pilot50/pilot_free_scored.jsonl")
    ap.add_argument("--out", type=Path, default=None); a = ap.parse_args()
    prim = {r["instance_id"]: r for r in map(json.loads, a.primary_rows.read_text().splitlines()) if r}
    W = {f.stem.replace("windows_gpu_", ""): json.loads(f.read_text())["rows"] for f in a.windows.glob("windows_gpu_*.json")}
    pop = [i for i, r in prim.items() if not r["excluded"] and r.get("visible_pass") and all(i in W.get(m, {}) for m in PANEL)]
    y = np.array([1 if prim[i]["g"] == 0 else 0 for i in pop]); grp = [prim[i]["repo"] for i in pop]
    # rank norms per critic on TRAIN rows' window scores
    fits = fit_rank_norms({m: [w["score"] for i in pop if prim[i]["split"] == "train" for w in W[m][i]["windows"] if w["score"] == w["score"]] for m in PANEL})
    def u_agg(i, how):
        n = W[PANEL[0]][i]["n_windows"]; us = []
        for k in range(n):
            rn = [fits[m].transform(W[m][i]["windows"][k]["score"]) for m in PANEL if k < W[m][i]["n_windows"] and W[m][i]["windows"][k]["score"] == W[m][i]["windows"][k]["score"]]
            if len(rn) >= 2: us.append(float(np.std(rn, ddof=1)))
        if not us: return float("nan")
        return {"max": max(us), "mean": float(np.mean(us)), "last": us[-1]}[how]
    rep = {"population_n": len(pop), "silent_failures": int(y.sum()), "critics_loaded": sorted(W), "n_windows_median": float(np.median([W[PANEL[0]][i]["n_windows"] for i in pop]))}
    c_prim = np.array([prim[i]["c"] for i in pop], float)
    for how in ("max", "mean", "last"):
        u = np.array([u_agg(i, how) for i in pop]); ok = ~np.isnan(u)
        t = cluster_bootstrap_delta(y[ok], -c_prim[ok], u[ok], [g for g, o in zip(grp, ok) if o], endpoint="auroc", n_boot=4000, seed=1)
        rep[f"u_{how}"] = {"AUROC": round(auroc_fast(y[ok], u[ok]), 3), "vs_primary_c": round(t.delta, 3), "CI95": [round(t.ci_lo, 3), round(t.ci_hi, 3)]}
    # --- added after the first readout (review item 2): permutation null for u_max, jackknife, per-critic aggregates ---
    rng = np.random.default_rng(3); grp_a = np.array(grp)
    def perm_p(score, within_repo):
        obs = auroc_fast(y, score); n_ge = 0; B = 4000
        for _ in range(B):
            yp = y.copy()
            if within_repo:
                for g in np.unique(grp_a):
                    idx = np.flatnonzero(grp_a == g); yp[idx] = y[idx][rng.permutation(len(idx))]
            else:
                yp = y[rng.permutation(len(y))]
            if auroc_fast(yp, score) >= obs - 1e-12: n_ge += 1
        return round((n_ge + 1) / (B + 1), 4)
    for how in ("max", "mean", "last"):
        u = np.array([u_agg(i, how) for i in pop])
        rep[f"u_{how}"]["p_perm"] = perm_p(u, False); rep[f"u_{how}"]["p_perm_within_repo"] = perm_p(u, True)
    u = np.array([u_agg(i, "max") for i in pop])
    jk = [auroc_fast(np.delete(y, k), np.delete(u, k)) for k in range(len(pop))]
    loro = [auroc_fast(y[grp_a != g], u[grp_a != g]) for g in np.unique(grp_a) if 0 < y[grp_a != g].sum() < (grp_a != g).sum()]
    rep["u_max_robustness"] = {"jackknife_min_max": [round(min(jk), 3), round(max(jk), 3)], "leave_one_repo_out_min_max": [round(min(loro), 3), round(max(loro), 3)],
                               "n_single_window_rows": int(sum(W[PANEL[0]][i]["n_windows"] == 1 for i in pop))}
    # where does the max sit? (fraction of multi-window rows whose u_max window is not the last window)
    def argmax_not_last(i):
        n = W[PANEL[0]][i]["n_windows"]
        if n < 2: return None
        us = []
        for k in range(n):
            rn = [fits[m].transform(W[m][i]["windows"][k]["score"]) for m in PANEL if k < W[m][i]["n_windows"]]
            us.append(float(np.std(rn, ddof=1)))
        return int(np.argmax(us)) != n - 1
    mw = [argmax_not_last(i) for i in pop]; mw = [v for v in mw if v is not None]
    rep["u_max_robustness"]["frac_multiwindow_rows_max_not_in_last_window"] = round(float(np.mean(mw)), 3)
    rep["per_critic_window_aggregates_EXPLORATORY"] = {}
    for m in sorted(W):
        agg = {}
        for how, fn in (("mean", np.nanmean), ("min", np.nanmin), ("last", lambda a: a[-1])):
            s_ = np.array([fn([w["score"] for w in W[m][i]["windows"]]) for i in pop]); agg[f"AUROC_neg{how}"] = round(auroc_fast(y, -s_), 3)
        rep["per_critic_window_aggregates_EXPLORATORY"][m] = agg
    um = rep["u_max"]["AUROC"]
    rep["A3_P1_beats_ruler_0p665"] = bool(um > 0.665); rep["A3_P2_gain_over_prefix_gt_0p10"] = bool(um - 0.508 > 0.10)
    if NAMED in W:
        cm = np.array([np.nanmean([w["score"] for w in W[NAMED][i]["windows"]]) for i in pop])
        rep["named_critic_mean_window"] = {"AUROC_negc": round(auroc_fast(y, -cm), 3), "A3_P3_gain_over_0p534_gt_0p10": bool(auroc_fast(y, -cm) - 0.534 > 0.10)}
    rep["caveat"] = "u_max is a max of noisy statistics: biased upward under the null; 12 positives; point estimates."
    print(json.dumps(rep, indent=2))
    if a.out: a.out.write_text(json.dumps(rep, indent=2))

if __name__ == "__main__": main()
