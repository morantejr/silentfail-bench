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
    ap.add_argument("--primary-rows", type=Path, default=ROOT / "data/swebench_pilot50/pilot_free_scored.jsonl"); a = ap.parse_args()
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
    um = rep["u_max"]["AUROC"]
    rep["A3_P1_beats_ruler_0p665"] = bool(um > 0.665); rep["A3_P2_gain_over_prefix_gt_0p10"] = bool(um - 0.508 > 0.10)
    if NAMED in W:
        cm = np.array([np.nanmean([w["score"] for w in W[NAMED][i]["windows"]]) for i in pop])
        rep["named_critic_mean_window"] = {"AUROC_negc": round(auroc_fast(y, -cm), 3), "A3_P3_gain_over_0p534_gt_0p10": bool(auroc_fast(y, -cm) - 0.534 > 0.10)}
    rep["caveat"] = "u_max is a max of noisy statistics: biased upward under the null; 12 positives; point estimates."
    print(json.dumps(rep, indent=2))

if __name__ == "__main__": main()
