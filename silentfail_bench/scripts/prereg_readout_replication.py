#!/usr/bin/env python3
"""Amendment 4e readout: R-P1 (prefix-u null replicates), R-P2 (u_max confirmatory), R-P3 (u_max vs lines added).

    python scripts/prereg_readout_replication.py --rows data/replication_multilingual/replication_scored.jsonl \
        --windows data/replication_multilingual/modal/windows [--full16k data/replication_multilingual/modal/full16k]
"""
from __future__ import annotations

import argparse, json, sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT / "src"))
from silentfail.scoring import fit_rank_norms  # noqa: E402
from silentfail.stats import auroc_fast, cluster_bootstrap_delta  # noqa: E402

PANEL = ["ray2333_grm_gemma2_2b_rewardmodel_ft", "ray2333_grm_llama3_2_3b_rewardmodel_ft", "skywork_skywork_reward_v2_llama_3_2_1b", "skywork_skywork_reward_v2_qwen3_1_7b"]
FLOOR = 20


def added_lines(diff: str) -> int:
    return sum(1 for l in diff.splitlines() if l.startswith("+") and not l.startswith("+++"))


def boot_auroc(y, s, grp, n=4000, seed=0):
    """Repo-clustered bootstrap CI on a single AUROC."""
    g = np.asarray(grp); uniq = np.unique(g); idx = {k: np.flatnonzero(g == k) for k in uniq}; rng = np.random.default_rng(seed); v = []
    for _ in range(n):
        ii = np.concatenate([idx[k] for k in rng.choice(uniq, size=len(uniq), replace=True)])
        if 0 < y[ii].sum() < len(ii): v.append(auroc_fast(y[ii], s[ii]))
    return [round(float(np.percentile(v, 2.5)), 3), round(float(np.percentile(v, 97.5)), 3)]


def perm_p(y, s, grp, within, n=4000, seed=3):
    g = np.asarray(grp); rng = np.random.default_rng(seed); obs = auroc_fast(y, s); c = 0
    for _ in range(n):
        yp = y.copy()
        if within:
            for k in np.unique(g):
                ii = np.flatnonzero(g == k); yp[ii] = y[ii][rng.permutation(len(ii))]
        else: yp = y[rng.permutation(len(y))]
        c += auroc_fast(yp, s) >= obs - 1e-12
    return round((c + 1) / (n + 1), 4)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", type=Path, required=True); ap.add_argument("--windows", type=Path, default=None); ap.add_argument("--full16k", type=Path, default=None)
    ap.add_argument("--out", type=Path, default=None); a = ap.parse_args()
    rows = [json.loads(l) for l in a.rows.read_text().splitlines() if l.strip()]
    pop = [r for r in rows if not r["excluded"] and r["visible_pass"]]
    y = np.array([1 if r["g"] == 0 else 0 for r in pop]); grp = [r["repo"] for r in pop]; ids = [r["instance_id"] for r in pop]
    rep = {"analysis": "Amendment 4e replication readout (pre-registered)", "n_rows": len(rows), "population_n": len(pop), "silent_failures": int(y.sum()),
           "silent_failure_rate": round(float(y.mean()), 3) if len(pop) else None, "n_repos": len(set(grp)), "inference_floor_met": bool(y.sum() >= FLOOR)}
    if y.sum() < FLOOR or y.sum() == len(pop):
        rep["conclusion"] = "below the pre-registered floor of 20 positives: descriptive only"
    u = np.array([r["u"] for r in pop]); c = np.array([r["c"] for r in pop]); ruler = np.array([added_lines(r["final_solution"]) for r in pop], float)
    t = cluster_bootstrap_delta(y, -c, u, grp, endpoint="auroc", n_boot=4000, seed=1)
    au = auroc_fast(y, u); ci = boot_auroc(y, u, grp)
    rep["primary_prefix_u"] = {"M1_AUROC_u": round(au, 3), "CI95_repo_clustered": ci, "M2_AUROC_negc": round(auroc_fast(y, -c), 3),
                               "M3_delta": round(t.delta, 3), "M3_CI95": [round(t.ci_lo, 3), round(t.ci_hi, 3)], "p_perm_within_repo": perm_p(y, u, grp, True)}
    rep["R_P1_null_replicates"] = bool(au < 0.60 and ci[0] <= 0.5 <= ci[1])
    rep["ruler_lines_added"] = {"AUROC": round(auroc_fast(y, ruler), 3), "CI95_repo_clustered": boot_auroc(y, ruler, grp)}
    if a.windows and a.windows.exists():
        W = {f.stem.replace("windows_gpu_", ""): json.loads(f.read_text())["rows"] for f in a.windows.glob("windows_gpu_*.json")}
        if all(m in W for m in PANEL):
            key = {r["instance_id"]: r for r in rows}
            fits = fit_rank_norms({m: [w["score"] for i in W[m] if key.get(i, {}).get("split") == "train" for w in W[m][i]["windows"] if w["score"] == w["score"]] for m in PANEL})
            def u_max(i):
                if any(i not in W[m] for m in PANEL): return float("nan")
                n = min(W[m][i]["n_windows"] for m in PANEL); us = []
                for k in range(n):
                    rn = [fits[m].transform(W[m][i]["windows"][k]["score"]) for m in PANEL if W[m][i]["windows"][k]["score"] == W[m][i]["windows"][k]["score"]]
                    if len(rn) >= 2: us.append(float(np.std(rn, ddof=1)))
                return max(us) if us else float("nan")
            um = np.array([u_max(i) for i in ids]); ok = ~np.isnan(um)
            if ok.sum() >= 10 and 0 < y[ok].sum() < ok.sum():
                yy, gg = y[ok], [g for g, o in zip(grp, ok) if o]; a_um = auroc_fast(yy, um[ok]); ci_um = boot_auroc(yy, um[ok], gg)
                pw = perm_p(yy, um[ok], gg, True); pf = perm_p(yy, um[ok], gg, False)
                tr = cluster_bootstrap_delta(yy, ruler[ok], um[ok], gg, endpoint="auroc", n_boot=4000, seed=2)
                rep["windowed_u_max"] = {"AUROC": round(a_um, 3), "CI95_repo_clustered": ci_um, "p_perm_within_repo": pw, "p_perm_free": pf, "rows_scored": int(ok.sum()),
                                        "vs_lines_added_delta": round(tr.delta, 3), "vs_lines_added_CI95": [round(tr.ci_lo, 3), round(tr.ci_hi, 3)]}
                rep["R_P2_u_max_confirmed"] = bool(a_um > 0.60 and ci_um[0] > 0.50 and pw < 0.05)
                rep["R_P3_u_max_beats_ruler"] = bool(a_um > rep["ruler_lines_added"]["AUROC"])
    for label, d in (("single_critics_prefix4k", None), ("single_critics_full16k", a.full16k)):
        if d is None or not d.exists(): continue
        out = {}
        for f in sorted(d.glob("scores_gpu_*.json")):
            sc = json.loads(f.read_text())["scores"]; s = np.array([sc.get(f"{r['instance_id']}#{r['selected_candidate_index']}", {}).get("score", np.nan) for r in pop], float)
            if np.isnan(s).all(): continue
            s = np.where(np.isnan(s), np.nanmedian(s), s); out[f.stem.replace("scores_gpu_", "")] = round(auroc_fast(y, -s), 3)
        rep[label] = out
    txt = json.dumps(rep, indent=1); print(txt)
    if a.out: a.out.write_text(txt)


if __name__ == "__main__":
    main()
