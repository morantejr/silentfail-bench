#!/usr/bin/env python3
"""Amendment 4f readout: random (unselected) arm vs committed arm, both benchmarks, paired by instance."""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT / "src"))
from silentfail.scoring import apply_panel_u, fit_rank_norms  # noqa: E402
from silentfail.stats import auroc_fast  # noqa: E402
PANEL = ["ray2333_grm_gemma2_2b_rewardmodel_ft", "ray2333_grm_llama3_2_3b_rewardmodel_ft", "skywork_skywork_reward_v2_llama_3_2_1b", "skywork_skywork_reward_v2_qwen3_1_7b"]
CSEL = "skywork_skywork_reward_v2_qwen3_4b"

def boot_diff(a, b, grp, n=4000, seed=6):
    g = np.asarray(grp); u = np.unique(g); idx = {k: np.flatnonzero(g == k) for k in u}; rng = np.random.default_rng(seed); d = []
    for _ in range(n):
        ii = np.concatenate([idx[k] for k in rng.choice(u, size=len(u), replace=True)]); d.append(a[ii].mean() - b[ii].mean())
    return [round(float(np.percentile(d, 2.5)), 3), round(float(np.percentile(d, 97.5)), 3)]

def boot_auroc(y, s, grp, n=4000, seed=7):
    g = np.asarray(grp); u = np.unique(g); idx = {k: np.flatnonzero(g == k) for k in u}; rng = np.random.default_rng(seed); v = []
    for _ in range(n):
        ii = np.concatenate([idx[k] for k in rng.choice(u, size=len(u), replace=True)])
        if 0 < y[ii].sum() < len(ii): v.append(auroc_fast(y[ii], s[ii]))
    return [round(float(np.percentile(v, 2.5)), 3), round(float(np.percentile(v, 97.5)), 3)]

def arm(rows_path, picks_path, report_path, scores_dir, csel_from_rows):
    rows = {r["instance_id"]: r for r in map(json.loads, Path(rows_path).read_text().splitlines()) if r}
    picks = {p["instance_id"]: p for p in map(json.loads, Path(picks_path).read_text().splitlines()) if p}
    rep = json.loads(Path(report_path).read_text()); g = {i: 1 for i in rep["resolved_ids"]}; g.update({i: 0 for i in rep["unresolved_ids"]})
    S = {m: json.loads((Path(scores_dir) / f"scores_gpu_{m}.json").read_text())["scores"] for m in PANEL + [CSEL]}
    ids = [i for i in picks if i in g and i in rows and not rows[i]["excluded"] and rows[i]["visible_pass"]]
    y_r = np.array([1 if g[i] == 0 else 0 for i in ids]); y_c = np.array([1 if rows[i]["g"] == 0 else 0 for i in ids]); grp = [rows[i]["repo"] for i in ids]
    def sc(m, i, k): return S[m].get(f"{i}#{k}", {}).get("score", np.nan)
    c_r = np.array([rows[i]["candidate_scores"][picks[i]["candidate"]] if csel_from_rows else sc(CSEL, i, picks[i]["candidate"]) for i in ids], float)
    c_c = np.array([rows[i]["c"] if csel_from_rows else sc(CSEL, i, picks[i]["committed"]) for i in ids], float)
    train_ids = [i for i in rows if rows[i]["split"] == "train" and not rows[i]["excluded"]]
    fits = fit_rank_norms({m: [sc(m, i, rows[i]["selected_candidate_index"]) for i in train_ids] for m in PANEL})
    u_r = np.array([apply_panel_u({m: sc(m, i, picks[i]["candidate"]) for m in PANEL}, fits)[1] for i in ids])
    u_c = np.array([apply_panel_u({m: sc(m, i, picks[i]["committed"]) for m in PANEL}, fits)[1] for i in ids])
    out = {"n_paired": len(ids), "failure_rate_random": round(float(y_r.mean()), 3), "failure_rate_committed": round(float(y_c.mean()), 3),
           "paired_diff_committed_minus_random": round(float(y_c.mean() - y_r.mean()), 3), "CI95": boot_diff(y_c, y_r, grp),
           "AUROC_negc_random": round(auroc_fast(y_r, -c_r), 3), "AUROC_negc_random_CI": boot_auroc(y_r, -c_r, grp), "AUROC_negc_committed_same_rows": round(auroc_fast(y_c, -c_c), 3),
           "AUROC_u_random": round(auroc_fast(y_r, u_r), 3), "AUROC_u_random_CI": boot_auroc(y_r, u_r, grp), "AUROC_u_committed_same_rows_same_panel": round(auroc_fast(y_c, u_c), 3),
           "random_arm_gold_errors": len(rep["error_ids"])}
    out["F_P1_selection_concentrates_failure"] = bool(out["paired_diff_committed_minus_random"] > 0 and out["CI95"][0] > 0)
    out["F_P2_selector_sees_more_on_random"] = bool(out["AUROC_negc_random"] - out["AUROC_negc_committed_same_rows"] > 0.10)
    return out

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out", type=Path, default=ROOT / "data/random_arm/prereg_readout_amendment4f.json"); a = ap.parse_args()
    rep = {"analysis": "Amendment 4f (pre-registered): unselected-candidate control, paired by instance",
           "verified": arm(ROOT / "data/swebench_pilot50/pilot_free_scored.jsonl", ROOT / "data/random_arm/verified_picks.jsonl", ROOT / "data/gold_modal/random_arm_verified/report.json", ROOT / "data/swebench_pilot50/l4/out/gpu_scores_16k", True),
           "multilingual": arm(ROOT / "data/replication_multilingual/replication_scored.jsonl", ROOT / "data/random_arm/multilingual_picks.jsonl", ROOT / "data/gold_modal/random_arm_multilingual/report.json", ROOT / "data/replication_multilingual/modal/prefix4k", False)}
    rep["verified"]["note"] = "c from the primary run's candidate scores (prefix 4k); u from the 16k L4 panel on both arms, as declared"
    a.out.write_text(json.dumps(rep, indent=1)); print(json.dumps(rep, indent=1))

if __name__ == "__main__": main()
