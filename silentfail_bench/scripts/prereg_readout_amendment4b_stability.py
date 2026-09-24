#!/usr/bin/env python3
"""Amendment 4b readout: r_fmt, r_shift, r_adj, r_far per critic and for per-window panel u."""
from __future__ import annotations

import argparse, json, sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from silentfail.scoring import apply_panel_u, fit_rank_norms  # noqa: E402

PANEL = ["ray2333_grm_gemma2_2b_rewardmodel_ft", "ray2333_grm_llama3_2_3b_rewardmodel_ft",
         "skywork_skywork_reward_v2_llama_3_2_1b", "skywork_skywork_reward_v2_qwen3_1_7b"]


def spearman(a, b) -> float:
    a, b = np.asarray(a, float), np.asarray(b, float); ok = ~(np.isnan(a) | np.isnan(b))
    if ok.sum() < 4: return float("nan")
    ra, rb = np.argsort(np.argsort(a[ok])), np.argsort(np.argsort(b[ok]))
    return float(np.corrcoef(ra, rb)[0, 1])


def stats(rows: dict, get) -> dict:
    """rows: instance -> list of window dicts; get(w) -> (orig, fmt, shift)."""
    o, f, s, adj_a, adj_b, far_a, far_b = [], [], [], [], [], [], []
    for iid, ws in rows.items():
        vals = [get(w) for w in ws]
        for v in vals: o.append(v[0]); f.append(v[1]); s.append(v[2])
        for v1, v2 in zip(vals, vals[1:]): adj_a.append(v1[0]); adj_b.append(v2[0])
        if len(vals) >= 2: far_a.append(vals[0][0]); far_b.append(vals[-1][0])
    return {"n_windows": len(o), "r_fmt": round(spearman(o, f), 3), "r_shift": round(spearman(o, s), 3),
            "r_adj": round(spearman(adj_a, adj_b), 3), "n_adj_pairs": len(adj_a),
            "r_far": round(spearman(far_a, far_b), 3), "n_far_pairs": len(far_a)}


def reading(r: dict) -> str:
    if r["r_fmt"] >= 0.8 and r["r_shift"] >= 0.7: return "stable to surface form -> low cross-window r reflects content"
    if r["r_fmt"] < 0.5: return "unstable -> instability claim stands"
    return "inconclusive"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stability", type=Path, required=True)
    ap.add_argument("--rows", type=Path, default=ROOT / "data/swebench_pilot50/pilot_free_scored.jsonl")
    ap.add_argument("--out", type=Path, default=ROOT / "data/swebench_pilot50/prereg_readout_amendment4b.json")
    a = ap.parse_args()
    prim = {r["instance_id"]: r for r in map(json.loads, a.rows.read_text().splitlines()) if r}
    crit = {}
    for f in sorted(a.stability.glob("stability_*.json")):
        d = json.loads(f.read_text()); crit[f.stem.replace("stability_", "")] = d
    rep = {"analysis": "Amendment 4b (pre-registered): Spearman r under content-preserving perturbations", "critics": {}, "panel_u": None}
    for name, d in crit.items():
        rows = {i: r["windows"] for i, r in d["rows"].items()}
        r = stats(rows, lambda w: (w["orig"], w["fmt"], w["shift"])); r["device"] = d.get("device"); r["reading"] = reading(r)
        rep["critics"][name] = r
    if all(m in crit for m in PANEL):
        common = set.intersection(*(set(crit[m]["rows"]) for m in PANEL))
        tr = {m: [w["orig"] for i in common if prim[i]["split"] == "train" for w in crit[m]["rows"][i]["windows"]] for m in PANEL}
        fits = fit_rank_norms(tr)
        rows = {}
        for i in sorted(common):
            nw = min(crit[m]["rows"][i]["n_windows"] for m in PANEL); ws = []
            for k in range(nw):
                ws.append({v: apply_panel_u({m: crit[m]["rows"][i]["windows"][k][v] for m in PANEL}, fits)[1] for v in ("orig", "fmt", "shift")})
            rows[i] = ws
        r = stats(rows, lambda w: (w["orig"], w["fmt"], w["shift"])); r["members"] = PANEL; r["reading"] = reading(r)
        rep["panel_u"] = r
    else:
        rep["panel_u"] = f"missing panel critics: {[m for m in PANEL if m not in crit]}"
    a.out.write_text(json.dumps(rep, indent=1)); print(json.dumps(rep, indent=1))


if __name__ == "__main__":
    main()
