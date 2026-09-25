#!/usr/bin/env python3
"""Amendment 2 readout: ingest gpu_scores/scores_gpu_*.json and report beside the primary.

Two analyses, both pre-registered:
  AUDIT-ONLY  (no new gold): on the PRIMARY committed set (34 usable), every
              GPU-scored critic as a detector of the 12 known silent failures
              (A2-P1 for AceCodeRM), and u from the full-length panel (A2-P2),
              plus prefix-vs-full correlation of c (A2-P3).
  RESELECT    (needs gold for newly committed candidates): for each selection
              critic, argmax over 4 candidates per instance; instances whose
              new committed candidate differs from the primary's need a gold
              pass -> written to needs_gold.jsonl (harness predictions format).
              M1-M3 for the reselected set are reported only once gold exists.
"""
from __future__ import annotations

import argparse, json, sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from silentfail.scoring import apply_panel_u, fit_rank_norms  # noqa: E402
from silentfail.stats import auroc_fast, cluster_bootstrap_delta  # noqa: E402

SEL_CRITICS = ["skywork_skywork_reward_v2_llama_3_1_8b", "tiger_lab_acecoderm_7b"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu-scores", type=Path, required=True)
    ap.add_argument("--bundle", type=Path, default=ROOT / "data/swebench_pilot50/rescoring_bundle.jsonl")
    ap.add_argument("--primary-rows", type=Path, default=ROOT / "data/swebench_pilot50/pilot_free_scored.jsonl")
    ap.add_argument("--gold-extra", type=Path, default=None, help="jsonl of {instance_id,candidate,g} for reselected patches")
    a = ap.parse_args()

    cands = {(c["instance_id"], c["candidate"]): c for c in map(json.loads, a.bundle.read_text().splitlines()) if c}
    prim = {r["instance_id"]: r for r in map(json.loads, a.primary_rows.read_text().splitlines()) if r}
    gpu = {}
    for f in sorted(a.gpu_scores.glob("scores_gpu_*.json")):
        d = json.loads(f.read_text()); gpu[f.stem.replace("scores_gpu_", "")] = d["scores"]
    print(f"[gpu] critics loaded: {list(gpu)}", flush=True)
    usable = [i for i, r in prim.items() if not r["excluded"]]
    pop = [i for i in usable if prim[i].get("visible_pass")]
    y = np.array([1 if prim[i]["g"] == 0 else 0 for i in pop]); grp = [prim[i]["repo"] for i in pop]
    key = lambda i: f"{i}#{prim[i]['selected_candidate_index']}"
    rep = {"population_n": len(pop), "silent_failures": int(y.sum()), "audit_only": {}, "reselect": {}}

    # ---- AUDIT-ONLY on the primary committed set ----
    for crit, sc in gpu.items():
        c = np.array([sc[key(i)]["score"] for i in pop], float)
        if np.isnan(c).any(): c = np.where(np.isnan(c), np.nanmedian(c), c)
        r = {"AUROC_negc_full": round(auroc_fast(y, -c), 3), "truncated_rows": int(sum(sc[key(i)]["truncated"] for i in pop))}
        if crit == "skywork_skywork_reward_v2_qwen3_4b":
            cp = np.array([prim[i]["c"] for i in pop], float); r["corr_prefix_vs_full_c"] = round(float(np.corrcoef(cp, c)[0, 1]), 3)
        rep["audit_only"][crit] = r
    # full-length u from the 4 registered auditors (+ AceCode as 5th, if present), rank norms on train only
    for label, members in (("panel_m4_full", ["ray2333_grm_gemma2_2b_rewardmodel_ft", "ray2333_grm_llama3_2_3b_rewardmodel_ft",
                                              "skywork_skywork_reward_v2_llama_3_2_1b", "skywork_skywork_reward_v2_qwen3_1_7b"]),
                           ("panel_m5_with_acecode", ["ray2333_grm_gemma2_2b_rewardmodel_ft", "ray2333_grm_llama3_2_3b_rewardmodel_ft",
                                              "skywork_skywork_reward_v2_llama_3_2_1b", "skywork_skywork_reward_v2_qwen3_1_7b", "tiger_lab_acecoderm_7b"])):
        if not all(m in gpu for m in members): rep["audit_only"][label] = "missing critic files"; continue
        tr = {m: [gpu[m][key(i)]["score"] for i in usable if prim[i]["split"] == "train"] for m in members}
        fits = fit_rank_norms(tr)
        u = np.array([apply_panel_u({m: gpu[m][key(i)]["score"] for m in members}, fits)[1] for i in pop])
        cp = np.array([prim[i]["c"] for i in pop], float)
        t = cluster_bootstrap_delta(y, -cp, u, grp, endpoint="auroc", n_boot=4000, seed=1)
        rep["audit_only"][label] = {"M1_AUROC_u_full": round(auroc_fast(y, u), 3), "primary_M1_prefix": 0.508,
                                   "A2_P2_gain_gt_0p10": bool(auroc_fast(y, u) - 0.508 > 0.10),
                                   "M3_vs_primary_c": round(t.delta, 3), "CI95": [round(t.ci_lo, 3), round(t.ci_hi, 3)]}

    # ---- RESELECT with each Amendment-2 selection critic ----
    gold_extra = {}
    if a.gold_extra and a.gold_extra.exists():
        for l in a.gold_extra.read_text().splitlines():
            if l.strip(): d = json.loads(l); gold_extra[(d["instance_id"], d["candidate"])] = d["g"]
    needs = []
    for crit in SEL_CRITICS:
        if crit not in gpu: rep["reselect"][crit] = "missing"; continue
        sc = gpu[crit]; new_commit = {}
        for i in usable:
            ks = [(k, sc.get(f"{i}#{k}", {}).get("score", float("-inf"))) for k in range(4) if cands[(i, k)]["final_solution"].strip()]
            if ks: new_commit[i] = max(ks, key=lambda x: x[1])[0]
        # g == -1 in gold_extra = harness could not apply/grade the patch -> gold_error exclusion, as in the pilot
        gold_err = [i for i in new_commit if gold_extra.get((i, new_commit[i])) == -1]
        for i in gold_err: del new_commit[i]
        changed = [i for i in new_commit if new_commit[i] != prim[i]["selected_candidate_index"]]
        have = {i: (prim[i]["g"] if new_commit[i] == prim[i]["selected_candidate_index"] else gold_extra.get((i, new_commit[i]))) for i in new_commit}
        missing = [i for i, g in have.items() if g is None]
        for i in missing: needs.append({"instance_id": i, "candidate": new_commit[i], "model_patch": cands[(i, new_commit[i])]["final_solution"], "model_name_or_path": f"a2_{crit}"})
        r = {"n_reselected": len(new_commit), "changed_vs_primary": len(changed), "gold_missing": len(missing), "gold_error_excluded": gold_err}
        if not missing and len(new_commit) >= 10:
            yy = np.array([1 if have[i] == 0 else 0 for i in new_commit]); gg = [prim[i]["repo"] for i in new_commit]
            c = np.array([sc[f"{i}#{new_commit[i]}"]["score"] for i in new_commit], float)
            panel = [m for m in ["ray2333_grm_gemma2_2b_rewardmodel_ft", "ray2333_grm_llama3_2_3b_rewardmodel_ft", "skywork_skywork_reward_v2_llama_3_2_1b",
                                 "skywork_skywork_reward_v2_qwen3_1_7b", "tiger_lab_acecoderm_7b", "skywork_skywork_reward_v2_llama_3_1_8b"] if m in gpu and m != crit]
            tr = {m: [gpu[m][f"{i}#{new_commit[i]}"]["score"] for i in new_commit if prim[i]["split"] == "train"] for m in panel}
            fits = fit_rank_norms(tr); u = np.array([apply_panel_u({m: gpu[m][f"{i}#{new_commit[i]}"]["score"] for m in panel}, fits)[1] for i in new_commit])
            t = cluster_bootstrap_delta(yy, -c, u, gg, endpoint="auroc", n_boot=4000, seed=1)
            r.update(failures=int(yy.sum()), M1_AUROC_u=round(auroc_fast(yy, u), 3), M2_AUROC_negc=round(auroc_fast(yy, -c), 3),
                     M3=round(t.delta, 3), CI95=[round(t.ci_lo, 3), round(t.ci_hi, 3)], p=round(t.p_one_sided, 4), panel=panel,
                     inference=("<10 positives" if yy.sum() < 10 else "eligible"))
        rep["reselect"][crit] = r
    if needs:
        p = a.gpu_scores / "needs_gold.jsonl"; p.write_text("".join(json.dumps(n) + "\n" for n in needs))
        rep["needs_gold_file"] = str(p); rep["needs_gold_n"] = len(needs)
    print(json.dumps(rep, indent=2))


if __name__ == "__main__":
    main()
