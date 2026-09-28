#!/usr/bin/env python3
"""Amendment 5 readout: the selector axis (S-P1, S-P2, S-P3).

DEPRECATED FOR PAPER NUMBERS. This script retains the defective population
filter documented in DEFECTS.md #7 / Amendment 5a: after gpt-5 re-selects, it
still filters with rows[i]["visible_pass"] from the *primary* committed
candidate. Authoritative analysis is scripts/readout_amendment5_corrected.py
→ data/selector_axis/readout_amendment5_corrected.json.

Re-selects over the existing candidate pools with a strong selector, then asks
whether panel disagreement adds anything on THAT selected set.
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT / "src"))
from silentfail.combine import _fit_standardizer, fit_logistic_irls  # noqa: E402
from silentfail.scoring import apply_panel_u, fit_rank_norms  # noqa: E402
from silentfail.stats import auroc_fast, cluster_bootstrap_delta  # noqa: E402

_DEPRECATION = (
    "prereg_readout_amendment5.py writes the DEFECTIVE 85-row population "
    "(DEFECTS #7). Use scripts/readout_amendment5_corrected.py for paper numbers. "
    "Re-run this script only with --write-defective-archive."
)

PANEL = ["ray2333_grm_gemma2_2b_rewardmodel_ft", "ray2333_grm_llama3_2_3b_rewardmodel_ft",
         "skywork_skywork_reward_v2_llama_3_2_1b", "skywork_skywork_reward_v2_qwen3_1_7b"]

def boot_auroc(y, s, grp, n=4000, seed=9):
    g = np.asarray(grp); u = np.unique(g); idx = {k: np.flatnonzero(g == k) for k in u}
    rng = np.random.default_rng(seed); v = []
    for _ in range(n):
        ii = np.concatenate([idx[k] for k in rng.choice(u, size=len(u), replace=True)])
        if 0 < y[ii].sum() < len(ii): v.append(auroc_fast(y[ii], s[ii]))
    return [round(float(np.percentile(v, 2.5)), 3), round(float(np.percentile(v, 97.5)), 3)] if v else None

def loro(X, y, grp):
    p = np.full(len(y), np.nan)
    for g in np.unique(grp):
        te = grp == g; tr = ~te
        if y[tr].sum() < 2 or (1 - y[tr]).sum() < 2: continue
        mu, sg = _fit_standardizer(X[tr]); coef, b = fit_logistic_irls((X[tr] - mu) / sg, y[tr].astype(float))
        p[te] = 1 / (1 + np.exp(-np.clip(((X[te] - mu) / sg) @ coef + b, -30, 30)))
    return p

def run(bench, rows_f, bundle_f, judge_cache, panel_dir, gold_extra_files, random_picks):
    rows = {r["instance_id"]: r for r in map(json.loads, Path(rows_f).read_text().splitlines()) if r}
    cands = {(c["instance_id"], c["candidate"]): c for c in map(json.loads, Path(bundle_f).read_text().splitlines()) if c}
    # selector scores: detection score = 100 - p_resolved, so HIGHER = worse; selection takes the argmin
    J = {}
    for d in map(json.loads, Path(judge_cache).read_text().splitlines()):
        if d.get("p_resolved") is not None: J[(d["instance_id"], d["candidate"])] = d["p_resolved"]
    # gold available per candidate: primary committed + random arm + any re-selection extras
    gold = {}
    for i, r in rows.items():
        if not r["excluded"]: gold[(i, r["selected_candidate_index"])] = r["g"]
    for f in gold_extra_files:
        if Path(f).exists():
            for d in map(json.loads, Path(f).read_text().splitlines()):
                if d.get("g") in (0, 1): gold[(d["instance_id"], d["candidate"])] = d["g"]
    if Path(random_picks).exists():
        picks = {d["instance_id"]: d["candidate"] for d in map(json.loads, Path(random_picks).read_text().splitlines())}
        rep_f = Path(random_picks).parent.parent / "gold_modal" / f"random_arm_{bench}" / "report.json"
        if rep_f.exists():
            rp = json.loads(rep_f.read_text())
            for i in rp["resolved_ids"]: gold[(i, picks[i])] = 1
            for i in rp["unresolved_ids"]: gold[(i, picks[i])] = 0

    out = {"benchmark": bench, "judge_scores": len(J), "candidates_with_gold": len(gold)}
    # ---- S-P1a: does the selector rank failures below passes WITHIN an instance? ----
    conc = tot = 0
    for i in rows:
        gs = [(k[1], gold[k]) for k in gold if k[0] == i and (i, k[1]) in J]
        pas = [c for c, g in gs if g == 1]; fal = [c for c, g in gs if g == 0]
        for a in pas:
            for b in fal:
                tot += 1; conc += J[(i, a)] > J[(i, b)]     # passing candidate scored higher
    out["pool_within_instance_concordance"] = round(conc / tot, 3) if tot else None
    out["pool_pairs_compared"] = tot
    # ---- pooled pool-level AUROC over every candidate that has gold and a score ----
    keys = [k for k in gold if k in J]
    yp = np.array([1 - gold[k] for k in keys]); sp = np.array([100 - J[k] for k in keys], float)
    gp = [rows[k[0]]["repo"] for k in keys]
    out["pool_AUROC_negc"] = round(auroc_fast(yp, sp), 3) if 0 < yp.sum() < len(yp) else None
    out["pool_AUROC_CI"] = boot_auroc(yp, sp, gp) if 0 < yp.sum() < len(yp) else None
    out["pool_n"] = len(keys)
    # ---- re-select and report what needs gold ----
    newc, needs = {}, []
    for i, r in rows.items():
        avail = [(c, J[(i, c)]) for c in range(4) if (i, c) in J and cands.get((i, c), {}).get("final_solution", "").strip()]
        if not avail: continue
        newc[i] = max(avail, key=lambda x: x[1])[0]
        if (i, newc[i]) not in gold:
            needs.append({"instance_id": i, "candidate": newc[i], "model_patch": cands[(i, newc[i])]["final_solution"],
                          "model_name_or_path": f"s5_judge_{bench}"})
    out["reselected"] = len(newc); out["changed_vs_primary"] = sum(1 for i in newc if newc[i] != rows[i]["selected_candidate_index"])
    out["needs_gold"] = len(needs)
    if needs:
        p = Path(f"data/selector_axis/needs_gold_{bench}.jsonl"); p.write_text("".join(json.dumps(n) + "\n" for n in needs))
        out["needs_gold_file"] = str(p)
    # ---- S-P1b/S-P2/S-P3 on the new committed set, where gold exists ----
    S = {m: json.load(open(Path(panel_dir) / f"scores_gpu_{m}.json"))["scores"] for m in PANEL}
    pop = [i for i in newc if (i, newc[i]) in gold and rows[i].get("visible_pass")]
    if len(pop) >= 10:
        y = np.array([1 - gold[(i, newc[i])] for i in pop]); grp = np.array([rows[i]["repo"] for i in pop])
        c = np.array([100 - J[(i, newc[i])] for i in pop], float)
        tr = [i for i in pop if rows[i]["split"] == "train"]
        fits = fit_rank_norms({m: [S[m][f"{i}#{newc[i]}"]["score"] for i in tr if f"{i}#{newc[i]}" in S[m]] for m in PANEL})
        u = np.array([apply_panel_u({m: S[m].get(f"{i}#{newc[i]}", {}).get("score", np.nan) for m in PANEL}, fits)[1] for i in pop])
        out["committed"] = {"n": len(pop), "failures": int(y.sum()), "failure_rate": round(float(y.mean()), 3),
                            "AUROC_negc": round(auroc_fast(y, c), 3), "AUROC_negc_CI": boot_auroc(y, c, grp),
                            "AUROC_u": round(auroc_fast(y, u), 3), "AUROC_u_CI": boot_auroc(y, u, grp)}
        pc, pcu = loro(c[:, None], y, grp), loro(np.column_stack([c, u]), y, grp)
        ok = ~np.isnan(pc) & ~np.isnan(pcu)
        if ok.sum() >= 10:
            t = cluster_bootstrap_delta(y[ok], pc[ok], pcu[ok], grp[ok], endpoint="auroc", n_boot=4000, seed=5)
            out["committed"]["endpoint_delta"] = round(t.delta, 3); out["committed"]["endpoint_CI"] = [round(t.ci_lo, 3), round(t.ci_hi, 3)]
        out["S_P1_selector_has_pool_signal"] = bool((out["pool_AUROC_negc"] or 0) > 0.65)
        out["S_P2_selection_consumes_signal"] = bool((out["pool_AUROC_negc"] or 0) - out["committed"]["AUROC_negc"] > 0.10)
        out["S_P3_u_adds_on_strong_selector"] = bool(out["committed"]["AUROC_u"] > 0.60 and out["committed"].get("endpoint_delta", -1) > 0.05
                                                     and (out["committed"].get("endpoint_CI") or [-1])[0] > 0)
    else:
        out["committed"] = f"only {len(pop)} rows with gold; below the inference floor"
    return out

def main():
    a = argparse.ArgumentParser(
        description=_DEPRECATION,
        epilog="Authoritative: python scripts/readout_amendment5_corrected.py",
    )
    a.add_argument("--out", type=Path, default=ROOT / "data/selector_axis/prereg_readout_amendment5.json")
    a.add_argument(
        "--write-defective-archive",
        action="store_true",
        help="Permit rewriting the archived defective readout (default: refuse).",
    )
    a = a.parse_args()
    if not a.write_defective_archive:
        print(_DEPRECATION, file=sys.stderr)
        sys.exit(2)
    print("WARNING: writing defective-population archive by explicit request.", file=sys.stderr)
    rep = {"analysis": "Amendment 5 (pre-registered): the selector axis — DEFECTIVE POPULATION (see DEFECTS #7; superseded by readout_amendment5_corrected.json)",
           "S_swerm": "NOT RUN. ksshumab/SWE-RM publishes 122 GB of Qwen3-MoE causal-LM weights with no value head, "
                      "no sequence-classification head, a 31-byte README and the stock Qwen3-Coder agent chat template "
                      "(no verdict or score token). There is no documented interface for extracting a scalar reward; "
                      "guessing one is the instrument-failure mode this project documents six times. Recorded as not "
                      "run, per the escape hatch declared in the amendment.",
           "verified": run("verified", ROOT/"data/swebench_pilot50/pilot_free_scored.jsonl", ROOT/"data/swebench_pilot50/rescoring_bundle.jsonl",
                           ROOT/"data/selector_axis/judge_verified.jsonl", ROOT/"data/swebench_pilot50/l4/out/gpu_scores_16k",
                           [ROOT/"data/swebench_pilot50/l4/out/gpu_scores_16k/gold_extra_merged.jsonl", ROOT/"data/selector_axis/gold_extra_verified.jsonl"], ROOT/"data/random_arm/verified_picks.jsonl"),
           "multilingual": run("multilingual", ROOT/"data/replication_multilingual/replication_scored.jsonl", ROOT/"data/replication_multilingual/rescoring_bundle.jsonl",
                               ROOT/"data/selector_axis/judge_multilingual.jsonl", ROOT/"data/replication_multilingual/modal/prefix4k",
                               [ROOT/"data/selector_axis/gold_extra_multilingual.jsonl"], ROOT/"data/random_arm/multilingual_picks.jsonl")}
    a.out.write_text(json.dumps(rep, indent=1)); print(json.dumps(rep, indent=1))

if __name__ == "__main__": main()
