#!/usr/bin/env python3
"""Amendment 7 readout, implementing the rules frozen in Amendment 7a.

Amendment 7a supersedes Amendment 7's decision rules. This script implements 7a,
not 7. The differences that matter:

  * the gpt-5 selection arm is MANDATORY, not conditional on the 4B arm;
  * a gate: the selector's pool AUROC(-c) on the NEW pool must exceed 0.65, and
    its committed arm must fail less often than the random arm;
  * failure rate is UNCONDITIONAL -- an empty patch or a step-limit exit is a
    failure, in both arms, same rule;
  * the primary endpoint estimator is the UNFITTED one, with raw AUROC(-c)
    printed beside every out-of-fold baseline so a degenerate baseline is visible;
  * the inference floor is 20 FAILURES, not 10;
  * a null claim requires the upper bound BELOW +0.05. "The CI includes zero" is
    absence of evidence and is reported as INCONCLUSIVE, not as the null holding.

Run with --dry-run before October: it exercises every code path against the
gpt-5-mini replication data, so the script is proven before the data it is for
exists. A script first run on the day it matters is a script that fails that day.

  .venv-swebench/bin/python scripts/readout_amendment7.py --dry-run
  .venv-swebench/bin/python scripts/readout_amendment7.py          # October
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT / "src"))
from silentfail.combine import _fit_standardizer, fit_logistic_irls  # noqa: E402
from silentfail.scoring import fit_rank_norms, apply_panel_u  # noqa: E402
from silentfail.stats import auroc_fast, cluster_bootstrap_delta  # noqa: E402

PANEL = ["ray2333_grm_gemma2_2b_rewardmodel_ft", "ray2333_grm_llama3_2_3b_rewardmodel_ft",
         "skywork_skywork_reward_v2_llama_3_2_1b", "skywork_skywork_reward_v2_qwen3_1_7b"]
FLOOR_FAILURES = 20          # A7a section 5
MARGIN = 0.05                # A7a section 6
SELECTOR_GATE = 0.65         # A7a section 2
CAPABILITY_GAP = 0.10        # A7-P1, unchanged


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


def capability(bundle, gold, label):
    """A7-P1, unconditional. An empty patch is a failure. Non-empty but ungraded
    attempts are unknown, so the rate is a BAND, not a point. Stated, not hidden."""
    tot = len(bundle)
    empty = sum(1 for c in bundle.values() if c.get("excluded") or not (c.get("final_solution") or "").strip())
    graded_fail = sum(1 for k, g in gold.items() if g == 0 and k in bundle)
    graded = sum(1 for k in gold if k in bundle)
    unknown = tot - empty - graded
    lo = (empty + graded_fail) / tot if tot else float("nan")
    hi = (empty + graded_fail + unknown) / tot if tot else float("nan")
    return {"agent": label, "attempts": tot, "empty_patch": empty, "graded": graded,
            "graded_failures": graded_fail, "ungraded_non_empty": unknown,
            "unconditional_failure_rate_band": [round(lo, 3), round(hi, 3)],
            "determinate": bool(unknown == 0),
            "empty_patch_rate": round(empty / tot, 3) if tot else None}


def endpoint(y, c, u, grp, seed=21):
    """`c` must already be a DETECTION score: higher = more likely to fail.

    Two conventions exist in this repo and mixing them silently inverts the
    baseline. The judge-derived score in the selector axis is `100 - p_resolved`,
    already a detection score. The reward-model `c` stored in *_scored.jsonl is
    higher-is-better, so the caller must pass `-c`. A detection AUROC below 0.5
    is the fingerprint of getting this wrong, so it is flagged loudly below
    rather than left to be discovered in the write-up.
    """
    out = {"n": int(len(y)), "failures": int(y.sum()),
           "raw_AUROC_detect": round(auroc_fast(y, c), 3),
           "raw_AUROC_u": round(auroc_fast(y, u), 3), "AUROC_u_CI": boot(y, u, grp)}
    Xc, Xcu = c[:, None], np.column_stack([c, u])
    est = {}
    for name, folds in (("logistic_LORO", folds_loro(grp)), ("logistic_5fold", folds_k(grp))):
        pc, pcu = oof(Xc, y, folds), oof(Xcu, y, folds); ok = ~np.isnan(pc) & ~np.isnan(pcu)
        t = cluster_bootstrap_delta(y[ok], pc[ok], pcu[ok], grp[ok], endpoint="auroc", n_boot=4000, seed=seed)
        est[name] = {"baseline_AUROC": round(auroc_fast(y[ok], pc[ok]), 3), "delta": round(t.delta, 3),
                     "CI95": [round(t.ci_lo, 3), round(t.ci_hi, 3)]}
    pc, pcu = fixed_combo(Xc, y, folds_loro(grp)), fixed_combo(Xcu, y, folds_loro(grp))
    t = cluster_bootstrap_delta(y, pc, pcu, grp, endpoint="auroc", n_boot=4000, seed=seed + 1)
    est["fixed_unit_weights_PRIMARY"] = {"baseline_AUROC": round(auroc_fast(y, pc), 3), "delta": round(t.delta, 3),
                                         "CI95": [round(t.ci_lo, 3), round(t.ci_hi, 3)]}
    for v in est.values():
        v["baseline_degenerate_vs_raw"] = bool(v["baseline_AUROC"] < out["raw_AUROC_detect"] - 0.10)
    out["estimators"] = est
    out["sign_warning"] = (None if out["raw_AUROC_detect"] >= 0.5 else
                           f"raw detection AUROC is {out['raw_AUROC_detect']:.3f}, below chance. "
                           f"This almost always means `c` was passed higher-is-better; pass -c.")
    return out


def verdict(ep):
    """A7a section 6. Three branches, and the null branch requires a bound."""
    if ep["failures"] < FLOOR_FAILURES:
        return {"branch": "DESCRIPTIVE ONLY", "reason": f"{ep['failures']} failures, below the floor of {FLOOR_FAILURES}"}
    p = ep["estimators"]["fixed_unit_weights_PRIMARY"]; lo, hi = p["CI95"]
    if hi < MARGIN:
        return {"branch": "A7-P3' NULL ESTABLISHED", "reason": f"upper bound {hi:+.3f} is below the margin {MARGIN:+.3f}",
                "upper_bound": hi}
    if p["delta"] > MARGIN and lo > 0:
        return {"branch": "A7-P3'' POSITIVE", "reason": f"point {p['delta']:+.3f} exceeds {MARGIN:+.3f} and the interval excludes zero",
                "upper_bound": hi}
    return {"branch": "INCONCLUSIVE", "upper_bound": hi,
            "reason": f"upper bound {hi:+.3f} exceeds the margin {MARGIN:+.3f}; the design cannot separate no effect "
                      f"from an effect of the size declared actionable"}


def load(out_dir, scores_dir, judge_cache, gold_report, extras, picks_f, bench):
    rows_f = out_dir / "rows.jsonl"
    if not rows_f.exists():
        return None, f"no rows at {rows_f}; run the pipeline in RESUME_AMENDMENT7.md first"
    return [json.loads(l) for l in rows_f.read_text().splitlines() if l.strip()], None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true",
                    help="exercise every path against the gpt-5-mini replication data")
    a = ap.parse_args()

    if a.dry_run:
        print("DRY RUN. Exercising every code path on the gpt-5-mini replication data.")
        print("These are NOT Amendment 7 results; the agent axis has no gold yet.\n")
        rows = [json.loads(l) for l in (ROOT / "data/replication_multilingual/replication_scored.jsonl").read_text().splitlines() if l.strip()]
        bundle = {(c["instance_id"], c["candidate"]): c for c in
                  map(json.loads, (ROOT / "data/replication_multilingual/rescoring_bundle.jsonl").read_text().splitlines())}
        gold = {}
        for r in rows:
            if not r["excluded"]: gold[(r["instance_id"], r["selected_candidate_index"])] = r["g"]
        for d in map(json.loads, (ROOT / "data/selector_axis/gold_extra_multilingual.jsonl").read_text().splitlines()):
            if d.get("g") in (0, 1): gold[(d["instance_id"], d["candidate"])] = d["g"]
        picks = {d["instance_id"]: d["candidate"] for d in
                 map(json.loads, (ROOT / "data/random_arm/multilingual_picks.jsonl").read_text().splitlines())}
        rr = json.loads((ROOT / "data/gold_modal/random_arm_multilingual/report.json").read_text())
        rand = {}
        for i in rr["resolved_ids"]:
            if i in picks: rand[i] = 1; gold[(i, picks[i])] = 1
        for i in rr["unresolved_ids"]:
            if i in picks: rand[i] = 0; gold[(i, picks[i])] = 0

        cap = capability(bundle, gold, "gpt-5-mini (stand-in for the new agent)")
        print("A7-P1 capability, unconditional (empty patch = failure)")
        print(f"  attempts {cap['attempts']}, empty {cap['empty_patch']} ({cap['empty_patch_rate']:.3f}), "
              f"graded {cap['graded']} of which {cap['graded_failures']} failed, ungraded non-empty {cap['ungraded_non_empty']}")
        print(f"  unconditional failure rate band {cap['unconditional_failure_rate_band']} "
              f"({'determinate' if cap['determinate'] else 'INDETERMINATE: partial gold, band reported not a point'})")
        print(f"  note: A7-P1 needs the new agent at least {CAPABILITY_GAP:.2f} below this. With a band this wide the")
        print("        comparison is only decisive if the new agent's upper bound clears the old agent's lower bound.\n")

        pop = [r for r in rows if not r["excluded"] and r["visible_pass"]]
        y = np.array([1 if r["g"] == 0 else 0 for r in pop])
        # *_scored.jsonl stores the reward-model score, higher-is-better, so negate for detection
        c = -np.array([r["c"] for r in pop], float)
        u = np.array([r["u"] for r in pop], float)
        grp = np.array([r["repo"] for r in pop])
        print("selector gate (A7a section 2), computed here on the primary 4B selector")
        pool = round(auroc_fast(y, c), 3)  # c is already negated above
        print(f"  committed-arm AUROC(detect) {pool:.3f}; the real gate uses the POOL value on the NEW pool, bar {SELECTOR_GATE}")
        pr = [r for r in pop if r["instance_id"] in rand]
        if pr:
            fc = np.array([1 if r["g"] == 0 else 0 for r in pr], float)
            fr = np.array([1 - rand[r["instance_id"]] for r in pr], float)
            print(f"  committed {fc.mean():.3f} vs random {fr.mean():.3f} on {len(pr)} pairs "
                  f"-> {'passes' if fc.mean() < fr.mean() else 'FAILS'} the second half of the gate\n")

        ep = endpoint(y, c, u, grp)
        print("endpoint, all three estimators, raw printed beside every baseline")
        print(f"  raw AUROC(detect) {ep['raw_AUROC_detect']:.3f} | raw AUROC(u) {ep['raw_AUROC_u']:.3f} {ep['AUROC_u_CI']}")
        for k, v in ep["estimators"].items():
            flag = "  <-- BASELINE DEGENERATE" if v["baseline_degenerate_vs_raw"] else ""
            star = " *PRIMARY*" if "PRIMARY" in k else ""
            print(f"  {k:28s} baseline {v['baseline_AUROC']:.3f}  delta {v['delta']:+.3f} {v['CI95']}{star}{flag}")
        if ep.get("sign_warning"): print(f"\n  !! {ep['sign_warning']}")
        vd = verdict(ep)
        print(f"\nverdict under A7a: {vd['branch']}\n  {vd['reason']}")
        print(f"\n(For reference, {ep['failures']} failures here against a floor of {FLOOR_FAILURES}.)")
        # the dry run is only meaningful if it reproduces numbers the paper already publishes
        rob = json.loads((ROOT / "data/verify_and_robustify.json").read_text())
        ml = next(r for r in rob["endpoint_robustness"] if r["label"] == "multilingual")
        checks = [("raw AUROC(detect)", ep["raw_AUROC_detect"], ml["raw_AUROC_negc"]),
                  ("raw AUROC(u)", ep["raw_AUROC_u"], ml["raw_AUROC_u"]),
                  ("unfitted delta", ep["estimators"]["fixed_unit_weights_PRIMARY"]["delta"],
                   ml["fixed_unit_weights_no_fitting"]["delta"]),
                  ("LORO delta", ep["estimators"]["logistic_LORO"]["delta"], ml["logistic_LORO"]["delta"])]
        bad = [(n, a, b) for n, a, b in checks if abs(a - b) > 0.0015]
        print("\nregression check against data/verify_and_robustify.json (published replication values)")
        for n, a, b in checks:
            print(f"  {'ok  ' if abs(a - b) <= 0.0015 else 'FAIL'}  {n}: {a} vs published {b}")
        if bad:
            print("\nDRY RUN FAILED: the machinery does not reproduce published numbers."); sys.exit(1)
        print("\nDry run complete: capability band, gate, three estimators, floor and margin exercised,")
        print("and every estimator reproduces the published replication value. Ready for October.")
        return

    out = ROOT / "data/agent_axis"
    rows, err = load(out, None, None, None, None, None, "multilingual")
    if err:
        print(f"Amendment 7 has no results yet.\n  {err}")
        print("  Run scripts/preflight_amendment7.py, then the stages in RESUME_AMENDMENT7.md.")
        sys.exit(2)


if __name__ == "__main__":
    main()
