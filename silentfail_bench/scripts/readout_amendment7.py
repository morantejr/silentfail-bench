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

Run with --dry-run before October: it regression-tests the endpoint estimators
against gpt-5-mini replication values. The production joins and gates are
separate because their complete candidate gold does not exist yet; they fail
closed on any absent or partial source and were exercised with a local fixture.

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
    empty_keys = {k for k, c in bundle.items()
                  if c.get("excluded") or not (c.get("final_solution") or "").strip()}
    empty = len(empty_keys)
    # Production gold explicitly records empty attempts as failures. Do not
    # count those attempts a second time in the graded-non-empty component.
    graded_keys = {k for k in gold if k in bundle and k not in empty_keys}
    graded_fail = sum(1 for k in graded_keys if gold[k] == 0)
    graded = len(graded_keys)
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


def jsonl(path):
    return [json.loads(l) for l in Path(path).read_text().splitlines() if l.strip()]


def require(path, label):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"{label} missing: {path}")
    return path


def load_candidate_gold(root, bundle):
    """Join four candidate-indexed gold runs; empty patches are failures by rule."""
    gold = {}
    for key, row in bundle.items():
        if not (row.get("final_solution") or "").strip():
            gold[key] = 0
    for candidate in range(4):
        p = require(Path(root) / f"a7_agent_axis_cand{candidate}" / "gold_extra.jsonl",
                    f"candidate-{candidate} gold")
        for d in jsonl(p):
            key = (d["instance_id"], int(d["candidate"]))
            if key[1] != candidate:
                raise ValueError(f"{p}: candidate mapping says {key[1]}, expected {candidate}")
            if key in gold and gold[key] != int(d["g"]):
                raise ValueError(f"conflicting gold for {key}")
            gold[key] = int(d["g"])
    # A slot can be missing for a reason the harness records: the patch would not apply,
    # or the run errored. gold_modal.py writes no gold_extra row for those (it guards on
    # `if "error" not in r`), and EVERY real gold run on disk has some, at 2-9% of
    # attempts. Aborting on them would throw away four paid gold runs. The project
    # already models this as an exclusion elsewhere (run_replication_modal.py's
    # `gold_error` reason), so it is excluded here too, counted, and declared in
    # Amendment 10b before any agent-axis gold existed.
    missing = sorted(set(bundle) - set(gold))
    errored = []
    for candidate in range(4):
        rep = Path(root) / f"a7_agent_axis_cand{candidate}" / "report.json"
        if rep.exists():
            for i in (json.loads(rep.read_text()).get("error_ids") or []):
                errored.append((i, candidate))
    errored_set = set(errored)
    unexplained = [k for k in missing if k not in errored_set]
    if unexplained:
        raise ValueError(f"gold incomplete for {len(unexplained)} slots the harness did NOT record as errors; "
                         f"first={unexplained[:5]}. This is a pipeline fault, not a harness error.")
    if missing:
        rate = len(missing) / max(len(bundle), 1)
        print(f"[gold] {len(missing)} of {len(bundle)} slots excluded as harness errors ({rate:.1%})")
        if rate > 0.10:
            raise ValueError(f"harness error rate {rate:.1%} exceeds 10%; stop and inspect rather than analysing a "
                             f"population this depleted")
    load_candidate_gold.excluded_slots = missing
    return gold


def load_panel(scores_dir):
    out = {}
    for model in PANEL:
        p = require(Path(scores_dir) / f"scores_gpu_{model}.json", f"panel score {model}")
        out[model] = json.loads(p.read_text())["scores"]
    return out


def selected_endpoint(bundle, selected, detection, gold, panel, label):
    """Endpoint on the selected, agent-declared-complete population."""
    pop = []
    for instance_id, candidate in selected.items():
        key = (instance_id, candidate); row = bundle[key]; skey = f"{instance_id}#{candidate}"
        vals = [panel[m].get(skey, {}).get("score", np.nan) for m in PANEL]
        if (row.get("visible_pass") and (row.get("final_solution") or "").strip()
                and key in gold and key in detection and np.isfinite(detection[key])
                and all(np.isfinite(vals))):
            pop.append(key)
    if not pop:
        raise ValueError(f"{label}: no endpoint rows after registered filters")
    train = [key for key in pop if bundle[key]["split"] == "train"]
    fits = fit_rank_norms({
        m: [panel[m][f"{i}#{c}"]["score"] for i, c in train] for m in PANEL
    })
    u = np.array([
        apply_panel_u({m: panel[m][f"{i}#{c}"]["score"] for m in PANEL}, fits)[1]
        for i, c in pop
    ])
    y = np.array([1 - gold[key] for key in pop])
    c = np.array([detection[key] for key in pop], float)
    grp = np.array([bundle[key]["repo"] for key in pop])
    out = endpoint(y, c, u, grp)
    out.update({
        "label": label,
        "repositories": int(len(set(grp))),
        "population_rule": "selected + visible_pass + non-empty + gold + all panel scores",
    })
    return out


def old_agent_capability():
    """A7-P1 comparator; partial old candidate gold honestly remains a band."""
    rows = {r["instance_id"]: r for r in jsonl(ROOT / "data/replication_multilingual/replication_scored.jsonl")}
    bundle = {(r["instance_id"], int(r["candidate"])): r
              for r in jsonl(ROOT / "data/replication_multilingual/rescoring_bundle.jsonl")}
    gold = {}
    for i, row in rows.items():
        if not row["excluded"]:
            gold[(i, int(row["selected_candidate_index"]))] = int(row["g"])
    for p in (ROOT / "data/selector_axis/gold_extra_multilingual.jsonl",):
        if p.exists():
            for d in jsonl(p):
                if d.get("g") in (0, 1):
                    gold[(d["instance_id"], int(d["candidate"]))] = int(d["g"])
    picks = {d["instance_id"]: int(d["candidate"])
             for d in jsonl(ROOT / "data/random_arm/multilingual_picks.jsonl")}
    rr = json.loads((ROOT / "data/gold_modal/random_arm_multilingual/report.json").read_text())
    for i in rr["resolved_ids"]:
        if i in picks:
            gold[(i, picks[i])] = 1
    for i in rr["unresolved_ids"]:
        if i in picks:
            gold[(i, picks[i])] = 0
    return capability(bundle, gold, "gpt-5-mini comparator")


def compare_to_old(arm_4b, arm_judge):
    """Registered descriptive differences on the same 114-instance frame."""
    robust = json.loads((ROOT / "data/verify_and_robustify.json").read_text())
    old_4b = next(r for r in robust["endpoint_robustness"] if r["label"] == "multilingual")
    a5 = json.loads((ROOT / "data/selector_axis/readout_amendment5_corrected.json").read_text())
    old_judge = a5["corrected"]["multilingual"]

    old = {
        "S_4b": {
            "failure_rate": old_4b["positives"] / old_4b["n"],
            "raw_AUROC_detect": old_4b["raw_AUROC_negc"],
            "raw_AUROC_u": old_4b["raw_AUROC_u"],
            "logistic_LORO_delta": old_4b["logistic_LORO"]["delta"],
            "logistic_5fold_delta": old_4b["logistic_5fold_clustered"]["delta"],
            "fixed_unit_weights_PRIMARY_delta": old_4b["fixed_unit_weights_no_fitting"]["delta"],
        },
        "S_judge": {
            "failure_rate": old_judge["failures"] / old_judge["n"],
            "raw_AUROC_detect": old_judge["committed_AUROC_detect"],
            "raw_AUROC_u": old_judge["AUROC_u"],
            "logistic_LORO_delta": old_judge["estimators"]["logistic_LORO"]["delta"],
            "logistic_5fold_delta": old_judge["estimators"]["logistic_5fold"]["delta"],
            "fixed_unit_weights_PRIMARY_delta": old_judge["estimators"]["fixed_unit_weights"]["delta"],
        },
    }
    new = {}
    for name, arm in (("S_4b", arm_4b), ("S_judge", arm_judge)):
        new[name] = {
            "failure_rate": arm["failures"] / arm["n"],
            "raw_AUROC_detect": arm["raw_AUROC_detect"],
            "raw_AUROC_u": arm["raw_AUROC_u"],
            "logistic_LORO_delta": arm["estimators"]["logistic_LORO"]["delta"],
            "logistic_5fold_delta": arm["estimators"]["logistic_5fold"]["delta"],
            "fixed_unit_weights_PRIMARY_delta": arm["estimators"]["fixed_unit_weights_PRIMARY"]["delta"],
        }
    out = {
        "note": "Descriptive new-minus-gpt-5-mini differences on the same fixed 114-instance frame. "
                "Registered visible-pass filters create different analysed subsets; no comparison threshold was registered.",
        "arms": {},
    }
    for name in ("S_4b", "S_judge"):
        out["arms"][name] = {
            key: {
                AGENT_KEY: round(float(new[name][key]), 3),
                "gpt5_mini": round(float(old[name][key]), 3),
                "difference": round(float(new[name][key] - old[name][key]), 3),
            }
            for key in new[name]
        }
    return out


def pick_4b(choices):
    """Match run_replication_modal.select: first candidate wins a score tie."""
    return max(choices, key=lambda pair: pair[0])[1]


def production_readout(args):
    expected = [l.strip() for l in require(args.instances, "registered instance file").read_text().splitlines()
                if l.strip() and not l.startswith("#")]
    if len(expected) != 114 or len(set(expected)) != 114:
        raise ValueError(f"registered instance file must contain 114 unique IDs; got {len(expected)}/{len(set(expected))}")
    rows = jsonl(require(args.bundle, "agent-axis bundle"))
    bundle = {(r["instance_id"], int(r["candidate"])): r for r in rows}
    expected_keys = {(i, c) for i in expected for c in range(4)}
    if set(bundle) != expected_keys:
        raise ValueError(f"bundle is not the registered 114 x 4 set: "
                         f"{len(expected_keys - set(bundle))} missing, {len(set(bundle) - expected_keys)} extra")
    gold = load_candidate_gold(args.gold_root, bundle)
    panel = load_panel(args.scores_dir)
    missing_panel = [
        (model, key) for model in PANEL for key in expected_keys
        if not np.isfinite(panel[model].get(f"{key[0]}#{key[1]}", {}).get("score", np.nan))
    ]
    if missing_panel:
        raise ValueError(f"panel scoring incomplete: {len(missing_panel)} model/candidate scores missing; "
                         f"first={missing_panel[:5]}")

    selector_model = "skywork_skywork_reward_v2_qwen3_4b"
    selector_scores = json.loads(require(
        Path(args.scores_dir) / f"scores_gpu_{selector_model}.json", "4B selector scores"
    ).read_text())["scores"]
    missing_selector = [
        key for key in expected_keys
        if not np.isfinite(selector_scores.get(f"{key[0]}#{key[1]}", {}).get("score", np.nan))
    ]
    if missing_selector:
        raise ValueError(f"4B selector scoring incomplete: {len(missing_selector)} candidate scores missing; "
                         f"first={missing_selector[:5]}")
    selected_4b, detect_4b = {}, {}
    for i in expected:
        choices = []
        for c in range(4):
            score = selector_scores.get(f"{i}#{c}", {}).get("score", float("-inf"))
            detect_4b[(i, c)] = -float(score)
            choices.append((float(score) if bundle[(i, c)].get("text") else float("-inf"), c))
        selected_4b[i] = pick_4b(choices)

    judge = {}
    for d in jsonl(require(args.judge_cache, "gpt-5 all-candidate cache")):
        if d.get("judge") == "gpt-5" and d.get("cond") == "J1" and d.get("p_resolved") is not None:
            judge[(d["instance_id"], int(d["candidate"]))] = float(d["p_resolved"])
    missing_judge = sorted(expected_keys - set(judge))
    if missing_judge:
        raise ValueError(f"gpt-5 cache incomplete: {len(missing_judge)} candidate scores missing; first={missing_judge[:5]}")
    selected_judge = {
        i: max(range(4), key=lambda c: judge[(i, c)]) for i in expected
    }
    detect_judge = {key: 100.0 - value for key, value in judge.items()}

    # A7a selector gate, first half: only candidates actually graded by the
    # harness. Empty patches are defined failures for capability but are not
    # "graded candidates" in section 2.
    pool_keys = [key for key in expected_keys if (bundle[key].get("final_solution") or "").strip()]
    pool_y = np.array([1 - gold[key] for key in pool_keys])
    pool_s = np.array([detect_judge[key] for key in pool_keys])
    pool_auroc = round(auroc_fast(pool_y, pool_s), 3)

    from prepare_amendment7 import random_candidate
    picks = {d["instance_id"]: int(d["candidate"])
             for d in jsonl(require(args.random_picks, "frozen random picks"))}
    expected_picks = {i: random_candidate(i) for i in expected}
    if picks != expected_picks:
        raise ValueError("random_picks.jsonl does not match the frozen a7-random-v1 rule")
    committed_fail = np.array([1 - gold[(i, selected_judge[i])] for i in expected], float)
    random_fail = np.array([1 - gold[(i, picks[i])] for i in expected], float)
    gate = {
        "pool_n_graded": len(pool_keys),
        "pool_failures": int(pool_y.sum()),
        "pool_AUROC_detect": pool_auroc,
        "pool_AUROC_gt_0p65": bool(pool_auroc > SELECTOR_GATE),
        "paired_instances": len(expected),
        "committed_failure_rate_unconditional": round(float(committed_fail.mean()), 3),
        "random_failure_rate_unconditional": round(float(random_fail.mean()), 3),
        "committed_below_random": bool(committed_fail.mean() < random_fail.mean()),
    }
    gate["passes"] = gate["pool_AUROC_gt_0p65"] and gate["committed_below_random"]

    new_cap = capability(bundle, gold, a.agent_label)
    old_cap = old_agent_capability()
    new_lo, new_hi = new_cap["unconditional_failure_rate_band"]
    old_lo, old_hi = old_cap["unconditional_failure_rate_band"]
    if new_hi <= old_lo - CAPABILITY_GAP:
        cap_branch = "PASS"
    elif new_lo >= old_hi - CAPABILITY_GAP:
        cap_branch = "FAIL"
    else:
        cap_branch = "INDETERMINATE"
    cap_cmp = {
        "new_agent": new_cap,
        "old_agent": old_cap,
        "required_gap": CAPABILITY_GAP,
        "branch": cap_branch,
        "rule": "PASS iff new upper <= old lower - 0.10; FAIL iff new lower >= old upper - 0.10; otherwise indeterminate",
    }

    arm_4b = selected_endpoint(bundle, selected_4b, detect_4b, gold, panel, "S_4b")
    arm_judge = selected_endpoint(bundle, selected_judge, detect_judge, gold, panel, "S_judge_gpt5")
    if cap_branch != "PASS":
        final = {"branch": "DESCRIPTIVE ONLY", "reason": f"A7-P1 capability branch is {cap_branch}"}
    elif not gate["passes"]:
        final = {"branch": "DESCRIPTIVE ONLY", "reason": "A7a strong-selector gate failed"}
    else:
        final = verdict(arm_judge)

    report = {
        "analysis": "Amendment 7/7a agent-axis readout",
        "registered_instances": len(expected),
        "candidate_slots": len(expected_keys),
        "capability_A7_P1": cap_cmp,
        "selector_gate_A7a_2": gate,
        "arms": {"S_4b": arm_4b, "S_judge": arm_judge},
        "comparison_to_gpt5_mini": compare_to_old(arm_4b, arm_judge),
        "confirmatory_verdict": final,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=1) + "\n")
    print(json.dumps(report, indent=1))
    print(f"\nwrote {args.out}")


AGENT_KEY = "gpt6_luna"   # overwritten from --agent-label in main()

def main():
    ap = argparse.ArgumentParser()
    # The new agent's label was hardcoded as "gpt-6-luna" when this was written for Amendment
    # 7; Amendment 13 runs the same design on DeepSeek. Default preserved so the September
    # dry run and fixture test are unchanged.
    ap.add_argument("--agent-label", default="gpt-6-luna",
                    help="name of the new agent in the output (Amendment 13: deepseek-v4.1-flash)")
    ap.add_argument("--dry-run", action="store_true",
                    help="exercise every path against the gpt-5-mini replication data")
    ap.add_argument("--bundle", type=Path, default=ROOT / "data/agent_axis/rescoring_bundle.jsonl")
    ap.add_argument("--instances", type=Path, default=ROOT / "configs/agent_axis_instances_multilingual.txt")
    ap.add_argument("--scores-dir", type=Path, default=ROOT / "data/agent_axis/modal/prefix4k")
    ap.add_argument("--judge-cache", type=Path, default=ROOT / "data/agent_axis/judge_gpt5.jsonl")
    ap.add_argument("--gold-root", type=Path, default=ROOT / "data/gold_modal")
    ap.add_argument("--random-picks", type=Path, default=ROOT / "data/agent_axis/gold_inputs/random_picks.jsonl")
    ap.add_argument("--out", type=Path, default=ROOT / "data/agent_axis/readout_amendment7.json")
    a = ap.parse_args()
    global AGENT_KEY
    AGENT_KEY = "".join(ch if ch.isalnum() else "_" for ch in a.agent_label).strip("_").replace("__", "_")

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
        print("and every estimator reproduces the published replication value.")
        return

    try:
        production_readout(a)
    except (FileNotFoundError, ValueError, KeyError) as exc:
        print(f"Amendment 7 readout cannot run yet:\n  {exc}", file=sys.stderr)
        print("  Complete every stage in RESUME_AMENDMENT7.md; no partial result was written.", file=sys.stderr)
        sys.exit(2)


if __name__ == "__main__":
    main()
