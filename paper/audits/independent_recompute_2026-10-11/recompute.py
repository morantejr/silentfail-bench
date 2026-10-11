#!/usr/bin/env python3
"""Independent recomputation of the headline numbers in paper/main.tex, from raw data only.

No project code is imported. Every statistic is re-implemented here from its definition.
"""
from __future__ import annotations
import json, sys, collections
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[3] / "silentfail_bench"  # paper/audits/<this>/recompute.py -> repo root
D = ROOT / "data"
OUT = Path(__file__).resolve().parent
RESULT = {}


def jsonl(p):
    return [json.loads(l) for l in Path(p).read_text().splitlines() if l.strip()]


# ---------------------------------------------------------------- primitives
def auroc(y, s):
    """Probability a random positive outranks a random negative; ties count 1/2. O(n_pos*n_neg), exact."""
    y = np.asarray(y).astype(int); s = np.asarray(s, float)
    pos, neg = s[y == 1], s[y == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    gt = (pos[:, None] > neg[None, :]).sum(); eq = (pos[:, None] == neg[None, :]).sum()
    return float((gt + 0.5 * eq) / (len(pos) * len(neg)))


def ecdf_rank(train_vals, x):
    """Empirical CDF of the training values evaluated at x: share of training values <= x (ties count)."""
    t = np.asarray(train_vals, float)
    return float((t <= x).sum() / len(t))


def panel_u(raw: dict, train: dict):
    """Sample SD (ddof=1) across critics of ECDF-rank-normalised scores."""
    rn = [ecdf_rank(train[k], raw[k]) for k in train]
    return float(np.std(rn, ddof=1))


def wilson(k, n, z=1.96):
    p = k / n; d = 1 + z * z / n; c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (c - h, c + h)


def cluster_boot(stat_fn, groups, n_boot, seed):
    """Repository-clustered bootstrap: resample clusters with replacement, all rows of a drawn cluster come along."""
    g = np.asarray(groups); uniq = np.unique(g)
    idx = {k: np.flatnonzero(g == k) for k in uniq}
    rng = np.random.default_rng(seed); vals = []
    for _ in range(n_boot):
        ii = np.concatenate([idx[k] for k in rng.choice(uniq, size=len(uniq), replace=True)])
        v = stat_fn(ii)
        if v is not None and np.isfinite(v):
            vals.append(v)
    vals = np.asarray(vals)
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5)), len(vals)


# ------------------------------------------------- items 1 & 2: committed population, rates, AUROC(u), AUROC(-c)
def load_bench(tag):
    if tag == "verified":
        rows = jsonl(D / "swebench_pilot50/pilot_free_scored.jsonl")
    else:
        rows = jsonl(D / "replication_multilingual/replication_scored.jsonl")
    usable = [r for r in rows if not r["excluded"]]
    pop = [r for r in usable if r["visible_pass"] and (r.get("final_solution") or "").strip()]
    return rows, usable, pop


def items_1_2():
    out = {}
    for tag in ("verified", "multilingual"):
        rows, usable, pop = load_bench(tag)
        y = np.array([1 if r["g"] == 0 else 0 for r in pop])
        c = np.array([r["c"] for r in pop], float)
        members = list(pop[0]["audit_scores"].keys())
        # rank norms fit on TRAINING-SPLIT rows (not excluded), per critic -- the definition the stored u uses
        train_usable = {m: [r["audit_scores"][m] for r in usable if r["split"] == "train"] for m in members}
        # alternative: fit on the committed population only (what the task prompt literally describes)
        train_pop = {m: [r["audit_scores"][m] for r in pop] for m in members}
        u_train = np.array([panel_u(r["audit_scores"], train_usable) for r in pop])
        u_pop = np.array([panel_u(r["audit_scores"], train_pop) for r in pop])
        u_stored = np.array([r["u"] for r in pop], float)
        grp = [r["repo"] for r in pop]
        lo, hi = wilson(int(y.sum()), len(y))
        ci_u = cluster_boot(lambda ii: auroc(y[ii], u_train[ii]) if 0 < y[ii].sum() < len(ii) else None, grp, 4000, 101)
        out[tag] = {
            "n_rows_file": len(rows), "n_usable_not_excluded": len(usable),
            "n_committed_population": len(pop), "silent_failures": int(y.sum()),
            "silent_failure_rate": float(y.mean()), "wilson95": [lo, hi],
            "n_repos": len(set(grp)),
            "n_train_rows_for_rank_norm": len(train_usable[members[0]]),
            "AUROC_u_recomputed_trainfit": auroc(y, u_train),
            "AUROC_u_recomputed_popfit": auroc(y, u_pop),
            "AUROC_u_from_stored_column": auroc(y, u_stored),
            "max_abs_diff_recomputed_vs_stored_u": float(np.max(np.abs(u_train - u_stored))),
            "AUROC_u_CI95_repo_clustered_mine": list(ci_u[:2]),
            "AUROC_negc": auroc(y, -c),
            "c_sel_id": pop[0]["c_sel_id"],
            "corr_c_u": float(np.corrcoef(c, u_train)[0, 1]),
        }
        # population variants, for the record
        pop_vis_only = [r for r in usable if r["visible_pass"]]
        out[tag]["n_visible_pass_not_excluded_ignoring_emptiness"] = len(pop_vis_only)
        out[tag]["n_empty_patch_among_them"] = sum(1 for r in pop_vis_only if not (r.get("final_solution") or "").strip())
    RESULT["item1_2"] = out
    return out


# ------------------------------------------------- item 3: pool (random arm) vs committed, same instances, Verified
def item_3():
    rows = {r["instance_id"]: r for r in jsonl(D / "swebench_pilot50/pilot_free_scored.jsonl")}
    picks = {p["instance_id"]: p for p in jsonl(D / "random_arm/verified_picks.jsonl")}
    rep = json.load(open(D / "gold_modal/random_arm_verified/report.json"))
    gold = {i: 1 for i in rep["resolved_ids"]}; gold.update({i: 0 for i in rep["unresolved_ids"]})
    ids = [i for i in picks if i in gold and i in rows and not rows[i]["excluded"] and rows[i]["visible_pass"]]
    y_r = np.array([1 if gold[i] == 0 else 0 for i in ids])
    y_c = np.array([1 if rows[i]["g"] == 0 else 0 for i in ids])
    c_r = np.array([rows[i]["candidate_scores"][picks[i]["candidate"]] for i in ids], float)
    c_c = np.array([rows[i]["c"] for i in ids], float)
    # sanity: committed index in picks equals the row's selected index, and c equals candidate_scores[selected]
    mism = [i for i in ids if picks[i]["committed"] != rows[i]["selected_candidate_index"]]
    c_mism = [i for i in ids if abs(rows[i]["candidate_scores"][rows[i]["selected_candidate_index"]] - rows[i]["c"]) > 1e-9]
    grp = [rows[i]["repo"] for i in ids]
    ci_r = cluster_boot(lambda ii: auroc(y_r[ii], -c_r[ii]) if 0 < y_r[ii].sum() < len(ii) else None, grp, 4000, 102)
    out = {"n_paired": len(ids), "picks_total": len(picks), "gold_errors": len(rep["error_ids"]),
           "random_failures": int(y_r.sum()), "committed_failures": int(y_c.sum()),
           "failure_rate_random": float(y_r.mean()), "failure_rate_committed": float(y_c.mean()),
           "AUROC_negc_random_arm": auroc(y_r, -c_r), "AUROC_negc_random_CI95_mine": list(ci_r[:2]),
           "AUROC_negc_committed_same_rows": auroc(y_c, -c_c),
           "committed_index_mismatches": mism, "c_vs_candidate_scores_mismatches": c_mism}
    RESULT["item3"] = out
    return out


# ------------------------------------------------- item 4: pooled registered endpoint, four estimators
def standardize_fit(X):
    mu = X.mean(axis=0); sg = X.std(axis=0); sg = np.where(sg < 1e-12, 1.0, sg)
    return mu, sg


def logistic_ridge(X, y, l2=1e-2, max_iter=200):
    """Own implementation: L2-penalised logistic regression (penalty on intercept too, as the project does),
    Newton steps with step-halving. Returns (coef, intercept)."""
    n, p = X.shape
    Xa = np.hstack([np.ones((n, 1)), X]); b = np.zeros(p + 1); reg = np.full(p + 1, l2)

    def nll(bb):
        z = np.clip(Xa @ bb, -50, 50); ph = 1 / (1 + np.exp(-z)); eps = 1e-12
        return -np.sum(y * np.log(ph + eps) + (1 - y) * np.log(1 - ph + eps)) + 0.5 * np.sum(reg * bb * bb)

    cur = nll(b)
    for _ in range(max_iter):
        z = np.clip(Xa @ b, -50, 50); ph = 1 / (1 + np.exp(-z)); W = ph * (1 - ph) + 1e-6
        grad = Xa.T @ (y - ph) - reg * b
        H = Xa.T @ (Xa * W[:, None]) + np.diag(reg)
        delta = np.linalg.solve(H, grad)
        sc = np.max(np.abs(delta))
        if sc > 10.0:
            delta *= 10.0 / sc
        step = 1.0; moved = False
        for _ in range(12):
            cand = b + step * delta; v = nll(cand)
            if np.isfinite(v) and v <= cur + 1e-9:
                b, cur, moved = cand, v, True; break
            step *= 0.5
        if not moved or np.linalg.norm(step * delta) < 1e-8:
            break
    return b[1:], float(b[0])


def oof_logistic(X, y, folds):
    p = np.full(len(y), np.nan)
    for te in folds:
        tr = ~te
        if y[tr].sum() < 2 or (1 - y[tr]).sum() < 2:
            continue
        mu, sg = standardize_fit(X[tr]); co, b0 = logistic_ridge((X[tr] - mu) / sg, y[tr].astype(float))
        p[te] = 1 / (1 + np.exp(-np.clip(((X[te] - mu) / sg) @ co + b0, -30, 30)))
    return p


def oof_fixed(X, y, folds):
    p = np.full(len(y), np.nan)
    for te in folds:
        mu, sg = standardize_fit(X[~te]); p[te] = ((X[te] - mu) / sg).sum(axis=1)
    return p


def loro_folds(grp):
    return [(grp == g) for g in np.unique(grp)]


def kfold_clustered(grp, k=5, seed=0):
    u = np.unique(grp); rng = np.random.default_rng(seed); u = u[rng.permutation(len(u))]
    return [np.isin(grp, u[i::k]) for i in range(k)]


def fold_avg_delta(Xc, Xcu, y, grp):
    num = den = 0.0; used = 0
    for g in np.unique(grp):
        te = grp == g; tr = ~te
        if y[tr].sum() < 2 or (1 - y[tr]).sum() < 2 or not (0 < y[te].sum() < te.sum()):
            continue
        pc = oof_logistic(Xc, y, [te]); pcu = oof_logistic(Xcu, y, [te])
        w = te.sum(); num += w * (auroc(y[te], pcu[te]) - auroc(y[te], pc[te])); den += w; used += 1
    return (num / den if den else float("nan")), used, den


def endpoint_block(pop, label, n_boot=4000, seed=200):
    y = np.array([1 if r["g"] == 0 else 0 for r in pop]); grp = np.array([r["_grp"] for r in pop])
    c = np.array([r["c"] for r in pop], float); u = np.array([r["u"] for r in pop], float)
    out = {"label": label, "n": len(y), "positives": int(y.sum()), "n_repos": int(len(np.unique(grp))),
           "raw_AUROC_negc": auroc(y, -c), "raw_AUROC_u": auroc(y, u)}
    Xc, Xcu = c[:, None], np.column_stack([c, u])

    # estimators 1 & 2: pooled-OOF logistic, LORO and 5-fold clustered
    for name, folds in (("logistic_LORO", loro_folds(grp)), ("logistic_5fold_clustered", kfold_clustered(grp))):
        pc, pcu = oof_logistic(Xc, y, folds), oof_logistic(Xcu, y, folds); ok = ~np.isnan(pc) & ~np.isnan(pcu)
        yy, pcc, pcuu, gg = y[ok], pc[ok], pcu[ok], grp[ok]
        point = auroc(yy, pcuu) - auroc(yy, pcc)
        lo, hi, nb = cluster_boot(lambda ii: (auroc(yy[ii], pcuu[ii]) - auroc(yy[ii], pcc[ii])) if 0 < yy[ii].sum() < len(ii) else None, gg, n_boot, seed)
        out[name] = {"AUROC_Mc": auroc(yy, pcc), "AUROC_Mcu": auroc(yy, pcuu), "delta": point, "CI95_mine": [lo, hi], "rows_used": int(ok.sum()), "boot_valid": nb}
        seed += 1
    # estimator 3: fixed unit weights, no fitting, LORO standardizer
    pc, pcu = oof_fixed((-c)[:, None], y, loro_folds(grp)), oof_fixed(np.column_stack([-c, u]), y, loro_folds(grp))
    point = auroc(y, pcu) - auroc(y, pc)
    lo, hi, nb = cluster_boot(lambda ii: (auroc(y[ii], pcu[ii]) - auroc(y[ii], pc[ii])) if 0 < y[ii].sum() < len(ii) else None, grp, n_boot, seed)
    # second seed, to show the seed-to-seed spread of the upper edge
    lo2, hi2, _ = cluster_boot(lambda ii: (auroc(y[ii], pcu[ii]) - auroc(y[ii], pc[ii])) if 0 < y[ii].sum() < len(ii) else None, grp, n_boot, seed + 50)
    lo3, hi3, _ = cluster_boot(lambda ii: (auroc(y[ii], pcu[ii]) - auroc(y[ii], pc[ii])) if 0 < y[ii].sum() < len(ii) else None, grp, n_boot, seed + 51)
    out["fixed_unit_weights"] = {"AUROC_negc": auroc(y, pc), "AUROC_negc_plus_u": auroc(y, pcu), "delta": point,
                                 "CI95_mine_seedA": [lo, hi], "CI95_mine_seedB": [lo2, hi2], "CI95_mine_seedC": [lo3, hi3], "boot_valid": nb}
    # estimator 4: fold-averaged LORO
    fa, used, covered = fold_avg_delta(Xc, Xcu, y, grp)
    out["fold_averaged_LORO"] = {"delta": fa, "folds_contributing": used, "rows_covered": int(covered)}
    return out


def item_4():
    V = [r for r in jsonl(D / "swebench_pilot50/pilot_free_scored.jsonl") if not r["excluded"] and r["visible_pass"]]
    M = [r for r in jsonl(D / "replication_multilingual/replication_scored.jsonl") if not r["excluded"] and r["visible_pass"]]
    for r in V: r["_grp"] = "verified:" + r["repo"]
    for r in M: r["_grp"] = "multilingual:" + r["repo"]
    out = {"verified": endpoint_block(V, "verified", seed=300), "multilingual": endpoint_block(M, "multilingual", seed=400),
           "pooled": endpoint_block(V + M, "pooled", seed=500)}
    RESULT["item4"] = out
    return out


# ------------------------------------------------- items 5 & 6: agent axis (Amendments 13 / 14)
PANEL = ["ray2333_grm_gemma2_2b_rewardmodel_ft", "ray2333_grm_llama3_2_3b_rewardmodel_ft",
         "skywork_skywork_reward_v2_llama_3_2_1b", "skywork_skywork_reward_v2_qwen3_1_7b"]
SEL4B = "skywork_skywork_reward_v2_qwen3_4b"


def load_axis():
    A = D / "agent_axis_deepseek"
    inst = [l.strip() for l in (ROOT / "configs/agent_axis_instances_multilingual.txt").read_text().splitlines() if l.strip() and not l.startswith("#")]
    bundle = {(r["instance_id"], int(r["candidate"])): r for r in jsonl(A / "rescoring_bundle.jsonl")}
    assert set(bundle) == {(i, c) for i in inst for c in range(4)}, "bundle is not the 114x4 set"
    gold = {}  # 1 = resolved, 0 = fail
    empties = {k for k, r in bundle.items() if not (r.get("final_solution") or "").strip()}
    for k in empties: gold[k] = 0
    n_gold_rows = 0; errs = []
    for c in range(4):
        for d in jsonl(D / f"gold_modal/a7_agent_axis_cand{c}/gold_extra.jsonl"):
            k = (d["instance_id"], int(d["candidate"])); assert k[1] == c
            assert k not in gold or gold[k] == int(d["g"])
            gold[k] = int(d["g"]); n_gold_rows += 1
        errs += json.load(open(D / f"gold_modal/a7_agent_axis_cand{c}/report.json")).get("error_ids") or []
    scores = {m: json.load(open(A / f"modal/prefix4k/scores_gpu_{m}.json"))["scores"] for m in PANEL + [SEL4B]}
    judges = {}
    for J, f in (("gpt-5", "judge_gpt5.jsonl"), ("gpt-6-sol", "judge_gpt-6-sol.jsonl"), ("gpt-6-luna", "judge_gpt-6-luna.jsonl")):
        judges[J] = {}
        for d in jsonl(A / f):
            if d.get("judge") == J and d.get("cond") == "J1" and d.get("p_resolved") is not None:
                judges[J][(d["instance_id"], int(d["candidate"]))] = float(d["p_resolved"])
    picks = {d["instance_id"]: int(d["candidate"]) for d in jsonl(A / "gold_inputs/random_picks.jsonl")}
    return inst, bundle, gold, empties, n_gold_rows, errs, scores, judges, picks


def committed_arm(inst, bundle, gold, scores, selected, detect, label):
    pop = []
    for i in inst:
        c = selected[i]; k = (i, c); r = bundle[k]; sk = f"{i}#{c}"
        vals = [scores[m].get(sk, {}).get("score", np.nan) for m in PANEL]
        if r.get("visible_pass") and (r.get("final_solution") or "").strip() and k in gold and k in detect and np.isfinite(detect[k]) and all(np.isfinite(vals)):
            pop.append(k)
    train = {m: [scores[m][f"{i}#{c}"]["score"] for i, c in pop if bundle[(i, c)]["split"] == "train"] for m in PANEL}
    u = np.array([panel_u({m: scores[m][f"{i}#{c}"]["score"] for m in PANEL}, train) for i, c in pop])
    y = np.array([1 - gold[k] for k in pop]); s = np.array([detect[k] for k in pop], float)
    grp = [bundle[k]["repo"] for k in pop]
    return {"label": label, "n": len(pop), "failures": int(y.sum()), "n_repos": len(set(grp)),
            "raw_AUROC_detect": auroc(y, s), "raw_AUROC_u": auroc(y, u),
            "n_train_rows_for_rank_norm": len(train[PANEL[0]])}


def items_5_6():
    inst, bundle, gold, empties, n_gold_rows, errs, scores, judges, picks = load_axis()
    out = {"registered_instances": len(inst), "slots": len(bundle), "empty_patches": len(empties),
           "gold_extra_rows": n_gold_rows, "harness_error_ids": errs,
           "slots_without_gold": sorted(set(bundle) - set(gold))}
    # A7-P1 DeepSeek unconditional failure rate
    graded = [k for k in bundle if k not in empties and k in gold]
    gfail = sum(1 for k in graded if gold[k] == 0)
    out["deepseek"] = {"attempts": len(bundle), "empty": len(empties), "graded_non_empty": len(graded), "graded_failures": gfail,
                       "ungraded_non_empty": len(bundle) - len(empties) - len(graded),
                       "unconditional_failure_rate": (len(empties) + gfail) / len(bundle),
                       "excluded_rows_in_bundle": sorted(k for k, r in bundle.items() if r.get("excluded")),
                       "exit_statuses": dict(collections.Counter(r.get("exit_status") for r in bundle.values()))}
    # gpt-5-mini comparator, 100 steps, k=1, set-union failure counting
    preds = {r["instance_id"]: r for r in jsonl(D / "agent_axis_gpt5mini_100/predictions_merged.jsonl")}
    rep = json.load(open(D / "gold_modal/a13_gpt5mini_100_merged/report.json"))
    res, unres, agent, infra = set(), set(), set(), set()
    AGENT_CAUSED = ("patch apply failed", "KeyError: 'tests_status'")
    for d in rep["details"]:
        if d.get("model_name_or_path", "").startswith("ORACLE") or d["instance_id"] not in inst:
            continue
        err = d.get("error")
        if err is None:
            (res if d.get("resolved") else unres).add(d["instance_id"])
        elif any(err.startswith(a) for a in AGENT_CAUSED):
            agent.add(d["instance_id"])
        else:
            infra.add(d["instance_id"])
    empty = {i for i in inst if not (preds.get(i, {}).get("model_patch") or "").strip()}
    fail_b = (unres | empty | agent) - infra
    out["gpt5mini_100"] = {"instances": len(inst), "pred_rows": len(preds), "resolved": len(res), "unresolved": len(unres),
                           "empty": len(empty), "empty_also_in_unresolved": len(empty & unres), "agent_crash": len(agent), "infra": len(infra),
                           "ungraded_non_empty": len(set(inst) - empty - res - unres - agent - infra),
                           "failure_set_union": len(fail_b), "failure_rate_B": len(fail_b) / (len(inst) - len(infra)),
                           "failure_rate_naive_sum_empty_plus_unresolved": (len(empty) + len(unres)) / len(inst)}
    # pool AUROCs over graded non-empty attempts
    pool = sorted(k for k in bundle if k not in empties and k in gold)
    y = np.array([1 - gold[k] for k in pool])
    out["pool"] = {"n": len(pool), "failures": int(y.sum()), "reward_models": {}, "judges": {}}
    for m in PANEL + [SEL4B]:
        s = np.array([scores[m][f"{i}#{c}"]["score"] for i, c in pool], float)
        out["pool"]["reward_models"][m] = auroc(y, -s)
    rms = list(out["pool"]["reward_models"].values())
    out["pool"]["reward_model_range"] = [min(rms), max(rms)]
    for J, jd in judges.items():
        pk = [k for k in pool if k in jd]
        out["pool"]["judges"][J] = {"n_scored": len(jd), "n_in_pool": len(pk),
                                    "AUROC_detect": auroc([1 - gold[k] for k in pk], [100 - jd[k] for k in pk])}
    # committed arms
    sel4b, det4b = {}, {}
    for i in inst:
        ch = []
        for c in range(4):
            sc = scores[SEL4B].get(f"{i}#{c}", {}).get("score", float("-inf")); det4b[(i, c)] = -float(sc)
            ch.append((float(sc) if bundle[(i, c)].get("text") else float("-inf"), c))
        sel4b[i] = max(ch, key=lambda p: p[0])[1]   # first candidate wins a tie (max keeps first maximal)
    out["arms"] = {"S_4b": committed_arm(inst, bundle, gold, scores, sel4b, det4b, "S_4b")}
    rand_fail = np.array([1 - gold[(i, picks[i])] for i in inst], float)
    out["random_arm_failure_rate"] = float(rand_fail.mean())
    out["random_arm_failures"] = int(rand_fail.sum())
    for J, jd in judges.items():
        selj = {}; ties = 0
        for i in inst:
            vals = [jd[(i, c)] for c in range(4)]; best = max(vals)
            if vals.count(best) > 1: ties += 1
            selj[i] = vals.index(best)   # first maximal, same as Python max over range
        detj = {k: 100.0 - v for k, v in jd.items()}
        arm = committed_arm(inst, bundle, gold, scores, selj, detj, f"S_judge_{J}")
        arm["argmax_ties_first_wins"] = ties
        cf = np.array([1 - gold[(i, selj[i])] for i in inst], float)
        arm["committed_failure_rate_unconditional_114"] = float(cf.mean()); arm["committed_failures_unconditional_114"] = int(cf.sum())
        out["arms"][f"S_judge_{J}"] = arm
    RESULT["item5_6"] = out
    return out


if __name__ == "__main__":
    np.set_printoptions(precision=4)
    items_1_2(); item_3(); item_4(); items_5_6()
    (OUT / "recompute_results.json").write_text(json.dumps(RESULT, indent=1, default=str))
    print(json.dumps(RESULT, indent=1, default=str))
