#!/usr/bin/env python3
"""S-P1's second registered condition, which the paper reported only in part.

Amendment 5 registered S-P1 as a CONJUNCTION: the selector's pool AUROC(-c)
above 0.65, AND "the committed-arm failure rate is below the random arm's
(a selector with signal picks better)". The paper reported the first half
(0.712 / 0.706) and declared S-P1 passed. The second half was never computed.

This computes it, paired within instance and repository-clustered, using the
same label assembly as prereg_readout_amendment5.py. No new gold, no API calls.
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT / "src"))


def boot_paired(diff, grp, n=4000, seed=17):
    """Repository-clustered bootstrap of a paired mean difference in failure rate."""
    g = np.asarray(grp); u = np.unique(g); idx = {k: np.flatnonzero(g == k) for k in u}
    rng = np.random.default_rng(seed); v = []
    for _ in range(n):
        ii = np.concatenate([idx[k] for k in rng.choice(u, size=len(u), replace=True)])
        v.append(float(np.mean(diff[ii])))
    return [round(float(np.percentile(v, 2.5)), 3), round(float(np.percentile(v, 97.5)), 3)]


def run(bench, rows_f, judge_cache, gold_extra_files, random_picks):
    rows = {r["instance_id"]: r for r in map(json.loads, Path(rows_f).read_text().splitlines()) if r}
    J = {}
    for d in map(json.loads, Path(judge_cache).read_text().splitlines()):
        if d.get("p_resolved") is not None: J[(d["instance_id"], d["candidate"])] = d["p_resolved"]
    gold = {}
    for i, r in rows.items():
        if not r["excluded"]: gold[(i, r["selected_candidate_index"])] = r["g"]
    for f in gold_extra_files:
        if Path(f).exists():
            for d in map(json.loads, Path(f).read_text().splitlines()):
                if d.get("g") in (0, 1): gold[(d["instance_id"], d["candidate"])] = d["g"]
    picks = {d["instance_id"]: d["candidate"] for d in map(json.loads, Path(random_picks).read_text().splitlines())}
    rep_f = Path(random_picks).parent.parent / "gold_modal" / f"random_arm_{bench}" / "report.json"
    rp = json.loads(rep_f.read_text())
    rand_gold = {}
    for i in rp["resolved_ids"]:
        if i in picks: rand_gold[i] = 1; gold[(i, picks[i])] = 1
    for i in rp["unresolved_ids"]:
        if i in picks: rand_gold[i] = 0; gold[(i, picks[i])] = 0

    # gpt-5's committed choice per instance (argmax of p_resolved over non-empty candidates)
    from itertools import chain
    newc = {}
    bundle = ROOT / ("data/swebench_pilot50/rescoring_bundle.jsonl" if bench == "verified"
                     else "data/replication_multilingual/rescoring_bundle.jsonl")
    cands = {(c["instance_id"], c["candidate"]): c for c in map(json.loads, bundle.read_text().splitlines()) if c}
    for i in rows:
        avail = [(c, J[(i, c)]) for c in range(4) if (i, c) in J and cands.get((i, c), {}).get("final_solution", "").strip()]
        if avail: newc[i] = max(avail, key=lambda x: x[1])[0]

    # paired: instances where BOTH the gpt-5 commit and the random pick have gold
    # per-candidate flag: the registered population is the COMMITTED trajectory, which here is
    # gpt-5's pick, not the 4B's. Using rows[i]["visible_pass"] was the defect in Amendment 5.
    pairs = [i for i in newc if (i, newc[i]) in gold and i in rand_gold and cands[(i, newc[i])].get("visible_pass")]
    out = {"benchmark": bench, "paired_instances": len(pairs)}
    if len(pairs) < 10:
        out["verdict"] = f"only {len(pairs)} paired instances; below the inference floor, descriptive only"
        return out
    f_comm = np.array([1 - gold[(i, newc[i])] for i in pairs], float)
    f_rand = np.array([1 - rand_gold[i] for i in pairs], float)
    grp = np.array([rows[i]["repo"] for i in pairs])
    diff = f_comm - f_rand
    out["committed_failure_rate"] = round(float(f_comm.mean()), 3)
    out["random_failure_rate"] = round(float(f_rand.mean()), 3)
    out["paired_difference"] = round(float(diff.mean()), 3)
    out["paired_difference_CI"] = boot_paired(diff, grp)
    # discordant pairs, the only ones that carry information
    out["discordant"] = {"committed_fails_random_passes": int(((f_comm == 1) & (f_rand == 0)).sum()),
                         "committed_passes_random_fails": int(((f_comm == 0) & (f_rand == 1)).sum())}
    # how often did gpt-5's pick differ from the random pick at all?
    out["picks_differ"] = int(sum(1 for i in pairs if newc[i] != picks[i]))
    out["identical_picks_structural_zeros"] = len(pairs) - out["picks_differ"]
    dif = [i for i in pairs if newc[i] != picks[i]]
    if len(dif) >= 10:
        fc2 = np.array([1 - gold[(i, newc[i])] for i in dif], float); fr2 = np.array([1 - rand_gold[i] for i in dif], float)
        out["informative_subset"] = {"n": len(dif), "committed_rate": round(float(fc2.mean()), 3),
                                     "random_rate": round(float(fr2.mean()), 3),
                                     "difference": round(float((fc2 - fr2).mean()), 3),
                                     "CI95": boot_paired(fc2 - fr2, np.array([rows[i]["repo"] for i in dif]))}
    out["S_P1b_committed_below_random"] = bool(out["paired_difference"] < 0)
    out["S_P1b_strictly_below_with_CI"] = bool(out["paired_difference_CI"][1] < 0)
    return out


V = run("verified", ROOT/"data/swebench_pilot50/pilot_free_scored.jsonl", ROOT/"data/selector_axis/judge_verified.jsonl",
        [ROOT/"data/swebench_pilot50/l4/out/gpu_scores_16k/gold_extra_merged.jsonl", ROOT/"data/selector_axis/gold_extra_verified.jsonl"],
        ROOT/"data/random_arm/verified_picks.jsonl")
M = run("multilingual", ROOT/"data/replication_multilingual/replication_scored.jsonl", ROOT/"data/selector_axis/judge_multilingual.jsonl",
        [ROOT/"data/selector_axis/gold_extra_multilingual.jsonl"], ROOT/"data/random_arm/multilingual_picks.jsonl")
rep = {"analysis": "S-P1's second registered condition: is the gpt-5 committed arm's failure rate below the random arm's?",
       "registered_text": "S-P1 (the regime is entered): for S_judge, AUROC(-c) in the candidate pool > 0.65, AND the committed-arm failure rate is below the random arm's.",
       "note": "The paper reported only the pool-AUROC half and declared S-P1 passed. This is the half that was missing.",
       "verified": V, "multilingual": M}
(ROOT / "data/readout_sp1b.json").write_text(json.dumps(rep, indent=2) + "\n")
print("=== S-P1's second condition: committed vs random failure rate (gpt-5 selector) ===")
for tag, r in (("Verified", V), ("Multilingual", M)):
    print(f"\n{tag}: {r['paired_instances']} paired instances")
    if "verdict" in r: print("  ", r["verdict"]); continue
    print(f"   committed failure rate {r['committed_failure_rate']:.3f} vs random {r['random_failure_rate']:.3f}")
    print(f"   paired difference {r['paired_difference']:+.3f} {r['paired_difference_CI']}  (negative = selector picks better)")
    print(f"   discordant pairs: committed fails/random passes {r['discordant']['committed_fails_random_passes']}, "
          f"committed passes/random fails {r['discordant']['committed_passes_random_fails']}")
    print(f"   gpt-5's pick differed from the random pick on {r['picks_differ']} of {r['paired_instances']}")
    print(f"   S-P1b (committed below random): {'MET' if r['S_P1b_committed_below_random'] else 'NOT MET'}"
          f" | strictly below with CI: {'yes' if r['S_P1b_strictly_below_with_CI'] else 'no'}")
print("\nwrote data/readout_sp1b.json")
