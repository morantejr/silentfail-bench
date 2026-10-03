#!/usr/bin/env python3
"""Mechanism numbers behind the paper's 'what disagreement measures' sentences (post hoc, 3 October 2026).

On the committed population and the batch-1 matrices (Verified 16k; Multilingual prefix4k), for the registered
four-critic panel and the seven-critic zoo:
  * two-way additive sum-of-squares shares of the RANK-NORMALISED score matrix (the scale the published u is built
    on), with the consistency ICC(C,1) [Shrout & Fleiss ICC(3,1)];
  * the same split on the raw scores, for contrast (critic location/scale dominates there and u never sees it);
  * the E2 pairwise error correlation (readout_pi_batch1 definition), its independence floor -- the value the same
    statistic takes when every critic's rank score is independent of the label and of every other critic -- and the
    band under 200 label permutations (seed 0), which keeps between-critic dependence and removes label dependence.
Argument-free; regenerates identically. Writes data/readout_critic_demeaning.json. Derived from the audit in
for_claude/algorithmic_to_7 and Cursor's review of it.
"""
from __future__ import annotations
import itertools, json, sys
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT / "src"))
from silentfail.scoring import fit_rank_norms  # noqa: E402

BENCH = {"verified": ("data/swebench_pilot50/pilot_free_scored.jsonl", "data/swebench_pilot50/l4/out/gpu_scores_16k"),
         "multilingual": ("data/replication_multilingual/replication_scored.jsonl", "data/replication_multilingual/modal/prefix4k")}
PANEL = ["ray2333_grm_gemma2_2b_rewardmodel_ft", "ray2333_grm_llama3_2_3b_rewardmodel_ft",
         "skywork_skywork_reward_v2_llama_3_2_1b", "skywork_skywork_reward_v2_qwen3_1_7b"]
N_PERM, SEED = 200, 0


def r3(x): return round(float(x), 3)


def ss_split(R):
    n, J = R.shape; gm = R.mean(); cm = R.mean(axis=0); tm = R.mean(axis=1)
    ss_c = n * ((cm - gm) ** 2).sum(); ss_t = J * ((tm - gm) ** 2).sum()
    ss_r = ((R - cm[None, :] - tm[:, None] + gm) ** 2).sum(); tot = ((R - gm) ** 2).sum()
    ms_t, ms_r = ss_t / (n - 1), ss_r / ((n - 1) * (J - 1))
    return {"share_critic": r3(ss_c / tot), "share_trajectory": r3(ss_t / tot), "share_residual": r3(ss_r / tot),
            "ICC_C1": r3((ms_t - ms_r) / (ms_t + (J - 1) * ms_r)), "n": n, "J": J}


def main() -> int:
    out = {"analysis": "critic demeaning / what disagreement measures", "post_hoc": True, "date": "2026-10-03", "benchmarks": {}}
    for tag, (rowf, sdir) in BENCH.items():
        rows = {}
        for l in (ROOT / rowf).read_text().splitlines():
            if l.strip(): r = json.loads(l); rows[r["instance_id"]] = r
        pop = [i for i in rows if not rows[i]["excluded"] and rows[i].get("visible_pass")]
        S = {f.stem.replace("scores_gpu_", ""): json.loads(f.read_text())["scores"] for f in sorted((ROOT / sdir).glob("scores_gpu_*.json"))}
        crit = sorted(S); key = lambda i: f"{i}#{rows[i]['selected_candidate_index']}"
        R = np.array([[S[m][key(i)]["score"] for m in crit] for i in pop], float)
        if np.isnan(R).any(): print("ABORT: missing cells"); return 2
        y = np.array([1 if rows[i]["g"] == 0 else 0 for i in pop]); train = np.array([rows[i]["split"] == "train" for i in pop])
        rn = {}
        for k, m in enumerate(crit):
            fit = fit_rank_norms({m: list(R[train, k])})[m]; rn[m] = np.array([fit.transform(v) for v in R[:, k]])
        b = {"n": len(pop), "failures": int(y.sum()), "repositories": len({rows[i]["repo"] for i in pop}), "matrix": sdir, "panels": {}}
        for pname, members in (("registered_panel4", PANEL), ("zoo7", crit)):
            cols = [crit.index(m) for m in members]; RN = np.column_stack([rn[m] for m in members])
            b["panels"][pname] = {"members": members, "rank_scale": ss_split(RN), "raw_scale": ss_split(R[:, cols])}
        # E2 error correlation over four-member subsets of the zoo, its independence floor, and the permutation band
        def band(yy):
            err = {m: rn[m] - (1 - yy) for m in crit}; v = []
            for combo in itertools.combinations(crit, 4):
                v.append(np.mean([np.corrcoef(err[a], err[c])[0, 1] for a, c in itertools.combinations(combo, 2)]))
            return np.array(v)
        obs = band(y); vy = float(np.var(y, ddof=1)); vr = {m: float(np.var(rn[m], ddof=1)) for m in crit}
        floor = np.array([np.mean([vy / np.sqrt((vr[a] + vy) * (vr[c] + vy)) for a, c in itertools.combinations(combo, 2)]) for combo in itertools.combinations(crit, 4)])
        rng = np.random.default_rng(SEED); pmin, pmax = [], []
        for _ in range(N_PERM):
            pb = band(rng.permutation(y)); pmin.append(pb.min()); pmax.append(pb.max())
        b["E2_error_correlation"] = {"definition": "rank_j(score) - (1 - y), rank norms fit on train rows; mean pairwise correlation over every four-member subset of the seven critics",
                                     "observed_range": [r3(obs.min()), r3(obs.max())],
                                     "independence_floor_range": [r3(floor.min()), r3(floor.max())],
                                     "label_permutation_range_mean": [r3(np.mean(pmin)), r3(np.mean(pmax))], "n_permutations": N_PERM, "seed": SEED,
                                     "reading": "the lower edge of the observed band meets the independence floor; the upper edge survives permuting the label, "
                                                "so it is dependence among the rank scores, which this statistic does not separate into shared error and shared signal"}
        out["benchmarks"][tag] = b
    (ROOT / "data/readout_critic_demeaning.json").write_text(json.dumps(out, indent=1, sort_keys=True) + "\n")
    for tag, b in out["benchmarks"].items():
        for pn, p in b["panels"].items():
            rs, ra = p["rank_scale"], p["raw_scale"]
            print(f"{tag}/{pn}: rank-scale shares critic {rs['share_critic']} / trajectory {rs['share_trajectory']} / residual {rs['share_residual']} ICC {rs['ICC_C1']} | raw-scale critic {ra['share_critic']}")
        e = b["E2_error_correlation"]; print(f"{tag}: E2 observed {e['observed_range']} floor {e['independence_floor_range']} permutation {e['label_permutation_range_mean']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
