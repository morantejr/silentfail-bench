#!/usr/bin/env python3
"""Four more PI questions, all from data on disk.
TB  did the agent verify its own work, and does that predict silent failure?
CAL are the LLM judges' stated probabilities calibrated, or only rank-useful?
E3  does the null deepen with selection pressure (best-of-2 vs best-of-4)?
XB  does anything transfer? Fit on one benchmark, test on the other.
"""
from __future__ import annotations
import json, re, sys, itertools
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT / "src"))
from silentfail.combine import _fit_standardizer, fit_logistic_irls  # noqa: E402
from silentfail.scoring import fit_rank_norms, apply_panel_u  # noqa: E402
from silentfail.stats import auroc_fast  # noqa: E402
TESTPAT = re.compile(r"pytest|python -m pytest|npm (?:run )?test|yarn test|go test|cargo test|mvn |gradle |bundle exec|phpunit|jest|vitest|tox|ctest|dotnet test", re.I)
BENCH = {"verified": ("data/swebench_pilot50/pilot_free_scored.jsonl", "data/swebench_pilot50/l4/out/gpu_scores_16k",
                      "data/swebench_pilot50/llm_judges/judgments.jsonl", "data/swebench_pilot50/llm_judges/judgments_gpt6.jsonl"),
         "multilingual": ("data/replication_multilingual/replication_scored.jsonl", "data/replication_multilingual/modal/prefix4k",
                          "data/replication_multilingual/llm_judges/judgments.jsonl", "data/replication_multilingual/llm_judges/judgments_gpt6.jsonl")}
PANEL = ["ray2333_grm_gemma2_2b_rewardmodel_ft","ray2333_grm_llama3_2_3b_rewardmodel_ft",
         "skywork_skywork_reward_v2_llama_3_2_1b","skywork_skywork_reward_v2_qwen3_1_7b"]

def boot(y, s, grp, n=4000, seed=1):
    g=np.asarray(grp); u=np.unique(g); idx={k:np.flatnonzero(g==k) for k in u}; rng=np.random.default_rng(seed); v=[]
    for _ in range(n):
        ii=np.concatenate([idx[k] for k in rng.choice(u,size=len(u),replace=True)])
        if 0<y[ii].sum()<len(ii): v.append(auroc_fast(y[ii],s[ii]))
    return [round(float(np.percentile(v,2.5)),3), round(float(np.percentile(v,97.5)),3)] if v else None

rep = {}
for tag,(rowf,scored,j5,j6) in BENCH.items():
    rows = {r["instance_id"]: r for r in map(json.loads, (ROOT/rowf).read_text().splitlines()) if r}
    pop = [i for i,r in rows.items() if not r["excluded"] and r.get("visible_pass")]
    y = np.array([1 if rows[i]["g"]==0 else 0 for i in pop]); grp=np.array([rows[i]["repo"] for i in pop])
    out = {"n": len(pop), "failures": int(y.sum())}
    # ---------- TB: did the agent verify? ----------
    ntest = np.array([len(TESTPAT.findall(rows[i]["trajectory_text"])) for i in pop], float)
    ran = (ntest > 0).astype(int)
    out["TB_agent_verification"] = {
        "frac_that_ran_any_test": round(float(ran.mean()),3),
        "failure_rate_if_ran_tests": round(float(y[ran==1].mean()),3) if (ran==1).any() else None,
        "failure_rate_if_never_tested": round(float(y[ran==0].mean()),3) if (ran==0).any() else None,
        "AUROC_never_tested_predicts_failure": round(auroc_fast(y, 1-ran),3),
        "AUROC_fewer_tests_predicts_failure": round(auroc_fast(y, -ntest),3),
        "CI": boot(y, -ntest, grp)}
    # ---------- CAL: judge calibration ----------
    J = {}
    for f in (j5, j6):
        p = ROOT/f
        if p.exists():
            for d in map(json.loads, p.read_text().splitlines()):
                if d.get("cond")=="J1" and d.get("p_resolved") is not None: J.setdefault(d["judge"],{})[d["instance_id"]] = d["p_resolved"]/100
    cal = {}
    for j, m in J.items():
        p = np.array([m.get(i, np.nan) for i in pop]); ok=~np.isnan(p)
        if ok.sum() < 10: continue
        pp, yy = p[ok], 1-y[ok]                       # yy = actually resolved
        brier = float(np.mean((pp-yy)**2))
        bins = np.clip((pp*5).astype(int), 0, 4); ece = 0.0
        for b in range(5):
            s = bins==b
            if s.any(): ece += s.mean()*abs(pp[s].mean()-yy[s].mean())
        cal[j] = {"brier": round(brier,3), "ECE": round(float(ece),3), "mean_stated_p_resolved": round(float(pp.mean()),3),
                  "actual_resolve_rate": round(float(yy.mean()),3)}
    out["CAL_judge_calibration"] = {"judges": cal, "note": "Brier: lower is better; 0.25 = always saying 50%. ECE: gap between stated and actual."}
    # ---------- E3: selection pressure ----------
    S = {f.stem.replace("scores_gpu_",""): json.loads(f.read_text())["scores"] for f in (ROOT/scored).glob("scores_gpu_*.json")}
    CSEL = "skywork_skywork_reward_v2_qwen3_4b"
    train = np.array([rows[i]["split"]=="train" for i in pop])
    e3 = {}
    rng = np.random.default_rng(0)
    for k in (2, 4):
        us, ys = [], []
        for idx, i in enumerate(pop):
            cs = rows[i]["candidate_scores"]
            avail = [j for j,v in enumerate(cs) if v == v and abs(v) != float("inf")]
            if len(avail) < k: continue
            sub = list(rng.choice(avail, size=k, replace=False)) if k < len(avail) else avail
            pick = max(sub, key=lambda j: cs[j])
            kk = f"{i}#{pick}"
            if not all(kk in S.get(m,{}) for m in PANEL): continue
            us.append({m: S[m][kk]["score"] for m in PANEL}); ys.append(y[idx])
        if len(ys) < 15: continue
        fits = fit_rank_norms({m: [d[m] for d,t in zip(us,[rows[i]["split"]=="train" for i in pop][:len(us)]) if t] or [d[m] for d in us] for m in PANEL})
        uu = np.array([apply_panel_u(d, fits)[1] for d in us]); yy = np.array(ys)
        e3[f"best_of_{k}"] = {"n": len(yy), "failures": int(yy.sum()), "AUROC_u": round(auroc_fast(yy, uu),3) if 0<yy.sum()<len(yy) else None}
    out["E3_selection_pressure"] = e3
    rep[tag] = out
# ---------- XB: cross-benchmark transfer ----------
def feats(tag):
    rowf = BENCH[tag][0]
    rows = [json.loads(l) for l in (ROOT/rowf).read_text().splitlines() if l.strip()]
    pop = [r for r in rows if not r["excluded"] and r["visible_pass"]]
    y = np.array([1 if r["g"]==0 else 0 for r in pop])
    add = lambda d: sum(1 for l in (d or "").splitlines() if l.startswith("+") and not l.startswith("+++"))
    X = np.column_stack([[add(r["final_solution"]) for r in pop], [r["step_count"] for r in pop],
                         [r["u"] for r in pop], [r["c"] for r in pop],
                         [len(TESTPAT.findall(r["trajectory_text"])) for r in pop]]).astype(float)
    return X, y
Xa, ya = feats("verified"); Xb, yb = feats("multilingual")
xb = {}
for name, cols in (("structure only", [0,1,4]), ("structure + c + u", [0,1,2,3,4]), ("u alone", [2])):
    for src, dst, sx, sy, dx, dy in (("verified->multilingual", None, Xa, ya, Xb, yb), ("multilingual->verified", None, Xb, yb, Xa, ya)):
        mu, sg = _fit_standardizer(sx[:, cols]); co, b = fit_logistic_irls((sx[:, cols]-mu)/sg, sy.astype(float))
        p = 1/(1+np.exp(-np.clip(((dx[:, cols]-mu)/sg)@co+b, -30, 30)))
        xb.setdefault(name, {})[src] = round(auroc_fast(dy, p), 3)
rep["XB_cross_benchmark_transfer"] = xb
(ROOT/"data/readout_pi_batch2.json").write_text(json.dumps(rep, indent=1))
for tag in BENCH:
    o = rep[tag]; print(f"\n=== {tag}: n={o['n']}, failures={o['failures']} ===")
    t = o["TB_agent_verification"]
    print(f"  TB  ran any test: {t['frac_that_ran_any_test']:.0%} | failure rate if tested {t['failure_rate_if_ran_tests']} vs never tested {t['failure_rate_if_never_tested']}")
    print(f"      'never ran a test' as a detector: AUROC {t['AUROC_never_tested_predicts_failure']} | fewer tests: {t['AUROC_fewer_tests_predicts_failure']} {t['CI']}")
    print(f"  CAL judge calibration (Brier / ECE / stated vs actual resolve rate):")
    for j, v in o["CAL_judge_calibration"]["judges"].items():
        print(f"      {j:12s} Brier {v['brier']:.3f}  ECE {v['ECE']:.3f}  says {v['mean_stated_p_resolved']:.2f} actual {v['actual_resolve_rate']:.2f}")
    print(f"  E3  {o['E3_selection_pressure']}")
print(f"\n=== XB cross-benchmark transfer (AUROC on the held-out benchmark) ===")
for k, v in rep["XB_cross_benchmark_transfer"].items(): print(f"  {k:20s} {v}")
