#!/usr/bin/env python3
"""POST HOC (9 Nov 2026; not pre-registered). Asked for by two symposium reviewers: if a single
strong judge ranks silent failures well on its own, does ADDING ITS SCORE to the selector's own
score give a gain on the committed population, under the paper's protocol?

Protocol as in the paper: committed population (not excluded, visible pass), positive class =
failure, selector baseline -c, leave-one-repository-out, repository-clustered bootstrap CI on the
paired AUROC difference (the registration's approximation: no refit inside draws).
Estimators: fixed unit weights (the Amendment 7a primary) and LORO logistic (secondary).
Judge score enters as -p_resolved (J1 condition), same direction as -c.
Writes data/readout_single_judge_incremental.json.
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT / "src"))
from silentfail.combine import _fit_standardizer, fit_logistic_irls, _sigmoid  # noqa: E402
from silentfail.stats import auroc_fast, cluster_bootstrap_delta_auroc  # noqa: E402

BENCH = {
    "verified":     ("data/swebench_pilot50/pilot_free_scored.jsonl",           "data/swebench_pilot50/llm_judges/judgments_gpt6.jsonl"),
    "multilingual": ("data/replication_multilingual/replication_scored.jsonl", "data/replication_multilingual/llm_judges/judgments_gpt6.jsonl"),
}
JUDGES = ("gpt-6-sol", "gpt-6-luna")

def load(tag):
    rowf, judf = BENCH[tag]
    rows = [json.loads(l) for l in (ROOT / rowf).read_text().splitlines() if l.strip()]
    pop = [r for r in rows if not r.get("excluded") and r.get("visible_pass")]
    J = {}
    for l in (ROOT / judf).read_text().splitlines():
        if not l.strip(): continue
        d = json.loads(l)
        if d.get("cond") == "J1" and d.get("p_resolved") is not None:
            J.setdefault(d["judge"], {})[d["instance_id"]] = float(d["p_resolved"])
    return pop, J

def loro_fixed(X, y, grp):
    oof = np.full(len(y), np.nan)
    for g in np.unique(grp):
        te = grp == g; mu, sg = _fit_standardizer(X[~te]); oof[te] = ((X[te] - mu) / sg).sum(axis=1)
    return oof

def loro_logistic(X, y, grp):
    oof = np.full(len(y), np.nan)
    for g in np.unique(grp):
        te = grp == g; mu, sg = _fit_standardizer(X[~te]); Xtr = (X[~te] - mu) / sg; Xte = (X[te] - mu) / sg
        coef, b0 = fit_logistic_irls(Xtr, y[~te].astype(float)); oof[te] = _sigmoid(Xte @ coef + b0)
    return oof

def ci_of(y, base, aug, grp):
    r = cluster_bootstrap_delta_auroc(y, base, aug, list(grp), n_boot=2000)
    lo = getattr(r, "ci_lo", None); hi = getattr(r, "ci_hi", None)
    if lo is None: lo, hi = r[3], r[4]          # DeltaTest(point, a_base, a_aug, ci_lo, ci_hi, ...)
    return [round(float(lo), 3), round(float(hi), 3)], r

def analyse(pop_by_tag, J_by_tag, tags, judge):
    y, sc, sj, grp = [], [], [], []
    for tag in tags:
        for r in pop_by_tag[tag]:
            if r["instance_id"] not in J_by_tag[tag].get(judge, {}): continue
            y.append(1 if r["g"] == 0 else 0); sc.append(-float(r["c"])); sj.append(-J_by_tag[tag][judge][r["instance_id"]]); grp.append(f"{tag}:{r['repo']}")
    y = np.array(y, float); sc = np.array(sc); sj = np.array(sj); grp = np.array(grp); X = np.column_stack([sc, sj])
    out = {"n": int(len(y)), "failures": int(y.sum()), "repositories": int(len(np.unique(grp))),
           "AUROC_selector_alone": round(float(auroc_fast(y, sc)), 3), "AUROC_judge_alone": round(float(auroc_fast(y, sj)), 3)}
    for name, fn in (("fixed_unit_weights_PRIMARY", loro_fixed), ("logistic_LORO", loro_logistic)):
        oof = fn(X, y, grp); a = float(auroc_fast(y, oof)); ci, _ = ci_of(y, sc, oof, grp)
        out[name] = {"AUROC_selector_plus_judge": round(a, 3), "delta_vs_selector": round(a - out["AUROC_selector_alone"], 3), "CI95": ci}
    return out

def main():
    pop, J = {}, {}
    for tag in BENCH: pop[tag], J[tag] = load(tag)
    rep = {"status": "POST HOC, not pre-registered; computed 2026-11-09 at two symposium reviewers' request",
           "protocol": "committed population; positive class failure; baseline -c; judge enters as -p_resolved (J1); LORO by repository; repository-clustered bootstrap CI (no refit inside draws, as registered)",
           "judges": {}}
    for judge in JUDGES:
        rep["judges"][judge] = {tag: analyse(pop, J, [tag], judge) for tag in BENCH}
        rep["judges"][judge]["pooled_descriptive"] = analyse(pop, J, list(BENCH), judge)
    out = ROOT / "data/readout_single_judge_incremental.json"
    out.write_text(json.dumps(rep, indent=1) + "\n"); print(f"wrote {out.relative_to(ROOT)}")
    for judge, d in rep["judges"].items():
        for tag, v in d.items():
            f = v["fixed_unit_weights_PRIMARY"]; l = v["logistic_LORO"]
            print(f"{judge:11s} {tag:19s} n={v['n']:3d} fail={v['failures']:2d} | -c {v['AUROC_selector_alone']:.3f} judge {v['AUROC_judge_alone']:.3f} | fixed: {f['AUROC_selector_plus_judge']:.3f} d={f['delta_vs_selector']:+.3f} CI{f['CI95']} | logistic: d={l['delta_vs_selector']:+.3f} CI{l['CI95']}")

if __name__ == "__main__":
    main()
