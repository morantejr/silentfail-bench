#!/usr/bin/env python3
"""Margin-stratified ('pivotal vote') analysis, after Shu (arXiv:2608.06940).

That paper finds, on code-correctness panels, that verification helps only on
decisions settled by a single vote, and that population-level metrics miss it.
Here the analogue is the SELECTION margin: how far ahead the committed
candidate was. If disagreement carries signal anywhere, the close calls are
where it should show. Uses only committed data already on disk.
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT / "src"))
from silentfail.stats import auroc_fast  # noqa: E402

def boot(y, s, grp, n=4000, seed=2):
    g = np.asarray(grp); u = np.unique(g); idx = {k: np.flatnonzero(g == k) for k in u}
    rng = np.random.default_rng(seed); v = []
    for _ in range(n):
        ii = np.concatenate([idx[k] for k in rng.choice(u, size=len(u), replace=True)])
        if 0 < y[ii].sum() < len(ii): v.append(auroc_fast(y[ii], s[ii]))
    return [round(float(np.percentile(v, 2.5)), 3), round(float(np.percentile(v, 97.5)), 3)] if v else None

def run(path, tag):
    rows = [json.loads(l) for l in Path(path).read_text().splitlines() if l.strip()]
    pop = [r for r in rows if not r["excluded"] and r["visible_pass"]]
    out = {"benchmark": tag, "n": len(pop)}
    margins = []
    for r in pop:
        cs = [x for x in r["candidate_scores"] if x == x and abs(x) != float("inf")]
        cs = sorted(cs, reverse=True)
        margins.append(cs[0] - cs[1] if len(cs) >= 2 else np.nan)
    m = np.array(margins, float)
    ok = ~np.isnan(m)
    y = np.array([1 if r["g"] == 0 else 0 for r in pop]); u = np.array([r["u"] for r in pop]); c = np.array([r["c"] for r in pop])
    grp = np.array([r["repo"] for r in pop])
    out["margin"] = {"median": round(float(np.median(m[ok])), 3), "iqr": [round(float(np.percentile(m[ok], 25)), 3), round(float(np.percentile(m[ok], 75)), 3)]}
    # does a close call predict failure at all?
    out["AUROC_small_margin_predicts_failure"] = round(auroc_fast(y[ok], -m[ok]), 3)
    out["AUROC_small_margin_CI"] = boot(y[ok], -m[ok], grp[ok])
    # stratify: pivotal (closest third) vs comfortable (widest third)
    q1, q2 = np.percentile(m[ok], [33.3, 66.7])
    strata = {"pivotal (closest third)": m <= q1, "middle": (m > q1) & (m <= q2), "comfortable (widest third)": m > q2}
    out["strata"] = {}
    for name, sel in strata.items():
        sel = sel & ok
        yy, uu, cc, gg = y[sel], u[sel], c[sel], grp[sel]
        d = {"n": int(sel.sum()), "failures": int(yy.sum())}
        if 0 < yy.sum() < sel.sum():
            d["AUROC_u"] = round(auroc_fast(yy, uu), 3); d["AUROC_u_CI"] = boot(yy, uu, gg)
            d["AUROC_negc"] = round(auroc_fast(yy, -cc), 3)
        out["strata"][name] = d
    return out

rep = {"analysis": "Margin-stratified ('pivotal vote') check, exploratory, no new data.",
       "reference": "Shu 2026, arXiv:2608.06940 - verification concentrates on single-vote-margin cases.",
       "verified": run(ROOT/"data/swebench_pilot50/pilot_free_scored.jsonl", "verified"),
       "multilingual": run(ROOT/"data/replication_multilingual/replication_scored.jsonl", "multilingual")}
for b in ("verified", "multilingual"):
    x = rep[b]; print(f"\n{b}: n={x['n']}  selection margin median {x['margin']['median']} IQR {x['margin']['iqr']}")
    print(f"  a narrow margin predicts failure: AUROC {x['AUROC_small_margin_predicts_failure']} {x['AUROC_small_margin_CI']}")
    for k, v in x["strata"].items():
        extra = f"  u={v.get('AUROC_u')} {v.get('AUROC_u_CI')}  -c={v.get('AUROC_negc')}" if "AUROC_u" in v else "  (one class only)"
        print(f"    {k:28s} n={v['n']:3d} failures={v['failures']:3d}{extra}")
(ROOT/"data/readout_pivotal.json").write_text(json.dumps(rep, indent=1))
