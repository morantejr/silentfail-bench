#!/usr/bin/env python3
"""Pre-registered M1-M3 for a scored JSONL, plus the Amendment-1 m=3 robustness.

Reports both the registered m=4 panel and the panel with `audit_3` dropped
(fully family-disjoint from a Qwen3 c_sel). Neither is chosen after the fact;
both are printed. Rank norms are refit on train rows only for the m=3 case.
"""
from __future__ import annotations

import json, sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from silentfail.scoring import apply_panel_u, fit_rank_norms  # noqa: E402
from silentfail.stats import auroc_fast, cluster_bootstrap_delta  # noqa: E402


def readout(rows, label, drop=()):
    if drop:
        train_raw = {}
        for r in rows:
            if r["split"] == "train":
                for k, v in r["audit_scores"].items():
                    if k not in drop: train_raw.setdefault(k, []).append(v)
        fits = fit_rank_norms(train_raw)
        u = np.array([apply_panel_u({k: v for k, v in r["audit_scores"].items() if k not in drop}, fits)[1] for r in rows])
    else:
        u = np.array([r["u"] for r in rows], float)
    strat = [i for i, r in enumerate(rows) if r.get("visible_pass") and not r["excluded"]]
    y = np.array([1 if rows[i]["g"] == 0 else 0 for i in strat]); us = u[strat]
    c = np.array([rows[i]["c"] for i in strat], float); grp = [rows[i]["repo"] for i in strat]
    pos, neg = int(y.sum()), int((1 - y).sum())
    out = {"panel": label, "n_visible_pass": len(strat), "silent_failures": pos, "successes": neg}
    if pos and neg:
        t = cluster_bootstrap_delta(y, -c, us, grp, endpoint="auroc", n_boot=4000, seed=1)
        out.update(M1_AUROC_u=round(auroc_fast(y, us), 3), M2_AUROC_negc=round(auroc_fast(y, -c), 3),
                   M3_delta=round(t.delta, 3), M3_CI95=[round(t.ci_lo, 3), round(t.ci_hi, 3)],
                   M3_p_one_sided=round(t.p_one_sided, 4),
                   status=("<10 positives: point estimates only, no conclusion" if pos < 10 else "eligible for test"))
    return out


def main():
    rows = [json.loads(l) for l in Path(sys.argv[1]).read_text().splitlines() if l.strip()]
    sel_idx = [r.get("selected_candidate_index") for r in rows if r.get("selected_candidate_index") is not None]
    hist = {}
    for i in sel_idx: hist[i] = hist.get(i, 0) + 1
    print(json.dumps({
        "file": sys.argv[1],
        "n_usable": sum(1 for r in rows if not r["excluded"]),
        "failures_total": sum(1 for r in rows if r["g"] == 0),
        "visible_pass_rate": round(float(np.mean([bool(r.get("visible_pass")) for r in rows])), 3),
        "P3_silent_failure_rate_visible_pass": round(sum(1 for r in rows if r["g"] == 0 and r.get("visible_pass")) / max(1, sum(1 for r in rows if r.get("visible_pass"))), 3),
        "selected_index_histogram": dict(sorted(hist.items())) if hist else "n/a",
        "corr_c_u": round(float(np.corrcoef([r["c"] for r in rows], [r["u"] for r in rows])[0, 1]), 3),
        "registered_m4": readout(rows, "m=4 (registered)"),
        "robustness_m3_drop_audit_3": readout(rows, "m=3 (audit_3 dropped; family-disjoint from Qwen3 c_sel)", drop=("audit_3",)),
    }, indent=2))


if __name__ == "__main__":
    main()
