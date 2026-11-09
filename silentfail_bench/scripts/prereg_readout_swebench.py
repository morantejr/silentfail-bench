#!/usr/bin/env python3
"""Pre-registered readout for the SWE-bench Stage-0 pilot (PREREGISTRATION_SWEBENCH_PILOT.md).

Executes the registered analysis and nothing else. Population, positive
class, metrics, robustness panel and go/no-go are all fixed in the document;
this script is the mechanical implementation.

Usage: prereg_readout_swebench.py <pilot_free_scored.jsonl> [candidates.jsonl]
"""
from __future__ import annotations

import json, sys, collections
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from silentfail.identifiers import repo_from_instance  # noqa: E402
from silentfail.scoring import apply_panel_u, fit_rank_norms  # noqa: E402
from silentfail.stats import auroc_fast, cluster_bootstrap_delta  # noqa: E402


def wilson(k: int, n: int, z: float = 1.96):
    if n == 0: return (float("nan"), float("nan"))
    p = k / n; d = 1 + z*z/n; c = (p + z*z/(2*n)) / d; h = z*np.sqrt(p*(1-p)/n + z*z/(4*n*n)) / d
    return (round(c - h, 3), round(c + h, 3))


def icc_anova(y, groups):
    g = np.unique(groups); k = len(g)
    if k < 2: return float("nan")
    n_i = np.array([(groups == r).sum() for r in g], float); m_i = np.array([y[groups == r].mean() for r in g])
    N = n_i.sum(); mu = y.mean()
    msb = np.sum(n_i*(m_i-mu)**2)/(k-1); msw = np.sum([np.sum((y[groups == r]-m_i[j])**2) for j, r in enumerate(g)])/max(1, N-k)
    n0 = (N - np.sum(n_i**2)/N)/(k-1)
    return float((msb-msw)/(msb+(n0-1)*msw)) if (msb+(n0-1)*msw) else float("nan")


def metrics(rows, label, drop=()):
    """M1-M3 on the registered population, optionally with a panel member dropped (rank norms refit on train)."""
    if drop:
        tr = {}
        for r in rows:
            if r["split"] == "train":
                for k, v in r["audit_scores"].items():
                    if k not in drop: tr.setdefault(k, []).append(v)
        fits = fit_rank_norms(tr)
        u = np.array([apply_panel_u({k: v for k, v in r["audit_scores"].items() if k not in drop}, fits)[1] for r in rows])
    else:
        u = np.array([r["u"] for r in rows], float)
    y = np.array([1 if r["g"] == 0 else 0 for r in rows]); c = np.array([r["c"] for r in rows], float)
    grp = [r["repo"] for r in rows]; pos, neg = int(y.sum()), int((1-y).sum())
    out = {"panel": label, "n_population": len(rows), "silent_failures": pos, "successes": neg}
    if pos and neg:
        t = cluster_bootstrap_delta(y, -c, u, grp, endpoint="auroc", n_boot=4000, seed=1)
        out.update(M1_AUROC_u=round(auroc_fast(y, u), 3), M2_AUROC_negc=round(auroc_fast(y, -c), 3),
                   M3_delta=round(t.delta, 3), M3_CI95_repo_clustered=[round(t.ci_lo, 3), round(t.ci_hi, 3)],
                   M3_p_one_sided=round(t.p_one_sided, 4), n_repos_in_population=len(set(grp)),
                   P1_M1_gt_0p5=bool(auroc_fast(y, u) > 0.5), P2_M2_le_0p5=bool(auroc_fast(y, -c) <= 0.5),
                   inference=("<10 positives: point estimates only; NO conclusion either way" if pos < 10 else "eligible"))
    else:
        out["inference"] = "one class absent; nothing computable"
    return out


def main():
    rows = [json.loads(l) for l in Path(sys.argv[1]).read_text().splitlines() if l.strip()]
    cands = [json.loads(l) for l in Path(sys.argv[2]).read_text().splitlines()] if len(sys.argv) > 2 else []
    rep = {"file": sys.argv[1], "prereg": "PREREGISTRATION_SWEBENCH_PILOT.md"}

    # --- M0: measurements over everything, nothing dropped ---
    if cands:
        st = collections.Counter(c.get("exit_status") or "None" for c in cands)
        empty = sum(1 for c in cands if not (c.get("final_solution") or "").strip())
        rep["M0a_candidates"] = {"n": len(cands), "exit_statuses": dict(st.most_common()),
                                 "empty_patch": empty, "exclusion_rate": round(empty/len(cands), 3),
                                 "cost_usd_total": round(sum(c.get("cost_usd") or 0 for c in cands), 2)}
    committed = [r for r in rows]
    usable = [r for r in committed if not r["excluded"]]
    fails = sum(1 for r in usable if r["g"] == 0)
    rep["M0b_committed"] = {"n_committed": len(committed), "n_excluded": len(committed)-len(usable),
                            "n_usable": len(usable), "failures": fails,
                            "failure_rate": round(fails/len(usable), 3) if usable else None,
                            "failure_rate_CI95": wilson(fails, len(usable))}
    # registered population: Submitted + non-empty patch  (visible_pass carries exit_status=="Submitted")
    pop = [r for r in usable if r.get("visible_pass")]
    sf = sum(1 for r in pop if r["g"] == 0)
    rep["M0c_population"] = {"n": len(pop), "silent_failures": sf,
                             "silent_failure_rate": round(sf/len(pop), 3) if pop else None,
                             "silent_failure_rate_CI95": wilson(sf, len(pop)),
                             "P3_exceeds_humaneval_8pct": (sf/len(pop) > 0.08) if pop else None}
    if usable:
        y = np.array([1 if r["g"] == 0 else 0 for r in usable]); g = np.array([repo_from_instance(r["instance_id"]) for r in usable])
        rep["M0d_icc_by_repo"] = {"icc": round(icc_anova(y, g), 3), "n_repos": int(len(set(g))),
                                  "caveat": "9 repos: unstable by construction; public reference 0.033-0.053",
                                  "per_repo_failure_rate": {r: round(float(y[g == r].mean()), 2) for r in sorted(set(g))}}
    hist = collections.Counter(r.get("selected_candidate_index") for r in rows if r.get("selected_candidate_index") is not None)
    rep["watch_items"] = {"selected_index_histogram": dict(sorted(hist.items())),
                          "corr_c_u": round(float(np.corrcoef([r["c"] for r in usable], [r["u"] for r in usable])[0, 1]), 3) if len(usable) > 2 else None}

    # --- M1-M3 on the population, registered panel and m=3 robustness ---
    rep["registered_m4"] = metrics(pop, "m=4 (registered)")
    rep["robustness_m3_drop_audit_3"] = metrics(pop, "m=3 (audit_3 dropped)", drop=("audit_3",))

    # --- go/no-go, exactly as registered ---
    reasons = []
    excl = rep.get("M0a_candidates", {}).get("exclusion_rate")
    if excl is not None and excl > 0.40: reasons.append(f"exclusion rate {excl} > 0.40")
    sfr = rep["M0c_population"]["silent_failure_rate"]
    # projected positives at a 500-instance confirmatory run with the observed population share
    if pop and sfr is not None:
        proj = sfr * (len(pop)/len(committed)) * 500
        rep["projected_silent_failures_at_500"] = round(proj)
        if proj < 10: reasons.append(f"projected silent failures at 500 instances = {proj:.0f} < 10")
    else:
        reasons.append("population empty")
    rep["go_no_go"] = {"decision": "GO" if not reasons else "NO_GO", "reasons": reasons,
                       "note": "AUROC(M_c) headroom check applies only if a held-out logistic is computable (>=10 test positives)"}
    print(json.dumps(rep, indent=2, default=str))


if __name__ == "__main__":
    main()
