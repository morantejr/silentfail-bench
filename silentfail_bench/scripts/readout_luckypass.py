#!/usr/bin/env python3
"""Lucky-pass contamination of the g=1 labels, and whether the headline survives it.

AgentLens (Sahoo et al., arXiv:2605.12925) documents that trajectories marked resolved on
SWE-bench can pass for the wrong reason. Our positive class is FAILURE, so a lucky pass
is a row mislabelled as a NEGATIVE: it inflates the apparent separability of both c
and u, and it is the one labelling error that could hide a real signal.

We cannot adjudicate semantic correctness without re-running gold, so we bound it.
Three cheap, executed-evidence flags per passing row:
  - disjoint  : the agent's patch touches none of the files the reference patch touches
  - one_test  : exactly one FAIL_TO_PASS test defines 'resolved' for this instance
  - tiny      : the patch adds two lines or fewer
Then we recompute the headline under the worst case for our claim: drop the flagged
passes, and separately relabel them as failures. If the null holds in both, label
contamination cannot be what produced it.

Costs nothing: gold patches come from the cached benchmark datasets, test counts from
the harness reports already on disk. No new gold runs, no API calls.
"""
from __future__ import annotations
import json, sys, glob, re, collections
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT / "src"))
from silentfail.combine import _fit_standardizer, fit_logistic_irls  # noqa: E402
from silentfail.stats import auroc_fast, cluster_bootstrap_delta  # noqa: E402

TEST_PAT = re.compile(r"(^|/)(tests?|testing|spec|specs)(/|$)|(^|/)test_[^/]*$|[^/]*_test\.[a-z]+$|[^/]*Test[s]?\.[a-z]+$|\.spec\.[a-z]+$")


def diff_files(diff: str) -> set[str]:
    """Paths touched by a unified diff, from the 'diff --git' headers and +++ lines."""
    out = set()
    for line in (diff or "").splitlines():
        if line.startswith("diff --git "):
            parts = line.split()
            for p in parts[2:]:
                out.add(re.sub(r"^[ab]/", "", p))
        elif line.startswith("+++ ") and not line.startswith("+++ /dev/null"):
            out.add(re.sub(r"^[ab]/", "", line[4:].strip().split("\t")[0]))
    return {p for p in out if p and p != "/dev/null"}


def added_lines(diff: str) -> int:
    return sum(1 for l in (diff or "").splitlines() if l.startswith("+") and not l.startswith("+++"))


def load_gold(name, split="test"):
    from datasets import load_dataset
    ds = load_dataset(name, split=split)
    return {r["instance_id"]: {"patch": r["patch"], "test_patch": r.get("test_patch", "")} for r in ds}


def verified_reports():
    out = {}
    for p in glob.glob(str(ROOT / "logs/run_evaluation/sf_*/silentfail/*/report.json")):
        for iid, v in json.load(open(p)).items():
            ts = (v.get("tests_status") or {}); f2p = ts.get("FAIL_TO_PASS", {}) or {}
            out[iid] = {"resolved": bool(v.get("resolved")),
                        "f2p_n": len(f2p.get("success", [])) + len(f2p.get("failure", []))}
    return out


def multilingual_reports():
    out = {}
    d = json.load(open(ROOT / "data/gold_modal/replication_multilingual/report.json"))
    for r in d.get("details", []):
        if r.get("model_name_or_path") != "silentfail_replication":
            continue
        try: n = int(str(r.get("f2p")).split("/")[1])
        except Exception: n = 0
        out[r["instance_id"]] = {"resolved": bool(r.get("resolved")), "f2p_n": n}
    return out


# ---------- endpoint estimators (mirrors verify_and_robustify.py) ----------
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

def endpoint(y, c, u, grp, seed=21):
    out = {"n": int(len(y)), "positives": int(y.sum()),
           "raw_AUROC_u": round(auroc_fast(y, u), 3), "raw_AUROC_negc": round(auroc_fast(y, -c), 3)}
    Xc, Xcu = c[:, None], np.column_stack([c, u])
    for name, folds in (("logistic_LORO", folds_loro(grp)), ("logistic_5fold", folds_k(grp))):
        pc, pcu = oof(Xc, y, folds), oof(Xcu, y, folds); ok = ~np.isnan(pc) & ~np.isnan(pcu)
        t = cluster_bootstrap_delta(y[ok], pc[ok], pcu[ok], grp[ok], endpoint="auroc", n_boot=4000, seed=seed)
        out[name] = {"delta": round(t.delta, 3), "CI95": [round(t.ci_lo, 3), round(t.ci_hi, 3)]}
    pc = fixed_combo((-c)[:, None], y, folds_loro(grp)); pcu = fixed_combo(np.column_stack([-c, u]), y, folds_loro(grp))
    t = cluster_bootstrap_delta(y, pc, pcu, grp, endpoint="auroc", n_boot=4000, seed=seed + 1)
    out["fixed_unit_weights"] = {"delta": round(t.delta, 3), "CI95": [round(t.ci_lo, 3), round(t.ci_hi, 3)]}
    return out


def run(tag, scored_path, reports, gold):
    rows = [json.loads(l) for l in open(scored_path) if l.strip()]
    pop = [r for r in rows if not r["excluded"] and r["visible_pass"]]
    have_gold = [r for r in pop if r["instance_id"] in gold]
    assert len(have_gold) == len(pop), f"{tag}: missing gold patch for {[r['instance_id'] for r in pop if r['instance_id'] not in gold]}"

    flags = {}
    for r in pop:
        iid = r["instance_id"]; sol = r.get("final_solution") or ""
        af = diff_files(sol); gf = diff_files(gold[iid]["patch"]); tf = diff_files(gold[iid]["test_patch"])
        flags[iid] = {
            "g": r["g"],
            "agent_files": len(af), "gold_files": len(gf),
            "overlap": len(af & gf),
            "disjoint": bool(af) and not (af & gf),
            "no_patch": not af,
            "one_test": reports[iid]["f2p_n"] == 1,
            "tiny": added_lines(sol) <= 2,
            "touched_gold_test_files": bool(af & tf),
            "touched_any_test_path": any(TEST_PAT.search(p) for p in af),
        }
    passes = [r for r in pop if r["g"] == 1]
    fails = [r for r in pop if r["g"] == 0]

    def rate(sel, of):
        return {"n": sum(1 for r in of if sel(flags[r["instance_id"]])), "of": len(of)}

    out = {"benchmark": tag, "n_analysed": len(pop), "n_passes": len(passes), "n_failures": len(fails),
           "among_passes": {
               "disjoint_from_reference_patch": rate(lambda f: f["disjoint"], passes),
               "guarded_by_exactly_one_test": rate(lambda f: f["one_test"], passes),
               "adds_two_lines_or_fewer": rate(lambda f: f["tiny"], passes),
               "edited_a_file_the_test_patch_touches": rate(lambda f: f["touched_gold_test_files"], passes),
               "edited_any_test_path": rate(lambda f: f["touched_any_test_path"], passes),
               "disjoint_AND_one_test": rate(lambda f: f["disjoint"] and f["one_test"], passes),
           },
           "among_failures_for_contrast": {
               "disjoint_from_reference_patch": rate(lambda f: f["disjoint"], fails),
               "edited_any_test_path": rate(lambda f: f["touched_any_test_path"], fails),
           },
           "median_file_overlap_passes": float(np.median([flags[r["instance_id"]]["overlap"] for r in passes])),
           "median_file_overlap_failures": float(np.median([flags[r["instance_id"]]["overlap"] for r in fails])) if fails else None}

    # ---- sensitivity: worst case for our claim ----
    suspect = {r["instance_id"] for r in passes if flags[r["instance_id"]]["disjoint"]}
    out["suspect_definition"] = "passing rows whose patch touches no file the reference patch touches"
    out["n_suspect"] = len(suspect)
    out["suspect_rate_of_passes"] = round(len(suspect) / len(passes), 3) if passes else None
    out["suspect_ids"] = sorted(suspect)

    def arrs(subset, relabel=False):
        y = np.array([1 if (r["g"] == 0 or (relabel and r["instance_id"] in suspect)) else 0 for r in subset])
        return (y, np.array([r["c"] for r in subset], float), np.array([r["u"] for r in subset], float),
                np.array([f"{tag}:{r['repo']}" for r in subset]))
    out["scenarios"] = {}
    out["scenarios"]["as_labelled"] = endpoint(*arrs(pop))
    kept = [r for r in pop if r["instance_id"] not in suspect]
    out["scenarios"]["drop_suspect_passes"] = endpoint(*arrs(kept))
    out["scenarios"]["relabel_suspect_as_failure"] = endpoint(*arrs(pop, relabel=True))
    out["_flags"] = flags
    out["_pop"] = pop
    return out


gV = load_gold("princeton-nlp/SWE-bench_Verified")
gM = load_gold("swe-bench/SWE-Bench_Multilingual")
V = run("verified", ROOT / "data/swebench_pilot50/pilot_free_scored.jsonl", verified_reports(), gV)
M = run("multilingual", ROOT / "data/replication_multilingual/replication_scored.jsonl", multilingual_reports(), gM)

# pooled, with the same suspect rule applied within each benchmark
popP = V["_pop"] + M["_pop"]
susP = set(V["suspect_ids"]) | set(M["suspect_ids"])
def arrsP(subset, relabel=False):
    y = np.array([1 if (r["g"] == 0 or (relabel and r["instance_id"] in susP)) else 0 for r in subset])
    return (y, np.array([r["c"] for r in subset], float), np.array([r["u"] for r in subset], float),
            np.array([f"{r['_b']}:{r['repo']}" for r in subset]))
for r in V["_pop"]: r["_b"] = "verified"
for r in M["_pop"]: r["_b"] = "multilingual"
pooled = {"n": len(popP), "n_suspect": len(susP),
          "suspect_rate_of_passes": round(len(susP) / sum(1 for r in popP if r["g"] == 1), 3),
          "as_labelled": endpoint(*arrsP(popP)),
          "drop_suspect_passes": endpoint(*arrsP([r for r in popP if r["instance_id"] not in susP])),
          "relabel_suspect_as_failure": endpoint(*arrsP(popP, relabel=True))}

rep = {"analysis": "Lucky-pass contamination of g=1 and headline sensitivity to it. Exploratory.",
       "reference": "AgentLens (Sahoo et al. 2026, arXiv:2605.12925) - resolved SWE-bench trajectories that pass for the wrong reason.",
       "why_it_matters": "positive class is failure, so a lucky pass is a mislabelled negative; it would inflate apparent separability of both c and u.",
       "verified": {k: v for k, v in V.items() if not k.startswith("_")},
       "multilingual": {k: v for k, v in M.items() if not k.startswith("_")},
       "pooled": pooled}
(ROOT / "data/readout_luckypass.json").write_text(json.dumps(rep, indent=2) + "\n")

print("=== lucky-pass contamination of the g=1 labels ===")
for tag, r in (("Verified", V), ("Multilingual", M)):
    print(f"\n{tag}: {r['n_passes']} passes, {r['n_failures']} failures")
    for k, d in r["among_passes"].items():
        print(f"   {d['n']:3d}/{d['of']:3d}  ({100*d['n']/d['of']:4.1f}%)  {k}")
    print(f"   median agent/reference file overlap: passes {r['median_file_overlap_passes']:.0f}, failures {r['median_file_overlap_failures']:.0f}")
    print(f"   contrast, among FAILURES: disjoint {r['among_failures_for_contrast']['disjoint_from_reference_patch']['n']}"
          f"/{r['among_failures_for_contrast']['disjoint_from_reference_patch']['of']}")
print(f"\nsuspect rate (disjoint passes): Verified {V['suspect_rate_of_passes']}, Multilingual {M['suspect_rate_of_passes']}, pooled {pooled['suspect_rate_of_passes']}")
print("   (our flag is a file-disjointness proxy, not a semantic judgement: an upper bound, not their quantity)")
print("\n=== does the headline survive the worst case? (pooled) ===")
for name in ("as_labelled", "drop_suspect_passes", "relabel_suspect_as_failure"):
    d = pooled[name]
    print(f"  {name:30s} n={d['n']:3d} pos={d['positives']:3d}  u={d['raw_AUROC_u']:.3f}  -c={d['raw_AUROC_negc']:.3f}"
          f"  dLORO={d['logistic_LORO']['delta']:+.3f} {d['logistic_LORO']['CI95']}"
          f"  d5fold={d['logistic_5fold']['delta']:+.3f}"
          f"  dUnit={d['fixed_unit_weights']['delta']:+.3f} {d['fixed_unit_weights']['CI95']}")
print("\nwrote data/readout_luckypass.json")
