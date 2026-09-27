#!/usr/bin/env python3
"""E5: a test-derived taxonomy of the silent failures, and whether the null hides
a subclass where disagreement works.

Every class here is read off the executed test report, not from a judge: the
harness records which FAIL_TO_PASS tests the patch turned green and which
PASS_TO_PASS tests it broke. Classes are mutually exclusive and exhaustive.

The scientific point is not the histogram. It is the per-class AUROC: if
disagreement were useful on, say, regressions but useless on no-progress
failures, the pooled null would be masking that. Classes below the
pre-registered inference floor (10 positives) are reported descriptively only.

Costs nothing: no new agent runs, no new gold, no API calls.
"""
from __future__ import annotations
import json, sys, glob, collections
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT / "src"))
from silentfail.stats import auroc_fast  # noqa: E402

FLOOR = 10  # pre-registered inference floor: fewer positives => descriptive only


def boot(y, s, grp, n=4000, seed=5):
    g = np.asarray(grp); u = np.unique(g); idx = {k: np.flatnonzero(g == k) for k in u}
    rng = np.random.default_rng(seed); v = []
    for _ in range(n):
        ii = np.concatenate([idx[k] for k in rng.choice(u, size=len(u), replace=True)])
        if 0 < y[ii].sum() < len(ii): v.append(auroc_fast(y[ii], s[ii]))
    return [round(float(np.percentile(v, 2.5)), 3), round(float(np.percentile(v, 97.5)), 3)] if v else None


def load_verified_reports() -> dict:
    """Union of the 14 batched harness runs for the committed arm. Full test names."""
    out = {}
    for p in glob.glob(str(ROOT / "logs/run_evaluation/sf_*/silentfail/*/report.json")):
        for iid, v in json.load(open(p)).items():
            ts = v.get("tests_status", {}) or {}
            f2p = ts.get("FAIL_TO_PASS", {}) or {}; p2p = ts.get("PASS_TO_PASS", {}) or {}
            out[iid] = {
                "patch_exists": bool(v.get("patch_exists")) and not v.get("patch_is_None"),
                "patch_applied": bool(v.get("patch_successfully_applied")),
                "resolved": bool(v.get("resolved")),
                "f2p_ok": len(f2p.get("success", [])), "f2p_n": len(f2p.get("success", [])) + len(f2p.get("failure", [])),
                "p2p_ok": len(p2p.get("success", [])), "p2p_n": len(p2p.get("success", [])) + len(p2p.get("failure", [])),
            }
    return out


def load_multilingual_reports() -> dict:
    """gold_modal details for the committed arm. f2p/p2p as 'ok/n' fractions."""
    out = {}
    d = json.load(open(ROOT / "data/gold_modal/replication_multilingual/report.json"))
    for r in d.get("details", []):
        if r.get("model_name_or_path") != "silentfail_replication":
            continue
        def frac(s):
            try: a, b = str(s).split("/"); return int(a), int(b)
            except Exception: return 0, 0
        fo, fn = frac(r.get("f2p")); po, pn = frac(r.get("p2p"))
        out[r["instance_id"]] = {
            "patch_exists": r.get("patch_applied") is not None,
            "patch_applied": bool(r.get("patch_applied")),
            "resolved": bool(r.get("resolved")),
            "f2p_ok": fo, "f2p_n": fn, "p2p_ok": po, "p2p_n": pn,
        }
    return out


def classify(rep: dict) -> str:
    """Mutually exclusive classes for a FAILED (g=0) committed trajectory."""
    if not rep["patch_exists"]:
        return "no patch produced"
    if not rep["patch_applied"]:
        return "patch did not apply"
    broke = rep["p2p_n"] > 0 and rep["p2p_ok"] < rep["p2p_n"]
    target_fixed = rep["f2p_n"] > 0 and rep["f2p_ok"] == rep["f2p_n"]
    partial = rep["f2p_n"] > 0 and 0 < rep["f2p_ok"] < rep["f2p_n"]
    if target_fixed and broke:
        return "fixed the bug, broke something else"
    if broke:
        return "broke something else, bug unfixed"
    if partial:
        return "partial fix"
    return "no progress on the bug, broke nothing"


ORDER = ["no progress on the bug, broke nothing", "partial fix", "broke something else, bug unfixed",
         "fixed the bug, broke something else", "patch did not apply", "no patch produced"]


def run(tag, scored_path, reports):
    rows = [json.loads(l) for l in open(scored_path) if l.strip()]
    pop = [r for r in rows if not r["excluded"] and r["visible_pass"]]
    missing = [r["instance_id"] for r in pop if r["instance_id"] not in reports]
    assert not missing, f"{tag}: no test report for {missing}"
    mism = [r["instance_id"] for r in pop if int(reports[r["instance_id"]]["resolved"]) != r["g"]]
    assert not mism, f"{tag}: report/g mismatch for {mism}"

    fails = [r for r in pop if r["g"] == 0]
    cls = {r["instance_id"]: classify(reports[r["instance_id"]]) for r in fails}
    counts = collections.Counter(cls.values())
    out = {"benchmark": tag, "n_analysed": len(pop), "n_failures": len(fails),
           "classes": {k: counts.get(k, 0) for k in ORDER if counts.get(k, 0)},
           "per_class": {}}

    # how many FAIL_TO_PASS tests guard a pass, and how many PASS_TO_PASS tests exist
    out["guard_tests"] = {
        "f2p_n_median_over_pop": int(np.median([reports[r["instance_id"]]["f2p_n"] for r in pop])),
        "pop_with_single_f2p_test": int(sum(1 for r in pop if reports[r["instance_id"]]["f2p_n"] == 1)),
    }

    # does disagreement detect one class better than another?
    y_all = np.array([1 if r["g"] == 0 else 0 for r in pop])
    u_all = np.array([r["u"] for r in pop]); c_all = np.array([r["c"] for r in pop])
    grp_all = np.array([r["repo"] for r in pop])
    passes = [r for r in pop if r["g"] == 1]
    for k in ORDER:
        sub = [r for r in fails if cls[r["instance_id"]] == k]
        if not sub:
            continue
        # one-vs-all-passes: can u separate THIS class of failure from the passes?
        keep = sub + passes
        y = np.array([1] * len(sub) + [0] * len(passes))
        u = np.array([r["u"] for r in keep]); c = np.array([r["c"] for r in keep])
        grp = np.array([r["repo"] for r in keep])
        d = {"n": len(sub), "vs_passes_n": len(keep),
             "below_inference_floor": len(sub) < FLOOR,
             "AUROC_u": round(auroc_fast(y, u), 3),
             "AUROC_negc": round(auroc_fast(y, -c), 3)}
        if len(sub) >= FLOOR:
            d["AUROC_u_CI"] = boot(y, u, grp)
        out["per_class"][k] = d
    out["AUROC_u_all_failures"] = round(auroc_fast(y_all, u_all), 3)
    out["AUROC_negc_all_failures"] = round(auroc_fast(y_all, -c_all), 3)
    out["_cls"] = cls
    return out


v = run("verified", ROOT / "data/swebench_pilot50/pilot_free_scored.jsonl", load_verified_reports())
m = run("multilingual", ROOT / "data/replication_multilingual/replication_scored.jsonl", load_multilingual_reports())

# pooled taxonomy
pooled = collections.Counter()
for r in (v, m):
    for k, n in r["classes"].items(): pooled[k] += n
rep = {"analysis": "E5 failure taxonomy, test-derived, exploratory (run after the confirmatory results).",
       "classes_are": "mutually exclusive and exhaustive over g=0 committed trajectories; read from the executed harness report, not a judge.",
       "inference_floor": FLOOR,
       "verified": {k: x for k, x in v.items() if k != "_cls"},
       "multilingual": {k: x for k, x in m.items() if k != "_cls"},
       "pooled_classes": {k: pooled[k] for k in ORDER if pooled.get(k)},
       "pooled_n_failures": sum(pooled.values())}
(ROOT / "data/readout_failure_taxonomy.json").write_text(json.dumps(rep, indent=2) + "\n")

print("=== E5 failure taxonomy (test-derived) ===")
for tag, r in (("Verified", v), ("Multilingual", m)):
    print(f"\n{tag}: {r['n_failures']} silent failures of {r['n_analysed']} analysed")
    for k, n in r["classes"].items():
        print(f"   {n:3d}  ({100*n/r['n_failures']:4.1f}%)  {k}")
    print(f"   median FAIL_TO_PASS tests guarding a pass: {r['guard_tests']['f2p_n_median_over_pop']}"
          f" | rows guarded by exactly one test: {r['guard_tests']['pop_with_single_f2p_test']}/{r['n_analysed']}")
print(f"\nPooled over {rep['pooled_n_failures']} failures:")
for k, n in rep["pooled_classes"].items():
    print(f"   {n:3d}  ({100*n/rep['pooled_n_failures']:4.1f}%)  {k}")
print("\n=== does disagreement work better on any one class? (class vs all passes) ===")
for tag, r in (("Verified", v), ("Multilingual", m)):
    print(f"  {tag}  (all failures: u={r['AUROC_u_all_failures']}, -c={r['AUROC_negc_all_failures']})")
    for k, d in r["per_class"].items():
        floor = " [below floor, descriptive]" if d["below_inference_floor"] else f" CI {d.get('AUROC_u_CI')}"
        print(f"     n={d['n']:3d}  u={d['AUROC_u']:.3f}  -c={d['AUROC_negc']:.3f}  {k}{floor}")
print("\nwrote data/readout_failure_taxonomy.json")
