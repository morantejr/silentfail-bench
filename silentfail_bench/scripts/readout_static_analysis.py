#!/usr/bin/env python3
"""Do classical static-analysis signals carry what the critics do not?

The paper's structural baseline is `lines added`, which is a rhetorical device
more than a deployable guardrail: it is trivially gamed and it measures size, not
structure. This asks the fairer question a reviewer should ask. If cheap,
non-LLM software-engineering signals -- the kind a linter already computes --
rank silent failures as well as reward models do, then the finding is not "models
are bad at this", it is "this population is separable by structure and the models
are not finding the structure".

Features are computed from the committed diff alone, which is what an
inference-time guardrail would have. Two families:

  lexical   : decision points, nesting, hunk count, files touched, error
              handling, comment ratio, signature changes. Language-agnostic,
              computed for all eight languages in the two benchmarks.
  syntactic : Python only (Verified is 100% Python). Added hunks are parsed with
              the standard library `ast` after dedent; parse failure is itself a
              feature, since a patch whose added code does not stand alone is a
              different kind of object.

Everything is repository-clustered. The combined model is evaluated
leave-one-repository-out so it is comparable to the paper's endpoint. Costs
nothing: no API, no GPU, no new gold.
"""
from __future__ import annotations
import ast, json, re, sys, textwrap
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT / "src"))
from silentfail.combine import _fit_standardizer, fit_logistic_irls  # noqa: E402
from silentfail.stats import auroc_fast  # noqa: E402

DECISION = re.compile(r"\b(if|elif|else\s+if|for|while|case|switch|catch|except|rescue|when|unless)\b|&&|\|\||\?\s*:")
ERRH = re.compile(r"\b(try|except|catch|rescue|recover|finally|raise|throw|panic)\b")
SIG = re.compile(r"\b(def|func|fn|function|class|struct|impl|public|private|protected|static)\b")
COMMENT = re.compile(r"^\s*(#|//|/\*|\*|--)")


def added_lines(diff: str) -> list[str]:
    return [l[1:] for l in (diff or "").splitlines() if l.startswith("+") and not l.startswith("+++")]

def removed_count(diff: str) -> int:
    return sum(1 for l in (diff or "").splitlines() if l.startswith("-") and not l.startswith("---"))

def touched_files(diff: str) -> set[str]:
    out = set()
    for l in (diff or "").splitlines():
        if l.startswith("diff --git "):
            for p in l.split()[2:]: out.add(re.sub(r"^[ab]/", "", p))
    return out

def hunks(diff: str) -> int:
    return sum(1 for l in (diff or "").splitlines() if l.startswith("@@"))


def lexical(diff: str) -> dict:
    add = added_lines(diff)
    code = [l for l in add if l.strip() and not COMMENT.match(l)]
    dec = sum(len(DECISION.findall(l)) for l in code)
    nest = max((len(l) - len(l.lstrip())) for l in code) if code else 0
    return {
        "added_lines": float(len(add)),
        "decision_points": float(dec),
        "cyclomatic_added": float(dec + 1),
        "decision_density": float(dec / len(code)) if code else 0.0,
        "max_indent": float(nest),
        "hunks": float(hunks(diff)),
        "files_touched": float(len(touched_files(diff))),
        "net_lines": float(len(add) - removed_count(diff)),
        "adds_error_handling": float(any(ERRH.search(l) for l in code)),
        "touches_signature": float(any(SIG.search(l) for l in code)),
        "comment_ratio": float(sum(1 for l in add if COMMENT.match(l)) / len(add)) if add else 0.0,
        "no_control_flow": float(dec == 0),
    }


def syntactic_python(diff: str) -> dict:
    """Parse each added hunk with ast. Only meaningful where the diff is Python."""
    blocks, cur = [], []
    for l in (diff or "").splitlines():
        if l.startswith("@@"):
            if cur: blocks.append(cur); cur = []
        elif l.startswith("+") and not l.startswith("+++"):
            cur.append(l[1:])
    if cur: blocks.append(cur)
    parsed = failed = nodes = fdefs = calls = 0
    for b in blocks:
        src = textwrap.dedent("\n".join(b))
        if not src.strip(): continue
        try:
            t = ast.parse(src)
        except SyntaxError:
            failed += 1; continue
        parsed += 1
        for n in ast.walk(t):
            nodes += 1
            if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)): fdefs += 1
            if isinstance(n, ast.Call): calls += 1
    tot = parsed + failed
    return {"ast_hunks_parsed": float(parsed), "ast_hunks_failed": float(failed),
            "ast_parse_fail_rate": float(failed / tot) if tot else 0.0,
            "ast_nodes": float(nodes), "ast_func_defs": float(fdefs), "ast_calls": float(calls)}


def loro_oof(X, y, grp):
    p = np.full(len(y), np.nan)
    for g in np.unique(grp):
        te = grp == g; tr = ~te
        if y[tr].sum() < 2 or (1 - y[tr]).sum() < 2: continue
        mu, sg = _fit_standardizer(X[tr]); coef, b = fit_logistic_irls((X[tr] - mu) / sg, y[tr].astype(float))
        p[te] = 1 / (1 + np.exp(-np.clip(((X[te] - mu) / sg) @ coef + b, -30, 30)))
    return p


def boot(y, s, grp, n=4000, seed=31):
    g = np.asarray(grp); u = np.unique(g); idx = {k: np.flatnonzero(g == k) for k in u}
    rng = np.random.default_rng(seed); v = []
    for _ in range(n):
        ii = np.concatenate([idx[k] for k in rng.choice(u, size=len(u), replace=True)])
        if 0 < y[ii].sum() < len(ii): v.append(auroc_fast(y[ii], s[ii]))
    return [round(float(np.percentile(v, 2.5)), 3), round(float(np.percentile(v, 97.5)), 3)] if v else None


def run(tag, path, python_only):
    rows = [json.loads(l) for l in open(path) if l.strip()]
    pop = [r for r in rows if not r["excluded"] and r["visible_pass"]]
    y = np.array([1 if r["g"] == 0 else 0 for r in pop])
    grp = np.array([r["repo"] for r in pop])
    feats = []
    for r in pop:
        f = lexical(r["final_solution"])
        if python_only: f.update(syntactic_python(r["final_solution"]))
        feats.append(f)
    names = sorted(feats[0])
    M = np.array([[f[n] for n in names] for f in feats], float)
    out = {"benchmark": tag, "n": len(pop), "failures": int(y.sum()), "repositories": int(len(set(grp))),
           "single_features": {}}
    for j, n in enumerate(names):
        col = M[:, j]
        if np.allclose(col, col[0]): out["single_features"][n] = "constant"; continue
        a = auroc_fast(y, col)
        # report the better-oriented direction, as a practitioner would use it
        out["single_features"][n] = {"AUROC": round(max(a, 1 - a), 3),
                                     "direction": "higher=worse" if a >= 0.5 else "lower=worse",
                                     "CI": boot(y, col if a >= 0.5 else -col, grp)}
    # combined model, out of fold, comparable to the paper's endpoint
    ok = ~np.all(np.isclose(M, M[0]), axis=0)
    p = loro_oof(M[:, ok], y, grp); m = ~np.isnan(p)
    out["combined_LORO"] = {"features": int(ok.sum()), "rows_scored": int(m.sum()),
                            "AUROC": round(auroc_fast(y[m], p[m]), 3) if 0 < y[m].sum() < m.sum() else None,
                            "CI": boot(y[m], p[m], grp[m]) if 0 < y[m].sum() < m.sum() else None}
    # the question the paper asks of disagreement, asked of static analysis:
    # does it add anything to the selector's own score, out of fold?
    from silentfail.stats import cluster_bootstrap_delta
    c = -np.array([r["c"] for r in pop], float)        # detection score
    u = np.array([r["u"] for r in pop], float)
    Xc = c[:, None]; Xcs = np.column_stack([c, M[:, ok]])
    pc, pcs = loro_oof(Xc, y, grp), loro_oof(Xcs, y, grp)
    good = ~np.isnan(pc) & ~np.isnan(pcs)
    t = cluster_bootstrap_delta(y[good], pc[good], pcs[good], grp[good], endpoint="auroc", n_boot=4000, seed=33)
    out["endpoint_static_adds_to_selector"] = {"delta": round(t.delta, 3), "CI95": [round(t.ci_lo, 3), round(t.ci_hi, 3)],
                                               "baseline_AUROC": round(auroc_fast(y[good], pc[good]), 3),
                                               "raw_AUROC_for_comparison": round(auroc_fast(y, c), 3)}
    # the logistic baseline above is depressed relative to raw, the failure mode this project
    # documented in DEFECTS.md. Repeat with the unfitted estimator, which cannot collapse.
    def fixed(X):
        q = np.full(len(y), np.nan)
        for g in np.unique(grp):
            te = grp == g
            mu, sg = _fit_standardizer(X[~te]); q[te] = ((X[te] - mu) / sg).sum(axis=1)
        return q
    qc, qcs = fixed(Xc), fixed(Xcs)
    t2 = cluster_bootstrap_delta(y, qc, qcs, grp, endpoint="auroc", n_boot=4000, seed=34)
    out["endpoint_unfitted"] = {"delta": round(t2.delta, 3), "CI95": [round(t2.ci_lo, 3), round(t2.ci_hi, 3)],
                                "baseline_AUROC": round(auroc_fast(y, qc), 3)}
    out["head_to_head"] = {"raw_AUROC_selector": round(auroc_fast(y, c), 3),
                           "raw_AUROC_panel_u": round(auroc_fast(y, u), 3),
                           "lines_added_alone": round(max(auroc_fast(y, M[:, names.index("added_lines")]),
                                                          1 - auroc_fast(y, M[:, names.index("added_lines")])), 3)}
    return out


V = run("verified", ROOT / "data/swebench_pilot50/pilot_free_scored.jsonl", python_only=True)
M = run("multilingual", ROOT / "data/replication_multilingual/replication_scored.jsonl", python_only=False)
rep = {"analysis": "Classical static-analysis baseline: do non-LLM structural signals rank silent failures?",
       "why": "the paper's lines-added baseline measures size, not structure, and is trivially gamed. This tests the "
              "signals a linter already computes, which is the fair version of the same question.",
       "exploratory": True, "verified": V, "multilingual": M}
(ROOT / "data/readout_static_analysis.json").write_text(json.dumps(rep, indent=2) + "\n")

for tag, r in (("Verified", V), ("Multilingual", M)):
    print(f"\n=== {tag}: {r['n']} rows, {r['failures']} failures, {r['repositories']} repositories ===")
    sf = {k: v for k, v in r["single_features"].items() if isinstance(v, dict)}
    for n, d in sorted(sf.items(), key=lambda kv: -kv[1]["AUROC"])[:8]:
        print(f"   {d['AUROC']:.3f} {str(d['CI']):>16s}  {n}  ({d['direction']})")
    c = r["combined_LORO"]
    print(f"   combined, {c['features']} features, leave-one-repo-out: {c['AUROC']} {c['CI']} on {c['rows_scored']} rows")
    h = r["head_to_head"]; e = r["endpoint_static_adds_to_selector"]
    print(f"   head to head: selector {h['raw_AUROC_selector']}, panel u {h['raw_AUROC_panel_u']}, lines-added {h['lines_added_alone']}")
    print(f"   static adds to selector, LORO logistic: {e['delta']:+.3f} {e['CI95']}  baseline {e['baseline_AUROC']} vs raw {e['raw_AUROC_for_comparison']}"
          + ("   <-- BASELINE DEPRESSED" if e['baseline_AUROC'] < e['raw_AUROC_for_comparison'] - 0.10 else ""))
    f = r["endpoint_unfitted"]
    print(f"   static adds to selector, unfitted (primary): {f['delta']:+.3f} {f['CI95']}  baseline {f['baseline_AUROC']}")
print("\nwrote data/readout_static_analysis.json")
