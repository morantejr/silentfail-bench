#!/usr/bin/env python3
"""Amendment 4c: single-family LLM-as-judge panel (OpenAI gpt-5 / gpt-5-mini / gpt-5-nano).

  run      call the judges (cached in judgments.jsonl; safe to re-run)
  readout  AUROCs, u_LLM, bars, paired test vs the primary critic

The API key is read from mini-swe-agent's global .env in-process and never printed.
"""
from __future__ import annotations

import argparse, json, os, re, sys, time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(Path(__file__).resolve().parent))
from silentfail.scoring import apply_panel_u, fit_rank_norms  # noqa: E402
from silentfail.stats import auroc_fast, cluster_bootstrap_delta  # noqa: E402

JUDGES = ["gpt-5", "gpt-5-mini", "gpt-5-nano"]
PRICE = {"gpt-5": (1.25, 10.0), "gpt-5-mini": (0.25, 2.0), "gpt-5-nano": (0.05, 0.40)}   # $/M in, out
TAIL_CHARS, DIFF_CAP, COST_CAP = 24_000, 60_000, 10.0
ENV = Path.home() / "Library/Application Support/mini-swe-agent/.env"
SYSTEM = ("You audit patches proposed by an automated software agent for a GitHub issue. "
          "You never see the maintainers' hidden tests. Estimate how likely the patch fully resolves the issue "
          "as those hidden tests would judge it. Respond with JSON only.")


def load_key() -> None:
    if os.environ.get("OPENAI_API_KEY"): return
    for line in ENV.read_text().splitlines():
        if line.startswith("OPENAI_API_KEY="):
            os.environ["OPENAI_API_KEY"] = line.split("=", 1)[1].strip().strip('"').strip("'"); return
    sys.exit("OPENAI_API_KEY not found; set it with: mini-extra config set OPENAI_API_KEY <key>")


def prompt(c: dict, cond: str) -> str:
    diff = c["final_solution"] or ""; note = ""
    if len(diff) > DIFF_CAP: diff, note = diff[:DIFF_CAP], "\n[diff truncated]"
    p = f"Problem statement:\n{c['problem_statement']}\n\nProposed patch (unified diff):\n{diff}{note}\n"
    if cond == "J2":
        from gpu_rescore_windows import prune
        tail = prune(c["text"])[-TAIL_CHARS:]
        p += f"\nFinal portion of the agent's transcript (its own commands and test runs; long outputs pruned):\n{tail}\n"
    p += ('\nRespond with JSON only: {"p_resolved": <integer 0-100>, "reason": "<at most 40 words>"}')
    return p


def parse(txt: str):
    try:
        d = json.loads(txt); v = int(round(float(d["p_resolved"]))); return max(0, min(100, v)), str(d.get("reason", ""))[:300]
    except Exception:
        m = re.search(r'"p_resolved"\s*:\s*(\d+)', txt)
        return (max(0, min(100, int(m.group(1)))), "regex-parsed") if m else (None, txt[:200])


def run(a) -> None:
    load_key()
    from openai import OpenAI
    client = OpenAI()
    cands = [json.loads(l) for l in a.bundle.read_text().splitlines() if l.strip()]
    committed = [c for c in cands if c["candidate"] == c["committed_index_primary"] and not c["excluded"]]
    if a.limit: committed = committed[: a.limit]
    a.cache.parent.mkdir(parents=True, exist_ok=True)
    done = {(d["judge"], d["cond"], d["instance_id"]) for d in map(json.loads, a.cache.read_text().splitlines())} if a.cache.exists() else set()
    spent = 0.0
    for cond in ("J1", "J2"):
        for judge in JUDGES:
            for c in committed:
                key = (judge, cond, c["instance_id"])
                if key in done: continue
                if spent > COST_CAP: sys.exit(f"cost cap ${COST_CAP} reached (est ${spent:.2f})")
                msgs = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": prompt(c, cond)}]
                rec = {"judge": judge, "cond": cond, "instance_id": c["instance_id"], "candidate": c["candidate"]}
                for attempt in range(3):
                    try:
                        kw = dict(model=judge, messages=msgs)
                        if attempt == 0: kw["response_format"] = {"type": "json_object"}
                        r = client.chat.completions.create(**kw)
                        txt = r.choices[0].message.content or ""
                        p, reason = parse(txt)
                        u = r.usage; cost = (u.prompt_tokens * PRICE[judge][0] + u.completion_tokens * PRICE[judge][1]) / 1e6
                        spent += cost
                        rec.update({"p_resolved": p, "reason": reason, "prompt_tokens": u.prompt_tokens, "completion_tokens": u.completion_tokens,
                                    "model_returned": r.model, "est_cost_usd": round(cost, 5), "attempt": attempt})
                        if p is not None or attempt == 2: break
                    except Exception as e:
                        rec.update({"p_resolved": None, "error": f"{type(e).__name__}: {str(e)[:200]}"}); time.sleep(3 * (attempt + 1))
                with a.cache.open("a") as f: f.write(json.dumps(rec) + "\n")
                done.add(key)
                print(f"  {cond} {judge:11s} {c['instance_id']:40s} p={rec.get('p_resolved')} (${spent:.2f} so far)", flush=True)
    print(f"[done] est spend ${spent:.2f}", flush=True)


def readout(a) -> None:
    prim = {r["instance_id"]: r for r in map(json.loads, a.rows.read_text().splitlines()) if r}
    usable = [i for i, r in prim.items() if not r["excluded"]]
    pop = [i for i in usable if prim[i].get("visible_pass")]
    y = np.array([1 if prim[i]["g"] == 0 else 0 for i in pop]); grp = [prim[i]["repo"] for i in pop]
    cp = np.array([prim[i]["c"] for i in pop], float)
    J = {}
    for d in map(json.loads, a.cache.read_text().splitlines()):
        if d.get("p_resolved") is not None: J[(d["judge"], d["cond"], d["instance_id"])] = 100 - d["p_resolved"]
    rep = {"analysis": "Amendment 4c (pre-registered): single-family LLM judge panel (OpenAI); gpt-5-mini is also the agent model",
           "population_n": len(pop), "silent_failures": int(y.sum()), "bar": 0.665, "conditions": {}}
    for cond in ("J1", "J2"):
        R = {"judges": {}}
        for j in JUDGES:
            s = np.array([J.get((j, cond, i), np.nan) for i in pop], float); ok = ~np.isnan(s)
            if ok.sum() < len(pop) - 3: R["judges"][j] = {"missing": int((~ok).sum())}; continue
            s = np.where(ok, s, np.nanmedian(s))
            t = cluster_bootstrap_delta(y, -cp, s, grp, endpoint="auroc", n_boot=4000, seed=7)
            R["judges"][j] = {"AUROC_detect": round(auroc_fast(y, s), 3), "missing": int((~ok).sum()),
                              "M3_vs_primary_c": round(t.delta, 3), "CI95": [round(t.ci_lo, 3), round(t.ci_hi, 3)]}
        have = [j for j in JUDGES if "AUROC_detect" in R["judges"][j]]
        if len(have) == 3:
            tr = {j: [J[(j, cond, i)] for i in usable if prim[i]["split"] == "train" and (j, cond, i) in J] for j in JUDGES}
            fits = fit_rank_norms(tr)
            u = np.array([apply_panel_u({j: J.get((j, cond, i), np.nan) for j in JUDGES}, fits)[1] for i in pop])
            t = cluster_bootstrap_delta(y, -cp, u, grp, endpoint="auroc", n_boot=4000, seed=8)
            R["u_LLM"] = {"AUROC": round(auroc_fast(y, u), 3), "M3_vs_primary_c": round(t.delta, 3), "CI95": [round(t.ci_lo, 3), round(t.ci_hi, 3)]}
            best = max(have, key=lambda j: R["judges"][j]["AUROC_detect"])
            R["best_single"] = {"judge": best, "AUROC": R["judges"][best]["AUROC_detect"]}
        rep["conditions"][cond] = R
    J1, J2 = rep["conditions"]["J1"], rep["conditions"]["J2"]
    if "best_single" in J1 and "best_single" in J2:
        bj = J1["best_single"]["judge"]
        gain = round(J2["judges"][bj]["AUROC_detect"] - J1["judges"][bj]["AUROC_detect"], 3)
        rep["predictions"] = {"A4c-P1_best_single_J1_gt_bar": bool(J1["best_single"]["AUROC"] > 0.665),
                              "A4c-P2_uLLM_J1_gt_bar": bool(J1["u_LLM"]["AUROC"] > 0.665),
                              "A4c-P3_J2_minus_J1_best_judge": gain, "A4c-P3_pass": bool(gain > 0.10)}
    a.out.write_text(json.dumps(rep, indent=1)); print(json.dumps(rep, indent=1))


def main() -> None:
    ap = argparse.ArgumentParser(); ap.add_argument("cmd", choices=["run", "readout"])
    ap.add_argument("--bundle", type=Path, default=ROOT / "data/swebench_pilot50/rescoring_bundle.jsonl")
    ap.add_argument("--rows", type=Path, default=ROOT / "data/swebench_pilot50/pilot_free_scored.jsonl")
    ap.add_argument("--cache", type=Path, default=ROOT / "data/swebench_pilot50/llm_judges/judgments.jsonl")
    ap.add_argument("--out", type=Path, default=ROOT / "data/swebench_pilot50/prereg_readout_amendment4c.json")
    ap.add_argument("--limit", type=int, default=None)
    a = ap.parse_args(); (run if a.cmd == "run" else readout)(a)


if __name__ == "__main__":
    main()
