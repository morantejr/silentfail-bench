#!/usr/bin/env python3
"""EXPLORATORY (PREREGISTRATION_SWEBENCH_PILOT.md, Amendment 1): re-score the
committed trajectories on the SUBMISSION WINDOW instead of the prefix.

Input to every critic = final submitted diff + last ~3,000 tokens of the
trajectory. Same models, same committed indices (from scores.json), rank
norms fit on train repos only. Writes scores_tail.json and a rows file with
the tail-window c/u, then prints M1-M3 next to the primary. Reported
separately; cannot be promoted to primary.

Run with the VM stopped (8 GB model). Usage:
  rescore_tail_window.py --out data/swebench_pilot50
"""
from __future__ import annotations

import argparse, json, sys, time
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from silentfail.rm_backend import RewardModel  # noqa: E402
from silentfail.scoring import apply_panel_u, fit_rank_norms  # noqa: E402
from silentfail.stats import auroc_fast, cluster_bootstrap_delta  # noqa: E402

TAIL_CHARS = 12_000   # ~3,000 tokens


def window(cand: dict) -> str:
    diff = (cand.get("final_solution") or "").strip()
    tail = (cand.get("text") or "")[-TAIL_CHARS:]
    return f"## Final submitted diff\n{diff}\n\n## Last steps of the trajectory\n{tail}"


def m123(rows, u_key, label):
    pop = [r for r in rows if r.get("visible_pass") and not r["excluded"]]
    y = np.array([1 if r["g"] == 0 else 0 for r in pop]); pos = int(y.sum()); neg = int((1 - y).sum())
    out = {"label": label, "n": len(pop), "silent_failures": pos, "successes": neg}
    if pos and neg:
        u = np.array([r[u_key] for r in pop], float); c = np.array([r["c_tail" if u_key == "u_tail" else "c"] for r in pop], float)
        grp = [r["repo"] for r in pop]
        t = cluster_bootstrap_delta(y, -c, u, grp, endpoint="auroc", n_boot=4000, seed=1)
        out.update(M1_AUROC_u=round(auroc_fast(y, u), 3), M2_AUROC_negc=round(auroc_fast(y, -c), 3),
                   M3_delta=round(t.delta, 3), M3_CI95=[round(t.ci_lo, 3), round(t.ci_hi, 3)], M3_p=round(t.p_one_sided, 4),
                   inference=("<10 positives: point estimates only" if pos < 10 else "eligible"))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--anchor", type=Path, default=ROOT / "configs" / "anchor_free_rm4_n16.yaml")
    a = ap.parse_args()
    cfg = yaml.safe_load(a.anchor.read_text())
    rows = [json.loads(l) for l in (a.out / "pilot_free_scored.jsonl").read_text().splitlines() if l.strip()]
    sj = json.loads((a.out / "scores.json").read_text())
    committed = {k: int(v) for k, v in sj["committed"].items()}
    cands = {}
    for l in (a.out / "candidates.jsonl").read_text().splitlines():
        d = json.loads(l); cands[(d["instance_id"], d["candidate"])] = d
    meta_code = ("from datasets import load_dataset; import json,sys; ids=set(json.loads(sys.argv[1]));"
                 "print(json.dumps({r['instance_id']:r['problem_statement'] for r in load_dataset('princeton-nlp/SWE-bench_Verified',split='test') if r['instance_id'] in ids}))")
    import subprocess
    meta = json.loads(subprocess.run([str(ROOT / ".venv-swebench/bin/python"), "-c", meta_code, json.dumps([r["instance_id"] for r in rows])],
                                     capture_output=True, text=True, check=True).stdout.strip().splitlines()[-1])
    inputs = {r["instance_id"]: window(cands[(r["instance_id"], committed[r["instance_id"]])]) for r in rows}

    # c_sel on the window (committed trajectory only -- selection itself is NOT redone; that would change the population)
    t0 = time.time(); rm = RewardModel.load(sj["c_sel_hf_id"], "c_sel_tail")
    c_tail = {i: rm.score(meta[i], inputs[i]) for i in inputs}; rm.unload()
    print(f"[c_sel tail] {len(c_tail)} in {time.time()-t0:.0f}s", flush=True)
    audit_tail = {i: {} for i in inputs}
    for m in cfg["audit_panel"]["members"]:
        t0 = time.time(); rmm = RewardModel.load(m["hf_id"], m["id"])
        for i in inputs: audit_tail[i][m["id"]] = rmm.score(meta[i], inputs[i])
        rmm.unload(); print(f"[{m['id']} tail] done in {time.time()-t0:.0f}s", flush=True)
    (a.out / "scores_tail.json").write_text(json.dumps({"c_tail": c_tail, "audit_tail": audit_tail, "window_chars": TAIL_CHARS}, indent=1))

    train_raw = {m["id"]: [] for m in cfg["audit_panel"]["members"]}
    for r in rows:
        if r["split"] == "train" and not r["excluded"]:
            for k, v in audit_tail[r["instance_id"]].items(): train_raw[k].append(v)
    fits = fit_rank_norms(train_raw)
    for r in rows:
        r["c_tail"] = float(c_tail[r["instance_id"]])
        r["audit_tail_rank_norm"], r["u_tail"] = apply_panel_u(audit_tail[r["instance_id"]], fits)
    (a.out / "pilot_free_scored_tail.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    print(json.dumps({"PRIMARY (prefix, registered)": m123(rows, "u", "prefix"),
                      "EXPLORATORY (submission window)": m123(rows, "u_tail", "tail")}, indent=2))


if __name__ == "__main__":
    main()
