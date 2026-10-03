#!/usr/bin/env python3
"""Amendment 13: pool AUROC of every critic for detecting failure over the agent's graded,
non-empty attempts, with both signs, so the paper's "every critic is at chance" sentence is
derived from a producer and not from an ad-hoc calculation (DEFECTS #16 lesson).

Writes data/agent_axis_deepseek/readout_amendment13_panel.json. Argument-free; regenerates
identically. Fails closed (exit 2) if any input is missing.
"""
from __future__ import annotations
import glob, json, sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
O = ROOT / "data/agent_axis_deepseek"


def auroc(y, s):
    y = np.asarray(y); s = np.asarray(s, float); pos = s[y == 1]; neg = s[y == 0]
    if len(pos) == 0 or len(neg) == 0: return float("nan")
    gt = (pos[:, None] > neg[None, :]).sum(); eq = (pos[:, None] == neg[None, :]).sum()
    return float((gt + 0.5 * eq) / (len(pos) * len(neg)))


def main() -> int:
    bundle = {}
    for l in (O / "rescoring_bundle.jsonl").read_text().splitlines():
        if l.strip(): r = json.loads(l); bundle[(r["instance_id"], int(r["candidate"]))] = r
    gold = {}
    for c in range(4):
        p = ROOT / f"data/gold_modal/a7_agent_axis_cand{c}/report.json"
        if not p.exists(): print(f"ABORT: missing {p}"); return 2
        for d in json.loads(p.read_text())["details"]:
            if d.get("model_name_or_path", "").startswith("a7_") and d.get("error") is None:
                gold[(d["instance_id"], c)] = 0 if d["resolved"] else 1   # 1 = failure
    pool = sorted(k for k in gold if (bundle[k].get("final_solution") or "").strip())
    y = [gold[k] for k in pool]
    out = {"amendment": 13, "pool_n_graded_non_empty": len(pool), "pool_failures": int(sum(y)),
           "sign_convention": "detection score: higher = more likely to fail; reward models are negated, the judge is 100 - p_resolved",
           "critics": {}}
    files = sorted(glob.glob(str(O / "modal/prefix4k/scores_gpu_*.json")))
    if len(files) != 5: print(f"ABORT: expected 5 score files, found {len(files)}"); return 2
    for f in files:
        sc = json.load(open(f))["scores"]; s = [sc[f"{i}#{c}"]["score"] for i, c in pool]
        name = Path(f).stem.replace("scores_gpu_", "")
        out["critics"][name] = {"kind": "reward_model", "AUROC_detect": round(auroc(y, [-x for x in s]), 3), "AUROC_sign_flipped": round(auroc(y, s), 3)}
    judge = {}
    for l in (O / "judge_gpt5.jsonl").read_text().splitlines():
        if l.strip():
            r = json.loads(l)
            if r.get("judge") == "gpt-5" and r.get("cond") == "J1": judge[(r["instance_id"], int(r["candidate"]))] = float(r["p_resolved"])
    pk = [k for k in pool if k in judge]; yj = [gold[k] for k in pk]
    out["critics"]["gpt5_judge_J1"] = {"kind": "llm_judge", "n": len(pk), "AUROC_detect": round(auroc(yj, [100 - judge[k] for k in pk]), 3), "AUROC_sign_flipped": round(auroc(yj, [judge[k] for k in pk]), 3)}
    rms = [v["AUROC_detect"] for k, v in out["critics"].items() if v["kind"] == "reward_model"]
    out["reward_models"] = {"n": len(rms), "min_AUROC_detect": min(rms), "max_AUROC_detect": max(rms)}
    out["selector_gate_bar"] = 0.65
    out["any_critic_above_bar"] = bool(any(v["AUROC_detect"] > 0.65 for v in out["critics"].values()))
    (O / "readout_amendment13_panel.json").write_text(json.dumps(out, indent=2) + "\n")
    for k, v in out["critics"].items(): print(f"  {k:48s} AUROC_detect={v['AUROC_detect']:.3f}  flipped={v['AUROC_sign_flipped']:.3f}")
    print(f"  reward models: n={len(rms)} range [{min(rms):.3f}, {max(rms):.3f}]; any critic above 0.65: {out['any_critic_above_bar']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
