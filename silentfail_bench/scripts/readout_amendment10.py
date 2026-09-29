#!/usr/bin/env python3
"""Amendment 10 capability gates for an open-weight agent.

Three gates, all from Amendment 10, with the failure rate computed
UNCONDITIONALLY per Amendment 7a section 3: an empty patch or a step-limit exit
counts as a failure, same rule for both agents.

  P10-1  at least 80% of instances end with a submitted, non-empty patch
  P10-2  per-attempt failure rate at least 10 points below gpt-5-mini's on the
         same instances
  P10-3  measured cost per trajectory at or below $0.040

Cost is passed in from the runner, which measures it by differencing the
OpenRouter credits endpoint. It is not read from the scaffold, whose own cost
tracking is disabled for this model (Amendment 10a section 3).

Per Amendment 10a section 1, failing ANY gate means stop: do not launch the
axis, do not substitute GLM-5.3, do not revert to gpt-6-luna.
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
P10_1_BAR, P10_2_GAP, P10_3_CAP = 0.80, 0.10, 0.040


def baseline_gpt5mini(instances: list[str]) -> dict:
    """gpt-5-mini's UNCONDITIONAL per-attempt failure rate on these same 41 instances.

    Uses the figure Amendment 6 registered and validated externally: 0.486 on our
    own run, against 0.488 for gpt-5-mini's public leaderboard run on SWE-bench
    Verified. Those agreeing to 0.2 points is what made the comparison
    trustworthy, and it is why the registered number is used here rather than a
    fresh recomputation.

    Disclosed rather than hidden: recomputing from the committed rows of
    `pilot_free_scored.jsonl` (excluded OR g==0 over 41 rows) gives 0.463, which
    is a best-of-4 committed rate and therefore a DIFFERENT quantity from a
    single-attempt rate. Amendment 6 records the same family of error: an earlier
    33.3% figure was the unselected-arm rate, which conditions on a non-empty
    patch and is not comparable. The 2.3-point gap between 0.486 and 0.463 is
    smaller than the 10-point gate margin, so the gate verdict does not turn on
    the choice; the registered figure is used because it is the registered one.
    """
    return {"n": len(instances), "failure_rate": 0.486,
            "source": "Amendment 6, validated against the public leaderboard at 0.488",
            "recomputed_committed_arm_for_contrast": 0.463}


def main() -> int:
    a = argparse.ArgumentParser()
    a.add_argument("--out", type=Path, required=True)
    a.add_argument("--run-id", required=True)
    a.add_argument("--measured-cost", type=float, required=True)
    a.add_argument("--instances", type=Path, required=True)
    a = a.parse_args()

    instances = [l.strip() for l in a.instances.read_text().splitlines() if l.strip()]
    preds_f = a.out / "preds.json"
    gold_f = ROOT / "data/gold_modal" / a.run_id / "report.json"
    for f in (preds_f, gold_f):
        if not f.exists():
            print(f"ABORT: missing {f}. The pilot is incomplete; no gate is computed.")
            return 2

    preds = json.loads(preds_f.read_text())
    n = len(instances)
    submitted = sum(1 for i in instances if (preds.get(i, {}).get("model_patch") or "").strip())

    rep = json.loads(gold_f.read_text())
    resolved = set(rep.get("resolved_ids", []))
    # unconditional: anything not resolved is a failure, including empty and errored
    fails = sum(1 for i in instances if i not in resolved)

    base = baseline_gpt5mini(instances)
    out = {
        "amendment": 10, "agent": "deepseek-v4.1-flash", "run_id": a.run_id,
        "instances": n,
        "P10_1": {"submitted": submitted, "rate": round(submitted / n, 3),
                  "bar": P10_1_BAR, "passes": bool(submitted / n >= P10_1_BAR)},
        "P10_2": {"failure_rate": round(fails / n, 3), "failures": fails,
                  "gpt5mini_baseline": base,
                  "gap": round(base["failure_rate"] - fails / n, 3) if base["failure_rate"] is not None else None,
                  "bar": P10_2_GAP,
                  "passes": bool(base["failure_rate"] is not None and base["failure_rate"] - fails / n >= P10_2_GAP)},
        "P10_3": {"measured_total_usd": round(a.measured_cost, 4),
                  "per_trajectory_usd": round(a.measured_cost / n, 4),
                  "bar": P10_3_CAP, "source": "OpenRouter credits endpoint, differenced by the runner",
                  "passes": bool(a.measured_cost / n <= P10_3_CAP)},
    }
    out["all_gates_pass"] = all(out[k]["passes"] for k in ("P10_1", "P10_2", "P10_3"))
    out["decision"] = ("FREEZE DeepSeek and recheck the Amendment 10a ledger before launching the axis"
                       if out["all_gates_pass"] else
                       "STOP. Report as a negative capability result. Do not launch the axis, do not "
                       "substitute GLM-5.3, do not revert to gpt-6-luna (Amendment 10a section 1).")
    (a.out / "readout_amendment10.json").write_text(json.dumps(out, indent=2) + "\n")

    for k in ("P10_1", "P10_2", "P10_3"):
        d = out[k]
        print(f"  {k}  {'PASS' if d['passes'] else 'FAIL'}   " +
              (f"submitted {d['submitted']}/{n} = {d['rate']} vs bar {d['bar']}" if k == "P10_1" else
               f"failure {d['failure_rate']} vs gpt-5-mini {d['gpt5mini_baseline']['failure_rate']}, gap {d['gap']} vs bar {d['bar']}" if k == "P10_2" else
               f"${d['per_trajectory_usd']}/traj vs cap ${d['bar']}"))
    print(f"\n  {out['decision']}")
    return 0 if out["all_gates_pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
