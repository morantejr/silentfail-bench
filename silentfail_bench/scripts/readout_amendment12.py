#!/usr/bin/env python3
"""Amendment 12 readout: GLM-5.3 under DeepSeek's qualifying settings, against Amendment 11's
gpt-5-mini@100 comparator. Reuses Amendment 11's classification (both ways) and step stats.
Fails closed (exit 2) on missing inputs or a non-positive measured cost. Exit 1 if any gate
fails -- which P12-3 is expected to -- and 0 only if all pass. Writes data/readout_amendment12.json.
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from readout_amendment11 import classify, rates, step_stats, submitted, load_json, BAR_SUBMIT, BAR_GAP, BAR_COST  # noqa: E402
LIMIT, MAXTOK = 100, 16000

def main() -> int:
    ap = argparse.ArgumentParser(); ap.add_argument("--measured-cost", type=float, required=True); a = ap.parse_args()
    if a.measured_cost <= 0:
        print("ABORT: --measured-cost must be > 0 (DEFECTS #17)"); return 2
    inst = [l.strip() for l in (ROOT / "configs/pilot_instances_swebench41_complete.txt").read_text().splitlines() if l.strip()]
    n = len(inst)
    gdir = ROOT / "data/pilot_a12_glm"
    preds = load_json(gdir / "preds.json"); gold = load_json(ROOT / "data/gold_modal/a12_glm/report.json")
    sub = submitted(inst, preds); cls = classify(gold, "z-ai/glm-5.3", inst); r = rates(n, *cls)
    steps, stat = step_stats([gdir], inst)
    # The config's cost_limit (0.25/trajectory) bound GLM where it never bound DeepSeek:
    # LimitsExceeded exits sit at $0.25-0.30 and a median of ~31 steps, nowhere near the
    # 100-step ceiling. Record which limit actually cut each run, so P12-1 is not reported
    # as a budget verdict it is not.
    cost_cut = sorted(i for i, (st_, ex, c) in steps.items() if ex == "LimitsExceeded" and c >= 0.25)
    step_cut = sorted(i for i, (st_, ex, c) in steps.items() if ex == "LimitsExceeded" and c < 0.25 and st_ >= LIMIT)
    other_lim = sorted(i for i, (st_, ex, c) in steps.items() if ex == "LimitsExceeded" and i not in cost_cut and i not in step_cut)
    # comparator from Amendment 11, reused
    a11 = load_json(ROOT / "data/readout_amendment11.json"); gm = a11["P11_2"]["gpt5mini_at_same_budget"]
    a10 = load_json(ROOT / "data/pilot_deepseek_a10/readout_amendment10.json")
    # GLM's own 40-step record, from the committed pilot
    old = load_json(ROOT / "data/pilot_glm/preds.json"); old_sub = submitted(inst, old)
    p1 = {"submitted": len(sub), "rate": round(len(sub) / n, 3), "bar": BAR_SUBMIT,
          "passes": len(sub) / n >= BAR_SUBMIT,
          "evaluable_as_registered": len(cost_cut) == 0,
          "binding_limit": {"cost_limit_0.25_cut": cost_cut, "step_limit_cut": step_cut, "other_limits_exceeded": other_lim},
          "reading": ("the registered budget question is NOT answered: the per-trajectory cost cap, a setting the "
                      "amendment did not name, cut these runs before the step budget could bind; reported descriptively"
                      if cost_cut else "evaluable as registered"),
          "at_40_steps_4k_tokens": {"submitted": len(old_sub), "rate": round(len(old_sub) / n, 3)}}
    gA = round(gm["failure_rate_A"] - r["failure_rate_A"], 3); gB = round(gm["failure_rate_B"] - r["failure_rate_B"], 3)
    p2 = {"glm": r, "gpt5mini_at_same_budget_from_amendment_11": gm, "gap_A": gA, "gap_B": gB, "bar": BAR_GAP,
          "passes_A": gA >= BAR_GAP, "passes_B": gB >= BAR_GAP, "passes": gA >= BAR_GAP and gB >= BAR_GAP,
          "classification_post_hoc": True}
    per = round(a.measured_cost / n, 4)
    p3 = {"measured_total_usd": round(a.measured_cost, 4), "per_trajectory_usd": per, "bar": BAR_COST, "passes": per <= BAR_COST,
          "source": "OpenRouter credits endpoint, differenced by the runner", "at_40_steps_scaffold_usd": 0.1739}
    out = {"amendment": 12, "step_limit": LIMIT, "max_tokens": MAXTOK, "instances": n,
           "status": "post hoc with respect to every setting; two settings changed (budget and token cap) so a pass is not attributed to either alone; cannot qualify GLM for the axis",
           "P12_1": p1, "P12_2": p2, "P12_3": p3, "steps_at_100": stat,
           "all_gates_pass": p1["passes"] and p2["passes"] and p3["passes"]}
    if cost_cut:
        out["decision"] = (f"NOT EVALUABLE AS REGISTERED: the $0.25 cost limit cut {len(cost_cut)} of {n} runs before the step "
                           f"budget could bind. Under that cap GLM submitted on {len(sub)} of {n}. P12-3 fails at any budget, so GLM "
                           f"stays disqualified for the axis; Amendments 10 and 11 unchanged; the papers' 'mechanism unmeasured' stands.")
    else:
        out["decision"] = ("GLM functions under DeepSeek's settings (P12-1) and " + ("is" if p2["passes"] else "is not") +
                           " more capable than gpt-5-mini at equal budget; it " + ("passes" if p3["passes"] else "FAILS") +
                           " the cost cap and " + ("could" if out["all_gates_pass"] else "cannot") + " qualify for the axis. "
                           "Amendments 10 and 11 unchanged.")
    (ROOT / "data/readout_amendment12.json").write_text(json.dumps(out, indent=2) + "\n")
    print(f"== Amendment 12 gates (GLM-5.3, step_limit={LIMIT}, max_tokens={MAXTOK}) ==")
    print(f"  P12_1  {'PASS' if p1['passes'] else ('NOT EVALUABLE' if cost_cut else 'FAIL')}   submitted {len(sub)}/{n} = {p1['rate']} vs {BAR_SUBMIT}   (40 steps/4k: {len(old_sub)}/{n})")
    print(f"  P12_2  {'PASS' if p2['passes'] else 'FAIL'}   gap A {gA} / B {gB} vs {BAR_GAP}   GLM fail {r['failure_rate_B']} vs gpt-5-mini {gm['failure_rate_A']}/{gm['failure_rate_B']}")
    print(f"  P12_3  {'PASS' if p3['passes'] else 'FAIL'}   ${per}/traj vs ${BAR_COST}")
    print(f"  limit  cost-cap cut {len(cost_cut)}, step-cap cut {len(step_cut)}, other {len(other_lim)}")
    print(f"  steps  max {stat['max']}, hit ceiling {stat['hit_ceiling']}, needed >40: {stat['needed_over_40']}; agent-caused crashes {len(r['agent_caused_crashes'])}, infra errors {len(r['infrastructure_errors'])}")
    print("  " + out["decision"])
    return 0 if out["all_gates_pass"] else 1

if __name__ == "__main__":
    sys.exit(main())
