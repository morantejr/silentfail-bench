#!/usr/bin/env python3
"""Amendment 11 readout: does a larger step budget change the Amendment 10 result?

Two arms at step_limit=100 on the same 41 Verified instances: DeepSeek V4.1 Flash
(the candidate) and gpt-5-mini (the P10-2 comparator, re-measured at the same budget
because its registered 0.486 was measured at 40 steps). Gates are identical to
Amendment 10's. The budget was chosen after Amendment 10's outcome was visible, so a
pass here is a NEW measurement and is never reported as P10-1 passing.

Classification of gold rows, declared here because it is post hoc. The gold wrapper
filed two agent-caused outcomes under "error": a malformed diff that does not apply,
and a patch that applies but makes the package un-importable so the eval script
exits before any test result is logged (swebench raises KeyError: 'tests_status').
Amendment 10b's "exclude harness errors, stop above 10%" rule was written for
infrastructure faults. Counting agent-caused crashes as exclusions would remove the
comparator's failures from a capability comparison, in its favour. So:

  way A  Amendment 10b as written: every "error" row excluded, population shrinks.
  way B  agent-caused rows (patch apply failed; KeyError tests_status with a test log
         present) counted as unresolved per Amendment 7a section 3; only genuine
         infrastructure errors excluded.

Both are reported. P11-2 passes only if it passes BOTH ways, so the reclassification
cannot be what decides the verdict.

Fails closed (exit 2) if any input is missing or the measured cost is not > 0.
Exit 1 if any gate fails, 0 if all pass. Writes data/readout_amendment11.json.
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BAR_SUBMIT, BAR_GAP, BAR_COST, LIMIT = 0.80, 0.10, 0.040, 100
AGENT_CAUSED = ("patch apply failed", "KeyError: 'tests_status'")


def load_json(p: Path):
    if not p.exists():
        print(f"ABORT: missing {p}. The amendment is incomplete; no gate is computed."); sys.exit(2)
    return json.loads(p.read_text())


def submitted(instances, preds_rows):
    return [i for i in instances if (preds_rows.get(i, {}).get("model_patch") or "").strip()]


def classify(report, model_name, instances):
    """Return resolved, unresolved, agent_caused, infra sets for one arm's rows."""
    res, unres, agent, infra = set(), set(), set(), set()
    for d in report["details"]:
        if d.get("model_name_or_path") != model_name or d["instance_id"] not in instances:
            continue
        iid = d["instance_id"]
        err = d.get("error")
        if err is None:
            (res if d.get("resolved") else unres).add(iid)
        elif any(err.startswith(a) for a in AGENT_CAUSED):
            agent.add(iid)
        else:
            infra.add(iid)
    seen = res | unres | agent | infra
    missing = set(instances) - seen
    if missing:
        print(f"ABORT: {len(missing)} instances have no gold row for {model_name}: {sorted(missing)[:5]}"); sys.exit(2)
    return res, unres, agent, infra


def rates(n, res, unres, agent, infra):
    pop_a = n - len(agent) - len(infra)          # way A: every error row excluded
    pop_b = n - len(infra)                       # way B: agent-caused rows are failures
    fail_a = len(unres)
    fail_b = len(unres) + len(agent)
    return {"population_A": pop_a, "failures_A": fail_a, "failure_rate_A": round(fail_a / pop_a, 3),
            "population_B": pop_b, "failures_B": fail_b, "failure_rate_B": round(fail_b / pop_b, 3),
            "resolved": len(res), "unresolved_plain": len(unres),
            "agent_caused_crashes": sorted(agent), "infrastructure_errors": sorted(infra)}


def step_stats(dirs, instances):
    """Assistant-turn counts per instance; later dirs override earlier (the w=1 rerun)."""
    steps = {}
    for d in dirs:
        for i in instances:
            f = d / i / f"{i}.traj.json"
            if f.exists():
                t = json.loads(f.read_text())
                steps[i] = (sum(1 for m in t.get("messages", []) if m.get("role") == "assistant"),
                            str((t.get("info") or {}).get("exit_status")),
                            float(((t.get("info") or {}).get("model_stats") or {}).get("instance_cost") or 0.0))
    vals = sorted(s for s, _, _ in steps.values())
    return steps, {"n": len(vals), "min": vals[0], "median": vals[len(vals) // 2], "max": vals[-1],
                   "hit_ceiling": sum(1 for s in vals if s >= LIMIT), "needed_over_40": sum(1 for s in vals if s > 40)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--deepseek-measured-cost", type=float, required=True,
                    help="OpenRouter spend for the DeepSeek arm, differenced externally by the runner")
    ap.add_argument("--instances", type=Path, default=ROOT / "configs/pilot_instances_swebench41_complete.txt")
    a = ap.parse_args()
    if a.deepseek_measured_cost <= 0:
        print("ABORT: --deepseek-measured-cost must be > 0; a zero cost is an unmeasured cost (DEFECTS #17)."); return 2

    inst = [l.strip() for l in a.instances.read_text().splitlines() if l.strip()]
    n = len(inst)
    if n != 41:
        print(f"ABORT: {n} instances, expected 41"); return 2

    # --- DeepSeek arm --------------------------------------------------------
    ds_dir = ROOT / "data/pilot_a11_deepseek"
    ds_preds = load_json(ds_dir / "preds.json")
    ds_gold = load_json(ROOT / "data/gold_modal/a11_deepseek/report.json")
    ds_sub = submitted(inst, ds_preds)
    ds_cls = classify(ds_gold, "deepseek/deepseek-v4.1-flash", inst)
    ds_steps, ds_stat = step_stats([ds_dir], inst)
    ds_rates = rates(n, *ds_cls)

    # --- gpt-5-mini comparator (merged: 38 rows at w=6 + 3 rerun at w=1) -----
    gm_dir, gm_rerun = ROOT / "data/pilot_a11_gpt5mini", ROOT / "data/pilot_a11_gpt5mini_rerun3"
    merged_path = gm_dir / "predictions_merged.jsonl"
    if not merged_path.exists():
        print(f"ABORT: missing {merged_path}"); return 2
    gm_preds = {r["instance_id"]: r for r in (json.loads(l) for l in merged_path.read_text().splitlines() if l.strip())}
    gm_gold = load_json(ROOT / "data/gold_modal/a11_gpt5mini_merged/report.json")
    gm_sub = submitted(inst, gm_preds)
    gm_cls = classify(gm_gold, "gpt-5-mini", inst)
    gm_steps, gm_stat = step_stats([gm_dir, gm_rerun], inst)
    gm_rates = rates(n, *gm_cls)
    gm_cost = round(sum(c for _, _, c in gm_steps.values()), 4)
    rerun_ids = sorted(i for i in inst if (gm_rerun / i / f"{i}.traj.json").exists())

    # --- the Amendment 10 record, for the budget comparison ------------------
    a10 = load_json(ROOT / "data/pilot_deepseek_a10/readout_amendment10.json")

    # --- gates ----------------------------------------------------------------
    p1 = {"submitted": len(ds_sub), "rate": round(len(ds_sub) / n, 3), "bar": BAR_SUBMIT,
          "passes": len(ds_sub) / n >= BAR_SUBMIT,
          "at_40_steps_amendment_10": {"submitted": a10["P10_1"]["submitted"], "rate": a10["P10_1"]["rate"]}}
    gap_a = round(gm_rates["failure_rate_A"] - ds_rates["failure_rate_A"], 3)
    gap_b = round(gm_rates["failure_rate_B"] - ds_rates["failure_rate_B"], 3)
    p2 = {"deepseek": ds_rates, "gpt5mini_at_same_budget": gm_rates,
          "gap_A_amendment_10b_exclusion": gap_a, "gap_B_agent_crashes_as_failures": gap_b, "bar": BAR_GAP,
          "passes_A": gap_a >= BAR_GAP, "passes_B": gap_b >= BAR_GAP,
          "passes": (gap_a >= BAR_GAP) and (gap_b >= BAR_GAP),
          "classification_post_hoc": True,
          "note": "passes only if both classifications agree; the post-hoc reclassification cannot decide the verdict",
          "gpt5mini_registered_40_step_baseline_NOT_used": a10["P10_2"]["gpt5mini_baseline"]["failure_rate"]}
    per_traj = round(a.deepseek_measured_cost / n, 4)
    p3 = {"measured_total_usd": round(a.deepseek_measured_cost, 4), "per_trajectory_usd": per_traj, "bar": BAR_COST,
          "passes": per_traj <= BAR_COST, "source": "OpenRouter credits endpoint, differenced by the runner",
          "gpt5mini_scaffold_reported_total_usd": gm_cost, "gpt5mini_per_trajectory_usd": round(gm_cost / n, 4)}

    out = {"amendment": 11, "step_limit": LIMIT, "instances": n,
           "status": "post hoc with respect to the budget: 100 was chosen after Amendment 10's 32/41 was visible; "
                     "a pass here is a new measurement and is not Amendment 10's P10-1 passing",
           "P11_1": p1, "P11_2": p2, "P11_3": p3,
           "budget_was_binding_at_40": {"deepseek_steps_at_100": ds_stat, "gpt5mini_steps_at_100": gm_stat,
                                        "reading": "no DeepSeek trajectory reached the 100-step ceiling, so the "
                                                   "100-step result is not budget-limited"},
           "comparator_rerun": {"instances": rerun_ids,
                                "why": "exited RateLimitError on OpenAI's 500k TPM ceiling at 6 workers; "
                                       "regenerated at 1 worker so a rate limit is not scored as a capability failure"},
           "all_gates_pass": p1["passes"] and p2["passes"] and p3["passes"]}
    out["decision"] = ("All three pass at a 100-step budget. The agent axis becomes runnable on DeepSeek V4.1 Flash "
                       "under its own amendment, and every result it produces must carry the post-hoc-budget caveat."
                       if out["all_gates_pass"] else
                       "Not all gates pass. Report as attempted; Amendment 10's verdicts stand; the axis stays closed.")
    (ROOT / "data/readout_amendment11.json").write_text(json.dumps(out, indent=2) + "\n")

    print(f"== Amendment 11 gates (step_limit={LIMIT}) ==")
    print(f"  P11_1  {'PASS' if p1['passes'] else 'FAIL'}   DeepSeek submitted {p1['submitted']}/{n} = {p1['rate']} vs bar {BAR_SUBMIT}"
          f"   (40 steps: {a10['P10_1']['submitted']}/{n} = {a10['P10_1']['rate']})")
    print(f"  P11_2  {'PASS' if p2['passes'] else 'FAIL'}   gap A {gap_a} / gap B {gap_b} vs bar {BAR_GAP}"
          f"   DeepSeek fail {ds_rates['failure_rate_B']} vs gpt-5-mini {gm_rates['failure_rate_A']} (A) / {gm_rates['failure_rate_B']} (B)")
    print(f"  P11_3  {'PASS' if p3['passes'] else 'FAIL'}   ${per_traj}/traj vs cap ${BAR_COST}   (gpt-5-mini ${p3['gpt5mini_per_trajectory_usd']}/traj)")
    print(f"  steps  DeepSeek max {ds_stat['max']}, hit ceiling {ds_stat['hit_ceiling']}; gpt-5-mini max {gm_stat['max']}, hit ceiling {gm_stat['hit_ceiling']}")
    print(f"  gpt-5-mini agent-caused crashes: {len(gm_rates['agent_caused_crashes'])}; infrastructure errors: {len(gm_rates['infrastructure_errors'])}")
    print()
    print("  " + out["decision"])
    return 0 if out["all_gates_pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
