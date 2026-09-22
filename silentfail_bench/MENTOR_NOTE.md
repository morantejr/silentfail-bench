# SilentFail-Bench — Stage-0 session note

**Date:** 2026-09-22 · **Session goal:** free-compute Stage-0 pilot with real gold, plus go/no-go.

---

## 1. The headline is a power problem, not a compute problem

The planned confirmatory run at **n = 200 is badly underpowered** for the
primary estimand. Using the corrected, null-calibrated clustered test, at an
observed-realistic failure rate:

| True effect | n for 80% power |
|---|---|
| ΔAUROC ≈ 0.045 | **~1,920 trajectories** |
| ΔAUROC ≈ 0.095 | **~960 trajectories** |
| ΔAUROC = 0.03 (the proposal's MDE) | **> 3,840** |

At n=200 the power against ΔAUROC≈0.045 is roughly **0.24**. This matters for
the "null is publishable" framing: at n=200 a null result would be
**indistinguishable from underpowering**, which is not a publishable null — it
is an uninformative one. Caveat in our favour: the new test is conservative
(actual α≈0.01 vs nominal 0.05), so these n's are pessimistic; a
better-calibrated test would need fewer, but the order of magnitude stands.

**Decision needed from you:** either raise n toward ~1,000, or re-scope the
primary estimand to an effect size worth powering for.

## 2. The old power function was reporting noise as signal

`pilot.simulate_power` defined power as P(ΔAUROC > 0) in-sample. That is ~0.5
under the null. Measured, with u carrying **zero** signal, it reported
**0.39–0.44 "power"**. Any go/no-go computed with it was meaningless. Replaced
with a paired cluster-bootstrap test (groups = repo / problem_id, models held
fixed); it now rejects at 0.010 under the null and scales properly with real
effects. Full detail in `DEFECTS.md`.

Three further defects would have corrupted a live run: hyphenated repo names
were truncated (`scikit-learn__scikit-learn-10297` → `scikit`), so the
repo-level split was not clean; builtin `hash()` made splits non-reproducible
across processes; and IRLS diverged under separation, producing NaN AUROC.
All four now have regression tests — suite is 5 → **18 tests**.

## 3. What the free pilot can and cannot claim

Real SWE-bench gold requires per-instance Docker images, and this machine has
no container runtime. Shrinking n or m does not help — it is orthogonal to
compute. So profile A substitutes HumanEval+ (gold = executed held-out tests,
agent loop genuinely multi-step) and accepts the cost honestly:

- **Not tested:** cross-repo generalization (split unit is `problem_id`).
- **Not RMs:** critics are LLM-as-judge (`llama3.1:8b` selecting;
  `gemma2:9b` + `mistral-nemo:12b` auditing), so `u` is judge disagreement,
  not RM-ensemble disagreement. Every logged row carries
  `critic_type: llm_judge`, `is_true_reward_model: false`.
- **Not quality-matched:** no RewardBench run locally, so the diverse-vs-seed
  panel contrast is out of scope.
- **Honored:** whole-trajectory `c`, disjoint c_sel ∉ audit panel,
  trajectory-level gold, train-only rank-norm fits.

Profile B (Colima + real SWE-bench Verified) restores all of it and is still
$0; it needs one install and an x86-emulation time budget.

## 4. Oracle validation caught a fake 100% failure rate

The first smoke run scored `g=0` on every trajectory — which would have read as
a spectacular failure rate. It was a harness bug: the sandboxed interpreter ran
with `-S`, so HumanEval+ gold tests could not `import numpy`. Caught only
because the EvalPlus **reference solutions are run through the gold harness
first**. After fixing that and a UTF-8 encoding bug, **163/164** references
pass. `HumanEval/32` is excluded as `gold_harness_invalid`.

Recommend keeping this control permanently: if the known-correct answer does
not pass, the oracle is broken, and every downstream number is fiction.

## 5. Cost / time for the confirmatory run

Free-pilot measured cost: **$0**, ~22 GB download, and the per-trajectory
timings in `pilot_report.json`. Extrapolation for the confirmatory run is in
§6 below once the pilot numbers land.
