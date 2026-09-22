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

**Decision needed from you.** Two viable routes, presented without a
recommendation because the second is a budget question that is yours:

**Route 1 — re-scope the MDE.** Pre-register that the study only claims to
detect large effects (ΔAUROC ≈ 0.10), which ~960 trajectories supports. Costs
nothing and keeps the current plan roughly intact. The price is that a null
then says only "no large effect", leaving a real-but-moderate effect
undetected and unclaimed.

**Route 2 — fund n ≈ 1,000–2,000.** Keeps ΔAUROC ≈ 0.045 detectable and
preserves the original claim. This is beyond local compute: an inference API
(e.g. OpenRouter) makes generation and scoring tractable for roughly **$5–25**
at n=2,000, but real SWE-bench gold still needs containers, so the gating cost
is the execution environment rather than the tokens.

A third option I tested and rejected: switching the primary endpoint to a
budgeted-escalation metric. See §3.

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

## 3. A tempting fix that does not work

Switching the primary endpoint from global AUROC to precision@k looked
attractive — it is what an escalation policy actually acts on, and it matches
the `BUSINESS.md` framing. Measured, it is much *worse* powered:

| endpoint | null rejection | power (u=0.5) | power (u=1.0) |
|---|---|---|---|
| **global AUROC** | 0.008 | **0.258** | **0.850** |
| precision@10% | 0.000 | 0.000 | 0.175 |
| partial AUROC (FPR≤20%) | 0.025 | 0.192 | 0.567 |

precision@k is a step function: only the top-k ordering moves it, so most
bootstrap resamples return the same value and sensitivity collapses. Adopting
it would deepen the power problem, not solve it. It is now logged as a
**secondary** metric, which is where it belongs.

## 4. What the free pilot can and cannot claim

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

## 5. Oracle validation caught a fake 100% failure rate

The first smoke run scored `g=0` on every trajectory — which would have read as
a spectacular failure rate. It was a harness bug: the sandboxed interpreter ran
with `-S`, so HumanEval+ gold tests could not `import numpy`. Caught only
because the EvalPlus **reference solutions are run through the gold harness
first**. After fixing that and a UTF-8 encoding bug, **163/164** references
pass. `HumanEval/32` is excluded as `gold_harness_invalid`.

Recommend keeping this control permanently: if the known-correct answer does
not pass, the oracle is broken, and every downstream number is fiction.

## 6. Pilot result — NO_GO, and why that is useful

| | |
|---|---|
| usable / excluded | 40 / 0 |
| failures (g=0) | 8 → **20%** [CI 7.6, 32.4] |
| visible-test pass rate | 87.5% |
| **silent failures** (visible PASS, gold FAIL) | **3** = 8.6% of visible-passers |
| corr(c, u) | 0.051 |
| decision | **NO_GO** |

Both stop conditions fired: failures 8 < 12 required, and AUROC(M_c) printed
1.000.

**The AUROC=1.000 is an artifact and must not be quoted.** The group split left
**7 test rows containing 2 failures**; held-out inference is not possible at
n=40. The rule fired mechanically on a degenerate split, which is the rule
working, not a finding about the critic.

### What the run establishes

1. **HumanEval+ is too easy.** Best-of-4 on a coder model leaves only 20%
   failures among selected trajectories. At that rate the power projection
   cannot reach ΔAUROC≈0.045 even at 1,920 trajectories. Real SWE-bench
   Verified is *better* on this axis — far lower solve rates mean far more
   failures per trajectory spent.
2. **The target phenomenon is real.** Three trajectories passed the visible
   tests and failed hidden gold, and the selecting critic scored **all three at
   c=8.0** — maximum confidence on exactly the cases it got wrong. That is the
   selection-conditional blind spot the proposal predicts.
3. **u did not detect them here.** Mean u was 0.163 for failures vs 0.189 for
   successes — slightly the wrong direction. With m=2, u is just |a−b|/√2, so
   this is weak evidence, not a result. c did carry signal (6.75 vs 7.56).

### Recommended next configuration

Harder tasks, more of them, and m≥4 — in that order of importance. The failure
rate, not the model quality, is what starved this pilot.

## 7. Cost / time

Measured: **$0**, ~22 GB of models, ~35 min wall clock for 40 trajectories
(generation 805s for candidate scoring alone). Per-trajectory cost is dominated
by critic scoring, not generation.

Extrapolating to n=1,000 locally: roughly 15 hours. Via an inference API,
roughly $5-25 and a couple of hours. True reward models add a separate local
cost — see the throughput note in DEFECTS.md.
