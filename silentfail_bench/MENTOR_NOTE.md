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

---

# 8. A structural limit that n cannot fix (found 2026-09-22, profile B setup)

SWE-bench Verified is **500 instances across only 12 repositories**, and the
distribution is severely imbalanced:

| repo | instances | share |
|---|---|---|
| django/django | 231 | **46.2%** |
| sympy/sympy | 75 | 15.0% |
| sphinx-doc/sphinx | 44 | 8.8% |
| matplotlib/matplotlib | 34 | 6.8% |
| scikit-learn/scikit-learn | 32 | 6.4% |
| *(7 more)* | 83 | 16.6% |
| pallets/flask | 1 | 0.2% |

The primary estimand is held-out ΔAUROC **across repositories**, so repo is the
clustering unit and the effective sample size is not n. With mean cluster size
41.7 — inflated to 129.2 by the imbalance — the design effect is severe:

| ICC | design effect | effective n (of 500) |
|---|---|---|
| 0.02 | 3.56 | **140** |
| 0.05 | 7.41 | **67** |
| 0.10 | 13.82 | **36** |

**This is a worse problem than the n=200 shortfall, and adding trajectories
does not solve it.** My earlier power simulations assumed one row per cluster
(groups = rows), which was optimistic. Under repo clustering, evaluating all
500 Verified instances may carry the power of roughly 67–140 independent
observations. A 70/30 repo split leaves about **4 test repos**, which is far
too few for a stable cluster bootstrap.

### Consequences

1. **n is the wrong lever.** 200 → 2,000 trajectories over 12 repos buys
   within-cluster data that the design effect heavily discounts.
2. **Repos are the scarce resource**, not instances.
3. **The unit of generalization may have to change.** Either widen the task
   pool beyond Verified (SWE-Gym, SWE-smith, or multiple benchmarks) to obtain
   more repositories, or retreat from "held-out repos" to "held-out instances"
   and state plainly that cross-repo generalization is untested.

### Measured (2026-09-23) — the ICC is no longer assumed

Estimated from six public SWE-bench Verified submissions that use the **exact
scaffold this proposal plans** (mini-swe-agent), via one-way ANOVA on
per-instance pass/fail grouped by repo. Data: `data/icc_public_swebench_verified.json`.

| model | failure rate | ICC | design effect | effective n (of 500) |
|---|---|---|---|---|
| gpt-5-mini | 43.8% | 0.033 | 5.25 | **95** |
| gemini-3-flash-high | 24.2% | 0.042 | 6.40 | **78** |
| deepseek-3-2-high | 30.0% | 0.050 | 7.44 | **67** |
| claude-4-5-sonnet-high | 28.6% | 0.036 | 5.60 | **89** |
| gpt-5-2-high | 27.2% | 0.052 | 7.70 | **65** |
| claude-4-6-opus | 24.4% | 0.053 | 7.83 | **64** |

The ICC is **0.033–0.053, tight across models** — it is a property of the
benchmark, not of any one agent. Effective n is **64–95**, in the middle of
the range assumed above. Concretely: pylint instances fail 60–80% for every
model; scikit-learn fails 6–25%. That spread *is* the clustering.

Two things this settles:

1. **The clustering bound is real and now sized.** Running all 500 Verified
   instances yields roughly the inferential weight of 64–95 independent
   observations for a repo-clustered estimand.
2. **SWE-bench supplies failures.** 24–44% failure rates versus 20% on
   HumanEval+ — the failure-starvation that produced the first NO_GO does not
   recur here. Repos, not failures, are the scarce resource.
