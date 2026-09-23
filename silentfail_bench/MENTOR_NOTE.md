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


---

# 9. Second pilot — 4 true reward models (2026-09-23)

Same 40 tasks, same seeds, same selection critic. The **only** change is the
audit panel: 2 LLM judges → 4 true scalar reward models across three base
families. Gold outcomes reproduced exactly (8 failures, the same 3 silent
ones), so the comparison isolates the panel.

## The aggregate looked wrong — and was hiding the signal

Mean `u` for failures was *lower* than for successes in both pilots. That
number conflates two kinds of failure that behave oppositely:

| stratum | n | mean u, 2 judges | mean u, 4 RMs |
|---|---|---|---|
| obvious failure — visible tests FAIL, gold FAIL | 5 | 0.094 | **0.018** |
| **silent failure — visible PASS, gold FAIL** | 3 | 0.279 | **0.206** |
| success — visible PASS, gold PASS | 32 | 0.189 | 0.171 |

On obvious failures the panel *agrees* — there is nothing to disagree about,
the agent's own tests already caught it — so `u` collapses toward zero and
drags the failure mean down. That is correct behaviour, not a miss. The
silent failures, the cell this whole project is about, sit **above** the
successes in both pilots.

## The selection-conditional cell

Restricting to trajectories that passed their own visible tests (n=35, of
which 3 are silent failures), and asking whether a signal ranks the silent
failures above the successes:

| signal | AUROC, 2 judges | AUROC, 4 true RMs |
|---|---|---|
| selecting critic score `c` (low = risky) | 0.391 | **0.406** |
| audit-panel disagreement `u` | 0.562 | **0.688** |

Two things to see. First, **`c` is below chance in this cell** — the
selecting critic rates the silent failures as *more* likely correct than the
genuine successes. That is the Goodhart signature from §L24/§L37 of the
proposal, measured. Second, `u` moved from near-useless to 0.688 when the
panel became real reward models, and the per-critic breakdown shows why: on
each silent failure the four RMs genuinely split (e.g. `HumanEval/76`:
0.79 / 0.58 / 0.18 / 0.48 rank-normalized). The disagreement is real, not
one outlier.

## What this is and is not

- **3 positives against 32 negatives.** 0.688 is 66 of 96 pairwise wins. No
  test reaches significance here and none should be run. This is a signal to
  *design around*, not a result.
- **The visible-pass stratification was chosen after seeing the data.** It is
  well-motivated — it is the selection-conditional cell in the most literal
  sense, the agent's own verification passed — but it must be **pre-registered**
  for the confirmatory run, not presented as a finding of this one.
- `corr(c, u)` rose from 0.05 to **0.45** with true RMs. `u` is less
  independent of `c` than the judge-based `u` was; the frozen logistic with
  `c` as a covariate handles this, but it is worth watching.
- Still **NO_GO** on the original gates, for the same reasons as pilot 1
  (failure count; AUROC=1.000 on a 7-row split, an artifact).

## Recommendation for the design

1. Pre-register the visible-pass (or, on SWE-bench, "agent's own tests
   pass") stratum as the primary analysis population. It is where the
   phenomenon lives and where `c` is demonstrably blind.
2. Panel = true reward models, m ≥ 4, three or more base families. The judge
   panel was not measuring the construct; the RM panel is, and it costs
   ~6 min per 40 trajectories.
3. Move to SWE-bench for failure supply (24–44% vs 20%), accepting the
   12-repo clustering limit from §8 and choosing the generalization claim
   accordingly.


---

# 10. Pilots 3 and 3b — selection pressure, and a correction (2026-09-23)

## Pilot 3 (n=16, LLM-judge c_sel) did not apply selection

The judge emitted only {5, 6, 8} over 640 candidates; on 21/40 tasks all 16
tied at the max and argmax took index 0. **39/40 committed trajectories were
byte-identical to the n=4 run.** P3 was untested. Pre-registration Amendment 1
swapped `c_sel` to a true reward model (Skywork-Reward-V2-Qwen3-4B) and
changed nothing else.

## Pilot 3b (n=16, true-RM c_sel) — selection happened, and the signal went away

Chosen-index histogram spread across all 16 slots. Only 8/40 committed
trajectories match pilot 2. This is the first run in which "best-of-n" meant
what it says.

| | pilot 2 (n=4, judge c) | pilot 3b (n=16, RM c) |
|---|---|---|
| total failures | 8 | **6** |
| visible-pass rate | 87.5% | 92.5% |
| silent failures / visible-pass | 3/35 = 8.6% | 3/37 = **8.1%** |
| corr(c, u) | 0.46 | 0.27 |

Pre-registered primary, visible-pass population (n=37, **3 positives**):

| metric | registered m=4 | robustness m=3 |
|---|---|---|
| M1 `AUROC(u)` | **0.520** | 0.495 |
| M2 `AUROC(−c)` | **0.647** | 0.647 |
| M3 `u − (−c)` | −0.127, CI [−0.50, 0.37], p=0.75 | −0.152 |

**Per the <10-positives rule, no conclusion is drawn.** But the point
estimates run against all three predictions: `u` is at chance, `c` is *above*
chance, and the silent-failure rate did not rise with pressure.

## The correction

§9 said the Goodhart signature had been "measured": `c` below chance at 0.406
while `u` reached 0.688. **That was over-read, and I am withdrawing it.** Both
numbers came from 3 positives scored by a critic with three output values,
on trajectories that had not actually been selected. Against a critic with
no resolution, `−c` at 0.41 is noise, and `u` looking good *relative to it* is
not evidence for `u`. With a functioning critic and genuine selection, `c`
does modestly discriminate (0.65) and `u` does not (0.52). The swing from
0.69 to 0.52 is entirely within what three positives can produce, which is
exactly why the pre-registration forbids concluding from it in either
direction — and it would have forbidden §9's framing too, had it existed
then.

The silent failures themselves: pilot 2 had {HumanEval/101, HumanEval/76, HumanEval/91}; 3b has
{HumanEval/101, HumanEval/50, HumanEval/76}; overlap {HumanEval/101, HumanEval/76}.

## What the three pilots establish, stated conservatively

1. **The instrument now works end to end** with a real reward-model critic,
   a real reward-model panel, real executed gold, and selection that
   actually selects. That was not true before today.
2. **On this task family the silent-failure rate is ~8% and flat in n.** Forty
   tasks yield three positives at any pressure tried. HumanEval+ cannot
   supply the positives the primary analysis needs; ~120 tasks would be
   required for ten, and even that is thin.
3. **A functioning critic improves the selected set** (8 → 6 failures). The
   first-order effect of selection is the expected one. The winner's-curse
   *concentration* effect — P3 — was not visible here. It may need harder
   tasks, stronger pressure, or both; it is not refuted at n=3.
4. **No evidence for H1 yet, and no evidence against it.** Three positives
   cannot carry either claim. Everything before pilot 3b that looked like
   evidence was an artifact of a critic that could not rank.

## What changes for the design

- `c_sel` must be a true RM. This is now demonstrated, not argued: a coarse
  judge silently turns best-of-n into first-draw.
- Positives in the visible-pass cell are the scarce resource on HumanEval+.
  SWE-bench (24–44% failure) is the fix for supply; the 12-repo clustering
  bound (§8) is the price.
- The pre-registration stands as written, amendment included. The next run
  that can test P1–P3 is one with ≥10 silent failures, which this task family
  will not produce at n=40.


---

# 11. SWE-bench smoke test (2026-09-23) — pipeline validated, 7B agent is not

Profile B ran end to end on a real instance: mini-swe-agent inside the x86
image under Rosetta, driven by the local 7B coder, trajectory captured in the
schema. The agent executed real shell commands, then looped and hit the step
limit with an empty patch (elapsed=109s, 30 calls, 3 distinct
commands out of 30).

A 14B local model was then tried on the same instance under the same
limits: it found the exact validation site the fix needs, then echoed its
conclusion 22 times without editing a file. Nine distinct commands, still no
patch. Local models up to 14B are exhausted on this host.

This is the expected result and it settles the last open question about
where the cost lives: **on SWE-bench, generation is capability-bound.** A 7B
model cannot drive the scaffold; every public mini-swe-agent result uses a
frontier model. Scoring and gold are local and cheap. The confirmatory run
therefore needs a capable generation model — via API — with selection,
audit and gold staying on this machine. That is the architecture, and it is
now the only remaining decision that costs money.
