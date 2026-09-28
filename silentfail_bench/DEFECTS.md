# Stage-0 defect log — 2026-09-22

Four defects found while preparing the free-compute pilot. All were verified by
execution, not inspection, and all now have regression tests in
[`tests/test_defect_regressions.py`](tests/test_defect_regressions.py).

---

### #1 — Repo grouping truncated hyphenated repo names  *(corrupts the primary design)*

`runner.repo_from_instance` split on the FIRST hyphen, so any repo whose name
contains one lost its identity:

| instance_id | was | now |
|---|---|---|
| `scikit-learn__scikit-learn-10297` | `scikit` | `scikit-learn__scikit-learn` |
| `sphinx-doc__sphinx-8721` | `sphinx` | `sphinx-doc__sphinx` |
| `pylint-dev__pylint-6506` | `pylint` | `pylint-dev__pylint` |

Since the held-out split groups by `repo`, mangled keys mean "held-out repos"
were not reliably held out. Fixed in [`identifiers.py`](src/silentfail/identifiers.py):
only a trailing `-<digits>` is treated as the issue number.

### #2 — Splits were not reproducible across runs

`pilot.py` used builtin `hash()`, which is salted per process
(`PYTHONHASHSEED`). Three consecutive runs put `django__django` in buckets
3, 5 and 6. Any rerun reshuffled train/test. Replaced with a SHA-256 bucket
(`assign_split`), verified identical across processes and calibrated to the
70/30 target (70.1% over 2000 keys).

### #3 — IRLS diverged under separation and unscaled controls

`length_chars` runs to 1e3–1e5. Fed unscaled to IRLS this overflowed the
Hessian (`divide by zero` / `overflow encountered in matmul`), and quasi-
separation at pilot n drove the intercept to ±inf, producing NaN AUROC.
Fixed with a TRAIN-ONLY standardizer stored on `LogisticModel` (also closes a
train→test leakage vector), ridge on the intercept, and step-halving that
accepts only finite, non-worsening steps.

### #4 — The power function was invalid  *(it gated the go/no-go)*

`pilot.simulate_power` defined power as P(ΔAUROC > 0) **in the test sample**.
That statistic is ~0.5 under the true null, so it reported "power" where there
was no signal whatsoever:

| | old, under the TRUE null (δ=0) | new |
|---|---|---|
| n=100 / 120 groups | **0.390** | 0.010 |
| n=200 / 240 groups | **0.395** | 0.010 |
| n=400 | **0.435** | — |

A second bug cut the other way: sims skipped by the `if too few fails:
continue` guard stayed in the `wins / n_sims` denominator.

Replacement in [`stats.py`](src/silentfail/stats.py): the two AUROCs are
computed on the same held-out rows, so the comparison is **paired and
correlated**; inference is a **cluster bootstrap over groups** (repo /
problem_id) with the fitted models held fixed. Skipped sims are reported
separately (`n_sims_valid` / `n_sims_skipped`).

**The new test is conservative**: actual rejection ≈0.010 against a nominal
α=0.05. Rejections are therefore trustworthy; required-n figures are
pessimistic.

### #5 (lower severity) — Rank-norm leakage

`runner.run_pilot_mock` fit rank-normalizations on the first half of usable
rows, which overlaps the test split assigned later in `pilot_summary`. Test
rows leaked into the normalization that defines `u`, contradicting the
`rank_norm_fit: train_split_only` freeze. Both the CI runner and
[`runner_free.py`](src/silentfail/runner_free.py) now fit on train groups only.

---

## Harness bugs found by control checks (profile A)

Caught because **the EvalPlus reference solutions are run through the gold
harness first** — if the known-correct answer fails, the oracle is broken.

1. **`-S` hid numpy.** The sandboxed interpreter ran with `-S`, but HumanEval+
   gold tests `import numpy`. Every gold test errored, so the first smoke run
   scored `g=0` on all trajectories — a 100% failure rate that looked like a
   finding. Fixed by re-injecting exactly this interpreter's `site-packages`.
2. **Locale encoding.** `Path.write_text` without an explicit encoding produced
   `SyntaxError: Non-UTF-8 code` on tests carrying non-ASCII literals
   (`HumanEval/161`). Now written as UTF-8 with a coding declaration.
3. **Wrong reference.** Base HumanEval canonical solutions genuinely fail the
   stricter `plus` tests on some instances (`HumanEval/122`, `/76`, `/44`) —
   that is the documented EvalPlus finding, not a bug. The control now uses
   EvalPlus's own canonical solution.

After these fixes: **163/164** reference solutions pass gold. The one holdout,
`HumanEval/32` (`find_zero`, numerically unstable), is excluded as
`gold_harness_invalid` rather than counted as an agent failure.

## Sandbox posture (verified, not assumed)

`sandbox-exec` deny-by-default, confirmed by adversarial test: network egress
blocked, DNS blocked, writes outside the temp dir blocked, reads of
`~/Library/CloudStorage`, `~/Desktop`, `~/.ssh` blocked (`PermissionError`),
CPU-limit and wall-clock kills both fire.

**Verified limitation:** macOS silently ignores `RLIMIT_AS` — inside the
sandbox `getrlimit` still reports unlimited and an 800 MB allocation succeeds
under a nominal 512 MB cap. Memory is NOT capped on this platform; the CPU and
wall-clock limits are the real controls.

---

## Endpoint power comparison — a hypothesis that did not survive testing

It seemed likely that a budgeted-escalation endpoint (precision@k) would be
better powered than global AUROC at fixed n, *and* more decision-relevant.
Measured at n=400 across 120 sims with the same clustered test
([`tools/endpoint_power_comparison.py`](tools/endpoint_power_comparison.py)):

| endpoint | null rejection | power (u=0.5) | power (u=1.0) |
|---|---|---|---|
| **global AUROC** | 0.008 | **0.258** | **0.850** |
| precision@10% | 0.000 | 0.000 | 0.175 |
| precision@20% | 0.000 | 0.033 | 0.342 |
| partial AUROC (FPR≤20%) | 0.025 | 0.192 | 0.567 |

Global AUROC wins decisively. precision@k is a coarse step function — only the
top-k ordering moves it, so most bootstrap resamples return an identical value
and sensitivity collapses. Identical numbers before and after the IRLS trust
region, so the ordering is not an artifact of the numerics.

**Consequence:** switching the primary endpoint to precision@k would make the
power problem *worse*. ΔAUROC stays primary; precision@k and lift@k are worth
logging as secondary, decision-relevant numbers for the `BUSINESS.md` framing,
but they cannot rescue n=200.

## IRLS instability, second pass

The trust region was added after the first fix still warned. The design matrix
was perfectly scaled (columns within ±3.5), so the cause was not bad input: once
`z` clips, `W ≈ 1e-6` everywhere, and with a near-free intercept the Newton step
runs to `grad/1e-6`, compounding toward ~1e300. Ridging the intercept at the
same strength plus a trust region on `|delta|_inf` took it from warning on most
fits to **1 in 600**, with null calibration unchanged (0.013) and
power(u=1.5, n=160) = 0.82.

---

## Reward-model throughput on Apple Silicon — a claim corrected twice

First measured at **58–116 s/score**, which suggested true RMs were an
overnight job. That was **GPU contention**: Ollama was saturating Metal for the
concurrently-running pilot. Re-measured on an idle GPU it looked like
**0.3 s/score**, which suggested the opposite. Both readings over-generalized
from one small model. Measured properly, across models, on the pilot's real
trajectory lengths (255–4124 chars), warm:

| model | size | s/score | n=2000 × 1 critic |
|---|---|---|---|
| Skywork-Reward-V2-Llama-3.2-1B | 1B | 0.75 | 25 min |
| GRM-gemma2-2B | 2B | 0.72 | 24 min |
| Skywork-Reward-V2-Qwen3-1.7B | 1.7B | 1.14 | 38 min |
| GRM-Llama3.2-3B | 3B | 1.75 | 58 min |
| **Skywork-Reward-V2-Llama-3.1-8B** | 8B | **133.98** | **74 hours** |

The rule is **not** "reward models are fast". It is **"≤3B is fast, 8B
thrashes"** — 16 GB of fp16 weights on a 24 GB machine pages rather than
computes. Two lessons worth keeping:

1. Never benchmark while another process holds the GPU.
2. Never generalize a throughput number from one model size on a synthetic
   input; the 8B is ~180× slower than the 2B, not ~4× as parameter count
   would suggest.

**Consequence:** a true-RM panel is viable locally at ≤3B (~2.4 h for n=2000
across four models), which closes the construct gap for free. Scoring is cheap;
**generation is the bottleneck**. That favors generation via an inference API
with scoring on local RMs — an API cannot serve RM heads at all.

Also unusable: `internlm/internlm2-1_8b-reward` — its custom tokenizer fails
`convert_slow_tokenizer` on transformers 4.57.

---

## Run crash on the 4-RM panel pilot (2026-09-23) — two runner defects

One selection-critic call ran the 8B judge past its 4,096-token context
(`n_tokens = 2886, truncated = 1`); Ollama hung for the full 600 s timeout and
returned 500. The runner had no guard, so one bad HTTP call aborted the run
**after generation had completed** — and the candidates existed only in
memory, so 17 minutes of generation were lost.

1. **No checkpoint between stages.** Generation is the expensive stage and
   was never persisted before scoring started. Now each task's candidates are
   appended to `candidates_raw.jsonl` the moment they finish, and a rerun
   resumes from that file instead of regenerating.
2. **No fault tolerance in scoring.** `judge_score` now never raises: a
   transport error, timeout or unparseable reply returns NaN, which the runner
   already maps to `judge_unparseable_score` exclusion. The prompt cap dropped
   from 12,000 to 6,000 chars (~1.5k tokens) so the judge stays inside its
   context window.

The lesson generalizes: the expensive stage must be durable before the cheap
stage begins, and the cheap stage must degrade per-item, never per-run.

---

## A coarse selection critic silently disables best-of-n (2026-09-23)

The LLM-judge `c_sel` asked for "a number from 0 to 10" and, over 640
candidates, returned only **{5, 6, 8}** — 55% of them exactly 8. On 21 of 40
tasks all 16 candidates tied at the max; argmax then took index 0. Raising
selection pressure from n=4 to n=16 changed the committed trajectory on
**1 of 40 tasks**.

| | n=4 | n=16 |
|---|---|---|
| committed trajectory identical to n=4 run | — | **39/40** |
| selected from the 12 new candidates | — | 1/40 |
| silent failures | 3 | 3 |

So pilots 1–3 were not selection-conditional in any meaningful sense: "best
of n" was "first draw". This is the sharpest case for the proposal's own
specification — `c_sel` must be a continuous scalar reward model, not a chat
model emitting integers — and it was found only because the candidate pool
was checkpointed with full text and could be inspected after the fact.

Fix: `scripts/reselect_from_candidates.py` re-runs selection over the
existing pool with a true RM. Pre-registration Amendment 1 records the swap.

## Replication infrastructure (2026-09-25): swe-rex Modal image layer stalled generation
**Symptom.** Amendment 4e generation on Modal produced 28 trajectories, then
nothing for an hour with every worker in "Starting modal sandbox".
**Cause.** swe-rex's Modal deployment builds a per-instance image layer
(`apt-get update && apt-get install pipx`) on top of every SWE-bench image
before it can start. On the Multilingual images (Ubuntu 22.04, no Python)
that layer takes 10+ minutes each; 114 of them through Modal's limited
build concurrency is effectively a stall. Verified: Modal's `add_python`
layer on the same image builds in 5 s; the apt layer had not finished
`apt-get update` after 5 minutes.
**Fix.** `swerex/deployment/modal.py` patched in the venv (copy kept at
`modal/swerex_modal_patched.py.txt`): registry images get
`add_python="3.11"`, no apt layer; `swe-rex` is pip-installed into that
python at sandbox start (~10 s). Sandboxes then start in 6–90 s.
**Also.** Killing the orchestrator does not stop its Modal sandboxes; they
run to `runtime_timeout` and hold concurrency. Stop the `swe-rex` app
explicitly (`modal app stop <id> -y`) before restarting.

## Amendment 7's paused checkpoint was invisible to its own runner (found 2026-09-27, before resume)

**Symptom (latent, not yet triggered).** The 100 trajectories generated before
Amendment 7 was paused sat at `data/agent_axis/cand0/`. The runner's generate
stage writes to `out/traj/cand{c}` and its bundle stage globs
`out.glob("traj/cand*/*/*.traj.json")`. Against `--out data/agent_axis` that
glob returned **0 files** while 100 existed one level up.

**What would have happened on 1 October.** `gen` would have created a fresh
`traj/cand0/` with no `preds.json`, so mini-swe-agent's skip logic
(`--redo-existing` defaults off; it skips instance ids already in
`preds.json`) would have had nothing to skip and regenerated all 114 from
scratch. The original 100 would have been silently ignored by `bundle` as
well. Cost of the silent redo: ~$0.4 of OpenAI plus ~$2.4 of Modal sandboxes,
and, worse, no error message anywhere.

**Why it was there.** The validated replication run uses
`data/replication_multilingual/traj/cand{0..3}/`. The agent-axis run was
launched with an output path one level shallower, and because generation was
stopped for budget rather than by a failure, nothing ever exercised the
bundle stage against it.

**Fix.** `mv data/agent_axis/cand0 data/agent_axis/traj/cand0`. Verified after
the move: the runner's glob returns 100, `preds.json` travelled with the
directory and still holds 100 entries, and the layout now matches the
validated run. Resume will skip those 100 and generate the remaining 14 of
cand0 before starting cand1.

**Class.** Same family as the two earlier silent no-ops recorded above: a path
or skip predicate that disagrees with where the data actually is, failing by
doing nothing rather than by raising. The check that catches it is to assert a
non-zero count from the consumer's own glob, not to trust that files exist.

## The population defect that manufactured a paper-reversing result (found 2026-09-27, in a deliberate error hunt)

**Symptom.** Amendment 5's registered leave-one-repository-out logistic reported
`ΔAUROC = +0.375 [+0.214, +0.522]` on Multilingual — large, apparently
significant, and enough to reverse the paper's conclusion — against a baseline
`M_c` that had collapsed to **0.150** out of fold while the raw `AUROC(-c)` on
the same rows was 0.591. We diagnosed it as a degenerate-baseline artifact of
the logistic estimator, reported the unfitted estimator's `+0.029` instead, and
cited the episode in Amendment 7a as the reason to demote logistic estimators.

**Actual cause, and it was not the estimator.** The population was wrong.
Amendment 5 re-selects a new committed candidate with `gpt-5`, then filtered the
analysis population with `rows[i]["visible_pass"]` — the agent-declared-complete
flag of the candidate the **4B critic** committed, which for three-quarters of
instances is a different trajectory. The scoring bundle carries `visible_pass`
and `excluded` per candidate, so the correct field was available. On Multilingual
the wrong field silently dropped **16 instances** whose gpt-5-committed
trajectory was complete, non-empty and graded, purely because the 4B critic's
pick had been empty. On the corrected population the same estimator reports
`+0.012` with a baseline of **0.575** against a raw 0.671. The anomaly was the
defect; the estimator was fine.

**Everything it moved.** Multilingual: 85 → 101 rows, 32 → 44 failures, 39 → 40
repositories, committed `AUROC(-c)` 0.591 → 0.671, `AUROC(u)` 0.614 → 0.591,
endpoint by three estimators +0.375/+0.195/+0.029 → +0.012/−0.024/−0.018. Two
registered verdicts changed: **S-P2 moved from pass to fail** (gap 0.115 → 0.035
against a bar of 0.10), and **S-P3 now fails on its first condition** because
`AUROC(u)` fell below the registered 0.60, so the paper no longer has to overrule
a criterion that was literally met.

**Scope, checked rather than assumed.** Verified is unaffected: both readings
give the identical population (32 rows, 9 failures) and identical statistics.
Amendment 2's re-selection rows are unaffected on Verified, verified by
reimplementing them and reproducing the published values exactly (8B
re-selection u 0.455, −c 0.602, delta −0.148, CI [−0.486, 0.260]).

**How it was found, and how it nearly was not.** Found by asking whether a
per-candidate `visible_pass` existed in the bundle, then diffing the two
readings. On the way to confirming it I produced a *false* alarm of my own: I
recomputed the endpoint with the baseline as `-c` when `c` in this readout is
already a detection score (`c = 100 - p_resolved`, higher = worse), which
inverted the baseline to 0.412 and inflated the delta to +0.122. Negating an
already-negated score is now stated in the corrected script's docstring.

**Second defect, same investigation.** The `endpoint` block in
`data/selector_axis/prereg_readout_amendment5.json`, including the `+0.029` the
paper quoted and its `"BASELINE DEGENERATE"` annotations, was **not produced by
any committed script**. A headline number with no reproducible provenance is how
a defect like this survives. Fixed by
`scripts/readout_amendment5_corrected.py`, which regenerates both populations
side by side so the correction stays auditable.

**Class.** A filter keyed to the wrong entity — the same family as the two
silent no-ops and the checkpoint-path defect recorded above. It failed by
quietly analysing a subset, not by raising. The general check: when an analysis
re-selects a different unit, every per-unit flag in the filter must be re-read
from the newly selected unit, and a test should assert the population size
against an independent count.

### #8 — Figure caption contradicted its own plotted points; short paper drifted after Amendment 5a *(reporting)*

Found in the 2026-09-27 PI audit against HEAD after Amendment 5a.

1. `fig:ladder` caption said "No circle crosses it" while `make_figures.py`
   plotted pruned-window $u_{\max}=0.686$ against the ruler at 0.665. The table
   caption already stated that only the last row clears the bar; the figure
   caption did not. Fixed by rewriting the caption to say exactly one circle
   crosses, and by extending `check_consistency.py` to ban the old sentence.

2. `make_figures.py` plotted 9 of 13 Table-1 rows. The checker only verified
   figure→table, so the missing rows (family-disjoint robustness, code-trained
   16k, LLM judges, execution-aware) were invisible. Fixed by plotting every
   table pair and adding the reverse check.

3. `data/readout_tier15.json` still said the widest sweep upper bound was
   +0.45 from the 5-RM code-trained row; the paper correctly quotes +0.63 from
   the 16k code-trained row. JSON updated; checker now asserts 0.63.

4. `main_short.tex` still said "about two-thirds" for the best detector and
   still described Amendment 5 as "inconclusive rather than null" in
   Limitations after Amendment 5a had moved S-P3 to a clean first-condition
   failure. Synced to the long paper.

5. Long-paper prose claimed "nothing built from disagreement is above chance
   on either benchmark" while $u_{\max}=0.686$ and several point estimates sit
   above 0.5. Narrowed to: aside from the unreplicated pilot $u_{\max}$, no
   disagreement statistic clears chance with a CI excluding 0.5.

**Class.** Caption/prose not derived from the plotted or tabulated values;
checker asymmetry; short-paper drift after a registered verdict moved.

6. `data/*/*` also swallowed `data/*/gpu_scores/scores*.json`, so Fig.~2's input was local-only. Re-included the directory and score files; committed the Kaggle/GPU score JSONs the figures need.

### #9 — E3 train-fold misalignment; Fig8 Verified hardcodes; ECE `jg and`; defective A5 script still runnable *(code / reporting)*

Found in the 2026-09-27 PI code-correctness audit against HEAD after #8.

1. `scripts/readout_pi_batch2.py` built E3 rank-norm fits with
   `[rows[i]["split"]=="train" for i in pop][:len(us)]` — a prefix of the
   *unfiltered* population's split flags — after skipping rows that lacked
   enough candidates or panel scores. Train folds were therefore misaligned
   with the filtered `us` list. On Verified best-of-4 this moved
   $\mathrm{AUROC}(u)$ from the published exploratory 0.444 to 0.462 after
   the fix. The paper never quoted either point estimate (only the
   exploratory existence of the $n{=}2$ vs $n{=}4$ check). Regenerated
   `data/readout_pi_batch2.json`; checker now pins 0.462 and bans stale 0.444.

2. `paper/check_consistency.py` loaded ECE via `jg and json.loads(...)`.
   Because `jg` is a non-empty dict it always short-circuited to the truthy
   dict rather than the intended batch2 load — except that the subsequent
   `[b]["CAL_judge_calibration"]...` indexing happened to work only when the
   expression was parsed as `(jg and json.loads(...))[b]...`, which in Python
   evaluates `json.loads` when `jg` is truthy. The idiom was still a landmine
   (swap `jg` for an empty container and calibration checks go silent).
   Replaced with an explicit `_batch2 = json.loads(...)` binding.

3. `paper/make_figures.py` hard-coded Fig.~8's Verified AUROCs (and a dead
   CI that was never drawn) while Multilingual already read from JSON. Point
   estimates now load from the same committed readouts the paper quotes.

4. `scripts/prereg_readout_amendment5.py` still wrote the defective 85-row
   population if re-run. It now exits 2 unless `--write-defective-archive` is
   passed, and its docstring points at `readout_amendment5_corrected.py`.

**Class.** Filter/fold alignment keyed to the wrong index stream; defensive
coding debt left next to a known population defect; figure values not wired
to their JSON sources.
