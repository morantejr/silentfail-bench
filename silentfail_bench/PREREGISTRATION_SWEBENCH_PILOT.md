# Pre-registration — SWE-bench Verified Stage-0 pilot (50 instances × 4 candidates)

**Registered:** 2026-09-23 19:35, during generation, **before any gold has
been computed** (gold runs in a later stage; no committed patch has been
graded). Code state: the commit that introduces this file.
Config: `configs/mini_swe_openai.yaml` + `configs/anchor_free_rm4_n16.yaml`
(panel), instances: `configs/pilot_instances_swebench50.txt`.

## What this run is for

This is the **Stage-0 pilot the proposal specifies** (§L163): it measures the
quantities the design needs and produces a go/no-go. It is **not** the
confirmatory test of H1. Anything below labelled *measurement* is reported as
a number with its count; anything labelled *prediction* can fail.

## Fixed design

| item | value |
|---|---|
| tasks | 50 SWE-bench Verified instances, seed 20260922 (9 repos; django 23) |
| agent | mini-swe-agent 2.4.6, `gpt-5-mini` via OpenAI, native tool calls, `step_limit=40`, cost cap $0.25/trajectory |
| candidates | n = 4 per instance, seeds 1000–1003 |
| selection critic `c` | `Skywork/Skywork-Reward-V2-Qwen3-4B`, scalar reward on (problem_statement, trajectory text); argmax committed |
| audit panel | 4 true RMs: GRM-gemma2-2B, GRM-Llama3.2-3B, Skywork-V2-Llama-3.2-1B, Skywork-V2-Qwen3-1.7B |
| `u` | sample std of rank-normalized audit scores; rank norms fit on **train repos only** |
| gold `g` | official `swebench` harness on the committed patch; `g=1` iff resolved |
| split | 70/30 **by repository**, SHA-256 bucket, seed 20260922 |

## Population and positive class

- **Analysis population:** committed trajectories whose agent **exited with
  `Submitted` and a non-empty patch**. This is the SWE-bench analogue of the
  visible-pass stratum: the agent believed it was finished. Trajectories that
  hit the step limit or submitted nothing are **excluded** and their rate is
  reported as a measurement, not silently dropped.
- **Positive class:** `g = 0` within the population — the agent declared
  success and the tests disagree. ("Silent failure.")

## Measurements (report all, with counts)

- M0a exclusion rate: `LimitsExceeded` / empty-patch share of the 200.
- M0b failure rate among committed trajectories, with a 95% CI.
- M0c silent-failure rate within the population, with a 95% CI.
- M0d ICC of failure by repository (one-way ANOVA), **with the caveat that
  9 repos is too few for a stable estimate**; compared against the public
  0.033–0.053.
- M0e per-candidate cost and wall-clock.

## Pre-registered metrics within the population

- **M1** `AUROC(u)` for silent failure.
- **M2** `AUROC(−c)` for silent failure.
- **M3** M1 − M2 with a **repo-clustered** bootstrap CI (4,000 resamples),
  one-sided p for M3 > 0.
- Confirmatory logistic `ΔAUROC` **only if the test split holds ≥ 10
  positives**; otherwise its absence is stated.

## Predictions

- **P1 (H1):** M1 > 0.5.
- **P2 (Goodhart):** M2 ≤ 0.5.
- **P3 (supply):** the silent-failure rate within the population exceeds the
  8% observed on HumanEval+ (pilots 2–3b).

## Go / no-go for a confirmatory run (the pilot's actual output)

GO requires all of: (i) ≥ 10 silent failures projected for the confirmatory
population at the planned size, from M0c; (ii) exclusion rate ≤ 40%;
(iii) `AUROC(M_c)` on held-out data, if computable, ≤ 0.90. Otherwise NO_GO
with the binding reason named.

## Reporting rules

- All 200 candidates and all 50 committed rows are reported; nothing dropped.
- With fewer than 10 positives, M1–M3 are point estimates and **no
  conclusion is drawn in either direction**.
- No stratum other than the one above is analysed as primary.
- `corr(c, u)` and the chosen-index histogram (did selection actually select?)
  are reported as watch items.

## Declared limits

- 9 repositories: the repo-clustered CI is unstable by construction and the
  cross-repo generalization claim is **not** tested here.
- `c_sel` (Qwen3-4B) and `audit_3` (Qwen3-1.7B) share a base family;
  checkpoint-level disjointness holds; M1–M3 are also reported with
  `audit_3` dropped (m=3).
- The reward models were trained on chat preferences, not code trajectories.

---

# Amendment 1 — 2026-09-24 06:22, after RM scoring, **before any gold**

**Finding that motivates it.** `RewardModel.score` truncates at 4,096 tokens
and SWE-bench trajectories run ~40k characters, so `c` and `u` as computed
score each trajectory's *prefix* (problem statement + early steps), not the
steps where the patch is written. This applies uniformly to every row and
so does not bias `c` versus `u`, but it limits what either can detect.

**What is NOT changed.** The primary analysis (population, positive class,
M0–M3, predictions, go/no-go) runs on the scores already computed and
frozen in `scores.json`. Nothing about it is altered by this amendment.

**What is added, as EXPLORATORY.** A second scoring pass, labelled
exploratory and reported separately:

- Input to every critic = the **final submitted diff** followed by the
  **last 3,000 tokens** of the trajectory (the submission window), same
  models, same rank-normalization fit on train repos only.
- Report M1–M3 on that input, alongside the primary. If the exploratory
  numbers differ materially from the primary, that is evidence about the
  *instrument* (where in a trajectory the signal lives), not evidence for or
  against H1, and will be described that way.
- It can inform the confirmatory design's scoring input; it cannot be
  promoted to primary for this pilot.

Declared with zero gold computed so it cannot be selected on the results.

---

# Amendment 2 — 2026-09-24, before any GPU rescoring

**Motivation.** §12–13 found chat-preference reward models carry ~no
information about patch correctness in either a 4,096-token prefix or a
3,000-token submission window. Two instrument limits remain untested: the
critic *class* (chat-preference vs code-trained) and the input *length*
(truncated vs full trajectory). A GPU removes both.

**What is NOT changed.** The 164 candidates, the 41 instances, the gold
labels, the 12 positives, the population definition, and the metrics M1–M3
with their repo-clustered bootstrap. The primary result (§12) stands as
registered and is not superseded by this amendment.

**What is added — EXPLORATORY, reported beside the primary.**

| | primary (§12) | Amendment 2 |
|---|---|---|
| input | 4,096-token prefix | **full trajectory** (up to 16k tokens; longer ones take head 4k + tail 12k, declared per row) |
| selection critic | Skywork-Reward-V2-Qwen3-4B | (a) **Skywork-Reward-V2-Llama-3.1-8B** — the proposal's named critic, now runnable; (b) **TIGER-Lab/AceCodeRM-7B** — trained on code with execution feedback |
| audit panel | 4 chat-preference RMs | the same 4, rescored on full input; plus AceCodeRM-7B as a fifth auditor when it is not the selector |

Two selection critics means two committed sets; each is analysed with the
panel that excludes it (disjointness preserved).

**Predictions, stated so they can fail.**
- **A2-P1:** with AceCodeRM-7B as a critic, `AUROC(−c)` on the population
  exceeds 0.6 (a code-trained critic sees correctness).
- **A2-P2:** `AUROC(u)` from the full-length panel exceeds the prefix
  panel's 0.508 by more than 0.10.
- **A2-P3:** the prefix-vs-full-length correlation of `c` exceeds 0.5 for
  the code-trained critic (a critic with a stable view of quality).

**What would count against the pivot.** All three fail → the null holds
across critic class and input length, and the negative-result paper is the
paper.

**Reporting.** Point estimates with counts; <10 positives in any committed
set → no conclusion for that set. Same script, same bootstrap, nothing else.
Declared with zero GPU scores computed.

**Amendment 2, clarification (same day, before any GPU scores).** Gold exists
only for the 34 patches the *primary* critic committed. Therefore:
(i) the **audit-only** analysis — every new critic as a detector on the
primary committed set, and `u` from the full-length panel — needs no new
gold and is the first thing reported; (ii) **re-selection** by a new critic
commits some different candidates, whose patches must be graded by the
harness before M1–M3 for that set can be computed. `needs_gold.jsonl` lists
them; nothing about those rows is looked at before the harness has run.

---

# Amendment 3 — 2026-09-24, before any windowed score exists

**Motivation.** Amendments 1–2 scored a prefix, a tail, and a head+tail. All
null. One structural objection remains: no critic ever saw the *middle* of
a trajectory, where an agent drifts into a silent failure. Two changes give
the panel a genuinely different chance without exceeding a 16 GB GPU.

**Population, positives, gold, metrics: unchanged.** Audit-only on the
primary committed set (n=34, 12 positives). No re-selection in this
amendment.

**Preprocessing (structural pruning, deterministic).** Tool outputs longer
than 24 lines are collapsed to their first 8 + last 8 lines with an
`[... N lines omitted ...]` marker. Diffs are never pruned. Applied
identically to every row before windowing.

**Windowing.** Each pruned trajectory is cut into overlapping windows of
≈3,500 tokens with 25% overlap, each prefixed by the problem statement
(≤800 tokens). The final window always ends at the trajectory's end
(patch included). At most 8 windows per trajectory; if more are needed,
8 are chosen evenly spaced including first and last. Window count is
recorded per row.

**Scores.** Every critic scores every window. Per critic and trajectory:
`mean`, `max`, `last`. Panel `u` is computed *per window* (std of
rank-normalized scores across the m=4 panel, rank norms fit on train rows'
window scores), then aggregated: **`u_max`** (primary), `u_mean`, `u_last`.

**Predictions.**
- **A3-P1 (the bar):** `AUROC(u_max) > 0.665` — the panel must beat the
  best structural covariate (patch added lines). Anything less is not a
  detector worth a confirmatory run.
- **A3-P2:** `AUROC(u_max) − 0.508 > 0.10` (gain over the prefix panel).
- **A3-P3:** the named critic's `mean`-window score beats its
  patch-in-view score (0.534) by > 0.10 — i.e. seeing the middle helps a
  single critic at all.

**What would count against the pivot.** All three fail → the panel design is
closed across prefix, tail, head+tail and multi-window inputs; the negative
result paper is the paper, with this as its strongest section.

**Reporting.** `u_max` is the maximum of noisy statistics and is biased
upward under the null; the repo-clustered bootstrap over instances handles
the paired comparison, but point estimates are reported with that caveat.
Declared with zero windowed scores computed.

**Amendment 2/3 addendum — L4 access (2026-09-24, before any L4 score).**
A 24 GB GPU makes the *originally declared* Amendment-2 input (16k tokens)
runnable. The Kaggle result at the 4k fallback stands as that instrument's
result and is not overwritten; the 16k run is reported beside it and is the
one carried forward. `TIGER-Lab/AceCodeRM-32B` (4-bit) is added to A2-P1 as
the stronger code-trained critic, with the same prediction. Amendment 3 runs
unchanged. Bars unchanged: 0.665.

---

# Amendment 4 — 2026-09-24, declared before any of the quantities below exist

**Motivation.** A skeptical-reviewer pass on draft v0.1 identified four ways
the null could be an artifact of scope rather than a property of the panel:
(a) the structural "ruler" may only be a proxy for task difficulty, and the
critics may add signal *conditional* on structure that the marginal AUROC
hides; (b) the mechanism sentence ("critics hold no stable judgment", from
cross-window r = 0.03–0.18) cannot distinguish critic noise from windows
legitimately containing different information; (c) only reward models were
tested, so "AI-critic disagreement" is broader than the evidence; (d) the
length ladder has two points. Each gets one pre-specified test. Everything
below is reported regardless of outcome; nothing is selected among
perturbations, judges, or feature sets after the fact.

**Population, gold, positive class: unchanged** (primary committed set,
n=34, 12 silent failures, repo-clustered bootstrap, 0.665 bar where a bar
applies).

## A4-a  Incremental value beyond structure and difficulty

- **Features.** `struct` = {patch_added_lines, step_count, patch_chars,
  patch_files_touched} (all read off the committed diff/transcript).
  `difficulty` = public solve rate: the fraction of six independent public
  mini-swe-agent v2.0.0 runs on SWE-bench Verified (gpt-5-mini,
  gemini-3-flash-high, deepseek-3-2-high, claude-4-5-sonnet-high,
  gpt-5-2-high, claude-4-6-opus; `SWE-bench/experiments`, fetched
  2026-09-24) that resolved the instance. None of those runs are ours.
- **Models** (ridge-IRLS logistic, standardizer fit on the training fold):
  `M_base` = struct + difficulty; `M_base+c` (primary c); `M_base+u`
  (primary u); `M_base+c+u`.
- **Evaluation.** Leave-one-repo-out (9 folds). Out-of-fold probabilities
  are pooled; AUROC on the pooled vector. In-sample AUROCs are not results.
  CI: repo-clustered bootstrap over the pooled out-of-fold predictions
  (approximation — models are not refit inside each draw; stated as such).
- **Predictions.** A4a-P1: `AUROC(M_base+u) − AUROC(M_base) ≥ 0.05` → u
  carries signal conditional on structure (a genuine, if modest, positive
  finding that must be reported). `≤ 0.02` → no incremental value; the
  ruler claim becomes "no incremental value," its defensible form. A4a-P2:
  the same for c. Also reported: AUROC(difficulty) alone, AUROC(M_base).
- **Overfit guard.** 34 rows, 5–7 features: out-of-fold only.

## A4-b  Test–retest stability of window scores

- **Input.** The Amendment 3 pruned windows (same `prune`/windowing code) on
  the 34 committed trajectories.
- **Perturbations, content-preserving, fixed now.**
  `P_fmt`: role headers re-cased (`## user` → `## User`, etc.), the pruning
  marker reworded (`[... N lines omitted ...]` → `[N lines elided]`), and one
  neutral line `(trajectory excerpt)` prepended. Zero information change.
  `P_shift`: window start moved by +128 tokens at the same length (−128 for
  a window that would run past the end; for a trajectory that fits in one
  window, the first 128 tokens are dropped). ≈3.7% content change.
- **Statistics, per critic.** Spearman r between original and `P_fmt`
  scores over all windows (`r_fmt`); same for `P_shift` (`r_shift`);
  `r_adj` between adjacent windows (25% overlap); `r_far` between first and
  last window of the same trajectory. The same four for per-window panel u
  (rank norms fit on train rows' original window scores).
- **Reading, fixed now.** `r_fmt ≥ 0.8` and `r_shift ≥ 0.7` → critics are
  stable to surface form; the low cross-window correlation reflects
  content, and the draft's mechanism sentence is **withdrawn** and replaced
  with "critic scores respond to window content; the panel's aggregation is
  what fails." `r_fmt < 0.5` → the instability claim stands. Between →
  reported as inconclusive. Conclusions are per critic where classes differ.
- **Compute.** ≤4B critics locally (MPS); 8B and AceCodeRM-7B on the L4
  (stage 5). Same script, same perturbations.

## A4-c  LLM-as-judge classes

- **Judges.** `gpt-5`, `gpt-5-mini`, `gpt-5-nano` (OpenAI API). One
  vendor: this is labelled a **single-family LLM judge panel**, never a
  "frontier panel". `gpt-5-mini` is also the agent model — a
  self-evaluation confound, flagged; `gpt-5` and `gpt-5-nano` are not the
  agent. Cross-family judges are a stated extension, not run here.
- **Conditions.** `J1` (patch-in-view): problem statement + final diff.
  `J2` (execution-aware): J1 + the pruned last ≈6,000 tokens (24,000
  characters) of the trajectory, which contain the agent's own test runs
  when it ran any. No judge sees gold tests or the reference patch.
- **Call.** One per (judge, condition, trajectory); default reasoning
  effort; no sampling parameters. Output JSON
  `{"p_resolved": 0–100, "reason": ≤40 words}`. Detection score =
  `100 − p_resolved`. Responses cached; a parse failure is retried once,
  then recorded as missing (never imputed as a score).
- **Metrics.** Per judge AUROC; panel `u_LLM` = std of rank-normalized
  (train-fit) scores across the three judges; AUROC(`u_LLM`). Paired vs
  primary −c by repo-clustered bootstrap. Bars: 0.665 for every quantity.
- **Predictions.** A4c-P1: best single judge in J1 > 0.665. A4c-P2:
  `u_LLM` (J1) > 0.665. A4c-P3: J2 − J1 gain > 0.10 for the best judge.
- **Readings.** P1 or P2 pass → the paper's claim narrows to *reward-model*
  critics. P3 passes while P1 fails → the useful safeguard is the agent's
  own execution feedback, not a critic panel. All fail → "AI-critic
  disagreement, within the classes tested".
- **Cost cap.** $10 of the OpenAI credits (~204 calls).

## A4-d  Length ladder third point (L4 stage 5)

An 8k point is added for all seven critics (same code, `--max-tokens 8000`)
so the ladder is 4k / 8k / 16k; per-critic paired differences with the
clustered bootstrap. The 16k run remains the Amendment 2 result. No
prediction beyond monotonicity reporting. `AceCodeRM-32B`: did not load on
24 GB (OOM twice); reported as not run.

## A4-e  Replication on a wide repository pool — designed, not run

The binding constraint on Verified is 12 repositories (django 46%), which
caps effective n regardless of instance count (paired u-vs-ruler Δ=0.10 has
power ≈0.3 at any n). Design: SWE-smith (or SWE-rebench) instances sampled
one per repository across ≥40 repositories, same agent, scaffold, k=4,
critic, panel, gold-by-execution. Stopping rule: stop when the repo-
clustered 95% CI on AUROC(u) has half-width ≤ 0.08, or when 40 repositories
have each contributed ≥ 3 usable trajectories, whichever first; never on a
favourable point estimate. Same bars. Estimated cost ≈ $5 generation +
container gold. Requires Docker again on this machine. Runs only on
explicit decision.

---

# Amendment 4e (revised) — wide-repository replication on SWE-bench Multilingual — 2026-09-25, before any trajectory exists

**Why revised.** 4e named SWE-smith / SWE-rebench. Neither is gradable by the
official `swebench` 5.x harness (their repositories have no harness specs),
and SWE-smith needs its own Docker-based harness, which this machine no
longer has. **SWE-bench Multilingual** (300 instances, **41 repositories**,
9 languages; `swe-bench/SWE-Bench_Multilingual`) is a self-describing 5.x
dataset with official images and eval scripts, supported by mini-swe-agent's
batch runner. It attacks the binding constraint directly: 41 independent
repository clusters against Verified's 12. It also changes the language mix,
which is a generalization axis, not a confound.

**Infrastructure change, declared.** No Colima. Agent environments run in
Modal sandboxes (`environment_class: swerex_modal`); the LLM calls stay
local through the same OpenAI key. Gold runs through `modal/gold_modal.py`,
which mirrors `swebench.harness.run_evaluation.run_instance` (5.0.2) step
for step in a Modal sandbox built from the dataset's own `image`, and grades
with the harness's own `get_eval_report`. **Precondition:** before any model
patch is graded, the runner must pass an oracle pair on Verified (reference
patch resolves; empty patch does not) and a reference-patch oracle on ≥ 3
Multilingual instances spanning ≥ 3 languages.

**Sampling.** Fixed seed 20260922. From each of the 41 repositories, up to 3
instances drawn uniformly without replacement (repositories with fewer take
all) → ≈115 instances. `k = 4`, `gpt-5-mini`, 40-step limit, same
`swebench.yaml` base config, same selection critic
(`Skywork-Reward-V2-Qwen3-4B`) and the same m = 4 audit panel as the primary.
Cost cap: $15 OpenAI, $15 Modal. Instances whose gold errors are excluded
(`gold_error`), as in the pilot.

**Population, positive class, metrics, splits: as the primary.** Rank norms
fit on train repositories only (70/30 by repository, same salt).

**Inference floor.** Fewer than **20** positives → no conclusion in either
direction; the run is reported descriptively only.

**Predictions (bars fixed now).**
- **R-P1 (null replication).** Primary-protocol panel `u` (4k prefix):
  `AUROC(u) < 0.60` and its repo-clustered 95% CI includes 0.50.
- **R-P2 (the marginal positive, confirmatory).** Amendment-3 protocol
  (same pruning, same windows) `AUROC(u_max) > 0.60` **and** the CI lower
  bound `> 0.50` **and** the within-repository permutation `p < 0.05`.
- **R-P3 (the ruler).** `AUROC(u_max) > AUROC(lines added)` on the same
  rows (point estimate; paired CI reported).
- Reported alongside, no prediction: every single critic at 4k and 16k;
  the LLM-judge panel (J1) if credits allow; incremental-value model.

**Readings.** R-P2 passes → windowed disagreement is a real, weak detector
and the paper's end state moves to "narrower positive"; the confirmatory
study the proposal promised is warranted for `u_max`, not for `u`. R-P2
fails with R-P1 passing → the Verified `u_max` result was a twelve-positive
fluctuation; end state "null survives", now with 41 repositories. R-P1
fails (u works here) → the Verified null does not generalize and the paper
is rewritten around the difference.

**Stopping rule.** Unchanged: stop when the repo-clustered 95% CI on the
primary quantity has half-width ≤ 0.08, or when 40 repositories have each
contributed ≥ 3 usable trajectories, whichever first; never on a
favourable point estimate. Since the sample is fixed at ≈115 instances, the
rule decides only whether a second draw is warranted.

**Amendment 4e — declared deviation, 2026-09-25 08:40, before any critic score exists.**
The $15 OpenAI cap was exceeded: recorded trajectory cost reached $16.76 at
420/456 trajectories because the account's rate limit (500 RPM / 500k TPM)
killed 138 partially-run trajectories overnight, whose spend is not even in
that figure. Stopping at 420 would drop the harder, longer instances first
and bias the failure rate, so the remaining 36 were generated (≈$1.30) and
the cap is recorded as ≈$18. Instances that still lack a candidate after
that pass are excluded before any scoring, by name, in the readout.
