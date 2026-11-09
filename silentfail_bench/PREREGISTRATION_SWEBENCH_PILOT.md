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

---

# Amendment 4f — unselected-candidate control — 2026-09-25, before any random candidate is graded

**Motivation.** The paper reports the silent-failure rate among *committed*
trajectories but has no unselected comparison, so it cannot say whether
selection concentrated failure or merely occurred. This control grades one
candidate per instance that the selector did **not** commit.

**Design.** For every instance in both benchmarks (41 Verified, 114
Multilingual), draw one non-committed candidate with a non-empty patch,
uniformly, seed 20260925. Grade it with the same harness path as the
committed one (Modal runner; oracle pairs already passed on both datasets).
Instances with no eligible candidate are reported as such, not replaced.

**Quantities, both arms paired by instance.**
- Failure rate among random-arm rows vs committed-arm rows (paired
  difference, repo-clustered bootstrap CI).
- `AUROC(-c)` and `AUROC(u)` on the random arm, computed with the same
  critic scores used for the committed arm: Multilingual — prefix-4k scores
  (all 456 candidates were scored); Verified — the selector's own
  candidate scores for `c` (all 164 were scored in the primary run) and the
  16k L4 panel for `u` (the only panel run that scored every candidate);
  the Verified committed-arm comparison therefore uses the 16k panel value
  (0.439), and this is declared.

**Predictions (bars fixed now).**
- **F-P1 (selection concentrates failure):** committed-arm failure rate
  exceeds the random-arm rate; CI on the paired difference excludes zero.
- **F-P2 (selection changes what the critic can see):** `AUROC(-c)` on the
  random arm exceeds `AUROC(-c)` on the committed arm by > 0.10 — the
  selector separates failures better among candidates it did not pick.
- **F-P3:** `AUROC(u)` on the random arm is reported; no prediction. If it
  exceeds 0.665 while the committed arm does not, that is the range-
  restriction reading of the null and is stated as such.

**Readings.** F-P1 passes → "selection concentrates silent failures" may be
claimed. Fails → the paper says "a third to a half of committed trajectories
are silent failures" and nothing about concentration. F-P2 passes →
selection-conditional evaluation is shown to matter, which is the paper's
protocol claim, measured. Cost ≈ $10 Modal.

---

# Amendment 5 — the selector axis — 2026-09-25, declared before any strong-selector score exists

**Motivation.** Three independent reviews converge on one objection: the
selecting critic is a 4B chat reward model that barely ranks failures, so
"what does an audit signal add after selection" has not been asked in the
regime the proposal named. Amendment 4f made this concrete — on Multilingual
the selector has no signal in either arm (0.43 unselected, 0.51 committed),
so there was nothing for selection to consume. This amendment re-runs the
entire protocol with selectors that do have signal, on the candidate pools
already generated. No new trajectories.

**Selectors (each re-selects independently over the same pools).**
1. `S_judge`: **gpt-5**, problem statement + final diff, the J1 prompt
   already used for auditing, scoring *every* candidate. As a detector on
   committed rows it reached 0.72 (Verified) and 0.66 (Multilingual), so it
   is known to carry signal.
2. `S_swerm`: **SWE-RM** (`arXiv:2512.21919`) if the weights are publicly
   available and loadable; otherwise recorded as not run, with the reason.
3. `S_4b`: the primary 4B chat RM, unchanged, as the reference arm.

**Population.** For each selector: argmax over the four candidates by that
selector's score; the committed candidate defines the population, with the
same exclusions (empty patch, gold error, no visible pass). Newly committed
candidates lacking gold are graded by the Modal runner whose oracle pairs
have already passed on both datasets. The audit panel is the same m=4 chat
RMs and is disjoint from every selector by construction.

**Quantities, per selector and benchmark.**
- Failure rate of the committed arm, and of the Amendment-4f random arm
  restricted to the same instances (paired).
- `AUROC(-c)` **in the candidate pool**: within-instance, whether the
  selector's score ranks failing candidates below passing ones across all
  graded candidates available for that instance.
- `AUROC(-c)` **on the committed set**.
- `AUROC(u)` on the committed set, panel rank norms fit on training
  repositories of that selector's committed rows.
- Registered endpoint: LORO out-of-fold `ΔAUROC(M_{c+u} − M_c)`.

**Predictions (bars fixed now).**
- **S-P1 (the regime is entered):** for `S_judge`, `AUROC(-c)` in the
  candidate pool > 0.65, and the committed-arm failure rate is *below* the
  random arm's (a selector with signal picks better). If S-P1 fails, the
  selector is not strong on this population and S-P2/S-P3 are reported
  descriptively only.
- **S-P2 (selection consumes the selector's signal):** `AUROC(-c)` in the
  pool minus `AUROC(-c)` on the committed set > 0.10. This is the 0.81 → 0.61
  effect, tested on a selector that has signal, on both benchmarks.
- **S-P3 (the paper's question, confirmatory):** on the committed set of a
  selector that passed S-P1, `AUROC(u) > 0.60` **and** the registered
  endpoint `ΔAUROC > +0.05` with a repository-clustered CI excluding zero.

**Readings.** S-P3 passes → disagreement *does* carry residual signal once
the selector is strong, the null is specific to weak selectors, and the
paper is rewritten around that condition. S-P3 fails with S-P1 and S-P2
passing → the null holds in the regime the proposal named, which is the
strongest form of the result and removes the reviews' main objection.
S-P1 fails for every selector → the study cannot enter the regime with
these pools and says so.

**Inference floor.** Unchanged: fewer than 10 positives in a selector's
committed set → descriptive only for that selector.

**Cost cap.** $40 OpenAI, $30 Modal. Declared before any call.

**Amendment 5 — NOT RUN (2026-09-25).** Declared as above, then stopped by
author decision before any analysis. 42 of 620 candidate scores from
`S_judge` were written (`data/selector_axis/`, $0.83) and **were not read,
analysed, or used in any reported quantity**; `S_swerm` was never launched.
The amendment stands as a pre-registered design for a future study. No
result in this paper depends on it, and v0.5's conclusions are unchanged.

**Amendment 5 — RUN (2026-09-27), outcome.** `S_judge` (gpt-5 over all 620
candidates, $13.05) executed as declared; `S_swerm` recorded as **not run**
under the escape hatch above (122 GB of causal-LM weights, no value head, no
score token, 31-byte README — no documented scalar interface). **S-P1 passes
on both benchmarks** (pool AUROC 0.712 / 0.706). **S-P2 passes on
Multilingual** (0.706 → 0.591) and fails on Verified, which has 9 positives
and is below the inference floor. **S-P3 does not hold**: on Multilingual the
criterion as literally written is met by the estimator it names, but that
estimator's baseline is degenerate (M_c = 0.150 out of fold against a raw
0.591) and the apparent gain tracks the baseline's deficit; under the
assumption-free estimator the endpoint is +0.029 [−0.082, +0.137], and
AUROC(u) = 0.614 has a CI of [0.488, 0.736] including chance. Recorded as not
holding, on the four-estimator standard adopted in Amendment 4.

---

# Amendment 6 — capability pilot for a newer agent — 2026-09-27, before any trajectory exists

**Motivation.** The mentor asks for a more capable agent. Its feasibility turns
entirely on price: the five stronger public runs cost $0.356–$0.658 per
trajectory against our $0.040, which puts the experiment at $226–$678.
`gpt-6-luna` is listed at $0.10/$0.50 per M tokens — **cheaper than the agent
we used** — but no GPT-6 model has a published SWE-bench score, so its
capability at agentic coding is unknown. This pilot measures it before any
money is committed to a full run.

**Design.** The **same 41 SWE-bench Verified instances** as the primary pilot,
so the comparison is paired on identical tasks. `k = 1` (one attempt per
instance; this is a capability measurement, not a selection experiment), the
same mini-swe-agent scaffold, the same 40-step limit, the same harness gold.

**Quantities.** Fraction of instances where the agent submits a non-empty
patch; per-attempt failure rate under executed gold; cost per trajectory; and
the paired difference in failure rate against gpt-5-mini on the same 41
instances.

**Predictions (bars fixed now).**
- **P6-1 (it functions):** ≥ 80% of instances end with a submitted, non-empty
  patch. Below that the scaffold does not drive this model and the option is
  closed, as it was for the local 7B/14B agents in the pilot record.
- **P6-2 (it is more capable):** per-attempt failure rate is at least 10 points
  below gpt-5-mini's on the same instances.
- **P6-3 (it is cheap):** measured cost per trajectory ≤ $0.040, i.e. no more
  than the agent we used.

**Readings.** All three pass → the mentor's agent axis is affordable inside
the remaining credit and is worth proposing at full scale. P6-1 fails → record
as a non-starter with the reason. P6-2 fails → the model is cheap but not more
capable, so it does not address the objection.

**This is explicitly a capability pilot and is not inferential.** No claim
about disagreement, selection or silent failures is drawn from it; 41
instances at k=1 cannot support one. Cost cap: $5.

**Amendment 6 — RUN (2026-09-27), outcome.** All three predictions pass.
**P6-1:** 41/41 instances submitted a non-empty patch (bar ≥80%).
**P6-2:** per-attempt failure 30.0% against gpt-5-mini's 48.6% on the same 41
instances, unconditional — **+18.6 points** (bar ≥10). Our own gpt-5-mini
figure agrees with the public leaderboard run on those instances (48.8%) to
0.2 points. An earlier 33.3% figure was the unselected-arm rate, which
conditions on a non-empty patch and is not comparable.
**P6-3:** $0.0050 per trajectory against $0.0401 — **8× cheaper** (bar
≤$0.040). Total pilot cost $0.20.

*Instrument failure #7, recorded.* The first run of this amendment returned
0/41 submitted patches, which reads as "this model cannot drive the scaffold".
It was our own misconfiguration: `model_class: litellm` unconditionally sends
native function tools, which gpt-6-luna rejects on `/v1/chat/completions`;
`drop_params: true` then dropped them silently, the model replied in correct
text, and the model class — parsing only tool calls — aborted. Caught by
sending the agent's own prompt directly to the model and seeing the expected
output. Fixed with `model_class: litellm_response`.

**Consequence.** The mentor's agent axis costs **≈$27 for 212 instances**
(35 silent failures, matching the current study) using this agent, against
$376 with gemini-3-flash-high. It fits inside the remaining credit.

---

# Amendment 7 — the agent axis — 2026-09-27, declared before any trajectory exists

**Motivation.** The remaining half of the mentor's objection: the null may be
an artifact of a weak agent. Amendment 6 established that `gpt-6-luna` is
both more capable than our agent (30.0% vs 48.6% per-attempt failure on
identical instances) and 8x cheaper, which makes this testable inside the
remaining credit.

**Design — paired on instances.** The **same 114 SWE-bench Multilingual
instances** as the replication, `k = 4`, the same scaffold, the same 40-step
limit, the same m=4 audit panel, the same harness gold. The only thing that
changes is the agent. Multilingual is chosen over Verified because its 41
repositories are the axis that binds precision, and because the replication's
selector-axis result lives there.

**Two selection regimes, as in the rest of the study.**
`S_4b`: the primary 4B chat reward model, for comparability with the primary
protocol. `S_judge`: gpt-5 over all candidates, for the strong-selector
condition, run only if the first regime clears the inference floor.

**Quantities.** Silent-failure rate among committed trajectories; `AUROC(u)`;
`AUROC(-c)`; the registered endpoint under all four estimators; and the paired
difference in each against the gpt-5-mini run on these same instances.

**Predictions (bars fixed now).**
- **A7-P1 (the agent is capable here):** per-attempt failure rate on these
  instances is at least 10 points below gpt-5-mini's on the same instances.
  If this fails the agent is not more capable on Multilingual and the
  amendment is descriptive only.
- **A7-P2 (the null holds):** `AUROC(u) < 0.60` and its repository-clustered
  CI includes 0.50.
- **A7-P3 (the endpoint holds):** the assumption-free estimator of
  `ΔAUROC(M_{c+u} − M_c)` has a CI including zero.

**Readings.** A7-P2 and A7-P3 hold → the null survives a capable agent and a
capable selector, and the mentor's objection is retired in full. Either fails
→ the null is specific to weak agents, the paper's claim becomes conditional,
and a confirmatory run is warranted before any positive claim is made.

**Inference floor.** Unchanged: fewer than 10 silent failures in the committed
set → descriptive only.

**Cost, declared (original Amendment 7 estimate).** ~$2 generation and ~$9
judge on OpenAI credit; ~$7 of Modal, of which ~$3 is free-tier and the
remainder is a small overage before the 1 October reset. Cap: $15 OpenAI,
$12 Modal. **Superseded for the October resume by Amendment 7b:** four
candidate-indexed gold runs raise Modal to ~\$8.40 generation + ~\$12 gold
≈ **\$20.40**, cap **\$22** inside the \$30 credit that resets 1 October; no
paid Modal overage.

**Amendment 7 — PAUSED (2026-09-27), not analysed.** Generation stopped
partway through the first of four candidate passes: 100 of 456 trajectories
(99 with a non-empty patch, 1 step-limited). Stopped for budget, not for any
result — Modal sandboxes on Multilingual images cost $0.0236 per trajectory
against the $0.013 measured on Verified, which would have put the run ~$9
past the free tier before the 1 October reset. **No gold has been run on
these trajectories and no quantity has been computed from them.** Generation
resumes after the reset. Should the design change before then, this note
stands as the record that the first 100 trajectories existed and were not
inspected.

# Amendment 8 — failure taxonomy and label-contamination bound — 2026-09-27, declared after the confirmatory results and reported as exploratory

**Honesty note, first.** Unlike Amendments 1–7 this one is written *after* the
quantities below were computed. It is therefore exploratory in the strict
sense and carries no pre-registered thresholds, no multiplicity correction and
no power claim. It is recorded here because the project's rule is that every
analysis in the paper appears in this file, including the ones that do not
earn the protection of a prior declaration.

**Why these two.** Both are reviewer questions that the existing data can
answer for nothing. Neither can produce a positive finding for the safeguard;
both can only fail to rescue the null, which is the asymmetry a reader should
hold against them.

## E5 — what the silent failures actually are

**Question.** Does the pooled null average over failure classes, one of which
disagreement detects?

**Method.** Every committed trajectory's executed harness report is already on
disk. For Verified this is the union of the 14 batched `run_evaluation` runs
under `logs/run_evaluation/sf_*/silentfail/`, which carry full test names; for
Multilingual it is the `silentfail_replication` rows of
`data/gold_modal/replication_multilingual/report.json`, which carry
`FAIL_TO_PASS` and `PASS_TO_PASS` counts as fractions. Both joins were
validated by requiring that the report's `resolved` field reproduce the stored
`g` for every analysed row: 34/34 and 77/77, zero mismatches. Classes are
derived from the test counts alone, never from a model's opinion, and are
mutually exclusive and exhaustive. Per class we report `AUROC(u)` against the
passing trajectories, with the pre-registered inference floor of 10 positives
applied: classes below it are descriptive only.

**Result.** Of 47 silent failures: 29 made no progress and broke nothing, 8
broke something else with the bug unfixed, 5 fixed the bug and broke something
else, 4 were partial fixes, 1 patch did not apply. The one class above the
floor (no progress, Multilingual, n=21) gives `AUROC(u) = 0.534 [0.367,
0.697]`. No class is detectable. Incidental: the median analysed instance is
guarded by exactly one `FAIL_TO_PASS` test and 84 of 111 rows are guarded by
exactly one.

**Cost.** $0. No new gold, no new generation, no API calls. The ~$2 budgeted
for an LLM-judged taxonomy was not spent, because the executed tests answer
the question better than a judge would.

## Lucky-pass contamination of the `g=1` labels

**Question.** Our positive class is failure, so a trajectory that passes for
the wrong reason is a mislabelled negative, which inflates the apparent
separability of both `c` and `u`. Could that be what produced the null?

**Method.** Three flags per passing row, all from executed evidence and the
cached benchmark datasets: the patch touches no file the reference patch
touches; the instance is guarded by exactly one `FAIL_TO_PASS` test; the patch
adds two lines or fewer. Also recorded: whether the patch touched any file the
reference *test* patch touches, or any path matching a test convention. The
primary flag is file-disjointness from the reference patch. This is a **proxy
and an upper bound**, not a semantic judgement: a disjoint patch may be a
legitimate alternative fix. Sensitivity is then run two ways — drop the
flagged passes, and relabel them as failures — recomputing `AUROC(u)` and the
endpoint under the same three full-coverage estimators used in the paper. The
estimator implementation was validated by reproducing the published
`as_labelled` pooled values exactly (−0.030, −0.014, +0.013).

**Result.** Zero test-file tampering on either benchmark. Disjoint passes: 0
of 22 on Verified, 4 of 42 on Multilingual, pooled 6.2%; the four are five- to
fifty-line edits in a neighbouring file of the same module. Among failures the
disjoint rate is higher (10 of 35 on Multilingual), which is the expected
direction. Sensitivity: dropping moves the pooled LORO delta from −0.030 to
−0.053 [−0.107, −0.012] and the unfitted estimator from +0.013 to −0.007;
relabelling gives −0.024 under both. `AUROC(u)` falls from 0.542 to 0.516 and
0.484. Every scenario moves further from the safeguard working.

**Cost.** $0. Gold patches come from the already-cached benchmark datasets.

**Reading.** Label contamination cannot explain the null. It remains true, and
is stated in the paper's limitations, that harness gold is imperfect in both
directions and that every AUROC here is attenuated by label noise.

**Scripts.** `scripts/readout_failure_taxonomy.py`,
`scripts/readout_luckypass.py`. Outputs `data/readout_failure_taxonomy.json`,
`data/readout_luckypass.json`.

# Amendment 7a — clarification of Amendment 7's inference rules — 2026-09-27, appended BEFORE any gold exists for the agent axis

**Standing of this amendment.** No gold has been run on any agent-axis
trajectory and no quantity has been computed from the 100 that exist. Their
contents are hashed in `data/agent_axis/CHECKPOINT_HASHES.json` (100 files,
rollup `cb1968d5d5728902f4b24b83ad26f651f19586e2a98883834e6b2e1b27f92a9c`),
taken before this text was written, so a reader can verify that the rules
below were fixed before any outcome was visible. The trajectories are
preserved and will not be regenerated; see DEFECTS.md for the path defect that
would have silently discarded them.

**Why it is needed.** Amendment 7 as written commits a statistical error that
external review identified and that we accept. It treats `AUROC(u) < 0.60`
with a CI including 0.50, and an endpoint CI including zero, as *confirmation*
of the null. A confidence interval that includes the null is absence of
evidence, not evidence of absence, and with few positives a wide interval will
include the null almost regardless of the truth. As written, A7 could have
"confirmed" the null by being underpowered. Four further gaps: the strong
selector arm was conditional, there was no gate on the new selector actually
having signal on the new pool, the capability comparison did not fix how empty
patches enter the denominator, and the reading claimed the mentor's objection
would be "retired in full".

The following supersede the corresponding parts of Amendment 7. Everything
else in Amendment 7 stands.

## 1. The strong-selector arm is mandatory, not conditional

Amendment 7 said the `gpt-5` selection pass would run "only if the first regime
clears the inference floor". That is withdrawn. The cell the objection is
actually about is **capable agent crossed with capable selector**, so `gpt-5`
selection over the new candidate pool runs regardless of what the 4B arm does.
If budget permits only one selection arm, it is the `gpt-5` arm.

## 2. A gate on the selector, on the new pool

`gpt-5`'s pool `AUROC(-c)` over graded candidates of the new agent must exceed
**0.65**, and its committed arm's failure rate must be below the random arm's
(the same conjunction S-P1 registered, both halves reported this time). If the
gate fails, the strong-selector cell is not entered and every quantity from it
is descriptive only. Selector signal does not transfer across agents by
assumption: gpt-5 ranked the gpt-5-mini pool at 0.71, and that is not evidence
about a gpt-6-luna pool.

## 3. The capability comparison has a fixed denominator

A7-P1 compares per-attempt failure rates. **Failure rate is computed
unconditionally: every attempt is a denominator entry, and an empty patch or a
step-limit exit counts as a failure.** Both arms use the same rule. This is
stated because the ambiguity already produced a near-miss in Amendment 6, where
the conditional rate (30.0% against our 33.3%) and the unconditional rate
(30.0% against 48.6%) gave opposite verdicts, and only the unconditional rate
agreed with the public leaderboard (48.8%). Conditional rates may be reported
alongside, labelled as such, and never as the criterion.

## 4. The primary estimator is the unfitted one

The operational primary endpoint is the **fixed unit-weight estimator**
(`z(-c) + z(u)`, standardiser fit on the training fold, no fitted
coefficients). The leave-one-repository-out logistic and the 5-fold clustered
logistic are reported as secondary. This is a change of primary estimator and
we declare it as such. It is not opportunistic: the logistic's baseline
degenerated to 0.150 against a raw 0.591 in Amendment 5, which is the failure
mode that would otherwise decide this experiment, and the unfitted estimator's
baseline reproduces the raw value by construction. Raw `AUROC(-c)` will be
printed beside every out-of-fold baseline so that a degenerate baseline is
visible rather than inferred.

## 5. Inference floor raised to 20, and it is a floor on *failures*

Fewer than **20 silent failures** in the committed set → descriptive only, no
confirmatory reading in either direction. The old floor of 10 is retained only
as the threshold below which a quantity is not computed at all.

## 6. A null claim requires an upper bound below a declared margin

This replaces A7-P2 and A7-P3. The equivalence margin is **`ΔAUROC = +0.05`**,
the same threshold Amendment 5 registered as the effect size worth acting on.

- **A7-P3′ (the null is established):** the primary estimator's upper 95%
  confidence bound is **below +0.05**. Only this licenses "disagreement adds
  nothing in this regime".
- **A7-P3′′ (a positive):** the primary estimator's point estimate exceeds
  +0.05 and its lower bound excludes zero. This licenses "disagreement carries
  residual signal under a capable agent", and the paper is rewritten around it.
- **Otherwise: inconclusive.** We report the point estimate and the upper
  bound, and state that the design cannot separate no effect from an effect of
  the size that would matter.

**Declared in advance: the inconclusive branch is the likely one.** On 35
failures Amendment 5's comparable cell returned `+0.029 [-0.08, +0.14]`, an
upper bound nearly three times the margin. An upper bound below +0.05 needs a
CI half-width under about 0.05, which this design will probably not reach at
114 instances, and a capable agent produces *fewer* failures, which makes it
harder still. We write this down now so that reporting "inconclusive" later is
not read as a retreat, and so that a reader can see we did not expect to prove
equivalence. A7 is powered to detect a positive result, not to establish a
null.

## 7. The reading, restated

- A7-P3′′ fires → the null is specific to weak agents, and the paper's central
  claim becomes conditional on agent capability.
- A7-P3′ fires → the null holds in the strong-agent, strong-selector cell, and
  the mentor's objection is answered on the axis it named.
- Inconclusive → the paper reports the strong-agent cell as consistent with the
  null and bounded above by whatever the data gives, and says plainly that the
  objection is **narrowed, not retired**.

The phrase "retired in full" is withdrawn from Amendment 7. No single
experiment on one agent and one benchmark retires a capability objection in
full, and claiming otherwise was an overstatement.

## 8. What does not change

Population, benchmark, agent, panel, `u` definition, rank norms fit on training
repositories only, repository-clustered inference, and the cost caps are all as
in Amendment 7. This amendment changes decision rules and reporting, not the
instrument or the estimand.

# Amendment 7b — executable-readout and capability-denominator correction — 2026-09-27, before any agent-axis gold exists

**Standing.** Still no gold has been run on the agent-axis trajectories and no
outcome has been computed from them. This amendment was prompted by a second
PI audit of the October runbook and prewritten readout. It fixes implementation
gaps found without inspecting an outcome.

## The defect in the prepared workflow

The first runbook would have graded only the trajectory selected by the 4B
critic. That cannot answer A7-P1, whose denominator is all 456 attempts, and it
cannot compute `gpt-5`'s pool AUROC or its paired committed-versus-random gate.
The prewritten readout's dry-run path reproduced old numbers, but its production
path stopped after trying to load `rows.jsonl` (the pipeline writes
`replication_scored.jsonl`) and computed no Amendment 7 quantity. Calling that
workflow “fully prepared” was false.

## Gold and random arm, fixed before outcome

All four candidate-indexed arms will be graded. Empty patches do not consume a
gold run and count as failures; every non-empty attempt receives harness gold.
Four run IDs are necessary because the harness report is keyed by instance ID
and cannot represent four candidates with the same ID in one report.

The random arm is fixed locally before gold by
`sha256("a7-random-v1:" + instance_id)`, first eight bytes interpreted
big-endian, modulo four. `scripts/prepare_amendment7.py` writes those picks and
the four candidate-indexed gold inputs, and refuses any bundle other than the
registered 114 instances × four candidates.

`gpt-5` J1 scores are collected for every candidate, including empty attempts.
The selector-pool AUROC uses the candidates actually graded by the harness
(non-empty patches); the unconditional committed-versus-random comparison uses
all 114 paired instances, with empties and step-limit exits counted as failures
in either arm.

## A7-P1 when the old-agent comparator is only partially graded

The old `gpt-5-mini` pool has gold for only 213 of 456 candidate slots, so its
unconditional failure rate is an identification band, not a point. The
pre-outcome decision rule is conservative:

- **pass A7-P1** iff the new agent's upper failure-rate bound is at least 0.10
  below the old agent's lower bound;
- **fail A7-P1** iff the new agent's lower bound is no less than 0.10 below the old
  agent's upper bound;
- otherwise **indeterminate**, and every agent-axis endpoint is descriptive.

With complete new-agent gold its band collapses to a point. This rule does not
alter A7-P1's ten-point bar; it specifies how the promised comparison is made
when the registered comparator was not fully graded.

## “Paired difference in each,” made executable

Amendment 7 promised a paired difference against the `gpt-5-mini` run but did
not register a test or threshold for those differences. The executable reading
is descriptive: for each selector arm, report new-minus-old differences in
analysed failure rate, raw selector AUROC, raw disagreement AUROC, and each of
the three endpoint estimates. “Paired” means the same fixed 114-instance frame;
the registered candidate-specific `visible_pass` filter can produce different
analysed subsets across agents, so these are not row-paired outcomes and earn
no confirmatory verdict.

## Budget correction

The earlier ~$3 Modal-gold estimate covered one selected arm, not the all-attempt
denominator A7-P1 requires. Using the runbook's own estimate, four
candidate-indexed gold runs cost about **$12**, and the remaining generation
about **$8.40**. The Modal cap is therefore raised, before outcome, from $12 to
**$22**. This remains below the $30 free credit that resets on 1 October; no paid
overage is authorized. The OpenAI cap remains $15. If the free credit is not
available, the run pauses rather than spending money.

## Executable readout

`scripts/readout_amendment7.py` now validates the exact 114×4 bundle, complete
candidate gold, complete five-model panel/selector scores, all 456 `gpt-5`
scores, and the frozen random picks. It computes both 4B and `gpt-5` arms, the
capability branch, both selector-gate conditions, all three endpoint estimators,
the 20-failure floor and the A7-P3′/P3′′/inconclusive branch, then writes
`data/agent_axis/readout_amendment7.json`. Partial inputs exit non-zero and
write no result.

# Amendment 7c — selector tie rule — 2026-09-27, before agent-axis gold exists

The 4B selector's existing implementation in `run_replication_modal.py`
selects the first candidate when scores tie. The prepared Amendment 7 readout
instead selected the last. Four of the 114 earlier Multilingual candidate sets
have four-way score ties, so this difference is observable in existing data.
The Amendment 7 readout now uses the original first-candidate tie rule. No
agent-axis outcome has been inspected; this fixes the selector definition
before the new agent's gold exists.

# Post-result interpretation note — 2026-09-27

The existing pilot and replication have 47 analysed failures in 46 repositories.
Their pooled fixed-weight 95% interval reaches +0.12 AUROC, above the +0.05
effect this project called actionable. The two samples therefore replicate a
lack of detectable gain; they do not establish equivalence to zero or rule out
an actionable +0.05 gain. The paper and README now reserve “null replicates”
for the name of the registered R-P1 prediction and state the evidence in those
terms. This is an interpretation correction after results, not a new analysis
rule or an amendment to a preregistered verdict.

# Amendment 5a — correction of a population defect in Amendment 5 — 2026-09-27

**This is a bug fix, not a rule change.** No threshold, population definition,
estimand or estimator declared in Amendment 5 is altered. What changes is that
the code now implements the population Amendment 5 already specified. Three
registered verdicts move as a result, and two of them move against earlier
claims in the paper, so the correction is recorded here in full.

## The defect

Amendment 5 re-selects a new committed candidate with `gpt-5`, then filtered the
analysis population with `rows[i]["visible_pass"]`. That flag belongs to the
candidate the **4B critic** committed, which for three-quarters of instances is a
different trajectory. The registered population is "committed,
agent-declared-complete trajectories", so for this arm the flag must come from
the **gpt-5-committed** candidate. The scoring bundle carries `visible_pass` and
`excluded` per candidate, so the correct field was available throughout.

- **Verified: no effect.** Both readings give the identical population (n=32, 9
  failures) and identical statistics. Checked, not assumed.
- **Multilingual: 16 instances were wrongly dropped.** Each has a
  gpt-5-committed trajectory that is agent-declared-complete, has a non-empty
  patch, and has executed gold. They were excluded only because the 4B critic's
  pick had been empty. None were wrongly *included*; the published population is
  a strict subset of the correct one.

Amendment 2's re-selection rows were checked for the same defect and are
**unaffected**: on Verified the two readings coincide exactly, and the published
values reproduce (8B re-selection u 0.455, $-c$ 0.602, delta $-0.148$).

## What the correction does to the registered verdicts

| Quantity, Multilingual | As published | Corrected |
|---|---|---|
| n / failures / repositories | 85 / 32 / 39 | 101 / 44 / 40 |
| pool `AUROC` (unchanged by the fix) | 0.706 | 0.706 |
| committed-set `AUROC(-c)` | 0.591 | 0.671 |
| `AUROC(u)` | 0.614 [0.49, 0.73] | 0.591 [0.47, 0.70] |
| endpoint, LORO logistic | +0.375, baseline 0.150 | +0.012, baseline 0.575 |
| endpoint, 5-fold clustered | +0.195, baseline 0.377 | −0.024 |
| endpoint, unfitted (primary) | +0.029 [−0.07, +0.14] | −0.018 [−0.11, +0.07] |

- **S-P1 still passes.** The pool AUROC is computed over graded candidates and is
  untouched by the population filter (0.712 and 0.706). S-P1's second condition,
  the paired committed-versus-random failure contrast, also still passes on the
  corrected population: 0.258 against 0.323 on Verified (−0.065 [−0.20, 0.00],
  31 pairs) and 0.421 against 0.505 on Multilingual (−0.084 [−0.138, −0.033], 95
  pairs), with zero reverse-discordant pairs on either benchmark. Noted for
  honesty: 7 of 31 and 36 of 95 pairs are structural zeros because both arms
  chose the same candidate; restricted to pairs where the picks differ the
  contrast is −0.083 [−0.27, 0.00] and −0.136 [−0.22, −0.06].
- **S-P2 now FAILS on both benchmarks.** The registered bar is a pool-minus-
  committed gap above 0.10. Verified was already reported as failing (−0.080,
  and below the inference floor). On Multilingual the gap falls from 0.115 to
  **0.035**. The claim that selection measurably consumes a strong selector's
  own signal is therefore **not** supported by this amendment. It remains
  supported for the primary 4B selector on Verified (0.81 to 0.61), which is a
  different and weaker statement than the paper previously made.
- **S-P3 fails on its first condition.** `AUROC(u)` is 0.591, below the
  registered 0.60. The conjunction fails on its own terms, with no appeal to
  estimator validity.

## The consequence that matters most

The published LORO estimate of +0.375 with a baseline of 0.150 — the result that
looked like it would reverse the paper, and that Amendment 7a cited as the reason
to demote the logistic estimator — **was an artifact of the population defect**.
On the correct population that baseline is 0.575, close to the raw 0.671, and
the three estimators agree in sign for the first time in this study: +0.012,
−0.024, −0.018.

We are not withdrawing the four-estimator standard. It is what exposed the
anomaly that led to this bug being found, and the Verified cell still shows a
depressed logistic baseline (0.669 against a raw 0.792). Amendment 7a's choice
of the unfitted estimator as primary stands, on the general argument rather than
on this instance.

## Honest statement of what remains inconclusive

On the corrected population the unfitted estimator gives −0.018 with an upper
bound of **+0.073**. That is still above the +0.05 margin Amendment 7a declared,
so the strong-selector cell is consistent with the null, tighter than before,
and still short of establishing equivalence.

**Script.** `scripts/readout_amendment5_corrected.py`, which also fixes a
provenance gap: the `endpoint` block in
`data/selector_axis/prereg_readout_amendment5.json`, including the "+0.029" the
paper quoted and its "BASELINE DEGENERATE" annotations, was not produced by any
committed script. It is now regenerated by one, for both populations, with the
sign convention stated (`c` is a detection score; the baseline is `c`, not `-c`).

# Reporting audit — 2026-09-27, post hoc (no thresholds)

A PI-level audit after Amendment 5a. No new experiment, no new gold, no new
thresholds. Findings and fixes are recorded in DEFECTS.md #8 and in the paper's
claim-change appendix as v0.9.4. They concern figure captions, figure↔table
completeness, a superseded JSON field, short-paper drift, and one absolute
prose claim that outran the plotted data. None changes a registered prediction
or a corrected Amendment 5 verdict.

# Code-path audit — 2026-09-27, post hoc (no thresholds)

Continuation of the PI audit after the reporting pass. No new experiment, no
new gold, no new thresholds. Findings and fixes are in DEFECTS.md #9 and the
claim-change appendix as v0.9.6.

1. Exploratory E3 (selection pressure, best-of-2 vs best-of-4) had train-fold
   flags misaligned with the filtered score list in
   `scripts/readout_pi_batch2.py`. Regenerating moves Verified best-of-4
   $\mathrm{AUROC}(u)$ from 0.444 to 0.462. Neither value was quoted in either
   paper; the exploratory mention of the check is unchanged. Written after
   the regenerated number was visible.
2. Fig.~8 Verified points are now loaded from committed JSON (was hardcoded);
   checker ECE loader no longer uses a `jg and json.loads` idiom; the
   defective Amendment 5 script refuses to overwrite its archive without an
   explicit flag.

None of these changes a registered prediction or a corrected Amendment 5
verdict.

# Statistical-validity audit — 2026-09-29, post hoc (no thresholds)

Hostile read of estimator choice, equivalence language, and replication
narration. Recorded in DEFECTS.md #14 and claim-change appendix v0.9.10.

1. **R-P1.** The registered conjunction is still met on Multilingual
   (`AUROC(u)=0.565`, CI [0.43, 0.71]). The paper no longer headlines that as
   “the null replicates” or *\emph{Passed}*; it states failure to reject with
   upper bound 0.71, aligning replication prose with Amendment 7a's
   absence-of-evidence rule. Written after the wording fix was drafted.
2. **Instrument record.** Item (7) was caught by a registered-prediction audit,
   not an oracle or regression test; the instrument-failure prose now says so.
3. **Version.** Title block synced to changelog v0.9.10.

No registered threshold changed; no quoted number changed.

# Amendment 9 — static-analysis baseline and manual adjudication of the disjoint passes — 2026-09-27, declared after the confirmatory results and reported as exploratory

**Standing.** Both analyses below were run after the confirmatory results, carry
no pre-registered thresholds and no multiplicity correction, and cost nothing.
They exist because two reviewer objections deserved better answers than the
paper had: that `lines added` is a rhetorical baseline rather than a deployable
one, and that the lucky-pass bound rested entirely on a file-overlap heuristic.

## E7 — do classical static-analysis signals rank silent failures?

**Question.** If cheap non-LLM structural signals separate this population, the
finding is not "models are bad at this" but "the structure is there and the
models are not finding it". If they do not, the claim becomes broader and more
useful: the signal may not be cheaply available to anything.

**Method.** Features computed from the committed diff alone, which is what an
inference-time guardrail would have. Twelve lexical features across all eight
languages (decision points and density, maximum nesting, hunks, files touched,
net lines, error handling, signature changes, comment ratio, absence of control
flow). On Verified, which is 100% Python, six further features from parsing each
added hunk with the standard-library `ast` module, including whether the added
code parses standing alone. Single-feature AUROCs with repository-clustered
bootstrap; a combined model evaluated leave-one-repository-out so it is
comparable to the paper's endpoint; and the endpoint itself, asked of static
analysis exactly as it is asked of disagreement.

**Result.** No. Strongest single feature is net lines added, 0.705 [0.51, 0.94]
on Verified — the only structural feature anywhere whose interval excludes
chance — falling to 0.536 on Multilingual. The combined model reaches 0.644
[0.49, 0.78] on Multilingual, the highest any signal reaches there, and 0.436
[0.19, 0.60] on Verified, worse than chance. Direction reverses across
benchmarks.

**A caught artefact, recorded because it is the third instance of the class.**
The leave-one-repository-out logistic first reported static analysis adding
+0.169 [−0.01, +0.34] to the selector on Multilingual. Its baseline was 0.459
against a raw 0.560 — the depressed-baseline signature already documented for
the endpoint and for Amendment 5. Under this project's primary unfitted
estimator the same comparison gives −0.086 [−0.26, +0.08]. Had we reported the
logistic alone we would have published a positive finding for static analysis
that does not survive its own robustness check.

**Cost.** $0. Script `scripts/readout_static_analysis.py`, output
`data/readout_static_analysis.json`.

## Manual adjudication of the four disjoint passes

**Question.** The lucky-pass bound of 6.2% rests on a file-overlap heuristic: a
passing patch that touches none of the files the reference patch touches. A
heuristic cannot tell an alternative fix from a lucky one. Reading them can.

**Method.** All four flagged rows read by hand against the issue text and the
reference patch. No sampling: four is the whole flagged set.

**Result.** Three are unambiguous alternative fixes, and in each the agent's own
inline comments name the same root cause the reference patch addresses
(`gohugoio/hugo`: `url.Parse` treating `#` as a fragment; `immutable-js`: a
bitwise coercion folding `NaN` to zero; `micropython`: duplicate keyword keys,
same `TypeError`, different call site). The fourth, `vuejs/core`, is a five-line
change to subscriber notification order where the reference patch alters
batching and pause semantics; we cannot rule out that it passes without
addressing what the reference fixed, and we do not clear it. Contamination
supported by inspection is therefore **1 of 64, or 1.6%**, against the 6.2% the
automatic flag gives. The paper reports both and says which is which.

**Cost.** $0.

**Reading.** Neither analysis changes the registered endpoint or any registered
verdict. E7 broadens what the null is about; the adjudication tightens the
label-noise bound by a factor of four.

# Amendment 10 — the agent axis moves to open-weight models — 2026-09-28, declared before any trajectory exists for either model

**Origin.** The mentor who raised the capability objection was asked which models
to use and answered: capable open-weight models, naming **GLM-5.3** and
**DeepSeek V4.1 Flash**. This amendment adopts that, and supersedes Amendment 7's
choice of `gpt-6-luna` as the agent.

**Why this is better than a capability upgrade alone.** An API model can be
silently updated or retired, so a result obtained with one is not reproducible in
the strict sense: a reader cannot re-run it against the same weights. Open weights
are frozen and downloadable. For a paper whose central contribution is a
measurement protocol, and which documents seven instrument failures to make the
instrument auditable, an agent anyone can re-instantiate is worth more than a
marginally cheaper API. We state this as a reason in its own right and not as a
rationalisation of the mentor's preference.

**Capability, from published figures, not measured by us.** DeepSeek's V4 Flash
model card reports SWE-bench Verified at 73.7 non-thinking, 78.6 at high effort;
V4.1 Flash reports 74.2 on DeepSWE v1.1. GLM-5.3 reports open-source
state-of-the-art on Terminal-Bench 3.0 with large gains on SWE-style benchmarks.
Our own agent, `gpt-5-mini`, resolves about 56% of Verified. Either model is a
genuine capability step, which is the axis the objection named. **These are
vendor-reported numbers and we do not treat them as verified; the pilot below is
what we will act on.**

## A capability pilot is required first, and it is not optional

Amendment 6 exists because an untested agent in this scaffold produced 0 of 41
trajectories and an error that looked like a model limitation but was our own
configuration forcing function tools the model refused. That cost a day and is
recorded in DEFECTS.md. A new agent gets the same gate before any of the 456-
trajectory run is spent.

**Configuration, chosen with that failure in mind.**
`configs/mini_swe_deepseek.yaml` and `configs/mini_swe_glm.yaml` use
`model_class: openrouter_textbased`, which parses the action out of the model's
own text with a regex rather than sending function tools. Verified to resolve to
`OpenRouterTextbasedModel`. The API key is set by the operator through
`mini-extra config set OPENROUTER_API_KEY`, which writes outside this repository;
no key is stored here.

**Predictions, on the same 41 Verified instances as Amendment 6, k=1, 40-step
limit, harness gold.** Identical bars to Amendment 6 so the two are comparable.

- **P10-1 (it functions):** at least 80% of instances end with a submitted,
  non-empty patch.
- **P10-2 (it is more capable):** per-attempt failure rate at least 10 points
  below `gpt-5-mini`'s on the same instances, computed **unconditionally** with
  empty patches and step-limit exits counted as failures, per Amendment 7a §3.
- **P10-3 (it is affordable):** measured cost per trajectory at or below $0.040.

**Readings.** Both models pass → prefer the cheaper per trajectory, and record
the other as an available replication. One passes → use it. Neither passes →
report the attempt, keep `gpt-6-luna` as the Amendment 7 agent, and say plainly
that the open-weight agents could not be driven by this scaffold, which is a
finding about the scaffold and not about the models.

**Cost.** List prices: DeepSeek V4.1 Flash on OpenRouter is $0.0295 per million
input and $0.60 per million output, against `gpt-6-luna` at $0.10 and $0.50, so
input is roughly 3.4x cheaper and output roughly 1.2x dearer. `gpt-6-luna`
measured $0.005 per trajectory. We do not estimate from that ratio because
reasoning models emit more output tokens; the pilot measures it. Cap: **$5 per
model for the pilot**, which stops the run if either is unexpectedly expensive.
GLM-5.3 pricing is not asserted here because we have not verified it.

**What this changes downstream.** Amendment 7's agent, its
`--mini-config`, and its capability comparator all change to the selected model.
Everything else in Amendments 7 and 7a stands: mandatory `gpt-5` selection arm,
the 0.65 pool gate, unconditional failure rates, the unfitted primary estimator,
the floor of 20 failures, and the +0.05 margin for any null claim. The 100
`gpt-6-luna` trajectories already generated are **not** part of this arm; they
remain hashed and unanalysed, and if the agent changes they become a discarded
capability pilot rather than a partial run, which is stated now so that
discarding them later is not a silent decision.

# Amendment 10a — stop rule, budget ledger and cost enforcement for the open-model arm — 2026-09-29, before any Amendment 10 gold exists

Raised by external review of the October runbook. Three of its findings are
accepted and fixed here, before any gold is run and therefore before any outcome
is visible. A fourth is recorded as already fixed.

## 1. The stop rule was ambiguous, and the ambiguity is resolved against the fallback

Amendment 10's reading said: "Neither passes → report the attempt, keep
`gpt-6-luna` as the Amendment 7 agent." The October handoff says instead: if the
pilot fails any gate, stop and do not launch the axis. **These are different
instructions and the conflict is resolved here, in favour of stopping.**

`gpt-6-luna` is withdrawn as a fallback for the axis, for a stated reason rather
than for convenience. It is an API model, so running the axis on it does not
answer the objection that prompted Amendment 10, which was specifically about
*capable open-weight* agents. Amendment 10 also already declared that the 100
existing `gpt-6-luna` trajectories become a discarded capability pilot if the
agent changes. Reinstating them after a DeepSeek failure would be choosing the
arm after seeing which arm survived.

**The branch, fixed now:**

- **DeepSeek passes all three gates** → freeze it, recheck the ledger in §2, run
  the axis.
- **DeepSeek fails any gate** → **stop**. Report the pilot as a negative
  capability result, state that the agent axis remains untested, and leave the
  paper's existing limitation as it stands. Do **not** substitute GLM-5.3, which
  has already failed at 13/41 and is roughly six times over budget. Do **not**
  revert to `gpt-6-luna`. Any later attempt on a third agent requires a new
  amendment written before that agent is run.

## 2. Budget ledger, reconciled

Amendment 7b's $22 Modal cap was derived for the axis alone and does **not**
include Amendment 10's capability pilot or its gold. The full ledger against the
$30 free credit that resets on 1 October:

| Item | Modal | Basis |
|---|---|---|
| A10 pilot generation, 41 Verified trajectories | ~$0.53 | 41 × $0.013 measured Verified sandbox rate |
| A10 pilot gold, 41 instances | ~$2.26 | 41 × $0.055 from the Amendment 4f gold run |
| Axis generation, 456 Multilingual trajectories | ~$8.40 | Amendment 7b |
| Axis gold, four candidate-indexed runs | ~$12.00 | Amendment 7b |
| **Total** | **~$23.19** | |

**The Modal cap is therefore raised, before outcome, from $22 to $26**, which
leaves $4 of headroom inside the $30 credit and still authorises **no paid
overage**. If the credit is unavailable or the ledger is exceeded mid-run, the
run pauses rather than spending money. The OpenAI/OpenRouter cap for the agent
remains $15, against a measured DeepSeek rate of $0.008 per trajectory, which
puts the axis at roughly $3.65.

## 3. The cost cap is made independently enforceable

The DeepSeek config carries `cost_tracking: ignore_errors`, because OpenRouter
does not always return a cost field and mini-swe-agent hard-errors when it is
absent. That setting disables the scaffold's own `cost_limit`, so **the declared
cap was unenforceable as written**. Review was right to flag it.

Enforcement is now external and measured, not vendor-reported: the pilot runner
reads the OpenRouter credits endpoint immediately before and after the run and
computes spend by difference. The run aborts before launch if the projected cost
exceeds the cap, and the measured delta is written into the readout so P10-3 is
checked against an independent number rather than a self-report.

## 4. Recorded as already fixed

Review flagged "the null replicates" as an overclaim after the selector axis.
That was corrected in draft v0.9.10 before this review: R-P1 is narrated as the
"registered null-replication rule" with its upper bound of 0.71 stated and
equivalence explicitly disclaimed, and `check_consistency.py` bans the old
phrasing outside the history appendix. No further change.

## 5. Executable, not improvised

Review also noted the runbook had no runnable pilot command, which invites
improvisation and accidental overwrite of the existing directories.
`scripts/run_amendment10_pilot.sh` now pins the instance file, refuses to write
into an existing output directory, runs generation and gold, and computes the
three gates. The pre-October preflight validates the Amendment 10 config, the
pilot set and this ledger.

# Amendment 10b — five cross-stage defects in the October pipeline — 2026-09-30, before any Amendment 10 or agent-axis gold exists

Found by an independent trace of the pipeline, commissioned after a reviewer
caught the `preds.json` / `predictions.jsonl` mismatch. All five are the same
family as that one and as the earlier checkpoint-path defect: **an artefact that
disagrees across a boundary between two stages, failing only when the boundary
is crossed, and after the preceding stage has already been paid for.** None
changes an estimand. One changes an analysis rule and is declared here before any
outcome is visible.

## Fixed as plumbing, no analysis consequence

1. **The pilot pointed gold at a dataset repo without the harness columns.**
   `run_amendment10_pilot.sh` passed `--dataset princeton-nlp/SWE-bench_Verified`.
   Verified on disk: that repo has 500 rows but lacks `image`, `eval_script`,
   `log_parser` and `eval_type`, so `make_test_spec` raises `KeyError: 'image'`
   on the first instance. `SWE-bench/SWE-bench_Verified` has them, is
   `gold_modal.py`'s own default, and is what all seven real gold runs on disk
   used. The flag was the only thing introducing the wrong repo. It would have
   died immediately after the full agent spend, in the same position as the
   `preds.json` bug.

2. **The axis `--only` list passed slugs where repo ids are matched.**
   `RESUME_AMENDMENT7.md` listed `ray2333_grm_gemma2_2b_rewardmodel_ft`, but
   `gpu_rescore.py` compares against `Ray2333/GRM-gemma2-2B-rewardmodel-ft`. The
   slug form is only ever the output *filename*. Every critic would have been
   skipped, the volume fetch would have returned an empty directory, and the
   readout would have failed on a missing panel. Every other `--only` call site
   in the repository passes repo ids; the runbook was the sole exception.

3. **The axis generator hardcoded the tool-call base config.**
   `run_replication_modal.py` passed `-c swebench.yaml` unconditionally.
   `configs/mini_swe_deepseek.yaml` uses a `*_textbased` model class whose parser
   requires the ```` ```mswea_bash_command ```` fence, which `swebench.yaml` never
   emits and `swebench_backticks.yaml` does. Running the axis on DeepSeek would
   have produced `RepeatedFormatError` on all 456 trajectories: the exact failure
   already recorded twice in DEFECTS.md. A `--base-config` argument is added; the
   default is unchanged.

4. **Failed judge calls were cached as done.** `llm_judges.py` built its resume
   set without checking `p_resolved`, so a call that exhausted its retries was
   treated as complete and re-running the identical command never retried it,
   while the readout requires a non-null score for every slot. Not hypothetical:
   the committed replication cache holds 504 rows of which **14 are null, all
   rate-limit failures**, and re-running would have retried none of them. Only
   successes now count as done, which converts those 14 from unrecoverable into
   retryable.

## Declared analysis rule, because it changes what is analysed

5. **Gold-harness errors are excluded, not fatal.** `gold_modal.py` writes no
   `gold_extra.jsonl` row when an attempt errors, guarded by `if "error" not in
   r`. `readout_amendment7.py` required a row for every candidate slot and
   raised otherwise. Every real gold run on disk has errors, at **2% to 9% of
   attempts**, so across four candidate-indexed runs over roughly 340 non-empty
   attempts at least one is effectively certain. The readout would have aborted
   after all four gold runs were paid for, recoverable only by hand-writing gold.

   **The rule, fixed before any agent-axis gold exists:** a slot the harness
   records in `error_ids` is **excluded** from the analysed population and
   counted. This matches the project's existing treatment elsewhere, where
   `run_replication_modal.py` already models `gold_error` as an exclusion
   reason; the rule existed in one stage and not the other. A slot missing
   *without* a recorded harness error still aborts, because that is a pipeline
   fault rather than a harness outcome. If harness errors exceed **10%** of
   slots the readout stops rather than analysing a depleted population, and the
   excluded count is reported alongside every quantity.

   This can only shrink the analysed set, never select on outcome: exclusion is
   decided by the harness's own error record, which is written before any
   quantity is computed and is independent of whether the patch would have
   resolved.

---

# Tooling note, 30 September 2026 — checker coverage measured, not assumed

Written after the results and before the 1 October run. **No estimand, no
registered prediction, no reported number and no verdict changes here.** This
note exists because the project's guard against reporting errors had never been
measured, and when measured it was substantially weaker than assumed.

## What was measured

`paper/bite_test.py` runs `paper/check_consistency.py` with an instrumented JSON
loader that records every scalar the checker reads and where it came from. It then
re-runs the checker once per scalar with that one value perturbed, and requires a
non-zero exit. The perturbation is injected into the loader rather than written to
disk, so an interrupted sweep cannot leave a corrupted readout behind.

Perturbation size is the substantive design choice. The readouts store values
already rounded to three decimals and the papers print three decimals, so a
perturbation below 0.0005 changes no printed digit and no honest check could catch
it. Floats move by at least 0.05, which changes every rendering the papers use.

**Result: 42 of 148 scalars the checker read were not pinned by any assertion.**

## What that exposed

The largest was the loosest pooled upper bound, **+0.12**, which appears sixteen
times in `main.tex` and five in `main_short.tex` — abstract, plain-terms paragraph,
contributions, endpoint section, `tab:endpoint` caption, limitations, conclusion and
summary table. The only assertion reading it was `_upper > 0.05`, a one-sided
threshold serving a different claim, so the source could have drifted to any value
above 0.05 while every printed mention stayed at +0.12 and the checker stayed green.

The same shape held for five LLM-judge AUROCs the paper quotes by name, `u_max` on
the replication, three Amendment 5 values, and eight of the twelve `tab:endpoint`
rows. Each was read only inside a `max`, an `argmax` or a threshold.

Nineteen assertions now anchor these to sentences and table rows rebuilt from the
source JSON. The remaining 24 unpinned scalars are recorded with reasons in
`paper/bite_test_exemptions.json`; each participates in a maximum or argmax that
*is* pinned, and none is quoted in either paper. The harness fails if an unpinned
scalar has no recorded reason, and also if a recorded reason is no longer needed,
because a stale exemption can hide a later regression.

## Two errors of my own, recorded because they are the instructive part

1. **The previous bite test reported four failures that did not exist.** On
   29 September an uncommitted shell loop reported "12 caught, 4 NOT caught" for
   the lucky-pass and taxonomy assertions. Its perturbations were below printing
   precision. All four bite under perturbations that change the published value.
   The defect was the harness: no committed form, no stated perturbation rule, and
   a result I could not reproduce the next day.

2. **The first full sweep was invalid because I edited the checker while it ran.**
   The sweep re-executes the checker from disk each iteration, so edits changed what
   was being measured, and a transient crash in the checker returns non-zero and
   scores as "caught". That run reported "every pinned scalar bites" and it did not
   count. The harness now fingerprints every file the checker reads and aborts if any
   of them moved mid-run. The re-run on a frozen tree gives 167 caught, 24 exempt,
   zero unexplained, of 195 scalars.

I also nearly recorded the paper as wrong about the Verified unfitted upper bound.
The source is 0.245; Python's `format` renders that `+0.24` because it rounds the
binary double, while both the table and the prose print `+0.25`, rounding halves
away from zero. The paper's convention is the conventional one and the checker now
matches it. This is the fourth false alarm in this project's audit tooling, and the
standing rule held: reproduce a published number with your own code before calling
it wrong.

## Reproducibility defect fixed in the same pass

`readout_tier15.py` held its whole `D_simultaneous_bound` block as literals,
including the widest sweep upper bound `0.452`. When the paper superseded that with
+0.63 from the carried-forward 16k code-trained re-selection, the committed JSON was
corrected and the script was not, so re-running the script reverted the artifact to
the retracted value. The block is now derived from the readouts it describes, and
`scripts/check_readouts_regenerate.py` runs all six argument-free readout scripts and
requires each committed JSON to match what its script produces. It restores the tree
either way and is wired into `preflight_pre_oct1.sh`. Recorded as DEFECTS #15
and #16, and as changelog row v0.9.11.

---

# Amendment 10 outcome, 1 October 2026 — DeepSeek V4.1 Flash fails the functioning gate

Run at 00:03–00:38 local on the registered date, with no override. Gates were
registered in Amendment 10 before any trajectory existed and are identical to
Amendment 6's so the two models are comparable. Readout:
`data/pilot_deepseek_a10/readout_amendment10.json`.

| Gate | Registered bar | Measured | Verdict |
|---|---|---|---|
| P10-1 it functions | ≥80% submit a non-empty patch | 32/41 = 0.780 | **FAIL** |
| P10-2 it is more capable | ≥10 pts below `gpt-5-mini`'s 0.486 | 0.268, gap 0.218 | PASS |
| P10-3 it is affordable | ≤$0.040 per trajectory | $0.0135 | PASS |

Harness gold: 30 resolved, 11 unresolved, **0 harness errors**. Measured
OpenRouter spend $0.5527 against a $5.00 cap, differenced externally from the
credits endpoint because `cost_tracking` is disabled in the config.

**Decision: STOP.** `all_gates_pass` is false, so per Amendment 10a §1 the agent
axis is not launched, GLM-5.3 is not substituted, and the study does not revert to
`gpt-6-luna`. This is reported as a negative capability result.

## The failure is one instance, and the bar is not renegotiated

33 of 41 would have been 0.805 and a pass. The temptation to round 0.780 up, to
drop the instance that timed out, or to raise the step limit and re-run is exactly
what the stop rule was written against, and the ambiguity was already resolved
against the fallback in Amendment 10a §1. The bar was 80% before the outcome was
visible, so it is 80% now.

## The failing gate was checked for validity, which is a different question

Before accepting this as a capability result, the nine missing patches were
examined to see whether the gate had measured our scaffold rather than the model.
All nine exited `LimitsExceeded` at 33–39 of 40 steps. None failed to parse, none
crashed, and none was killed by the 120-second command timeout; 15 such timeouts
occurred during the run and the agent continued past each. `gpt-6-luna` cleared the
same bar at 41/41 under the same scaffold, the same 40-step limit and the same
harness gold (P6-1), which is why the bars were made identical. The gate measured
convergence within a fixed budget, which is what it was for, so the result stands
without an asterisk.

Whether DeepSeek would pass at 60 steps is a separate, untested question. Answering
it requires re-registering the comparison and re-running `gpt-6-luna` at the new
budget as well, because a budget changed for one arm destroys the comparability
that makes the number mean anything.

## A published mechanism claim is corrected by this run

`main_short.tex` previously explained the open-weight shortfall as ``not coding
ability but the ability to hold a text-driven action format for forty turns.''
That inference came from runs without gold and is **wrong for DeepSeek**: it holds
the format well enough to produce 32 valid patches and resolve 30 of 41, beating
the baseline agent by 21.8 points of failure rate. The mechanism is step-budget
exhaustion, not format drift. Both papers now say so.

The submit count also replicates: `pilot_deepseek_v2` gave 31 of 41 and this
registered run gives 32 of 41. The 13-of-41 figure quoted for DeepSeek elsewhere
belongs to the hung run recorded in DEFECTS.md, which is not a capability
measurement. GLM-5.3's 13 of 41 was never taken to gold, so its mechanism remains
unmeasured and no claim is made about it.

## Data committed with this outcome

The readout's two inputs, `data/pilot_deepseek_a10/preds.json` and
`data/gold_modal/a10_deepseek_pilot/report.json`, were initially excluded by the
blanket `data/*/*` rule — the sixth recurrence. Both are now tracked, along with
five further previously-dropped files the same fix recovered. `check_inputs_tracked.py`
now also checks runtime-constructed path families as globs, so a future run
directory is covered without editing the allow-list.

---

# Amendment 11, 1 October 2026 — does a larger step budget change the Amendment 10 result?

**Written before this run and after Amendment 10's outcome was visible. The budget below
was chosen knowing that DeepSeek submitted 32 of 41 against an 80% bar. That makes this
amendment POST HOC with respect to the step budget, and the consequence is stated
plainly: a pass here CANNOT be reported as Amendment 10's P10-1 passing. It is a new
measurement with a new name.**

## Why run it

Amendment 10 attributed its gate failure in part to this scaffold: all nine missing
patches exited `LimitsExceeded` at 33–39 of 40 steps, none failed to parse, and DeepSeek
resolved 30 of the 32 instances where it did submit. The obvious objection is that the
bar was set by our own budget. That objection deserves a measurement rather than an
argument, and the measurement is cheap: roughly $11 against $23 of remaining Modal
headroom inside the $30 credit.

## Both arms move, or the comparison is rigged

P10-2 compares DeepSeek's unconditional failure rate against `gpt-5-mini`'s 0.486,
**measured at 40 steps**. Raising the candidate's budget while leaving the comparator at
40 would hand DeepSeek the gap. So `gpt-5-mini` is re-run at the same budget on the same
41 instances.

**This costs us the leaderboard anchor for the re-run only.** Our 40-step `gpt-5-mini`
figure agrees with the public leaderboard to 0.2 points at the configuration the
leaderboard used, and that remains the paper's primary number. The 100-step `gpt-5-mini`
figure is used **only** as the comparator inside this amendment and is not reported as a
leaderboard-comparable result.

## Design

Same 41 Verified instances as Amendments 6 and 10, `k = 1`, harness gold, identical bars,
`step_limit = 100` for both arms. 100 is two and a half times the original budget, chosen
so that no reader can claim the budget was the binding constraint; it is not tuned to any
observed value.

- **P11-1 (it functions at a larger budget):** DeepSeek submits a non-empty patch on at
  least 80% of instances.
- **P11-2 (it is more capable at an equal budget):** DeepSeek's unconditional failure
  rate is at least 10 points below `gpt-5-mini`'s **at the same 100-step budget**, with
  empty patches and step-limit exits counted as failures per Amendment 7a §3.
- **P11-3 (still affordable):** measured cost at or below $0.040 per trajectory.

## Readings, fixed now

- **P11-1 fails.** The 40-step result was not a budget artefact. The scaffold objection is
  retired on evidence, Amendment 10's verdicts stand unchanged, and the agent axis stays
  closed. This is the outcome that most strengthens the paper.
- **All three pass.** A capable open-weight agent can be driven by this harness given a
  larger budget, and the axis becomes runnable on it — but under a budget chosen after
  seeing a failure. That caveat travels with any axis result produced this way and must be
  stated wherever the axis is reported. Launching the axis still requires its own
  amendment.
- **P11-1 passes and P11-2 fails.** DeepSeek functions but is not more capable at an equal
  budget, which would mean the 40-step capability gap was itself partly a budget artefact.
  Report it and stop.

Amendment 10's recorded verdicts do not change under any outcome, `gpt-6-luna` is not
reinstated, and the paper's primary 40-step figures stand.

## Caps and stop rules

Modal ≤ **$10** for this amendment (two arms of generation plus two gold runs), against
the $26 authorised cap of which ~$2.79 is consumed. OpenRouter ≤ **$3**, OpenAI ≤ **$3**,
both measured externally by differencing the provider's usage endpoint rather than trusting
the scaffold. **No paid overage.** If a cap is exceeded the run stops before gold. If
either arm has not finished generation within three hours it is stopped and reported as
incomplete rather than analysed.

## Amendment 11, deviation recorded before any run: the comparator arm is blocked

A one-instance smoke of the `gpt-5-mini` arm at `step_limit = 100`, run before committing
to 82 trajectories, returned `AuthenticationError` from OpenAI: *"Your API key has been
invalidated."* Cost $0.00, one API call, 27 seconds. The OpenRouter key is valid
(HTTP 200, usage $8.478 of $50 granted), so the DeepSeek arm is unaffected.

**Consequence for the three predictions:**

- **P11-1 (it functions at a larger budget)** — answerable with the DeepSeek arm alone;
  it references no comparator. This is the prediction that tests whether the 40-step bar
  was an artefact of our budget, which is the objection Amendment 11 exists to answer.
- **P11-3 (still affordable)** — answerable with the DeepSeek arm alone.
- **P11-2 (more capable at an equal budget)** — **blocked.** It requires `gpt-5-mini`
  re-measured at 100 steps. It will **not** be computed against the 40-step figure of
  0.486, because comparing a 100-step candidate with a 40-step comparator is the rigging
  this amendment was written to avoid.

**Declared now, before the run:** if the DeepSeek arm proceeds alone, P11-2 is reported as
*attempted and blocked on an invalidated credential*, not as passed, failed, or quietly
omitted. It becomes answerable only if the comparator is re-run at the same budget, which
requires a working key the operator sets themselves. Amendment 10's verdicts are unchanged
either way.

## Amendment 11 outcome, 1 October 2026 — all three gates pass at a 100-step budget

Readout: `data/readout_amendment11.json`, from `scripts/readout_amendment11.py
--deepseek-measured-cost 0.6272`. Both arms ran at `step_limit = 100` on the same 41
Verified instances in the same Modal sandbox environment.

| Gate | Bar | Measured | Verdict |
|---|---|---|---|
| P11-1 functions at a larger budget | ≥80% submit | DeepSeek **39/41 = 0.951** (40 steps: 32/41 = 0.780) | **PASS** |
| P11-2 more capable at an equal budget | gap ≥ 10 pts | DeepSeek failure **0.098**; `gpt-5-mini` **0.556** (way A) / **0.610** (way B); gap **0.458 / 0.512** | **PASS, both ways** |
| P11-3 still affordable | ≤ $0.040/traj | **$0.0153** (comparator $0.0357) | **PASS** |

**The 40-step bar was the binding constraint, and 100 was not.** No DeepSeek trajectory
reached the ceiling: the longest submitting run used 79 steps; seven of the 39 needed more
than 40. All seven instances that exited `LimitsExceeded` at 40 steps submitted at 100, and
nothing that submitted at 40 regressed. Gold: 37 resolved, 4 unresolved, 0 harness errors.
The two remaining non-submissions exited `RepeatedFormatError` at 20 and 50 steps — the
format problem is real but was never the binding constraint.

**The comparator, and the classification declared post hoc.** `gpt-5-mini` at 100 steps:
39 of 41 non-empty patches (two `Submitted` exits carried an empty diff), 16 resolved, 20
plainly unresolved, and five rows the gold wrapper filed as `error`. Those five are the
agent's own patches breaking the package before any test ran — an `IndentationError`, a
lost import, a circular import, a `SyntaxError`, one hunk that did not apply — reproduced
identically across two gold runs (DEFECTS #19). Amendment 10b's 10% exclusion rule was
written for harness faults, so the readout reports both:

- **Way A, Amendment 10b as written:** the five excluded, population 36, failure 20/36 = 0.556.
- **Way B, Amendment 7a §3 unconditionally:** the five counted as failures, population 41,
  failure 25/41 = 0.610.

P11-2 is declared to pass only if both ways clear the bar. Both do, by more than four times
the margin. The reclassification therefore does not decide the verdict, and it is recorded
here as post hoc because it was made after the rows were seen.

**Three comparator trajectories were regenerated at a single worker.** They had exited
`RateLimitError` on OpenAI's 500,000 tokens-per-minute ceiling at six workers, which is our
rate limit and not the model's ability; scoring them as capability failures would have
inflated the comparator's rate by up to 7.3 points. All three submitted on the rerun; one
needed 86 steps. The merge is recorded in `data/pilot_a11_gpt5mini/MERGE_NOTE.md`.

**An observation, not a finding.** `gpt-5-mini` resolved 16 of 41 at 100 steps against the
≈21 of 41 implied by its registered 40-step rate of 0.486. The two measurements come from
different runs and the earlier one is leaderboard-anchored under a different environment,
so this is not a controlled comparison and is not claimed; it is consistent with a larger
budget letting a weaker agent rewrite whole files and break them, which is what the five
crashed rows look like, and it would need its own registered test.

**What this does and does not change.** Amendment 10's recorded verdicts stand: at 40
steps DeepSeek failed the functioning gate, and that remains the registered result at that
budget. Amendment 11 is a new measurement at a budget chosen after that outcome. The agent
axis becomes **runnable** on DeepSeek V4.1 Flash, under its own amendment written before
it runs, and every result it produces must state that its agent qualified under a post-hoc
budget. The paper's primary 40-step figures, including the leaderboard-anchored
`gpt-5-mini` number, are unchanged.

**Spend.** OpenRouter: $0.6272 for the DeepSeek arm, measured externally; $0.0000 during
the comparator arm, as sequencing intended. OpenAI, scaffold-reported: $1.3298 for the
six-worker run plus $0.2506 for the three reruns. Modal is not readable from the
repository; at the ledger's rates, two generation runs and **three** gold runs — the third,
on the merged comparator, was unplanned and absorbed by the cap — come to roughly $8
against the $10 authorised for this amendment and ~$23 of headroom. The operator should
read the dashboard figure rather than rely on this estimate.

One cost hazard the preflight caught afterwards: mini-swe-agent's `swerex_modal`
environment leaves its `swe-rex` deployment up after a run (`ModalDeployment.stop` is
never awaited), and the one created at 00:03 was still deployed at 11:30 with 46 tasks
on record. It was stopped by hand. `modal app list` after every run is now part of the
no-spend gate's definition of done.

---

# Amendment 12, 1 October 2026 — GLM-5.3 under DeepSeek's qualifying settings

**Written before this run, after Amendments 10 and 11 were visible. Post hoc with respect
to every setting below.** Amendment 10a §1 ruled GLM-5.3 out of the axis on two gates —
13 of 41 submitted, and cost over the cap — and Amendment 11 then showed that the 40-step
budget was the binding constraint for DeepSeek. This amendment asks whether the same is
true of GLM. It exists for completeness and for the sentence the papers currently carry,
"GLM-5.3's mechanism is unmeasured"; it **cannot** qualify GLM for the axis, for the reason
in P12-3 below, and no outcome here changes which agent the axis would run on.

## What the 40-step record actually shows

From the committed trajectories: 22 of 41 exited `LimitsExceeded`, 6 `RepeatedFormatError`,
13 `Submitted`; median 35 steps, maximum 40. Scaffold-reported cost $0.1739 per trajectory,
**4.3×** the $0.040 cap — Amendment 10a's "roughly six times" was an overstatement and is
corrected here. More than half the runs hit the ceiling, a stronger budget signal than
DeepSeek's nine.

## Two settings change, and attribution between them is not claimed

GLM ran with `max_tokens: 4000`; DeepSeek's config raised its own cap to 16000 after
recording that a tight cap starves the action and the turn parses to zero actions. GLM's
six `RepeatedFormatError` exits are consistent with that mechanism. Testing the budget
alone would leave a known instrument defect in place, so GLM runs under **DeepSeek's
qualifying settings**: `step_limit = 100` and `max_tokens = 16000`, same 41 instances,
same Modal environment, same text-based model class. The question answered is "does GLM
qualify under the settings DeepSeek qualified under," which is the one the axis cares
about. Because two things changed, a pass is not attributed to the budget alone.

## Comparator

`gpt-5-mini` at 100 steps is already measured on these instances in this environment
(Amendment 11, `data/gold_modal/a11_gpt5mini_merged/report.json`). It is reused, not
re-run. The classification of its five agent-caused crashes is reported both ways, as in
Amendment 11, and P12-2 passes only if both agree.

## Predictions

- **P12-1 (it functions):** GLM submits a non-empty patch on at least 80% of instances.
- **P12-2 (more capable at equal budget):** unconditional failure rate at least 10 points
  below `gpt-5-mini`'s at 100 steps, under both classifications.
- **P12-3 (affordable):** at or below $0.040 per trajectory. **Expected to fail.** The
  40-step rate was already 4.3× over, and more steps and a larger token cap raise it. A
  P12-3 failure keeps GLM disqualified for the axis whatever P12-1 and P12-2 show.

## Readings, fixed now

- P12-1 passes → the 40-step result was an artefact of the scaffold settings for GLM
  too, and the papers' "mechanism unmeasured" becomes a measured statement. GLM stays
  disqualified on cost.
- P12-1 fails → GLM does not function under the settings that worked for DeepSeek; the
  shortfall is not explained by budget or token cap. Reported as such.
- Amendment 10's and 11's verdicts are unchanged under every outcome.

## Caps and stop rules

A one-instance smoke runs first. Its scaffold-reported cost is multiplied by 41; if the
projection exceeds **$20** the full run does not start. OpenRouter is capped at **$20**
for this amendment, measured externally by differencing the credits endpoint (balance
before: $40.89 of $50). Modal ≤ **$5** (one generation run, one gold run) against $27.10
read from the dashboard. No paid overage. Three-hour stop on generation.

## Amendment 12, deviation recorded mid-run: a third setting was binding, and it was not the budget

With 33 of 41 trajectories written, 20 had exited `LimitsExceeded` at a median of 31 steps
and a maximum of 51 — none near the 100-step ceiling. Their scaffold-reported costs are
$0.252–$0.300, every one just past the `cost_limit: 0.25` per trajectory that both
open-weight configs carry. That limit never bound DeepSeek at $0.015 a run; it binds GLM
at $0.17–0.30 a run. "DeepSeek's qualifying settings" therefore contained a setting that
was inert for DeepSeek and decisive for GLM, and this amendment did not name it.

**Consequence, declared before the readout is computed.** P12-1 as registered — does GLM
function at a 100-step budget — is **not answered by this run**. What the run measures is
GLM under a $0.25 per-trajectory cost cap, and that is how it is reported: descriptively,
with the number of runs the cap cut. The registered reading "P12-1 fails → the shortfall is
not explained by budget or token cap" does **not** apply, because a third cap explains it.
The run is allowed to finish so the record is complete; its remaining cost is about $2.

**What answering the registered question would take.** A rerun with the cost limit raised
far enough not to bind — on this evidence at least $1.00 per trajectory — at roughly $25–40
of OpenRouter for 41 instances, which exceeds this amendment's $20 cap and most of the
remaining balance, to settle a question on which no decision rides: P12-3 fails at every
budget, so GLM is disqualified for the axis regardless. It is not run under this amendment.
If it is ever run, it is a new amendment with its own cap.

The 40-step record is re-read in the same light: GLM's mean cost there was $0.1739 with a
maximum of $0.2663, so some of its 22 `LimitsExceeded` exits at 40 steps were also the cost
cap rather than the step ceiling. The papers' sentence that GLM's mechanism is unmeasured
remains the accurate one, now with the reason stated.

**Checked for the results already reported.** The same `cost_limit: 0.25` sits in DeepSeek's
config. Across Amendment 11's DeepSeek arm the most expensive trajectory cost $0.123 and
across the gpt-5-mini comparator $0.086; zero runs on either arm reached the cap. The
Amendment 11 verdicts are therefore not confounded by it. Amendment 10's DeepSeek arm: maximum
$0.058, zero at the cap.


## Amendment 12 outcome, 1 October 2026 — not evaluable as registered; GLM stays disqualified on cost

Readout: `data/readout_amendment12.json`, from `scripts/readout_amendment12.py --measured-cost 9.0378`.

| Gate | Bar | Measured | Verdict |
|---|---|---|---|
| P12-1 functions | ≥80% submit | **16/41 = 0.390** (40 steps/4k: 13/41) | **NOT EVALUABLE** — 23 of 41 runs cut by the $0.25 cost cap, 0 by the step ceiling |
| P12-2 more capable at equal budget | gap ≥ 10 pts | GLM failure 0.610 vs `gpt-5-mini` 0.556 / 0.610; gap -0.054 / 0.0 | **FAIL** |
| P12-3 affordable | ≤ $0.040/traj | **$0.2204** (5.5× over; 40 steps: $0.1739) | **FAIL, as declared** |

Gold: **16 resolved, 25 unresolved, 0 harness errors**, oracle controls correct.
External OpenRouter measurement $9.0378 against the scaffold's $8.8111: agreement to 2.5%.

**The one thing this run does establish.** GLM submitted 16 patches and **all 16 resolved**.
It is not an agent that produces wrong answers; it is an agent that, at $0.22 a trajectory, is
cut off by a per-trajectory cap before it finishes on 23 of 41 instances. Steps used by the
submitting runs: median 40, maximum 69, none at the ceiling.

**What it does not establish.** Whether GLM would clear 80% with the cap lifted. The
registered question stays open, for the reason recorded in the mid-run deviation above, and
no decision rides on it: P12-3 fails at any budget. GLM remains disqualified for the axis on
cost. The papers' sentence that its mechanism is unmeasured stands, now stated with its cause.
Amendments 10 and 11 are unchanged.

**Spend.** OpenRouter $9.0378 of a $20 cap, measured externally. Modal: one generation run
and one gold run; the dashboard read $27.10 before this amendment and should be re-read.


---

# Amendment 13, 1 October 2026 — the agent axis, on the agent that qualified

**Written before any axis trajectory exists on this agent. Post hoc in one declared respect:**
the agent qualified under a budget (100 steps) chosen after Amendment 10's failure at 40, and
every number this axis produces carries that caveat. Nothing else below is tuned to an outcome.

## What runs

Amendment 7's design, unchanged in every registered respect except the agent and its budget:
the **same 114 SWE-bench Multilingual instances** as the replication, `k = 4` (456 trajectories),
the same m=4 audit panel, the same 4B selector, the mandatory `gpt-5` strong-selector arm, the
same harness gold, analysed by `scripts/readout_amendment7.py` under **Amendment 7a's rules**
(unconditional failure rates, unfitted primary estimator, 20-failure floor, null claim only
below +0.05, the 0.65 pool gate, the frozen random arm). The agent is **DeepSeek V4.1 Flash**
under `configs/mini_swe_deepseek_axis.yaml`, which names every limit: `step_limit 100`,
`max_tokens 16000`, `cost_limit 1.00`. The cost limit is a ceiling that must not bind; a
trajectory that reaches it is reported as a cut, as Amendment 12 taught.

Output goes to a **fresh** directory, `data/agent_axis_deepseek`. The 100 `gpt-6-luna`
trajectories in `data/agent_axis` are untouched and, per Amendment 10, are a discarded
capability pilot; they are not mixed in and their hashes still verify.

## The capability comparison, two ways, both declared now

Registered **A7-P1** compares against `gpt-5-mini`'s replication rate, which is a band
(0.476–0.752) because only 213 of 456 replication slots have gold, and which was measured at
40 steps. Amendment 11 established that comparing a 100-step candidate with a 40-step
comparator hands the candidate the gap. So this amendment adds an **equal-budget comparator**:
`gpt-5-mini`, `k = 1`, 100 steps, same 114 instances, `configs/mini_swe_openai_axis.yaml`,
gold run `a13_gpt5mini_100`. Both comparisons are reported. The registered A7-P1 verdict uses
the registered band; the equal-budget figure is reported beside it, and if the two disagree
the disagreement is stated rather than resolved by choosing one.

## Predictions

A7-P1, A7-P2, A7-P3 exactly as registered in Amendment 7 and read under Amendment 7a. No new
thresholds. The agent-axis predictions the papers currently list as "registered and
unanswered" become answered — or, under 7a, INCONCLUSIVE, which is a legitimate reading and
was predicted to be the likely one.

## Stages, in order, each gated

1. One-instance smoke of the DeepSeek axis config on a Multilingual instance (`--limit 1`,
   `k=1`); must submit; cost projected ×456 must be under the OpenRouter cap. Deleted after.
2. Generate 456 trajectories, 6 workers. OpenRouter spend measured externally by differencing
   the credits endpoint; the runner's own cap is a second line.
3. Bundle; must be exactly 114 × 4 slots.
4. Panel scoring on Modal GPU, 4k prefix, the four registered auditors plus the 4B selector,
   by HuggingFace repo id (the slug form silently scores nothing).
5. `gpt-5` judges every candidate, J1 only.
6. `prepare_amendment7.py` against the new bundle: four candidate-indexed gold inputs and the
   frozen random arm.
7. Four gold runs, `a7_agent_axis_cand{0..3}`, oracle controls on cand0, on
   `swe-bench/SWE-Bench_Multilingual` (verified to carry `image`, `eval_script`, `log_parser`,
   `eval_type`; the replication's gold used it).
8. The equal-budget comparator: generate `k=1` at 100 steps, gold `a13_gpt5mini_100`.
9. `readout_amendment7.py` pointed at the new directory, plus the comparator figure.

## Caps and stop rules

OpenRouter ≤ **$20** for generation (balance $31.85 before; DeepSeek's Multilingual rate is
expected around $0.03 a trajectory, ≈ $14). OpenAI ≤ **$20** for the judge (~$9) and the
comparator (~$8); the operator confirms the account holds it before stage 5. Modal ≤ **$22**
(Amendment 7b's cap) against $27.10 on the dashboard before Amendment 12, re-read before
launch. **No paid overage.** If any cap is exceeded the pipeline stops before the next paid
stage and the partial state is reported, not analysed. The `swe-rex` deployment is stopped at
the end; `modal app list` is part of done. Wall clock: roughly twelve hours; a stage that has
not finished within twice its estimate is stopped and reported.

## Amendment 13, deviation recorded mid-run: the sandboxes never closed, and the run was paused at 100 of 456

Recorded 1 October 2026, 15:10 EDT, after the generation stage was killed and before it was
resumed. Generation started 13:02 EDT and was killed at 14:49 EDT with candidate 0 at 100 of
114 trajectories (96 `Submitted`, 4 `RepeatedFormatError`) and candidates 1–3 not started.

**What happened.** `modal container list` showed 100 live containers on the `swe-rex` app:
one per finished trajectory. Two upstream defects stack (DEFECTS #21): mini-swe-agent 2.4.6's
batch runner never calls `env.stop()`, and swerex's `ModalDeployment.stop()` terminates a
sandbox only if it has *already exited*. Every sandbox therefore lived to the config's
`deployment_timeout` (7200 s). At 100 idle containers Modal's workspace cap was reached, and
the remaining 14 candidate-0 instances (tokio-rs__axum-691, -734, tokio-rs__tokio-6724,
-6752, -7139, uutils__coreutils-6377, -6575, -6731, valkey-io__valkey-1499, -1842, -790,
vuejs__core-11589, -11870, -11915) each died in `SandboxTimeoutError` while waiting for a
sandbox, **with zero model calls**. The generator then logged candidate 0 as complete at 100
and would have stalled the same way on candidate 1.

**Fix, verified before the resume.** `envs/swerex_modal_closing.py` subclasses the swerex
Modal environment and terminates the sandbox when the trajectory releases the environment
(and at process exit); `envs/run_swebench_closing.py` is a drop-in for `mini-extra swebench`
that points the registry name `swerex_modal` at that class, so the registered YAMLs are
byte-identical and the runner's image selection still applies. Nothing the agent sees
changes: same image, commands, limits and timeouts. The trajectory's
`info.config.environment_type` records which class ran, so the first 100 and the rest are
distinguishable. A one-instance smoke through the new path (apache__druid-14092: `Submitted`,
19 steps, $0.0102 external) left **zero** containers 35 s after the process ended, with no
app stop. One earlier smoke attempt, minutes after the mass teardown, failed at sandbox
start with `SandboxTimeoutError`; the identical attempt passed on retry and a direct swerex
probe on the same image started in 7 s, so that failure is recorded as transient.

**Resume accounting.** The generator skips existing trajectories (`--redo-existing` is off),
so the resume regenerates exactly the 14 instances above under candidate 0's seed and then
candidates 1–3. Those 14 are retried because no model call was made; they are not re-rolls.
OpenRouter spend to the pause is $1.89 (18.161482 → 20.053937, in the ledger as
`openrouter_at_pause`); the $20 cap is measured from the **first** `openrouter_before`, so
the segments add. The 4 `RepeatedFormatError` exits stand as failures, as registered. Modal
spend during the leak is read from the dashboard by the operator and reported in the
outcome; the 100 idle containers are the cost of this defect, not of the measurement.

**What does not change.** Predictions, gates, caps, stop rules, the comparator design and
the 7a inference rules are untouched. The only post-hoc element added is the environment
class, which cannot affect `c`, `u` or gold.

## Amendment 13, second deviation recorded mid-run: slow sandbox starts were reported as failures

Recorded 1 October 2026, 16:25 EDT, after the resumed generation was killed at candidate 1 =
47 of 114 and before the second resume. Candidate 0 had completed (114: 110 `Submitted`,
4 `RepeatedFormatError`) once the 14 instances from the first deviation were regenerated;
they are in the data under candidate 0's seed with the closing environment class recorded.

**What happened.** In candidate 1, 66 instances died in `SandboxTimeoutError` at ~51 s with
**zero model calls**, while 47 ran normally. The cause is a second upstream defect (DEFECTS
#22): swerex waits for the sandbox's tunnel with Modal's `Sandbox.tunnels()` default of 50 s,
not the registered `startup_timeout` of 600 s, so any sandbox whose image pull or scheduling
takes longer than 50 s is reported as failed before it starts. Candidate 0's first pass was
not affected because its images were warm; by candidate 1 they were not. mini-swe-agent
again wrote the failed instances into `preds.json` with an empty patch, and its resume logic
skips by that key, so each candidate directory now carries a `RESUME_NOTE.md` listing the
removed placeholders with a backup of the file as it was.

**Fix, verified before the resume.** A one-line patch to the venv's swerex
(`tunnels.aio(timeout=max(50, startup_timeout))`, marked `silentfail patch`), checked by the
preflight and asserted at import by `envs/swerex_modal_closing.py`, which refuses to run
without it. A one-instance smoke through the patched path: `Submitted`, 20 steps, $0.0118
external, container self-closed. The smoke took seven minutes end to end, which is the slow
start the patch now tolerates.

**What this means for the measurement.** Nothing about `c`, `u` or gold changes; the 66
instances are regenerated under candidate 1's seed exactly as the 14 were. Wall clock is
longer than estimated. OpenRouter spend to this pause is $2.97 (`openrouter_at_pause2`).

## Amendment 13, third deviation recorded mid-run: two instances declared endpoint failures

Recorded 2 October 2026, 05:20 EDT, before the fifth resume. Candidate 1 stands at 112 of 114.
The remaining two, hashicorp__terraform-34900 and valkey-io__valkey-790, were attempted twice
in full (00:50–02:50 and 03:05–05:12). In both attempts the agent ran normally for 20–30 steps
and then every model call failed at the HTTP layer (`RemoteDisconnected`, read timeouts) and
was retried without limit by mini-swe-agent while the sandbox sat idle; the first attempt
ended when the sandbox reached its 7200 s lifetime, the second was killed by hand after an
hour of retries. The same instance (valkey-790) completed normally in candidate 0, so this is
the endpoint failing on a particular request, not the agent failing on the task.

**Declared rule.** An instance whose two full attempts both end this way is an *endpoint
failure*: no trajectory, no capability verdict. It enters the bundle as an excluded row
(`exit_status: EndpointFailure`, `exclusion_reason: endpoint_failure`) so that the bundle keeps
456 rows over 114 instances; the selector chooses among that instance's three remaining
candidates. The Amendment 13 outcome reports A7-P1 both ways — with the two rows excluded
(the registered Amendment 10b treatment of infrastructure errors) and with them counted as
failures — and P7-style inference is unaffected unless the two readings disagree, in which
case the stricter one is reported. The list is `data/agent_axis_deepseek/ENDPOINT_FAILURES.json`;
the recovery script will not regenerate anything on it; any later instance meeting the same
rule is added to that file with its evidence before the run continues.

No prediction, gate, cap or stop rule changes. Trajectories written so far are untouched.

**Addendum to the third deviation, 3 October 2026, 00:25 EDT.** Six candidate-2 trajectories
(apache__druid-15402, axios__axios-4731, -5085, -6539, astral-sh__ruff-15356, -15626) ended
with `exit_status: OpenRouterAPIError` whose cause is `Failed to resolve 'openrouter.ai'` — the
operator's laptop lost DNS while asleep on 2 October between 10:38 and 11:06, mid-trajectory.
These are not endpoint failures (the endpoint was never reached) and not agent failures; they
are no attempt. Rule: a trajectory whose terminal error is a network failure on the operator's
machine is quarantined (`data/agent_axis_deepseek/quarantine/`) and the instance regenerates
under the same seed. `scripts/resume_amendment13.sh` applies this automatically. The six
quarantined files are kept for inspection and are not in the bundle.

**Addendum, 3 October 2026, 01:55 EDT.** One candidate-2 trajectory (projectlombok__lombok-2792)
was written at the moment the generator was killed after an 86-minute hang, carrying the
agent's own `RepeatedFormatError` exit at step 45. Whether the hang influenced the exit cannot
be established. It is **kept as a failure** — regenerating a failure would be a favourable
re-roll — and disclosed in the outcome as ambiguous. The quarantine rule above applies only to
network-level errors on the operator's machine, never to an exit the agent itself produced.

## Amendment 13 outcome, recorded 3 October 2026, 09:45 EDT

**Generation.** 454 trajectories plus 2 declared endpoint failures = 456 slots over 114
Multilingual instances, k=4, 100 steps, cost_limit $1.00, 16k max tokens. External OpenRouter
spend $8.77 (cap $20). Exits over the 454: 436 Submitted, 16 RepeatedFormatError, 2 LimitsExceeded; plus the 2
declared EndpointFailure slots = 20 empty slots, all counted as failures. Submit rate 0.956
(436/456). Median 31 assistant turns. The two LimitsExceeded exits are at 91 and 99 turns with
costs $0.11 and $0.04: the **step** budget bound them, not the $1.00 cost cap, which bound
nothing. The budget therefore shaped 2 of 456 attempts (0.4%); the two step-limited attempts are cand2 hashicorp__terraform-34900 and
cand3 sharkdp__bat-1892. Wall clock ~40 h across 1–3 October, dominated by
the operator machine sleeping (three deviations and two addenda above); none touched c, u or gold.

**Gold.** All four candidates fully graded: 365/456 resolved (80.0% per attempt), 0 harness
errors, 24/24 oracle checks pass. Unconditional failure rate 0.200 (empties as failures).
102/114 instances have at least one resolved candidate. Reports carry the stale label
`a7_gpt6luna_cand{N}` (cosmetic; the readout keys by instance and candidate).

**A7-P1 (capability): PASS.** The equal-budget comparator (gpt-5-mini, k=1, 100 steps, the
same 114 instances, `data/agent_axis_gpt5mini_100`): 114 attempts, 11 empty patches, 48
resolved, 66 unresolved by gold (the wrapper evaluates empty patches too; the 11 are among the
66), 0 agent-caused crashes, 0 infrastructure errors — so the two classification ways coincide:
unconditional failure rate **0.579**. Seventeen instances were regenerated at two workers
(16 `RateLimitError` exits on OpenAI's TPM ceiling at six workers, plus lombok-3009, whose
sandbox the idle-reaper stopped during a retry loop; `MERGE_NOTE.md`): 15 submitted, 2 hit
the step limit. One gold was run on the merged predictions (`a13_gpt5mini_100_merged`, 0
errors, oracles clean). DeepSeek's unconditional failure rate is 0.200 counting the two
declared endpoint failures as failures (way B) and 0.196 excluding them (way A, 89/454); the
gap is **0.379** (way B) against the registered bar of 0.10. For reference the September
40-step comparator band was 0.476–0.752; the 100-step budget made gpt-5-mini *worse*
(0.579 within that band's upper half), which is consistent with Amendment 11's reading that
the budget was not what separated the agents.

**A7a §2 selector gate: FAIL.** The registered selector is the gpt-5 judge (J1). Its pool
AUROC for detecting failure over the 436 graded non-empty attempts is **0.552** (bar 0.65),
reproduced independently with the detection sign checked (0.448 with the sign flipped). The
selected set does fail less than the frozen random picks (0.158 vs 0.175), so the second
half of the gate holds, but the gate is conjunctive. For completeness the five reward
models' pool AUROCs on the same attempts are 0.464, 0.487, 0.517, 0.419, 0.523 (registered
sign): every critic is at chance on this agent.

**Inference floor.** Committed failures: 17 (judge arm, n=113) and 19 (4B arm, n=109),
both below the registered floor of 20.

**Endpoint (descriptive only).** Primary fixed-unit-weights ΔAUROC: judge arm +0.020
[−0.120, +0.149]; 4B arm +0.032 [−0.079, +0.140]. AUROC(u): 0.544 [0.400, 0.697] and
0.502 [0.356, 0.649]. Upper bounds far above +0.05: had the gate passed, the reading would
have been INCONCLUSIVE, as declared likely in advance. The readout's `sign_warning` on the
within-selected detection AUROCs (0.468/0.481) is the post-selection truncation the
heuristic cannot distinguish from a sign error; the pool-level check above settles the sign.

**Verdict: DESCRIPTIVE ONLY**, on the registered rule, for two independent reasons: the
selector gate fails and the committed set is under the floor. Reading: on the capable agent,
no available critic — five reward models or the gpt-5 judge — separates failures from
successes well enough to *be* a selector, so the strong-agent, strong-selector cell the
mentor's objection names could not be constructed with these instruments. The objection is
not retired and not narrowed by an equivalence bound; it is re-posed: the premise of
critic-selected trajectories is what fails first on a capable agent.

**Costs.** OpenRouter (DeepSeek generation, measured externally) $8.77; OpenAI: gpt-5 judge
$13.50 (scaffold estimate), gpt-5-mini comparator and reruns on the scaffold estimate; Modal
(sandboxes, GPU panel, six gold runs): the dashboard read $27.10 before Amendment 12 and $14.30 after
Amendment 13 (operator reading, 3 October), so Amendments 12 and 13 together cost **$12.80** against the
registered $22 cap. OpenAI balance after the run: $23.23 (operator reading). Registered
readout: `data/agent_axis_deepseek/readout_amendment7_deepseek.json`; the comparator function
double-counted evaluated empty patches in its first run (0.675) and was corrected to the set
union before this record was written (0.579, equal to the raw 66/114).

**Corrections and a late-recorded deviation, 3 October 2026, after Cursor's review of the
artifacts.** (i) The outcome's phrase "every critic is at chance on this agent" overstates:
three of the six point estimates exceed 0.5 (0.517, 0.523, 0.552). The registered statement
is that no critic clears the selector bar of 0.65; the artifacts now say that. (ii) The gloss
"it was 0.671 on gpt-5-mini" referred to the judge's detection AUROC *inside the selected arm*
of the September run (`comparison_to_gpt5_mini.arms.S_judge.raw_AUROC_detect.gpt5_mini`), a
different population from this pool AUROC; removed from the artifacts and stated correctly.
(iii) **Fourth deviation, recorded late.** Stage 2 above registers six workers. Generation was
resumed at 18 workers at 17:58 EDT on 2 October (ledger line `workers_raised_6_to_18`, commit
`b6a134e`) because the model endpoint's per-step latency bound a six-worker pool to about
0.2 trajectories a minute; seeds, limits and configs were unchanged. It was written to the
ledger at the time but omitted from this section until the review. (iv) "24 of 24 oracle checks" in the outcome above was **wrong**: only candidate 0's gold run
carried the oracle pairs (`--oracle 3`), so the four reports hold six oracle rows, all clean.
The artifacts now say 6 of 6 and the checker derives the count from the reports.

# Amendment 14, 6 October 2026 — a stronger judge as the selector on the Amendment 13 pool

**Written before any judge call on this pool by a gpt-6 model.** Amendment 13's selector gate
failed because no critic in the panel or the gpt-5 judge separated DeepSeek V4.1 Flash's
failures from its successes above the registered bar (pool AUROC 0.552; reward models
0.419–0.523; bar 0.65). The mentor asked the obvious question: the gpt-6 judges rank failures
at 0.71–0.82 on the gpt-5-mini trajectories (`readout_judges_gpt6.json`: gpt-6-luna 0.824 on
Verified, 0.707 on Multilingual; gpt-6-sol 0.776–0.826 in the paper). Were they tried here? They
were not. This amendment tries them, as selectors, on the pool that already exists.

## What runs
Two judges, gpt-6-sol and gpt-6-luna, condition J1 (the registered condition), every one of the
456 Amendment 13 candidates, through `scripts/llm_judges.py` as for the gpt-5 arm. No new
trajectories, no new gold, no Modal: the readout reuses the four candidate gold reports and the
frozen random picks. One smoke of 3 instances (12 candidates) per judge projects the cost first.

## Predictions, read under Amendment 7a exactly as for the gpt-5 arm
- **A14-P1 (a selector exists):** at least one of the two judges has pool AUROC for detecting
  failure over the 436 graded non-empty attempts **above 0.65**, and its selected set fails less
  often than the frozen random picks. If both fail the gate, the Amendment 13 reading stands
  and this amendment reports two more critics at or near chance on this agent.
- **A14-P2 / A14-P3:** for a judge that clears the gate, A7-P2 and A7-P3 as registered,
  including the floor of 20 committed failures and the +0.05 margin. Declared in advance: a
  judge that clears the gate commits *fewer* failures, so the floor is the likely obstacle, and
  the expected reading is INCONCLUSIVE or descriptive only; Amendment 15 exists for that case.
- Each judge is reported separately; no pooling of judges; no best-of-two selection after the
  fact. Both arms are reported whatever they show.

## Caps and stop rules
Price table in `llm_judges.py`: gpt-6-sol $2/$10 per million tokens, gpt-6-luna $0.10/$0.50.
Projected from the gpt-5 arm ($0.025 per judgment): about $15 for sol and $1 for luna. Cap
**$40 for both** on the scaffold estimate; a smoke projecting over $30 for one judge aborts.
OpenAI balance before the run is read by the operator and recorded in
`data/agent_axis_deepseek/spend_ledger_a14.txt`. No Modal spend. Output:
`data/agent_axis_deepseek/readout_amendment14_<judge>.json`, produced by
`scripts/readout_amendment7.py --judge <judge>` (the flag added for this amendment; default
unchanged, dry run and fixture test still pass).

# Amendment 15, 6 October 2026 — the remaining 186 Multilingual instances, conditional

**Written before Amendment 14's outcome is known. Runs only if Amendment 14 finds a judge above
the selector bar; otherwise it is not launched and this section stands as the reason.**

Amendment 13's committed sets held 17 and 19 failures against a floor of 20, and a selector
that clears the gate commits fewer failures still. SWE-bench Multilingual has 300 instances;
114 were registered for Amendment 13. This amendment extends the same run to the other 186
under identical settings: DeepSeek V4.1 Flash, k=4, 100 steps, cost_limit $1.00, 16k tokens,
`configs/mini_swe_deepseek_axis.yaml`, the closing sandbox environment, 18 workers (recorded as
an execution setting). Every candidate graded by the harness; the panel scored on Modal GPU;
the gate-clearing judge from Amendment 14 judges every candidate; the equal-budget gpt-5-mini
comparator extended to the same 186 at k=1. Predictions are A7-P1, A7-P2, A7-P3 as registered,
read on the union of 300 instances with the Amendment 14 selector; the floor of 20 and the
+0.05 margin unchanged; the readout reports the 114 and the 186 separately as well as pooled.

Caps: OpenRouter $25 (projected ~$14 for 744 attempts at the measured $0.019), Modal $15
(gold and panel), OpenAI $30 (judge and comparator). Stop rules as Amendment 13: a cap exceeded
stops the pipeline before the next paid stage; the stage order and the resume script are those
of Amendment 13. Wall clock: about one day with the machine awake and plugged in.

## Amendment 14 outcome, recorded 6 October 2026, 14:05 EDT

Both judges judged all 456 candidates (J1): gpt-6-sol $3.64, gpt-6-luna $0.33 on the scaffold
estimate, against the $40 cap. **A14-P1 fails for both.** Pool AUROC for detecting failure over the 436
graded non-empty attempts: **gpt-6-sol 0.619**, **gpt-6-luna 0.569** (bar 0.65; gpt-5 was 0.552),
each reproduced independently with the detection sign checked. The second half of the gate also fails for
both: the selected sets fail more often than the frozen random picks (0.193 and
0.211 against 0.175). The selected arms hold 21 and 23 failures, above the
floor of 20, which shows the floor was not the binding constraint: the selector bar was. Descriptive
only, both arms: AUROC(u) 0.421 and 0.39; primary ΔAUROC -0.117 and
-0.141, intervals including zero. Readouts: `data/agent_axis_deepseek/readout_amendment14_<judge>.json`.

**Amendment 15 is not launched.** Its condition — a judge above the selector bar — is not met, and the
registration stands as the reason: more instances raise the failure count but cannot make a cell
inferential whose selector cannot select.

**Addendum, 11 October 2026 (defect #24).** The "fail more often than the frozen random picks"
sentence above depends on how a tied top score is broken. The judges' scores are integers and
tie at the top on 41 (gpt-6-sol) and 38 (gpt-6-luna) of 114 instances; the readout keeps the
first attempt. Under the last-attempt rule gpt-6-sol's picks fail at 0.175, the random rate
exactly, and gpt-6-luna's at 0.202. The gate's first half, the pool AUROC, involves no picks and
is unchanged, so A14-P1 still fails for both and Amendment 15 stays unlaunched.
