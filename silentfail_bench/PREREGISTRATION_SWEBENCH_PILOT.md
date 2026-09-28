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

**Cost, declared.** ~$2 generation and ~$9 judge on OpenAI credit; ~$7 of
Modal, of which ~$3 is free-tier and the remainder is a small overage before
the 1 October reset. Cap: $15 OpenAI, $12 Modal.

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
