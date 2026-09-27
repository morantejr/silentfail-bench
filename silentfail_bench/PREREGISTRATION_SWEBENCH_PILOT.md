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
