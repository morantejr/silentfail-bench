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
