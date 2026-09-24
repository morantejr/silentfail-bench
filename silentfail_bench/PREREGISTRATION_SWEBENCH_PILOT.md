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
