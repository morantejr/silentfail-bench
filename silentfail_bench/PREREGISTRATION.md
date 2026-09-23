# Pre-registration — Pilot 3: selection pressure n=16, visible-pass primary population

**Registered:** 2026-09-23, before any n=16 data existed. Code state: the git
commit that introduces this file. Config: `configs/anchor_free_rm4_n16.yaml`.

This document exists because the visible-pass stratum in pilot 2 was chosen
*after* seeing the data. Everything below is fixed before pilot 3 runs; any
analysis not listed here is exploratory and will be labelled as such.

## Population

Committed trajectories (best-of-*n* argmax by the selection critic) whose
**agent-visible tests passed** on the final step. Rationale: this is the
selection-conditional cell in the most literal sense — the agent's own
verification passed — and it is where a deployed system would ship the
trajectory. Trajectories whose visible tests failed are excluded from the
primary analysis (the panel correctly agrees they are bad; there is no silent
failure to detect).

## Positive class

`g = 0` within the population: passed visible tests, failed held-out gold.
("Silent failure.")

## Fixed design

| item | value |
|---|---|
| tasks | the same 40 HumanEval+ instances as pilots 1–2 (seed 20260922; `HumanEval/32` excluded, invalid oracle) |
| agent | `qwen2.5-coder:7b`, max 3 steps, temperature 0.7 |
| selection pressure | **n = 16** (was 4). Candidates 0–3 are byte-identical to pilot 2 by seed construction; 4–15 are new |
| selection critic `c` | `llama3.1:8b` LLM-judge, unchanged from pilots 1–2, whole-trajectory score, argmax retained |
| audit panel | 4 true scalar reward models, unchanged from pilot 2: GRM-gemma2-2B, GRM-Llama3.2-3B, Skywork-V2-Llama-3.2-1B, Skywork-V2-Qwen3-1.7B |
| `u` | sample std of rank-normalized audit scores; rank norms fit on **train groups only** |
| split | 70/30 by `problem_id`, SHA-256 bucket, seed 20260922 |
| exclusions | generation error, unparseable judge score, sandbox error |

## Primary metrics (Stage-0 pilot)

Within the population, on raw scores (no model fitting — positives will be
too few to fit a logistic honestly):

- **M1** `AUROC(u)` for silent failure.
- **M2** `AUROC(−c)` for silent failure (higher `c` = critic more confident).
- **M3** `AUROC(u) − AUROC(−c)`, with a group-clustered bootstrap CI
  (4,000 resamples, groups = `problem_id`), one-sided p for M3 > 0.

## Confirmatory metric (only if eligible)

Held-out `ΔAUROC = AUROC(M_{c+u}) − AUROC(M_c)` from the frozen logistic,
fit on train groups, evaluated on test groups, **reported only if the test
split of the population contains ≥ 10 positives.** Otherwise it is not
computed, and its absence is stated.

## Predictions, stated so they can fail

- **P1 (H1):** M1 > 0.5.
- **P2 (Goodhart):** M2 ≤ 0.5 — the selecting critic is at or below chance
  at spotting the failures it selected.
- **P3 (winner's curse):** the silent-failure rate among visible-pass
  trajectories is **higher at n=16 than at n=4** (pilot 2: 3/35 = 8.6%).

## What would count against H1

M1's clustered 95% CI including 0.5 with ≥ 10 positives. With fewer than
10 positives no conclusion is drawn in either direction; the run is then a
pressure measurement (P3) only.

## Reporting rules

- All three pilots reported together; nothing dropped.
- No stratum other than visible-pass is analysed as primary. Any other cut
  is labelled exploratory.
- Point estimates are reported with their positive count beside them, always.
- `corr(c, u)` is reported as a watch item; it was 0.45 in pilot 2.

## Known limits, declared up front

- Substitute task family (HumanEval+), problem-level split: **no cross-repo
  claim** is made from this pilot.
- The selection critic is an LLM judge, not a reward model, for continuity
  with pilots 1–2. The proposal specifies an RM; that swap is a separate,
  later comparison.
- Two audit RMs share a Llama-3.2 base. The panel is three families, not four.
