# SilentFail-Bench

**Among critic-*selected* LLM-agent trajectories, does disjoint audit-panel
uncertainty $u$ improve held-out failure detection *beyond* the selecting
critic's own score $c$?**

Primary estimand: $\Delta\mathrm{AUROC} = \mathrm{AUROC}(M_{c+u}) - \mathrm{AUROC}(M_c)$
on held-out repositories. Positive class = failure ($g=0$).

Theory in one line: under best-of-$n$, selection concentrates critic mistakes
(Goodhart / winner's curse), so the trajectories a critic is most confident
about are exactly where its blind spots collect. **A null result is publishable**
— provided the study is powered well enough for a null to mean something.

> **Status: draft v0.6 (2026-09-25) — registered endpoint computed four ways: disagreement's incremental value over the selector is indistinguishable from zero under all of them, loosest upper bound +0.12 AUROC, pooled over 47 positives.**
> SWE-bench Verified pilot (41 instances, 12 positives) plus a pre-registered
> replication on SWE-bench Multilingual (114 instances, 41 repos, 35 positives),
> seven reward-model critics at 4k/8k/16k and windowed, three LLM judges,
> test–retest, official executed gold throughout. **H1 not supported and the
> null replicates:** panel disagreement 0.565 [0.43, 0.71] vs the critic's own
> 0.560; the Verified windowed positive (0.686) fell to 0.541 [0.40, 0.68];
> the lines-added baseline fell to 0.508; best-of-4 by a 4B reward model did
> not concentrate failure (control), but selection removed the selector's own
> signal (0.81 → 0.61 on Verified). Paper:
> [`paper/main.pdf`](paper/main.pdf). See [`UPDATES.md`](UPDATES.md).

## Layout

| Path | What it is |
|---|---|
| [`silentfail_bench/`](silentfail_bench/) | The runnable benchmark: harness, configs, schema, tests |
| [`silentfail_bench/FREEZES.md`](silentfail_bench/FREEZES.md) | Locked Stage-0 decisions, plus what the free-compute path cannot honor |
| [`silentfail_bench/DEFECTS.md`](silentfail_bench/DEFECTS.md) | Defects found before the first real run, and the verified sandbox posture |
| [`silentfail_bench/MENTOR_NOTE.md`](silentfail_bench/MENTOR_NOTE.md) | Current open decisions |
| [`UPDATES.md`](UPDATES.md) | **Research log — one entry per stage: done / learned / changed** |
| `SilentFail_AAAI_Proposal.pdf` / `.tex` | The current proposal (the only PDF in the repo) |
| [`archive/drafts/`](archive/drafts/) | Earlier drafts and write-ups, kept for the record |

## Two things a reader should know up front

**1. The n=200 plan is underpowered.** Using a null-calibrated clustered test,
detecting $\Delta\mathrm{AUROC} \approx 0.045$ at 80% power needs roughly
**1,900 trajectories**; the proposal's 0.03 MDE needs more than 3,800. At
n=200, power against 0.045 is about **0.24** — so a null there would be
indistinguishable from underpowering. This is an open design decision, not a
settled one. (The test is conservative, α≈0.01 against a nominal 0.05, so
these figures are pessimistic.)

**2. The free-compute pilot measures a different construct than the proposal
theorizes about.** Real SWE-bench gold needs per-instance Docker images; on a
container-less host the pilot substitutes HumanEval+ (gold is genuinely
executed, but the split unit is `problem_id`, so cross-repo generalization is
untested), and the critics are LLM-as-judge rather than trained reward models,
so $u$ is judge disagreement rather than RM-ensemble disagreement. Every logged
row carries `critic_type` and `is_true_reward_model` so this can never be
quietly overstated. See `FREEZES.md` for the full honored/not-honored table.

## Quick start

```bash
cd silentfail_bench
python3 -m pip install -e ".[dev]"
python3 -m pytest -q                      # mock path, no GPU, no network
python3 scripts/fetch_humaneval.py        # rebuild derived task data
```

The free-compute pilot (Ollama, ~22 GB, $0) is documented in
[`silentfail_bench/README.md`](silentfail_bench/README.md).

## Design commitments

- $c$ is the selection critic's score of the **whole committed trajectory**, not
  a mean of per-step scores. Best-of-$n$ ranks full trajectories and commits the argmax.
- The selection critic is **never** a member of the audit panel.
- Gold is **trajectory-level** — failed tests do not imply every step was wrong.
- Rank-normalizations are fit on **train groups only**.
- If the known-correct reference solution does not pass the gold harness, that
  instance is excluded as an invalid oracle rather than counted as a failure.
