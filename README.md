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

> **Status: draft v0.9.22 (2026-10-13) — the registered endpoint, computed four
> Public mirror of `main`: https://github.com/morantejr/silentfail-bench (the links in the papers point there; the working repository is `silentfail-bench-private`).
> ways, puts disagreement's incremental value over the selector's own score at
> indistinguishable from zero, loosest pooled upper bound +0.12 over 47 failures
> and 46 repositories.** A SWE-bench Verified pilot (41 instances, 12 positives)
> plus a pre-registered replication on SWE-bench Multilingual (114 instances, 37
> analysed repositories, 35 positives): eight reward-model critics at 4k/8k/16k
> and windowed, five LLM judges across two model generations, test–retest, two
> selection regimes, official executed gold throughout. **H1 not supported and
> the lack of detectable gain replicates:** panel disagreement 0.565 [0.43, 0.71] against the
> critic's own 0.560; the Verified windowed positive (0.686) fell to 0.541; the
> lines-added baseline fell to 0.508. Best-of-4 by a 4B reward model did not
> concentrate failure, though selection did remove that selector's own signal on
> Verified (0.81 → 0.61). Re-selecting with `gpt-5` leaves the result unchanged
> in direction; that cell is reported as consistent with the null rather than
> establishing it, since its upper bound still exceeds the +0.05 this project
> declared actionable. Newer judges sharpen the dissociation: `gpt-6-sol` is the
> best single detector in the study at 0.776–0.826 while the disagreement of that
> newer pair is the worst at 0.41. **The agent axis ran** (Amendment 13, 1–3 October):
> DeepSeek V4.1 Flash, 456 attempts over the registered 114 Multilingual instances at
> 100 steps, all graded. The capability gate passed (failure rate 0.200 against 0.579
> for gpt-5-mini at the same 100-step cap, four attempts per instance versus one), but the registered selector gate failed: the gpt-5
> judge's pool AUROC for detecting this agent's failures is 0.552 against a bar of 0.65
> and the reward models sit at 0.419–0.523, so the cell is reported as descriptive only; a stronger judge as selector (gpt-6-sol 0.619, gpt-6-luna 0.569, Amendment 14) does not change that.
> No critic cleared the registered selector bar on that agent. The registered readout is
> `silentfail_bench/data/agent_axis_deepseek/readout_amendment7_deepseek.json`; the
> deviations are in the Amendment 13 section of the preregistration.
> Twenty-two numbered defects (including seven critic/instrument failures) are recorded in
> [`silentfail_bench/DEFECTS.md`](silentfail_bench/DEFECTS.md), including one that
> manufactured a result large enough to reverse the paper. Paper:
> [`paper/main.pdf`](paper/main.pdf), 9-page submission version (body within seven)
> [`paper/main_short.pdf`](paper/main_short.pdf). See [`UPDATES.md`](UPDATES.md).
Before spending Modal credit (resets **1 Oct**): [`CLAUDE_HANDOFF_OCT1.md`](CLAUDE_HANDOFF_OCT1.md)
and `silentfail_bench/scripts/preflight_pre_oct1.sh`.

## Layout

| Path | What it is |
|---|---|
| [`silentfail_bench/`](silentfail_bench/) | The runnable benchmark: harness, configs, schema, tests |
| [`silentfail_bench/FREEZES.md`](silentfail_bench/FREEZES.md) | Locked Stage-0 decisions, plus what the free-compute path cannot honor |
| [`silentfail_bench/DEFECTS.md`](silentfail_bench/DEFECTS.md) | Defects found before the first real run, and the verified sandbox posture |
| [`silentfail_bench/MENTOR_NOTE.md`](silentfail_bench/MENTOR_NOTE.md) | Current open decisions |
| [`UPDATES.md`](UPDATES.md) | **Research log — one entry per stage: done / learned / changed** |
| `SilentFail_AAAI_Proposal.pdf` / `.tex` | The original proposal |
| `paper/main.pdf` | The paper (full version) |
| `paper/main_short.pdf` | Nine-page submission version; body within seven pages, then references |
| `paper/main_agentverify.pdf` | Anonymous IEEE version, 8 pages including references; **generated** by `paper/make_agentverify.py` from `main_short.tex` |
| `paper/main_symposium.pdf` | Two-page extended abstract for the Agent Evaluation Science symposium on the venue's AES template (`paper/aes/`, XeLaTeX); named, references after page 2 as the call allows, numbers pinned by the checker |
| `paper/frontier_main.pdf` | The long paper with its contribution framed at the strongest level the data carry (measurement, bound, critic collapse, follow-up); **generated** by `paper/make_frontier.py` from `main.tex`, no new numbers; not a submission artifact |
| `silentfail_bench/data/agent_axis_deepseek/readout_amendment7_deepseek.json` | Amendment 13 registered readout: A7-P1 PASS, selector gate FAIL, descriptive only; every input it reads is tracked |
| `silentfail_bench/data/agent_axis_deepseek/readout_amendment13_panel.json` | Pool AUROC of every critic on the DeepSeek attempts, both signs; **generated** by `scripts/readout_amendment13_panel.py` |
| `paper/audits/independent_recompute_2026-10-11/` | Independent recomputation of every abstract number from the raw files with code that imports nothing from this repository (`recompute.py`, `RESULTS.md`); all reproduce; surfaced defect #24 |
| `paper/submissions/` | One venue-formatted PDF per target workshop (four AAAI-27 workshops, AGENT and LLM4Code at ICSE, the SANER track), **generated** by `paper/make_submissions.py` from `main_short.tex`; never hand-edited |
| `silentfail_bench/data/agent_axis_deepseek/ENDPOINT_FAILURES.json` | The two declared endpoint failures with their evidence and the rule (Amendment 13, third deviation) |
| [`archive/drafts/`](archive/drafts/) | Earlier drafts and write-ups, kept for the record |

## Where this is being submitted

Recorded here because the repository that builds the submission did not say, while
`paper/make_agentverify.py` already hard-codes the venue's format requirements.

| Target | Artifact | Deadline | Status |
|---|---|---|---|
| AGENTVERIFY 2027 @ ICSE | `paper/main_agentverify.pdf` — anonymous IEEE, 8 pp incl. refs | abstract 1 Nov, paper 6 Nov (workshop CFP) **vs** 13 Nov (ICSE umbrella) | **deadline disputed — ask the chairs** |
| Agent Evaluation Science, Fall 2026 NYC | 4–6 pp body or 2 pp abstract, refs excluded, not anonymous | registration 20 Oct, submission 25 Oct | non-archival, presentation only; requires attendance |

These dates come from the author's venue trackers dated 2026-09-29, not from a CFP
read by this repository, and the Nov 6 / Nov 13 conflict is unresolved. Treat 6 Nov as
the conservative planning date until the organisers confirm. `main_short.pdf`'s body is
seven pages and therefore does **not** conform to the symposium's 4–6 page limit; the
2 pp abstract is the conforming option.

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


## Large artifacts and data handling

- `silentfail_bench/data/replication_multilingual/rescoring_bundle.jsonl` (39 MB) is tracked
  deliberately: it is the input every critic scored, and without it no result in the paper can
  be re-derived. It is kept by an explicit `.gitignore` exception.
- `silentfail_bench/data/**/traj/` (raw agent trajectories) is **not** tracked. It contains
  machine-local absolute paths, and one trajectory captured a test-fixture RSA private key and
  webhook secret that the agent read from the upstream Docusaurus repository. Both were
  redacted in place; neither ever reached a tracked file or any commit (verified with
  `git log --all -S`). If you copy this folder rather than cloning it, that directory travels
  with it.
- Top-level LaTeX build artifacts (`*.aux`, `*.log`, `*.out`) are ignored but present on disk.
