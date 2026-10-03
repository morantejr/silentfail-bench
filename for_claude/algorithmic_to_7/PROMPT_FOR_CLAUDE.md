# Prompt for Claude

Open `/Users/jmorante/Downloads/xtail_spec` and work from the repository root.

Read these files in order before doing anything:

1. `for_claude/algorithmic_to_7/README.md`
2. `for_claude/algorithmic_to_7/PROOF.md`
3. `for_claude/algorithmic_to_7/PAPERS_AND_RESOURCES.md`
4. `for_claude/algorithmic_to_7/DATA.md`
5. `paper/main.tex`, especially lines 262–266 and 283
6. `silentfail_bench/scripts/readout_pi_batch1.py`, especially the E2 and learned-combiner sections

## Goal

Determine whether the existing score matrices support an algorithmic contribution strong enough to move SilentFail-Bench's algorithmic novelty from roughly 4/10 toward 7/10.

The proposed contribution is **critic-demeaned disagreement**:

\[
u_{\perp,i}
=
\operatorname{sd}_j\!\left(r_{j,i}-\widehat{\alpha}_j\right),
\]

where each critic effect \(\widehat{\alpha}_j\) is fit on training rows only. Its stated property is invariance to any trajectory-specific common shock applied equally to every critic. Use Afrin and Shihab, Proposition 4, as the citable identification result. Do not present their theorem as ours.

This is a post-hoc analysis. Its value may be a null result. Do not optimize choices after seeing AUROCs.

## First: audit the proposal

Before writing code, assess the mathematical proposal critically.

1. Verify the projector argument and the specialization of Afrin and Shihab Proposition 4.
2. Confirm that taking within-trajectory spread after full two-way double-centering is algebraically identical to taking spread after critic demeaning. Treat any numerical difference as a bug.
3. State precisely what \(u_\perp\) removes and what it cannot identify.
4. Decide whether calling \(u_\perp\) an “algorithm” is defensible. If it is only a diagnostic statistic, say so directly.
5. Identify any claim in `PROOF.md` that is too strong, incorrectly attributed, or unsupported by the cited source.

Record this audit at the beginning of:

`for_claude/algorithmic_to_7/RESULTS.md`

## Analysis

Use only existing, tracked inputs. Run the analysis separately on:

- SWE-bench Verified
- SWE-bench Multilingual

For each benchmark, analyze:

- the registered four-critic panel;
- the seven-critic zoo;
- the committed population used by the existing batch-1 E2/learned-combiner analysis.

Fit critic means on `split == "train"` rows only and apply them to all analyzed rows. Never fit normalization or critic effects on held-out rows. Positive class is failure: `y = 1` when `g == 0`.

Compute and report:

1. analyzed rows, failures, and repository count;
2. missing score cells and the rule used for them;
3. two-way additive sum-of-squares shares:
   - critic main effect;
   - trajectory main effect;
   - residual;
4. AUROC of:
   - selector score \(c\), with the sign convention matching the paper;
   - published disagreement \(u\);
   - critic-demeaned disagreement \(u_\perp\);
5. the existing paper endpoint for adding \(u_\perp\) to \(c\), using the same four estimators and repository-aware implementation already used by the project:
   - leave-one-repository-out logistic;
   - clustered five-fold logistic;
   - within-fold size-weighted;
   - fixed unit weights;
6. repository-clustered 95% intervals, matching the project's existing bootstrap and seed conventions;
7. pairwise critic error-correlation ranges before and after critic demeaning, using the same error definition as `readout_pi_batch1.py`;
8. a numerical identity check showing that double-centered residual spread equals \(u_\perp\) to floating-point tolerance.

Do not use DeLong. Do not substitute a row bootstrap for repository clustering. Do not pool Amendment 13 DeepSeek rows with the original benchmarks.

## Implementation rules

- Use `silentfail_bench/.venv-swebench/bin/python`.
- No paid APIs, Modal, GPU work, downloads, or trajectory regeneration.
- Reuse existing project functions for population construction, rank normalization, estimators, and clustered intervals wherever possible.
- Do not edit anything under `silentfail_bench/data/`.
- Do not edit `paper/main.tex`, `paper/main_short.tex`, `paper/main_agentverify.tex`, README, preregistration, mentor note, checker, or existing readouts.
- Put any new analysis script under:

  `for_claude/algorithmic_to_7/`

- The script may write only into that folder.
- Make the script deterministic and print its inputs, population counts, and seed.
- Do not commit.

## Deliverables

Create:

1. `for_claude/algorithmic_to_7/analyze_residual_disagreement.py`
2. `for_claude/algorithmic_to_7/results.json`
3. `for_claude/algorithmic_to_7/RESULTS.md`

`RESULTS.md` must contain:

- the mathematical audit;
- one compact results table per benchmark;
- exact commands used;
- input paths and git-tracking confirmation;
- limitations;
- a hostile-reviewer assessment;
- a clear recommendation: **promote**, **keep as exploratory support**, or **drop**.

## Decision standard

Recommend **promote** only if all of these hold:

1. the statistic has a clear property beyond merely being another feature;
2. the property is stated without claiming common-mode errors become observable;
3. the empirical table validates the paper's existing mechanism claim or materially improves the endpoint;
4. the result is not a tautology created by double-centering;
5. the contribution is distinguishable from Afrin and Shihab rather than a repackaging of Proposition 4;
6. the framing survives the objection: “This is a post-hoc rater fixed effect plus an existing impossibility result.”

Otherwise recommend **keep as exploratory support** or **drop**. Do not force a 7/10 conclusion.

End `RESULTS.md` with:

- a 1–10 algorithmic-novelty score before and after this analysis;
- the strongest defensible one-sentence contribution;
- the strongest reviewer objection;
- whether the evidence justifies changing the abstract;
- a proposed paper paragraph only if the recommendation is **promote**.

Run the script twice and verify byte-identical `results.json`. Then run:

```bash
python3 paper/check_consistency.py
git status --short
```

The consistency checker must remain green. The final status should show changes only inside `for_claude/algorithmic_to_7/` plus any unrelated pre-existing untracked files. Report any violation and stop rather than repairing files outside the folder.
