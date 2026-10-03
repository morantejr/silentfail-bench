# Read this first

Folder for Claude. Goal: raise the **algorithmic** score of SilentFail-Bench from about 4 toward 7 by turning one existing diagnosis into a procedure with a stated property.

The paper already has the evaluation-method contribution (selected population, selector baseline, executed gold, pre-registered bars). That score is already high. Do not spend this folder on another agent, another reward model, or a sharper abstract. Those do not move the algorithmic number.

Repo: `/Users/jmorante/Downloads/xtail_spec`, branch `main`. Today is 3 October 2026. Amendment 13 has already run and is **descriptive only**. Do not describe it as "the null holds for capable agents."

## Read order

1. `PROOF.md` — the claim, the invariance that makes a naive residual vacuous, and the measurement that is not vacuous.
2. `PAPERS_AND_RESOURCES.md` — open PDFs and docs. Afrin & Shihab, Proposition 4, is the theorem to specialize. Do not invent a stronger theorem than that paper proves.
3. `DATA.md` — paths and fields for the matrices that already exist. No new trajectories.

## What "7" means here

A reader can point to a named statistic, a property it has, and a table on these data.

- **Statistic.** Critic-demeaned disagreement: subtract each critic's mean on the training rows, then take the within-trajectory spread. Call it \(u_{\perp}\).
- **Property.** \(u_{\perp}\) is invariant to any shock that hits every critic on a trajectory by the same amount. A mistake shared with the selector is exactly such a shock. So \(u_{\perp}\) cannot audit it. Afrin & Shihab, Proposition 4, is the external statement of this; `PROOF.md` specializes it to one score per cell.
- **Table.** On the committed rows, variance components of the score matrix (critic main effect, trajectory main effect, residual) and \(\Delta\)AUROC of \(u_{\perp}\) over \(c\), beside the published \(u\). Label it post-hoc. The diversity analysis that measured pairwise error correlation 0.71–0.89 is already exploratory (`paper/main.tex` line 283).

If the residual spread matches published \(u\), write that. The contribution is then the property plus the variance split, not a new AUROC. If critic-demeaning moves the AUROC, that movement is the empirical half. Either result is allowed to be null.

## Hard constraints

- Python: `silentfail_bench/.venv-swebench/bin/python` for project code. `statsmodels` may be absent there; double-centering needs only NumPy.
- Do not call a paid API. Do not spend Modal or GPU. Do not regenerate trajectories.
- Do not edit `silentfail_bench/data/`, readout JSON, `paper/main.tex`, `paper/main_short.tex`, or `paper/main_agentverify.tex` unless the PI asks. This folder is the work surface.
- A new analysis is post-hoc. Date it. Do not call it a registered endpoint.
- The checker `python3 paper/check_consistency.py` must stay exit 0 if any paper sentence changes later. Not this task.
- Do not cite a URL this folder marks UNVERIFIED.

## Provenance

Gathered 3 October 2026 by three subagents, then checked here against the Afrin HTML for Proposition 4:

- Papers: [Open papers](0c787974-f30c-41a0-8444-fbae74be5fb5)
- Free notes and code: [Free resources](8e210e2b-57c1-4de1-bb11-bddb1141ae6a)
- Matrix paths: [Panel matrices](86ab27b1-e0fd-4564-988a-184b23aac408)

Proposition 4's statement in `PROOF.md` was read from the arXiv HTML, not from the subagent's paraphrase.
