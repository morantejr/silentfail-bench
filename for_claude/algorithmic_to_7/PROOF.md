# Proof sketch for Claude to tighten

This is a sketch. The citable theorem is Afrin & Shihab, Proposition 4, arXiv:2608.08002, proved in their Appendix A.1. Specialize it. Do not claim a new identification theorem unless the specialization adds an assumption they do not have (one score per critic–trajectory cell, no replicate ratings).

## Setup, matched to this repo

For committed trajectory \(i = 1,\ldots,n\) and critic \(j = 1,\ldots,J\):

\[
r_{j,i} = q_i + b_j + c_i + u_{j,i}.
\]

- \(q_i\): true quality. Gold failure is \(g_i = 0\), and the paper's positive class for AUROC is failure, \(y_i = 1\{g_i = 0\}\).
- \(b_j\): critic stringency. Constant across trajectories. This is the "between-critic heterogeneity" in `paper/main.tex` line 262.
- \(c_i\): common-mode error. Same addend for every critic on trajectory \(i\). Depends on the trajectory. This is the shared mistake.
- \(u_{j,i}\): idiosyncratic residual, with \(\sum_j u_{j,i} = 0\) after the usual constraint, or mean zero across \(j\).

\(J\) is 4 on the registered panel and 7 on the zoo used for the diversity check. There is one score per cell. Shrout & Fleiss (1979) already note that with no replicate ratings the judge-by-target interaction and the pure error are not separable. Do not pretend the residual is pure noise.

The published \(u\) is the standard deviation of rank-normalized panel scores. It mixes \(b_j\) (through ranks) and \(u_{j,i}\). It does not see \(c_i\), for the reason below.

## The invariance

Let \(P = I - \mathbf{1}\mathbf{1}^\top / J\) be the projector off the all-ones vector. For a fixed trajectory, stack the \(J\) scores as \(\mathbf{r}_i\).

\[
P\mathbf{r}_i = P(q_i\mathbf{1} + \mathbf{b} + c_i\mathbf{1} + \mathbf{u}_i) = P\mathbf{b} + P\mathbf{u}_i,
\]

because \(P\mathbf{1} = 0\). So \(q_i\) and \(c_i\) are both annihilated. Any spread computed from \(P\mathbf{r}_i\) — standard deviation, variance, range — is unchanged if a constant is added to every critic on that trajectory.

**Property.** Critic-demeaned disagreement

\[
u_{\perp,i} = \mathrm{sd}_j(r_{j,i} - \bar{r}_{j\cdot})
\]

does not depend on \(c_i\) or \(q_i\). A failure that every critic shares with the selector is invisible to \(u_{\perp}\) and to published \(u\).

Afrin & Shihab, Proposition 4, is the same fact in their notation. For any scalar function \(h\) with mean zero under their prompt measure, replacing quality \(q\) by \(q+h\) and the error \(\mathbf{e}\) by \(\mathbf{e} - h\mathbf{1}\) leaves the score vector \(\mathbf{r}\) unchanged, leaves \(P\mathbf{e}\) unchanged, and shifts only the ensemble mean. Their one-sentence consequence: internal judge scores and their disagreement cannot identify response-dependent common-mode error without an external anchor or extra structure. Gold in this project is that external anchor for \(y\), and it still does not put \(c_i\) into \(u\).

Appendix A.1 of that paper is six lines: substitute, use \(P\mathbf{1} = 0\), observe that every statistic of \(\mathbf{r}\) is unchanged while the common mode moved. Quote that argument. Do not restate it as original.

## What is vacuous, so you do not file it as the result

Double-centering

\[
\tilde{r}_{j,i} = r_{j,i} - \bar{r}_{j\cdot} - \bar{r}_{\cdot i} + \bar{r}_{\cdot\cdot}
\]

removes both main effects. The within-trajectory standard deviation of \(\tilde{r}_{\cdot,i}\) equals \(u_{\perp,i}\), because \(\bar{r}_{\cdot i}\) is constant in \(j\). If a script reports that two-way residual spread differs from critic-demeaned spread, the script is wrong.

The trajectory effect \(\beta_i = q_i + c_i\) is identified only up to the usual ANOVA constraints (Scholz notes; sum-to-zero). It is confounded with true quality. Ranking by \(\beta_i\) asks whether the panel mean beats the selector. That is a different, weaker question, and the learned combiner in `readout_pi_batch1.py` already asked a version of it.

## What is not vacuous

1. **Variance components** on the committed score matrix. Report the share of sum of squares in the critic main effect, the trajectory main effect, and the residual \((J-1)(n-1)\) term. The prose at `paper/main.tex` line 262 says the spread is dominated by between-critic heterogeneity. The table either supports that sentence or forces it to be narrowed. Brennan's NCME module, Equation for the crossed decomposition, is the language: rater main-effect variance enters absolute error and drops out of relative error. \(u_{\perp}\) is the relative-error piece.

2. **\(\Delta\)AUROC of \(u_{\perp}\) over \(c\)** on the committed rows, same positive class (failure), same repository clustering the paper already uses. Compare to published \(u\) on the same rows. Post-hoc. If they match, say the rank-normalization was already doing most of the critic-demeaning, and the algorithmic content is the invariance plus the variance split.

3. **Tie to the correlation that is already measured.** `readout_pi_batch1.json` has `E2_diversity.error_corr_range` of about 0.71–0.89 on Verified and 0.74–0.84 on Multilingual (`paper/main.tex` line 264). In the exchangeable model Afrin writes as \(e_j = c + u_j\),

\[
\mathbb{E}[D] = \frac{J-1}{J}\mathrm{Var}(u_j), \qquad \mathrm{Var}(\bar{e}) = \mathrm{Var}(c) + \mathrm{Var}(u_j)/J.
\]

High pairwise error correlation means \(\mathrm{Var}(c)\) is most of the error variance, which is the component \(u_{\perp}\) deletes. Compute the correlation of the residuals \(r_{j,i} - \bar{r}_{j\cdot}\) the same way the script computes `error_corr`, and report it next to 0.71–0.89. The drop, if any, is how much of that correlation was critic location rather than shared trajectory error.

## What this does not prove

- It does not prove disagreement is useless on every agent. Amendment 13's selector gate failed for a different reason (no critic cleared 0.65), and that cell is descriptive only.
- It does not produce a statistic that recovers the 0.71–0.75 single-critic AUROC. Those numbers are individual critics seeing \(q_i + c_i + u_{j,i}\). Removing \(c_i\) cannot increase what a shared error hides.
- Nested \(\Delta\)AUROC is not a DeLong test. Demler, Pencina, and D'Agostino (2012, PMC3684152, Section 4) and Pepe et al. (2013, PMC3625503, Theorem 1) are why the paper already uses clustered out-of-fold estimators. Reuse that estimator. Do not introduce DeLong.
- Smith & Winkler (2006, §2.3) is the selection fact already in the paper: the argmax score is biased high. The invariance above is about the auditor, not a new proof of the optimizer's curse.

## Writing target, one paragraph

State \(u_{\perp}\), state the invariance in one sentence with the projector, cite Proposition 4 for the identification limit, cite the variance-component table and the post-hoc \(\Delta\)AUROC on the committed rows, and say the 0.71–0.89 band is the empirical reason the deleted component is the large one. If the numbers are null, the paragraph still stands. The property is the algorithm.
