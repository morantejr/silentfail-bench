# Papers and free resources

URLs below were returned by web search on 3 October 2026. Proposition 4 was re-read from the arXiv HTML in this session. Other entries were checked by the papers subagent and were not all re-opened here. If a PDF 404s, drop the citation. Do not replace it with a guessed arXiv id.

## The one theorem to specialize

**Afrin, Fariya, and Ibne Farabi Shihab.** Evaluator Ensembles Under Reward Hacking: Covariance Geometry and Finite-Search Guarantees. arXiv:2608.08002, 2026.

- HTML: https://arxiv.org/html/2608.08002
- PDF: https://arxiv.org/pdf/2608.08002
- Read Proposition 4 (identification limit) and Appendix A.1. The statement, from the HTML: for a centered scalar \(h\), the shifted pair \(q^{(h)} = q+h\), \(\mathbf{e}^{(h)} = \mathbf{e}-h\mathbf{1}\) leaves the scores unchanged, leaves \(P\mathbf{e}\) unchanged, and shifts only the ensemble mean. Internal scores and disagreement do not identify response-dependent common-mode error.
- Also their display (13): under \(e_j = c + u_j\), disagreement energy is \(\mathrm{Var}(u_j)\) and ensemble-mean variance keeps \(\mathrm{Var}(c)\). That is the split \(u_{\perp}\) implements.
- The paper's own scope note says this is narrower than introducing ensembles or disagreement sampling. Cite it as the identification lemma, not as a result this project discovered.

## Identification and rater decomposition

| Source | Free URL | Read | Use |
|---|---|---|---|
| Shrout & Fleiss 1979, ICC | https://aliquote.org/cours/2012_biomed/biblio/Shrout1979.pdf | Eq. 2; ICC(2,1) vs ICC(3,1). APA original is paywalled. | One rating per cell: interaction and error are not separable. |
| Brennan 1992, Generalizability Theory, NCME module | https://ncme.org/wp-content/uploads/2025/10/Module-14-Generalizability-Theory-Brennan-Winter-1.pdf | Crossed decomposition; absolute vs relative error | Rater main effect drops out of relative error. \(u_{\perp}\) is relative error. |
| Scholz, STAT 421 constraints notes | http://faculty.washington.edu/fscholz/DATAFILES/constraints.pdf | Additive model, sum-to-zero | \(\alpha_j+\beta_i\) needs a constraint; levels are not unique. |
| Dawid & Skene 1979 | https://crowdsourcing-class.org/readings/downloads/ml/EM.pdf | §§2–3 | Without labels, observer error rates are not unique (latent-class version of the same limit). JSTOR: https://www.jstor.org/stable/2346806 |
| Raykov, uncorrelated errors in psychometric theory | https://pmc.ncbi.nlm.nih.gov/articles/PMC5965613/ | Parallel tests, models M1 vs M2 | \(\mathrm{Cov}(X_1,X_2)=\mathrm{Var}(T)+\mathrm{Cov}(E_1,E_2)\). Shared error is not testable from the raters alone. |
| Williams, Identification of the Linear Factor Model | https://www2.gwu.edu/~forcpgm/2018-002.pdf | Footnote 3 | An extra factor for the shared correlation is often unidentified with few measurements. |
| Two-way ANOVA, one observation per cell | https://www.math.wichita.edu/~cma/stat764/ch5.pdf | §6 | \(Y_{ij}=\mu+\alpha_i+\beta_j+\varepsilon_{ij}\). |

Lord & Novick 1968 has no free Chapter 2. Archive.org borrow record only. Do not cite a PDF of it.

## Correlated judges, the empirical neighbors

These support the sentence "the shared component is large." They are not the proof.

| Paper | Free URL | Read |
|---|---|---|
| Goel et al. 2025, Great Models Think Alike. ICML. | https://arxiv.org/html/2502.04313 and PMLR https://proceedings.mlr.press/v267/goel25b.html | §5, error correlation rising with capability |
| Kim, Peng, Garg et al. 2025, Correlated Errors in Large Language Models. ICML. | https://arxiv.org/abs/2506.07962 and PMLR https://proceedings.mlr.press/v267/kim25e.html | §4, LLM-as-judge case |
| Kohli 2026, Nine Judges, Two Effective Votes. arXiv:2605.29800 | https://arxiv.org/html/2605.29800 | Effective sample size under correlation. Already in the paper's related work. |
| Hossain, Yousefi, Lim 2026, Agreement Overstates Evidence. arXiv:2609.22512 | https://arxiv.org/abs/2609.22512 | Pairwise error correlation cuts independent votes. |
| Liu 2026, LLMs as a Jury. arXiv:2607.10139 | https://arxiv.org/pdf/2607.10139 | §5.2 shared-error floor when the panel agrees on a wrong answer |
| Yu et al. 2026, Heterogeneous Judge-Aware Ranking. arXiv:2605.05073 | https://arxiv.org/pdf/2605.05073 | §4, \(S=\gamma\mu^\top+UV^\top\), Proposition 1. Consensus and residual disagreement separated only after their normalizations. |
| CARE, Zhao et al. 2026. arXiv:2603.00039 | https://arxiv.org/pdf/2603.00039 | Proposition 4.1. The contrast: latent quality is separable only under their extra assumptions. Afrin explicitly positions Proposition 4 as the no-assumption boundary against this paper. |
| Lail 2026, Decomposing LLM-Judge Uncertainty. arXiv:2609.06444 | https://arxiv.org/pdf/2609.06444 | §3 and Appendix B. Spread sums disagreement and ignorance. |
| CyclicJudge, Zhu et al. 2026. arXiv:2603.01865 | https://arxiv.org/pdf/2603.01865 | §3.1: a centered judge offset is not the common mode; shared bias stays in the panel mean. |
| Verga et al. 2024, Replacing Judges with Juries. arXiv:2404.18796 | https://arxiv.org/abs/2404.18796 | The pooling method the identification result restricts. |
| Baan et al. 2022, Stop Measuring Calibration When Humans Disagree. arXiv:2210.16133 | https://arxiv.org/pdf/2210.16133 | Disagreement is not calibration. Outside the 2023–2026 window; still the right older cite. |

Jiang, DiscoUQ, arXiv:2603.20975 treats disagreement as a confidence feature. Cite only as the proposal this property restricts.

## Nested AUROC and selection, already the paper's estimators

Do not replace the paper's clustered estimators with these tests. Read them so the new paragraph does not cite DeLong.

| Paper | Free URL | Read |
|---|---|---|
| Demler, Pencina, D'Agostino 2012 | https://pmc.ncbi.nlm.nih.gov/articles/PMC3684152/ | §4, DeLong on nested models |
| Pepe, Kerr, Longton, Wang 2013 | https://pmc.ncbi.nlm.nih.gov/articles/PMC3625503/ | Theorem 1: no incremental value iff \(P(D=1\mid X,Y)=P(D=1\mid X)\) |
| Smith & Winkler 2006, The Optimizer's Curse | https://jimsmith.host.dartmouth.edu/wp-content/uploads/2022/04/The_Optimizers_Curse.pdf | §2.3. INFORMS page is paywalled. |
| Hernán, Hernández-Díaz, Robins 2004 | https://journals.lww.com/epidem/fulltext/2004/09000/a_structural_approach_to_selection_bias.20.aspx | Selection bias as conditioning on a common effect. No numbered theorem. |

## Code to copy

Balanced grid only. Missing cells: do not one-shot double-center; the committed matrices should be checked for NaNs first (`DATA.md`).

Double-centering equals the two-way residual:

```text
r_tilde_ji = r_ji - mean_j - mean_i + grand_mean
```

Sources:

- CausalPy panel fixed effects: https://causalpy.readthedocs.io/en/latest/notebooks/panel_fixed_effects.html
- scikit-learn spectral biclustering, log normalization: https://scikit-learn.org/stable/modules/biclustering.html#spectral-biclustering
- ICCDesign, arXiv:2606.02059, §2.1, the residual sum of squares \(\sum_{ij}(Y_{ij}-\bar{Y}_{i\cdot}-\bar{Y}_{\cdot j}+\bar{Y})^2\): https://arxiv.org/html/2606.02059
- Why the common shock leaves the residual: https://carlos-mendez.org/post/r_demeaning_twfe/ — the trajectory mean removes a shock common to all critics. That page is about TWFE in panels; the algebra is the same, the setting is not econometrics.

statsmodels, if installed: `smf.ols('s ~ C(critic) + C(trajectory)', data=df).fit()` then `.resid`. Docs: https://www.statsmodels.org/stable/example_formulas.html

Prefer NumPy. The project venv may not have statsmodels. Within-trajectory SD after subtracting critic means is the feature, and it does not need the trajectory dummies (see `PROOF.md`).

AUROC: https://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc_auc_score.html

```python
roc_auc_score(y_fail, u_perp)
```

Clustered interval: resample repositories, refit critic means on the resampled training rows, recompute \(u_{\perp}\) and \(\Delta\)AUROC. Do not use `scipy.stats.bootstrap` on rows. estimagic's clustered bootstrap is documented at https://estimagic.org/en/v0.2.0/how_to_guides/inference/how_to_do_bootstrap_inference.html — the cluster rule is what to copy; the OLS example is not an AUROC callable. The paper's own clustered bootstrap in the readout scripts is the implementation to match.

ICC as a summary, not as the feature: https://pingouin-stats.org/generated/pingouin.intraclass_corr.html — ICC(C,1) ignores rater shifts; ICC(A,1) treats them as error. pingouin may be absent. The variance-component table does not need it.

## Skip

- Wasserman, *All of Statistics*: not on the author's free page.
- Shalizi's free draft: wrong model (smooth additive regression).
- Casella & Berger: not free.
- Any PDF that search labels as a full textbook mirror.
