# RESULTS — critic-demeaned disagreement $u_\perp$ (post-hoc, 3 October 2026)

Post-hoc analysis on tracked inputs only. Not a registered endpoint. No paper file was edited. Positive class is failure ($y=1\{g=0\}$); the selector enters as $-c$; critic means are fit on `split == "train"` rows and applied to every committed row.

## 1. Audit of the proposal (before any number was computed)

**1.1 Projector argument and the specialisation of Afrin & Shihab, Proposition 4.** Verified. With $P=I-\mathbf{1}\mathbf{1}^\top/J$ and $\mathbf r_i=(q_i+c_i)\mathbf 1+\mathbf b+\mathbf u_i$, $P\mathbf r_i=P\mathbf b+P\mathbf u_i$ because $P\mathbf 1=0$; any within-trajectory spread is unchanged by adding a constant to every critic on that trajectory. Proposition 4 as retrieved from the arXiv HTML (arXiv:2608.08002) states: for any scalar $h(x,a)$ with $\mathbb E_{\nu_x}[h]=0$, the pair $q^{(h)}=q+h$, $\mathbf e^{(h)}=\mathbf e-h\mathbf 1$ leaves the scores and the centering convention unchanged, $P\mathbf e^{(h)}=P\mathbf e$, $\bar e^{(h)}=\bar e-h$; hence internal judge scores and their disagreement cannot identify response-dependent common-mode error without an external anchor or additional structure. Their Appendix A.1 is the substitution, $P\mathbf 1=0$, and the observation that every statistic of the scores is unchanged. Our $c_i$ is their $h$ (our $c_i$ need not be centred; the centring is absorbed into $q$ and is a labelling convention). Their model already has **one score per evaluator per response**, so "one score per cell" is not an assumption our specialisation adds; there is no new identification theorem here, only a citation. Proposition 4 says nothing about rank normalisation, AUROC, or selection; the selection conditioning this paper studies is outside it and does not change the per-trajectory algebra.

**1.2 Double-centring equals critic demeaning for the within-trajectory spread.** Algebraically $\tilde r_{j,i}=r_{j,i}-\bar r_{j\cdot}-\bar r_{\cdot i}+\bar r_{\cdot\cdot}$ differs from $r_{j,i}-\bar r_{j\cdot}$ by a term constant in $j$, so $\mathrm{sd}_j$ is identical **provided the same critic means are used in both**. Numerically (table 3): $\max|\mathrm{sd}_j(\tilde r)-u_\perp| \le 9\times10^{-16}$ on every matrix. Textbook double-centring with all-row critic means differs from train-fit $u_\perp$ by up to the values in table 3; that is the train/held-out rule, not a bug, and the brief's rule to treat any difference as a bug holds only for like-for-like means.

**1.3 What $u_\perp$ removes and what it cannot identify.** Demeaning removes critic location $b_j$. The within-trajectory spread removes the trajectory-common component $q_i+c_i$ — but **every** within-row spread does, including the published $u$. $u_\perp$ cannot separate $c_i$ from $q_i$ (both sit in the trajectory main effect), cannot separate critic-by-trajectory interaction from noise (one score per cell; Shrout & Fleiss 1979), and does not remove critic scale or shape differences (demeaning is location-only; the rank normalisation the published $u$ uses removes scale as well). **The decisive audit fact:** per-critic empirical-CDF rank normalisation fit on train rows gives every critic the same train-mean rank up to ties, so critic-demeaning the rank matrix cannot change its within-row spread. Numerically, $u_\perp$ computed on the rank-normalised scores equals the published $u$ to $1\times10^{-16}$ where there are no ties and to $\le 1.0\times10^{-3}$ on the GPU matrices (ties). The published $u$ already has the stated invariance; $u_\perp$ is the same statistic on the raw scale.

**1.4 Is $u_\perp$ an algorithm?** No. It is two operations — a per-critic centring and a within-row standard deviation — and on the rank scale it coincides with the existing statistic. It is a diagnostic statistic, and the honest name for the contribution is a *property statement about the statistic the paper already uses*, plus a variance decomposition.

**1.5 Claims in `PROOF.md` / `PAPERS_AND_RESOURCES.md` that are too strong, mis-attributed, or unsupported.**
(a) "The published $u$ … mixes $b_j$ (through ranks)": **false**; per-critic rank normalisation removes $b_j$ exactly (monotone per-critic map), which is why $u_\perp$ on the rank scale reproduces $u$. (b) "High pairwise error correlation means $\mathrm{Var}(c)$ is most of the error variance": **unsupported by that statistic**. The E2 error is $\mathrm{rank}_j-(1-y)$, so every pair shares the label term; two critics independent of the label and of each other would show correlation $\mathrm{Var}(y)/\sqrt{(\mathrm{Var}(\mathrm{rn}_a)+\mathrm{Var}(y))(\mathrm{Var}(\mathrm{rn}_b)+\mathrm{Var}(y))}$, which on these data is 0.71–0.75 (table 4). The lower edge of the observed band meets that floor; the upper edge survives a label permutation and is dependence among the rank scores, which the band does not decompose into shared error versus shared signal. (c) "The component $u_\perp$ deletes" conflates two operations: demeaning deletes $b_j$; the spread deletes $q_i+c_i$. (d) "Unless the specialisation adds an assumption they do not have (one score per cell)": they already assume it. (e) `PAPERS_AND_RESOURCES.md`: "Afrin explicitly positions Proposition 4 as the no-assumption boundary against [CARE]": the text I retrieved positions the *paper* against CARE on scope (static aggregation with latent confounders versus optimisation-aware analysis of continuous scores), not Proposition 4 against CARE's Proposition 4.1; do not cite it in that form without reading the section. (f) "Appendix A.1 is six lines": four sentences; immaterial. (g) The display $\mathbb E[D]=\tfrac{J-1}{J}\mathrm{Var}(u_j)$, $\mathrm{Var}(\bar e)=\mathrm{Var}(c)+\mathrm{Var}(u_j)/J$ matches their (13) under their independence and centring assumptions — correctly attributed. (h) The brief's "the table either supports the line-262 sentence or narrows it" needs a qualification: on the raw scale the critic main effect (0.67–0.93 of SS) is score-scale heterogeneity that $u$ never sees; the relevant split is on the rank scale (table 2b), and it narrows the sentence to *critic-by-trajectory interaction*, not critic location and not a shared trajectory level.

## 2. Results

Population: the batch-1 committed population (`not excluded and visible_pass`). Matrices, stated before any number was seen: **primary** = the 16k pass named in `DATA.md`; **like-for-like** = the pass the published $u$ was computed on (Verified: the rows' own `audit_scores`, the September MPS pass, four critics; Multilingual: `prefix4k`, which also fed batch-1 E2/LC). Published $u$ reproduces exactly from the like-for-like matrix (max difference 0.0 on Multilingual; the Verified rows' `audit_scores` match no tracked GPU pass, so their own values are the like-for-like matrix).

### 2a. Rows, failures, repositories, missing cells

| benchmark | matrix | critics | n | failures | repos | train rows | missing cells |
|---|---|---|---|---|---|---|---|
| verified | 16k_primary | 7 | 34 | 12 | 9 | 30 | 0 (rule: abort if any) |
| verified | rows_audit_scores_like_for_like | 4 | 34 | 12 | 9 | 30 | 0 (rule: abort if any) |
| multilingual | 16k_primary | 7 | 77 | 35 | 37 | 59 | 0 (rule: abort if any) |
| multilingual | prefix4k_like_for_like | 7 | 77 | 35 | 37 | 59 | 0 (rule: abort if any) |

### 2b. Two-way additive sum-of-squares shares (descriptive, all analysed rows)

Raw scores, then the rank-normalised scores the published $u$ is built from. ICC(C,1) is the consistency ICC (critic location ignored).

| benchmark / matrix / panel | raw: critic | raw: trajectory | raw: residual | raw ICC(C,1) | rank: critic | rank: trajectory | rank: residual | rank ICC(C,1) |
|---|---|---|---|---|---|---|---|---|
| verified / 16k_primary / panel4 (J=4) | 0.934 | 0.033 | 0.033 | +0.335 | 0.004 | 0.441 | 0.555 | +0.257 |
| verified / 16k_primary / zoo7 (J=7) | 0.875 | 0.052 | 0.073 | +0.318 | 0.003 | 0.351 | 0.646 | +0.244 |
| verified / rows_audit_scores_like_for_like / panel4 (J=4) | 0.666 | 0.077 | 0.257 | -0.025 | 0.002 | 0.208 | 0.790 | -0.055 |
| multilingual / 16k_primary / panel4 (J=4) | 0.892 | 0.041 | 0.067 | +0.169 | 0.005 | 0.286 | 0.709 | +0.050 |
| multilingual / 16k_primary / zoo7 (J=7) | 0.814 | 0.071 | 0.116 | +0.275 | 0.005 | 0.291 | 0.704 | +0.175 |
| multilingual / prefix4k_like_for_like / panel4 (J=4) | 0.797 | 0.062 | 0.141 | +0.075 | 0.001 | 0.374 | 0.624 | +0.166 |
| multilingual / prefix4k_like_for_like / zoo7 (J=7) | 0.812 | 0.048 | 0.140 | +0.131 | 0.005 | 0.290 | 0.705 | +0.173 |

### 2c. AUROC (positive class = failure; repo-clustered 95% intervals, n=4000, seed 9)

| benchmark / matrix / panel | AUROC($-c$) | AUROC($u$ published) | AUROC($u$ recomputed on this matrix) | AUROC($u_\perp$) | corr($u_\perp$, $u$ recomputed) |
|---|---|---|---|---|---|
| verified / 16k_primary / panel4 | 0.621 | 0.508 | 0.439 [+0.222, +0.864] | 0.530 [+0.268, +0.771] | 0.733 |
| verified / 16k_primary / zoo7 | 0.621 | 0.508 | 0.557 [+0.379, +0.952] | 0.629 [+0.382, +0.906] | 0.705 |
| verified / rows_audit_scores_like_for_like / panel4 | 0.621 | 0.508 | 0.508 [+0.318, +0.850] | 0.496 [+0.280, +0.893] | 0.815 |
| multilingual / 16k_primary / panel4 | 0.560 | 0.565 | 0.549 [+0.405, +0.685] | 0.569 [+0.419, +0.705] | 0.860 |
| multilingual / 16k_primary / zoo7 | 0.560 | 0.565 | 0.582 [+0.446, +0.708] | 0.561 [+0.414, +0.703] | 0.848 |
| multilingual / prefix4k_like_for_like / panel4 | 0.560 | 0.565 | 0.565 [+0.425, +0.703] | 0.587 [+0.455, +0.719] | 0.848 |
| multilingual / prefix4k_like_for_like / zoo7 | 0.560 | 0.565 | 0.583 [+0.438, +0.725] | 0.556 [+0.415, +0.693] | 0.852 |

### 2d. Endpoint: $\Delta$AUROC from adding $u_\perp$ to $-c$ (the project's four estimators; repo-clustered bootstrap, n=4000 seeds 21/22; within-fold n=2000 seed 13)

| benchmark / matrix / panel | logistic LORO | logistic 5-fold | within-fold size-weighted | fixed unit weights (primary) | fixed, refitting $\hat\alpha$ inside the bootstrap |
|---|---|---|---|---|---|
| verified / 16k_primary / panel4 | -0.045 [-0.088, +0.000] | -0.212 [-0.480, -0.054] | -0.012 [-1.000, +0.690] | -0.049 [-0.233, +0.017] | [-0.175, +0.059] |
| verified / 16k_primary / zoo7 | -0.049 [-0.170, +0.035] | -0.114 [-0.347, +0.015] | +0.012 [-1.000, +1.000] | +0.004 [-0.125, +0.059] | [-0.117, +0.137] |
| verified / rows_audit_scores_like_for_like / panel4 | -0.121 [-0.214, +0.080] | -0.220 [-0.328, +0.000] | -0.187 [-0.498, +1.000] | -0.015 [-0.170, +0.273] | [-0.222, +0.262] |
| multilingual / 16k_primary / panel4 | +0.007 [-0.099, +0.105] | +0.020 [-0.071, +0.118] | -0.048 [-0.395, +0.480] | +0.016 [-0.074, +0.105] | [-0.080, +0.104] |
| multilingual / 16k_primary / zoo7 | -0.035 [-0.097, +0.019] | -0.031 [-0.077, +0.019] | -0.048 [-0.414, +0.458] | -0.010 [-0.106, +0.080] | [-0.104, +0.088] |
| multilingual / prefix4k_like_for_like / panel4 | -0.048 [-0.101, -0.008] | -0.027 [-0.056, -0.002] | -0.145 [-0.364, +0.350] | +0.027 [-0.062, +0.114] | [-0.067, +0.104] |
| multilingual / prefix4k_like_for_like / zoo7 | -0.071 [-0.141, -0.018] | -0.041 [-0.086, +0.003] | -0.145 [-0.402, +0.370] | +0.007 [-0.093, +0.106] | [-0.093, +0.104] |

Same estimators for the **published** $u$ on the same rows (the paper's existing comparison, recomputed here with identical code):

| benchmark / matrix | logistic LORO | logistic 5-fold | within-fold size-weighted | fixed unit weights |
|---|---|---|---|---|
| verified / 16k_primary | -0.076 [-0.159, +0.089] | -0.189 [-0.283, +0.075] | -0.152 [-1.000, +0.400] | -0.030 [-0.165, +0.240] |
| verified / rows_audit_scores_like_for_like | -0.076 [-0.159, +0.089] | -0.189 [-0.283, +0.075] | -0.152 [-1.000, +0.400] | -0.030 [-0.165, +0.240] |
| multilingual / 16k_primary | -0.022 [-0.133, +0.080] | -0.018 [-0.079, +0.052] | -0.145 [-0.375, +0.421] | +0.048 [-0.074, +0.175] |
| multilingual / prefix4k_like_for_like | -0.022 [-0.133, +0.080] | -0.018 [-0.079, +0.052] | -0.145 [-0.375, +0.421] | +0.048 [-0.074, +0.175] |

### 3. Identity checks (floating point)

| benchmark / matrix / panel | max $|\mathrm{sd}_j(\tilde r_{\text{train-fit}})-u_\perp|$ | max $|\mathrm{sd}_j(\tilde r_{\text{all-row}})-u_\perp|$ | max $|u_\perp^{\text{rank}}-u|$ |
|---|---|---|---|
| verified / 16k_primary / panel4 | 4.4e-16 | 1.27e-01 | 1.0e-03 |
| verified / 16k_primary / zoo7 | 8.9e-16 | 1.88e-01 | 8.2e-04 |
| verified / rows_audit_scores_like_for_like / panel4 | 8.9e-16 | 1.44e-01 | 1.1e-16 |
| multilingual / 16k_primary / panel4 | 8.9e-16 | 2.07e-01 | 9.7e-04 |
| multilingual / 16k_primary / zoo7 | 8.9e-16 | 3.12e-01 | 7.5e-04 |
| multilingual / prefix4k_like_for_like / panel4 | 8.9e-16 | 1.43e-01 | 2.7e-04 |
| multilingual / prefix4k_like_for_like / zoo7 | 8.9e-16 | 1.36e-01 | 1.6e-04 |

### 4. E2 pairwise error correlation (batch-1 definition: $\mathrm{rank}_j-(1-y)$, rank norms fit on train; mean over all four-member subsets)

| benchmark / matrix / panel | subsets | before critic demeaning | after critic demeaning | max change | independence floor | label-permutation range (mean of 200) |
|---|---|---|---|---|---|---|
| verified / 16k_primary / panel4 | 1 | [0.785, 0.785] | [0.785, 0.785] | 0.0e+00 | [0.725, 0.725] | [0.796, 0.796] |
| verified / 16k_primary / zoo7 | 35 | [0.710, 0.891] | [0.710, 0.891] | 0.0e+00 | [0.711, 0.725] | [0.720, 0.898] |
| verified / rows_audit_scores_like_for_like / panel4 | 1 | [0.729, 0.729] | [0.729, 0.729] | 0.0e+00 | [0.732, 0.732] | [0.718, 0.718] |
| multilingual / 16k_primary / panel4 | 1 | [0.759, 0.759] | [0.759, 0.759] | 0.0e+00 | [0.752, 0.752] | [0.765, 0.765] |
| multilingual / 16k_primary / zoo7 | 35 | [0.748, 0.914] | [0.748, 0.914] | 0.0e+00 | [0.747, 0.754] | [0.743, 0.917] |
| multilingual / prefix4k_like_for_like / panel4 | 1 | [0.769, 0.769] | [0.769, 0.769] | 0.0e+00 | [0.745, 0.745] | [0.789, 0.789] |
| multilingual / prefix4k_like_for_like / zoo7 | 35 | [0.737, 0.839] | [0.737, 0.839] | 0.0e+00 | [0.736, 0.749] | [0.750, 0.849] |

The zoo rows on the batch-1 matrices reproduce `readout_pi_batch1.json` exactly (Verified 16k 0.710–0.891; Multilingual prefix4k 0.737–0.839). Critic demeaning changes nothing because the error is built from per-critic ranks, which a per-critic shift leaves unchanged; Pearson correlation of raw columns is shift-invariant for the same reason. The floor is what the same statistic would show if every critic's rank score were independent of the label and of every other critic. Permuting the label (200 draws, seed 0) removes label dependence but keeps between-critic rank dependence; the permuted band's upper edge matches the observed upper edge, so the top of the band is dependence among the rank scores and survives shuffling the label. Only the lower edge is the floor.

Raw-score pairwise correlations (not the E2 error): verified/16k_primary/panel4 mean +0.350 range [-0.099, +0.733]; verified/16k_primary/zoo7 mean +0.311 range [-0.120, +0.892]; verified/rows_audit_scores_like_for_like/panel4 mean -0.028 range [-0.180, +0.128]; multilingual/16k_primary/panel4 mean +0.071 range [-0.399, +0.555]; multilingual/16k_primary/zoo7 mean +0.189 range [-0.399, +0.831]; multilingual/prefix4k_like_for_like/panel4 mean +0.083 range [-0.189, +0.397]; multilingual/prefix4k_like_for_like/zoo7 mean +0.131 range [-0.286, +0.506].

## 5. Commands, inputs, tracking

```
cd /Users/jmorante/Downloads/xtail_spec/silentfail_bench
.venv-swebench/bin/python ../for_claude/algorithmic_to_7/analyze_u_perp.py   # run twice; results.json byte-identical (sha256 prefix 1d3bebdc4df6550b)
.venv-swebench/bin/python ../for_claude/algorithmic_to_7/make_results_md.py
cd /Users/jmorante/Downloads/xtail_spec && python3 paper/check_consistency.py && git status --short   # the checker opens paper/main.tex relative to the cwd
```

| input | git-tracked files |
|---|---|
| `silentfail_bench/data/swebench_pilot50/pilot_free_scored.jsonl` | 1 |
| `silentfail_bench/data/replication_multilingual/replication_scored.jsonl` | 1 |
| `silentfail_bench/data/swebench_pilot50/l4/out/gpu_scores_16k` | 10 |
| `silentfail_bench/data/replication_multilingual/modal/full16k` | 7 |
| `silentfail_bench/data/replication_multilingual/modal/prefix4k` | 7 |
| `silentfail_bench/data/readout_pi_batch1.json` | 1 |
| `silentfail_bench/scripts/readout_amendment7.py` | 1 |
| `silentfail_bench/scripts/verify_and_robustify.py` | 1 |

Estimators imported from `scripts/readout_amendment7.py` (`boot`, `folds_loro`, `folds_k`, `oof`, `fixed_combo`) and `silentfail.stats.cluster_bootstrap_delta`; the within-fold size-weighted estimator is `verify_and_robustify.fold_avg_with_ci`'s logic copied verbatim (that script runs at import). No DeLong; no row bootstrap; no Amendment 13 rows.

## 6. Limitations

- Verified has 12 failures over 9 repositories; several bootstrap intervals are degenerate (the within-fold estimator's interval reaches ±1) and the 5-fold logistic can only overfit at this size. Nothing on Verified is inferential.
- This is post hoc and no inference is drawn from any $\Delta$AUROC here. (An earlier draft imported Amendment 7a's 20-failure floor; that floor belongs to a different registered setting, and Multilingual's 35 failures would clear it in any case: Cursor review, 3 October.)
- The raw-scale variance split mixes score-scale differences across reward models into the critic main effect; the rank-scale split is the one that speaks to the published statistic.
- The trajectory main effect carries $q_i+c_i$ inseparably; nothing here estimates $\mathrm{Var}(c)$.
- The uninformative-critic floor assumes rank scores independent of the label and of each other; it is a reference point for reading the E2 band, not a decomposition of it.
- Post hoc: two matrices per benchmark and two panels are reported; none was chosen after seeing an AUROC, but a reader may still discount the smallest intervals.

## 7. Hostile-reviewer assessment

*"This is a post-hoc rater fixed effect plus an existing impossibility result."* Correct as stated, and the identity in §1.3 makes it worse for the proposal: the paper's own $u$ already is the critic-demeaned statistic on the rank scale, so $u_\perp$ is not a new estimator with a new property; it is the old estimator on a worse scale (it keeps critic scale differences that the ranks remove). The fixed-unit-weight points are inside $\pm0.05$ and their intervals cover zero, including the positive points $+0.016$ and $+0.027$. Some logistic points are positive. Some logistic intervals lie entirely below zero, including Multilingual prefix4k leave-one-repository-out $[-0.101,-0.008]$ and Verified five-fold $[-0.480,-0.054]$. $u_\perp$ does not improve the endpoint on any estimator by more than the margin, and the fitted estimators sometimes lose. The one cell whose primary upper bound falls below $+0.05$ (Verified 16k panel-4, $-0.049$ $[-0.233,+0.017]$) has 12 failures and is not evidence of anything.

What the reviewer cannot dismiss: (i) the rank-scale variance split shows that 56–79% of the spread the published $u$ measures is critic-by-trajectory interaction, with a trajectory-common share of 21–44% and ICC(C,1) between $-0.06$ and $+0.26$ — a quantitative version of line 262 that is narrower and more defensible than "between-critic heterogeneity"; (ii) the E2 error correlation of 0.71–0.89 that line 264 reports as "no decorrelated panel to be had" has a lower edge at the independence floor, 0.71-0.75, and an upper edge that remains after permuting the label; the lower edge cannot distinguish shared critic error from uninformative critics, and the paper should not let a reader infer $\mathrm{Var}(c)$ from it.

## 8. Recommendation: **keep as exploratory support** (do not promote)

Against the decision standard: (1) the property is real but belongs to the published $u$ already, so $u_\perp$ is not a statistic with a property beyond being another feature — **fails**; (2) stated without claiming common-mode error becomes observable — passes; (3) the table narrows the mechanism sentence usefully but does not improve the endpoint — **half**; (4) not a tautology created by double-centring — passes, but the rank-scale identity makes it a near-tautology relative to $u$ — **fails in spirit**; (5) distinguishable from Afrin & Shihab — **fails**: the invariance is Proposition 4 specialised with no added assumption; (6) survives "post-hoc rater fixed effect plus an existing impossibility result" — **fails**. Two of six, so not promote.

- **Algorithmic-novelty score:** before 4/10; after **4/10** for $u_\perp$ as a contribution. The *mechanism statement* improves (a citable identification limit, a variance split, and a floor for the error-correlation band), which is evaluation-method content, not algorithmic content.
- **Strongest defensible one-sentence contribution:** The disagreement statistic the paper evaluates is invariant to critic location and to any trajectory-common shock (Afrin & Shihab 2026, Prop. 4), so what it measures is critic-by-trajectory interaction, which accounts for 56–79% of rank-normalised spread on the committed rows; demeaning on the raw scale ($u_\perp$) reproduces the published statistic's null ($\Delta$AUROC within $\pm0.05$ on every primary estimator, intervals including zero).
- **Strongest reviewer objection:** "$u_\perp$ equals your existing $u$ up to the choice of scale, and the invariance is a four-sentence appendix in someone else's paper."
- **Change the abstract?** No. Nothing here is a registered result or moves a headline number.
- **Paper paragraph:** not supplied (recommendation is not *promote*). If the PI wants the exploratory support in the paper, the places are the line-262 sentence (narrow "between-critic heterogeneity" to "critic-by-trajectory interaction", citing the rank-scale split) and no line-264 caveat that the whole band is the floor (only its lower edge is); the exploratory narrowing of line 262 would need a tracked readout (rank-scale residual share, the independence floor, the label-permutation range) and checker pins before any edit, since `results.json` is untracked before any edit.

