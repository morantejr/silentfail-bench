# Independent recomputation of the headline numbers in paper/main.tex

Read-only, fresh code: `recompute.py` in this directory (no project module imported; every statistic re-implemented
from its definition). Raw outputs: `recompute_results.json`, `run.log`, `ties_and_digits.log`.
Repository root is two levels above this folder (`paper/audits/independent_recompute_2026-10-11/`); all paths below are relative to `silentfail_bench/data/`.

Match legend: **exact** = identical to the quoted precision and to the project's stored JSON at 3 d.p.;
**rounding** = agrees once rounded as the paper rounds; **boot-noise** = bootstrap interval edge, agrees within seed-to-seed
spread; **MISMATCH** = disagrees; **caveat** = reproduces but depends on a definition choice the paper does not state.

## Table

| # | Quantity (paper) | Paper value | Recomputed | Match | Raw files used | Population |
|---|---|---|---|---|---|---|
| 1a | Silent-failure rate, SWE-bench Verified | 35% (35.3% [21.5, 52.1]) | 12/34 = 0.3529, Wilson [0.215, 0.521] | exact | `swebench_pilot50/pilot_free_scored.jsonl` (41 rows) | 34 committed = not excluded & visible_pass; 7 excluded; 9 repos |
| 1b | Silent-failure rate, SWE-bench Multilingual | 45% (45.5%) | 35/77 = 0.4545, Wilson [0.348, 0.565] | exact | `replication_multilingual/replication_scored.jsonl` (114 rows) | 77 committed; 37 excluded (30 empty_patch, 7 gold_error); 37 repos |
| 2a | AUROC(u), Verified | 0.508 (abstract: 0.51) | 0.50758 | exact | same as 1a; `audit_scores` raw columns, rank norms fit on 30 train-split rows | 34 rows, 12 positives |
| 2b | AUROC(u), Multilingual | 0.565 (abstract: 0.57) | 0.56463 | rounding (see note A: 2-d.p. value is 0.56, abstract's 0.57 is a double rounding) | same as 1b; rank norms fit on 59 train-split rows | 77 rows, 35 positives |
| 2c | AUROC(-c), Verified (selector's own score) | 0.621 | 0.62121 | exact | `c` column of 1a (`c_sel_skywork-reward-v2-qwen3-4b`) | 34 rows, 12 positives |
| 2d | AUROC(-c), Multilingual (used for the "0.560" comparison) | 0.560 | 0.55952 | exact | `c` column of 1b | 77 rows, 35 positives |
| 3a | Selector ranks failures among candidates it passed over, Verified | 0.81 [0.74, 1.00] | 0.80952, my repo-clustered CI [0.741, 1.000] | exact | `random_arm/verified_picks.jsonl` (39), `gold_modal/random_arm_verified/report.json` (24 resolved / 12 unresolved / 3 error), `candidate_scores[pick]` from 1a | 31 paired rows (39 picks − 3 gold errors − 5 excluded/not visible-pass); 10 random-arm failures |
| 3b | ... among the ones it committed, same instances | 0.61 | 0.61111 | exact | `c` from 1a on the same 31 instances | 31 rows, 9 committed failures |
| 4a | Pooled failures / repositories | 47 failures, 46 repositories | 47 positives of 111 rows; 46 clusters (9 Verified + 37 Multilingual) | exact | 1a + 1b, cluster key `bench:repo` | 111 rows |
| 4b | Pooled point estimates of the registered endpoint, "−0.03, −0.01 and +0.01" | −0.03 / −0.01 / +0.01 | logistic LORO −0.0296; logistic 5-fold clustered −0.0136; fixed unit weights +0.0126 | exact (project JSON: −0.030, −0.014, +0.013) | 1a + 1b (`c`, stored `u`, `g`, `repo`) | 111 rows, 47 positives |
| 4c | Fourth estimator (fold-averaged LORO) | not quoted in abstract; project −0.119 | −0.1189 (15 folds contributing, 53 rows covered) | exact | same | same |
| 4d | Loosest pooled 95% upper bound | +0.12 | fixed-unit-weights upper edge: +0.1212 / +0.1196 / +0.1220 over three bootstrap seeds (4000 resamples each, repo-clustered percentile) | boot-noise (project: +0.120) | same | 46 clusters resampled |
| 5a | DeepSeek V4.1 Flash unconditional failure rate, 100-step cap | 0.200 of 456 | (20 empty + 71 graded failures)/456 = 0.1996 | exact | `agent_axis_deepseek/rescoring_bundle.jsonl` (456), `gold_modal/a7_agent_axis_cand{0..3}/gold_extra.jsonl` (110+108+111+107 = 436), `report.json` error_ids all empty | 456 attempts; 0 ungraded non-empty |
| 5b | gpt-5-mini, k=1, same cap | 0.579 of 114 | 66/114 = 0.5789 (set union of unresolved ∪ empty ∪ agent-crash; the 11 empties are all already in the 66 unresolved) | exact | `agent_axis_gpt5mini_100/predictions_merged.jsonl` (114), `gold_modal/a13_gpt5mini_100_merged/report.json` (48 resolved / 66 unresolved / 0 error) | 114 instances |
| 5c | gpt-5 judge pool AUROC | 0.552 | 0.5520 | exact | `agent_axis_deepseek/judge_gpt5.jsonl` (judge=gpt-5, cond=J1), score = 100 − p_resolved | pool = 436 graded non-empty attempts, 71 failures |
| 5d | Reward models on this axis | 0.419–0.523 | gemma2-2b 0.464, llama3.2-3b 0.487, llama3.2-1b 0.517, qwen3-1.7b 0.419, qwen3-4b 0.523 → [0.419, 0.523] | exact | `agent_axis_deepseek/modal/prefix4k/scores_gpu_*.json` (5 files), AUROC(y, −score) | same pool |
| 5e | Committed failures, judge arm / 4B arm | 17 (n=113) / 19 (n=109) | 17/113 (41 repos) / 19/109 (41 repos) | exact | bundle + gold + panel scores + judge cache / 4B selector scores | population rule: selected, visible_pass, non-empty, gold present, all 4 panel scores finite |
| 6a | gpt-6-sol as selector, pool AUROC | 0.619 | 0.6185 | exact | `agent_axis_deepseek/judge_gpt-6-sol.jsonl` (456 rows) | 436-attempt pool |
| 6b | gpt-6-luna as selector, pool AUROC | 0.569 | 0.5687 | exact | `agent_axis_deepseek/judge_gpt-6-luna.jsonl` (456 rows) | 436-attempt pool |
| 6c | Their picks' unconditional failure rate | 0.193 / 0.211 | 22/114 = 0.1930 / 24/114 = 0.2105 with first-max tie-breaking | exact, **caveat** (note B: tie rule changes sol to 0.175–0.193) | same + gold | 114 instances |
| 6d | Random picks' failure rate | 0.175 | 20/114 = 0.1754 | exact | `agent_axis_deepseek/gold_inputs/random_picks.jsonl` + gold | 114 instances |
| 6e | Committed failures in the judge arm | 21 / 23 | 21/113 (sol) / 23/113 (luna) | exact, caveat (note B: 19–21 / 22–23 under other tie rules) | same population rule as 5e | 113 rows each |

Every number in the abstract and the summary paragraph that I was asked to check reproduces from the raw files. No MISMATCH.

## Definition choices I had to make

1. **Committed population** = rows with `excluded == False` and `visible_pass == True` in the `*_scored.jsonl` row file. I also
   required a non-empty `final_solution`; that changed nothing (0 empty patches among visible-pass, non-excluded rows on either
   benchmark). Silent failure = `g == 0` on such a row (`g` is the harness verdict stored on the row; 1 = resolved).
2. **Rank normalisation for u.** The task prompt describes "rank per critic over the committed population". That is *not* what the
   stored `u` uses and *not* what reproduces 0.508/0.565. The stored `u` is the ECDF rank of each critic's raw `audit_scores`
   value against the **training-split** rows (`split == "train"`, not excluded, regardless of visible_pass: 30 rows on Verified,
   59 on Multilingual), computed as share of training values `<= x` (ties count; equivalent to `searchsorted(..., side="right")`).
   `u` = sample SD with ddof=1 over the four critics. Recomputing this way reproduces the stored `u` column to 0.0 max abs
   difference on both benchmarks, hence AUROC(u) 0.50758 / 0.56463. Fitting the ECDF on the committed population instead gives
   0.5227 / 0.5721, which is exactly the "leaky" variant the paper says it rejected (0.523 / 0.572 in `verify_and_robustify.json`).
   So the prompt's phrasing describes the rejected definition; the paper's numbers come from the train-fit definition.
3. **AUROC** = P(score of random positive > score of random negative) with ties counted 1/2, computed by exhaustive pairwise
   comparison (not rank-based); identical quantity to the project's mid-rank Mann-Whitney.
4. **Score direction.** `c` in `*_scored.jsonl` is the reward model's raw score (higher = better), so the detector is `−c`.
   Judge scores are `100 − p_resolved`. Reward-model pool scores on the agent axis are negated.
5. **Item 3 pairing.** Random arm: `candidate_scores[picks.candidate]` from the row file, gold from the random-arm Modal report;
   committed arm: row `c` and row `g`. Instances kept only if in both gold and the committed population (31 of 39 picks; 3 of the
   8 drops are harness `error_ids`). I verified that `picks.committed == selected_candidate_index` and
   `c == candidate_scores[selected_candidate_index]` on all 31 rows.
6. **Item 4 estimators.** (i) Pooled out-of-fold logistic, leave-one-repository-out, ridge 1e-2 on all coefficients including
   the intercept, Newton with step-halving (own implementation); (ii) same with 5 repository-clustered folds, repos permuted
   with `default_rng(0)` and dealt round-robin — I copied this fold assignment because the fold split is part of the estimator's
   definition, not a random seed one can vary and still call it the same estimand; (iii) fixed unit weights: sum of z-scores of
   `−c` and `u` with the standardiser fit on the LORO training fold, no fitted coefficients; (iv) fold-averaged LORO: in-fold
   ΔAUROC weighted by fold size, skipping folds with one class absent. Cluster key is `benchmark:repo`, so the same repository
   name on both benchmarks would count as two clusters (none overlap in practice: 9 + 37 = 46).
7. **Bootstrap CI (item 4d).** My own repository-clustered percentile bootstrap: draw 46 clusters with replacement, concatenate
   their rows, recompute the paired ΔAUROC on the fixed OOF scores (scores are *not* refit inside the bootstrap, same as the
   project), skip resamples with one class absent, 4000 draws, 2.5/97.5 percentiles. Seeds 500/550/551 give upper edges
   0.1212 / 0.1196 / 0.1220; the project's seed-12 run gives 0.120. An exact match is not expected (different seed, different
   RNG call pattern) and was not obtained; the quoted +0.12 is well inside the seed-to-seed spread of ±0.002.
8. **Item 5a.** Empty patch (whitespace-only `final_solution`) = failure without a gold run; every other slot has exactly one
   gold row. The 20 excluded rows in the bundle (16 RepeatedFormatError, 2 EndpointFailure, 2 LimitsExceeded) coincide
   exactly with the 20 empty-patch slots, so "excluded" vs "empty" is the same set here.
9. **Item 5b.** Failure set = (unresolved ∪ empty ∪ agent-caused-crash) − infrastructure-error, divided by
   (114 − infrastructure errors). There were 0 crashes and 0 infra errors; the 11 empty-patch instances are all already graded
   unresolved by the harness, so the set-union count (66) differs from the naive sum (77, rate 0.675). The paper's 0.579 is the
   union; the prompt's description of this is correct.
10. **Selection rules on the agent axis.** 4B arm: argmax of the 4B critic's prefix-4k score over candidates with non-empty
    `text`, first candidate wins ties. Judge arms: argmax of `p_resolved`, first candidate wins ties (Python `max` semantics).
    Panel rank norms in each arm are fit on that arm's own train-split rows (80 rows for the 4B arm, 84 for the judge arms).

## Notes and findings

**A. Abstract's "0.57" for Multilingual AUROC(u) is a double rounding.** The exact value is 0.564626. Rounded directly to two
decimals that is 0.56; the paper rounds first to 0.565 (3 d.p.) and then to 0.57. `check_consistency.py` enforces the string
"0.51 and 0.57", so this is deliberate, but a reader rounding 0.565 themselves would get 0.56 or 0.57 depending on convention.
Not a numerical error; a presentation choice worth a footnote or switching to "0.51 and 0.56".

**B. The gpt-6 "picks fail more often than random" comparison is tie-rule dependent for gpt-6-sol.** Judge scores are integer
percentages, so argmax ties are common: 41 of 114 instances under gpt-6-sol and 38 under gpt-6-luna have two or more candidates
sharing the top `p_resolved` (23 under gpt-5). The readout breaks ties by taking the lowest candidate index. Two tied instances
per gpt-6 judge have tied candidates with *different* gold, so the tie rule moves the numbers:

| judge | tie rule | committed failures (n=113) | unconditional rate over 114 |
|---|---|---|---|
| gpt-6-sol | first index (project) | 21 | 0.193 |
| gpt-6-sol | last index | 19 | 0.175 |
| gpt-6-sol | uniform random (seed 0) | 20 | 0.184 |
| gpt-6-luna | first index (project) | 23 | 0.211 |
| gpt-6-luna | last index | 22 | 0.202 |
| gpt-6-luna | uniform random (seed 0) | 23 | 0.211 |
| gpt-5 | any of the three | 17 | 0.158 |

Under last-index tie-breaking gpt-6-sol's picks fail at exactly the random rate (0.175 vs 0.175), so the paper's sentence that
both judges' picks "fail more often than the frozen random picks (0.193 and 0.211 against 0.175)" holds for gpt-6-luna under
every rule but for gpt-6-sol only under the first-index rule. The pool AUROCs (6a, 6b) and the "below 0.65" conclusion are
unaffected by ties. The paper does not state the tie rule; it should, or it should report the range.

**C. Everything else is reproduced exactly from the raw files, including the counts** (34/12, 77/35, 31 pairs, 111/47/46,
456/20/436/71, 114/66, 113/17, 109/19, 113/21, 113/23, 20 random-arm failures).

**D. Not recomputed (out of scope or no raw input):** the per-benchmark M3 intervals in Table 1 (only the pooled bound was
asked for), and the 0.671 "same judge on gpt-5-mini trajectories" figure, which lives in the Amendment 5 corrected readout
and was not in the list.
