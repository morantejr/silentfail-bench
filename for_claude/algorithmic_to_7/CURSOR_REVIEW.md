# Review brief for Cursor — critic-demeaned disagreement (u_perp), 3 October 2026

**Scope.** Everything in `for_claude/algorithmic_to_7/`. Nothing outside this folder was edited; the
paper, the readouts and `silentfail_bench/data/` are untouched. Verify that first:
`git status --short` should show only `for_claude/` (plus one pre-existing untracked file,
`silentfail_bench/data/agent_axis_deepseek/gold_inputs/MANIFEST.json`, which is not from this work).

**What was asked.** Whether critic-demeaned disagreement
u_perp_i = sd_j(r_ji − alpha_hat_j), alpha_hat_j fit on train rows, is an algorithmic contribution
that moves the paper's algorithmic novelty from ~4/10 toward 7/10. Post hoc; a null is allowed.

**What `RESULTS.md` concludes.** Keep as exploratory support; do not promote; score stays 4/10.
Three claims carry that conclusion — attack them:

1. **The published u already is critic-demeaned disagreement on the rank scale.** Per-critic ECDF
   rank normalisation fit on train rows gives every critic the same train-mean rank (up to ties), so
   demeaning the rank matrix cannot change its within-row sd. Table 3: |u_perp(rank) − u| ≤ 1e-16
   (no ties) / ≤ 1.0e-3 (GPU matrices, ties). If this is right, u_perp is the same statistic on a
   worse scale and `PROOF.md`'s "u mixes b_j through ranks" is false.
2. **Adding u_perp to −c is null or negative on every estimator** (table 2d), including with the
   critic means refit inside the repo-clustered bootstrap.
3. **The E2 error correlation (0.71–0.89, paper line 264) sits on an uninformative-critic floor of
   0.71–0.75** (table 4): err_j = rank_j − (1 − y) shares the label term, so critics independent of
   the label and of each other would show correlation Var(y)/sqrt((Var(rn_a)+Var(y))(Var(rn_b)+Var(y))).
   If this is right, the band cannot support "Var(c) dominates", and line 264 needs a caveat.

## Read in this order
1. `RESULTS.md` §1 (the audit) — check the algebra and the attribution to Afrin & Shihab Prop. 4
   (arXiv:2608.08002; statement quoted there from the HTML).
2. `analyze_u_perp.py` — the computation. It imports the paper's own estimators from
   `silentfail_bench/scripts/readout_amendment7.py` and copies `fold_avg_with_ci` from
   `verify_and_robustify.py`. Confirm the imports are the paper's code paths, not re-implementations.
3. `results.json` — every number in the tables comes from here via `make_results_md.py`.
4. `silentfail_bench/scripts/readout_pi_batch1.py` lines 61–85 — the E2 definition the floor is derived for.

## Questions (yes/no with a file:line pointer each)
- Is the rank-scale identity (claim 1) correct as algebra, and is the 1e-3 residual really ties and
  nothing else? (Check `RankNormFit.transform` in `silentfail_bench/src/silentfail/scoring.py`.)
- Is the floor formula (claim 3) the right independence benchmark for the E2 definition, and is
  Var(y) with ddof=1 the right choice? Would a permutation of y across rows give the same floor?
- Are critic means fit on `split == "train"` only, everywhere, including inside the refit bootstrap?
- Is the sign convention right (−c as the detection score; positive class failure)? AUROC(−c) on
  Verified is 0.621, which equals `prereg_readout.json` M2; the paper's 0.61 is Amendment 4f's
  same-rows number. Agree?
- Does the two-way SS decomposition on all analysed rows (descriptive) leak anything into the AUROCs?
  (It should not; it is not a feature.)
- Is "keep as exploratory support" the right call against the six-point decision standard in
  `PROMPT_FOR_CLAUDE.md`, or is there a defensible promote framing I missed?
- Are the two proposed narrowings (line 262 → "critic-by-trajectory interaction"; line 264 floor
  caveat) worth putting in the paper, and if so what checker pins would they need?

## Verify, don't trust
```
cd /Users/jmorante/Downloads/xtail_spec/silentfail_bench
.venv-swebench/bin/python ../for_claude/algorithmic_to_7/analyze_u_perp.py   # ~90 s; run twice, results.json must be byte-identical
.venv-swebench/bin/python ../for_claude/algorithmic_to_7/make_results_md.py
python3 ../paper/check_consistency.py                                        # must stay ALL CONSISTENT
```

## What not to do
Do not edit anything outside `for_claude/algorithmic_to_7/`. Do not run paid APIs or Modal. Propose
changes to the audit or tables as diffs against the two scripts; `RESULTS.md` is generated.
