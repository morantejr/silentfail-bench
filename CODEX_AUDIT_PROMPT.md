# Codex prompt: third independent audit of SilentFail-Bench

Paste everything below the line into Codex with the repo open.

---

You are the third independent reviewer of this research repository. Two AI
agents have already audited it and committed fixes. Your value is **not** in
repeating what they did. It is in finding what two auditors with similar
instincts both missed.

**Repository:** `/Users/jmorante/Downloads/xtail_spec` (git, branch `main`)
**Python:** `silentfail_bench/.venv-swebench/bin/python` — numpy, datasets, and a
populated HuggingFace cache with SWE-bench Verified and Multilingual gold patches
**LaTeX:** `export PATH="$HOME/Library/TinyTeX/bin/universal-darwin:$PATH"`
**Must pass:** `python3 paper/check_consistency.py` exits 0

## The claim under audit

Does disagreement `u` among a checkpoint-disjoint panel of critics detect silent
failures in critic-selected SWE-agent trajectories, beyond the selecting critic's
own score `c`? Positive class is failure. The paper says no: pooled over 111
rows, 47 failures and 46 repositories, three full-coverage estimators give
−0.030, −0.014 and +0.013, with a loosest upper bound of +0.12.

## What the first two audits already covered

Do not redo these. Read their results and move past them.

- Every number in both `.tex` files checked against source JSON.
- Every named pre-registered prediction checked for unreported conditions.
  Two real errors found: a conjunction where only half was reported, and a
  criterion that was met and silently overruled.
- Figure captions checked against their own plotted data; three were false.
- A population defect found in the selector-axis analysis that moved two
  registered verdicts and dissolved a +0.375 anomaly into +0.012.
- A train/test misalignment found in one exploratory script.
- Roughly 25 text and count errors fixed.

## Where to look instead

### 1. Statistical implementation, not statistical description

Both prior audits were strong on text and weak on numerical code. Read the
analysis scripts as a statistician reading an implementation, and re-derive
rather than review:

- `scripts/verify_and_robustify.py` — the four-estimator endpoint the paper's
  central table rests on. Check the fold construction, the standardiser fitting,
  the clustered bootstrap, and whether the out-of-fold probability pooling does
  what the paper says it does.
- `scripts/readout_amendment5_corrected.py`, `readout_sp1b.py`,
  `readout_failure_taxonomy.py`, `readout_luckypass.py`,
  `readout_pi_batch1.py`, `readout_pi_batch2.py`, `readout_pivotal.py`.
- `src/silentfail/` — `stats.py`, `combine.py`, `scoring.py`. The rank
  normalisation, the panel `u` statistic, and `cluster_bootstrap_delta` are used
  by everything. A defect there is a defect everywhere.

Specific things worth checking rather than assuming: whether the bootstrap
resamples clusters correctly when cluster sizes are unequal; whether rank norms
fit on training repositories leak through ties; whether AUROC handles ties the
same way in every call site; whether any seed is load-bearing.

### 2. Reproduce from scratch

For each committed readout JSON, run its script and diff the output. Anything
that does not regenerate, or regenerates differently, is a finding. One file,
`data/selector_axis/prereg_readout_amendment5.json`, is known to hold a
superseded population while a second file holds the corrected one; decide
whether the repository should tolerate two files disagreeing about the same
quantity.

### 3. The code written for data that does not exist

`scripts/readout_amendment7.py` and `scripts/preflight_amendment7.py` run on
1 October against data nobody has seen. Read `Amendment 7a` in
`silentfail_bench/PREREGISTRATION_SWEBENCH_PILOT.md` and check the readout
implements every rule in it. Then break each of the readout's four self-assertions
and each of the preflight's five checks, and confirm they actually fail. A
self-test that cannot fail is decoration.

### 4. The questions a reviewer asks that an auditor does not

This is the part most likely to be genuinely new.

- **Is the estimand the right one?** The paper measures whether `u` adds to `c`
  out of fold. Is that the quantity an operator deploying an escalation trigger
  would care about, or is it a proxy that is easier to compute than to act on?
- **Does the conclusion match the evidence, or exceed it?** The study has 47
  positives. Read the abstract and conclusion against that number and say
  whether any sentence claims more than 47 positives can support.
- **Is the null informative?** A negative result is only publishable if the
  design could have detected the effect. State plainly what effect size this
  design had power to detect, and whether the paper says so honestly.
- **Is the disjointness real?** The panel is described as checkpoint-disjoint
  from the selector. Check that claim in the code and the data, not the prose.
- **Does the selection story hold?** The paper now says selection did not
  concentrate failure, and that S-P2 fails on both benchmarks. Ask whether the
  framing in the introduction still follows from that.

### 5. Things two auditors would both take on trust

- `paper/make_figures.py` hardcodes values rather than reading JSON. Re-derive
  every plotted value from source data.
- `silentfail_bench/MENTOR_NOTE.md` section 25 and the `README.md` status block
  were written late and checked by nobody.
- `silentfail_bench/RESUME_AMENDMENT7.md` contains commands that will spend real
  money. Check every flag against each script's argument parser.

## Working rules, learned the expensive way

- **Reproduce a published number with your own code before calling it wrong.**
  Two false alarms have already been raised here: a regex whose ordered
  alternation matched `json` inside `jsonl`, and a recomputation that negated a
  score already encoded as a detection score. If your recomputation disagrees,
  suspect yourself first.
- **Two sign conventions coexist.** The judge-derived score is `100 - p_resolved`
  and is already a detection score. The reward-model `c` in `*_scored.jsonl` is
  higher-is-better and must be negated. A detection AUROC below 0.5 is the
  fingerprint of getting this wrong.
- **Never trust a commit.** A heredoc that raises partway leaves the file
  unwritten while `git commit` still succeeds. Grep the file after every edit.
- **Check `git check-ignore -v` before assuming a file is tracked.** A blanket
  `data/*/*` rule in `silentfail_bench/.gitignore` has silently excluded four
  things. An inline `#` comment on a negation line is part of the pattern.
- **Anchor consistency checks to exact sentences built from source JSON**, never
  to a bare number. Values like `0.542` and `0.484` appear in several tables, and
  loose checks have passed on corrupted input twice here.
- **No spending.** No API calls, no GPU, no Modal runs. Everything is on disk.
  Do not regenerate the 100 paused trajectories in `data/agent_axis`; they are
  hashed in `CHECKPOINT_HASHES.json` as a pre-outcome record.

## Deliverables

Fix what you confirm. For anything that changes a number, a verdict or a scope,
also add a dated entry to the pre-registration, a `DEFECTS.md` entry if it was a
bug, a version-history row in `main.tex`, and a checker assertion proved by
injecting the error it catches. Both PDFs must build with zero overfull boxes;
`main_short.tex` body stays under six pages.

End commit messages with:

```
Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>
```

Then report:

1. Confirmed defects, ranked, with `file:line` and the source value each was
   checked against.
2. What you checked and found correct, in aggregate, so the search is auditable.
3. Your answer to section 4, as a reviewer rather than an auditor. That section
   is worth more to me than another list of numeric corrections.
4. Anything you could not verify, and what you would need to verify it.
