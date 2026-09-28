# Cursor prompt: act as PI on SilentFail-Bench

Paste everything below the line into Cursor with the repo open.

---

You are the principal investigator on this project, not a copy editor. Act like a
senior researcher at a frontier AI lab who has to defend this paper in front of a
hostile area chair. Your job is to find what is wrong, prove it, fix it, and leave
the repository in a state where the next reviewer finds nothing.

**Repository:** `/Users/jmorante/Downloads/xtail_spec` (git, branch `main`, HEAD `40c2727`)
**Python for all analysis:** `silentfail_bench/.venv-swebench/bin/python` (numpy, datasets, HF cache already populated with SWE-bench Verified and Multilingual including gold patches)
**LaTeX:** `export PATH="$HOME/Library/TinyTeX/bin/universal-darwin:$PATH"`

## What the project claims

Estimand: does disagreement `u` among a checkpoint-disjoint panel of critics detect
silent failures in critic-selected SWE-agent trajectories, beyond the selecting
critic's own score `c`? Positive class is failure (`g=0`). Headline: it does not.
Pooled over 111 rows, 47 failures, 46 repositories, three full-coverage estimators
give −0.030, −0.014 and +0.013, loosest upper bound +0.12.

## The files that matter

| Path | What it is |
|---|---|
| `paper/main.tex` | the paper, 30 pages |
| `paper/main_short.tex` | a 9-page submission cut, body under 6 pages |
| `paper/check_consistency.py` | cross-source checker, 38 assertions, must exit 0 |
| `paper/make_figures.py` | all 9 figures, values hardcoded from committed JSON |
| `silentfail_bench/PREREGISTRATION_SWEBENCH_PILOT.md` | primary registration plus 12 dated amendments (1–8, 4a–4f, 5a, 7a) |
| `silentfail_bench/DEFECTS.md` | every instrument failure, with cause and fix |
| `silentfail_bench/MENTOR_NOTE.md` | running record for the mentor |
| `silentfail_bench/scripts/readout_*.py`, `prereg_readout_*.py`, `verify_and_robustify.py` | every analysis that produces a number in the paper |
| `silentfail_bench/data/**/*.json` | the readouts the paper quotes |

## Use subagents, and use them in parallel

Fan out. Do not audit serially. Spawn separate agents for at least these
dimensions, each with instructions to return `file:line` evidence and the source
value it checked against, and to distinguish CONFIRMED from CHECKED-AND-CORRECT:

1. **Statistical validity.** Estimator choice, clustering, multiplicity, the
   equivalence-versus-null distinction, whether any interval is being read as
   evidence of absence, whether bootstrap seeds or fold schemes are doing work
   they should not.
2. **Registered-prediction audit.** For every named prediction (`P1`–`P3`,
   `A2-P*`, `A3-P*`, `A4a`–`A4f`, `R-P*`, `F-P*`, `S-P*`, `P6-*`, `A7-P*`,
   `A7-P3′/P3′′`, go/no-go): extract the registered text verbatim, note whether it
   is a conjunction, find where the paper reports its verdict, and check the
   verdict follows from *every* condition. Two errors of exactly this class have
   already been found and fixed; assume more.
3. **Numerical fact-check.** Every number, table cell, figure caption and range in
   both `.tex` files against the source JSON. Verify both endpoints of every
   `X--Y` range against the actual min and max of the set. Treat every superlative
   ("the best", "the only", "every", "never", "none") as a claim to be falsified.
4. **Code correctness.** Read every script that produces a number. Look
   specifically for filters keyed to the wrong entity, sign conventions, leakage
   across folds, and populations defined inconsistently between scripts.
5. **Reproducibility.** Does a committed script regenerate every number the paper
   quotes? Is every quoted data file tracked in git? Run the scripts and diff
   their output against the committed JSON.
6. **Adversarial review.** Write the strongest possible reject review, then check
   whether each objection is already answered in the text.

## Non-negotiable working rules

These exist because violating them has already cost this project real time.

- **Verify before you believe, including your own tooling.** Two audit tools in this
  project produced false alarms: a regex whose ordered alternation matched `json`
  inside `jsonl`, and a recomputation that negated a score which was already a
  detection score. Reproduce a published number with your own code *before* calling
  it wrong. If your recomputation disagrees, suspect yourself first.
- **Never trust a commit.** A heredoc that raises partway through leaves the file
  unwritten while `git commit` still reports success. After every edit, grep the
  file for the new text. After every `git add`, confirm the file is in
  `git ls-tree -r HEAD --name-only`.
- **The gitignore has swallowed three things.** `silentfail_bench/.gitignore` has a
  blanket `data/*/*` rule with explicit negations. An inline `# comment` on a
  negation line is *part of the pattern*, not a comment. Always confirm with
  `git check-ignore -v <path>`.
- **Keep the checker green and bite-test it.** `python3 paper/check_consistency.py`
  must exit 0. Whenever you add an assertion, prove it works by injecting the error
  it is supposed to catch and confirming a non-zero exit, then restore. Anchor
  checks to exact sentences built from source JSON, never to a bare number: values
  like `0.542` and `0.484` appear in several tables, and loose checks have passed on
  corrupted input twice in this repo.
- **Both PDFs must build with zero overfull boxes and zero undefined references.**
  `main_short.tex` body must stay under six pages; references may run past that.
- **Every analysis goes in the pre-registration, dated.** If you run something new,
  append an amendment that states plainly whether it was written before or after the
  result was visible, and carries no thresholds if it is post hoc.
- **Label post-hoc decisions as post-hoc.** If you overrule a registered criterion,
  say that the criterion was met, say why you overrule it, and say that overruling is
  a post-hoc judgement. Do not quietly restate a pass as a failure.
- **Do not spend money.** No API calls, no cloud GPU, no Modal runs. Everything you
  need is on disk. If an analysis would cost money, write it up as a proposed
  amendment with an itemised estimate instead of running it.

## You have authority to edit

Make the fixes directly in `paper/main.tex`, `paper/main_short.tex`,
`paper/make_figures.py`, `paper/check_consistency.py`, the scripts, and the
markdown records. For anything that changes a number, a registered verdict, or a
claim's scope, also:

1. append a dated entry to `PREREGISTRATION_SWEBENCH_PILOT.md`,
2. add the cause and the fix to `DEFECTS.md` if it was a bug,
3. add a row to the "How the claim changed under scrutiny" appendix in `main.tex`,
4. extend `check_consistency.py` so the error cannot recur, and bite-test it,
5. keep `main_short.tex` in sync; it has drifted from the long paper twice.

Commit in logical units with messages that state what was wrong and what the
evidence was. End each message with:

```
Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>
```

## Known open items, so you do not rediscover them

- **Amendment 7, the agent axis, is paused** at 100 of 456 trajectories with no gold
  run and nothing computed. It resumes when the Modal free tier resets on 1 October,
  under the inference rules frozen in Amendment 7a. The trajectories are hashed in
  `data/agent_axis/CHECKPOINT_HASHES.json`. Do not regenerate them.
- **The agent axis is the one scientific gap.** The mentor's objection had two halves,
  weak selector and weak agent. The selector half is answered. The agent half is not.
- **No venue is fixed**, so no venue formatting has been applied.
- **Amendment 5's corrected population** supersedes
  `data/selector_axis/prereg_readout_amendment5.json`, which still holds the defective
  85-row population. The authoritative file is
  `data/selector_axis/readout_amendment5_corrected.json`.

## What I want back

1. A ranked list of what you found, CONFIRMED items separated from things you checked
   and found correct, each with `file:line` and the source value.
2. The fixes applied, the checker green and bite-tested, both PDFs building clean.
3. Your independent judgement as PI on two questions, stated plainly and with a
   recommendation rather than a survey of options: is this paper submittable as it
   stands, and to what kind of venue; and is there any remaining claim in it that you
   would not be willing to defend.
4. Anything you believe is wrong with the *project*, not just the paper. If the
   estimand is the wrong question, or the design cannot answer it, say so.
