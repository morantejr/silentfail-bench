# Cursor prompt 2: verify the repairs, including your own

Paste everything below the line into Cursor with the repo open.

---

You audited this repository already and committed fixes (`bad8aaf`, `8c1d711`,
`717991b`, `abc440a`). Separately, an AI assistant made a large set of changes
before and after yours. This round is different from the last one: **you are
auditing repair work, including your own, not the original paper.**

That matters because repairs are where errors concentrate. In the last two
rounds of work on this repository:

- a fix introduced a false claim that had to be retracted the next day (the
  assertion that the 4B selector failed a condition it actually passed);
- two consistency checks were written that passed on deliberately corrupted
  input, because they searched for a bare number that also occurs elsewhere;
- a commit reported success while the file it claimed to change was never
  written;
- an audit tool produced three false alarms from a regex whose ordered
  alternation matched `json` inside `jsonl`;
- a recomputation "found" an error that did not exist, by negating a score that
  was already negated.

Assume this round contains errors of the same kind. **Treat your own four
commits as suspect on equal terms with everyone else's.** Self-review blindness
is the specific failure mode here: you are checking work you did, and the
easiest thing to miss is the thing you already decided was correct.

**Repository:** `/Users/jmorante/Downloads/xtail_spec` (git, branch `main`)
**Python:** `silentfail_bench/.venv-swebench/bin/python`
**LaTeX:** `export PATH="$HOME/Library/TinyTeX/bin/universal-darwin:$PATH"`

Start by reading `git log --oneline 40c2727..HEAD` and the diff of every commit
in that range. That is the work under audit.

## Priority 1: the correction that changed two registered verdicts

`scripts/readout_amendment5_corrected.py` claims Amendment 5's analysis filtered
its population using the wrong candidate's eligibility flag, and that fixing it
moves S-P2 from pass to fail and drops `AUROC(u)` below its registered bar so
S-P3 fails outright. Two registered verdicts changed on the strength of one
code change.

**Verify it independently.** Do not read the script and agree with it. Write
your own implementation from the pre-registration's definition of the
population, and see whether you get 101 rows, 44 failures, `AUROC(u)` 0.591, and
an unfitted endpoint of −0.018. Then ask the questions the script does not:

- Is the corrected population *right*, or merely different? The registered
  population is "committed, agent-declared-complete trajectories". Confirm from
  the bundle schema which field actually encodes that for a re-selected arm.
- The 16 added instances have gold. Where did that gold come from, and is its
  availability independent of the outcome? If those instances were graded
  because they happened to coincide with the random arm, check whether that
  introduces selection on the label.
- `prereg_readout_amendment5.py` still writes the old, defective population to
  `data/selector_axis/prereg_readout_amendment5.json`. Two files now disagree
  about the same quantity. Decide whether that is acceptable provenance or
  whether the old script should refuse to run.

## Priority 2: code written for data that does not exist yet

`scripts/readout_amendment7.py` and `scripts/preflight_amendment7.py` were
written in September for a run that happens on 1 October. Neither has ever seen
real Amendment 7 data.

- Read `Amendment 7a` in the pre-registration and check the readout implements
  **all** of it: mandatory strong-selector arm, the 0.65 pool gate on the new
  pool, unconditional failure rates, unfitted estimator primary, floor of 20
  failures, and a null claim requiring the upper bound below +0.05. Find the
  rules it does not implement.
- The `--dry-run` asserts it reproduces four published values. Break each one
  deliberately and confirm the dry run fails. A self-test that cannot fail is
  decoration.
- `preflight_amendment7.py` claims to catch five specific failures. Induce each
  one, in a scratch copy, and confirm it exits non-zero. At least one of its
  checks is probably weaker than it looks.
- The capability test A7-P1 is reported as a *band* because gold is partial.
  Check that band is computed correctly, and decide whether A7-P1 can be
  resolved at all with this coverage. If it cannot, say so now rather than on
  1 October.
- `configs/agent_axis_instances_multilingual.txt` pins 114 instances. Verify
  that set is the right scope and that all 100 existing trajectories are inside
  it.

## Priority 3: what your own compression may have lost

Your commit `abc440a` shortened the version-history rows in `main.tex` to fit
the table on one page. Compression is where accuracy quietly dies.

- Diff each shortened row against what it replaced and confirm no claim changed
  meaning, no number was dropped that another passage depends on, and no
  now-false summary was created.
- The table runs to v0.9.6 and the title block carries a version. Confirm they
  agree and that every row's date matches its commit.

## Priority 4: prose nobody has verified

Two documents were written late and checked by nobody.

- `silentfail_bench/MENTOR_NOTE.md` section 25 reports the correction to the
  mentor. Every number in it must match the corrected readout. It also claims
  Amendment 2 is unaffected and Verified is unaffected; verify both.
- `README.md`'s status block was rewritten. Check every claim in it, including
  the critic and judge counts, the repository counts, and the judge scores.
- `silentfail_bench/RESUME_AMENDMENT7.md` gives commands that will be run with
  real money on 1 October. Check the flags against each script's actual argument
  parser. A wrong flag there costs a day.

## Priority 5: the standing questions

Re-run the whole checker and bite-test **every** assertion, not only new ones.
Then answer, as PI:

1. Is there any number in either PDF that you cannot trace to a committed script
   plus committed data? Name it.
2. Is the paper's central claim still exactly supported by what the corrected
   analyses show, or has it drifted during the repairs?
3. Given the corrected S-P2 failure, is the contribution still what the
   introduction says it is?

## Rules

- `python3 paper/check_consistency.py` must exit 0, and every assertion you add
  must be proved by injecting the error it catches.
- Both PDFs build with zero overfull boxes and zero undefined references;
  `main_short.tex` body stays under six pages.
- Anything that changes a number, a verdict, or a scope gets a dated
  pre-registration entry, a `DEFECTS.md` entry if it was a bug, a version-history
  row, and a checker assertion.
- No spending. No API calls, no GPU, no Modal runs. Everything needed is on disk.
  Do not regenerate the 100 paused trajectories; they are hashed.
- Reproduce a published number with your own code before calling it wrong. If
  your recomputation disagrees, suspect yourself first.
- After every edit, grep the file for the new text. After every `git add`,
  confirm with `git ls-tree -r HEAD --name-only`. `git check-ignore -v` before
  assuming a file is tracked.

End commit messages with:

```
Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>
```

## What I want back

A ranked list of confirmed defects with `file:line` and the source value, what
you fixed, what you could not verify and why, and a plain statement of anything
in the last two rounds of repair that you believe is still wrong. If you find
nothing in your own four commits, say so explicitly and say what you checked,
so I can judge whether the search was real.
