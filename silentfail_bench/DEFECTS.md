# Stage-0 defect log — 2026-09-22

Four defects found while preparing the free-compute pilot. All were verified by
execution, not inspection, and all now have regression tests in
[`tests/test_defect_regressions.py`](tests/test_defect_regressions.py).

---

### #1 — Repo grouping truncated hyphenated repo names  *(corrupts the primary design)*

`runner.repo_from_instance` split on the FIRST hyphen, so any repo whose name
contains one lost its identity:

| instance_id | was | now |
|---|---|---|
| `scikit-learn__scikit-learn-10297` | `scikit` | `scikit-learn__scikit-learn` |
| `sphinx-doc__sphinx-8721` | `sphinx` | `sphinx-doc__sphinx` |
| `pylint-dev__pylint-6506` | `pylint` | `pylint-dev__pylint` |

Since the held-out split groups by `repo`, mangled keys mean "held-out repos"
were not reliably held out. Fixed in [`identifiers.py`](src/silentfail/identifiers.py):
only a trailing `-<digits>` is treated as the issue number.

### #2 — Splits were not reproducible across runs

`pilot.py` used builtin `hash()`, which is salted per process
(`PYTHONHASHSEED`). Three consecutive runs put `django__django` in buckets
3, 5 and 6. Any rerun reshuffled train/test. Replaced with a SHA-256 bucket
(`assign_split`), verified identical across processes and calibrated to the
70/30 target (70.1% over 2000 keys).

### #3 — IRLS diverged under separation and unscaled controls

`length_chars` runs to 1e3–1e5. Fed unscaled to IRLS this overflowed the
Hessian (`divide by zero` / `overflow encountered in matmul`), and quasi-
separation at pilot n drove the intercept to ±inf, producing NaN AUROC.
Fixed with a TRAIN-ONLY standardizer stored on `LogisticModel` (also closes a
train→test leakage vector), ridge on the intercept, and step-halving that
accepts only finite, non-worsening steps.

### #4 — The power function was invalid  *(it gated the go/no-go)*

`pilot.simulate_power` defined power as P(ΔAUROC > 0) **in the test sample**.
That statistic is ~0.5 under the true null, so it reported "power" where there
was no signal whatsoever:

| | old, under the TRUE null (δ=0) | new |
|---|---|---|
| n=100 / 120 groups | **0.390** | 0.010 |
| n=200 / 240 groups | **0.395** | 0.010 |
| n=400 | **0.435** | — |

A second bug cut the other way: sims skipped by the `if too few fails:
continue` guard stayed in the `wins / n_sims` denominator.

Replacement in [`stats.py`](src/silentfail/stats.py): the two AUROCs are
computed on the same held-out rows, so the comparison is **paired and
correlated**; inference is a **cluster bootstrap over groups** (repo /
problem_id) with the fitted models held fixed. Skipped sims are reported
separately (`n_sims_valid` / `n_sims_skipped`).

**The new test is conservative**: actual rejection ≈0.010 against a nominal
α=0.05. Rejections are therefore trustworthy; required-n figures are
pessimistic.

### #5 (lower severity) — Rank-norm leakage

`runner.run_pilot_mock` fit rank-normalizations on the first half of usable
rows, which overlaps the test split assigned later in `pilot_summary`. Test
rows leaked into the normalization that defines `u`, contradicting the
`rank_norm_fit: train_split_only` freeze. Both the CI runner and
[`runner_free.py`](src/silentfail/runner_free.py) now fit on train groups only.

---

## Harness bugs found by control checks (profile A)

Caught because **the EvalPlus reference solutions are run through the gold
harness first** — if the known-correct answer fails, the oracle is broken.

1. **`-S` hid numpy.** The sandboxed interpreter ran with `-S`, but HumanEval+
   gold tests `import numpy`. Every gold test errored, so the first smoke run
   scored `g=0` on all trajectories — a 100% failure rate that looked like a
   finding. Fixed by re-injecting exactly this interpreter's `site-packages`.
2. **Locale encoding.** `Path.write_text` without an explicit encoding produced
   `SyntaxError: Non-UTF-8 code` on tests carrying non-ASCII literals
   (`HumanEval/161`). Now written as UTF-8 with a coding declaration.
3. **Wrong reference.** Base HumanEval canonical solutions genuinely fail the
   stricter `plus` tests on some instances (`HumanEval/122`, `/76`, `/44`) —
   that is the documented EvalPlus finding, not a bug. The control now uses
   EvalPlus's own canonical solution.

After these fixes: **163/164** reference solutions pass gold. The one holdout,
`HumanEval/32` (`find_zero`, numerically unstable), is excluded as
`gold_harness_invalid` rather than counted as an agent failure.

## Sandbox posture (verified, not assumed)

`sandbox-exec` deny-by-default, confirmed by adversarial test: network egress
blocked, DNS blocked, writes outside the temp dir blocked, reads of
`~/Library/CloudStorage`, `~/Desktop`, `~/.ssh` blocked (`PermissionError`),
CPU-limit and wall-clock kills both fire.

**Verified limitation:** macOS silently ignores `RLIMIT_AS` — inside the
sandbox `getrlimit` still reports unlimited and an 800 MB allocation succeeds
under a nominal 512 MB cap. Memory is NOT capped on this platform; the CPU and
wall-clock limits are the real controls.
