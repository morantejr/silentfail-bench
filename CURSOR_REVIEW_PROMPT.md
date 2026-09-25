Read the repository at the repository root (this folder) in full. It is a
complete, finished empirical ML study with its paper, data, code and
pre-registration trail. You have reviewed the PDF three times (v0.4, v0.5,
v0.6) and every wording/statistics item you raised has been applied. This
time I want you to use **file access**, which you did not have before, to do
the things a PDF reviewer cannot.

## What the project is

Does disagreement among a panel of off-the-shelf critics detect silent
failures in SWE-agent trajectories that a selecting critic already committed?
Answer: no, replicated across two benchmarks and ~50 repositories, with an
estimator-robust upper bound of +0.12 AUROC on the pre-registered incremental
endpoint. Full detail in `paper/main.pdf`.

## Map

- `paper/main.tex`, `paper/main.pdf` — the paper (v0.8, 23 pp). `paper/refs.bib`
  (64 refs), `paper/make_figures.py` (every figure regenerates from committed JSON).
- `silentfail_bench/PREREGISTRATION_SWEBENCH_PILOT.md` — the spine. Primary
  pre-registration plus dated Amendments 1, 2, 3, 4, 4a–4f, 5. Amendment 5 is
  declared and **NOT RUN**.
- `silentfail_bench/MENTOR_NOTE.md` — §1–21, the full internal record.
- `silentfail_bench/DEFECTS.md` — six instrument failures, each of which would
  have produced a false positive.
- `silentfail_bench/REVIEW_skeptical_2026-09-25.md` — a prior adversarial review.
- `UPDATES.md` — research log, one entry per stage.
- `silentfail_bench/scripts/prereg_readout_*.py` — one readout per amendment.
- `silentfail_bench/data/` — every scored row, critic score, harness report.
  Key files: `swebench_pilot50/pilot_free_scored.jsonl` (pilot rows),
  `replication_multilingual/replication_scored.jsonl` (replication rows),
  `verify_and_robustify.json`, `readout_tier15.json`, `random_arm/`.
- `SilentFail_AAAI_Proposal.pdf/.tex` — the original proposal.

## What I want

**1. Numerical audit (highest value).** Every number in `paper/main.tex` should
trace to a committed JSON file or be derivable by running a script in
`silentfail_bench/scripts/`. Check them. Report any number in the paper that
you cannot reproduce from the data, any that disagrees with its source file,
and any that appears in two places with different values. Pay particular
attention to Tables 1–5 and the abstract.

**2. Reproducibility.** Could a stranger clone this and re-derive the results?
Try: does `python3 paper/make_figures.py` run from the repo root? Do the
readout scripts run against the committed data without network or GPU? List
what is missing (undocumented env setup, hard-coded paths, absent inputs).
The venv is `silentfail_bench/.venv-swebench`.

**3. Pre-registration integrity.** Cross-check the amendments against the
readouts and the paper. For every prediction with a numeric bar, confirm the
paper reports the outcome and reports it in the declared direction. Flag any
quantity in the paper that looks like it was chosen after seeing data and is
not labelled exploratory.

**4. Release hygiene.** The repo is private and about to be shared with a
mentor. Find: secrets or tokens in any committed file or in git history;
absolute paths that leak the machine layout; build clutter at the top level
(`*.aux`, `*.log`, `a1.log`…`e2.log`, stray `a2_*.json` in
`silentfail_bench/`); anything large that should be gitignored. Propose a
cleanup commit; do not run it.

**5. The venue cut.** The paper is 23 pages. AAAI main track is 7 + references.
Propose a specific cut to 7 pages: what moves to the appendix, what is deleted,
what must survive. Give the section-by-section page budget.

## Rules

- Do not re-litigate the wording or the statistics; that is settled.
- Do not propose new experiments. Amendment 5 is a known, pre-registered,
  deliberately unrun option; its cost is itemized in `MENTOR_NOTE.md` §20 and
  its power problem in §21.
- If you find a discrepancy, quote the file, the line, and the conflicting
  value. Do not assume the paper is right.
- If a check passes, say so in one line and move on.
