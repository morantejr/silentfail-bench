# SilentFail arXiv freeze — v0.9.24 (2026-11-09)

Supersedes the v0.9.23 freeze of 28 Oct, which was prepared but never uploaded.
arXiv v1 is therefore paper v0.9.24: the symposium reviews applied,
plus the design-figure box fix. No registered number changed between v0.9.23 and v0.9.24.

## What changed since the v0.9.23 freeze

- Rows accounted for (41 Verified instances to 34 committed rows; 114 Multilingual to 77)
- Plain-language glossary for u, −c and LORO; the four panel models named; attempt counts explicit
- Pooling labelled as added after registration; Amendment 5a population distinguished from the replication set
- One post-hoc analysis, marked as such: a single strong judge's own score added to the selector,
  +0.17 [+0.10, +0.25] AUROC pooled; one abstract sentence says so
- Fig. design schematic: boxes widened so no label spills outside its rectangle
- Changelog + `CLAIM_HISTORY.md` rows for v0.9.24

## Artifacts to upload

| Role | File |
|---|---|
| **arXiv PDF (primary)** | `paper/release/Morante_Auditing-the-Winner_LONG.pdf` (= `paper/main.pdf`, 36 pp) |
| **arXiv source** | `paper/release/arxiv_SilentFail-Bench_v1_source.tar.gz` |
| Short cut (workshop / mirror) | `paper/release/Morante_Auditing-the-Winner_SHORT.pdf` (10 pp) |
| Frontier framing (optional) | `paper/release/Morante_Auditing-the-Winner_FRONTIER.pdf` |
| IEEE anon cut | `paper/release/Morante_Auditing-the-Winner_IEEE-anon.pdf` (8 pp incl. refs) |

The source tarball holds `main.tex`, `main.bbl` and the nine `figures/*.pdf` the paper includes.
The packaged `main.tex` differs from `paper/main.tex` by one leading line, `\pdfoutput=1`,
so arXiv's AutoTeX chooses pdflatex for the PDF figures. It was test-compiled in a clean directory
with pdflatex alone (two passes, no bibtex): 36 pages, 0 overfull boxes, 0 undefined references,
and its extracted text is identical to `paper/main.pdf`.

## Suggested arXiv metadata

See `paper/release/arxiv_SilentFail-Bench_v1_METADATA.txt` (title, authors, categories, comments,
abstract with LaTeX removed). Categories: `cs.SE` primary, `cs.AI`, `cs.LG`.

## Pre-upload checks

1. `cd` repo root (`xtail_spec/`) → `python3 paper/check_consistency.py` → `ALL CONSISTENT`
2. Confirm the text never mentions the archival venue (chairs' condition for a preprint during review)
3. Upload source tar + PDF; do **not** upload the venue-anonymised copies as the arXiv named version

## Out of this freeze

- Any new experiment or claim edit
- Replacing the venue PDFs (separate step; see README venue table)
