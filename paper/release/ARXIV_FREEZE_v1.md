# SilentFail arXiv freeze — v0.9.23 (2026-10-28)

No claim or number changed. Presentation freeze only.

## What changed

- Banner: `Working draft v0.9.23 --- Frozen for arXiv; …`
- Removed empty post-abstract `\fbox` and unused `\pending` / `\todo` macros
- Dates on long + short: **October 28, 2026**
- Changelog + `CLAIM_HISTORY.md` row for v0.9.23
- Regenerated `frontier_main.tex` / `main_ieee.tex`
- Rebuilt PDFs with tectonic 0.17.0

## Artifacts to upload

| Role | File |
|---|---|
| **arXiv PDF (primary)** | `paper/main.pdf` (long, ~35 pp) |
| **arXiv source** | `paper/release/arxiv_SilentFail-Bench_v1_source.tar.gz` |
| Short cut (workshop / mirror) | `paper/main_short.pdf` (~9 pp; body through conclusion before refs) |
| Frontier framing (optional) | `paper/frontier_main.pdf` |
| IEEE anon cut | `paper/main_ieee.pdf` (8 pp incl. refs) |

## Suggested arXiv metadata

- **Title:** Auditing the Winner: An Upper Bound on What Critic Disagreement Adds Beyond the Selector on SWE-Agent Trajectories
- **Authors:** Joseph Morante (Algoverse AI Research)
- **Categories:** `cs.SE`, `cs.AI`, `cs.LG` (primary `cs.SE` or `cs.AI`)
- **Comments:** 35 pages. Short 9-page cut and venue builds in companion materials. Pre-registered; null is failure-to-detect (+0.12 loosest pooled upper bound), not equivalence.
- **Abstract:** use the abstract from `main.tex` verbatim

## Pre-upload checks

1. `cd` repo root (`xtail_spec/`) → `python3 paper/check_consistency.py` → `ALL CONSISTENT`
2. Confirm short PDF: Conclusion and References start ≤ page 7; ieee ≤ 8 pages
3. Upload source tar + PDF; do **not** upload OneDrive venue-anonymised copies as the arXiv named version unless you want those workshops cited

## Out of this freeze

- Re-running the venue-format builds (optional after arXiv)
- Bite-test full sweep (~230 runs)
- Any new experiment or claim edit
