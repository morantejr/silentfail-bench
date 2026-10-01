#!/usr/bin/env python3
"""Generate the anonymous IEEE submission from main_short.tex. Never hand-edit the output.

AGENTVERIFY wants IEEE format, double-blind, 8 pages including references.
Maintaining that as a third hand-written .tex would invite the drift that has
already hit main_short.tex twice, and defect #16 is the same lesson one level
down: correcting an artifact instead of its generator does not hold. So the
submission is derived mechanically and `paper/check_consistency.py` regenerates
it and requires a byte-identical match.

Transformations, all of them mechanical:
  * IEEE conference preamble, two-column, replacing the article preamble
  * the author block becomes anonymous; no name, affiliation, date or repository
  * figures and tables are promoted to full-width starred floats, because a
    column-width fig_ladder is unreadable
  * IEEEtran bibliography style

Run:  python3 paper/make_agentverify.py        (writes paper/main_agentverify.tex)
      python3 paper/make_agentverify.py --check  (exit 1 if the file is stale)
"""
from __future__ import annotations
import argparse, re, sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
SRC, OUT = HERE / "main_short.tex", HERE / "main_agentverify.tex"

PREAMBLE = r"""% GENERATED FILE -- do not edit.
% Produced by paper/make_agentverify.py from paper/main_short.tex.
% Anonymous IEEE conference version for a double-blind workshop submission.
% Edit main_short.tex and regenerate; check_consistency.py enforces that this
% file matches its source.
\documentclass[conference]{IEEEtran}
\IEEEoverridecommandlockouts
\usepackage{amsmath,amssymb,booktabs,enumitem,tabularx,array,graphicx,xcolor}
\usepackage{url}
\usepackage[numbers,sort&compress]{natbib}
\usepackage{stfloats}
\usepackage[hidelinks]{hyperref}
\newcommand{\gzero}{g{=}0}\newcommand{\DeltaAUROC}{\Delta\mathrm{AUROC}}

\title{Auditing the Winner: Evaluate Failure Detectors on the\\ Population the Selector Actually Commits}
\author{\IEEEauthorblockN{Anonymous submission}\IEEEauthorblockA{Paper under double-blind review}}

\begin{document}\maketitle
"""


def generate() -> str:
    s = SRC.read_text()
    body_start = s.index(r"\begin{document}\maketitle") + len(r"\begin{document}\maketitle")
    body = s[body_start:]
    # Full-width floats: a two-column IEEE page makes column-width figures unreadable.
    body = re.sub(r"\\begin\{(figure|table)\}\[t\]", r"\\begin{\1*}[t]", body)
    body = re.sub(r"\\end\{(figure|table)\}", r"\\end{\1*}", body)
    body = body.replace(r"\bibliographystyle{plainnat}", r"\bibliographystyle{IEEEtran}")
    out = PREAMBLE + body
    # The source names the author only in the preamble, which is discarded above;
    # assert that nothing identifying rode along in the body.
    for token in ("Morante", "Algoverse", "morantejr", "github.com"):
        if token in out:
            raise SystemExit(f"ABORT: '{token}' survives into the anonymous version; "
                             f"remove it from main_short.tex's body before generating.")
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="exit 1 if the output is stale")
    a = ap.parse_args()
    want = generate()
    if a.check:
        have = OUT.read_text() if OUT.exists() else None
        if have == want:
            print(f"  ok    {OUT.name} matches what main_short.tex generates")
            return 0
        print(f"  FAIL  {OUT.name} is {'missing' if have is None else 'stale'}; "
              f"run python3 paper/make_agentverify.py")
        return 1
    OUT.write_text(want)
    print(f"wrote {OUT.relative_to(HERE.parent)} ({len(want)} chars)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
