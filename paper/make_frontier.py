#!/usr/bin/env python3
"""Generate frontier_main.tex from main.tex: the same paper with its contribution framed at the strongest
level the data carry -- the executed selection-conditional measurement, the bound, the measured critic
collapse on a capable agent, and the follow-up it points to. No number is new; every figure below already
appears in main.tex and is pinned there by check_consistency.py. The submission artifacts are untouched.
Regenerated on every build so it cannot drift from the pinned text.   python3 make_frontier.py
"""
from pathlib import Path
H = Path(__file__).resolve().parent
s = (H / "main.tex").read_text()
def sub(old, new):
    global s
    assert s.count(old) == 1, old[:80]; s = s.replace(old, new)
sub("Working draft v0.9.21 --- SilentFail-Bench; two benchmarks, 46 repositories analysed, pre-registered throughout",
    "Frontier framing of v0.9.21 (generated from the working draft; the submission artifacts are the draft and its nine-page cut) --- SilentFail-Bench; two benchmarks, 46 repositories analysed, pre-registered throughout")
sub("What had not been done is the measurement: put a disjoint audit panel on the population a critic actually committed, with executed gold, structural baselines and pre-registered bars, and ask what it adds over the selector. Our contributions are:",
    "We found no prior work that measures a second audit signal's incremental value over the selector's own score on critic-committed agent trajectories with executed gold: candidate-pool evaluations~\\citep{shum2025swerm,rahman2026scout} stop before selection, and the identification limit on common-mode error~\\citep{afrin2026ensembles} is proved, not measured. This paper is that measurement, and what it returns is a bound. Our contributions are:")
sub("\\item \\textbf{A selection-conditional evaluation protocol for audit signals} --- selected population, selector baseline, executed gold, pre-registered bars --- with its first two-benchmark instantiation on SWE-agent trajectories.",
    "\\item \\textbf{An executed, selection-conditional measurement of residual auditor value} --- selected population, selector baseline, executed gold, pre-registered bars --- and the protocol that defines it, instantiated on two SWE benchmarks.")
sub("\\item \\textbf{Diagnostics and an instrument record:}",
    "\\item \\textbf{A measured instance of critic collapse on a capable agent.} On \\texttt{DeepSeek-V4.1-Flash} (456 attempts; unconditional failure rate 0.200 against 0.579 for \\texttt{gpt-5-mini} at the same 100-step cap, four attempts per instance versus one), no critic cleared the registered selector bar of 0.65: the \\texttt{gpt-5} judge's pool AUROC for detecting failure was 0.552 and the five reward models scored on this axis sat at 0.419--0.523, in the direction the correlated-judge literature predicts for stronger agents. The cell is descriptive only, and we report it as the sharpest form of the open question rather than as a result (\\S\\ref{sec:limits}).\n\\item \\textbf{Diagnostics and an instrument record:}")
sub("the capable-agent cell was run under Amendment 13 and is descriptive only, because no critic cleared the selector bar of 0.65.",
    "the capable-agent cell was run under Amendment 13 and is descriptive only, because no critic cleared the selector bar of 0.65. The remedy this diagnosis points to is an auditor that holds information the critic panel does not share --- generated tests, repository invariants, a verifier trained on labelled failures --- evaluated on the committed population against the selector's own score, on a capable agent, with a selector that clears the gate and enough repository-diverse failures for the $+0.05$ margin. That is a registered study we have not run; this paper is its diagnosis.")
(H / "frontier_main.tex").write_text(s); print("frontier_main.tex generated from main.tex")
