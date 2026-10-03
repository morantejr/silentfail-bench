# Review brief for Cursor — Amendment 13 (the agent axis), 3 October 2026

**Scope.** Review commits `40def07..c9400fd` (1–3 October). They add the agent-axis run on
DeepSeek V4.1 Flash under Amendment 13 and carry its result into the four artifacts. Nothing
before Amendment 13 changed in substance; the September numbers are untouched.

**The claim to attack.** On a capable agent (unconditional failure rate 0.200 vs 0.579 for
gpt-5-mini at the same 100-step budget), no critic separates failures from successes above
chance — gpt-5 judge pool AUROC 0.552, five reward models 0.419–0.523, registered bar 0.65 —
so the registered selector gate fails, the committed sets (17 and 19 failures) sit under the
20-failure floor, and the cell is reported as **descriptive only**. The papers now say the
critic, not the disagreement, is what fails first on strong agents.

## Read these, in this order
1. `silentfail_bench/PREREGISTRATION_SWEBENCH_PILOT.md` — the Amendment 13 section: what was
   registered before the run, then three mid-run deviations, two addenda, and the outcome.
2. `silentfail_bench/data/agent_axis_deepseek/readout_amendment7_deepseek.json` — the
   registered readout. `readout_amendment13_panel.json` — every critic's pool AUROC, both signs.
3. `paper/main.tex` — abstract (last two sentences) and the limitations item beginning "A
   capable open-weight agent was tested; the axis ran; no critic qualified as its selector."
4. `paper/main_short.tex` — same item; `paper/main_agentverify.tex` is generated from it.
5. `paper/main_symposium.tex` — the "The agent axis." paragraph.
6. `silentfail_bench/MENTOR_NOTE.md` §29; `silentfail_bench/DEFECTS.md` #21, #22 and the
   #21 addendum (which records that my first fix was inert).

## Questions I want answered (yes/no with a pointer, not an essay)
- Does every number in the artifact sentences trace to the two readout JSONs? (The checker
  says yes; try to break it: `python3 paper/check_consistency.py`.)
- Is "descriptive only" used consistently, and is it ever allowed to drift into "the null
  holds for capable agents"? It must not.
- Is the interpretation ("the premise of critic selection fails first") supported by the
  numbers, or is it reaching? Say where it reaches.
- The two declared endpoint failures and the six quarantined DNS-killed trajectories: is the
  rule stated before the data, and does regenerating them favour the agent in any way I missed?
- One cand2 trajectory (lombok-2792) is a RepeatedFormatError written at the second an
  operator kill landed. It is kept as a failure. Agree, or should it be excluded with disclosure?
- The comparator's 17 rate-limited instances were rerun at 2 workers and merged (A11
  precedent). Does the merge note make the comparison fair to gpt-5-mini?
- Worker count went 6 → 18 mid-run as an execution setting. Does anything in the registered
  text make that a deviation that should have been recorded as one?
- The abstracts gained one sentence each. Too long? Does it earn its place?

## Verify, don't trust
```
python3 paper/check_consistency.py                       # ALL CONSISTENT, 27 Amendment 13 assertions
python3 paper/bite_test.py                               # ~40 min; 256 scalars, 24 standing exemptions
cd silentfail_bench && .venv-swebench/bin/python scripts/check_readouts_regenerate.py   # 7 readouts
.venv-swebench/bin/python scripts/check_inputs_tracked.py                               # 0 untracked
.venv-swebench/bin/python scripts/readout_amendment13_panel.py                          # recompute the panel AUROCs
```
Trajectory files (257 MB) are untracked, as for every earlier run; everything the readout
reads is tracked.

## What not to do
Do not edit data files or readouts. Propose text changes as diffs against `main.tex` /
`main_short.tex` (the AGENTVERIFY cut regenerates from the short paper; never edit it directly).
