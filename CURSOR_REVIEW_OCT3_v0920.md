# Review brief for Cursor — v0.9.20 manuscript corrections, 3 October 2026

**Scope.** Commits `82fc725..192de4a` (this afternoon). They apply your second review's five
corrections and the positioning/lead items to the four artifacts, with every number pinned to a
tracked readout. The post-hoc u_perp analysis itself stays untracked in `for_claude/`
(recommendation: keep as exploratory support; not adopted).

**Verify first, from the repo root** (the checker opens `paper/main.tex` relative to the cwd):
```
python3 paper/check_consistency.py                         # ALL CONSISTENT, 360+ assertions
cd silentfail_bench && .venv-swebench/bin/python scripts/check_readouts_regenerate.py   # 9 readouts identical
.venv-swebench/bin/python scripts/check_inputs_tracked.py  # 0 untracked
python3 ../paper/bite_test.py   # optional, ~45 min: last run 269 caught / 24 standing exemptions / 0 crashes
git status --short              # only for_claude/ and the pre-existing gold_inputs/MANIFEST.json
```

## What changed, where, and what pins it
1. **Defect #23 (learned-combiner sign).** `scripts/readout_pi_batch1.py` now also writes
   `fixed_weights_oof_detect` (reward-model columns and c negated); the published key is kept.
   `paper/main.tex` line 266 rewritten: unit-weight panel sum 0.57 / 0.59 against disagreement
   0.50 / 0.57; logistic 0.64 / 0.38; the inverted 0.43 / 0.41 named as the error. `DEFECTS.md` #23.
   Pins: `check_consistency.py` "learned-combiner sentence (DEFECTS #23)" block.
2. **Mechanism sentence (line 262).** New tracked producer `scripts/readout_critic_demeaning.py`
   → `data/readout_critic_demeaning.json`. Sentence states the rank-scale decomposition:
   critic location <1%, shared trajectory level 29–44%, critic-by-trajectory interaction 56–71%.
   Pins: range sentences + sum-to-one per panel.
3. **E2 band (line 264).** Two qualifications added: lower edge meets the independence floor
   (0.71–0.73 Verified; 0.74–0.75 Multilingual); upper edge survives 200 label permutations
   (0.90 / 0.85). Not called an artifact. Pins: floor and permutation sentences derive from the JSON;
   observed band still reproduces 0.71–0.89 / 0.74–0.84.
4. **Candidate diversity** in the Amendment 4f limitations item (overlap 0.18 / 0.22; near-identical
   6% / 3%; selector-score SD 2.34 / 1.35), pinned to `readout_pi_batch1.json` CD.
5. **Wording.** Conclusion: "they are at chance" → failure to reject with intervals that permit
   effects. Amendment 13 comparison stated *per attempt* (four attempts vs one) in all four
   artifacts, README and MENTOR_NOTE. "40 steps throughout" → "throughout the main study (Amendments
   11 and 13 ran at 100)". Abstract denominators: "of the committed population (visible-pass,
   non-empty, gradable)".
6. **Positioning sentence** in related work ("SWE agents, verifiers and gold"): Afrin & Shihab
   (identification limit) / SWE-RM and Scout (candidate pools) / this paper (committed trajectory,
   residual value, executed gold).
7. **Abstract order.** The endpoint-and-bound sentences moved up to follow the framing sentence
   (moved, not duplicated). History row v0.9.20; README version; MENTOR_NOTE §30.

## Questions (yes/no with file:line)
- Is the line-266 rewrite now a fair reading of the JSON — in particular "slightly above the spread"
  for 0.57 vs 0.50 and 0.59 vs 0.57, and "unstable across benchmarks" for logistic 0.64 / 0.38?
- Does the line-262 sentence claim anything the rank-scale decomposition does not support? (It
  says interaction is "the component disagreement measures".)
- Line 264: do the two qualifications read as qualifications, or do they undercut the sentence
  they qualify ("no decorrelated panel to be had")? Say which.
- Is "per-attempt budget" the right phrase for an abstract, or does it need "(four attempts per
  instance against one)" there too? The long paper's limitations item has the full statement.
- Abstract order: does leading with the endpoint before the benchmark sentence read as
  result-before-setup in a way a reviewer would dislike?
- The positioning sentence cites `rahman2026scout` as "Scout"; confirm the bib entry's title
  ("Locating Hidden Failures Makes Long-Horizon Agents More Reliable") is the paper you meant by Scout.
- Anything in the four PDFs that still says or implies the agent axis is unrun, or that the null
  holds for capable agents? (The checker scans for the phrases we agreed; look for new ones.)

## What not to do
Do not edit `data/`, readouts or `for_claude/`. Propose text changes as diffs against `main.tex` /
`main_short.tex`; `main_agentverify.tex` is generated.
