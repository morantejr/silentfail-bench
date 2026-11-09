# For Blake — open-weight agents, pilot results (29 Sep 2026)

**Your note:** *"Try to test the bench on more capable models... capable open weight
models like glm 5.3 and deepseek 4.1 flash."*

Adopted as Amendment 10, which supersedes the earlier `gpt-6-luna` plan. Both
models were run on the same 41 SWE-bench Verified instances as the existing
capability pilot, so the three are directly comparable. Nothing below is a
result about the paper's hypothesis; these are capability gates that decide
which agent the real experiment uses.

## Gate 1: can the model drive the agent scaffold at all?

| Agent | Submitted a patch | Hit the 40-step limit | Died on output format |
|---|---|---|---|
| `gpt-5-mini` (the paper's agent) | — | — | — |
| `gpt-6-luna` (API, Amendment 6) | **41/41 (100%)** | 0 | 0 |
| `deepseek-v4.1-flash` (open weights) | 31/41 (76%) | 7 | 3 |
| `glm-5.3` (open weights) | **13/41 (32%)** | 22 | 6 |

The registered bar is 80%. **Neither open-weight model clears it**, and GLM-5.3
misses by a wide margin.

**One caveat on DeepSeek, in its favour.** That run used a 4,000-token output
cap. This is a reasoning model that spends output budget on thinking before it
writes the command, and a tight cap can consume the whole budget and leave an
empty turn. All three format deaths are consistent with that. The cap is now
16,000 and **the pilot will be rerun before DeepSeek is judged**. The seven
step-limit exits are genuine agent behaviour and would not be affected.

**This is the unplanned finding, and I think it is the interesting one.** The
gap is not about coding ability. It is about whether a model can reliably
operate a text-driven agent loop for forty turns. The commercial model did it
41 times out of 41; the open-weight models managed 31 and 13. If that holds up
it is a practical obstacle to open-weight agents in evaluation harnesses
generally, not a fact about these two models.

## Gate 3: cost

| Agent | Per trajectory | Full 456-trajectory axis |
|---|---|---|
| `gpt-6-luna` | $0.005 | ~$2.30 |
| `deepseek-v4.1-flash` | **$0.008** | ~$3.65 |
| `glm-5.3` | ~$0.19 | ~$87 |

DeepSeek is affordable and comfortably inside the $15 cap the pre-registration
declares. **GLM-5.3 is not**, by roughly a factor of six, which is why I am not
proposing it as the study agent regardless of the reliability result.

## Gate 2: is it actually more capable?

Not yet measured. It needs the official test suites run in containers, which
needs Modal credit. **That resets 1 October** and the run is already prepared
and gated.

## What happens next

1. **1 Oct:** rerun DeepSeek's 41-instance pilot with the corrected cap, then
   grade it. Apply all three gates.
2. **If it passes:** freeze DeepSeek and run the full 114-instance, four-candidate
   axis, about $3.65 of API and inside the $26 Modal cap (Amendment 10a reconciled the ledger to $23.19 and raised the cap from $22 before outcome).
3. **If it fails:** stop and report. Do not launch the full axis.

## The honest forecast, written before the run

The pre-registration says in advance that the likely outcome is **inconclusive**,
not a confirmation. With roughly 47 failures the design can detect a large
effect but cannot establish equivalence, and a more capable agent produces
*fewer* failures, which makes the statistics harder rather than easier. I would
rather run it and report a bound than leave the axis untested, but I want you to
know the expected result before the credit is spent.

## Where the paper stands

Draft v0.9.10. The primary hypothesis is not supported on either benchmark, the
strong-selector axis is complete, and the claim is stated as a bounded
failure-to-reject rather than a demonstrated absence. The agent axis is the one
open question and the paper says so.

**What I would value your view on:** whether the reliability gap above is worth
reporting as a finding in its own right, and whether an inconclusive agent-axis
result changes your view on submitting to the ICLR/ICML workshop track.
