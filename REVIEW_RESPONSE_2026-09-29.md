# Response to the October runbook review — 29 Sep 2026

All six findings addressed across `bb84e4e` and `103bfef`. Verification for
every claim below: `silentfail_bench/scripts/preflight_pre_oct1.sh` exits 0,
`python3 paper/check_consistency.py` is green, and the new preflight checks were
bite-tested by breaking each one and confirming a non-zero exit (3 of 3 caught).

## The execution bug was real, and worse than described

You flagged that the runner passes `predictions.jsonl` while mini-swe-agent
writes `preds.json`. Confirmed, and there is a second mismatch underneath it:
`preds.json` is a **dict keyed by instance id**, while `modal/gold_modal.py`
(line 75) reads `--preds` as **JSONL**, one object per line. Pointing at the
right filename would still have failed.

The timing is the expensive part, as you implied: generation completes first, so
all 41 trajectories and their OpenRouter spend would have been paid before gold
aborted on a missing file.

Fixed with an explicit conversion stage that reads `preds.json`, asserts every
row carries `instance_id` and `model_patch`, writes one object per line, and
aborts if the result is empty. Verified against the existing DeepSeek pilot
output: 41 rows, clean round-trip, both required fields present.

**Why the preflight missed it**, which I think is the more useful finding: it
checked that the runner existed and that its guards were present, never that the
file one stage writes is the file the next stage reads. That is the same class
as the Amendment 7 checkpoint-path defect already in `DEFECTS.md`. Two checks
added for that boundary specifically.

## The handoff leftover was real

You were right that the status table still carried the overclaim after the paper
had been corrected. It read "null **replicates** after selector axis". That
matters more than a stale sentence usually would, because the handoff is the
first document a fresh agent reads. It now states the bounded failure-to-reject
with the $+0.12$ pooled bound against the $+0.05$ actionable margin, and the
preflight bans the old phrasing so it cannot return.

## The other five

| Finding | Resolution |
|---|---|
| Stop-rule conflict | Amendment 10a §1 resolves it **against** the fallback. A DeepSeek failure means stop: no GLM-5.3 substitution, no `gpt-6-luna` revert. Reasoning stated rather than asserted — `gpt-6-luna` is an API model and so does not answer the objection that prompted Amendment 10, and those 100 trajectories were already declared discarded if the agent changed, so reinstating them after seeing DeepSeek fail would be choosing the arm after seeing which survived. A third agent requires a new amendment written before it runs. |
| Budget not reconciled | Amendment 10a §2. Itemised: pilot generation $0.53, pilot gold $2.26, axis generation $8.40, axis gold $12.00, total **$23.19**. Modal cap raised before outcome from $22 to **$26**, inside the $30 credit, no paid overage. |
| Cost cap unenforceable | Amendment 10a §3. Correct diagnosis on your part: `cost_tracking: ignore_errors` silently disables the scaffold's own `cost_limit`. Enforcement is now external — the runner differences the OpenRouter credits endpoint before and after, aborts before gold if spend exceeds the cap, and passes the measured figure into the readout so P10-3 is checked against an independent number. |
| No executable command | `scripts/run_amendment10_pilot.sh`. Pins the 41-instance file, refuses to write into an existing directory, verifies the config is on the corrected 16k cap and the text-based model class, and refuses to run before 1 October without an explicit override. Verified: it refuses today. |
| Preflight ignored Amendment 10 | Thirteen checks added covering config, pilot set, fresh output directory, ledger, stop rule, external cost enforcement, the two conversion checks above, and that the gate readout fails closed with no data. Two bugs surfaced while writing them: paths relative to the wrong directory, and `set -e` killing the script on the readout's intentional exit 2 before the check could read it. |

## One thing you did not raise, disclosed rather than buried

The P10-2 comparator has two defensible values. Amendment 6 registered
gpt-5-mini's unconditional rate at **0.486**, validated against the public
leaderboard's 0.488 on SWE-bench Verified. Recomputing from the committed rows
of `pilot_free_scored.jsonl` gives **0.463**, which is a best-of-4 committed rate
and therefore a different quantity from a single-attempt rate.

The registered figure is used, because it is the registered one and because it
has external validation. The alternative is recorded in the readout with the
note that the 2.3-point gap is smaller than the 10-point gate margin, so the
gate verdict does not turn on the choice. Amendment 6 documents the same family
of error: an earlier 33.3% figure was the unselected-arm rate, which conditions
on a non-empty patch and is not comparable.

## Current state

Draft v0.9.10. Preflight green, consistency checker green, no Modal or API spend
since the review. The pilot runs on 1 October under
`scripts/run_amendment10_pilot.sh`, and its stop branch is fixed in advance.
