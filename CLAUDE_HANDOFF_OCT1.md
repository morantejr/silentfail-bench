# Claude handoff — SilentFail-Bench through 1 October 2026

Use this file when continuing work **before and on** the Modal \$30 credit reset (1 Oct 2026). The repo is **`/Users/jmorante/Downloads/xtail_spec`** (git, branch `main`).

---

## Current state (verified 2026-09-29)

| Check | Result |
|--------|--------|
| `python3 -m pytest -q` in `silentfail_bench/` | 18 passed |
| `python3 paper/check_consistency.py` (repo root) | **ALL CONSISTENT** |
| `scripts/preflight_amendment7.py` | **READY** (100 trajectories, hashes OK) |
| `scripts/readout_amendment7.py --dry-run` | Reproduces published replication estimators |
| `scripts/test_readout_amendment7.py` | Production-path fixture **PASS** |
| Paper | `paper/main.pdf`, `paper/main_short.pdf` — draft **v0.9.10** |
| Scientific headline | H1 **not supported**; disagreement \(u\) does not beat selector \(c\) on held-out failure detection; null **replicates** after selector axis (Amendment 5) |

**Only open experiment:** the agent axis. **Amendment 10 supersedes Amendment
7's `gpt-6-luna` choice** and adopts the mentor's requested open-weight agents:
**DeepSeek V4.1 Flash** and **GLM-5.3**. They must pass a 41-instance capability
pilot before either is used for the 114 × 4 = 456-trajectory axis.

Current pilot evidence (not yet an Amendment 10 capability verdict):

- DeepSeek V4.1 Flash submitted 31/41 patches (75.6%), below the 80% P10-1
  gate. Three runs died under the old 4k output cap; the corrected config uses
  16k and a pinned provider, so this pilot must be rerun before judging it.
- GLM-5.3 submitted 13/41 (31.7%), with 22 step-limit and six format exits. It
  is not viable in the current scaffold unless a specific harness/config defect
  is demonstrated and fixed.
- The 100 `gpt-6-luna` trajectories are checkpoint-hashed historical data.
  Never edit or silently combine them with an open-model arm. If DeepSeek is
  selected, use a **fresh output directory**.

**Do not spend Modal** until 1 October unless the user explicitly approves paid overage. Prereg caps Modal at **\$26** inside the \$30 free credit (Amendment 10a §2 reconciles the ledger: the old \$22 covered the axis only, not the Amendment 10 pilot and its gold) (Amendment 7b).

---

## No-spend gate (run first every session)

```bash
chmod +x /Users/jmorante/Downloads/xtail_spec/silentfail_bench/scripts/preflight_pre_oct1.sh
/Users/jmorante/Downloads/xtail_spec/silentfail_bench/scripts/preflight_pre_oct1.sh
```

If anything fails, fix before generation or gold.

---

## What to do **before** 1 October (no Modal bill)

1. **Read Amendment 10 first**, then Amendments 7a–7c in
   `silentfail_bench/PREREGISTRATION_SWEBENCH_PILOT.md`. The old runbook's
   `gpt-6-luna` generation command is superseded.
2. **Confirm secrets:** mini-swe-agent has `OPENROUTER_API_KEY`; Modal CLI is
   logged in. Do not print or commit keys.
3. **Confirm pilot set:** the same 41 Verified instances used by Amendment 6.
   The 114-instance Multilingual file is only for the full axis after a pilot
   passes.
4. **Dry-run only** unless user says otherwise:
   - Re-run `preflight_pre_oct1.sh` after any code edit.
   - If touching paper: `python3 paper/check_consistency.py` must stay green.
5. **Optional paper polish (no new data):** typos, mentor-facing one-pager, ICLR workshop abstract from `main_short.tex` — **do not** change claimed numbers without updating JSON readouts and re-running `check_consistency.py`.
6. **Algoverse cohort:** user is targeting **Sep 20 – Dec 13** (ICLR & ICML workshops). Submission narrative = rigorous **failure-to-reject** on panel disagreement, not a positive H1.

---

## What to do **on/after** 1 October (Modal credit available)

Execute this decision tree:

| Stage | Command / action | Spend |
|--------|------------------|--------|
| 0 | Run the no-spend gate and confirm no Modal tasks are active | \$0 |
| 1 | Rerun the **corrected DeepSeek** 41-instance capability pilot in a fresh directory (`mini_swe_deepseek.yaml`) | Amendment 10 cap: \$5 API |
| 2 | Run official harness gold for all 41 attempts; count empty/limit/format exits as failures | Modal credit |
| 3 | Compute P10-1 submission ≥80%, P10-2 failure ≥10 points below matched `gpt-5-mini`, P10-3 cost ≤\$0.040/trajectory | \$0 |
| 4a | If all pass: freeze DeepSeek as the agent and run a **fresh** 114×4 axis | Budget must be rechecked before launch |
| 4b | If any fail: report honestly; do not launch 456 trajectories or substitute GLM without a preregistered, evidence-based reason | \$0 |
| 5 | After an agent is selected, reuse downstream bundle → panel → gpt-5 all-candidate → four gold arms → readout stages from the old runbook | Per Amendment 7b |

**Operational rules**

- Killing the local orchestrator **does not** stop Modal sandboxes → run `scripts/modal_orphans.py` and stop `swe-rex` app if needed.
- Failed instances may be wrongly marked in `preds.json` → `scripts/replication_cleanup.py`.
- Do not call submission rate capability. P10-2 requires official executed gold.
- Do not resume the 100 `gpt-6-luna` rows as though they were DeepSeek rows.
- After readout: update `paper/main.tex` / `main_short.tex` **only** with numbers from `readout_amendment7.json`; run `check_consistency.py`; rebuild PDFs if user wants PDF.
- Interpretation per Amendment 7a: **INCONCLUSIVE** is acceptable; null claim needs **upper bound &lt; +0.05**, not merely CI includes zero.

---

## Key paths

- Governing design: Amendment 10 in `silentfail_bench/PREREGISTRATION_SWEBENCH_PILOT.md`
- Downstream runbook (generation stage superseded): `silentfail_bench/RESUME_AMENDMENT7.md`
- Defects: `silentfail_bench/DEFECTS.md`
- Research log: `UPDATES.md`
- Venv: `silentfail_bench/.venv-swebench/` (SWE-bench + Modal)
- Candidate configs: `silentfail_bench/configs/mini_swe_deepseek.yaml`,
  `silentfail_bench/configs/mini_swe_glm.yaml`

---

## Copy-paste prompt for Claude

```text
You are continuing the SilentFail-Bench project at /Users/jmorante/Downloads/xtail_spec.

Read CLAUDE_HANDOFF_OCT1.md and Amendment 10 plus Amendments 7a–7c in silentfail_bench/PREREGISTRATION_SWEBENCH_PILOT.md. Read RESUME_AMENDMENT7.md only for downstream stages; its gpt-6-luna generation command is superseded.

Today’s date: [FILL IN]. Modal $30 free credit resets 2026-10-01. Do NOT start Modal generation or gold before that date unless I explicitly approve paid overage.

First action every session:
  /Users/jmorante/Downloads/xtail_spec/silentfail_bench/scripts/preflight_pre_oct1.sh
Then, on/after 1 Oct, the pilot is one command, not improvised:
  silentfail_bench/scripts/run_amendment10_pilot.sh
If it fails, fix the code/docs before spending money.

Context: Paper is draft v0.9.10; primary H1 is not supported on Verified + Multilingual, and Amendment 5's strong-selector axis is done. Mentor feedback requests capable open-weight agents, specifically DeepSeek V4.1 Flash and GLM-5.3. Amendment 10 adopts them and supersedes gpt-6-luna. Existing evidence: DeepSeek submitted 31/41 under an old 4k cap (now corrected to 16k); GLM submitted 13/41 and is not viable in the current scaffold. The 100 gpt-6-luna trajectories are checkpointed historical data and must not be mixed with the open-model arm.

Before Oct 1: only no-spend verification and documentation. After Oct 1: first rerun the corrected DeepSeek 41-instance capability pilot in a fresh output directory and run official gold. Apply all Amendment 10 gates: ≥80% submitted; unconditional failure rate ≥10 points better than matched gpt-5-mini; ≤$0.040/trajectory. If it fails any gate, stop and report — do not launch the 456-trajectory axis, do not substitute GLM-5.3, and do not revert to gpt-6-luna. That branch is fixed in Amendment 10a §1, which resolves the conflict between this handoff and Amendment 10's original fallback reading. If it passes, freeze DeepSeek and then adapt the downstream Amendment 7 stages using a fresh open-model output directory, after rechecking the budget.

When you change code or numbers, say what you ran to verify. Do not amend git commits unless I ask.
```
