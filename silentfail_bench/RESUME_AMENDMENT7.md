# Resuming Amendment 7 on 1 October 2026

Everything below is one sitting. Run the preflight first; it exits non-zero if
anything would waste money or silently discard the 100 paused trajectories.

```bash
cd /Users/jmorante/Downloads/xtail_spec/silentfail_bench
.venv-swebench/bin/python scripts/preflight_amendment7.py
```

## Decide the instance set before you start

Amendment 7 declared **456 trajectories**, which is 114 instances at four
candidates. The replication instance file holds **128**, because 14 of those
never produced a usable row. Using the larger file would generate 512
trajectories and cost about **$1.30 more** than the amendment declared, for
instances the replication could not analyse.

`configs/agent_axis_instances_multilingual.txt` is the 114-instance set,
generated from the rows the replication actually analysed. All 100 already
generated trajectories are inside it, so nothing is wasted. **Use it**, and the
run matches the declared scope.

## The stages

```bash
V=.venv-swebench/bin/python
I=configs/agent_axis_instances_multilingual.txt
O=data/agent_axis

# 1. generate — resumes; the 100 instances in preds.json are skipped automatically
$V scripts/run_replication_modal.py --stage generate \
    --instances-file $I --out $O \
    --mini-config configs/mini_swe_gpt6luna.yaml \
    --k 4 --workers 8 --step-limit 40 --cost-cap 12

# 2. bundle — must be exactly 114 x 4 candidate slots
$V scripts/run_replication_modal.py --stage bundle --instances-file $I --out $O

# 3. panel scoring on Modal, 4k prefix (modal/rescore.py --mode prefix)
#    needs only the four registered auditors plus the 4B selector
.venv-swebench/bin/modal volume put silentfail-bundles \
    $O/rescoring_bundle.jsonl /agent_axis/rescoring_bundle.jsonl
.venv-swebench/bin/python -m modal run modal/rescore.py \
    --run agent_axis --out prefix4k --max-tokens 4000 --mode prefix \
    --only "ray2333_grm_gemma2_2b_rewardmodel_ft ray2333_grm_llama3_2_3b_rewardmodel_ft skywork_skywork_reward_v2_llama_3_2_1b skywork_skywork_reward_v2_qwen3_1_7b skywork_skywork_reward_v2_qwen3_4b"
.venv-swebench/bin/modal volume get silentfail-out /agent_axis $O/modal

# 4. gpt-5 scores EVERY candidate (mandatory strong-selector arm; J1 only)
$V scripts/llm_judges.py run --bundle $O/rescoring_bundle.jsonl \
    --cache $O/judge_gpt5.jsonl --all-candidates \
    --judges gpt-5 --conds J1 --workers 8

# 5. prepare four candidate-indexed gold runs and the frozen random arm.
#    This is local-only and refuses a bundle other than the registered 114 x 4.
$V scripts/prepare_amendment7.py

# 6. gold every non-empty attempt. Four run IDs are required because the
#    harness keys reports by instance_id and cannot distinguish duplicate IDs.
#    Empty patches are omitted from Modal and count as failures in the readout.
for C in 0 1 2 3; do
  $V modal/gold_modal.py --dataset swe-bench/SWE-Bench_Multilingual \
      --preds $O/gold_inputs/cand${C}_predictions.jsonl \
      --candidates $O/gold_inputs/cand${C}_mapping.jsonl \
      --run-id a7_agent_axis_cand${C} $([ "$C" = 0 ] && echo --oracle 3)
done

# 7. analysis. Refuses partial candidate gold, incomplete judge/panel scores,
#    a changed random arm, or a bundle other than the registered 456 slots.
$V scripts/readout_amendment7.py
```

## The readout

`scripts/readout_amendment7.py` implements Amendment 7a's rules, not Amendment
7's. It was written and **dry-run in September**, before the data it is for
exists, because a script first run on the day it matters is a script that fails
that day. The dry run regression-tests the three endpoint estimators against
the gpt-5-mini replication values the paper already publishes. The production
candidate/gold/judge joins are separate (those complete sources do not exist
yet); they were exercised with a local 114×4 fixture and fail closed on partial
inputs:

```bash
.venv-swebench/bin/python scripts/readout_amendment7.py --dry-run
.venv-swebench/bin/python scripts/test_readout_amendment7.py
```

The first exits non-zero if the estimators stop reproducing published values.
The second builds deterministic synthetic candidate gold in a temporary
directory, exercises the full production path, then deletes one gold record
and proves partial input is rejected without writing a result. Run both before
October in case anything upstream has drifted. The rules enforced:

- `gpt-5` selection is **mandatory**, not conditional on the 4B arm.
- Gate: the selector's pool `AUROC(-c)` on the **new** graded pool must exceed 0.65,
  and its committed arm must fail less often than the random arm.
- Failure rate is **unconditional**: empty patches and step-limit exits count as
  failures, in both arms.
- Primary estimator is the **unfitted** one; print raw `AUROC(-c)` beside every
  out-of-fold baseline so a degenerate baseline is visible.
- Inference floor is **20 failures**, not 10.
- A null claim requires the **upper bound below +0.05**. "The CI includes zero"
  is not equivalence.

Two things the dry run already established, worth knowing before October:

- **The capability comparison may be indeterminate.** A7-P1 asks for an
  unconditional per-attempt failure rate, but only 213 of the replication's 456
  candidate slots have gold, so gpt-5-mini's rate is a band, 0.476 to 0.752, not
  a point. The comparison is only decisive if the new agent's upper bound clears
  the old agent's lower bound. The script reports the band and says so rather
  than quoting a false point estimate.
- **Under A7a's rules the replication cell itself would read INCONCLUSIVE**, with
  an upper bound of +0.175 against a margin of +0.05. That is the branch
  Amendment 7a predicted in advance would be the likely one. It is not a failure
  of the experiment; it is the design being honest about its power.

## Budget

| Item | Estimate |
|---|---|
| Modal sandboxes, 356 remaining trajectories at $0.0236 | ~$8.40 |
| Modal gold, 4 candidate-indexed runs (up to 456 attempts) | ~$12 |
| OpenAI generation and judge | ~$11 |

Modal's free credit resets to $30 on 1 October, so the Modal side is covered.
The OpenAI side comes from the ~$46 remaining. Amendment 7b raises the Modal
cap from $12 to $22 before gold exists: the earlier $3 gold estimate covered
only one selected arm and could not establish A7-P1's per-attempt capability
criterion. The revised estimate is ~$8.40 generation + ~$12 gold = ~$20.40,
inside the $30 free credit. Do not incur paid overage.

## Things that have already gone wrong here

- Killing the orchestrator does **not** stop Modal sandboxes. They run to
  `runtime_timeout`. Stop the `swe-rex` app explicitly, and
  `scripts/modal_orphans.py` terminates strays.
- The batch runner skips by `preds.json`, not by trajectory files, so a failed
  instance can be marked done. `scripts/replication_cleanup.py` prunes it.
- OpenAI rate limits killed 138 partial trajectories once. Twelve concurrent
  workers and a high retry count is what worked.
- Do not regenerate the paused 100. They are hashed in
  `data/agent_axis/CHECKPOINT_HASHES.json`.
