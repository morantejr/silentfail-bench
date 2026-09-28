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

# 2. bundle — candidates -> rescoring_bundle.jsonl
$V scripts/run_replication_modal.py --stage bundle --instances-file $I --out $O

# 3. panel scoring on Modal, 4k prefix (modal/rescore.py --mode prefix)
#    needs only the four registered auditors plus the 4B selector

# 4. select — argmax of the selection critic's prefix-4k scores -> predictions.jsonl
$V scripts/run_replication_modal.py --stage select --instances-file $I --out $O \
    --scores-dir $O/modal/prefix4k

# 5. gold
$V modal/gold_modal.py --dataset swe-bench/SWE-Bench_Multilingual \
    --preds $O/predictions.jsonl --run-id a7_agent_axis --oracle 3

# 6. rows
$V scripts/run_replication_modal.py --stage rows --instances-file $I --out $O \
    --scores-dir $O/modal/prefix4k --gold-report data/gold_modal/a7_agent_axis/report.json
```

## What still has to be written

There is **no readout script for Amendment 7 yet**. It must implement the rules
frozen in Amendment 7a, not the ones in Amendment 7:

- `gpt-5` selection is **mandatory**, not conditional on the 4B arm.
- Gate: the selector's pool `AUROC(-c)` on the **new** pool must exceed 0.65,
  and its committed arm must fail less often than the random arm.
- Failure rate is **unconditional**: empty patches and step-limit exits count as
  failures, in both arms.
- Primary estimator is the **unfitted** one; print raw `AUROC(-c)` beside every
  out-of-fold baseline so a degenerate baseline is visible.
- Inference floor is **20 failures**, not 10.
- A null claim requires the **upper bound below +0.05**. "The CI includes zero"
  is not equivalence.

`scripts/readout_amendment5_corrected.py` is the closest template: it already
computes the three estimators, prints raw beside out-of-fold, reports the upper
bound against the margin, and carries the paired committed-versus-random
contrast. Copy its structure.

## Budget

| Item | Estimate |
|---|---|
| Modal sandboxes, 356 remaining trajectories at $0.0236 | ~$8.40 |
| Modal gold | ~$3 |
| OpenAI generation and judge | ~$11 |

Modal's free credit resets to $30 on 1 October, so the Modal side is covered.
The OpenAI side comes from the ~$46 remaining. Amendment 7 declared caps of $15
OpenAI and $12 Modal; the estimates above sit inside both.

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
