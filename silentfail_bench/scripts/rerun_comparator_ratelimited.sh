#!/usr/bin/env bash
# Amendment 13 comparator: regenerate the instances that exited RateLimitError (OpenAI's TPM
# ceiling at 6 workers, not the model's behaviour) at a single worker, merge, and run one gold on
# the merged predictions -- the Amendment 11 precedent (pilot_a11_gpt5mini/MERGE_NOTE.md).
#
#   bash scripts/rerun_comparator_ratelimited.sh
set -euo pipefail
cd "$(dirname "$0")/.."
V=.venv-swebench/bin/python
OC=data/agent_axis_gpt5mini_100
RR=data/agent_axis_gpt5mini_100_rerun_w1
I=configs/agent_axis_instances_multilingual.txt
[ -f $OC/predictions.jsonl ] || { echo "ABORT: comparator stage has not produced $OC/predictions.jsonl"; exit 2; }
pgrep -f "run_amendment13_axis.sh comparator" >/dev/null && { echo "ABORT: the comparator stage is still running"; exit 2; }

IDS=$($V - <<'PY'
import json, glob
ids = []
for f in glob.glob("data/agent_axis_gpt5mini_100/traj/cand0/*/*.traj.json"):
    if json.load(open(f))["info"].get("exit_status") in ("RateLimitError", "KeyboardInterrupt"): ids.append(f.split("/")[-2])
import os
ids += [i for i in os.environ.get("EXTRA_IDS", "").split() if i not in ids]   # operator-declared, e.g. a kill after a reaped sandbox
print(" ".join(sorted(ids)))
PY
)
N=$(echo $IDS | wc -w | tr -d ' ')
echo "== $N comparator instances exited RateLimitError: $IDS =="
[ "$N" -gt 0 ] || { echo "nothing to rerun"; exit 0; }
FLT="^($(echo $IDS | sed 's/ /|/g'))$"
mkdir -p $RR/traj/cand0
echo "== rerun at 2 workers, then any that rate-limit again at 1 worker (same config, same seed) =="
env -u OPENAI_API_BASE -u OPENAI_BASE_URL PYTHONPATH=$PWD $V -m envs.run_swebench_closing --subset multilingual --split test \
    --filter "$FLT" -o $RR/traj/cand0 -w 2 -c swebench.yaml -c configs/mini_swe_openai_axis.yaml \
    -c model.model_kwargs.seed=1000 -c agent.step_limit=100 > $RR/gen.log 2>&1 || true
AGAIN=$($V - <<'PY2'
import json, glob
print(" ".join(sorted(f.split("/")[-2] for f in glob.glob("data/agent_axis_gpt5mini_100_rerun_w1/traj/cand0/*/*.traj.json")
                       if json.load(open(f))["info"].get("exit_status") == "RateLimitError")))
PY2
)
if [ -n "$AGAIN" ]; then
  echo "== $(echo $AGAIN | wc -w | tr -d ' ') rate-limited again at 2 workers; rerunning at 1: $AGAIN =="
  $V - "$AGAIN" <<'PY3'
import json, sys
from pathlib import Path
ids = sys.argv[1].split(); p = Path("data/agent_axis_gpt5mini_100_rerun_w1/traj/cand0/preds.json"); d = json.loads(p.read_text())
for i in ids: d.pop(i, None); 
p.write_text(json.dumps(d, indent=2))
PY3
  FLT2="^($(echo $AGAIN | sed 's/ /|/g'))$"
  env -u OPENAI_API_BASE -u OPENAI_BASE_URL PYTHONPATH=$PWD $V -m envs.run_swebench_closing --subset multilingual --split test \
      --filter "$FLT2" -o $RR/traj/cand0 -w 1 --redo-existing -c swebench.yaml -c configs/mini_swe_openai_axis.yaml \
      -c model.model_kwargs.seed=1000 -c agent.step_limit=100 >> $RR/gen.log 2>&1 || true
fi
$V - "$IDS" <<'PY'
import json, sys, glob, collections
from pathlib import Path
ids = sys.argv[1].split()
OC, RR = Path("data/agent_axis_gpt5mini_100"), Path("data/agent_axis_gpt5mini_100_rerun_w1")
orig = {json.loads(l)["instance_id"]: json.loads(l) for l in (OC / "predictions.jsonl").read_text().splitlines() if l.strip()}
rer = json.loads((RR / "traj/cand0/preds.json").read_text()) if (RR / "traj/cand0/preds.json").exists() else {}
ex = collections.Counter()
for i in ids:
    f = RR / f"traj/cand0/{i}/{i}.traj.json"
    ex[json.loads(f.read_text())["info"].get("exit_status") if f.exists() else "MISSING"] += 1
still = [i for i in ids if i not in rer or not (rer[i].get("model_patch") or "").strip()]
merged = dict(orig)
for i in ids:
    if i in rer: merged[i] = rer[i]
(OC / "predictions_merged.jsonl").write_text("".join(json.dumps(merged[i]) + "\n" for i in sorted(merged)))
(OC / "MERGE_NOTE.md").write_text(f"""# predictions_merged.jsonl

{len(merged)} rows. {len(merged) - len(ids)} from this directory's own run (6 workers); {len(ids)} regenerated at a
two workers (then one for any that rate-limited again) in `../agent_axis_gpt5mini_100_rerun_w1` because they exited `RateLimitError` on
OpenAI's TPM ceiling rather than on the model's own behaviour:
{', '.join(ids)}.

Rerun exits: {dict(ex)}. Still empty after the rerun (counted as failures): {still or 'none'}.
Counting a trajectory our rate limit killed as a capability failure would inflate the comparator's
failure rate; the Amendment 11 precedent (pilot_a11_gpt5mini/MERGE_NOTE.md) is followed exactly.
One gold run is made on the merged predictions as `a13_gpt5mini_100_merged`.
""")
print(f"merged: {len(merged)} rows; rerun exits {dict(ex)}; still empty {still}")
PY
echo "== one gold on the merged predictions =="
$V modal/gold_modal.py --dataset swe-bench/SWE-Bench_Multilingual --preds $OC/predictions_merged.jsonl --run-id a13_gpt5mini_100_merged --oracle 3
echo "== done: readout should use data/gold_modal/a13_gpt5mini_100_merged/report.json and $OC/predictions_merged.jsonl =="
