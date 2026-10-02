#!/usr/bin/env bash
# Amendment 13: the agent axis on DeepSeek V4.1 Flash, stage by stage, each gated.
#
#   bash scripts/run_amendment13_axis.sh smoke      # one instance, project cost, delete
#   bash scripts/run_amendment13_axis.sh generate   # 456 trajectories, external cost gate
#   bash scripts/run_amendment13_axis.sh score      # bundle, Modal GPU panel, gpt-5 judge
#   bash scripts/run_amendment13_axis.sh gold       # prepare + four candidate-indexed gold runs
#   bash scripts/run_amendment13_axis.sh comparator # gpt-5-mini k=1 at 100 steps + gold
#   bash scripts/run_amendment13_axis.sh readout    # Amendment 7a readout on the new directory
#
# Stages are separate commands on purpose: a twelve-hour pipeline that is one process cannot
# be inspected between the parts that spend money. Every stage refuses to run unless the
# registration and the preflight say it may.
set -euo pipefail
cd "$(dirname "$0")/.."
V=.venv-swebench/bin/python
MODAL=.venv-swebench/bin/modal
I=configs/agent_axis_instances_multilingual.txt
O=data/agent_axis_deepseek
OC=data/agent_axis_gpt5mini_100
CFG=configs/mini_swe_deepseek_axis.yaml
CFGC=configs/mini_swe_openai_axis.yaml
OR_CAP=20.00
STAGE="${1:-}"
LEDGER=$O/spend_ledger.txt

grep -q "^# Amendment 13, 1 October 2026" PREREGISTRATION_SWEBENCH_PILOT.md || { echo "ABORT: Amendment 13 not recorded"; exit 2; }

spend() {
  $V - <<'INNERPY'
import os, sys, requests
from dotenv import load_dotenv
load_dotenv(os.path.expanduser("~/Library/Application Support/mini-swe-agent/.env"), override=True)
d = requests.get("https://openrouter.ai/api/v1/credits",
                 headers={"Authorization": "Bearer " + os.environ["OPENROUTER_API_KEY"]}, timeout=30).json()
if "data" not in d or "total_usage" not in d["data"]:
    print(f"ABORT: unexpected /credits payload, keys={list(d)}", file=sys.stderr); sys.exit(4)
print(f"{d['data']['total_usage']:.6f}")
INNERPY
}

case "$STAGE" in
smoke)
  $V scripts/preflight_amendment13.py >/dev/null 2>&1 || { echo "ABORT: preflight failed; run it to see why"; exit 2; }
  echo "== smoke: first Multilingual instance, k=1, DeepSeek axis config =="
  rm -rf data/smoke_a13
  B=$(spend); echo "   OpenRouter usage before: \$$B"
  $V scripts/run_replication_modal.py --stage generate --instances-file $I --out data/smoke_a13 \
      --mini-config $CFG --base-config swebench_backticks.yaml --k 1 --workers 1 --step-limit 100 --cost-cap 2 --limit 1 >/dev/null 2>&1 || true
  A=$(spend)
  $V - data/smoke_a13 "$B" "$A" "$OR_CAP" <<'INNERPY'
import json, sys, glob
from pathlib import Path
out, before, after, cap = Path(sys.argv[1]), float(sys.argv[2]), float(sys.argv[3]), float(sys.argv[4])
fs = glob.glob(f"{out}/traj/cand0/*/*.traj.json")
if not fs: print("ABORT: smoke wrote no trajectory"); sys.exit(2)
d = json.loads(Path(fs[0]).read_text()); info = d.get("info", {}); st = info.get("model_stats") or {}
cfg = info.get("config") or {}
steps = sum(1 for m in d.get("messages", []) if m.get("role") == "assistant")
ext = after - before
print(f"   {Path(fs[0]).parent.name}: exit={info.get('exit_status')} steps={steps} api_calls={st.get('api_calls')} "
      f"scaffold=${float(st.get('instance_cost') or 0):.4f} external=${ext:.4f}")
print(f"   applied: step_limit={cfg.get('agent',{}).get('step_limit')} cost_limit={cfg.get('agent',{}).get('cost_limit')} "
      f"max_tokens={(cfg.get('model',{}).get('model_kwargs') or {}).get('max_tokens')}")
print(f"   environment_type: {cfg.get('environment_type')}")
if "swerex_modal_closing" not in str(cfg.get("environment_type")):
    print("ABORT: the sandbox-closing environment did not apply (DEFECTS #21)"); sys.exit(2)
if cfg.get("agent", {}).get("step_limit") != 100 or cfg.get("agent", {}).get("cost_limit") != 1.0:
    print("ABORT: the axis config did not apply"); sys.exit(2)
proj = ext * 456
print(f"   projected x456 from the external measurement: ${proj:.2f} (cap ${cap:.2f})")
if proj > cap: print("ABORT: projection exceeds the OpenRouter cap; do not generate"); sys.exit(3)
INNERPY
  grep -h "terminated sandbox" data/smoke_a13/traj/cand0/*.log 2>/dev/null | head -1 | sed 's/^/   /' || true
  rm -rf data/smoke_a13
  echo "== smoke OK =="
  ;;
generate)
  $V scripts/preflight_amendment13.py >/dev/null 2>&1 || { echo "ABORT: preflight failed; run it to see why"; exit 2; }
  mkdir -p $O
  if [ "${A13_RESUME:-}" = "1" ] && grep -q "^openrouter_before " $LEDGER 2>/dev/null; then
    # Resuming: the cost gate is measured from the FIRST before-mark, so paused segments add up.
    B=$(grep "^openrouter_before " $LEDGER | head -1 | awk '{print $2}'); echo "openrouter_resume $(spend)" >> $LEDGER
    echo "== generate (RESUME): 456 trajectories, existing ones skipped (usage at first start \$$B) =="
  else
    B=$(spend); echo "openrouter_before $B" >> $LEDGER; echo "== generate: 456 trajectories (usage before \$$B) =="
  fi
  $V scripts/run_replication_modal.py --stage generate --instances-file $I --out $O \
      --mini-config $CFG --base-config swebench_backticks.yaml --k 4 --workers "${A13_WORKERS:-6}" --step-limit 100 --cost-cap $OR_CAP
  A=$(spend); echo "openrouter_after_generate $A" >> $LEDGER
  COST=$($V -c "print(f'{$A - $B:.4f}')")
  N=$(ls $O/traj/cand*/*/*.traj.json 2>/dev/null | wc -l | tr -d ' ')
  echo "   trajectories: $N / 456   measured OpenRouter spend: \$$COST (cap \$$OR_CAP)"
  $V - $O <<'INNERPY'
import json, glob, sys
cut = [f for f in glob.glob(f"{sys.argv[1]}/traj/cand*/*/*.traj.json")
       if float(((json.loads(open(f).read()).get("info") or {}).get("model_stats") or {}).get("instance_cost") or 0) >= 1.0]
print(f"   trajectories at the $1.00 cost ceiling: {len(cut)}" + ("  <- reported as cuts in the readout" if cut else ""))
INNERPY
  $V -c "import sys; sys.exit(0 if $COST <= $OR_CAP else 1)" || { echo "ABORT: spend exceeded the cap; do not proceed to scoring"; exit 3; }
  EF=$($V -c "import json,sys; d=json.load(open('$O/ENDPOINT_FAILURES.json')) if __import__('os').path.exists('$O/ENDPOINT_FAILURES.json') else {}; print(sum(len(v) for k,v in d.items() if k.startswith('cand')))")
  echo "   declared endpoint failures (ENDPOINT_FAILURES.json): $EF"
  [ $((N + EF)) -eq 456 ] || { echo "ABORT: $N trajectories + $EF declared endpoint failures != 456; generation incomplete -- inspect before scoring"; exit 3; }
  ;;
score)
  echo "== bundle =="
  $V scripts/run_replication_modal.py --stage bundle --instances-file $I --out $O
  $V - $O <<'INNERPY'
import json, sys
rows = [json.loads(l) for l in open(f"{sys.argv[1]}/rescoring_bundle.jsonl") if l.strip()]
inst = {r["instance_id"] for r in rows}
if len(rows) != 456 or len(inst) != 114: print(f"ABORT: bundle has {len(rows)} rows over {len(inst)} instances, expected 456 over 114"); sys.exit(3)
print(f"   bundle: {len(rows)} slots over {len(inst)} instances; submitted {sum(r['visible_pass'] for r in rows)}; empty {sum(r['excluded'] for r in rows)}")
INNERPY
  echo "== panel scoring on Modal GPU (repo ids, not slugs) =="
  $MODAL volume put silentfail-bundles $O/rescoring_bundle.jsonl /agent_axis_deepseek/rescoring_bundle.jsonl --force
  $V -m modal run modal/rescore.py --run agent_axis_deepseek --out prefix4k --max-tokens 4000 --mode prefix \
      --only "Ray2333/GRM-gemma2-2B-rewardmodel-ft Ray2333/GRM-Llama3.2-3B-rewardmodel-ft Skywork/Skywork-Reward-V2-Llama-3.2-1B Skywork/Skywork-Reward-V2-Qwen3-1.7B Skywork/Skywork-Reward-V2-Qwen3-4B"
  mkdir -p $O/modal
  $MODAL volume get silentfail-out /agent_axis_deepseek $O/modal --force
  ls $O/modal/prefix4k/scores_gpu_*.json | wc -l | xargs echo "   score files:"
  [ "$(ls $O/modal/prefix4k/scores_gpu_*.json 2>/dev/null | wc -l | tr -d ' ')" -ge 5 ] || { echo "ABORT: fewer than 5 score files; --only probably matched nothing"; exit 3; }
  echo "== gpt-5 judges every candidate, J1 =="
  $V scripts/llm_judges.py run --bundle $O/rescoring_bundle.jsonl --cache $O/judge_gpt5.jsonl --all-candidates --judges gpt-5 --conds J1 --workers 8
  ;;
gold)
  echo "== prepare gold inputs and the frozen random arm =="
  $V scripts/prepare_amendment7.py --bundle $O/rescoring_bundle.jsonl --instances $I --out $O/gold_inputs
  for C in 0 1 2 3; do
    echo "== gold: a7_agent_axis_cand$C =="
    $V modal/gold_modal.py --dataset swe-bench/SWE-Bench_Multilingual \
        --preds $O/gold_inputs/cand${C}_predictions.jsonl --candidates $O/gold_inputs/cand${C}_mapping.jsonl \
        --run-id a7_agent_axis_cand$C $([ "$C" = 0 ] && echo --oracle 3)
  done
  ;;
comparator)
  mkdir -p $OC
  echo "== equal-budget comparator: gpt-5-mini k=1 at 100 steps =="
  $V scripts/run_replication_modal.py --stage generate --instances-file $I --out $OC \
      --mini-config $CFGC --base-config swebench.yaml --k 1 --workers 6 --step-limit 100 --cost-cap 20
  $V - $OC <<'INNERPY'
import json, sys
from pathlib import Path
out = Path(sys.argv[1]); src = out / "traj/cand0/preds.json"
if not src.exists(): print(f"ABORT: {src} missing"); sys.exit(2)
d = json.loads(src.read_text()); rows = list(d.values()) if isinstance(d, dict) else d
(out / "predictions.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
print(f"   comparator: {len(rows)} rows -> predictions.jsonl")
INNERPY
  $V modal/gold_modal.py --dataset swe-bench/SWE-Bench_Multilingual --preds $OC/predictions.jsonl --run-id a13_gpt5mini_100 --oracle 3
  ;;
readout)
  echo "== Amendment 7a readout on the DeepSeek axis =="
  $V scripts/readout_amendment7.py --bundle $O/rescoring_bundle.jsonl --instances $I \
      --scores-dir $O/modal/prefix4k --judge-cache $O/judge_gpt5.jsonl \
      --random-picks $O/gold_inputs/random_picks.jsonl --out $O/readout_amendment7_deepseek.json \
      --agent-label deepseek-v4.1-flash
  echo "== stop the swe-rex deployment =="
  for a in $($MODAL app list 2>/dev/null | grep -E "deployed|running" | grep -oE "ap-[A-Za-z0-9]+"); do $MODAL app stop --yes "$a" >/dev/null 2>&1 || true; done
  $MODAL app list 2>/dev/null | grep -cE "deployed|running|stopping" | xargs echo "   Modal apps still live:"
  ;;
*)
  echo "usage: $0 {smoke|generate|score|gold|comparator|readout}"; exit 2 ;;
esac
