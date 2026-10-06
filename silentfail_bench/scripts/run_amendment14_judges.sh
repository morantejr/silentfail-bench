#!/usr/bin/env bash
# Amendment 14: can a gpt-6 judge be the selector on the Amendment 13 pool?  Stage by stage, each gated.
#   bash scripts/run_amendment14_judges.sh smoke        # 3 instances x 4 candidates per judge, cost projected, nothing kept
#   bash scripts/run_amendment14_judges.sh judge        # all 456 attempts, both judges, external cost check
#   bash scripts/run_amendment14_judges.sh readout      # Amendment 7a readout per judge on the EXISTING gold; no new gold, no Modal
set -euo pipefail
cd "$(dirname "$0")/.."
V=.venv-swebench/bin/python
O=data/agent_axis_deepseek
I=configs/agent_axis_instances_multilingual.txt
JUDGES="gpt-6-sol gpt-6-luna"
CAP=40.00
LEDGER=$O/spend_ledger_a14.txt
grep -q "^# Amendment 14, 6 October 2026" PREREGISTRATION_SWEBENCH_PILOT.md || { echo "ABORT: Amendment 14 not recorded"; exit 2; }
[ -f $O/rescoring_bundle.jsonl ] || { echo "ABORT: no Amendment 13 bundle"; exit 2; }
case "${1:-}" in
smoke)
  rm -rf data/smoke_a14; mkdir -p data/smoke_a14
  head -3 $I > data/smoke_a14/instances.txt
  $V - $O data/smoke_a14 <<'PY'
import json, sys
from pathlib import Path
O, S = Path(sys.argv[1]), Path(sys.argv[2]); keep = set(S.joinpath("instances.txt").read_text().split())
rows = [l for l in O.joinpath("rescoring_bundle.jsonl").read_text().splitlines() if l.strip() and json.loads(l)["instance_id"] in keep]
S.joinpath("bundle.jsonl").write_text("\n".join(rows) + "\n"); print(f"   smoke bundle: {len(rows)} candidates over {len(keep)} instances")
PY
  for J in $JUDGES; do
    $V scripts/llm_judges.py run --bundle data/smoke_a14/bundle.jsonl --cache data/smoke_a14/judge_$J.jsonl --all-candidates --judges $J --conds J1 --workers 4
    $V - data/smoke_a14/judge_$J.jsonl $J <<'PY'
import json, sys
rows = [json.loads(l) for l in open(sys.argv[1]) if l.strip()]
cost = sum(float(r.get("est_cost_usd") or 0) for r in rows); n = len(rows)
print(f"   {sys.argv[2]}: {n} calls, est ${cost:.3f} -> x456 = ${cost/n*456:.2f}" if n else "   no rows")
if n and cost / n * 456 > 30: print("   ABORT: projected over $30 for one judge"); sys.exit(3)
PY
  done
  rm -rf data/smoke_a14; echo "== smoke OK =="
  ;;
judge)
  for J in $JUDGES; do
    echo "== $J judges all 456 candidates, J1 ==" | tee -a $LEDGER
    $V scripts/llm_judges.py run --bundle $O/rescoring_bundle.jsonl --cache $O/judge_$J.jsonl --all-candidates --judges $J --conds J1 --workers 8
    $V - $O/judge_$J.jsonl $J $CAP <<'PY' | tee -a $LEDGER
import json, sys
rows = [json.loads(l) for l in open(sys.argv[1]) if l.strip()]
cost = sum(float(r.get("est_cost_usd") or 0) for r in rows)
print(f"   {sys.argv[2]}: {len(rows)} judgments, scaffold-estimated ${cost:.2f} (cap ${float(sys.argv[3]):.2f} for both)")
if len(rows) < 456: print(f"   ABORT: {len(rows)} of 456 judged"); sys.exit(3)
PY
  done
  ;;
readout)
  for J in $JUDGES; do
    echo "== Amendment 7a readout with $J as the selector =="
    $V scripts/readout_amendment7.py --bundle $O/rescoring_bundle.jsonl --instances $I \
        --scores-dir $O/modal/prefix4k --judge-cache $O/judge_$J.jsonl --judge $J \
        --random-picks $O/gold_inputs/random_picks.jsonl --out $O/readout_amendment14_$J.json \
        --agent-label deepseek-v4.1-flash \
        --comparator-gold data/gold_modal/a13_gpt5mini_100_merged/report.json --comparator-preds data/agent_axis_gpt5mini_100/predictions_merged.jsonl
    $V -c "
import json; r=json.load(open('$O/readout_amendment14_$J.json')); g=r['selector_gate_A7a_2']; a=r['arms']
k=[x for x in a if x.startswith('S_judge')][0]
print(f'   $J: pool AUROC {g[\"pool_AUROC_detect\"]} (bar 0.65) gate passes={g[\"passes\"]} | committed failures {a[k][\"failures\"]} of {a[k][\"n\"]} | verdict {r[\"confirmatory_verdict\"][\"branch\"]}')"
  done
  ;;
*) echo "usage: $0 {smoke|judge|readout}"; exit 2 ;;
esac
