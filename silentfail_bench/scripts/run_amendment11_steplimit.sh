#!/usr/bin/env bash
# Amendment 11: does a larger step budget change the Amendment 10 result?
#
# Two arms at step_limit=100 on the same 41 Verified instances, because P10-2's
# comparator (gpt-5-mini, 0.486) was measured at 40 steps and raising only the
# candidate's budget would hand DeepSeek the gap. The 100-step gpt-5-mini figure is
# the comparator for THIS amendment only and is not leaderboard-comparable; the
# 40-step figure remains the paper's primary.
#
# POST HOC with respect to the budget: 100 was chosen knowing DeepSeek submitted
# 32 of 41 against an 80% bar. A pass here is not Amendment 10's P10-1 passing.
#
# Base config MUST match each arm's model class. swebench_backticks.yaml supplies the
# ```mswea_bash_command fence for the text-based class; swebench.yaml supplies native
# tool-call prompts. Crossing them produced zero parseable actions and cost a day.
set -euo pipefail
cd "$(dirname "$0")/.."

LIMIT=100
# Arm selector. The gpt-5-mini comparator needs a valid OPENAI_API_KEY; a smoke test on
# 2026-10-01 returned "Your API key has been invalidated", so P11-2 is blocked and the
# deviation is recorded in the pre-registration. "deepseek" answers P11-1 and P11-3 alone.
ARMS="${1:-both}"
case "$ARMS" in deepseek|gpt5mini|both) ;; *)
  echo "ABORT: arm must be deepseek, gpt5mini or both (got '$ARMS')"; exit 2 ;; esac
V=.venv-swebench/bin/python
MINI=.venv-swebench/bin/mini-extra
INSTANCES=configs/pilot_instances_swebench41_complete.txt
N=$(grep -c . "$INSTANCES")
OR_CAP=3.00          # OpenRouter, measured externally
ARM_CAP=3.00         # per-arm scaffold-reported cost ceiling

echo "== Amendment 11: step budget $LIMIT, two arms, $N instances =="

# --- guards -------------------------------------------------------------
[ "$N" -eq 41 ] || { echo "ABORT: instance file has $N entries, expected 41"; exit 2; }
grep -q "^# Amendment 11, 1 October 2026" PREREGISTRATION_SWEBENCH_PILOT.md || {
  echo "ABORT: Amendment 11 is not recorded in the pre-registration. Record it before running."; exit 2; }
grep -q "step_limit = 100" PREREGISTRATION_SWEBENCH_PILOT.md || {
  echo "ABORT: the pre-registration does not declare step_limit = 100; the budget must be registered, not improvised."; exit 2; }
for d in data/pilot_a11_deepseek data/pilot_a11_gpt5mini; do
  case "$ARMS:$d" in
    deepseek:*gpt5mini|gpt5mini:*deepseek) continue ;;
  esac
  [ -e "$d" ] && { echo "ABORT: $d already exists; refusing to overwrite a prior run"; exit 2; }
done

spend() {  # OpenRouter total usage. Uses python-dotenv rather than hand-parsing the env
           # file: a hand-rolled split sends surrounding quotes along with the value and
           # the endpoint answers 401, which reads as a revoked key rather than a parse bug.
  $V - <<'INNERPY'
import os, sys, requests
from dotenv import load_dotenv
load_dotenv(os.path.expanduser("~/Library/Application Support/mini-swe-agent/.env"), override=True)
r = requests.get("https://openrouter.ai/api/v1/credits",
                 headers={"Authorization": "Bearer " + os.environ["OPENROUTER_API_KEY"]},
                 timeout=30)
d = r.json()
# Assert the shape. A 200 with a changed payload would otherwise yield 0.000000 for both
# readings, making the measured cost 0.00 and passing the cap check silently.
if "data" not in d or "total_usage" not in d["data"]:
    print(f"ABORT: unexpected /credits payload, keys={list(d)}", file=sys.stderr); sys.exit(4)
print(f"{d['data']['total_usage']:.6f}")
INNERPY
}

BEFORE=$(spend) || { echo "ABORT: could not read OpenRouter usage; the cost gate would be blind."; exit 4; }
echo "   OpenRouter usage before: \$$BEFORE"
$V -c "import sys; sys.exit(0 if $BEFORE > 0 else 1)" || {
  echo "ABORT: usage reads exactly 0.000000, which is a payload change rather than a fresh account."; exit 4; }

FILTER="^($(paste -sd'|' "$INSTANCES"))$"

run_arm() {  # name base overlay outdir
  local name="$1" base="$2" overlay="$3" out="$4"
  echo
  echo "== arm $name: $base + $overlay at step_limit=$LIMIT =="
  $MINI swebench --subset verified --split test --filter "$FILTER" \
    -o "$out" -w 6 -c "$base" -c "$overlay" -c "agent.step_limit=$LIMIT"
  # scaffold-reported cost, summed from the trajectories
  $V - "$out" "$ARM_CAP" <<'PY'
import json, sys
from pathlib import Path
out, cap = Path(sys.argv[1]), float(sys.argv[2])
tot, n = 0.0, 0
for f in out.glob("*/*.traj.json"):
    st = (json.loads(f.read_text()).get("info") or {}).get("model_stats") or {}
    c = st.get("instance_cost")
    if c is not None:
        tot += float(c); n += 1
print(f"   scaffold-reported cost: ${tot:.4f} over {n} trajectories (${tot/max(n,1):.4f} each)")
if tot > cap:
    print(f"ABORT: arm cost ${tot:.4f} exceeded the ${cap:.2f} cap. Gold NOT run."); raise SystemExit(3)
PY
}

[ "$ARMS" = gpt5mini ] || run_arm deepseek swebench_backticks.yaml configs/mini_swe_deepseek.yaml data/pilot_a11_deepseek
[ "$ARMS" = deepseek ] || run_arm gpt5mini swebench.yaml configs/mini_swe_openai_modal.yaml data/pilot_a11_gpt5mini

AFTER=$(spend)
COST=$($V -c "print(f'{$AFTER - $BEFORE:.4f}')")
echo
echo "   measured OpenRouter spend across both arms: \$$COST (cap \$$OR_CAP)"
$V -c "import sys; sys.exit(0 if $COST <= $OR_CAP else 1)" || {
  echo "ABORT: OpenRouter spend \$$COST exceeded the \$$OR_CAP cap. Gold NOT run."; exit 3; }

# --- convert predictions, per arm --------------------------------------
OUTS=""
[ "$ARMS" = gpt5mini ] || OUTS="$OUTS data/pilot_a11_deepseek"
[ "$ARMS" = deepseek ] || OUTS="$OUTS data/pilot_a11_gpt5mini"
for out in $OUTS; do
  echo "== convert $out/preds.json -> predictions.jsonl =="
  $V - "$out" <<'CONV'
import json, sys
from pathlib import Path
out = Path(sys.argv[1]); src = out / "preds.json"
if not src.exists():
    print(f"ABORT: {src} not written by generation"); sys.exit(2)
d = json.loads(src.read_text())
rows = list(d.values()) if isinstance(d, dict) else d
bad = [r for r in rows if not {"instance_id", "model_patch"} <= set(r)]
if bad:
    print(f"ABORT: {len(bad)} prediction rows lack instance_id/model_patch"); sys.exit(2)
dst = out / "predictions.jsonl"
dst.write_text("".join(json.dumps(r) + "\n" for r in rows))
print(f"   wrote {dst} with {len(rows)} rows")
CONV
  [ -s "$out/predictions.jsonl" ] || { echo "ABORT: $out/predictions.jsonl empty or missing"; exit 2; }
done

# --- gold, per arm ------------------------------------------------------
# SWE-bench/SWE-bench_Verified, not princeton-nlp/: the latter lacks image,
# eval_script, log_parser and eval_type and KeyErrors on the first instance.
PAIRS=""
[ "$ARMS" = gpt5mini ] || PAIRS="data/pilot_a11_deepseek:a11_deepseek"
[ "$ARMS" = deepseek ] || PAIRS="$PAIRS data/pilot_a11_gpt5mini:a11_gpt5mini"
for pair in $PAIRS; do
  set -- ${pair%%:*} ${pair##*:}
  echo "== gold: $2 =="
  $V modal/gold_modal.py --dataset SWE-bench/SWE-bench_Verified \
    --preds "$1/predictions.jsonl" --run-id "$2" --oracle 3
done

echo
echo "== Amendment 11 generation and gold complete; run scripts/readout_amendment11.py =="
