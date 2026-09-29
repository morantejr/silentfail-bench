#!/usr/bin/env bash
# Amendment 10 capability pilot for DeepSeek V4.1 Flash. Run on/after 2026-10-01.
#
# Refuses to improvise: the instance set, config and output directory are pinned,
# it will not write into an existing directory, and it aborts before launch if the
# projected cost exceeds the declared cap. Cost is measured externally from the
# OpenRouter credits endpoint, because cost_tracking is disabled in the config and
# the scaffold's own cost_limit is therefore inert (Amendment 10a §3).
set -euo pipefail
cd "$(dirname "$0")/.."

V=.venv-swebench/bin/python
MINI=.venv-swebench/bin/mini-extra
INSTANCES=configs/pilot_instances_swebench41_complete.txt
CONFIG=configs/mini_swe_deepseek.yaml
OUT=data/pilot_deepseek_a10
RUN_ID=a10_deepseek_pilot
CAP_USD=5.00                      # Amendment 10, per model
N=$(grep -c . "$INSTANCES")

echo "== Amendment 10 pilot: DeepSeek V4.1 Flash =="
echo "   instances $N   config $CONFIG   out $OUT   cap \$$CAP_USD"

# --- guards -------------------------------------------------------------
[ "$N" -eq 41 ] || { echo "ABORT: instance file has $N entries, expected 41"; exit 2; }
[ -e "$OUT" ] && { echo "ABORT: $OUT already exists. Move it or pick a new directory; refusing to overwrite."; exit 2; }
grep -q "max_tokens: 16000" "$CONFIG" || { echo "ABORT: config is not on the corrected 16k output cap"; exit 2; }
grep -q "openrouter_textbased" "$CONFIG" || { echo "ABORT: config is not using the text-based model class"; exit 2; }
if [ "$(date +%Y-%m-%d)" \< "2026-10-01" ]; then
  echo "ABORT: today is $(date +%Y-%m-%d); Modal credit resets 2026-10-01. Re-run then, or pass --i-accept-paid-overage."
  [ "${1:-}" = "--i-accept-paid-overage" ] || exit 2
fi

spend() { $V - <<'PY'
import os, requests
from dotenv import load_dotenv
load_dotenv(os.path.expanduser("~/Library/Application Support/mini-swe-agent/.env"), override=True)
c = requests.get("https://openrouter.ai/api/v1/credits",
                 headers={"Authorization": "Bearer " + os.environ["OPENROUTER_API_KEY"]},
                 timeout=30).json().get("data", {})
print(f"{c.get('total_usage', 0):.6f}")
PY
}

BEFORE=$(spend); echo "   OpenRouter usage before: \$$BEFORE"

# --- 1. generation ------------------------------------------------------
FILTER="^($(paste -sd'|' "$INSTANCES"))$"
echo "== generation =="
$MINI swebench --subset verified --split test --filter "$FILTER" \
  -o "$OUT" -w 6 -c swebench_backticks.yaml -c "$CONFIG" -c agent.step_limit=40

AFTER=$(spend)
COST=$($V -c "print(f'{$AFTER - $BEFORE:.4f}')")
PER=$($V -c "print(f'{($AFTER - $BEFORE)/$N:.4f}')")
echo "   measured OpenRouter spend: \$$COST  (\$$PER per trajectory)"
$V -c "import sys; sys.exit(0 if $COST <= $CAP_USD else 1)" || {
  echo "ABORT: spend \$$COST exceeded the \$$CAP_USD cap. Gold NOT run."; exit 3; }

# --- 2. convert predictions -------------------------------------------
# mini-swe-agent writes preds.json as a DICT keyed by instance_id; gold_modal.py
# reads --preds as JSONL, one object per line. Passing the dict, or a
# predictions.jsonl that generation never wrote, fails at gold after the agent
# spend is already sunk. Convert explicitly and assert the result.
echo "== convert preds.json -> predictions.jsonl =="
$V - "$OUT" <<'CONV'
import json, sys
from pathlib import Path
out = Path(sys.argv[1])
src = out / "preds.json"
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
[ -s "$OUT/predictions.jsonl" ] || { echo "ABORT: predictions.jsonl empty or missing"; exit 2; }

# --- 3. gold ------------------------------------------------------------
echo "== gold =="
$V modal/gold_modal.py --dataset princeton-nlp/SWE-bench_Verified \
  --preds "$OUT/predictions.jsonl" --run-id "$RUN_ID" --oracle 3

# --- 4. gates -----------------------------------------------------------
echo "== Amendment 10 gates =="
$V scripts/readout_amendment10.py --out "$OUT" --run-id "$RUN_ID" \
  --measured-cost "$COST" --instances "$INSTANCES"
