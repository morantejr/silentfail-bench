#!/usr/bin/env bash
# Amendment 12: GLM-5.3 under DeepSeek's qualifying settings (step_limit=100, max_tokens=16000).
# Smoke one instance, project cost x41, abort if over cap; then the full 41; external
# OpenRouter cost measurement; convert; gold. Comparator is Amendment 11's gpt-5-mini@100.
set -euo pipefail
cd "$(dirname "$0")/.."
LIMIT=100; MAXTOK=16000
V=.venv-swebench/bin/python
MINI=.venv-swebench/bin/mini-extra
INSTANCES=configs/pilot_instances_swebench41_complete.txt
CONFIG=configs/mini_swe_glm.yaml
OUT=data/pilot_a12_glm
RUN_ID=a12_glm
OR_CAP=20.00
N=$(grep -c . "$INSTANCES")

echo "== Amendment 12: GLM-5.3 at step_limit=$LIMIT, max_tokens=$MAXTOK, $N instances =="
[ "$N" -eq 41 ] || { echo "ABORT: $N instances, expected 41"; exit 2; }
grep -q "^# Amendment 12, 1 October 2026" PREREGISTRATION_SWEBENCH_PILOT.md || { echo "ABORT: Amendment 12 not recorded"; exit 2; }
grep -q "max_tokens = 16000" PREREGISTRATION_SWEBENCH_PILOT.md || { echo "ABORT: the registration does not declare max_tokens = 16000"; exit 2; }
[ -e "$OUT" ] && { echo "ABORT: $OUT exists; refusing to overwrite"; exit 2; }
grep -q "openrouter_textbased" "$CONFIG" || { echo "ABORT: GLM config is not the text-based class"; exit 2; }

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
BEFORE=$(spend) || { echo "ABORT: cannot read OpenRouter usage; the cost gate would be blind"; exit 4; }
$V -c "import sys; sys.exit(0 if $BEFORE > 0 else 1)" || { echo "ABORT: usage reads 0.000000"; exit 4; }
echo "   OpenRouter usage before: \$$BEFORE"

# --- smoke: one instance, project the cost ------------------------------
SMOKE=$(head -1 "$INSTANCES")
echo "== smoke: $SMOKE =="
rm -rf data/smoke_a12_glm
$MINI swebench --subset verified --split test --filter "^${SMOKE}$" \
  -o data/smoke_a12_glm -w 1 -c swebench_backticks.yaml -c "$CONFIG" \
  -c "agent.step_limit=$LIMIT" -c "model.model_kwargs.max_tokens=$MAXTOK" >/dev/null 2>&1 || true
$V - "$SMOKE" "$N" "$OR_CAP" <<'INNERPY'
import json, sys
from pathlib import Path
inst, n, cap = sys.argv[1], int(sys.argv[2]), float(sys.argv[3])
f = Path(f"data/smoke_a12_glm/{inst}/{inst}.traj.json")
if not f.exists():
    print("ABORT: smoke wrote no trajectory"); sys.exit(2)
info = json.loads(f.read_text()).get("info", {})
st = info.get("model_stats") or {}
c = float(st.get("instance_cost") or 0.0)
print(f"   smoke exit={info.get('exit_status')} api_calls={st.get('api_calls')} cost=${c:.4f} -> projected x{n} = ${c*n:.2f} (cap ${cap:.2f})")
if info.get("exit_status") in ("AuthenticationError",):
    print("ABORT: authentication failed on the smoke"); sys.exit(2)
if c * n > cap:
    print(f"ABORT: projected ${c*n:.2f} exceeds the ${cap:.2f} cap; full run NOT started"); sys.exit(3)
INNERPY
rm -rf data/smoke_a12_glm

# --- generation -----------------------------------------------------------
FILTER="^($(paste -sd'|' "$INSTANCES"))$"
echo "== generation =="
$MINI swebench --subset verified --split test --filter "$FILTER" \
  -o "$OUT" -w 6 -c swebench_backticks.yaml -c "$CONFIG" \
  -c "agent.step_limit=$LIMIT" -c "model.model_kwargs.max_tokens=$MAXTOK"
AFTER=$(spend)
COST=$($V -c "print(f'{$AFTER - $BEFORE:.4f}')")
echo "   measured OpenRouter spend: \$$COST (\$$($V -c "print(f'{($AFTER-$BEFORE)/$N:.4f}')") per trajectory; cap \$$OR_CAP)"
$V -c "import sys; sys.exit(0 if $COST <= $OR_CAP else 1)" || { echo "ABORT: spend exceeded cap. Gold NOT run."; exit 3; }

# --- convert ---------------------------------------------------------------
echo "== convert preds.json -> predictions.jsonl =="
$V - "$OUT" <<'INNERPY'
import json, sys
from pathlib import Path
out = Path(sys.argv[1]); src = out / "preds.json"
if not src.exists(): print(f"ABORT: {src} missing"); sys.exit(2)
d = json.loads(src.read_text()); rows = list(d.values()) if isinstance(d, dict) else d
bad = [r for r in rows if not {"instance_id", "model_patch"} <= set(r)]
if bad: print(f"ABORT: {len(bad)} rows lack fields"); sys.exit(2)
(out / "predictions.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
print(f"   wrote predictions.jsonl with {len(rows)} rows")
INNERPY
[ -s "$OUT/predictions.jsonl" ] || { echo "ABORT: predictions.jsonl empty"; exit 2; }

# --- gold ------------------------------------------------------------------
echo "== gold: $RUN_ID =="
$V modal/gold_modal.py --dataset SWE-bench/SWE-bench_Verified --preds "$OUT/predictions.jsonl" --run-id "$RUN_ID" --oracle 3
echo "== done; run scripts/readout_amendment12.py --measured-cost $COST =="
