#!/usr/bin/env bash
# No-spend verification before Amendment 7 Modal work (run any time before Oct 1).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
SB="$ROOT/silentfail_bench"
V="$SB/.venv-swebench/bin/python"

echo "== silentfail_bench unit tests (system python) =="
cd "$SB"
python3 -m pytest -q

echo "== Amendment 7 preflight =="
"$V" scripts/preflight_amendment7.py

echo "== Amendment 7 readout dry-run + production fixture =="
"$V" scripts/readout_amendment7.py --dry-run
"$V" scripts/test_readout_amendment7.py

echo "== paper / JSON cross-check (slow) =="
cd "$ROOT"
python3 paper/check_consistency.py

echo ""

echo
echo "audit: Amendment 10 open-model arm (config, pilot set, ledger, enforcement):"
a10_fail=0
chk() { if [ "$1" = "1" ]; then echo "  ok    $2"; else echo "  FAIL  $2"; a10_fail=1; fi; }

grep -q "max_tokens: 16000" silentfail_bench/configs/mini_swe_deepseek.yaml && chk 1 "DeepSeek config on the corrected 16k output cap" || chk 0 "DeepSeek config on the corrected 16k output cap"
grep -q "openrouter_textbased" silentfail_bench/configs/mini_swe_deepseek.yaml && chk 1 "DeepSeek config uses the text-based model class" || chk 0 "DeepSeek config uses the text-based model class"
grep -q "allow_fallbacks: false" silentfail_bench/configs/mini_swe_deepseek.yaml && chk 1 "provider pinned with fallbacks disabled" || chk 0 "provider pinned with fallbacks disabled"
[ "$(grep -c . silentfail_bench/configs/pilot_instances_swebench41_complete.txt)" = "41" ] && chk 1 "pilot set is the Amendment 6 41 instances" || chk 0 "pilot set is the Amendment 6 41 instances"
[ ! -e silentfail_bench/data/pilot_deepseek_a10 ] && chk 1 "Amendment 10 output directory is fresh (no overwrite risk)" || chk 0 "Amendment 10 output directory already exists"
[ -x silentfail_bench/scripts/run_amendment10_pilot.sh ] && chk 1 "executable pilot runner present" || chk 0 "executable pilot runner present"
[ -f silentfail_bench/scripts/readout_amendment10.py ] && chk 1 "Amendment 10 gate readout present" || chk 0 "Amendment 10 gate readout present"
grep -q "Amendment 10a" silentfail_bench/PREREGISTRATION_SWEBENCH_PILOT.md && chk 1 "Amendment 10a recorded (stop rule, ledger, cost enforcement)" || chk 0 "Amendment 10a recorded"
grep -q "raised, before outcome, from .22 to .26" silentfail_bench/PREREGISTRATION_SWEBENCH_PILOT.md && chk 1 "Modal ledger reconciled and cap raised to \$26 before outcome" || chk 0 "Modal ledger reconciled"
! grep -q "null \*\*replicates\*\*" CLAUDE_HANDOFF_OCT1.md && chk 1 "handoff does not overclaim the null as replicated" || chk 0 "handoff overclaims the null as replicated"
grep -q "preds.json -> predictions.jsonl" silentfail_bench/scripts/run_amendment10_pilot.sh && chk 1 "runner converts preds.json to the JSONL gold actually reads" || chk 0 "runner converts preds.json for gold"
grep -q "ABORT: predictions.jsonl empty or missing" silentfail_bench/scripts/run_amendment10_pilot.sh && chk 1 "runner asserts the converted file exists before spending on gold" || chk 0 "runner asserts the converted file exists"
grep -q "credits endpoint" silentfail_bench/scripts/run_amendment10_pilot.sh && chk 1 "cost cap enforced externally, not by the scaffold" || chk 0 "cost cap enforced externally"
grep -q "do not revert to" silentfail_bench/scripts/readout_amendment10.py && chk 1 "gate readout encodes the Amendment 10a stop rule" || chk 0 "gate readout encodes the stop rule"
a10_rc=0
silentfail_bench/.venv-swebench/bin/python silentfail_bench/scripts/readout_amendment10.py --out silentfail_bench/data/pilot_deepseek_a10 --run-id a10_deepseek_pilot --measured-cost 0 --instances silentfail_bench/configs/pilot_instances_swebench41_complete.txt >/dev/null 2>&1 || a10_rc=$?
[ "$a10_rc" -eq 2 ] && chk 1 "gate readout fails closed with no pilot data (exit 2)" || chk 0 "gate readout fails closed"
if [ "$a10_fail" = "1" ]; then echo; echo "AMENDMENT 10 PREFLIGHT FAILED"; exit 1; fi

echo "PRE-OCT1 OK — safe to wait for Modal credit; see CLAUDE_HANDOFF_OCT1.md"
