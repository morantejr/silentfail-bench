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
# Before 1 Oct this asserted the output directory was still fresh. The pilot has
# now run, so the meaningful property is the opposite: the readout exists and
# records the STOP decision, which is what keeps the axis from being launched.
a10_done=0
[ -f silentfail_bench/data/pilot_deepseek_a10/readout_amendment10.json ] && \
  silentfail_bench/.venv-swebench/bin/python -c "import json,sys; d=json.load(open('silentfail_bench/data/pilot_deepseek_a10/readout_amendment10.json')); sys.exit(0 if d['all_gates_pass'] is False and d['P10_3']['per_trajectory_usd'] > 0 else 1)" && a10_done=1
[ "$a10_done" = "1" ] && chk 1 "Amendment 10 pilot has run; readout records STOP with a non-zero measured cost" || chk 0 "Amendment 10 readout missing, or it records a passing gate set or a zero cost"
# Amendment 11 re-ran both arms at 100 steps. Its readout must exist, record every gate
# passing under BOTH classifications of the comparator's crashed rows, and carry a
# non-zero measured cost, or the papers' "runnable" claim has nothing behind it.
a11_done=0
[ -f silentfail_bench/data/readout_amendment11.json ] && \
  silentfail_bench/.venv-swebench/bin/python -c "import json,sys; d=json.load(open('silentfail_bench/data/readout_amendment11.json')); sys.exit(0 if d['all_gates_pass'] is True and d['P11_2']['passes_A'] and d['P11_2']['passes_B'] and d['P11_3']['measured_total_usd'] > 0 else 1)" && a11_done=1
[ "$a11_done" = "1" ] && chk 1 "Amendment 11 has run; readout records all gates passing under both classifications, non-zero cost" || chk 0 "Amendment 11 readout missing, or a gate fails, or the classifications disagree, or cost is zero"
[ -x silentfail_bench/scripts/run_amendment10_pilot.sh ] && chk 1 "executable pilot runner present" || chk 0 "executable pilot runner present"
[ -f silentfail_bench/scripts/readout_amendment10.py ] && chk 1 "Amendment 10 gate readout present" || chk 0 "Amendment 10 gate readout present"
grep -q "Amendment 10a" silentfail_bench/PREREGISTRATION_SWEBENCH_PILOT.md && chk 1 "Amendment 10a recorded (stop rule, ledger, cost enforcement)" || chk 0 "Amendment 10a recorded"
grep -q "raised, before outcome, from .22 to .26" silentfail_bench/PREREGISTRATION_SWEBENCH_PILOT.md && chk 1 "Modal ledger reconciled and cap raised to \$26 before outcome" || chk 0 "Modal ledger reconciled"
! grep -q "null \*\*replicates\*\*" CLAUDE_HANDOFF_OCT1.md && chk 1 "handoff does not overclaim the null as replicated" || chk 0 "handoff overclaims the null as replicated"
! grep -qE "^[^#]*--dataset[[:space:]]+princeton-nlp/" silentfail_bench/scripts/run_amendment10_pilot.sh && chk 1 "pilot gold does not use the princeton-nlp mirror (it lacks image/eval_script)" || chk 0 "pilot gold uses the princeton-nlp mirror, which KeyErrors in make_test_spec"
inputs_rc=0
silentfail_bench/.venv-swebench/bin/python silentfail_bench/scripts/check_inputs_tracked.py >/dev/null 2>&1 || inputs_rc=$?
[ "$inputs_rc" -eq 0 ] && chk 1 "every data input a committed script reads is itself committed" || chk 0 "a committed script reads an untracked data input (run scripts/check_inputs_tracked.py)"
grep -q "preds.json -> predictions.jsonl" silentfail_bench/scripts/run_amendment10_pilot.sh && chk 1 "runner converts preds.json to the JSONL gold actually reads" || chk 0 "runner converts preds.json for gold"
grep -q "ABORT: predictions.jsonl empty or missing" silentfail_bench/scripts/run_amendment10_pilot.sh && chk 1 "runner asserts the converted file exists before spending on gold" || chk 0 "runner asserts the converted file exists"
grep -q "credits endpoint" silentfail_bench/scripts/run_amendment10_pilot.sh && chk 1 "cost cap enforced externally, not by the scaffold" || chk 0 "cost cap enforced externally"
grep -q "do not revert to" silentfail_bench/scripts/readout_amendment10.py && chk 1 "gate readout encodes the Amendment 10a stop rule" || chk 0 "gate readout encodes the stop rule"
a10_rc=0
# Probe fail-closed behaviour against an empty scratch directory and a run-id with
# no gold. Pointing this at the real pilot directory overwrote the committed
# readout with $0.0000 once the run existed, and P10-3 then "passed" on it.
a10_probe=$(mktemp -d)
silentfail_bench/.venv-swebench/bin/python silentfail_bench/scripts/readout_amendment10.py --out "$a10_probe" --run-id a10_probe_no_such_run --measured-cost 0.01 --instances silentfail_bench/configs/pilot_instances_swebench41_complete.txt >/dev/null 2>&1 || a10_rc=$?
rm -rf "$a10_probe"
[ "$a10_rc" -eq 2 ] && chk 1 "gate readout fails closed with no pilot data (exit 2)" || chk 0 "gate readout fails closed"
regen_rc=0
silentfail_bench/.venv-swebench/bin/python silentfail_bench/scripts/check_readouts_regenerate.py >/dev/null 2>&1 || regen_rc=$?
[ "$regen_rc" -eq 0 ] && chk 1 "every committed readout matches what its own script produces" || chk 0 "a committed readout no longer regenerates (run scripts/check_readouts_regenerate.py)"

if [ "$a10_fail" = "1" ]; then echo; echo "AMENDMENT 10 PREFLIGHT FAILED"; exit 1; fi

echo "PREFLIGHT OK — Amendment 10 failed at 40 steps; Amendment 11 passes every gate at 100 steps (post hoc); the agent axis is RUNNABLE under its own amendment and has NOT been launched"
