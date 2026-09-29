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
echo "PRE-OCT1 OK — safe to wait for Modal credit; see CLAUDE_HANDOFF_OCT1.md"
