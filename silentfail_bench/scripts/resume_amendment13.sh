#!/usr/bin/env bash
# Resume Amendment 13 generation after an interruption (wifi drop, sleep, kill).
#
#   bash scripts/resume_amendment13.sh
#
# Safe to run any number of times. It refuses to start if a generator is already running,
# stops stray Modal apps (DEFECTS #21), removes the empty preds.json placeholders that
# mini-swe-agent writes for instances that never ran (DEFECTS #21/#22) -- with a backup and
# a RESUME_NOTE entry per candidate -- and resumes with every finished trajectory skipped.
# Finished trajectories are never touched. Cost gate stays anchored at the first start.
set -euo pipefail
cd "$(dirname "$0")/.."
V=.venv-swebench/bin/python
MODAL=.venv-swebench/bin/modal
O=data/agent_axis_deepseek

pgrep -f run_replication_modal >/dev/null && { echo "ABORT: a generator is already running (pgrep -f run_replication_modal)"; exit 2; }

echo "== stopping stray Modal apps =="
for a in $($MODAL app list 2>/dev/null | grep -E "deployed|running" | grep -oE "ap-[A-Za-z0-9]+"); do
  $MODAL app stop --yes "$a" >/dev/null 2>&1 && echo "   stopped $a"; done
for i in $(seq 1 18); do
  live=$($MODAL app list 2>/dev/null | grep -cE "deployed|running|stopping" || true)   # grep -c exits 1 on zero under pipefail
  [ "$live" = 0 ] && break
  echo "   waiting for $live app(s) to finish stopping..."; sleep 10
done
echo "   swe-rex containers live: $($MODAL container list 2>/dev/null | grep -c swe-rex)"

echo "== removing placeholders (empty patch, no trajectory) =="
$V - "$O" <<'PY'
import json, glob, shutil, sys, datetime
from pathlib import Path
O = Path(sys.argv[1]); stamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
ef_path = O / "ENDPOINT_FAILURES.json"
EF = {k: set(v) for k, v in json.loads(ef_path.read_text()).items() if k.startswith("cand")} if ef_path.exists() else {}
for c in range(4):
    p = O / f"traj/cand{c}/preds.json"
    if not p.exists(): print(f"   cand{c}: no preds.json yet"); continue
    d = json.loads(p.read_text()); traj = {q.split("/")[-2] for q in glob.glob(f"{O}/traj/cand{c}/*/*.traj.json")}
    ph = sorted(i for i, r in d.items() if not (r.get("model_patch") or "").strip() and i not in traj and i not in EF.get(f"cand{c}", set()))
    if ph:
        shutil.copy(p, p.with_name(f"preds.json.{stamp.replace(' ', '_').replace(':', '')}_with_{len(ph)}_empty_entries"))
        for i in ph: del d[i]
        p.write_text(json.dumps(d, indent=2))
        with open(O / f"traj/cand{c}/RESUME_NOTE.md", "a") as f:
            f.write(f"\n## Resume {stamp}\n\n{len(ph)} placeholder entries (empty patch, no trajectory) removed so they regenerate:\n"
                    + "\n".join(ph) + "\n")
    print(f"   cand{c}: {len(traj)} trajectories on disk, {len(ph)} placeholders removed, {len(d)} entries kept")
PY

echo "== preflight (resume mode) =="
A13_RESUME=1 $V scripts/preflight_amendment13.py 2>&1 | grep -E "FAIL|PREFLIGHT" || true
A13_RESUME=1 $V scripts/preflight_amendment13.py >/dev/null 2>&1 || { echo "ABORT: preflight failed; run it to see why"; exit 2; }

echo "== resuming generation (finished trajectories skipped) =="
A13_RESUME=1 exec bash scripts/run_amendment13_axis.sh generate
