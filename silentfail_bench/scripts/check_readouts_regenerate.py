#!/usr/bin/env python3
"""Every committed readout must be reproducible by its own script.

This exists because one was not. `readout_tier15.py` hardcoded the widest sweep
upper bound as a literal. When the paper superseded that bound, the committed
JSON was corrected by hand and the script was not, so re-running the script
reverted the artifact to a retracted value. A repository that cannot regenerate
its own numbers cannot be audited, and a fix applied to an artifact rather than
to its producer is not a fix.

Each script below writes its own output path and needs no arguments, no network
and no GPU. The check snapshots the target bytes, runs the script, compares, and
restores the snapshot either way, so a failure never leaves a modified readout
behind.

Usage:  silentfail_bench/.venv-swebench/bin/python \\
            silentfail_bench/scripts/check_readouts_regenerate.py [--only NAME]
"""
from __future__ import annotations
import argparse, json, subprocess, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# script stem -> the readout it writes, relative to silentfail_bench/
TARGETS = {
    "readout_failure_taxonomy":     "data/readout_failure_taxonomy.json",
    "readout_luckypass":            "data/readout_luckypass.json",
    "readout_sp1b":                 "data/readout_sp1b.json",
    "readout_pi_batch2":            "data/readout_pi_batch2.json",
    "readout_tier15":               "data/readout_tier15.json",
    "readout_amendment5_corrected": "data/selector_axis/readout_amendment5_corrected.json",
    "readout_amendment13_panel": "data/agent_axis_deepseek/readout_amendment13_panel.json",
    "readout_pi_batch1": "data/readout_pi_batch1.json",
    "readout_critic_demeaning": "data/readout_critic_demeaning.json",
}


def norm(b: bytes):
    """Compare parsed JSON, so key order and trailing-newline churn are not failures."""
    try:
        return json.loads(b.decode())
    except Exception:
        return b


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="", help="substring filter on the script stem")
    a = ap.parse_args()
    todo = {k: v for k, v in TARGETS.items() if a.only in k}
    if not todo:
        print(f"no readout matches --only {a.only!r}; known: {', '.join(sorted(TARGETS))}")
        return 2

    bad = []
    for stem, rel in sorted(todo.items()):
        out = ROOT / rel
        script = ROOT / "scripts" / f"{stem}.py"
        if not script.exists():
            print(f"  MISSING SCRIPT  {stem}.py"); bad.append(stem); continue
        before = out.read_bytes() if out.exists() else None
        try:
            r = subprocess.run([sys.executable, str(script)], capture_output=True, text=True, cwd=ROOT)
            after = out.read_bytes() if out.exists() else None
            if r.returncode != 0:
                print(f"  FAILED TO RUN   {stem}: exit {r.returncode}")
                print("     " + (r.stderr or r.stdout).strip().splitlines()[-1][:160])
                bad.append(stem)
            elif after is None:
                print(f"  WROTE NOTHING   {stem}: expected {rel}")
                bad.append(stem)
            elif norm(before) != norm(after):
                print(f"  DRIFTED         {stem}: {rel} differs from what its script produces")
                bad.append(stem)
            else:
                print(f"  ok              {stem} -> {rel}")
        finally:
            if before is not None:
                out.write_bytes(before)

    print()
    if bad:
        print(f"{len(bad)} readout(s) not reproducible: {', '.join(bad)}")
        print("Fix the script, not the committed JSON.")
        return 1
    print(f"all {len(todo)} readouts regenerate identically")
    return 0


if __name__ == "__main__":
    sys.exit(main())
