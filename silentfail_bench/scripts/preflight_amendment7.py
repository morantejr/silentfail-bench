#!/usr/bin/env python3
"""Preflight for resuming Amendment 7 on 1 October. Read-only, costs nothing.

Verifies, in order, the things that have actually gone wrong in this project:
  1. the paused checkpoint is where the RUNNER looks for it, not one level up
  2. its contents still match the hashes taken before Amendment 7a was written
  3. preds.json will make the agent SKIP the 100 done instances, not redo them
  4. no Modal app or sandbox is still running and quietly billing
  5. the swe-rex image patch is still in the venv (without it, generation stalls)
Exits non-zero if anything would waste money or silently discard work.
"""
from __future__ import annotations
import hashlib, json, subprocess, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data/agent_axis"
fail: list[str] = []

def check(ok: bool, msg: str, detail: str = "") -> None:
    print(f"  {'ok  ' if ok else 'FAIL'}  {msg}{(' -> ' + detail) if detail else ''}")
    if not ok: fail.append(msg)

print("Amendment 7 preflight\n")
print("checkpoint")
traj = OUT / "traj/cand0"
files = sorted(traj.glob("*/*.traj.json"))
check(traj.is_dir(), "checkpoint sits under traj/cand0, where the runner globs", str(traj))
check(len(files) == 100, "100 trajectories present", f"{len(files)} found")
runner_glob = list(OUT.glob("traj/cand*/*/*.traj.json"))
check(len(runner_glob) == len(files), "the runner's own glob finds them", f"{len(runner_glob)} via traj/cand*/*/*.traj.json")

print("\nintegrity")
hp = OUT / "CHECKPOINT_HASHES.json"
if not hp.exists():
    check(False, "CHECKPOINT_HASHES.json present")
else:
    rec = json.loads(hp.read_text())
    per = {}
    for p in files:
        per[str(p.relative_to(OUT))] = hashlib.sha256(p.read_bytes()).hexdigest()
    roll = hashlib.sha256("".join(f"{k}:{v}" for k, v in sorted(per.items())).encode()).hexdigest()
    check(roll == rec["rollup_sha256"], "contents match the pre-outcome hashes", roll[:16] + "...")
    pj = traj / "preds.json"
    check(pj.exists() and hashlib.sha256(pj.read_bytes()).hexdigest() == rec["preds_json_sha256"],
          "preds.json unchanged since the hashes were taken")

print("\nresume behaviour")
pj = traj / "preds.json"
if pj.exists():
    n = len(json.loads(pj.read_text()))
    check(n == 100, "preds.json lists 100 instances, so the agent skips them", f"{n} entries")
    ids = set(json.loads(pj.read_text()).keys())
    inst = (ROOT / "configs/replication_instances_multilingual.txt").read_text().split()
    check(ids <= set(inst), "every done instance is in the registered instance list")
    print(f"        remaining in cand0: {len(set(inst) - ids)} of {len(inst)}; then cand1-3 run fresh")
else:
    check(False, "preds.json present")

print("\nnothing still billing")
modal = ROOT / ".venv-swebench/bin/modal"
if modal.exists():
    r = subprocess.run([str(modal), "app", "list"], capture_output=True, text=True, timeout=120)
    rows = [l for l in r.stdout.splitlines() if l.strip().startswith("│")]
    check(not rows, "no Modal apps running", f"{len(rows)} row(s)")
else:
    check(False, "modal CLI present in the venv")

print("\nenvironment")
sw = ROOT / ".venv-swebench/lib/python3.12/site-packages/swerex/deployment/modal.py"
if sw.exists():
    t = sw.read_text()
    check("add_python" in t, "swe-rex image patch still applied (else generation stalls on apt)")
else:
    check(False, "swerex modal deployment present")
cfg = ROOT / "configs/mini_swe_gpt6luna.yaml"
check(cfg.exists() and "litellm_response" in cfg.read_text(),
      "agent config uses litellm_response (litellm forces tool calls gpt-6-luna rejects)")

print()
if fail:
    print(f"{len(fail)} PROBLEM(S) — do not start: " + "; ".join(fail)); sys.exit(1)
print("READY. Amendment 7 can resume; see RESUME_AMENDMENT7.md for the commands.")
