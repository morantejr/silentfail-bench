#!/usr/bin/env python3
"""Terminate swe-rex sandboxes not referenced by the currently running batch logs (each killed batch leaves its sandboxes alive)."""
import glob, re, subprocess, sys
import modal
PY = sys.executable
live = set()
for f in glob.glob("data/replication_multilingual/traj/cand*/gen.log"):
    txt = open(f, errors="ignore").read().rsplit("Running on ", 1)[-1]
    live |= set(re.findall(r"sb-[A-Za-z0-9]+", txt))
if "--all" in sys.argv: live = set()
apps = subprocess.run([PY, "-m", "modal", "app", "list"], capture_output=True, text=True).stdout
ids = re.findall(r"(ap-[A-Za-z0-9]+)\s*│\s*swe-rex", apps)
killed = kept = 0
for app_id in ids:
    for sb in modal.Sandbox.list(app_id=app_id):
        if sb.object_id in live: kept += 1; continue
        try: sb.terminate(); killed += 1
        except Exception as e: print("terminate failed", sb.object_id, str(e)[:80])
print(f"swe-rex apps {ids}: terminated {killed}, kept {kept} live")
