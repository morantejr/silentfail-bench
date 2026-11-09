#!/usr/bin/env python3
"""Drop error trajectories (rate limits, dead sandboxes) and prune preds.json so the batch runner re-attempts them."""
import json, glob, os, shutil
from pathlib import Path
KEEP = {"Submitted", "LimitsExceeded"}
n = 0
for p in glob.glob("data/replication_multilingual/traj/cand*/*/*.traj.json"):
    try: st = json.load(open(p))["info"].get("exit_status")
    except Exception: st = None
    if st not in KEEP: shutil.rmtree(os.path.dirname(p)); n += 1
for c in range(4):
    d = Path(f"data/replication_multilingual/traj/cand{c}"); p = d / "preds.json"
    if p.exists():
        preds = json.loads(p.read_text()); keep = {k: v for k, v in preds.items() if (d / k / f"{k}.traj.json").exists()}
        p.write_text(json.dumps(keep, indent=1))
have = len(glob.glob("data/replication_multilingual/traj/cand*/*/*.traj.json"))
print(f"removed {n} error trajectories; usable trajectories {have}/456; missing {456 - have}")
