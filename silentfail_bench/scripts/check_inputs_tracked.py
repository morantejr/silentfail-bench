#!/usr/bin/env python3
"""Every data file a committed script reads must itself be committed.

The blanket `data/*/*` rule in .gitignore has now silently excluded load-bearing
inputs five separate times, each found by a human or an audit rather than by a
gate. This checks the property directly: scan the scripts that must run, extract
the data paths they reference literally, and assert each is tracked in git.

Exit 1 and name the offenders if any script reads something a fresh clone would
not have.
"""
from __future__ import annotations
import re, subprocess, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ["scripts/readout_amendment5_corrected.py", "scripts/readout_sp1b.py",
           "scripts/readout_amendment7.py", "scripts/readout_amendment10.py",
           "scripts/preflight_amendment7.py", "scripts/readout_failure_taxonomy.py",
           "scripts/readout_luckypass.py", "scripts/readout_pi_batch1.py",
           "scripts/readout_pi_batch2.py", "scripts/readout_pivotal.py",
           "scripts/readout_static_analysis.py", "scripts/verify_and_robustify.py",
           "scripts/readout_tier15.py", "paper/check_consistency.py"]
PAT = re.compile(r'["\']((?:data|silentfail_bench/data)/[A-Za-z0-9_./-]+\.(?:jsonl|json|txt|md))["\']')

tracked = set(subprocess.run(["git", "ls-files"], cwd=ROOT.parent, capture_output=True, text=True).stdout.split())
missing, checked = [], 0
for s in SCRIPTS:
    f = ROOT.parent / s if s.startswith("paper/") else ROOT / s
    if not f.exists():
        continue
    for m in set(PAT.findall(f.read_text())):
        if "*" in m:
            continue
        rel = m if m.startswith("silentfail_bench/") else f"silentfail_bench/{m}"
        disk = ROOT.parent / rel
        if not disk.exists():
            continue                      # produced by the run, not an input
        checked += 1
        if rel not in tracked:
            missing.append((s, rel))

for s, rel in sorted(set(missing)):
    print(f"  UNTRACKED INPUT  {rel}\n                   read by {s}")
print(f"  {checked} literal data inputs checked across {len(SCRIPTS)} scripts, {len(set(missing))} untracked")
sys.exit(1 if missing else 0)
