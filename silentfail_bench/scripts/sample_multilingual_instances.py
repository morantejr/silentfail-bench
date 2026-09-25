#!/usr/bin/env python3
"""Amendment 4e sample: up to N instances per repository from SWE-bench Multilingual, fixed seed."""
from __future__ import annotations
import argparse, collections, random
from pathlib import Path
from datasets import load_dataset

ap = argparse.ArgumentParser(); ap.add_argument("--per-repo", type=int, default=3); ap.add_argument("--seed", type=int, default=20260922)
ap.add_argument("--out", type=Path, default=Path(__file__).resolve().parents[1] / "configs/replication_instances_multilingual.txt"); a = ap.parse_args()
ds = load_dataset("swe-bench/SWE-Bench_Multilingual", split="test")
by = collections.defaultdict(list)
for r in ds: by[r["repo"]].append(r["instance_id"])
rng = random.Random(a.seed); chosen = []
for repo in sorted(by):
    ids = sorted(by[repo]); rng.shuffle(ids); chosen += ids[: a.per_repo]
a.out.write_text(f"# SWE-bench Multilingual, up to {a.per_repo} per repo, seed {a.seed}, {len(by)} repos, {len(chosen)} instances\n" + "\n".join(chosen) + "\n")
print(len(by), "repos;", len(chosen), "instances ->", a.out)
