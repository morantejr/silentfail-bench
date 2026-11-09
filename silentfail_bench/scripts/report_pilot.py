#!/usr/bin/env python3
"""Recompute pilot report + go/no-go + power from an existing scored JSONL."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from silentfail.pilot import go_no_go, load_jsonl, pilot_summary, required_n_for_power  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("jsonl", type=Path)
    p.add_argument("--mde", type=float, default=0.03)
    p.add_argument("--min-failures", type=int, default=None)
    args = p.parse_args()

    rows = load_jsonl(args.jsonl)
    summary = pilot_summary(rows)
    min_fail = args.min_failures or max(15, int(0.05 * summary.get("n_usable", 0)))
    decision = go_no_go(summary, min_failures=min_fail)
    rate = summary.get("failure_rate") or 0.3
    if rate != rate:
        rate = 0.3
    report = {
        "summary": summary,
        "go_no_go": decision,
        "power_80": required_n_for_power(rate, args.mde, 0.8),
        "power_90": required_n_for_power(rate, args.mde, 0.9),
    }
    print(json.dumps(report, indent=2, default=str))


if __name__ == "__main__":
    main()
