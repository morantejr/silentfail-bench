#!/usr/bin/env python3
"""Run Stage-0 pilot (mock by default; set SILENTFAIL_BACKEND=vllm|openai for live)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from silentfail.pilot import go_no_go, pilot_summary, required_n_for_power  # noqa: E402
from silentfail.runner import run_pilot_mock  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--anchor", type=Path, default=ROOT / "configs" / "anchor.yaml")
    p.add_argument("--out", type=Path, default=ROOT / "data" / "pilot" / "pilot.jsonl")
    p.add_argument("--n", type=int, default=None, help="Override trajectory count (default 200)")
    p.add_argument("--mde", type=float, default=0.03, help="Planning MDE for power grid")
    args = p.parse_args()

    scored = run_pilot_mock(args.anchor, args.out, n=args.n)
    rows = []
    with scored.open() as f:
        for line in f:
            rows.append(json.loads(line))

    summary = pilot_summary(rows)
    # Provisional min failures: 5% of usable or 15, whichever larger — replaced by power script after real pilot
    min_fail = max(15, int(0.05 * summary.get("n_usable", 0)))
    decision = go_no_go(summary, min_failures=min_fail)

    rate = summary.get("failure_rate") or 0.3
    power = required_n_for_power(rate if rate == rate else 0.3, args.mde, target_power=0.8)
    power90 = required_n_for_power(rate if rate == rate else 0.3, args.mde, target_power=0.9)

    report = {
        "scored_log": str(scored),
        "summary": summary,
        "go_no_go": decision,
        "power_80": power,
        "power_90": power90,
        "mde": args.mde,
    }
    report_path = scored.with_name("pilot_report.json")
    report_path.write_text(json.dumps(report, indent=2, default=str))
    print(json.dumps(report, indent=2, default=str))
    print(f"\nWrote {report_path}")


if __name__ == "__main__":
    main()
