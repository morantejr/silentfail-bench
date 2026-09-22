#!/usr/bin/env python3
"""Rebuild data/humaneval/ from the HF datasets-server (no `datasets` dep, no auth).

Produces the merged task file used by profile A, with the visible/held-out test
separation, plus the gold-oracle control list.

  visible_test : base HumanEval asserts   (agent may run these)
  gold_test    : HumanEval+ extended tests (defines g; never shown)
"""

from __future__ import annotations

import json
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "humaneval"
sys.path.insert(0, str(ROOT / "src"))


def fetch(dataset: str, config: str, split: str = "test") -> list[dict]:
    rows, off = [], 0
    while True:
        q = urllib.parse.urlencode(
            {"dataset": dataset, "config": config, "split": split, "offset": off, "length": 100}
        )
        with urllib.request.urlopen(
            f"https://datasets-server.huggingface.co/rows?{q}", timeout=60
        ) as r:
            d = json.load(r)
        rows += [x["row"] for x in d["rows"]]
        if len(rows) >= d["num_rows_total"]:
            return rows
        off = len(rows)
        time.sleep(0.3)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    plus = {r["task_id"]: r for r in fetch("evalplus/humanevalplus", "default")}
    base = {r["task_id"]: r for r in fetch("openai/openai_humaneval", "openai_humaneval")}
    merged = [
        {
            "task_id": tid,
            "entry_point": base[tid]["entry_point"],
            "prompt": base[tid]["prompt"],
            "visible_test": base[tid]["test"],
            "gold_test": plus[tid]["test"],
            # EvalPlus's OWN canonical solution: base HumanEval canonicals are
            # known-wrong on some instances and fail the stricter plus tests.
            "reference_solution": plus[tid]["prompt"] + plus[tid]["canonical_solution"],
        }
        for tid in sorted(base, key=lambda t: int(t.split("/")[1]))
        if tid in plus
    ]
    (OUT / "humaneval_plus_merged.json").write_text(json.dumps(merged))
    print(f"wrote {len(merged)} tasks")

    # Oracle control: if the known-correct answer fails gold, the oracle is
    # broken and that instance must be excluded, not counted as an agent failure.
    from silentfail.local_agent import grade_gold
    from silentfail.tasks_local import load_tasks

    bad = [t.task_id for t in load_tasks() if not grade_gold(t, t.reference_solution, timeout=30)[0]]
    (OUT / "gold_control_failures.json").write_text(json.dumps(bad))
    print(f"gold control: {len(load_tasks()) - len(bad)}/{len(load_tasks())} references pass")
    print(f"excluded as gold_harness_invalid: {bad}")


if __name__ == "__main__":
    main()
