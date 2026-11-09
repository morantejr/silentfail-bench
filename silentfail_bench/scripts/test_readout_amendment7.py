#!/usr/bin/env python3
"""No-network production-path test for the pre-outcome Amendment 7 readout.

Uses the committed 114x4 replication bundle, panel scores and gpt-5 cache, but
replaces candidate outcomes with deterministic synthetic labels. The labels
have no scientific meaning. The test exists only to exercise all production
joins, filters, gates, endpoint estimators, JSON writing and fail-closed checks
before agent-axis gold exists.
"""
from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PY = sys.executable


def run(*args: str, expect: int = 0) -> subprocess.CompletedProcess:
    result = subprocess.run([PY, *args], cwd=ROOT, capture_output=True, text=True)
    if result.returncode != expect:
        print(result.stdout)
        print(result.stderr, file=sys.stderr)
        raise AssertionError(f"{args} exited {result.returncode}, expected {expect}")
    return result


def main() -> None:
    from readout_amendment7 import pick_4b
    assert pick_4b([(2.0, c) for c in range(4)]) == 0
    assert pick_4b([(1.0, 0), (2.0, 1), (2.0, 2), (0.0, 3)]) == 1
    bundle_path = ROOT / "data/replication_multilingual/rescoring_bundle.jsonl"
    rows = [json.loads(line) for line in bundle_path.read_text().splitlines() if line.strip()]
    with tempfile.TemporaryDirectory(prefix="a7_readout_test_") as td:
        tmp = Path(td)
        inputs = tmp / "inputs"
        run(
            "scripts/prepare_amendment7.py",
            "--bundle", str(bundle_path),
            "--instances", "configs/agent_axis_instances_multilingual.txt",
            "--out", str(inputs),
        )
        gold_root = tmp / "gold"
        for candidate in range(4):
            out = gold_root / f"a7_agent_axis_cand{candidate}"
            out.mkdir(parents=True)
            gold = []
            for row in rows:
                if row["candidate"] != candidate or not (row.get("final_solution") or "").strip():
                    continue
                byte = hashlib.sha256(
                    f"a7-production-test:{row['instance_id']}:{candidate}".encode()
                ).digest()[0]
                gold.append({
                    "instance_id": row["instance_id"],
                    "candidate": candidate,
                    "g": int(byte % 3 != 0),
                })
            (out / "gold_extra.jsonl").write_text(
                "".join(json.dumps(record) + "\n" for record in gold)
            )

        output = tmp / "readout.json"
        run(
            "scripts/readout_amendment7.py",
            "--bundle", str(bundle_path),
            "--scores-dir", "data/replication_multilingual/modal/prefix4k",
            "--judge-cache", "data/selector_axis/judge_multilingual.jsonl",
            "--gold-root", str(gold_root),
            "--random-picks", str(inputs / "random_picks.jsonl"),
            "--out", str(output),
        )
        report = json.loads(output.read_text())
        assert report["registered_instances"] == 114
        assert report["candidate_slots"] == 456
        assert set(report["arms"]) == {"S_4b", "S_judge"}
        assert report["capability_A7_P1"]["new_agent"]["determinate"] is True
        assert report["selector_gate_A7a_2"]["paired_instances"] == 114
        assert set(report["comparison_to_gpt5_mini"]["arms"]) == {"S_4b", "S_judge"}
        assert "fixed_unit_weights_PRIMARY_delta" in \
            report["comparison_to_gpt5_mini"]["arms"]["S_judge"]
        assert report["confirmatory_verdict"]["branch"] in {
            "DESCRIPTIVE ONLY", "A7-P3' NULL ESTABLISHED",
            "A7-P3'' POSITIVE", "INCONCLUSIVE",
        }

        # Bite: one missing candidate gold must fail and must not write output.
        candidate_zero = gold_root / "a7_agent_axis_cand0/gold_extra.jsonl"
        lines = candidate_zero.read_text().splitlines()
        candidate_zero.write_text("\n".join(lines[:-1]) + "\n")
        bad_output = tmp / "must_not_exist.json"
        result = run(
            "scripts/readout_amendment7.py",
            "--bundle", str(bundle_path),
            "--scores-dir", "data/replication_multilingual/modal/prefix4k",
            "--judge-cache", "data/selector_axis/judge_multilingual.jsonl",
            "--gold-root", str(gold_root),
            "--random-picks", str(inputs / "random_picks.jsonl"),
            "--out", str(bad_output),
            expect=2,
        )
        assert "gold incomplete" in (result.stdout + result.stderr)
        assert not bad_output.exists()

    print("A7 production-path fixture: PASS (114x4 joins, both arms, gates, JSON, partial-gold bite)")


if __name__ == "__main__":
    main()
