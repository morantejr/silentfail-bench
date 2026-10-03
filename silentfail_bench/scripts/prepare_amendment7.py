#!/usr/bin/env python3
"""Prepare frozen, local-only inputs for Amendment 7 gold and random control.

This script makes no API calls. It validates the completed candidate bundle
against the 114-instance registered set, emits one gold-input file per
candidate index (gold_modal cannot distinguish duplicate instance IDs in one
run), and fixes the random arm by a deterministic hash chosen before gold.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def random_candidate(instance_id: str, k: int = 4) -> int:
    """Deterministic uniform pick from candidate indices; fixed pre-outcome."""
    digest = hashlib.sha256(f"a7-random-v1:{instance_id}".encode()).digest()
    return int.from_bytes(digest[:8], "big") % k


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bundle", type=Path, default=ROOT / "data/agent_axis/rescoring_bundle.jsonl")
    ap.add_argument("--instances", type=Path, default=ROOT / "configs/agent_axis_instances_multilingual.txt")
    ap.add_argument("--out", type=Path, default=ROOT / "data/agent_axis/gold_inputs")
    ap.add_argument("--agent-label", default="gpt6luna",
                    help="label written into model_name_or_path (a7_<label>_cand{N}); cosmetic for the readout, which keys gold by (instance, candidate)")
    args = ap.parse_args()

    expected = [line.strip() for line in args.instances.read_text().splitlines()
                if line.strip() and not line.startswith("#")]
    rows = read_jsonl(args.bundle)
    by_key = {(r["instance_id"], int(r["candidate"])): r for r in rows}
    expected_keys = {(instance_id, candidate) for instance_id in expected for candidate in range(4)}
    actual_keys = set(by_key)
    if actual_keys != expected_keys:
        missing = sorted(expected_keys - actual_keys)
        extra = sorted(actual_keys - expected_keys)
        print(f"bundle does not match the registered 114 x 4 set: "
              f"{len(missing)} missing, {len(extra)} extra", file=sys.stderr)
        if missing:
            print(f"  first missing: {missing[:5]}", file=sys.stderr)
        if extra:
            print(f"  first extra: {extra[:5]}", file=sys.stderr)
        sys.exit(1)

    args.out.mkdir(parents=True, exist_ok=True)
    for candidate in range(4):
        preds, mapping = [], []
        for instance_id in expected:
            row = by_key[(instance_id, candidate)]
            mapping.append({"instance_id": instance_id, "candidate": candidate})
            patch = row.get("final_solution") or ""
            if patch.strip():
                preds.append({
                    "instance_id": instance_id,
                    "model_name_or_path": f"a7_{args.agent_label}_cand{candidate}",
                    "model_patch": patch,
                })
        (args.out / f"cand{candidate}_predictions.jsonl").write_text(
            "".join(json.dumps(r) + "\n" for r in preds)
        )
        (args.out / f"cand{candidate}_mapping.jsonl").write_text(
            "".join(json.dumps(r) + "\n" for r in mapping)
        )
        print(f"cand{candidate}: {len(preds)} non-empty patches; "
              f"{len(expected) - len(preds)} empties count as failures without a gold run")

    picks = [{"instance_id": instance_id, "candidate": random_candidate(instance_id)}
             for instance_id in expected]
    (args.out / "random_picks.jsonl").write_text(
        "".join(json.dumps(r) + "\n" for r in picks)
    )
    manifest = {
        "registered_instances": len(expected),
        "candidate_slots": len(expected_keys),
        "random_rule": "sha256('a7-random-v1:' + instance_id), first 8 bytes big-endian mod 4",
        "instance_file_sha256": hashlib.sha256(args.instances.read_bytes()).hexdigest(),
        "bundle_sha256": hashlib.sha256(args.bundle.read_bytes()).hexdigest(),
    }
    (args.out / "MANIFEST.json").write_text(json.dumps(manifest, indent=1) + "\n")
    print(f"wrote local gold inputs and frozen random picks to {args.out}")


if __name__ == "__main__":
    main()
