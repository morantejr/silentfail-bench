"""Profile-B gold: real SWE-bench Verified unit tests via the official harness.

This is the only honest source of g for the primary design, and it is why
profile B exists: gold requires per-instance containers, so no amount of
shrinking n or m substitutes for it.

Runs under a separate Python >=3.10 interpreter (`.venv-swebench`), because the
`swebench` package uses PEP-604 annotations at runtime and this host's system
Python is 3.9. The rest of the package stays on 3.9.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
SWEBENCH_PY = ROOT / ".venv-swebench" / "bin" / "python"
DATASET = "SWE-bench/SWE-bench_Verified"   # the 5.x harness needs the new-org dataset: the princeton-nlp mirror lacks the image field


def available() -> bool:
    return SWEBENCH_PY.exists()


def write_predictions(preds: list[dict[str, str]], out: Path) -> Path:
    """preds: [{instance_id, model_patch}] -> harness-format JSONL."""
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as f:
        for p in preds:
            f.write(json.dumps({
                "instance_id": p["instance_id"],
                "model_name_or_path": p.get("model_name_or_path", "silentfail"),
                "model_patch": p["model_patch"],
            }) + "\n")
    return out


def run_evaluation(
    predictions_path: Path,
    run_id: str,
    *,
    instance_ids: list[str] | None = None,
    max_workers: int = 2,
    timeout: int = 1800,
    namespace: str | None = "swebench",
) -> dict[str, Any]:
    """Invoke the official harness. Returns the parsed report.

    `namespace='swebench'` pulls prebuilt images from Docker Hub instead of
    building locally -- far cheaper in time and disk, but the images are
    x86_64, so on Apple Silicon they run under Rosetta.
    """
    cmd = [
        str(SWEBENCH_PY), "-m", "swebench.harness.run_evaluation",
        "--dataset_name", DATASET,
        "--predictions_path", str(predictions_path),
        "--run_id", run_id,
        "--max_workers", str(max_workers),
        "--timeout", str(timeout),
    ]
    # swebench 5.x has no --cache_level / --namespace; images are pulled from
    # the swebench/ Docker Hub namespace by default and are x86_64-only, so the
    # caller sets DOCKER_DEFAULT_PLATFORM=linux/amd64 (done below).
    if instance_ids:
        cmd += ["--instance_ids", *instance_ids]

    import os
    env = dict(os.environ, DOCKER_DEFAULT_PLATFORM="linux/amd64")
    proc = subprocess.run(cmd, cwd=str(ROOT), capture_output=True, text=True, timeout=timeout * 4, env=env)
    report = None
    for cand in sorted(ROOT.glob(f"*.{run_id}.json")) + sorted(ROOT.glob(f"**/*{run_id}*.json")):
        try:
            report = json.loads(cand.read_text())
            break
        except Exception:
            continue
    return {
        "returncode": proc.returncode,
        "stdout_tail": proc.stdout[-4000:],
        "stderr_tail": proc.stderr[-4000:],
        "report": report,
    }


def gold_from_report(report: dict[str, Any], instance_ids: list[str]) -> dict[str, int]:
    """g=1 iff resolved; g=0 iff the harness COMPLETED and did not resolve.

    Instances the harness errored on, skipped as empty, or never completed
    are returned as -1 (gold_error) and must be EXCLUDED by the caller --
    treating them as failures manufactured a 41/41 failure rate when every
    image pull failed on a fresh VM.
    """
    resolved = set(report.get("resolved_ids", []) or [])
    unresolved = set(report.get("unresolved_ids", []) or [])
    out: dict[str, int] = {}
    for iid in instance_ids:
        if iid in resolved:
            out[iid] = 1
        elif iid in unresolved:
            out[iid] = 0
        else:
            out[iid] = -1
    return out
