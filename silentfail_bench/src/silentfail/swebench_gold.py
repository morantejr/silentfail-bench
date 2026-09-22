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
DATASET = "princeton-nlp/SWE-bench_Verified"


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
        "--cache_level", "env",          # keep env images, drop instance images
    ]
    if namespace:
        cmd += ["--namespace", namespace]
    if instance_ids:
        cmd += ["--instance_ids", *instance_ids]

    proc = subprocess.run(cmd, cwd=str(ROOT), capture_output=True, text=True, timeout=timeout * 4)
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
    """g = 1 iff the instance is in `resolved_ids`. Everything else is failure."""
    resolved = set(report.get("resolved_ids", []) or [])
    return {iid: (1 if iid in resolved else 0) for iid in instance_ids}
