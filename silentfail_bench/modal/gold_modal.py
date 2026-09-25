#!/usr/bin/env python3
"""Official SWE-bench gold on Modal, without Docker on this machine.

swebench 5.0.2's own `--modal` path is broken (its remote code reads a TestSpec field that no longer
exists). This mirrors `swebench.harness.run_evaluation.run_instance` step for step in a Modal
Sandbox built from the dataset's own `image` field, then grades the captured log with
`swebench.harness.grading.get_eval_report` locally. Same patch-apply chain, same eval script,
same parser, same resolution rule.

    python modal/gold_modal.py --preds preds.jsonl --run-id NAME [--dataset SWE-bench/SWE-bench_Verified]
                               [--oracle 1] [--workers 6] [--candidates needs_gold.jsonl]
Writes data/gold_modal/<run-id>/{report.json, gold_extra.jsonl, logs/<instance>/test_output.txt}.
--oracle N: on the first N instances also runs the dataset's reference patch (must resolve) and an
empty patch (must not) before trusting the runner.
"""
from __future__ import annotations

import argparse, concurrent.futures, json, sys, time
from pathlib import Path

import modal
from datasets import load_dataset
from swebench.harness.grading import get_eval_report
from swebench.harness.utils import make_test_spec

ROOT = Path(__file__).resolve().parents[1]
GIT_APPLY_CMDS = ["git apply --verbose", "git apply --verbose --3way", "git apply --verbose --reject", "patch --batch --forward --fuzz=5 -p1 -i"]
app = modal.App("silentfail-gold")


def run_one(ts, pred: dict, log_dir: Path, timeout: int) -> dict:
    log_dir.mkdir(parents=True, exist_ok=True)
    rec = {"instance_id": ts.instance_id, "model_name_or_path": pred["model_name_or_path"]}
    img = modal.Image.from_registry(ts.image, add_python="3.11")
    t0 = time.time()
    sb = modal.Sandbox.create("sleep", "infinity", image=img, app=app, timeout=timeout + 900, cpu=2, memory=6144, workdir="/testbed")
    try:
        def sh(cmd: str, to: int | None = None) -> tuple[int, str]:
            p = sb.exec("bash", "-c", cmd + " 2>&1", timeout=to); out = p.stdout.read(); p.wait(); return p.returncode, out
        def put(path: str, data: str) -> None:          # write a file through the process's stdin (no filesystem API needed)
            p = sb.exec("bash", "-c", f"cat > {path}"); p.stdin.write(data.encode()); p.stdin.write_eof(); p.wait()
        put("/tmp/patch.diff", pred["model_patch"] or "")
        applied, out = False, ""
        if (pred["model_patch"] or "").strip():
            for i, cmd in enumerate(GIT_APPLY_CMDS):
                if i: sh("git checkout -- . ; git clean -fd")
                rc, out = sh(f"{cmd} /tmp/patch.diff")
                if rc == 0: applied = True; break
            if not applied: applied = sh("git apply --check --reverse /tmp/patch.diff")[0] == 0
            if not applied:
                (log_dir / "patch_apply.txt").write_text(out); rec.update(error="patch apply failed"); return rec
        put("/eval.sh", ts.eval_script)
        rc, test_output = sh("/bin/bash /eval.sh", to=timeout)
        (log_dir / "test_output.txt").write_text(test_output)
        report = get_eval_report(ts, pred, str(log_dir / "test_output.txt"), include_tests_status=True)[ts.instance_id]
        rec.update(resolved=bool(report["resolved"]), patch_applied=applied, runtime_s=round(time.time() - t0),
                   f2p=f"{len(report['tests_status']['FAIL_TO_PASS']['success'])}/{len(ts.FAIL_TO_PASS)}",
                   p2p=f"{len(report['tests_status']['PASS_TO_PASS']['success'])}/{len(ts.PASS_TO_PASS)}")
    except Exception as e:
        rec.update(error=f"{type(e).__name__}: {str(e)[:300]}")
    finally:
        try: sb.terminate()
        except Exception: pass
    return rec


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--preds", type=Path, required=True); ap.add_argument("--run-id", required=True)
    ap.add_argument("--dataset", default="SWE-bench/SWE-bench_Verified"); ap.add_argument("--split", default="test")
    ap.add_argument("--workers", type=int, default=6); ap.add_argument("--timeout", type=int, default=1800)
    ap.add_argument("--oracle", type=int, default=0); ap.add_argument("--candidates", type=Path, default=None, help="jsonl with instance_id,candidate to emit gold_extra.jsonl")
    a = ap.parse_args()
    out = ROOT / "data/gold_modal" / a.run_id; out.mkdir(parents=True, exist_ok=True)
    preds = [json.loads(l) for l in a.preds.read_text().splitlines() if l.strip()]
    ds = {r["instance_id"]: r for r in load_dataset(a.dataset, split=a.split)}
    jobs = []
    for i, p in enumerate(preds):
        inst = ds[p["instance_id"]]; ts = make_test_spec(inst)
        if i < a.oracle:
            jobs.append((ts, {"instance_id": p["instance_id"], "model_name_or_path": "ORACLE_reference_patch", "model_patch": inst["patch"]}, out / "logs" / f"{p['instance_id']}__oracle_ref"))
            jobs.append((ts, {"instance_id": p["instance_id"], "model_name_or_path": "ORACLE_empty_patch", "model_patch": ""}, out / "logs" / f"{p['instance_id']}__oracle_empty"))
        jobs.append((ts, p, out / "logs" / p["instance_id"]))
    print(f"[gold] {len(jobs)} evaluations ({a.oracle} oracle pairs) on Modal, workers={a.workers}", flush=True)
    results = []
    with modal.enable_output(), app.run():
        with concurrent.futures.ThreadPoolExecutor(a.workers) as ex:
            for rec in ex.map(lambda j: run_one(j[0], j[1], j[2], a.timeout), jobs):
                results.append(rec); print(f"  {rec['instance_id']:40s} {rec['model_name_or_path'][:24]:24s} resolved={rec.get('resolved')} f2p={rec.get('f2p')} p2p={rec.get('p2p')} {rec.get('error','')}", flush=True)
    real = [r for r in results if not r["model_name_or_path"].startswith("ORACLE")]
    report = {"run_id": a.run_id, "dataset": a.dataset, "total": len(real),
              "resolved_ids": sorted(r["instance_id"] for r in real if r.get("resolved") is True),
              "unresolved_ids": sorted(r["instance_id"] for r in real if r.get("resolved") is False),
              "error_ids": sorted(r["instance_id"] for r in real if "error" in r),
              "oracle": [r for r in results if r["model_name_or_path"].startswith("ORACLE")], "details": results}
    (out / "report.json").write_text(json.dumps(report, indent=1))
    if a.candidates:
        cand = {json.loads(l)["instance_id"]: json.loads(l)["candidate"] for l in a.candidates.read_text().splitlines() if l.strip()}
        with (out / "gold_extra.jsonl").open("w") as f:
            for r in real:
                if "error" not in r: f.write(json.dumps({"instance_id": r["instance_id"], "candidate": cand[r["instance_id"]], "g": 1 if r["resolved"] else 0}) + "\n")
    ok = [o for o in report["oracle"]]
    print(f"[gold] resolved {len(report['resolved_ids'])} / unresolved {len(report['unresolved_ids'])} / errors {len(report['error_ids'])}; oracle: " +
          ", ".join(f"{o['model_name_or_path'].split('_',1)[1]}={o.get('resolved')}" for o in ok), flush=True)


if __name__ == "__main__":
    main()
