#!/usr/bin/env python3
"""Preflight for Amendment 13 (the axis on DeepSeek). Read-only, costs nothing.
Carries forward the two checks from preflight_amendment7.py that guard real money (no Modal
app still billing; the swe-rex image patch without which generation stalls) and adds the
ones this amendment needs. Exits non-zero if anything would waste money or silently mix the
discarded gpt-6-luna checkpoint into the new arm.
"""
from __future__ import annotations
import hashlib, json, os, subprocess, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
fail: list[str] = []
def check(ok, msg, detail=""):
    print(f"  {'ok  ' if ok else 'FAIL'}  {msg}{(' -> ' + detail) if detail else ''}")
    if not ok: fail.append(msg)

print("Amendment 13 preflight\n\nregistration")
pre = (ROOT / "PREREGISTRATION_SWEBENCH_PILOT.md").read_text()
check("# Amendment 13, 1 October 2026" in pre, "Amendment 13 is recorded before the run")
for s in ("step_limit 100", "max_tokens 16000", "cost_limit 1.00", "a13_gpt5mini_100"):
    check(s in pre, f"registration names {s}")

print("\nconfigs")
import yaml
ax = yaml.safe_load((ROOT / "configs/mini_swe_deepseek_axis.yaml").read_text())
check(ax["agent"]["step_limit"] == 100 and ax["model"]["model_kwargs"]["max_tokens"] == 16000
      and ax["agent"]["cost_limit"] == 1.0 and ax["model"]["model_class"] == "openrouter_textbased",
      "DeepSeek axis config: 100 steps, 16k tokens, $1.00 ceiling, text-based class")
cm = yaml.safe_load((ROOT / "configs/mini_swe_openai_axis.yaml").read_text())
check(cm["agent"]["step_limit"] == 100 and cm["agent"]["cost_limit"] == 1.0 and cm["model"]["model_class"] == "litellm",
      "comparator config: 100 steps, $1.00 ceiling, litellm (native tool calls -> swebench.yaml)")
inst = [l.strip() for l in (ROOT / "configs/agent_axis_instances_multilingual.txt").read_text().splitlines() if l.strip() and not l.startswith("#")]
check(len(inst) == 114, "114 registered instances", str(len(inst)))

print("\noutput directories")
for d in ("data/agent_axis_deepseek", "data/agent_axis_gpt5mini_100"):
    p = ROOT / d
    check(not p.exists() or not any(p.glob("traj/cand*/*/*.traj.json")) or os.environ.get("A13_RESUME") == "1",
          f"{d} is fresh (or A13_RESUME=1 to resume)", "exists with trajectories" if p.exists() else "absent")

print("\nthe discarded checkpoint must stay discarded, and intact")
hp = ROOT / "data/agent_axis/CHECKPOINT_HASHES.json"
h = json.loads(hp.read_text())
files = sorted((ROOT / "data/agent_axis/traj/cand0").glob("*/*.traj.json"))
# Canonical method from preflight_amendment7.py: sha256 over "relpath:digest" pairs sorted by
# path. A first draft here hashed bare digests and reported a tampered checkpoint that was not.
OUTA = ROOT / "data/agent_axis"
per = {str(f.relative_to(OUTA)): hashlib.sha256(f.read_bytes()).hexdigest() for f in files}
roll = hashlib.sha256("".join(f"{k}:{v}" for k, v in sorted(per.items())).encode()).hexdigest()
check(len(files) == h["n_trajectories"], "100 gpt-6-luna trajectories still present", str(len(files)))
check(roll == h["rollup_sha256"], "their rollup hash still matches the pre-outcome record")

print("\nkeys (presence only)")
env = Path.home() / "Library/Application Support/mini-swe-agent/.env"
t = env.read_text() if env.exists() else ""
for k in ("OPENROUTER_API_KEY", "OPENAI_API_KEY"):
    check(any(l.startswith(k + "=") and len(l) > len(k) + 1 for l in t.splitlines()), f"{k} present in mini-swe-agent's .env")

print("\nModal")
modal = ROOT / ".venv-swebench/bin/modal"
check(modal.exists(), "modal CLI present in the venv")
if modal.exists():
    r = subprocess.run([str(modal), "app", "list"], capture_output=True, text=True, timeout=120)
    live = [l for l in r.stdout.splitlines() if any(s in l for s in ("deployed", "running", "stopping"))]
    check(not live, "no Modal app still deployed or running (and quietly billing)", f"{len(live)} live")
sw = ROOT / ".venv-swebench/lib/python3.12/site-packages/swerex/deployment/modal.py"
check(sw.exists() and "add_python" in sw.read_text(), "swe-rex image patch still applied (else generation stalls on apt)")

print("\ndataset")
try:
    from datasets import load_dataset
    ds = load_dataset("swe-bench/SWE-Bench_Multilingual", split="test")
    cols = set(ds.column_names)
    check({"image", "eval_script", "log_parser", "eval_type"} <= cols, "Multilingual mirror carries the harness columns")
    check(set(inst) <= set(ds["instance_id"]), "all 114 instances exist in the mirror")
except Exception as e:
    check(False, "Multilingual mirror loads", f"{type(e).__name__}: {str(e)[:80]}")

print()
if fail:
    print(f"{len(fail)} PROBLEM(S) -- do not start: " + "; ".join(fail)); sys.exit(1)
print("AMENDMENT 13 PREFLIGHT OK")
