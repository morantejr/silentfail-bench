#!/usr/bin/env python3
"""Profile B orchestrator: SWE-bench Verified, mini-swe-agent, true-RM selection + audit, harness gold.

UNTESTED at time of writing beyond the single-instance smoke run; every stage
checkpoints so a crash resumes rather than regenerates.

Stages, staged by model as the free-compute config requires:
  1. generate  : k candidates per instance via `mini-extra swebench-single`
                 (distinct seed per candidate) -> candidates.jsonl
  2. c_sel     : true RM scores ALL candidates; argmax committed per instance
  3. audit     : 4 true RMs score the committed trajectory
  4. gold      : official swebench harness on committed patches -> g
  5. rows      : split by repo (stable hash), rank norms fit on TRAIN only

Trajectory text scored by every critic = the agent's messages, serialized.
Task prompt given to each RM = the instance's problem_statement.
"""
from __future__ import annotations

import argparse, json, os, re, subprocess, sys, time, uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import yaml  # noqa: E402
from silentfail.identifiers import assign_split, repo_from_instance  # noqa: E402
from silentfail.logging_schema import make_row  # noqa: E402
from silentfail.scoring import apply_panel_u, fit_rank_norms  # noqa: E402

VENV_PY = ROOT / ".venv-swebench" / "bin" / "python"
MINI = ROOT / ".venv-swebench" / "bin" / "mini-extra"


def traj_text(traj: dict) -> str:
    parts = []
    for m in traj.get("messages", []):
        role = m.get("role", "?")
        if role == "system":
            continue
        # Native tool-call format puts command output under role="tool"; keep it,
        # it is where the diff lives.
        c = m.get("content")
        if isinstance(c, list):
            c = "\n".join(x.get("text", "") for x in c if isinstance(x, dict))
        parts.append(f"## {role}\n{c}")
    return "\n\n".join(parts)


def image_for(inst: str) -> str:
    return f"docker.io/swebench/sweb.eval.x86_64.{inst.replace('__', '_1776_')}:latest".lower()


def prune_image(inst: str, env: dict) -> None:
    """Each instance image is ~4.2 GB and the VM holds ~19 GB; keep nothing
    between instances. Gold re-pulls later (two passes, ~42 s each)."""
    subprocess.run(["docker", "rmi", "-f", image_for(inst)], env=env, capture_output=True)


def load_instances(ids: list[str]) -> dict[str, dict]:
    code = ("from datasets import load_dataset; import json,sys; ids=set(json.loads(sys.argv[1]));"
            "print(json.dumps({r['instance_id']:{'problem_statement':r['problem_statement'],'repo':r['repo']} "
            "for r in load_dataset('princeton-nlp/SWE-bench_Verified',split='test') if r['instance_id'] in ids}))")
    out = subprocess.run([str(VENV_PY), "-c", code, json.dumps(ids)], capture_output=True, text=True, check=True)
    return json.loads(out.stdout.strip().splitlines()[-1])


def generate(inst: str, k: int, out_dir: Path, env: dict, mini_cfg: Path, step_limit: int, base_cfg: str) -> list[dict]:
    """k candidates for one instance via mini-swe-agent BATCH mode, one run per seed.

    Batch mode (`mini-extra swebench`) is non-interactive and writes
    info.exit_status / info.submission itself. The single-instance runner
    drops into an "Agent wants to finish -- Enter to quit" prompt at the end
    that -y does not cover; a smoke run sat on it for 90 minutes.
    """
    res = []
    for c in range(k):
        run_dir = out_dir / "traj" / f"cand{c}"
        traj_path = run_dir / inst / f"{inst}.traj.json"
        if not traj_path.exists():
            run_dir.mkdir(parents=True, exist_ok=True)
            t0 = time.time()
            subprocess.run([str(MINI), "swebench", "--subset", "verified", "--split", "test",
                            "--filter", f"^{re.escape(inst)}$", "-o", str(run_dir), "-w", "1", "--redo-existing",
                            "-c", base_cfg, "-c", str(mini_cfg),
                            "-c", f"model.model_kwargs.seed={1000 + c}", "-c", f"agent.step_limit={step_limit}"],
                           env=env, stdout=open(run_dir / f"{inst}.cand{c}.log", "w"), stderr=subprocess.STDOUT)
            print(f"  [gen] {inst} cand {c}: {time.time()-t0:.0f}s", flush=True)
        try:
            d = json.loads(traj_path.read_text())
            res.append({"instance_id": inst, "candidate": c, "text": traj_text(d),
                        "final_solution": d["info"].get("submission") or "",
                        "exit_status": d["info"].get("exit_status"), "steps": len(d.get("messages", [])),
                        "cost_usd": (d["info"].get("model_stats") or {}).get("instance_cost")})
        except Exception as e:
            res.append({"instance_id": inst, "candidate": c, "text": "", "final_solution": "",
                        "exit_status": f"traj_unreadable:{type(e).__name__}", "steps": 0, "cost_usd": None})
    return res


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--anchor", type=Path, default=ROOT / "configs" / "anchor_free_rm4_n16.yaml")
    ap.add_argument("--mini-config", type=Path, default=ROOT / "configs" / "mini_swe_openai.yaml")
    ap.add_argument("--mini-base-config", default="swebench.yaml",
                    help="swebench.yaml = native tool calls (frontier); swebench_backticks.yaml = text-based (local 7B/14B)")
    ap.add_argument("--keep-images", action="store_true", help="skip pruning (needs a big disk)")
    ap.add_argument("--instances", nargs="*", default=[])
    ap.add_argument("--instances-file", type=Path, default=None,
                    help="one instance id per line, # comments allowed; avoids shell word-splitting (zsh does not split $VAR)")
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--step-limit", type=int, default=30)
    ap.add_argument("--out", type=Path, default=ROOT / "data" / "swebench_pilot")
    ap.add_argument("--c-sel-hf-id", default="Skywork/Skywork-Reward-V2-Qwen3-4B")
    ap.add_argument("--stage", choices=["all", "generate", "score"], default="all")
    a = ap.parse_args()
    if a.instances_file:
        a.instances += [l.strip() for l in a.instances_file.read_text().splitlines() if l.strip() and not l.startswith("#")]
    if not a.instances:
        ap.error("no instances: pass --instances-file or --instances")
    if any(" " in i for i in a.instances):
        ap.error("an instance id contains a space: the shell passed several ids as one argument")
    cfg = yaml.safe_load(a.anchor.read_text()); seed0 = int(cfg["split"]["split_seed"])
    a.out.mkdir(parents=True, exist_ok=True)
    # Only Docker plumbing goes into the agent's environment. The model
    # endpoint comes from the mini-config alone: an Ollama overlay sets its own
    # api_base, and the OpenAI overlay must NOT inherit one. Injecting the
    # Ollama base URL here sent gpt-5-mini to localhost:11434 -> NotFoundError.
    env = {k: v for k, v in os.environ.items() if k not in ("OPENAI_API_BASE", "OPENAI_BASE_URL")}
    env["DOCKER_HOST"] = f"unix://{Path.home()}/.colima/default/docker.sock"
    if "ollama" in str(a.mini_config):
        env.update(OPENAI_API_KEY="ollama", OPENAI_API_BASE="http://127.0.0.1:11434/v1",
                   MSWEA_COST_TRACKING="ignore_errors")
    meta = load_instances(a.instances)

    # ---- 1. generate (checkpointed per candidate via traj files) ----
    cand_path = a.out / "candidates.jsonl"
    pool: dict[str, list[dict]] = {}
    for inst in a.instances:
        pool[inst] = generate(inst, a.k, a.out, env, a.mini_config, a.step_limit, a.mini_base_config)
        if not a.keep_images:
            prune_image(inst, env)
    cand_path.write_text("".join(json.dumps(c) + "\n" for cs in pool.values() for c in cs))
    empty = sum(1 for cs in pool.values() for c in cs if not c["final_solution"].strip())
    print(f"[gen] {sum(len(v) for v in pool.values())} candidates; empty patches: {empty}", flush=True)
    if a.stage == "generate":
        return

    # ---- 2. c_sel over ALL candidates, then 3. audit (one RM resident at a time) ----
    from silentfail.rm_backend import RewardModel
    t0 = time.time(); rm = RewardModel.load(a.c_sel_hf_id, "c_sel")
    scores = {i: [rm.score(meta[i]["problem_statement"], c["text"]) if c["text"] else float("-inf") for c in cs]
              for i, cs in pool.items()}
    rm.unload(); print(f"[c_sel] done in {time.time()-t0:.0f}s", flush=True)
    committed = {i: max(range(len(s)), key=lambda k: s[k]) for i, s in scores.items()}
    print(f"[select] chosen idx: { {i: committed[i] for i in list(committed)[:12]} }", flush=True)
    audit: dict[str, dict[str, float]] = {i: {} for i in pool}
    for m in cfg["audit_panel"]["members"]:
        t0 = time.time(); rmm = RewardModel.load(m["hf_id"], m["id"])
        for i in pool:
            audit[i][m["id"]] = rmm.score(meta[i]["problem_statement"], pool[i][committed[i]]["text"])
        rmm.unload(); print(f"[{m['id']}] done in {time.time()-t0:.0f}s", flush=True)

    # ---- 4. gold: official harness on the committed patches ----
    from silentfail.swebench_gold import gold_from_report, run_evaluation, write_predictions
    preds = write_predictions([{"instance_id": i, "model_patch": pool[i][committed[i]]["final_solution"],
                                "model_name_or_path": "silentfail"} for i in pool], a.out / "predictions.jsonl")
    run_id = f"sf_{int(time.time())}"
    ev = run_evaluation(preds, run_id, instance_ids=list(pool), max_workers=1, namespace=None)
    if not ev["report"]:
        print("[gold] harness produced no report:\n", ev["stderr_tail"][-1500:], flush=True); sys.exit(2)
    g = gold_from_report(ev["report"], list(pool))
    if not a.keep_images:
        for i in pool:
            prune_image(i, env)
    print(f"[gold] failures {sum(1 for v in g.values() if v == 0)}/{len(g)}  (resolved: {ev['report'].get('resolved_instances')})", flush=True)

    # ---- 5. rows ----
    rows = []
    for i in pool:
        c = pool[i][committed[i]]
        rows.append(make_row(
            trajectory_id=str(uuid.uuid4()), instance_id=i, repo=repo_from_instance(i),
            split=assign_split(repo_from_instance(i), train_frac=cfg["split"]["train_frac"], salt=str(seed0)),
            seed=seed0, n_candidates=len(pool[i]), selection_rule="best_of_n",
            candidate_scores=scores[i], selected_candidate_index=committed[i],
            c=float(scores[i][committed[i]]), c_sel_id="c_sel_" + a.c_sel_hf_id.split("/")[-1].lower(),
            audit_scores=audit[i], u=float("nan"), g=g[i],
            visible_pass=(c["exit_status"] == "Submitted"),     # agent believed it was done
            length_chars=len(c["text"]), step_count=c["steps"], mean_agent_entropy=None,
            excluded=not c["final_solution"].strip(), exclusion_reason=("empty_patch" if not c["final_solution"].strip() else None),
            agent_git_commit="mini-swe-agent==2.4.6", wall_time_sec=None, backend="mini-swe-agent+ollama+transformers_mps",
            critic_type="reward_model", is_true_reward_model=True, trajectory_text=c["text"], final_solution=c["final_solution"],
        ))
    train_raw = {m["id"]: [] for m in cfg["audit_panel"]["members"]}
    for r in rows:
        if r["split"] == "train" and not r["excluded"]:
            for k, v in r["audit_scores"].items(): train_raw[k].append(v)
    fits = fit_rank_norms(train_raw)
    for r in rows:
        r["audit_scores_rank_norm"], r["u"] = apply_panel_u(r["audit_scores"], fits)
    (a.out / "pilot_free_scored.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    print(f"[write] {len(rows)} rows -> {a.out/'pilot_free_scored.jsonl'}", flush=True)


if __name__ == "__main__":
    main()
