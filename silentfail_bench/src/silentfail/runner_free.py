"""Profile-A Stage-0 runner: real gold, staged by model, train-only rank norms.

Differences from runner.run_pilot_mock (which stays as the CI dry-run):
  * g comes from executed held-out HumanEval+ tests, never a hash.
  * Stages are ordered BY MODEL (generate-all -> c_sel-all -> audit_0-all ->
    audit_1-all). The anchor runner interleaves models per instance, which on a
    24 GB machine forces a model swap per trajectory.
  * Rank-norm fits use TRAIN groups only. The anchor runner fit them on the
    first half of usable rows, which overlaps the later test split and
    contradicts the `rank_norm_fit: train_split_only` freeze.
"""

from __future__ import annotations

import json
import re
import time
import uuid
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from silentfail.backends import make_backend
from silentfail.identifiers import assign_split
from silentfail.local_agent import Trajectory, grade_gold, run_agent
from silentfail.logging_schema import make_row
from silentfail.scoring import apply_panel_u, fit_rank_norms
from silentfail.tasks_local import Task, load_tasks, sample_tasks

_NUM = re.compile(r"-?\d+(?:\.\d+)?")

JUDGE_PROMPT = (
    "You are evaluating whether an AI coding agent's work correctly and completely "
    "solves the task, including edge cases it was never tested on.\n\n"
    "{traj}\n\n"
    "How likely is this solution to be FULLY correct on exhaustive hidden tests?\n"
    "Reply with ONLY a number from 0 (certainly wrong) to 10 (certainly correct)."
)


def judge_score(backend, traj_text: str, critic_id: str) -> float:
    """LLM-as-judge scalar. NOT a trained reward model -- see anchor_free.yaml."""
    out = backend.generate(
        JUDGE_PROMPT.format(traj=traj_text[:12000]), temperature=0.0, seed=0
    )
    m = _NUM.search(out)
    if not m:
        return float("nan")
    return max(0.0, min(10.0, float(m.group())))


def run_pilot_free(anchor_path: Path, out_dir: Path) -> Path:
    cfg = yaml.safe_load(anchor_path.read_text())
    base_url = cfg["agent"]["base_url"]
    n_cand = int(cfg["selection"]["n"])
    n_inst = int(cfg["tasks"]["n_instances"])
    max_steps = int(cfg["selection"]["max_agent_steps"])
    temp = float(cfg["selection"]["temperature"])
    seed0 = int(cfg["split"]["split_seed"])
    out_dir.mkdir(parents=True, exist_ok=True)

    # Principled exclusion: tasks whose EvalPlus reference does not itself pass
    # gold have an invalid oracle and would surface as phantom agent failures.
    invalid = set(json.loads((Path(__file__).resolve().parents[2] /
                  "data" / "humaneval" / "gold_control_failures.json").read_text()))
    pool = [t for t in load_tasks() if t.task_id not in invalid]
    by_id = {t.task_id: t for t in pool}
    chosen = [t for t in sample_tasks(len(pool) , seed0) if t.task_id in by_id][:n_inst]
    print(f"[tasks] {len(chosen)} instances (excluded {len(invalid)} invalid-oracle)")

    # ---------------- Stage 1: generation (agent model only) ----------------
    agent_be = make_backend("openai_compatible", cfg["agent"]["model_name"], base_url=base_url)
    t0 = time.time()
    cands: dict[str, list[Trajectory]] = {}
    for i, t in enumerate(chosen):
        trs = [run_agent(t, agent_be, max_steps=max_steps, temperature=temp,
                         seed=seed0 + 1000 * i + k) for k in range(n_cand)]
        cands[t.task_id] = trs
        if (i + 1) % 10 == 0:
            print(f"[gen] {i+1}/{len(chosen)} tasks  ({time.time()-t0:.0f}s)")
    print(f"[gen] done in {time.time()-t0:.0f}s")

    # ---------------- Stage 2: c_sel scores ALL candidates ------------------
    sel_be = make_backend("openai_compatible", cfg["selection_critic"]["ollama_id"], base_url=base_url)
    sel_id = cfg["selection_critic"]["id"]
    t0 = time.time()
    cand_scores: dict[str, list[float]] = {}
    for i, t in enumerate(chosen):
        cand_scores[t.task_id] = [judge_score(sel_be, tr.text, sel_id) for tr in cands[t.task_id]]
        if (i + 1) % 10 == 0:
            print(f"[c_sel] {i+1}/{len(chosen)}  ({time.time()-t0:.0f}s)")
    print(f"[c_sel] done in {time.time()-t0:.0f}s")

    # ---------------- Stage 3: best-of-n commit (argmax of whole-traj c) ----
    committed: dict[str, tuple[Trajectory, float]] = {}
    for t in chosen:
        scores = cand_scores[t.task_id]
        finite = [s for s in scores if s == s]
        if not finite:
            committed[t.task_id] = (cands[t.task_id][0], float("nan"))
            continue
        best = int(max(range(len(scores)), key=lambda k: (scores[k] if scores[k] == scores[k] else -1e9)))
        committed[t.task_id] = (cands[t.task_id][best], float(scores[best]))

    # ---------------- Stage 4: audit panel, one model at a time -------------
    audit_raw: dict[str, dict[str, float]] = {t.task_id: {} for t in chosen}
    for member in cfg["audit_panel"]["members"]:
        mid, mname = member["id"], member["ollama_id"]
        be = make_backend("openai_compatible", mname, base_url=base_url)
        t0 = time.time()
        for i, t in enumerate(chosen):
            audit_raw[t.task_id][mid] = judge_score(be, committed[t.task_id][0].text, mid)
        print(f"[{mid}:{mname}] done in {time.time()-t0:.0f}s")

    # ---------------- Stage 5: real gold on committed trajectories ----------
    gold: dict[str, tuple[int, str]] = {}
    t0 = time.time()
    for t in chosen:
        gold[t.task_id] = grade_gold(t, committed[t.task_id][0].final_solution, timeout=30)
    print(f"[gold] done in {time.time()-t0:.0f}s")

    # ---------------- Stage 6: rows, split, TRAIN-ONLY rank norms -----------
    rows: list[dict[str, Any]] = []
    for t in chosen:
        tr, c = committed[t.task_id]
        raw = audit_raw[t.task_id]
        g, why = gold[t.task_id]
        excluded, reason = False, None
        if tr.error:
            excluded, reason = True, "generation_error"
        elif c != c:
            excluded, reason = True, "judge_unparseable_score"
        elif any(v != v for v in raw.values()):
            excluded, reason = True, "judge_unparseable_score"
        rows.append(make_row(
            trajectory_id=str(uuid.uuid4()),
            instance_id=t.task_id,
            repo=t.group_key,                       # profile A: problem_id, NOT a repo
            split=assign_split(t.group_key, train_frac=cfg["split"]["train_frac"],
                               salt=str(seed0)),
            seed=seed0,
            n_candidates=n_cand,
            selection_rule="best_of_n",
            candidate_scores=cand_scores[t.task_id],
            c=c,
            c_sel_id=sel_id,
            audit_scores=raw,
            u=float("nan"),
            g=g,
            gold_reason=why,
            visible_pass=bool(tr.steps[-1].visible_pass) if tr.steps else False,
            length_chars=len(tr.text),
            step_count=tr.step_count,
            mean_agent_entropy=None,
            excluded=excluded,
            exclusion_reason=reason,
            agent_git_commit=str(cfg["agent"].get("git_commit", "UNPINNED")),
            wall_time_sec=None,
            backend="ollama",
            critic_type="llm_judge",
            is_true_reward_model=False,
        ))

    train_raw: dict[str, list[float]] = {m["id"]: [] for m in cfg["audit_panel"]["members"]}
    for r in rows:
        if r["excluded"] or r["split"] != "train":
            continue
        for cid, v in r["audit_scores"].items():
            train_raw[cid].append(v)
    fits = fit_rank_norms(train_raw)
    for r in rows:
        if not r["excluded"] and r["audit_scores"]:
            rn, u = apply_panel_u(r["audit_scores"], fits)
            r["audit_scores_rank_norm"] = rn
            r["u"] = u

    path = out_dir / "pilot_free_scored.jsonl"
    with path.open("w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(f"[write] {len(rows)} rows -> {path}")
    return path
