"""Stage-0 BoN runner: generate n candidates, score with c_sel, audit, gold stub."""

from __future__ import annotations

import json
import time
import uuid
from pathlib import Path
from typing import Any

import yaml

from silentfail.backends import LLMBackend, make_backend
from silentfail.identifiers import assign_split, repo_from_instance, stable_hash
from silentfail.logging_schema import make_row
from silentfail.scoring import RankNormFit, apply_panel_u, fit_rank_norms


def load_anchor(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text())


def build_prompt(instance_id: str) -> str:
    return (
        f"Solve SWE-bench instance {instance_id}. "
        "Produce a minimal patch. (Stage-0 harness wires real problem text later.)"
    )


def run_bon_for_instance(
    instance_id: str,
    *,
    cfg: dict[str, Any],
    agent_backend: LLMBackend,
    sel_backend: LLMBackend,
    audit_backends: dict[str, LLMBackend],
    rank_fits: dict[str, RankNormFit] | None,
    split: str,
    seed: int,
) -> dict[str, Any]:
    n = int(cfg["selection"]["n"])
    temp = float(cfg["selection"]["temperature"])
    prompt = build_prompt(instance_id)
    candidates: list[str] = []
    cand_scores: list[float] = []
    t0 = time.time()

    for i in range(n):
        traj = agent_backend.generate(prompt, temperature=temp, seed=seed + i)
        candidates.append(traj)
        cand_scores.append(sel_backend.score_trajectory(traj, cfg["selection_critic"]["id"]))

    if not candidates:
        return make_row(
            trajectory_id=str(uuid.uuid4()),
            instance_id=instance_id,
            repo=repo_from_instance(instance_id),
            split=split,
            seed=seed,
            n_candidates=0,
            selection_rule="best_of_n",
            candidate_scores=[],
            c=float("nan"),
            c_sel_id=cfg["selection_critic"]["id"],
            audit_scores={},
            u=float("nan"),
            g=0,
            length_chars=0,
            step_count=0,
            mean_agent_entropy=None,
            excluded=True,
            exclusion_reason="empty_candidate_pool",
            agent_git_commit=str(cfg["agent"].get("git_commit", "UNPINNED")),
            wall_time_sec=time.time() - t0,
            backend=cfg["agent"]["model_backend"],
        )

    best_i = int(max(range(n), key=lambda i: cand_scores[i]))
    committed = candidates[best_i]
    c = float(cand_scores[best_i])

    raw_audit = {
        mid: audit_backends[mid].score_trajectory(committed, mid)
        for mid in audit_backends
    }

    if any(v != v for v in [c, *raw_audit.values()]):  # NaN check
        excluded, reason = True, "critic_score_nan"
        u = float("nan")
        rn: dict[str, float] = {}
    else:
        excluded, reason = False, None
        if rank_fits is None:
            # Pilot first pass: use raw z-score across panel as temporary u
            import numpy as np

            vals = list(raw_audit.values())
            u = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
            rn = {}
        else:
            rn, u = apply_panel_u(raw_audit, rank_fits)

    # Gold: Stage-0 mock uses hash bit; real harness replaces with unit tests.
    # Deterministic across processes (defect #2). CI dry-run only:
    # anchor_free.yaml sets gold.mock_g_allowed=false for real pilots.
    g = 1 if stable_hash(committed) % 5 == 0 else 0

    return make_row(
        trajectory_id=str(uuid.uuid4()),
        instance_id=instance_id,
        repo=repo_from_instance(instance_id),
        split=split,
        seed=seed,
        n_candidates=n,
        selection_rule="best_of_n",
        candidate_scores=cand_scores,
        c=c,
        c_sel_id=cfg["selection_critic"]["id"],
        audit_scores=raw_audit,
        audit_scores_rank_norm=rn,
        u=u,
        g=g,
        length_chars=len(committed),
        step_count=committed.count("\n") + 1,
        mean_agent_entropy=None,
        excluded=excluded,
        exclusion_reason=reason,
        agent_git_commit=str(cfg["agent"].get("git_commit", "UNPINNED")),
        wall_time_sec=time.time() - t0,
        backend=cfg["agent"]["model_backend"] if cfg["agent"]["model_backend"] != "vllm" else "vllm",
    )


def run_pilot_mock(anchor_path: Path, out_jsonl: Path, n: int | None = None) -> Path:
    cfg = load_anchor(anchor_path)
    instances = [
        line.strip()
        for line in (anchor_path.parent / "pilot_instances.txt").read_text().splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]
    if n is not None:
        instances = instances[:n]
    # Expand placeholders to reach requested pilot size for dry-run
    target = n or int(cfg.get("pilot_n_trajectories", 200))
    while len(instances) < target:
        instances.append(f"mock__repo-{len(instances)}")

    agent_backend = make_backend("mock", cfg["agent"]["model_name"])
    sel_backend = make_backend("mock", cfg["selection_critic"]["hf_id"])
    audit_backends = {
        m["id"]: make_backend("mock", m["hf_id"]) for m in cfg["audit_panel"]["members"]
    }

    out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    with out_jsonl.open("w") as f:
        for i, inst in enumerate(instances[:target]):
            row = run_bon_for_instance(
                inst,
                cfg=cfg,
                agent_backend=agent_backend,
                sel_backend=sel_backend,
                audit_backends=audit_backends,
                rank_fits=None,
                split="pilot_unassigned",
                seed=20260922 + i,
            )
            rows.append(row)
            f.write(json.dumps(row) + "\n")

    # Second pass: fit rank norms on the TRAIN split only.
    # Previously this took the first half of usable rows, which overlaps the
    # test split assigned later in pilot_summary -- test rows leaked into the
    # rank-normalization that defines u, contradicting the freeze
    # `rank_norm_fit: train_split_only`.
    usable = [r for r in rows if not r["excluded"]]
    for r in usable:
        r["split"] = assign_split(r["repo"], train_frac=float(cfg["split"]["train_frac"]),
                                  salt=str(cfg["split"]["split_seed"]))
    train = [r for r in usable if r["split"] == "train"] or usable[: max(1, len(usable) // 2)]
    train_raw: dict[str, list[float]] = {m["id"]: [] for m in cfg["audit_panel"]["members"]}
    for r in train:
        for cid, val in r["audit_scores"].items():
            train_raw[cid].append(val)
    fits = fit_rank_norms(train_raw)

    final_path = out_jsonl.with_name(out_jsonl.stem + "_scored.jsonl")
    with final_path.open("w") as f:
        for r in rows:
            if not r["excluded"] and r["audit_scores"]:
                rn, u = apply_panel_u(r["audit_scores"], fits)
                r["audit_scores_rank_norm"] = rn
                r["u"] = u
            f.write(json.dumps(r) + "\n")
    return final_path
