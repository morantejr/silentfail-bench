#!/usr/bin/env python3
"""Pilot 3b: re-run selection over an existing candidate pool with a true-RM c_sel.

No generation. Loads every checkpointed candidate (text + final solution),
scores all of them with a scalar reward model, commits the argmax per task,
audits the committed trajectory with the 4-RM panel, grades gold, splits,
fits rank norms on train only, and writes the standard row schema.

Exists because the LLM-judge c_sel emitted only {5, 6, 8} over 640
candidates, tying all 16 at the max on 21/40 tasks; argmax then took
candidate 0, so 39/40 "selected" trajectories were the first draw. Selection
pressure was never applied.
"""
from __future__ import annotations

import argparse, json, sys, time, uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import yaml  # noqa: E402
from silentfail.identifiers import assign_split  # noqa: E402
from silentfail.local_agent import grade_gold  # noqa: E402
from silentfail.logging_schema import make_row  # noqa: E402
from silentfail.rm_backend import RewardModel  # noqa: E402
from silentfail.scoring import apply_panel_u, fit_rank_norms  # noqa: E402
from silentfail.tasks_local import load_tasks  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--anchor", type=Path, required=True)
    ap.add_argument("--candidates", type=Path, required=True, help="candidates.jsonl with text+final_solution")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--c-sel-hf-id", required=True)
    a = ap.parse_args()
    cfg = yaml.safe_load(a.anchor.read_text())
    seed0 = int(cfg["split"]["split_seed"])
    a.out.mkdir(parents=True, exist_ok=True)
    tasks = {t.task_id: t for t in load_tasks()}

    pool: dict[str, list[dict]] = {}
    for line in a.candidates.read_text().splitlines():
        d = json.loads(line); pool.setdefault(d["instance_id"], []).append(d)
    for v in pool.values(): v.sort(key=lambda d: d["candidate"])
    n_cand = max(len(v) for v in pool.values())
    print(f"[pool] {sum(len(v) for v in pool.values())} candidates over {len(pool)} tasks (n={n_cand})", flush=True)

    # ---- Stage 2': c_sel = true RM over ALL candidates ----
    t0 = time.time(); rm = RewardModel.load(a.c_sel_hf_id, "c_sel")
    cand_scores = {tid: [rm.score(tasks[tid].prompt, c["text"]) for c in cs] for tid, cs in pool.items()}
    rm.unload(); print(f"[c_sel:{a.c_sel_hf_id}] {sum(len(v) for v in cand_scores.values())} scores in {time.time()-t0:.0f}s", flush=True)
    sel_id = "c_sel_" + a.c_sel_hf_id.split("/")[-1].replace("-", "_").lower()

    committed = {}
    for tid, scores in cand_scores.items():
        best = max(range(len(scores)), key=lambda k: scores[k])
        committed[tid] = (pool[tid][best], float(scores[best]), best)
    idx_hist = {}
    for _, (_, _, b) in committed.items(): idx_hist[b] = idx_hist.get(b, 0) + 1
    print(f"[select] chosen-index histogram: {dict(sorted(idx_hist.items()))}", flush=True)

    # ---- Stage 4: audit panel (true RMs), one at a time ----
    audit_raw = {tid: {} for tid in pool}
    for m in cfg["audit_panel"]["members"]:
        t0 = time.time(); rmm = RewardModel.load(m["hf_id"], m["id"])
        for tid in pool: audit_raw[tid][m["id"]] = rmm.score(tasks[tid].prompt, committed[tid][0]["text"])
        rmm.unload(); print(f"[{m['id']}:{m['hf_id']}] done in {time.time()-t0:.0f}s", flush=True)

    # ---- Stage 5: gold ----
    gold = {tid: grade_gold(tasks[tid], committed[tid][0]["final_solution"], timeout=30) for tid in pool}
    print(f"[gold] failures: {sum(1 for g,_ in gold.values() if g==0)}/{len(gold)}", flush=True)

    rows = []
    for tid in pool:
        c, cscore, bidx = committed[tid]
        g, why = gold[tid]
        text = c["text"]
        vis = "Visible tests PASSED." in text.split("## Step")[-1] if "## Step" in text else False
        rows.append(make_row(
            trajectory_id=str(uuid.uuid4()), instance_id=tid, repo=tid,
            split=assign_split(tid, train_frac=cfg["split"]["train_frac"], salt=str(seed0)),
            seed=seed0, n_candidates=len(pool[tid]), selection_rule="best_of_n",
            candidate_scores=cand_scores[tid], selected_candidate_index=bidx,
            c=cscore, c_sel_id=sel_id, audit_scores=audit_raw[tid], u=float("nan"),
            g=g, gold_reason=why, visible_pass=bool(vis),
            length_chars=len(text), step_count=text.count("## Step") // 2 or 1,
            mean_agent_entropy=None, excluded=False, exclusion_reason=None,
            agent_git_commit=str(cfg["agent"].get("git_commit", "UNPINNED")), wall_time_sec=None,
            backend="transformers_mps", critic_type="reward_model", is_true_reward_model=True,
            trajectory_text=text, final_solution=c["final_solution"],
        ))
    train_raw = {m["id"]: [] for m in cfg["audit_panel"]["members"]}
    for r in rows:
        if r["split"] == "train":
            for k, v in r["audit_scores"].items(): train_raw[k].append(v)
    fits = fit_rank_norms(train_raw)
    for r in rows:
        r["audit_scores_rank_norm"], r["u"] = apply_panel_u(r["audit_scores"], fits)
    out = a.out / "pilot_free_scored.jsonl"
    out.write_text("".join(json.dumps(r) + "\n" for r in rows))
    print(f"[write] {len(rows)} rows -> {out}", flush=True)


if __name__ == "__main__":
    main()
