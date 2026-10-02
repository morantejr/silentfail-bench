#!/usr/bin/env python3
"""Amendment 4e: wide-repository replication on SWE-bench Multilingual, agent sandboxes and gold on Modal.

Stages (each checkpointed):
  generate : k candidates per instance, mini-swe-agent batch mode, environment_class swerex_modal
  bundle   : candidates -> rescoring_bundle.jsonl (same schema as the pilot's; committed index filled by `select`)
  select   : argmax of the selection critic's PREFIX-4k scores (from modal/rescore.py --mode prefix) -> preds for gold
  rows     : gold + audit prefix scores -> rows jsonl in the pilot's schema (u rank-normed on train repos)
The RM scoring itself runs on Modal:  modal volume put silentfail-bundles <bundle> /<run>/rescoring_bundle.jsonl
                                      python -m modal run modal/rescore.py --run <run> --out prefix4k --max-tokens 4000 --mode prefix
Gold: python modal/gold_modal.py --dataset swe-bench/SWE-Bench_Multilingual --preds <out>/predictions.jsonl --run-id <run> --oracle 3
"""
from __future__ import annotations

import argparse, json, os, re, subprocess, sys, time, uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "scripts"))
import yaml  # noqa: E402
from run_swebench_pilot import MINI, traj_text  # noqa: E402
from silentfail.identifiers import assign_split, repo_from_instance  # noqa: E402
from silentfail.logging_schema import make_row  # noqa: E402
from silentfail.scoring import apply_panel_u, fit_rank_norms  # noqa: E402

DATASET = "swe-bench/SWE-Bench_Multilingual"
C_SEL = "skywork_skywork_reward_v2_qwen3_4b"
PANEL = {"grm_gemma2_2b": "ray2333_grm_gemma2_2b_rewardmodel_ft", "grm_llama3_2_3b": "ray2333_grm_llama3_2_3b_rewardmodel_ft",
         "skywork_llama3_2_1b": "skywork_skywork_reward_v2_llama_3_2_1b", "skywork_qwen3_1_7b": "skywork_skywork_reward_v2_qwen3_1_7b"}


def gen(ids: list[str], k: int, out: Path, cfg: Path, workers: int, step_limit: int, cost_cap: float,
        base_config: str = "swebench.yaml") -> None:
    """base_config MUST match the agent config's model class. swebench.yaml supplies
    tool-call prompts; swebench_backticks.yaml supplies the ```mswea_bash_command fence
    that the *_textbased model classes parse. Pairing a textbased class with swebench.yaml
    produces RepeatedFormatError on every turn, which is the documented gpt-6-luna and
    DeepSeek failure (DEFECTS.md)."""
    flt = "^(" + "|".join(re.escape(i) for i in ids) + ")$"
    for c in range(k):
        run_dir = out / "traj" / f"cand{c}"; run_dir.mkdir(parents=True, exist_ok=True); t0 = time.time()
        # Launched through envs/run_swebench_closing.py, not mini-extra: identical CLI, but each
        # Modal sandbox is terminated when its trajectory ends (DEFECTS #21: upstream never does).
        subprocess.run([sys.executable, "-m", "envs.run_swebench_closing", "--subset", "multilingual", "--split", "test", "--filter", flt, "-o", str(run_dir),
                        "-w", str(workers), "-c", base_config, "-c", str(cfg), "-c", f"model.model_kwargs.seed={1000 + c}",
                        "-c", f"agent.step_limit={step_limit}"],
                       stdout=open(run_dir / "gen.log", "a"), stderr=subprocess.STDOUT, cwd=str(ROOT),
                       env={**{k: v for k, v in os.environ.items() if k not in ("OPENAI_API_BASE", "OPENAI_BASE_URL")},   # never inherit a local model base URL
                            "PYTHONPATH": str(ROOT)})
        cost = spent(out); print(f"[gen] cand {c}: {time.time()-t0:.0f}s, {len(list(run_dir.glob('*/*.traj.json')))} trajectories, cumulative est ${cost:.2f}", flush=True)
        if cost > cost_cap: print(f"[gen] COST CAP ${cost_cap} reached; stopping after cand {c}", flush=True); return


def spent(out: Path) -> float:
    tot = 0.0
    for p in out.glob("traj/cand*/*/*.traj.json"):
        try: tot += float((json.loads(p.read_text())["info"].get("model_stats") or {}).get("instance_cost") or 0)
        except Exception: pass
    return tot


def bundle(ids: list[str], k: int, out: Path, seed: int, train_frac: float) -> None:
    from datasets import load_dataset
    meta = {r["instance_id"]: r for r in load_dataset(DATASET, split="test") if r["instance_id"] in set(ids)}
    ef_path = out / "ENDPOINT_FAILURES.json"
    endpoint_failures = {k: v for k, v in json.loads(ef_path.read_text()).items() if k.startswith("cand")} if ef_path.exists() else {}
    rows = []
    for i in ids:
        for c in range(k):
            p = out / "traj" / f"cand{c}" / i / f"{i}.traj.json"
            if not p.exists():
                # A declared endpoint failure (ENDPOINT_FAILURES.json) is an excluded row, not a missing one.
                if i in endpoint_failures.get(f"cand{c}", {}):
                    rows.append({"instance_id": i, "candidate": c, "problem_statement": meta[i]["problem_statement"], "repo": repo_from_instance(i),
                                 "split": assign_split(repo_from_instance(i), train_frac=train_frac, salt=str(seed)), "text": "", "final_solution": "",
                                 "exit_status": "EndpointFailure", "steps": 0, "visible_pass": False, "excluded": True,
                                 "exclusion_reason": "endpoint_failure", "committed_index_primary": -1, "g": None})
                continue
            try: d = json.loads(p.read_text()); text = traj_text(d); sub = d["info"].get("submission") or ""; st = d["info"].get("exit_status"); steps = len(d.get("messages", []))
            except Exception as e: text, sub, st, steps = "", "", f"traj_unreadable:{type(e).__name__}", 0
            repo = repo_from_instance(i)
            rows.append({"instance_id": i, "candidate": c, "problem_statement": meta[i]["problem_statement"], "repo": repo,
                         "split": assign_split(repo, train_frac=train_frac, salt=str(seed)), "text": text, "final_solution": sub,
                         "exit_status": st, "steps": steps, "visible_pass": st == "Submitted", "excluded": not sub.strip(),
                         "committed_index_primary": -1, "g": None})
    (out / "rescoring_bundle.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    n_inst = len({r["instance_id"] for r in rows}); print(f"[bundle] {len(rows)} candidates over {n_inst} instances; submitted {sum(r['visible_pass'] for r in rows)}; empty patches {sum(r['excluded'] for r in rows)}; declared endpoint failures {sum(r.get('exit_status') == 'EndpointFailure' for r in rows)}", flush=True)


def select(out: Path, scores_dir: Path) -> None:
    sc = json.loads((scores_dir / f"scores_gpu_{C_SEL}.json").read_text())["scores"]
    rows = [json.loads(l) for l in (out / "rescoring_bundle.jsonl").read_text().splitlines() if l.strip()]
    by: dict[str, list[dict]] = {}
    for r in rows: by.setdefault(r["instance_id"], []).append(r)
    committed, preds = {}, []
    for i, cs in by.items():
        best = max(cs, key=lambda r: sc.get(f"{i}#{r['candidate']}", {}).get("score", float("-inf")) if r["text"] else float("-inf"))
        committed[i] = best["candidate"]
        if best["final_solution"].strip(): preds.append({"instance_id": i, "model_name_or_path": "silentfail_replication", "model_patch": best["final_solution"]})
    for r in rows: r["committed_index_primary"] = committed[r["instance_id"]]
    (out / "rescoring_bundle.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    (out / "predictions.jsonl").write_text("".join(json.dumps(p) + "\n" for p in preds))
    import collections; print(f"[select] {len(committed)} committed; index histogram {dict(collections.Counter(committed.values()))}; {len(preds)} patches to grade", flush=True)


def make_rows(out: Path, scores_dir: Path, gold_report: Path, seed: int) -> None:
    rep = json.loads(gold_report.read_text()); g = {i: 1 for i in rep["resolved_ids"]}; g.update({i: 0 for i in rep["unresolved_ids"]}); err = set(rep["error_ids"])
    S = {k: json.loads((scores_dir / f"scores_gpu_{v}.json").read_text())["scores"] for k, v in PANEL.items()}
    sc = json.loads((scores_dir / f"scores_gpu_{C_SEL}.json").read_text())["scores"]
    rows_b = [json.loads(l) for l in (out / "rescoring_bundle.jsonl").read_text().splitlines() if l.strip()]
    by: dict[str, list[dict]] = {}
    for r in rows_b: by.setdefault(r["instance_id"], []).append(r)
    rows = []
    for i, cs in by.items():
        ci = cs[0]["committed_index_primary"]; c = next(r for r in cs if r["candidate"] == ci); key = f"{i}#{ci}"
        cand_scores = [sc.get(f"{i}#{r['candidate']}", {}).get("score", float("nan")) for r in sorted(cs, key=lambda r: r["candidate"])]
        empty = not c["final_solution"].strip()
        rows.append(make_row(
            trajectory_id=str(uuid.uuid4()), instance_id=i, repo=c["repo"], split=c["split"], seed=seed, n_candidates=len(cs),
            selection_rule="best_of_n", candidate_scores=cand_scores, selected_candidate_index=ci,
            c=float(sc.get(key, {}).get("score", float("nan"))), c_sel_id="c_sel_skywork-reward-v2-qwen3-4b",
            audit_scores={k: float(S[k].get(key, {}).get("score", float("nan"))) for k in PANEL}, u=float("nan"),
            g=g.get(i, 0), visible_pass=c["visible_pass"], length_chars=len(c["text"]), step_count=c["steps"], mean_agent_entropy=None,
            excluded=empty or (i in err) or (i not in g and not empty), exclusion_reason=("empty_patch" if empty else ("gold_error" if (i in err or i not in g) else None)),
            agent_git_commit="mini-swe-agent==2.4.6", wall_time_sec=None, backend="mini-swe-agent+swerex_modal+modal_gpu",
            critic_type="reward_model", is_true_reward_model=True, trajectory_text=c["text"], final_solution=c["final_solution"]))
    train_raw = {k: [r["audit_scores"][k] for r in rows if r["split"] == "train" and not r["excluded"]] for k in PANEL}
    fits = fit_rank_norms(train_raw)
    for r in rows: r["audit_scores_rank_norm"], r["u"] = apply_panel_u(r["audit_scores"], fits)
    (out / "replication_scored.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    pop = [r for r in rows if not r["excluded"] and r["visible_pass"]]
    print(f"[rows] {len(rows)} rows; population {len(pop)}; silent failures {sum(1 for r in pop if r['g'] == 0)}; repos {len({r['repo'] for r in pop})}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["generate", "bundle", "select", "rows"], required=True)
    ap.add_argument("--instances-file", type=Path, default=ROOT / "configs/replication_instances_multilingual.txt")
    ap.add_argument("--out", type=Path, default=ROOT / "data/replication_multilingual")
    ap.add_argument("--mini-config", type=Path, default=ROOT / "configs/mini_swe_openai_modal.yaml")
    ap.add_argument("--base-config", default="swebench.yaml",
                    help="swebench.yaml for tool-call model classes; swebench_backticks.yaml for *_textbased (e.g. DeepSeek)")
    ap.add_argument("--anchor", type=Path, default=ROOT / "configs/anchor_free_rm4_n16.yaml")
    ap.add_argument("--k", type=int, default=4); ap.add_argument("--workers", type=int, default=8); ap.add_argument("--step-limit", type=int, default=40)
    ap.add_argument("--cost-cap", type=float, default=15.0); ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--scores-dir", type=Path, default=None, help="modal/rescore.py prefix4k output dir")
    ap.add_argument("--gold-report", type=Path, default=None)
    a = ap.parse_args(); a.out.mkdir(parents=True, exist_ok=True)
    ids = [l.strip() for l in a.instances_file.read_text().splitlines() if l.strip() and not l.startswith("#")]
    if a.limit: ids = ids[: a.limit]
    cfg = yaml.safe_load(a.anchor.read_text()); seed = int(cfg["split"]["split_seed"]); tf = float(cfg["split"]["train_frac"])
    if a.stage == "generate": gen(ids, a.k, a.out, a.mini_config, a.workers, a.step_limit, a.cost_cap, base_config=a.base_config)
    elif a.stage == "bundle": bundle(ids, a.k, a.out, seed, tf)
    elif a.stage == "select": select(a.out, a.scores_dir or a.out / "modal/prefix4k")
    elif a.stage == "rows": make_rows(a.out, a.scores_dir or a.out / "modal/prefix4k", a.gold_report or ROOT / "data/gold_modal/replication_multilingual/report.json", seed)


if __name__ == "__main__":
    main()
