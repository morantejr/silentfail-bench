#!/usr/bin/env python3
"""Amendment 3: structural pruning + overlapping windows, every critic scores every window.

Runs beside gpu_rescore.py (same directory; imports its loaders). Audit-only:
scores ONLY the primary committed candidate per instance (34 rows), all
windows, writes windows_gpu_<slug>.json per critic immediately.

    python gpu_rescore_windows.py --bundle rescoring_bundle.jsonl --out gpu_windows [--limit 2]
"""
from __future__ import annotations

import argparse, gc, json, re, time
from pathlib import Path

import torch
from gpu_rescore import CRITICS, load, score, slug

WIN_TOK, OVERLAP, MAX_WIN, PROB_TOK = 3_500, 0.25, 8, 800
PRUNE_ABOVE, KEEP = 24, 8


def prune(text: str) -> str:
    """Collapse long tool outputs; never touch diffs."""
    out, buf, in_diff = [], [], False
    def flush():
        nonlocal buf
        if len(buf) > PRUNE_ABOVE and not in_diff:
            out.extend(buf[:KEEP]); out.append(f"[... {len(buf) - 2*KEEP} lines omitted ...]"); out.extend(buf[-KEEP:])
        else:
            out.extend(buf)
        buf = []
    for line in text.splitlines():
        if line.startswith("## "):            # role boundary from traj_text()
            flush(); in_diff = False; out.append(line); continue
        if line.startswith("diff --git"): in_diff = True
        buf.append(line)
    flush()
    return "\n".join(out)


def windows(tok, problem: str, text: str):
    ids = tok(text, add_special_tokens=False)["input_ids"]
    step = int(WIN_TOK * (1 - OVERLAP))
    starts = list(range(0, max(1, len(ids) - WIN_TOK + 1), step))
    if not starts or starts[-1] + WIN_TOK < len(ids): starts.append(max(0, len(ids) - WIN_TOK))
    if len(starts) > MAX_WIN:
        idx = [round(i * (len(starts) - 1) / (MAX_WIN - 1)) for i in range(MAX_WIN)]
        starts = [starts[i] for i in idx]
    prob_ids = tok(problem, add_special_tokens=False)["input_ids"][:PROB_TOK]
    prob = tok.decode(prob_ids)
    for s in starts:
        chunk = tok.decode(ids[s:s + WIN_TOK])
        msgs = [{"role": "user", "content": prob}, {"role": "assistant", "content": chunk}]
        try: yield tok.apply_chat_template(msgs, tokenize=False), s, len(ids)
        except Exception: yield f"User: {prob}\n\nAssistant: {chunk}", s, len(ids)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bundle", type=Path, required=True); ap.add_argument("--out", type=Path, default=Path("gpu_windows"))
    ap.add_argument("--load-8bit", action="store_true"); ap.add_argument("--limit", type=int, default=None); ap.add_argument("--only", nargs="*", default=None)
    a = ap.parse_args(); a.out.mkdir(parents=True, exist_ok=True)
    cands = [json.loads(l) for l in a.bundle.read_text().splitlines() if l.strip()]
    committed = [c for c in cands if c["candidate"] == c["committed_index_primary"] and not c["excluded"]]
    if a.limit: committed = committed[: a.limit]
    print(f"[bundle] {len(committed)} primary-committed rows | GPUs: {torch.cuda.device_count()}", flush=True)
    for repo, kind in CRITICS:
        if a.only and repo not in a.only: continue
        if kind != "seqcls" and not a.only: continue      # AceCodeRM windows only when explicitly requested
        dest = a.out / f"windows_gpu_{slug(repo)}.json"
        if dest.exists(): print(f"[skip] {repo}", flush=True); continue
        t0 = time.time()
        try: tok, model = load(repo, kind, a.load_8bit)
        except Exception as e: print(f"[FAIL load] {repo}: {type(e).__name__}: {str(e)[:200]}", flush=True); continue
        dev = next(model.parameters()).device
        res = {}
        for k, c in enumerate(committed):
            text = prune(c["text"]); ws = []
            for s, start, total in windows(tok, c["problem_statement"], text):
                enc = tok(s, return_tensors="pt", truncation=True, max_length=WIN_TOK + PROB_TOK + 256).to(dev)
                try: v = score(model, kind, enc)
                except torch.cuda.OutOfMemoryError: v = float("nan"); torch.cuda.empty_cache()
                ws.append({"start": start, "score": v})
            res[c["instance_id"]] = {"n_windows": len(ws), "n_tokens_pruned": total if ws else None, "windows": ws}
            if (k + 1) % 10 == 0: print(f"  {repo.split('/')[-1]}: {k+1}/{len(committed)} ({(time.time()-t0)/(k+1):.0f}s/row)", flush=True)
        dest.write_text(json.dumps({"critic": repo, "win_tok": WIN_TOK, "overlap": OVERLAP, "max_win": MAX_WIN, "prune_above": PRUNE_ABOVE, "elapsed_s": round(time.time()-t0), "rows": res}, indent=1))
        print(f"[write] {dest} ({len(res)} rows, {time.time()-t0:.0f}s)", flush=True)
        del model, tok; gc.collect(); torch.cuda.empty_cache()
    print("[done]", flush=True)


if __name__ == "__main__":
    main()
