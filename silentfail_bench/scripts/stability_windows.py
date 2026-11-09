#!/usr/bin/env python3
"""Amendment 4b: test-retest stability of window scores under content-preserving perturbations.

For every Amendment-3 window of the 34 committed trajectories, score three inputs:
  orig   the window as Amendment 3 built it
  fmt    same tokens; role headers re-cased, pruning marker reworded, one neutral line prepended
  shift  the window moved by +128 tokens (or -128 / first-128-dropped at the ends)
Writes stability_<slug>.json per critic. Runs on CUDA (all critics) or Apple MPS (seqcls critics).
"""
from __future__ import annotations

import argparse, gc, json, re, sys, time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from gpu_rescore import CRITICS, load as load_cuda, score, slug  # noqa: E402
from gpu_rescore_windows import MAX_WIN, OVERLAP, PROB_TOK, WIN_TOK, prune  # noqa: E402

SHIFT = 128


def window_starts(n: int) -> list[int]:
    step = int(WIN_TOK * (1 - OVERLAP))
    starts = list(range(0, max(1, n - WIN_TOK + 1), step))
    if not starts or starts[-1] + WIN_TOK < n: starts.append(max(0, n - WIN_TOK))
    if len(starts) > MAX_WIN:
        idx = [round(i * (len(starts) - 1) / (MAX_WIN - 1)) for i in range(MAX_WIN)]
        starts = [starts[i] for i in idx]
    return starts


def shifted_start(s: int, n: int) -> int:
    for s2 in (s + SHIFT, s - SHIFT):
        if 0 <= s2 < n and s2 != s: return s2
    return s + SHIFT   # trajectory shorter than SHIFT tokens: degenerate, recorded


def perturb_fmt(chunk: str) -> str:
    s = re.sub(r"^## ([a-z])", lambda m: "## " + m.group(1).upper(), chunk, flags=re.M)
    s = re.sub(r"\[\.\.\. (\d+) lines omitted \.\.\.\]", r"[\1 lines elided]", s)
    return "(trajectory excerpt)\n" + s


def chat(tok, prob: str, chunk: str) -> str:
    msgs = [{"role": "user", "content": prob}, {"role": "assistant", "content": chunk}]
    try: return tok.apply_chat_template(msgs, tokenize=False)
    except Exception: return f"User: {prob}\n\nAssistant: {chunk}"


def load_any(repo: str, kind: str):
    if torch.cuda.is_available(): return load_cuda(repo, kind, False), "cuda"
    if kind != "seqcls": raise RuntimeError(f"{repo}: needs CUDA")
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(repo, trust_remote_code=True)
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    model = AutoModelForSequenceClassification.from_pretrained(repo, num_labels=1, trust_remote_code=True,
                                                               dtype=torch.float16 if dev != "cpu" else torch.float32).to(dev).eval()
    if getattr(model.config, "pad_token_id", None) is None: model.config.pad_token_id = tok.pad_token_id
    return (tok, model), dev


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bundle", type=Path, required=True); ap.add_argument("--out", type=Path, default=Path("gpu_stability"))
    ap.add_argument("--limit", type=int, default=None); ap.add_argument("--only", nargs="*", default=None)
    a = ap.parse_args(); a.out.mkdir(parents=True, exist_ok=True)
    cands = [json.loads(l) for l in a.bundle.read_text().splitlines() if l.strip()]
    committed = [c for c in cands if c["candidate"] == c["committed_index_primary"] and not c["excluded"]]
    if a.limit: committed = committed[: a.limit]
    print(f"[bundle] {len(committed)} primary-committed rows", flush=True)
    for repo, kind in CRITICS:
        if a.only and repo not in a.only: continue
        if kind != "seqcls" and not a.only: continue
        dest = a.out / f"stability_{slug(repo)}.json"
        if dest.exists(): print(f"[skip] {repo}", flush=True); continue
        t0 = time.time()
        try: (tok, model), dev = load_any(repo, kind)
        except Exception as e: print(f"[FAIL load] {repo}: {type(e).__name__}: {str(e)[:200]}", flush=True); continue
        print(f"[ok] {repo} on {dev}", flush=True)
        res = {}
        for k, c in enumerate(committed):
            ids = tok(prune(c["text"]), add_special_tokens=False)["input_ids"]; n = len(ids)
            prob = tok.decode(tok(c["problem_statement"], add_special_tokens=False)["input_ids"][:PROB_TOK])
            ws = []
            for s in window_starts(n):
                s2 = shifted_start(s, n)
                chunk = tok.decode(ids[s:s + WIN_TOK])
                variants = {"orig": chat(tok, prob, chunk), "fmt": chat(tok, prob, perturb_fmt(chunk)),
                            "shift": chat(tok, prob, tok.decode(ids[s2:s2 + WIN_TOK]))}
                out = {"start": s, "shift_start": s2}
                for name, text in variants.items():
                    enc = tok(text, return_tensors="pt", truncation=True, max_length=WIN_TOK + PROB_TOK + 256).to(dev)
                    try: out[name] = score(model, kind, enc)
                    except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
                        out[name] = float("nan"); print(f"  [oom/err] {c['instance_id']} s={s} {name}: {str(e)[:80]}", flush=True)
                        if dev == "cuda": torch.cuda.empty_cache()
                        elif dev == "mps": torch.mps.empty_cache()
                    del enc
                ws.append(out)
            res[c["instance_id"]] = {"n_tokens_pruned": n, "n_windows": len(ws), "windows": ws}
            print(f"  {repo.split('/')[-1]}: {k+1}/{len(committed)} ({len(ws)} win, {(time.time()-t0)/(k+1):.0f}s/row)", flush=True)
        dest.write_text(json.dumps({"critic": repo, "device": dev, "win_tok": WIN_TOK, "overlap": OVERLAP, "shift": SHIFT,
                                    "elapsed_s": round(time.time() - t0), "rows": res}, indent=1))
        print(f"[write] {dest} ({time.time()-t0:.0f}s)", flush=True)
        del model, tok; gc.collect()
        if dev == "cuda": torch.cuda.empty_cache()
        elif dev == "mps": torch.mps.empty_cache()
    print("[done]", flush=True)


if __name__ == "__main__":
    main()
