#!/usr/bin/env python3
"""Amendment 2 GPU rescoring. Runs unchanged on Kaggle (2xT4), Colab (1xT4, --load-8bit),
an L4, or AWS. Needs only the rescoring bundle; no repo, no dataset downloads, no tokens.

    pip install -q transformers accelerate sentencepiece bitsandbytes   # once, in the notebook
    python gpu_rescore.py --bundle rescoring_bundle.jsonl --out gpu_scores [--load-8bit] [--limit 4]

Per critic it scores ALL candidates on the FULL trajectory (problem statement +
text; if > MAX_TOKENS it keeps head 4k + tail 12k and flags the row), writes
scores_gpu_<slug>.json IMMEDIATELY, then frees the model. A 2-candidate
self-test runs first so an incompatible model fails in a minute, not an hour.
"""
from __future__ import annotations

import argparse, gc, json, re, time
from pathlib import Path

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

MAX_TOKENS, HEAD, TAIL = 16_000, 4_000, 12_000

CRITICS = [  # order: cheap standard heads first, then the two Amendment-2 critics
    ("Ray2333/GRM-gemma2-2B-rewardmodel-ft",        "seqcls"),
    ("Ray2333/GRM-Llama3.2-3B-rewardmodel-ft",      "seqcls"),
    ("Skywork/Skywork-Reward-V2-Llama-3.2-1B",      "seqcls"),
    ("Skywork/Skywork-Reward-V2-Qwen3-1.7B",        "seqcls"),
    ("Skywork/Skywork-Reward-V2-Qwen3-4B",          "seqcls"),   # the primary's c_sel, now at full length
    ("Skywork/Skywork-Reward-V2-Llama-3.1-8B",      "seqcls"),   # the proposal's named critic
    ("TIGER-Lab/AceCodeRM-7B",                      "acecode"),  # code-trained, custom head
]


def slug(repo: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", repo.lower()).strip("_")


def build_input(tok, problem: str, text: str, max_tokens: int = MAX_TOKENS):
    """Chat-formatted (user=problem, assistant=trajectory); long tails keep head+tail, flagged."""
    ids = tok(text, add_special_tokens=False)["input_ids"]
    truncated = len(ids) > max_tokens
    if truncated:
        head, tail = max_tokens // 4, max_tokens - max_tokens // 4
        text = tok.decode(ids[:head]) + "\n\n[... middle of trajectory omitted ...]\n\n" + tok.decode(ids[-tail:])
    msgs = [{"role": "user", "content": problem}, {"role": "assistant", "content": text}]
    try:
        s = tok.apply_chat_template(msgs, tokenize=False)
    except Exception:
        s = f"User: {problem}\n\nAssistant: {text}"
    return s, truncated


def load(repo: str, kind: str, load_8bit: bool):
    tok = AutoTokenizer.from_pretrained(repo, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    # sdpa = PyTorch memory-efficient attention: O(L) memory instead of the O(L^2)
    # map eager attention materializes (16k tokens -> 8-32 GiB per model on a T4).
    kw = dict(trust_remote_code=True, device_map="auto", attn_implementation="sdpa")
    if load_8bit:
        from transformers import BitsAndBytesConfig
        kw["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
    else:
        kw["dtype"] = torch.float16
    if kind == "seqcls":
        model = AutoModelForSequenceClassification.from_pretrained(repo, num_labels=1, **kw).eval()
    else:
        # AceCodeRM's value head is not in transformers' auto-map: AutoModel loads the
        # bare Qwen2Model and silently drops v_head. Use the authors' class.
        try:
            try:
                from acecoder.rm_utils import AceCodeRM   # bypass __init__ (it also imports eval_test_cases)
            except ImportError:
                from acecoder import AceCodeRM
        except Exception as e:
            raise RuntimeError(f"AceCodeRM import failed: {type(e).__name__}: {e}") from e
        model = AceCodeRM.from_pretrained(repo, **kw).eval()   # Qwen2ForCausalLM + ValueHead; forward() returns rm_scores
    if getattr(model.config, "pad_token_id", None) is None:
        model.config.pad_token_id = tok.pad_token_id
    return tok, model


@torch.no_grad()
def score(model, kind: str, enc) -> float:
    out = model(**enc)
    if kind == "seqcls":
        return float(out.logits[0].float().flatten()[0])
    if isinstance(out, torch.Tensor):               # AceCodeRM returns rm_scores directly
        return float(out.float().flatten()[0])
    # AceCodeRM (Qwen2ForCausalRM): probe the common reward-output conventions.
    for attr in ("end_scores", "rewards", "reward", "score", "scores", "values", "value", "logits"):
        v = getattr(out, attr, None)
        if v is None and isinstance(out, dict): v = out.get(attr)
        if v is not None:
            v = v.float()
            return float(v.flatten()[-1]) if attr == "logits" else float(v.flatten()[0])
    raise RuntimeError(f"cannot find a reward in output type {type(out)} with attrs {list(getattr(out, 'keys', lambda: [])())}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bundle", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=Path("gpu_scores"))
    ap.add_argument("--load-8bit", action="store_true", help="single 16 GB GPU (Colab T4)")
    ap.add_argument("--limit", type=int, default=None, help="smoke test on the first N candidates")
    ap.add_argument("--only", nargs="*", default=None, help="subset of critic repos")
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    cands = [json.loads(l) for l in a.bundle.read_text().splitlines() if l.strip()]
    if a.limit: cands = cands[: a.limit]
    print(f"[bundle] {len(cands)} candidates | GPUs: {torch.cuda.device_count()} | 8bit={a.load_8bit}", flush=True)

    for repo, kind in CRITICS:
        if a.only and repo not in a.only: continue
        dest = a.out / f"scores_gpu_{slug(repo)}.json"
        if dest.exists():
            print(f"[skip] {repo}: {dest.name} exists", flush=True); continue
        t0 = time.time()
        try:
            tok, model = load(repo, kind, a.load_8bit)
        except Exception as e:
            print(f"[FAIL load] {repo}: {type(e).__name__}: {str(e)[:200]}", flush=True); continue
        dev = next(model.parameters()).device
        # self-test: 2 candidates, must produce two finite, distinct-or-equal floats
        try:
            for c in cands[:2]:
                ok = False
                for cap in (MAX_TOKENS, 8_000, 4_000):
                    s, _ = build_input(tok, c["problem_statement"], c["text"], cap)
                    enc = tok(s, return_tensors="pt", truncation=True, max_length=cap + 512).to(dev)
                    try:
                        v = score(model, kind, enc); assert v == v; ok = True; break
                    except torch.cuda.OutOfMemoryError:
                        del enc; gc.collect(); torch.cuda.empty_cache()
                if not ok: raise RuntimeError("OOM even at 4k tokens")
            print(f"[ok] {repo} loaded in {time.time()-t0:.0f}s, self-test passed", flush=True)
        except Exception as e:
            print(f"[FAIL self-test] {repo}: {type(e).__name__}: {str(e)[:300]}", flush=True)
            del model; gc.collect(); torch.cuda.empty_cache(); continue
        res = {}
        t1 = time.time()
        for k, c in enumerate(cands):
            v, used, trunc, ntok = float("nan"), None, None, None
            for cap in (MAX_TOKENS, 8_000, 4_000):          # fall back on OOM, record the cap that worked
                s, trunc = build_input(tok, c["problem_statement"], c["text"], cap)
                enc = tok(s, return_tensors="pt", truncation=True, max_length=cap + 512).to(dev)
                try:
                    v = score(model, kind, enc); used = cap; ntok = int(enc["input_ids"].shape[1]); break
                except torch.cuda.OutOfMemoryError:
                    del enc; gc.collect(); torch.cuda.empty_cache()
            if used is None: print(f"  OOM at every cap: {c['instance_id']}#{c['candidate']}", flush=True)
            res[f"{c['instance_id']}#{c['candidate']}"] = {"score": v, "truncated": trunc, "n_tokens": ntok, "max_tokens_used": used}
            if (k + 1) % 20 == 0:
                print(f"  {repo.split('/')[-1]}: {k+1}/{len(cands)} ({(time.time()-t1)/(k+1):.1f}s each)", flush=True)
        dest.write_text(json.dumps({"critic": repo, "kind": kind, "max_tokens": MAX_TOKENS, "head": HEAD, "tail": TAIL,
                                    "load_8bit": a.load_8bit, "elapsed_s": round(time.time()-t0), "scores": res}, indent=1))
        print(f"[write] {dest} ({len(res)} scores, {time.time()-t0:.0f}s total)", flush=True)
        del model, tok; gc.collect(); torch.cuda.empty_cache()
    print("[done] download the gpu_scores/ folder", flush=True)


if __name__ == "__main__":
    main()
