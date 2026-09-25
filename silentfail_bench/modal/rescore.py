"""Generic reward-model rescoring on Modal (any critic subset, any length/mode, windows), bundle read from a Volume.

    modal volume put silentfail-bundles data/<run>/rescoring_bundle.jsonl /<run>/rescoring_bundle.jsonl
    python -m modal run modal/rescore.py --run <run> --out prefix4k --max-tokens 4000 --mode prefix
    python -m modal run modal/rescore.py --run <run> --out full16k
    python -m modal run modal/rescore.py --run <run> --out windows --windows
    modal volume get silentfail-out /<run> data/<run>/modal
Outputs land in Volume silentfail-out under /<run>/<out>/.
"""
from __future__ import annotations

import modal

app = modal.App("silentfail-rescore")
hf_cache = modal.Volume.from_name("silentfail-hf-cache", create_if_missing=True)
out_vol = modal.Volume.from_name("silentfail-out", create_if_missing=True)
bundles = modal.Volume.from_name("silentfail-bundles", create_if_missing=True)
image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("torch", "transformers", "accelerate", "sentencepiece", "bitsandbytes", "evalplus", "numpy")
    .pip_install("https://github.com/TIGER-AI-Lab/AceCoder/archive/refs/heads/main.zip")
    .add_local_dir("scripts", remote_path="/root/scripts")
)


@app.function(image=image, gpu="L4", timeout=6 * 3600, volumes={"/hf": hf_cache, "/out": out_vol, "/bundles": bundles})
def run(run_name: str, out: str, max_tokens: int, mode: str, windows: bool, only: list[str] | None, limit: int) -> dict:
    import importlib.util, os, subprocess, sys, time
    os.environ.update(HF_HOME="/hf", HF_HUB_CACHE="/hf/hub")
    spec = importlib.util.find_spec("acecoder")
    if spec:
        f = os.path.join(os.path.dirname(spec.origin), "rm_utils.py"); s = open(f).read()
        open(f, "w").write(s.replace("index=(attention_mask.sum(dim=-1, keepdim=True) - 1)", "index=(attention_mask.sum(dim=-1, keepdim=True) - 1).to(value.device)"))
    print(subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"], capture_output=True, text=True).stdout.strip(), flush=True)
    sys.path.insert(0, "/root/scripts")
    bundle = f"/bundles/{run_name}/rescoring_bundle.jsonl"; dest = f"/out/{run_name}/{out}"; t0 = time.time()
    extra = (["--only", *only] if only else []) + (["--limit", str(limit)] if limit else [])
    if windows:
        import gpu_rescore_windows as gw
        sys.argv = ["gpu_rescore_windows.py", "--bundle", bundle, "--out", dest] + (extra or ["--only", *[r for r, _ in gw.CRITICS]])
        gw.main()
    else:
        import gpu_rescore as gr
        sys.argv = ["gpu_rescore.py", "--bundle", bundle, "--out", dest, "--max-tokens", str(max_tokens), "--mode", mode] + extra
        gr.main()
    out_vol.commit()
    return {"elapsed_s": round(time.time() - t0), "files": sorted(os.listdir(dest))}


@app.local_entrypoint()
def main(run: str, out: str, max_tokens: int = 16000, mode: str = "headtail", windows: bool = False, only: str = "", limit: int = 0):
    print(run.remote(run_name=run, out=out, max_tokens=max_tokens, mode=mode, windows=windows, only=only.split() or None, limit=limit))
