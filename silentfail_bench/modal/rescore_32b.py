"""AceCodeRM-32B on Modal (A100-80GB, fp16) — the one Amendment-2 critic that never fit a 24 GB card.

Pre-registered in the Amendment 2/3 L4 addendum (A2-P1, same prediction); this runs it in fp16
instead of the declared 4-bit, which is the stronger, not weaker, version of the critic.

    cd silentfail_bench
    .venv-swebench/bin/python -m modal run modal/rescore_32b.py --limit 2      # smoke (~cents)
    .venv-swebench/bin/python -m modal run modal/rescore_32b.py                # all 164 + windows
    .venv-swebench/bin/python -m modal volume get silentfail-out / data/swebench_pilot50/modal
"""
from __future__ import annotations

import modal

app = modal.App("silentfail-32b")
hf_cache = modal.Volume.from_name("silentfail-hf-cache", create_if_missing=True)
out_vol = modal.Volume.from_name("silentfail-out", create_if_missing=True)
image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("torch", "transformers", "accelerate", "sentencepiece", "bitsandbytes", "evalplus", "numpy")
    .pip_install("https://github.com/TIGER-AI-Lab/AceCoder/archive/refs/heads/main.zip")
    .add_local_dir("scripts", remote_path="/root/scripts")
    .add_local_file("data/swebench_pilot50/rescoring_bundle.jsonl", remote_path="/root/data/rescoring_bundle.jsonl")
)
CRITIC = [("TIGER-Lab/AceCodeRM-32B", "acecode")]


@app.function(image=image, gpu="A100-80GB", timeout=4 * 3600, volumes={"/hf": hf_cache, "/out": out_vol})
def run(limit: int = 0, windows: bool = True) -> dict:
    import importlib.util, os, subprocess, sys, time
    os.environ.update(HF_HOME="/hf", HF_HUB_CACHE="/hf/hub")
    # AceCoder's value-head gather assumes one device; same one-line patch as on the L4
    spec = importlib.util.find_spec("acecoder"); f = os.path.join(os.path.dirname(spec.origin), "rm_utils.py"); s = open(f).read()
    s2 = s.replace("index=(attention_mask.sum(dim=-1, keepdim=True) - 1)", "index=(attention_mask.sum(dim=-1, keepdim=True) - 1).to(value.device)")
    open(f, "w").write(s2)
    print(subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"], capture_output=True, text=True).stdout.strip(), flush=True)
    sys.path.insert(0, "/root/scripts")
    import gpu_rescore as gr
    gr.CRITICS_32B = CRITIC                      # fp16 on 80 GB instead of nf4 on 24 GB
    t0 = time.time()
    sys.argv = ["gpu_rescore.py", "--bundle", "/root/data/rescoring_bundle.jsonl", "--out", "/out/gpu_scores_32b", "--include-32b", "--only", CRITIC[0][0]] + (["--limit", str(limit)] if limit else [])
    gr.main(); out_vol.commit()
    if windows:
        import gpu_rescore_windows as gw
        gw.CRITICS = CRITIC
        sys.argv = ["gpu_rescore_windows.py", "--bundle", "/root/data/rescoring_bundle.jsonl", "--out", "/out/gpu_windows_32b", "--only", CRITIC[0][0]] + (["--limit", str(limit)] if limit else [])
        gw.main(); out_vol.commit()
    return {"elapsed_s": round(time.time() - t0), "files": sorted(os.listdir("/out/gpu_scores_32b")) + (sorted(os.listdir("/out/gpu_windows_32b")) if windows else [])}


@app.local_entrypoint()
def main(limit: int = 0, windows: bool = True):
    print(run.remote(limit=limit, windows=windows))
