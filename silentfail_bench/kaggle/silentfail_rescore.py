# Kaggle script kernel: Amendment-2 rescoring on 2xT4. Writes gpu_scores/ to /kaggle/working.
import subprocess, sys, os, shutil, glob
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-U", "transformers", "accelerate", "sentencepiece", "bitsandbytes"], check=True)
# acecoder is not on PyPI; install from source, NON-fatal: if it fails only AceCodeRM is skipped.
r = subprocess.run([sys.executable, "-m", "pip", "install", "-q", "git+https://github.com/TIGER-AI-Lab/AceCoder.git"], check=False)
print("acecoder install:", "ok" if r.returncode == 0 else f"FAILED (rc={r.returncode}); AceCodeRM will be skipped", flush=True)
r2 = subprocess.run([sys.executable, "-m", "pip", "install", "-q", "evalplus"], check=False)   # acecoder/__init__ imports it
print("evalplus install:", "ok" if r2.returncode == 0 else "FAILED (rm_utils will be loaded by path instead)", flush=True)
# rm_utils gathers the last-token reward with an index on the input device; with the model
# split across two GPUs the value head is on the other one. Move the index to value.device.
import glob as _g
for f in _g.glob("/usr/local/lib/python3*/dist-packages/acecoder/rm_utils.py"):
    src = open(f).read()
    src2 = src.replace("index=(attention_mask.sum(dim=-1, keepdim=True) - 1)", "index=(attention_mask.sum(dim=-1, keepdim=True) - 1).to(value.device)")
    open(f, "w").write(src2); print("patched rm_utils gather device:", src != src2, flush=True)
subprocess.run([sys.executable, "-m", "pip", "show", "acecoder"], check=False)
subprocess.run([sys.executable, "-c", "import traceback\ntry:\n    import acecoder.rm_utils as m; print('acecoder.rm_utils OK:', m.__file__)\nexcept Exception:\n    traceback.print_exc()"], check=False)
print("input tree:", flush=True)
for root, dirs, files in os.walk("/kaggle/input"):
    for f in files: print("  ", os.path.join(root, f), flush=True)
scripts = glob.glob("/kaggle/input/**/gpu_rescore.py", recursive=True)
bundles = glob.glob("/kaggle/input/**/rescoring_bundle.jsonl", recursive=True)
if not scripts or not bundles:
    raise SystemExit(f"inputs not found: scripts={scripts} bundles={bundles}")
SCRIPT, BUNDLE = scripts[0], bundles[0]
print("using", SCRIPT, BUNDLE, flush=True)
os.makedirs("/kaggle/working/gpu_scores", exist_ok=True)
print("smoke test (2 candidates, every critic):", flush=True)
subprocess.run([sys.executable, SCRIPT, "--bundle", BUNDLE, "--out", "/kaggle/working/gpu_scores_smoke", "--limit", "2", "--only", "TIGER-Lab/AceCodeRM-7B"], check=False)
print("full run:", flush=True)
subprocess.run([sys.executable, SCRIPT, "--bundle", BUNDLE, "--out", "/kaggle/working/gpu_scores", "--only", "TIGER-Lab/AceCodeRM-7B"], check=False)
shutil.make_archive("/kaggle/working/gpu_scores", "zip", "/kaggle/working", "gpu_scores")
print("done:", os.listdir("/kaggle/working"), flush=True)
