# Kaggle script kernel: Amendment-2 rescoring on 2xT4. Writes gpu_scores/ to /kaggle/working.
import subprocess, sys, os, shutil
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-U", "transformers", "accelerate", "sentencepiece", "bitsandbytes"], check=True)
D = "/kaggle/input/silentfail-rescore"
print("smoke test (2 candidates, every critic):", flush=True)
subprocess.run([sys.executable, f"{D}/gpu_rescore.py", "--bundle", f"{D}/rescoring_bundle.jsonl", "--out", "/kaggle/working/gpu_scores_smoke", "--limit", "2"], check=False)
print("full run:", flush=True)
subprocess.run([sys.executable, f"{D}/gpu_rescore.py", "--bundle", f"{D}/rescoring_bundle.jsonl", "--out", "/kaggle/working/gpu_scores"], check=False)
shutil.make_archive("/kaggle/working/gpu_scores", "zip", "/kaggle/working", "gpu_scores")
print("done:", os.listdir("/kaggle/working"), flush=True)
