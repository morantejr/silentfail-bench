#!/bin/bash
# Algoverse L4 (24 GB): the whole Amendment 2 + 3 program in one command.
#     bash silentfail_l4_package/l4/run_l4.sh
# Results: silentfail_l4_package/l4_results.tar.gz  (download from the Jupyter file browser)
set -u
cd "$(dirname "$0")/.."
OUT=l4_out; mkdir -p $OUT
log(){ echo "[$(date +%T)] $*" | tee -a $OUT/run.log; }

# ---- 0. find a Python that has torch (the Jupyter kernel's, not the shell's) ----
PY=""
for cand in "$(jupyter kernelspec list --json 2>/dev/null | python3 -c 'import json,sys; d=json.load(sys.stdin)["kernelspecs"]; k=d.get("python3") or list(d.values())[0]; print(k["spec"]["argv"][0])' 2>/dev/null)" \
            /opt/conda/bin/python /opt/conda/envs/*/bin/python /usr/local/bin/python3 python3 python; do
  [ -n "$cand" ] && command -v "$cand" >/dev/null 2>&1 && "$cand" -c "import torch" >/dev/null 2>&1 && { PY="$cand"; break; }
done
if [ -z "$PY" ]; then
  log "no Python with torch found; installing torch into python3 user site (CUDA wheel, ~2.5 GB)"
  PY=python3; $PY -m pip install --user -q torch --index-url https://download.pytorch.org/whl/cu121 2>&1 | tail -2 | tee -a $OUT/run.log
fi
log "python: $PY ($($PY -c 'import sys;print(sys.version.split()[0])'))  torch: $($PY -c 'import torch;print(torch.__version__, "cuda", torch.cuda.is_available())')"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | tee -a $OUT/run.log

# ---- 1. deps (errors VISIBLE; retry with --user if the site is read-only) ----
pipi(){ $PY -m pip install -q -U "$@" 2>&1 | grep -viE "warning|already satisfied|^\s*$" | tail -3; }
log "deps: core"; pipi transformers accelerate sentencepiece bitsandbytes evalplus || pipi --user transformers accelerate sentencepiece bitsandbytes evalplus
$PY -c "import transformers, accelerate, bitsandbytes, evalplus; print('  core ok:', transformers.__version__)" 2>&1 | tee -a $OUT/run.log
log "deps: acecoder (git, then zip fallback)"
{ pipi "git+https://github.com/TIGER-AI-Lab/AceCoder.git" || pipi "https://github.com/TIGER-AI-Lab/AceCoder/archive/refs/heads/main.zip" || pipi --user "https://github.com/TIGER-AI-Lab/AceCoder/archive/refs/heads/main.zip"; } | tee -a $OUT/run.log
$PY - <<'PY' 2>&1 | tee -a $OUT/run.log
import importlib.util, glob, sys
spec = importlib.util.find_spec("acecoder")
if not spec: print("  acecoder NOT importable -> AceCodeRM will be skipped"); sys.exit(0)
import os; d = os.path.dirname(spec.origin); f = os.path.join(d, "rm_utils.py")
s = open(f).read(); s2 = s.replace("index=(attention_mask.sum(dim=-1, keepdim=True) - 1)", "index=(attention_mask.sum(dim=-1, keepdim=True) - 1).to(value.device)")
open(f, "w").write(s2); print("  acecoder ok at", d, "| gather patched:", s != s2)
PY

B=data/swebench_pilot50/rescoring_bundle.jsonl
log "1/4 smoke: 2 candidates, every critic incl. 32B"
$PY scripts/gpu_rescore.py --bundle $B --out $OUT/smoke --limit 2 --include-32b 2>&1 | grep -E "^\[|OOM|Error" | tee -a $OUT/run.log
log "2/4 Amendment 2 at TRUE 16k: all 164 candidates, 7 critics + 32B"
$PY scripts/gpu_rescore.py --bundle $B --out $OUT/gpu_scores_16k --include-32b 2>&1 | grep -E "^\[|OOM|/164|Error" | tee -a $OUT/run.log
log "3/4 Amendment 3 windows: 34 committed, all critics incl. AceCodeRM"
$PY scripts/gpu_rescore_windows.py --bundle $B --out $OUT/gpu_windows --only $($PY -c "import sys;sys.path.insert(0,'scripts');from gpu_rescore import CRITICS;print(' '.join(r for r,_ in CRITICS))") 2>&1 | grep -E "^\[|/34|Error" | tee -a $OUT/run.log
log "4/4 package"
tar czf l4_results.tar.gz $OUT && log "DONE -> download silentfail_l4_package/l4_results.tar.gz from the Jupyter file browser"
