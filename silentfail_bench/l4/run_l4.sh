#!/bin/bash
# Algoverse L4: root disk is 100% full, but /opt/dlami/nvme has ~2.9 TB. Build EVERYTHING there.
#     bash silentfail_l4_package/l4/run_l4.sh
# Results: ~/silentfail_l4_results.tar.gz (small: JSON only) -> download from the Jupyter file browser.
set -u
PKG="$(cd "$(dirname "$0")/.." && pwd)"
W=/opt/dlami/nvme/silentfail_${USER:-user}; mkdir -p $W/tmp $W/pip $W/hf $W/out
export TMPDIR=$W/tmp PIP_CACHE_DIR=$W/pip HF_HOME=$W/hf HF_HUB_CACHE=$W/hf/hub TRANSFORMERS_CACHE=$W/hf/hub
OUT=$W/out; log(){ echo "[$(date +%T)] $*" | tee -a $OUT/run.log; }
log "workdir $W  ($(df -h $W | tail -1 | awk '{print $4}') free)"

# ---- 0. private venv on the NVMe (root disk has ~900 MB free; torch alone is 2.5 GB) ----
if [ ! -x $W/venv/bin/python ]; then
  log "creating venv on NVMe"; python3 -m venv $W/venv 2>&1 | tail -2 | tee -a $OUT/run.log
fi
PY=$W/venv/bin/python; $PY -m pip install -q -U pip 2>&1 | tail -1
pipi(){ $PY -m pip install -q -U "$@" 2>&1 | grep -viE "warning|already satisfied|^\s*$" | tail -3; }
log "deps: torch (CUDA wheel, ~2.5 GB)"; pipi torch | tee -a $OUT/run.log
$PY -c "import torch;print('  torch', torch.__version__, '| cuda', torch.cuda.is_available(), '|', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NO GPU')" 2>&1 | tee -a $OUT/run.log
log "deps: core"; pipi transformers accelerate sentencepiece bitsandbytes evalplus | tee -a $OUT/run.log
$PY -c "import transformers, accelerate, bitsandbytes, evalplus; print('  core ok:', transformers.__version__)" 2>&1 | tee -a $OUT/run.log
log "deps: acecoder"; { pipi "https://github.com/TIGER-AI-Lab/AceCoder/archive/refs/heads/main.zip" || pipi "git+https://github.com/TIGER-AI-Lab/AceCoder.git"; } | tee -a $OUT/run.log
$PY - <<'PY' 2>&1 | tee -a $OUT/run.log
import importlib.util, os, sys
spec = importlib.util.find_spec("acecoder")
if not spec: print("  acecoder NOT importable -> AceCodeRM will be skipped"); sys.exit(0)
f = os.path.join(os.path.dirname(spec.origin), "rm_utils.py"); s = open(f).read()
s2 = s.replace("index=(attention_mask.sum(dim=-1, keepdim=True) - 1)", "index=(attention_mask.sum(dim=-1, keepdim=True) - 1).to(value.device)")
open(f, "w").write(s2); print("  acecoder ok | gather patched:", s != s2)
PY
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | tee -a $OUT/run.log

cd "$PKG"; B=data/swebench_pilot50/rescoring_bundle.jsonl
log "1/4 smoke: 2 candidates, every critic incl. 32B"
$PY scripts/gpu_rescore.py --bundle $B --out $OUT/smoke --limit 2 --include-32b 2>&1 | grep -E "^\[|OOM|Error" | tee -a $OUT/run.log
log "2/4 Amendment 2 at TRUE 16k: all 164 candidates, 7 critics + 32B"
$PY scripts/gpu_rescore.py --bundle $B --out $OUT/gpu_scores_16k --include-32b 2>&1 | grep -E "^\[|OOM|/164|Error" | tee -a $OUT/run.log
log "3/4 Amendment 3 windows: 34 committed, all critics incl. AceCodeRM"
$PY scripts/gpu_rescore_windows.py --bundle $B --out $OUT/gpu_windows --only $($PY -c "import sys;sys.path.insert(0,'scripts');from gpu_rescore import CRITICS;print(' '.join(r for r,_ in CRITICS))") 2>&1 | grep -E "^\[|/34|Error" | tee -a $OUT/run.log
log "4/4 package (JSON only, to home for download)"
tar czf ~/silentfail_l4_results.tar.gz -C $W out --exclude='*.safetensors' && log "DONE -> download ~/silentfail_l4_results.tar.gz from the Jupyter file browser ($(du -h ~/silentfail_l4_results.tar.gz | cut -f1))"
