#!/bin/bash
# Algoverse L4 (24 GB): the whole Amendment 2 + 3 program in one command.
# Run from a Jupyter terminal after uploading the repo zip (or cloning):
#     bash silentfail_bench/l4/run_l4.sh
# Results land in silentfail_bench/l4_results.zip -> download from the Jupyter file browser.
set -u
cd "$(dirname "$0")/.."
OUT=l4_out; mkdir -p $OUT
log(){ echo "[$(date +%T)] $*" | tee -a $OUT/run.log; }
log "deps"
pip install -q -U transformers accelerate sentencepiece bitsandbytes evalplus >/dev/null 2>&1 && log "  core ok"
pip install -q git+https://github.com/TIGER-AI-Lab/AceCoder.git >/dev/null 2>&1 && log "  acecoder ok" || log "  acecoder FAILED (AceCodeRM will be skipped)"
python - <<'PY'
import glob
for f in glob.glob("/usr/**/site-packages/acecoder/rm_utils.py", recursive=True)+glob.glob("/opt/**/site-packages/acecoder/rm_utils.py", recursive=True)+glob.glob("/home/**/site-packages/acecoder/rm_utils.py", recursive=True):
    s=open(f).read(); s2=s.replace("index=(attention_mask.sum(dim=-1, keepdim=True) - 1)","index=(attention_mask.sum(dim=-1, keepdim=True) - 1).to(value.device)")
    open(f,"w").write(s2); print("patched", f, s!=s2)
PY
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | tee -a $OUT/run.log
B=data/swebench_pilot50/rescoring_bundle.jsonl
log "1/4 smoke: 2 candidates, every critic incl. 32B"
python scripts/gpu_rescore.py --bundle $B --out $OUT/smoke --limit 2 --include-32b 2>&1 | grep -E "^\[|OOM" | tee -a $OUT/run.log
log "2/4 Amendment 2 at TRUE 16k: all 164 candidates, 7 critics + 32B"
python scripts/gpu_rescore.py --bundle $B --out $OUT/gpu_scores_16k --include-32b 2>&1 | grep -E "^\[|OOM|/164" | tee -a $OUT/run.log
log "3/4 Amendment 3 windows: 34 committed, all critics incl. AceCodeRM"
python scripts/gpu_rescore_windows.py --bundle $B --out $OUT/gpu_windows --only $(python -c "import sys;sys.path.insert(0,'scripts');from gpu_rescore import CRITICS;print(' '.join(r for r,_ in CRITICS))") 2>&1 | grep -E "^\[|/34" | tee -a $OUT/run.log
log "4/4 zip"
zip -qr l4_results.zip $OUT && log "DONE -> download silentfail_bench/l4_results.zip from the Jupyter file browser"
