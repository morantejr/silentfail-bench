#!/bin/bash
# Amendment 4b/4d on the L4, after run_l4.sh has finished (reuses its venv on the NVMe).
#     bash silentfail_l4_package/l4/run_l4_stage5.sh
# Results: ~/silentfail_l4_results_stage5.tar.gz
set -u
PKG="$(cd "$(dirname "$0")/.." && pwd)"
W=/opt/dlami/nvme/silentfail_${USER:-user}
export TMPDIR=$W/tmp PIP_CACHE_DIR=$W/pip HF_HOME=$W/hf HF_HUB_CACHE=$W/hf/hub TRANSFORMERS_CACHE=$W/hf/hub
OUT=$W/out; PY=$W/venv/bin/python; log(){ echo "[$(date +%T)] $*" | tee -a $OUT/run.log; }
[ -x $PY ] || { echo "venv missing at $W/venv - run run_l4.sh first"; exit 1; }
cd "$PKG"; B=data/swebench_pilot50/rescoring_bundle.jsonl
ALL=$($PY -c "import sys;sys.path.insert(0,'scripts');from gpu_rescore import CRITICS;print(' '.join(r for r,_ in CRITICS))")
log "5a/5c Amendment 4d: 8k ladder point, 7 critics, all 164 candidates"
$PY scripts/gpu_rescore.py --bundle $B --out $OUT/gpu_scores_8k --max-tokens 8000 2>&1 | grep -E "^\[|/164|Error" | tee -a $OUT/run.log
log "5b/5c Amendment 4b: window test-retest, 7 critics, 34 committed rows x 3 variants"
$PY scripts/stability_windows.py --bundle $B --out $OUT/gpu_stability --only $ALL 2>&1 | grep -E "^\[|/34|Error|oom" | tee -a $OUT/run.log
log "5c/5c package"
tar czf ~/silentfail_l4_results_stage5.tar.gz -C $W out --exclude='*.safetensors' && log "DONE -> download ~/silentfail_l4_results_stage5.tar.gz ($(du -h ~/silentfail_l4_results_stage5.tar.gz | cut -f1))"
