#!/bin/zsh
# One command after a fresh ~/.kaggle/kaggle.json is in place.
set -e
K=~/Library/Python/3.9/bin/kaggle
S=$(dirname "$0")
echo "== auth =="; $K kernels list --mine --page-size 1 >/dev/null && echo "  ok"
echo "== dataset (private, 12 MB) =="; cd $S/dataset && $K datasets create -p . --dir-mode skip 2>&1 | tail -1
for i in {1..12}; do sleep 10; ST=$($K datasets status jomorante/silentfail-rescore 2>&1 | tail -1); echo "  $ST"; echo "$ST" | grep -qi "ready\|complete" && break; done
echo "== kernel (private, GPU T4x2, internet) =="; cd $S/kernel && $K kernels push -p . 2>&1 | tail -1
sleep 8; $K kernels status jomorante/silentfail-rescore-run 2>&1 | tail -1
