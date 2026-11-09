# Running the Amendment-2 rescoring on Kaggle (free, 2×T4) — or Colab

**Upload once.** Kaggle → *Datasets* → *New Dataset* → upload
`silentfail_bench/data/swebench_pilot50/rescoring_bundle.jsonl` (12 MB) and
`silentfail_bench/scripts/gpu_rescore.py`. Name it `silentfail-rescore`.

**Notebook settings.** Accelerator: **GPU T4 ×2**. Internet: **On** (model
downloads). Persistence: off is fine — you download the results.

**Cells, in order.**

```python
# 1 — deps (~3 min)
!pip install -q -U transformers accelerate sentencepiece bitsandbytes
```
```python
# 2 — smoke test: 2 candidates through every critic (~10 min incl. downloads).
#     Any critic that fails to load or self-test is skipped and printed; the rest proceed.
!python /kaggle/input/silentfail-rescore/gpu_rescore.py \
    --bundle /kaggle/input/silentfail-rescore/rescoring_bundle.jsonl \
    --out /kaggle/working/gpu_scores_smoke --limit 2
```
```python
# 3 — full run (~2 h on 2×T4). Each critic writes its JSON the moment it finishes.
!python /kaggle/input/silentfail-rescore/gpu_rescore.py \
    --bundle /kaggle/input/silentfail-rescore/rescoring_bundle.jsonl \
    --out /kaggle/working/gpu_scores
```
```python
# 4 — bundle for download
!cd /kaggle/working && zip -r gpu_scores.zip gpu_scores && ls -la gpu_scores.zip
```
Download `gpu_scores.zip` from the Output panel. Send it to me; the readout
runs locally in seconds:
```bash
python3 scripts/prereg_readout_amendment2.py --gpu-scores gpu_scores/
```

**Colab instead?** Same cells, but Runtime → T4, and add `--load-8bit` to
steps 2 and 3 (one 16 GB card can't hold an 8B model in fp16). Download from
the Files panel. Expect disconnects; the script skips critics already done,
so just re-run step 3.

**If a session dies mid-run:** re-run step 3 — finished critics are skipped.
