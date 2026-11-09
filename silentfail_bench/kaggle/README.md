# Kaggle runner for the Amendment-2 rescoring

Runs `scripts/gpu_rescore.py` on Kaggle's free 2×T4 as a **private** script
kernel against a **private** dataset holding the rescoring bundle. Launched
2026-09-24 via the Kaggle API; nothing here contains credentials.

- `dataset-metadata.json` — the dataset (`rescoring_bundle.jsonl` + `gpu_rescore.py`)
- `kernel-metadata.json` — script kernel, GPU + internet enabled, dataset attached
- `silentfail_rescore.py` — installs deps, smoke-tests 2 candidates through every critic, then runs all 164, zips `gpu_scores/`
- `run_kaggle.sh` — create dataset → push kernel → print status

Auth: Kaggle CLI 1.7.4 does not read the newer access-token file natively;
export it as `KAGGLE_API_TOKEN` from that file. Phone verification on the
account is required for GPU and internet. Output (`gpu_scores.zip` + log) is
retrievable only after the kernel finishes:
`kaggle kernels output <user>/silentfail-rescore-run -p <dir>`.
