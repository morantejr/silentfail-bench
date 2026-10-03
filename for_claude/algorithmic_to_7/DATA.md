# Where the matrices are

All of these paths were in `git ls-files` on 3 October 2026. Do not assume a deeper data path is tracked. Confirm with `git ls-files -- <path>` before depending on it. Do not edit these files.

## Committed rows

| | Verified | Multilingual |
|---|---|---|
| Rows | `silentfail_bench/data/swebench_pilot50/pilot_free_scored.jsonl` | `silentfail_bench/data/replication_multilingual/replication_scored.jsonl` |
| 16k critic matrices | `silentfail_bench/data/swebench_pilot50/l4/out/gpu_scores_16k/scores_gpu_*.json` | `silentfail_bench/data/replication_multilingual/modal/full16k/scores_gpu_*.json` |
| Other tracked pass | `silentfail_bench/data/swebench_pilot50/gpu_scores/scores_gpu_*.json` | `silentfail_bench/data/replication_multilingual/modal/prefix4k/scores_gpu_*.json` |

Use the 16k matrices for the variance split. The prose you are formalizing is about critics scored at length.

Seven critics, stem after `scores_gpu_`:

- `ray2333_grm_gemma2_2b_rewardmodel_ft`
- `ray2333_grm_llama3_2_3b_rewardmodel_ft`
- `skywork_skywork_reward_v2_llama_3_1_8b`
- `skywork_skywork_reward_v2_llama_3_2_1b`
- `skywork_skywork_reward_v2_qwen3_1_7b`
- `skywork_skywork_reward_v2_qwen3_4b`
- `tiger_lab_acecoderm_7b`

Registered 4-member panel, the one whose published \(u\) is the headline: the two GRMs, Skywork Llama-3.2-1B, Skywork Qwen3-1.7B. Run the variance split twice: panel of 4, and the zoo of 7. The 0.71–0.89 range is over 4-member subsets of the 7.

## Fields

| Quantity | Where |
|---|---|
| Raw critic score | `scores_gpu_*.json` → `scores["{instance_id}#{candidate}"]["score"]` |
| Gold | `g` in the scored jsonl. `g == 0` is failure. AUROC positive class: `y = 1 if g == 0 else 0` |
| Selector score | `c` |
| Published disagreement | `u` |
| Rank-normalized panel | `audit_scores_rank_norm` |
| Committed population | `not excluded` and `visible_pass`. The batch-1 script uses this and gets Verified n=34, Multilingual n=77. |
| Which candidate | `selected_candidate_index` |
| Repository cluster | `repo` |
| Train fold for rank norms | `split == "train"` |

Matrix top-level keys: `critic`, `scores`. Score entry keys: `score`, `truncated`, `n_tokens`, `max_tokens_used`.

## Where the published correlation is computed

- Script: `silentfail_bench/scripts/readout_pi_batch1.py` lines 61–85 (`E2_diversity`, `error_corr`) and lines 87–101 (learned combiner).
- Stored: `silentfail_bench/data/readout_pi_batch1.json` field `E2_diversity.error_corr_range`.
- Paper: `paper/main.tex` lines 262 and 264. Exploratory label: line 283.

There is no two-way fixed-effect code in the repo. The `resid()` at `readout_pi_batch1.py` lines 76–77 residualizes AUROC(\(u\)) on panel quality for a partial correlation. It is not a rater model. Do not extend that function in place. A new script belongs outside `data/` and should write a new JSON the checker does not read, until the PI asks to wire it in.

## Load sketch

```python
import json
from pathlib import Path
import numpy as np

ROOT = Path("/Users/jmorante/Downloads/xtail_spec/silentfail_bench")
rowf = ROOT / "data/swebench_pilot50/pilot_free_scored.jsonl"
scored = ROOT / "data/swebench_pilot50/l4/out/gpu_scores_16k"

rows = [json.loads(l) for l in rowf.read_text().splitlines() if l.strip()]
rows = {r["instance_id"]: r for r in rows}
S = {}
for f in scored.glob("scores_gpu_*.json"):
    S[f.stem.replace("scores_gpu_", "")] = json.loads(f.read_text())["scores"]

pop = [i for i, r in rows.items() if not r["excluded"] and r.get("visible_pass")]
key = lambda i: f"{i}#{rows[i]['selected_candidate_index']}"

y = np.array([1 if rows[i]["g"] == 0 else 0 for i in pop])
c = np.array([rows[i]["c"] for i in pop], float)
u = np.array([rows[i]["u"] for i in pop], float)
repo = [rows[i]["repo"] for i in pop]

critics = sorted(S)
R = np.column_stack([
    [S[m].get(key(i), {}).get("score", np.nan) for i in pop] for m in critics
])  # shape (n, J)
# Drop or report any row with a NaN before demeaning.
# Critic means: fit on rows with split == "train" only, then transform all of pop.
# u_perp[i] = sd across critics of (R[i, j] - critic_mean[j])
```

Multilingual: swap `rowf` to `replication_scored.jsonl` and `scored` to `modal/full16k`.

## Population rules that are easy to get wrong

- Fit critic means on `split == "train"` only, then apply them to every committed row. Fitting on all rows leaks the test repositories into \(u_{\perp}\). The paper's rank norms already follow this rule (`fit_rank_norms` on `train`).
- Positive class is failure, not pass. `roc_auc_score(y, c)` will not match the paper if `y` is `g`.
- Verified analysed n and Multilingual analysed n in the headline are not 34 and 77 for every table. The batch-1 filter is the right one for a comparison against `E2` and the learned combiner. Say the n in the output.
- Amendment 13 DeepSeek scores are a different matrix and a descriptive-only cell. Do not pool them with these rows. Do not refit this decomposition on the agent axis to rescue a selector gate that failed.

## Output to leave in this folder

Write `RESULTS.md` here when the table exists: n, failures, repositories, SS shares, AUROC(\(u\)), AUROC(\(u_{\perp}\)), AUROC(\(c\)), and the error-correlation of raw scores versus critic-demeaned scores. No paper edit in the same step.
