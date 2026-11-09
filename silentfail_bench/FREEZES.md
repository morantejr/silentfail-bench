# SilentFail-Bench Stage-0 Freezes

**Status:** Frozen for Stage-0 pilot (mentor may swap checkpoint hashes before first run).  
**Date:** 2026-09-22  
**Source plan:** SilentFail Readiness Assessment

---

## One-score-per-trajectory rule (locked)

> **$c$ is the selection critic’s scalar score of the full committed trajectory text** (concatenated agent messages + final patch/diff), not a mean of per-step scores. Best-of-$n$ ranks $n$ full candidate trajectories by this same whole-trajectory score and commits the argmax.

---

## Anchor freezes

| Item | Frozen value |
|---|---|
| **Agent scaffold** | `SWE-agent` light harness via **mini-swe-agent** pattern; pin commit at install time into `configs/anchor.yaml` → `agent.git_commit` |
| **Tasks** | SWE-bench Verified; Stage-0 pilot = **200** instance IDs from `configs/pilot_instances.txt` |
| **Candidate rule** | Best-of-$n$: generate **$n=16$ independent full trajectories** (different seeds / temperatures); score each with $c_{\mathrm{sel}}$; **retain top-1** |
| **Selection critic** | `Skywork/Skywork-Reward-V2-Qwen3-8B` (or mentor override); role = `$c_{\mathrm{sel}}$` only |
| **Audit panel ($m=4$), disjoint** | See `configs/anchor.yaml` — none may equal selection critic |
| **$u$** | Sample std of train-fit rank-normalized audit scores |
| **Quality-match metric** | Pairwise ranking accuracy on a held-out preference slice **or** RewardBench subset score; panels compared only after match within ±2 pp or via regression control |
| **Max steps / timeout** | `max_agent_steps=50`; `per_trajectory_timeout_sec=1800`; exclude on timeout / empty candidates / critic NaN |
| **Repo split** | Group by `repo` field; 70/30 train/test by unique repo (seed=`split_seed`) |
| **Primary endpoint** | Held-out $\Delta$AUROC = AUROC($M_{c+u}$) − AUROC($M_c$); positive class = failure ($g=0$) |
| **Primary contrast** | Disjoint quality-matched diverse panel vs seed panel (seed panel = Stage-0 exploratory if only one panel available) |
| **Compute path** | **vLLM** for open agent + critics **or** OpenAI-compatible API; logistic models on CPU |

---

## Amendments — 2026-09-22 (free-compute pilot)

These do not relax the anchor freezes; they record what the free path cannot
honor, so nothing is quietly claimed.

| Freeze | Status on profile A | Why |
|---|---|---|
| Tasks = SWE-bench Verified | **Not honored** — HumanEval+ | No container runtime on host; SWE-bench gold needs per-instance Docker |
| Split unit = repository | **Not honored** — `problem_id` | No repo field in the substitute family; cross-repo generalization untested |
| Selection critic = Skywork-Reward-V2 | **Not honored** — LLM-as-judge | No torch/CUDA; Ollama serves chat models, not scalar RM heads |
| Audit panel m=4 | **Reduced to m=2** | u becomes \|a−b\|/√2 — magnitude only, no shape |
| Panels quality-matched ±2 pp | **Skipped** | No RewardBench run, no local preference slice → diverse-vs-seed contrast out of scope |
| Disjoint c_sel ∉ audit panel | **Honored** | c_sel `llama3.1:8b`; panel `gemma2:9b`, `mistral-nemo:12b` |
| c = whole-trajectory score | **Honored** | BoN ranks full trajectories; no step-mean |
| Trajectory-level gold only | **Honored** | g = executed held-out HumanEval+ tests |
| rank_norm_fit = train_split_only | **Honored (newly)** | Was violated by the original runner — see DEFECTS.md #5 |

## Go / no-go (numeric thresholds filled after pilot)

Continue to Stage 1 only if **both**:

1. Selected-set failures $N_{fail}$ ≥ pilot-simulated count for 80% power at MDE $\Delta$AUROC ≥ 0.03 (script: `scripts/power_from_pilot.py`).
2. Held-out AUROC($M_c$) ≤ 0.90 (headroom for $u$); if $M_c$ already > 0.90, stop or pivot.

Until pilot runs, thresholds are **not** fixed numbers in Ideal Results.

---

## Explicitly out of scope for Stage 0

WebArena, ALFWorld, step-level MC labels, shared-panel ablation, full diversity ladder, E4–E7.
