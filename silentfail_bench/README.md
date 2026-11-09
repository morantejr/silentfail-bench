# SilentFail-Bench (Stage-0)

Selection-conditional, beyond-score detection of **critic blind spots** on LLM-agent trajectories.

Primary unit: one committed trajectory $\tau$. Gold $g$ = trajectory task success (not step credit). Disjoint selection critic vs audit panel. Held-out $\Delta$AUROC of logistic $M_{c+u}$ vs $M_c$.

## Free-compute pilot (profile A) — runs on an Apple Silicon Mac, $0

No container runtime is required, but note what that costs: real SWE-bench gold
needs per-instance Docker images, so profile A substitutes a task family whose
gold tests execute locally. Gold is REAL (executed HumanEval+ held-out tests),
the agent loop is genuinely multi-step, and the split unit becomes `problem_id`
— so this profile does **not** test cross-repo generalization. See
[`configs/anchor_free.yaml`](configs/anchor_free.yaml) for what it can and
cannot claim, and [`DEFECTS.md`](DEFECTS.md) for what was wrong before it ran.

```bash
# 1. Ollama without sudo (Homebrew on this host is permission-broken)
mkdir -p ~/.local/ollama && cd ~/.local/ollama
curl -L -o o.tgz https://github.com/ollama/ollama/releases/download/v0.34.2/ollama-darwin.tgz
tar xzf o.tgz && ./ollama serve &

# 2. Models (~22 GB, one resident at a time)
./ollama pull qwen2.5-coder:7b && ./ollama pull llama3.1:8b
./ollama pull gemma2:9b && ./ollama pull mistral-nemo:12b

# 3. Run + report
python3 -c "import sys;sys.path.insert(0,'src');from pathlib import Path;\
from silentfail.runner_free import run_pilot_free;\
run_pilot_free(Path('configs/anchor_free.yaml'), Path('data/pilot_free'))"
python3 scripts/report_pilot_free.py data/pilot_free/pilot_free_scored.jsonl
```

Critics on this path are **LLM-as-judge**, not trained reward models, so `u` is
judge disagreement rather than RM-ensemble disagreement. Every row carries
`critic_type: llm_judge` and `is_true_reward_model: false`.

## Quick start (mock dry-run — no GPU)

```bash
cd silentfail_bench
pip install -e ".[dev]"
python scripts/run_stage0_pilot.py --n 40
pytest -q
```

## Live run (vLLM or API)

1. Pin `agent.git_commit` in `configs/anchor.yaml` after cloning mini-swe-agent.
2. Replace placeholder SWE-bench IDs:
   ```bash
   pip install -e ".[data]"
   python scripts/sample_pilot_instances.py --n 200
   ```
3. Serve models with vLLM (OpenAI-compatible) **or** set API keys:
   ```bash
   export VLLM_BASE_URL=http://127.0.0.1:8000/v1
   # or
   export OPENAI_API_KEY=...
   export OPENAI_BASE_URL=https://api.openai.com/v1
   ```
4. Wire the real SWE-bench harness into `silentfail/runner.py` (`build_prompt` / gold unit tests) — mock `g` is for dry-runs only.
5. Run pilot:
   ```bash
   python scripts/run_stage0_pilot.py --n 200
   python scripts/report_pilot.py data/pilot/pilot_scored.jsonl
   ```

**GPU:** required only if you self-host open weights via vLLM. API-only path needs no local GPU. Logistic combining models run on CPU.

## Layout

| Path | Purpose |
|---|---|
| [`FREEZES.md`](FREEZES.md) | Locked Stage-0 decisions |
| [`DEFECTS.md`](DEFECTS.md) | Defects found 2026-09-22 + verified sandbox posture |
| [`configs/anchor_free.yaml`](configs/anchor_free.yaml) | Free-compute overlay (profile A / B) |
| [`src/silentfail/stats.py`](src/silentfail/stats.py) | Calibrated clustered dAUROC test + power |
| [`src/silentfail/sandbox.py`](src/silentfail/sandbox.py) | sandbox-exec isolation for generated code |
| [`BUSINESS.md`](BUSINESS.md) | Business use cases |
| [`configs/anchor.yaml`](configs/anchor.yaml) | Scaffold, critics, $n$, exclusions |
| [`schema/trajectory_log.schema.json`](schema/trajectory_log.schema.json) | One-row-per-trajectory log |
| [`src/silentfail/`](src/silentfail/) | Schema, scoring, combine, backends, runner, pilot |
| [`scripts/`](scripts/) | Pilot runner, report, instance sampler |

## Go / no-go

See pilot report JSON: continue to Stage 1 only if failure counts meet power-derived minimum and `AUROC(M_c)` does not saturate (>0.90).
