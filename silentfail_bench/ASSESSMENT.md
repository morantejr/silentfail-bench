# SilentFail-Bench: Pre-Experiment Assessment (Implemented)

**Overall rating: 78 / 100**  
**Date:** 2026-09-22

## Relevance and novelty

- **Relevant:** yes — post-commit triggers for critic-guided agents are a live industry/research need.
- **Novel:** yes as a **narrow cell** (post-selection, trajectory gold, disjoint audit, held-out $c$ vs $c{+}u$), not as the whole “agent failures” field.

## GPU vs vLLM

| Path | Local GPU? |
|---|---|
| vLLM serving open agent + RMs | Yes |
| OpenAI-compatible hosted API | No |
| Logistic $M_c$ / $M_{c+u}$ | CPU only |
| SWE-bench tests | CPU / sandbox |

Stage-0 code supports **mock** (dry-run), **vLLM**, and **API** backends.

## Business use cases

See [`BUSINESS.md`](BUSINESS.md): coding agents, support/ops escalation, budgeted verification, RM procurement, compliance. Value = escalation policy, not a new foundation model.

## What was locked / built

| Deliverable | Location |
|---|---|
| Freezes (scaffold, score rule, critics, exclusions) | [`FREEZES.md`](FREEZES.md), [`configs/anchor.yaml`](configs/anchor.yaml) |
| Logging schema (1 row / trajectory) | [`schema/trajectory_log.schema.json`](schema/trajectory_log.schema.json) |
| Runner + backends | [`src/silentfail/runner.py`](src/silentfail/runner.py), [`backends.py`](src/silentfail/backends.py) |
| Pilot + go/no-go + power | [`scripts/run_stage0_pilot.py`](scripts/run_stage0_pilot.py), [`pilot.py`](src/silentfail/pilot.py) |
| Tests | 5 passed (`pytest`) |
| Proposal anchor updated | `../SilentFail_AAAI_Proposal.tex` |

## Before a live 200-trajectory pilot

1. Pin `agent.git_commit` in `configs/anchor.yaml`
2. `python scripts/sample_pilot_instances.py --n 200` (needs `datasets`)
3. Replace mock gold with real SWE-bench unit tests in the harness
4. Start vLLM or set API keys
5. Run `python scripts/run_stage0_pilot.py --n 200` and honor go/no-go
