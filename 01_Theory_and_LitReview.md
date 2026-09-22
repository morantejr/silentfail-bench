# SilentFail — Theoretical Framework + Lit Review Memo

**Date:** 2026-09-21  
**Purpose:** Lock motivation/theory, stress-test novelty, decide what to change before the conference-templated proposal.

---

## 1. Theoretical framework

### Motivation (why anyone should care)

Critic-guided agents (best-of-\(n\), tree search, PRMs) **commit by argmax of an imperfect proxy**. Mid-task mistakes that still look good to the critic compound. The missing piece for repair loops (Reflexion-style) is a **trigger**: when is a high-scoring selected action actually derailing the task?

### Main claim (1 sentence)

**Among actions selected by critic argmax, critic-ensemble uncertainty carries residual information about task-verified failure beyond the critic score itself, and that residual is governed by critic error decorrelation (diversity), not by ensembling per se.**

### What kind of theory is this?

Not a new learning algorithm. It is an **empirical causal / selection-bias theory** sitting at the intersection of:

| Theory lens | How it applies |
|---|---|
| **Goodhart’s law** | When a proxy (\(c\)) is optimized (selected on), it ceases to track the goal (\(g\)). |
| **Winner’s curse / selection bias** | Argmax over noisy proxy scores systematically picks overestimates → enriches the selected set for critic mistakes. |
| **Range restriction** | Conditioning on high \(c\) compresses score variance; beyond-score detectors must beat this. |
| **Epistemic UQ via ensembles** | Disagreement ≈ lack of shared knowledge; only works if errors are **not** perfectly correlated. |
| **Optimization pressure** | Larger \(n\) / greedier selection → stronger concentration on shared blind spots (predicts \(\Delta\)AUROC decay; diversity flattens it). |

**Falsifiable null (also a contribution):** even diverse critic panels share blind spots under selection → beyond-score UQ cannot trigger safe backtrack.

### Research questions

1. Does \(u\) add discrimination beyond \(c\) on the selected set? (\(\Delta\)AUROC)  
2. Is the increment mediated by error decorrelation across a diversity ladder?  
3. Does the signal survive cheap UQ and help at a fixed verification budget?

---

## 2. Lit review — closest priors (honest)

### Tier A — must cite and distinguish (novelty risk)

| Paper | What they did | Overlap | Our difference |
|---|---|---|---|
| **False Success / Silent Failure** arXiv:2606.09863 | Characterizes agents asserting success when env says otherwise; LLM judges fail (AUROC≤0.65); TF-IDF detectors work | Same *phrase* “silent failure”; agent reliability | They study **self-declared success** vs env gold, not **critic-selected high-\(c\)** actions. Detectors are lexical/telemetry, not critic-ensemble \(\Delta\)AUROC under selection. **Rename / reposition required.** |
| **Last Step Matters** arXiv:2608.29685 | Early verbal/perplexity UQ poorly predicts long-horizon failure (AUROC≤0.60 @50%); final-step confidence works | Agent UQ for failure prediction | Different signal (agent confidence ≠ critic panel disagreement); different conditioning (full trajectory progress ≠ selected high-\(c\) set). Their negative early-UQ result **motivates** testing a different UQ source. |
| **Real-Time Detect+Repair** arXiv:2608.02464 | Telemetry monitors + deterministic verify + rollback repair | Failure detection + repair trigger | No critic score / beyond-score / diversity mediation. Complementary baseline, not a preempt. |
| **AgentForesight** arXiv:2605.08715 | Trained online auditor for early alarm + step localization | Online failure auditing | Method paper (train 7B auditor), not selection-conditional critic UQ estimand. |
| **URM/URME** arXiv:2410.00847 | Uncertainty-aware RMs; ensemble filter low-quality outputs | Critic UQ | Unconditional / non-agentic preference quality — **not** agent selected-set \(\Delta\)AUROC. |
| **Coste et al. / Eisenstein et al.** | RM ensembles mitigate but don’t eliminate overoptimization; diversity matters | Mechanism for H2/H3 | Non-agentic BoN/PPO; we move to agents + task-verified \(g\) + mediation test. |
| **Inference-time reward hacking / winner’s curse** (e.g. 2506.19248) | BoN overestimates proxy as \(n\) grows | Theoretical backbone | We use this to **predict** silent failures under selection and test residual UQ. |
| **CalVerT** arXiv:2606.21777 | Calibrated confidence + grounding telemetry for commit decisions | Verifier telemetry as trigger | Knowledge-intensive QA agents; not critic-argmax selected-set beyond-score protocol. |
| **Ask Strong Judge when RM Uncertain** (NeurIPS 2025) | Route uncertain RM pairs to expensive judge | UQ as selective verification trigger | RLHF preference pairs, not agent action selection. Closest *practice* cousin for E7. |

### Tier B — scaffolding (cite, not novelty threats)

PRMs (Lightman; Math-Shepherd), Generative Verifiers, AgentPRM, Reflexion, SWE-bench / WebArena / ALFWorld, RewardBench, UQ-in-agents survey (2602.05073).

### Novelty verdict (honest)

| Claim | Strength |
|---|---|
| “First paper on silent failures in agents” | **False / fragile** — 2606.09863 already uses the term for false success. |
| “First agent failure detector” | **False** — AgentForesight, detect+repair, false-success detectors. |
| “First use of RM ensemble uncertainty” | **False** — URM/URME, Coste, Eisenstein. |
| “First **selection-conditional, beyond-score \(\Delta\)AUROC** protocol for **critic-guided agent actions** with **diversity mediation** and **task-verified gold**” | **Plausibly open** — no prior found that jointly does this. Soften “first”; say “we introduce / we evaluate.” |
| Benchmark contribution (SilentFail overlay) | **Strong packaging** — reviewers/advisors asked for a bench; this is the artifact. |

**Do not write:** “No prior work studies silent failures.”  
**Do write:** “Prior work studies false success, telemetry monitors, and unconditional RM UQ; none evaluate beyond-score critic uncertainty **conditional on critic selection** in agent trajectories.”

---

## 3. What to improve (given new context)

1. **Rename for collision avoidance**  
   Prefer: *Critic Blind Spots* / *Selected-Set Failures* / *Beyond-Score Silent Failures under Critic Selection* — keep “silent failure” only with a precise definition that cites 2606.09863 as related but distinct.

2. **Lead with theory, not just estimand**  
   Explicit Goodhart + winner’s curse → why selected set is the right conditioning event.

3. **Mandatory related-work axes**  
   (A) false success / agent failure monitors, (B) PRMs & agent critics, (C) RM ensembles & overoptimization. Three-column gap table.

4. **Benchmark-first**  
   SilentFail-Bench = logging protocol + splits + \(\Delta\)AUROC eval. Method novelty is secondary until Stage-0 shows headroom.

5. **Stage-0 gate**  
   Report selected-set high-\(c\), \(g=0\) rate before full diversity matrix. If tiny or trivials saturate → pivot (like attention-scaling → pause-tuning).

6. **Baselines from 2026**  
   Add: score-only, agent entropy/verbal confidence (Last Step Matters), telemetry monitor (if feasible), TF-IDF false-success features (2606.09863) as non-critic controls.

7. **Numeric ideal results** (VMED style) — hypotheses, not promises.

8. **Venue**  
   AAAI-27 workshop paper style (2–3pp, AAAI two-column conventions like VMED). Target ~Nov 20 workshop CFPs or ICLR 2027 workshops (~Feb).

9. **Claims hygiene**  
   Every gap sentence mentor-checked; no “will enable reliable agents” without E7 evidence.

10. **Null is first-class**  
   Match Eisenstein: ensembles may not rescue shared blind spots — that’s a headline.

---

## 4. Compressed narrative for the new proposal

**Problem:** Critic argmax concentrates Goodhart errors (silent failures).  
**Gap:** Failure detectors and RM-UQ exist, but not selection-conditional beyond-score tests with diversity mediation.  
**Artifact:** SilentFail-Bench + \(\Delta\)AUROC protocol.  
**Tests:** detection, diversity mediation, pressure×diversity, cheap UQ, offline selective verify.  
**Theory:** Goodhart + winner’s curse + epistemic disagreement.
