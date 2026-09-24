# SilentFail-Bench: Catching Silent Failures in Critic-Selected Agent Actions

**For:** `#project-proposals`  
**Author:** Joseph M. Morante Jr.  
**Date:** September 2026  
**Status:** Draft for mentor discussion — items marked **[UNSURE]** need input  

> Proposals must be specific. Mark uncertainty explicitly (as below).  
> Precedent: Dynamic Attention Scaling → eventually [Pause-Tuning, arXiv:2502.20405](https://arxiv.org/abs/2502.20405) (method pivoted; proposal specificity enabled the pivot).

---

## Relevant Past Papers

1. **Let’s Verify Step by Step** (Lightman et al., 2023) — arXiv:2305.20050  
   Process reward models score intermediate steps; best-of-n / search uses those scores to select.  
   *Gap for us:* scoring/selection only — no uncertainty-beyond-score analysis in the *selected* set.

2. **Math-Shepherd** (Wang et al., 2023) — arXiv:2312.08935  
   Step-wise verification + MC-rollout style labels for “on a succeeding path.”  
   *Useful:* step-level gold construction we can copy.  
   *Gap:* not agentic selection-conditional silent-failure detection.

3. **Generative Verifiers** (Zhang et al., 2024) — arXiv:2408.15240  
   Reward modeling as next-token prediction / self-consistency style verification.  
   *Useful:* cheap single-critic uncertainty baseline.  
   *Gap:* not conditioned on agent selection.

4. **AgentPRM** (2025) — arXiv:2511.08325  
   Process reward models for LLM agents (step-wise promise/progress).  
   *Closest agent scoring work.*  
   *Gap:* still score-to-select; does not ask whether uncertainty adds over the score among selected actions.

5. **Reflexion** (Shinn et al., 2023) — arXiv:2303.11366  
   Verbal self-correction / rollback after failure.  
   *Gap:* needs a *trigger*; assumes you know when to reflect. Silent failures are exactly when you don’t.

6. **URM / URME** (ICLR 2025) — arXiv:2410.00847  
   Uncertainty-aware reward models; ensemble disagreement filters low-quality outputs.  
   *Closest UQ prior.*  
   *Gap:* unconditional / non-agentic BoN quality filtering — **not** selection-conditional, beyond-score ΔAUROC in agents.

7. **Reward model ensembles & overoptimization** — Coste et al. arXiv:2310.02743; Eisenstein et al. arXiv:2312.09244; Gao et al. arXiv:2210.10760  
   Ensembles mitigate overoptimization; diversity matters; Goodhart under pressure.  
   *Useful:* theory for H2/H3 (diversity × selection pressure).  
   *Gap:* not agent silent-failure detection with task-verified gold.

8. **Agent failure detection** — AgentForesight arXiv:2605.08715; Real-time detect+repair arXiv:2608.02464; UQ-in-agents survey arXiv:2602.05073  
   Online monitoring / foresight / repair for agent failures.  
   *Gap (claim to verify with mentor):* do **not** evaluate whether critic uncertainty adds discrimination **beyond the critic score** inside the **selected** action set, with diversity manipulated.

9. **SWE-bench / WebArena / ALFWorld** — Jimenez et al. arXiv:2310.06770; Zhou et al. arXiv:2307.13854; Shridhar et al. arXiv:2010.03768  
   Verifiable agent environments (unit tests / env success).  
   *Role for us:* **task sources**, not the contribution. Contribution = SilentFail overlay + estimand.

**[UNSURE]** Exact novelty vs AgentForesight / 2608.02464 — need mentor lit check before any “first to…” sentence.

---

## Motivation

LLM agents often choose the next action by taking the **argmax of a critic** (best-of-n, tree search, process reward model). That operation concentrates on the critic’s own mistakes: actions that look good to the critic but derail the task — **silent failures**.

Existing work either (a) improves the score, (b) repairs after a *known* failure, or (c) measures uncertainty **unconditionally**. Nobody (that we’ve found) asks the deployment-relevant question:

> Among the actions the agent *already selected*, does critic uncertainty catch silent failures **beyond the score itself**, and only when critics are genuinely diverse?

That signal is the missing **trigger** for backtracking / selective verification. A clean **null** is also valuable: it would mean agent safeguards built on beyond-score critic UQ inherit the critic’s shared blind spots.

**Why a benchmark (not just a method paper):** the contribution needs a fixed, reusable protocol (candidates + selection + scores + uncertainty + task gold + ΔAUROC) so others can compare detectors. SWE-bench alone is a *task*; SilentFail is the *detection* suite.

---

## Key Contributions / Ideas

1. **Define silent failures under selection:** selected action with high critic score \(c\), task-verified failure \(g=0\).
2. **Primary estimand:** selection-conditional beyond-score discrimination  
   \(\Delta\mathrm{AUROC} = \mathrm{AUROC}(c,u\mid\mathrm{sel}) - \mathrm{AUROC}(c\mid\mathrm{sel})\).
3. **SilentFail-Bench:** offline logging protocol + eval suite on verifiable agent tasks.
4. **Mechanism tests:** critic diversity ladder (seed → data → base → public zoo) with error-correlation mediation — “more critics” vs “different critics.”
5. **Practice tests:** cheap UQ vs true ensembles; offline selective-verification risk–coverage.

---

## Methods

### Setup (concrete)

1. Run a **frozen** open agent scaffold on verifiable tasks (primary: SWE-bench Verified).  
2. At each decision point, log:
   - full **candidate** actions (or BoN trajectories)
   - critic score(s) \(c\)
   - which action was **selected** (taken / best-of-\(n\) / top-\(k\%\))
   - agent metadata: step index, action length, token entropy  
3. Label gold \(g\):
   - **Primary:** trajectory-level task success (tests pass / env reward) — clean, free.
   - **Secondary [UNSURE feasibility]:** step-level “still on a succeeding path” via Math-Shepherd-style MC rollouts + noise sensitivity.  
4. Score every selected action with a **critic panel** \(\{c_1,\ldots,c_m\}\); rank-normalize before disagreement.  
5. Uncertainty \(u\):
   - Primary: ensemble std / range / rank disagreement  
   - Cheap: dropout, quantile-width, generative-verifier self-consistency  
6. Fit conditional discrimination: predict \(g=0\) from \(c\) alone vs \(c+u\) (+ trivial covariates). Report \(\Delta\)AUROC, partial AUC (low-alarm), LR test; cluster by task/trajectory.

### Diversity ladder (manipulated)

| Level | What varies | Expected error correlation |
|---|---|---|
| Seed | same model, different seeds | high |
| Data | different preference / PRM training data | medium |
| Base | different base LLMs | lower |
| Zoo | public HF reward / PRM zoo | lowest **[UNSURE]** |

**Mediation check:** if a diversity step does not reduce pairwise error correlation on gold, it should not raise \(\Delta\)AUROC.

### Selection constructions (both reported)

- (a) Fixed-percent: top \(k\%\) by \(c\)  
- (b) Taken / best-of-\(n\): deployment-relevant; pressure roughly \(\mathrm{KL} \approx \log n - (n-1)/n\)

### Anchor cell (pre-register)

Base-diverse critics, \(m=4\), ensemble-std, BoN \(n=16\), \(k=5\%\), SWE-bench Verified, **one** open agent scaffold.  
Primary endpoint: \(\Delta\)AUROC at anchor, diverse vs seed, clustered-bootstrap 95% CI.

**[UNSURE]** Which exact open scaffold (OpenHands / SWE-agent / Mini-SWE-agent / other) — pick with mentor for logging cost.

**[UNSURE]** Exact adjustment for multi-critic scale mismatch beyond rank-normalization — may need calibration / pairwise ranking agreement instead of raw std.

---

## Experimental Setup

### How we prove it works

| ID | Question | Design | Success criterion |
|---|---|---|---|
| **E0** | Do silent failures exist? Do trivials saturate? | Selected-set failure rate; AUROC of length / step / entropy | Non-trivial \(g=0\) rate; trivials leave headroom |
| **E1** | Beyond-score detection (H1) | Anchor ± selection sweep (\(n\), \(k\)) | \(\Delta\)AUROC $>0$; LR significant; shrinks as \(n\)↑ |
| **E2** | Diversity mediated (H2) | Diversity ladder × \(m\) | \(\Delta\)AUROC rises with diversity; mediated by error corr. |
| **E3** | Pressure × diversity (H3) | Seed vs base-diverse × \(n\) grid | Flatter \(\Delta\)AUROC decay for diverse panels |
| **E4** | Cheap UQ (H4) | Estimator sweep at anchor | Report parity gap vs ensemble |
| **E5** | Failure taxonomy (H5) | Label: tool error / goal drift / hallucinated success / doom loop / verifier gaming | Heterogeneous per-class \(\Delta\)AUROC |
| **E6** | Generality | WebArena and/or ALFWorld | Same qualitative pattern |
| **E7** | Offline impact | Selective verify at budget \(b\%\) using \(u\) | Failures caught vs score-only risk–coverage |

### Ablations / confound control

- Always control: \(c\), action length, step index, agent entropy.  
- Report \(c\)–\(u\) correlation under selection (**range restriction** is expected, not a nuisance).  
- Diversity vs \(m\): hold \(m\) fixed when comparing diversity levels; hold diversity when sweeping \(m\).  
- Seed-only ensemble = negative control for “ensembling per se.”  
- Stage-0 vacuousness gate: if silent-failure rate ≈ 0 or trivials saturate → **stop / pivot** before full matrix.

### Visualizations / stats

- Fig 1: selected set = on-track vs silent failures; size = disagreement.  
- Fig 2: logging pipeline.  
- Fig 3: diversity → error corr → \(\Delta\)AUROC.  
- Tables: \(\Delta\)AUROC + CI; mediation coefficients; risk–coverage curves.  
- Inference: clustered bootstrap; Benjamini–Hochberg FDR on non-primary contrasts.

---

## Datasets and Evaluation

| Resource | Role |
|---|---|
| **SWE-bench Verified** | Primary tasks; gold = unit tests |
| **WebArena** | Long-horizon / goal-drift regime |
| **ALFWorld** | Third domain for E6 |
| **Critics** | PRMs, generative verifiers, AgentPRM-style, HF reward zoo; quality anchored to RewardBench |
| **Agent logs** | Frozen open scaffold; candidates + taken actions |

**Metrics:** \(\Delta\)AUROC (primary), partial AUC, LR test, unconditional AUROC (secondary), per-class \(\Delta\)AUROC, error-correlation mediation, risk–coverage at budget \(b\%\).

**Baselines:** score-only; trivial covariates; seed ensemble; single-critic cheap UQ.

---

## Ideal Results

1. Non-trivial silent-failure rate in the selected set (E0 passes).  
2. \(\Delta\)AUROC $>0$ at the diverse anchor; smaller/null for seed-only.  
3. Mediation: diversity steps that decorrelate errors raise \(\Delta\)AUROC.  
4. Under stronger BoN pressure, diverse panels lose less signal than seed panels.  
5. Cheap UQ recovers a usable fraction of ensemble signal (or we clearly report the gap).  
6. At a fixed verification budget, \(u\)-triggered selective check catches more silent failures than score-only thresholding.

**Also ideal (and publishable):** clean null with diverse critics → safety-relevant negative result paper.

---

## Potential Limitations

- Step-level gold is approximate (MC rollouts); trajectory gold is clean primary.  
- Offline E7 cannot fully simulate adaptive backtrack feedback loops.  
- Critic panels are heterogeneous (formats/scales); rank-norm may be imperfect.  
- Logging cost on SWE-bench Verified may force a subset — **[UNSURE]** target \(N\) tasks / steps for power.  
- If selection already correlates \(c\) and \(u\) too tightly, beyond-score headroom may vanish (range restriction) — that is a result, not only a flaw.  
- Novelty vs concurrent 2026 agent-monitoring papers needs continuous lit refresh.

---

## Feasibility / Staging

| Stage | Goal | Go / no-go |
|---|---|---|
| **0** | Vacuousness/power on one scaffold × SWE-bench subset | Silent failures exist; trivials don’t saturate |
| **1** | Pre-registered anchor + diversity ladder | Signal → promote; flat → negative-result paper |
| **2** | Full E1–E8, multi-bench, taxonomy, step gold | Workshop / main-conference writeup |

**Compute:** offline scoring + regression after one logging pass; no RL. Public critic zoo is download + inference.

**Writing path:** proposal → SilentFail-Bench paper outline → mentor review → PI Slack review (≥1 week before deadline). Do **not** ask PI until near-final + mentor feedback.

---

## Open questions for mentor (please discuss)

1. Confirm gap vs AgentForesight / real-time repair / URM — is the selection-conditional beyond-score framing still open?  
2. Which open agent scaffold to freeze for Stage-0?  
3. Target venue: AAAI-27 workshops (~Nov 20) vs ICLR 2027 workshops (~Feb 1)? Nov is tight for a real bench release.  
4. Is Stage-0 on SWE-bench Verified subset enough for `#project-proposals` approval, or do they want WebArena from day one?  
5. Benchmark-first vs method-first framing — we currently recommend **benchmark + estimand**.

---

## One-line pitch (for Slack)

**SilentFail-Bench** tests whether critic *uncertainty* catches high-score / task-fail agent actions *beyond the score itself* among *selected* actions — and whether that only works when critics are diverse, not clones.
