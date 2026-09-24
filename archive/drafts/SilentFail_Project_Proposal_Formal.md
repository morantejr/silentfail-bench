# SilentFail-Bench: Selection-Conditional Detection of Critic Blind Spots in LLM Agents

**Joseph M. Morante Jr.**  
Algoverse  
September 2026  

> Formatted to match successful Algoverse `#project-proposals` (VMED / NNSJ / AAAS style).  
> Items needing mentor input are marked **[UNSURE]**.

---

## 1 Introduction

### Motivation

LLM agents fail silently mid-task—tool errors, goal drift, doom loops, fabricated success—and catching those failures before they compound is a reliability bottleneck. In practice, many agents commit to the next action by taking the **argmax of a critic** (best-of-\(n\), tree search, or a process reward model). Selecting the maximum of an imperfect critic concentrates on the critic’s own mistakes: actions with **high critic score** that **derail the task**. We call these *silent failures*.

Prior work either (i) improves step scoring (PRMs, generative verifiers, AgentPRM), (ii) repairs after a *known* failure (Reflexion-style), or (iii) uses reward-model uncertainty **unconditionally** on non-agentic best-of-\(n\) quality (URM/URME, RM ensembles). What is missing is the deployment-relevant question: **among the actions already selected**, does critic uncertainty discriminate task-verified failures **beyond the critic score itself**, and does that signal depend on critics being genuinely diverse rather than near-clones?

A calibrated “this selected action is secretly failing” signal is exactly the missing **trigger** for backtracking / selective verification. A clean **null** is also safety-relevant: it would mean agent safeguards built on beyond-score critic UQ inherit the critic’s shared blind spots.

### Contribution

We propose **SilentFail-Bench**, a weight-light, offline evaluation framework (benchmark + estimand) for selection-conditional silent-failure detection. Contributions:

1. **Formal estimand.** Define silent failures under selection and the primary metric \(\Delta\)AUROC of uncertainty \(u\) over critic score \(c\), conditional on selection.
2. **Benchmark protocol.** Frozen open agents on verifiable tasks; log candidates, selected actions, critic panels, and task-verified gold; release standardized splits and eval scripts.
3. **Mechanism + practice tests.** Manipulate critic diversity (seed → data → base → public zoo) with error-correlation mediation; compare cheap UQ to ensembles; measure offline selective-verification risk–coverage.
4. **Falsifiable null.** Pre-register that a flat \(\Delta\)AUROC even with diverse critics is a publishable safety result.

---

## 2 Related Works and Critical Gap

| Paper | What it does | Gap for us |
|---|---|---|
| Let’s Verify Step by Step (arXiv:2305.20050) | PRMs score steps; BoN/search selects | Score-to-select only; no beyond-score UQ in selected set |
| Math-Shepherd (arXiv:2312.08935) | Step verify + MC-rollout path labels | Useful gold recipe; not agentic silent-failure detection |
| Generative Verifiers (arXiv:2408.15240) | RM as next-token / self-consistency | Cheap UQ baseline; not selection-conditional |
| AgentPRM (arXiv:2511.08325) | PRMs for agents | Closest agent scoring; still score-to-select |
| Reflexion (arXiv:2303.11366) | Verbal repair / rollback | Needs a *trigger*; silent failures lack one |
| URM/URME (arXiv:2410.00847) | Uncertainty-aware RMs; ensemble filter | Unconditional / non-agentic BoN quality |
| Coste / Eisenstein / Gao overoptimization | RM ensembles mitigate Goodhart | Theory for diversity × pressure; not agent silent fails |
| AgentForesight / detect+repair / UQ-agents survey | Online failure monitoring | **[UNSURE]** Do they ask beyond-score \(\Delta\)AUROC *inside* the selected set with diversity mediation? Mentor lit-check required. |
| SWE-bench / WebArena / ALFWorld | Verifiable agent tasks | Task sources only; not a silent-failure detection bench |

**Critical gap:** No prior work (that we have verified) provides a **selection-conditional, beyond-score, task-verified** silent-failure detection suite for critic-guided LLM agents with **critic diversity manipulated and mediated**. Current methods either score/select, repair after known failure, or measure uncertainty unconditionally outside the selected set.

---

## 3 Methodology

We frame silent-failure detection as **incremental discrimination under selection**: among actions the agent commits to, does uncertainty \(u\) improve prediction of task failure beyond score \(c\)?

### 3.1 Setup

A frozen agent \(\pi\) runs on tasks \(x\) from verifiable benchmarks. At decision points it proposes candidates \(\{a_i\}\), scored by critic \(c\). Selection `sel` is either:

- **(a)** top-\(k\%\) by \(c\), or  
- **(b)** taken / best-of-\(n\) (deployment-relevant).

Gold \(g\in\{0,1\}\):

- **Primary:** trajectory success (unit tests / env reward).  
- **Secondary [UNSURE power/cost]:** step lies on a still-succeeding path via Math-Shepherd-style MC rollouts + noise sensitivity.

### 3.2 Primary estimand

\[
\Delta\mathrm{AUROC}
= \mathrm{AUROC}(c,u \mid \mathrm{sel})
- \mathrm{AUROC}(c \mid \mathrm{sel}).
\]

Also report partial AUC (low-alarm regime), likelihood-ratio test for \(u\), and clustered bootstrap CIs (cluster = task/trajectory). Always control action length, step index, and agent token entropy. Report the selection-induced \(c\)–\(u\) correlation as a **range-restriction diagnostic**.

### 3.3 Uncertainty \(u\)

**Primary:** disagreement across critic panel \(\{c_1,\ldots,c_m\}\) (std / range / rank disagreement after rank-normalization).  

**Cheap:** dropout, quantile-width, generative-verifier self-consistency.

### 3.4 Critic diversity ladder (manipulated)

| Level | Variation | Expected pairwise error corr. on gold |
|---|---|---|
| Seed | same model, different seeds | high |
| Data | different PRM / preference data | medium |
| Base | different base LLMs | lower |
| Zoo | public HF reward/PRM zoo | lowest **[UNSURE]** |

**Mediation:** a diversity step that fails to decorrelate critic errors must fail to raise \(\Delta\)AUROC.

### 3.5 Anchor cell (pre-register)

Base-diverse critics, \(m=4\), ensemble-std, BoN \(n=16\), \(k=5\%\), SWE-bench Verified, **one** open agent scaffold.  
Primary endpoint: \(\Delta\)AUROC at anchor, diverse vs seed.

**[UNSURE]** Exact scaffold (OpenHands / SWE-agent / other) — choose for logging cost.

---

## 4 Experimental Setup

| Component | Details |
|---|---|
| **Primary tasks** | SWE-bench Verified (gold = unit tests) |
| **Generality** | WebArena; ALFWorld (E6) |
| **Agent** | One frozen open coding/web scaffold **[UNSURE which]** |
| **Critics** | PRMs, generative verifiers, AgentPRM-style, HF zoo; RewardBench quality report |
| **Selection** | taken / BoN \(n\in\{4,8,16,32\}\); top \(k\in\{1\%,5\%,10\%\}\) |
| **Diversity** | seed / data / base / zoo; \(m\in\{2,4,8\}\) |
| **Metrics** | \(\Delta\)AUROC (primary), partial AUC, LR test, per-class \(\Delta\)AUROC, error-corr mediation, risk–coverage @ budget \(b\%\) |
| **Baselines** | score-only; trivial covariates; seed ensemble; single-critic cheap UQ |
| **Stats** | clustered bootstrap 95% CI; BH-FDR on non-primary contrasts; Stage-0 vacuousness gate |

### Concrete test matrix (VMED-style)

| Test | Protocol |
|---|---|
| **Vacuousness (E0)** | Silent-failure rate in selected set; AUROC of length/step/entropy alone |
| **Beyond-score (E1)** | Anchor ± \(n\), \(k\) sweep; \(\Delta\)AUROC and LR vs selection pressure |
| **Diversity mediation (E2)** | Diversity ladder × \(m\); mediate via pairwise error correlation |
| **Pressure × diversity (E3)** | Seed vs base-diverse × \(n\) grid; slope of \(\Delta\)AUROC vs \(\log n\) |
| **Cheap UQ (E4)** | Dropout / quantile / self-consistency vs ensemble at anchor |
| **Taxonomy (E5)** | Label tool error / goal drift / hallucinated success / doom loop / verifier gaming |
| **Generality (E6)** | Repeat E1 on WebArena / ALFWorld |
| **Selective verify (E7)** | Offline: failures caught at fixed verification budget \(b\%\) vs score-only |
| **Feature ablation** | Drop \(u\), drop trivials, drop diversity — measure \(\Delta\)AUROC drop |
| **Range restriction** | Report \(c\)–\(u\) corr. under selection; sensitivity if residual headroom vanishes |

---

## 5 Datasets and Evaluation

**Performance metrics**

- \(\Delta\)AUROC (higher better; primary)  
- FPR / TPR in low-alarm regime (partial AUC)  
- LR test \(p\)-value for adding \(u\)  
- Failures caught @ \(b\in\{5\%,10\%,20\%\}\) verification budget  
- Scanning cost: critic-panel inference time per selected step (secondary)

**Validation protocol**

- Stage-0 must pass before full matrix (non-trivial silent-failure rate; trivials leave headroom).  
- Pre-register primary endpoint; other contrasts FDR-controlled.  
- Trajectory gold is primary; step gold flagged as approximate.

---

## 6 Ideal Results

1. Selected-set silent-failure rate high enough for powered tests (e.g. \(\gtrsim 10\)–\(20\%\) **[UNSURE empirical]**).  
2. \(\Delta\)AUROC \(> 0\) at diverse anchor with significant LR; seed-only panel smaller/null.  
3. Diversity steps that reduce error correlation raise \(\Delta\)AUROC (mediation holds).  
4. Under larger \(n\), diverse panels lose less \(\Delta\)AUROC than seed panels.  
5. Cheap UQ recovers a usable fraction of ensemble signal **or** we clearly quantify the gap.  
6. At fixed budget \(b\%\), \(u\)-triggered selective verification catches more silent failures than score-only thresholding.  
7. **Acceptable ideal alternative:** clean null with diverse critics → negative-result safety paper.

---

## 7 Potential Limitations

1. Step-level gold is approximate (MC rollouts); trajectory gold is clean but coarser.  
2. Offline E7 cannot fully simulate adaptive backtrack feedback.  
3. Critic panels differ in scale/format; rank-normalization may be imperfect.  
4. SWE-bench logging cost may force a subset — **[UNSURE]** \(N\) for power.  
5. Range restriction may erase beyond-score headroom (itself an informative result).  
6. Concurrent 2026 agent-monitoring papers may partially preempt novelty — continuous lit refresh + mentor check.

---

## 8 Feasibility / Staging

| Stage | Goal | Go / no-go |
|---|---|---|
| **0** | One scaffold × SWE-bench subset; E0 | Silent failures exist; trivials don’t saturate |
| **1** | Pre-registered anchor + diversity ladder | Signal → promote; flat → negative paper |
| **2** | Full matrix E1–E7 + multi-bench | Workshop / conference writeup |

Compute: one logging pass, then offline scoring + regression; no RL. Public critic zoo ≈ download + inference.

---

## 9 Open questions for mentor

1. Is the critical-gap sentence still true vs AgentForesight / detect+repair / URM?  
2. Which frozen open scaffold for Stage-0?  
3. Venue: AAAI-27 workshops (~Nov 20) vs ICLR 2027 workshops (~Feb 1)?  
4. Benchmark-first framing confirmed?  
5. Quantitative targets in Ideal Results — keep as hypotheses or wait for Stage-0?

---

## One-line Slack pitch

**SilentFail-Bench:** among critic-*selected* agent actions, does uncertainty catch high-score / task-fail “silent failures” *beyond the score*, and only when critics are diverse—not clones?
