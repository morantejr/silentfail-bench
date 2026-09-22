# SilentFail — Paper Outline (Algoverse process)

**Status:** Stage 0 outline only. Do **not** treat this as submission-ready.  
**Source proposal:** `Agent_Silent_Failures_Proposal.pdf`  
**Working venue (TBD):** AAAI-27 workshop (papers due ~Nov 20, 2026) or ICLR 2027 workshop (~Feb 1, 2027).  
**Paper type:** Benchmark + evaluation protocol (not a new agent architecture).

---

## Process gates (from program rules — non-negotiable)

| Gate | Requirement |
|---|---|
| 1. Mentor review | Mentor gives feedback **before** any PI Slack ask |
| 2. Near-final draft | Complete sections; no missing Method / Experiments / Related Work |
| 3. PI Slack review | Ask for a paper-review slot only after 1–2 |
| 4. Lead time | Submit to PI **≥ 1 week before conference deadline** (comments → revise → re-review) |
| 5. Submission ethics | Do not submit without program review; they will withdraw if submitted unreviewed |

### Reverse schedule (pick one target)

**If AAAI-27 workshops (≈ Nov 20, 2026):**

| Date | Deliverable |
|---|---|
| Now–Oct 5 | Title + skeleton + intro bullets (this doc) → mentor pass |
| Oct 6–20 | Stage-0 data: silent-failure rate on selected set; empty tables filled |
| Oct 21–Nov 5 | Full draft v1 → mentor review → revise |
| **≤ Nov 13** | Near-final → **PI Slack review request** |
| Nov 13–19 | PI comments → revise → second pass |
| Nov 20 | Workshop submission |

**If ICLR 2027 workshops (≈ Feb 1, 2027):** same gates, more realistic for a real SilentFail release. Prefer this unless Stage-0 is already strong by mid-October.

---

## Compress to 1–3 claims (paper narrative)

**One cohesive theme:** selection-conditional, beyond-score detection of silent failures in critic-guided agents.

1. **Existence / setup claim:** Under critic argmax selection (best-of-n / top-k%), a non-trivial fraction of selected actions are *silent failures* (high critic score \(c\), task-verified failure \(g=0\)).
2. **Detection claim:** Conditional on selection, critic uncertainty \(u\) adds discrimination beyond \(c\) alone (\(\Delta\)AUROC \(> 0\); LR test), after controlling for cheap covariates.
3. **Mechanism claim:** The increment rises with critic *diversity* (error decorrelation), not merely with ensemble size; seed-only panels underperform base-diverse / zoo panels.

**Falsifiable / publishable null:** If \(\Delta\)AUROC \(\approx 0\) even with diverse critics → agent safeguards that trust beyond-score critic UQ inherit the critic’s shared blind spots.

**Do not claim (until evidenced):** online backtrack gains; SOTA agent accuracy; that cheap UQ always matches ensembles; that step-level MC-rollout gold is exact.

---

## 0. Working title options

1. SilentFail-Bench: Selection-Conditional Detection of Critic Blind Spots in LLM Agents  
2. Beyond the Score: Does Critic Uncertainty Catch Silent Failures in Selected Agent Actions?  
3. When Argmax Concentrates Mistakes: A Benchmark for Silent Failures under Critic Selection  

*(Pick one with mentor before writing intro prose.)*

---

## 1. Paper skeleton (sections / subsections)

1. Abstract  
2. Introduction  
3. Problem Setup and Estimand *(Background + formal defs — keep short)*  
4. SilentFail-Bench  
   - 4.1 Source environments and agents  
   - 4.2 Logging protocol (candidates, selection, scores)  
   - 4.3 Gold labels (trajectory primary; step-level secondary)  
   - 4.4 Critic panels and diversity ladder  
   - 4.5 Uncertainty estimators  
   - 4.6 Evaluation protocol and metrics  
5. Experiments  
   - 5.1 Stage-0 vacuousness / power checks  
   - 5.2 Beyond-score detection (H1)  
   - 5.3 Diversity and mediation (H2)  
   - 5.4 Selection pressure × diversity (H3)  
   - 5.5 Cheap uncertainty (H4)  
   - 5.6 Failure taxonomy (H5)  
   - 5.7 Cross-benchmark generality  
   - 5.8 Offline selective-verification impact  
6. Results and Analysis *(answer RQs; self-contained captions)*  
7. Related Work *(near end; 3 axes)*  
8. Conclusion  
9. Limitations *(required if *ACL; still useful elsewhere)*  
10. Appendix  

**Figure plan (placeholders now):**

- Fig 1 (page 1): selected set = mix of on-track vs silent failures; estimand \(\Delta\)AUROC  
- Fig 2: SilentFail logging pipeline  
- Fig 3: diversity ladder → error correlation → \(\Delta\)AUROC  
- Tables: empty shells with row/column headers before numbers exist  

---

## 2. Intro skeleton (bullets only — Celine structure)

### ¶ Motivation — why care?
- LLM agents commit mid-task and fail silently (loops, tool errors, goal drift, fabricated success); catching failures before they compound is the reliability bottleneck.  
- Need citation cluster: agent failure surveys / foresight / repair papers from proposal.  
- A calibrated “this *selected* step is secretly failing” signal is the missing trigger for backtrack / selective verification (Reflexion-style repair needs a trigger).

### ¶ Context / gap — existing approaches aren’t enough
- Agents often select via critic argmax (BoN, tree search, PRMs, generative verifiers).  
- Prior work: (i) scores steps, (ii) repairs after *known* failure, (iii) detects failures without asking whether uncertainty adds **beyond the score in the selected set**, (iv) RM ensemble UQ shown **unconditionally** on non-agentic BoN quality.  
- Gap sentence (careful, no overclaim): *No prior work evaluates selection-conditional, beyond-score silent-failure detection with task-verified gold and a manipulated diversity ladder.*  
- **Must lit-check with mentor before this sentence ships.**

### ¶ Insight + research questions
- Insight: selecting the max of an imperfect critic concentrates on the critic’s mistakes → selected set is enriched for silent failures; score alone cannot sort them.  
- RQ1: Among selected actions, does \(u\) discriminate \(g=0\) beyond \(c\)?  
- RQ2: Does critic diversity (via error decorrelation) govern that increment?  
- RQ3: How much survives under cheap UQ and as an offline selective-verify trigger?

### ¶ Method summary (1 short block)
- Release / define SilentFail-Bench: log candidates under frozen open agents on verifiable tasks; score with critic panels; label \(g\) from task success; evaluate \(\Delta\)AUROC on the selected set; sweep diversity and selection pressure.

### ¶ Contributions (explicit list — end of intro)
1. Formalize the selection-conditional beyond-score estimand for silent failures.  
2. Introduce SilentFail-Bench: protocol + (eventual) public logs/splits + standardized metrics.  
3. Empirical answers to RQ1–RQ3 (detection, diversity mediation, cheap UQ, offline risk–coverage), including informative nulls.  

---

## 3. Related work skeleton (3 axes; fill ≥10–20 papers)

**Axis A — Agent failure detection / monitoring / repair**  
- AgentForesight; real-time detect+repair; UQ-in-agents survey; Reflexion  
- Diff: we ask beyond-score \(\Delta\)AUROC *conditional on critic selection*, not generic failure prediction.

**Axis B — Process reward models / generative verifiers / agent PRMs**  
- Let’s Verify Step by Step; Math-Shepherd; Generative Verifiers; AgentPRM  
- Diff: they score/select; we study residual risk *among selected* actions.

**Axis C — Reward-model ensembles, overoptimization, uncertainty**  
- URM/URME; Coste et al.; Eisenstein et al.; Gao et al. overoptimization; RewardBench  
- Diff: prior UQ is mostly unconditional / non-agentic; we condition on selection and manipulate diversity with mediation.

*(Each citation: one true claim + how we differ. Mentor-check every “no prior work…” sentence.)*

---

## 4. Method / Bench skeleton (procedural — no metaphors)

### Setup
- Frozen agent \(\pi\), tasks \(x\) from verifiable benchmarks.  
- At decision points: candidates \(\{a_i\}\), critic scores \(c(a_i)\), selection `sel` = taken / BoN-\(n\) or top-\(k\%\).  
- Gold \(g \in \{0,1\}\): trajectory = task success (primary); step = MC-rollout path-success (secondary, sensitivity).  
- Uncertainty \(u\): ensemble disagreement (primary); dropout / quantile-width / gen-verifier self-consistency (cheap).  
- Diversity ladder: seed → data → base model → public-critic zoo; measure pairwise error correlation on selected and full candidate sets.

### Primary estimand
\[
\Delta\mathrm{AUROC} = \mathrm{AUROC}(c,u \mid \mathrm{sel}) - \mathrm{AUROC}(c \mid \mathrm{sel})
\]
Also: partial AUC (low-alarm), LR test for \(u\), clustered bootstrap CIs; always control length, step index, agent entropy; report range-restriction (\(c\)–\(u\) correlation under selection).

### Anchor cell (pre-register)
- Base-diverse critics, \(m=4\), ensemble-std, BoN \(n=16\), \(k=5\%\), SWE-bench Verified, one open scaffold.  
- Primary endpoint: \(\Delta\)AUROC at anchor, diverse vs seed.

---

## 5. Experiments skeleton (empty tables first)

| Exp | Claim supported | Design | Metric |
|---|---|---|---|
| E0 | Silent failures exist; trivial features don’t saturate | Stage-0 on selected set | failure rate; AUROC of length/entropy/step |
| E1 | H1 beyond-score | anchor ± \(n\), \(k\) | \(\Delta\)AUROC, LR, partial AUC |
| E2 | H2 diversity mediated | diversity × \(m\) | \(\Delta\)AUROC; error-corr mediation |
| E3 | H3 pressure × diversity | seed vs diverse × \(n\) | slope of \(\Delta\)AUROC vs \(\log n\) |
| E4 | H4 cheap UQ | estimator sweep | incremental \(\Delta\)AUROC |
| E5 | H5 taxonomy | labeled failure classes | per-class \(\Delta\)AUROC |
| E6 | Generality | WebArena / ALFWorld | agreement with SWE |
| E7 | Impact (offline) | selective verify budget \(b\%\) | risk–coverage; failures caught |

**Baselines always present:** score-only; trivial covariates; seed ensemble; single-critic UQ.

---

## 6. Writing rules checklist (apply while drafting)

- [ ] Every non-obvious claim has a citation that *actually* supports it  
- [ ] No novelty exaggeration; gap sentences mentor-verified  
- [ ] No logical leaps; no aspirational “will enable reliable agents” without evidence  
- [ ] Pattern-match style from target venue + closest cited papers  
- [ ] Captions self-contained with takeaway  
- [ ] Results answer RQs; no number dumping  
- [ ] Related Work near end (unless mentor prefers early)  
- [ ] Conclusion = takeaways only; limitations separate  
- [ ] Anonymous links via anonymous.4open.science; no acknowledgements at submit  
- [ ] OpenReview account created now (signup takes time)  
- [ ] Overleaf only after venue chosen (use that template / page limit)  

---

## 7. Immediate next actions (this week)

1. **Mentor sync:** agree title + 3 claims + AAAI-27 vs ICLR-2027 target.  
2. **Lit check:** mentor validates gap sentence vs URM/URME, AgentForesight, AgentPRM, etc.  
3. **Stage-0:** log one scaffold on SWE-bench Verified subset; report silent-failure rate in selected set.  
4. **Do not** ask PI on Slack until mentor has marked a near-final draft.  
5. Create OpenReview profile: https://openreview.net/signup  

---

## 8. Hierarchical writing order (Devi Parikh)

Do **not** jump ahead until mentor converges on earlier stage:

1. ~~Title options~~ → pick with mentor  
2. ~~Paper skeleton~~ (this file §1)  
3. ~~Intro skeleton bullets~~ (this file §2)  
4. Intro filled (sentences) — **next writing task after mentor OK**  
5. Related work skeleton → filled  
6. Approach/bench skeleton → filled (split into two mentor passes)  
7. Results skeleton (empty tables) → filled after experiments  
8. Conclusion + abstract last  

When on Overleaf later: **do not mark mentor/PI comments resolved**; reply “done” and leave visible.
