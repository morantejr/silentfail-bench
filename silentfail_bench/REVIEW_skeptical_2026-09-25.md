# Skeptical assessment of SilentFail-Bench (draft v0.3), 2026-09-25

Written against the paper as it stands after the 41-repository replication, not
against the pre-replication numbers in the review request. Citations were
resolved from primary pages on 2026-09-25; unverifiable items are flagged.

## 0. Premise correction

The request describes the Verified-only state (u_max 0.686, ruler 0.665, best
RM 0.712, judges 0.72–0.75). All four of those fell to chance on SWE-bench
Multilingual (114 instances, 41 repositories, 35 positives): u_max 0.541
[0.40, 0.68]; lines-added 0.508 [0.35, 0.65]; best RM at 16k 0.577; judges
0.58–0.66. The prefix-panel null replicated (0.565 [0.43, 0.71] vs the critic's
own 0.560). The evidence base is therefore two benchmarks, ~50 distinct
repositories, 47 positives, executed gold throughout.

## 1. Reconstruction

Original hypothesis (proposal). Conditional on best-of-n selection by critic c,
disagreement u among a checkpoint-disjoint audit panel carries residual
information about executed failure beyond c; lower panel error correlation
yields larger residual value (E2). Confirmatory: held-out ΔAUROC of a logistic
M_{c+u} over M_c. E3–E7 exploratory (pressure×diversity, cheap u, taxonomy,
other domains, selective-verification budget curves).

Current hypothesis (v0.3). Reduced to M1–M3 raw-score AUROCs because the
held-out logistic was never eligible (<10 test-split positives); the claim is
that u is not above chance and not above c, on one window of any length, for
reward models and single-family LLM judges, on two benchmarks. Mechanism claim
from v0.1 ("critics hold no judgment") withdrawn after test–retest.

Design. gpt-5-mini via mini-swe-agent, k=4, Skywork-Reward-V2-Qwen3-4B selects
on a 4k prefix, four chat RMs (1–3B) audit, official swebench harness gold.
Amendments 1–4e were each dated before their data existed.

Confirmatory analyses actually run: primary M1–M3 (Verified); R-P1–R-P3
(Multilingual). Amendments 1–3 and 4a–4d were pre-registered with bars but are
secondary. Exploratory: covariate ruler, per-critic window aggregates, 8k
ladder point, m5 panels.

Main findings. Silent-failure rates 35% and 45%; u ≈ c ≈ 0.5–0.6 everywhere;
Verified point estimates that beat 0.665 (u_max, ruler, 1B RM, judges) did not
replicate; critics are test–retest stable (r ≥ 0.98); nothing adds
out-of-fold value beyond structure on Verified; on Multilingual the structure
itself is at chance.

Actual contribution implied. A pre-registered, executed-gold null for
disagreement-based auditing of critic-selected SWE-agent trajectories, plus a
documented catalogue of instrument failures. Not a mechanism paper, not a
method paper.

Discrepancies between presentation and evidence.
- The abstract and §"What the critics carry" still foreground the Verified
  single-critic numbers (0.71, 0.72–0.75) that did not replicate. Lead with
  the two-benchmark result; demote those to "Verified-only, not replicated".
- The title "Disagreement Is Not the Signal" is a universal; the evidence is
  one agent, off-the-shelf ≤8B chat RMs and one LLM family. Scope it.
- E2 (diversity vs error correlation), the proposal's mechanism experiment,
  was never run; the paper should say so rather than let the m=3 family-
  disjoint robustness row stand in for it.
- The "premise holds" framing (35%/45% silent failures) is presented as a
  finding about selection; without an unselected control (failure rate of a
  random candidate), it does not show that selection *concentrates* failure.
  The chosen-index histogram shows selection happened, not that it enriched
  failures. This is the largest gap between claim and evidence in the paper.

## 2. Novelty audit

Conceptual: LOW. "Correlated critics share blind spots; consensus/disagreement
does not track correctness; BoN over-optimizes a proxy" is established
(Gao et al. 2022; Eisenstein et al. 2023; Coste et al. 2023; Goel et al. 2025;
Kim et al. 2025; Kohli 2026; Hossain et al. 2026; Afrin & Shihab 2026). The
principle "evaluate verifiers conditionally on the selection policy" is
implicit in all BoN reward-hacking work and explicit in Emadi 2026 ("success
counted before an answer is chosen") and in Prop. 40 of the Behavioral
Credibility Trilemma (2605.25739: BoN degrades calibration even under a
perfect ranker). It is underexplored as an *evaluation protocol* for auditors,
not as an idea. Classification: closely anticipated.

Experimental design: MODERATE. Disjoint selector/panel, evaluation on the
committed population, executed gold, structural baselines, pre-registration
with dated amendments, test–retest, and a wide-repository replication in one
study. No located paper does this combination on SWE trajectories.
Classification: incrementally to plausibly novel (the combination, not any
element).

Statistical/methodological: LOW–MODERATE. Repo-clustered bootstrap on paired
AUROC, within-repository permutation for max-statistics, LORO incremental
model, null-calibrated power: all standard tools applied correctly; the
observation that Verified's 12-repository structure caps effective n at
~65–95 of 500 is useful and, as far as located, unstated elsewhere.
Classification: incrementally novel.

Empirical: MODERATE. Two-benchmark, 50-repository, 47-positive null for panel
disagreement on selected SWE trajectories; the non-replication of the
lines-added ruler; the test–retest stability numbers for reward models on
long windows. Classification: plausibly novel as measurements.

Practical: MODERATE. "Do not build an escalation trigger from off-the-shelf
RM disagreement; a single frontier judge reading the diff is the best cheap
signal and it is only ~0.66" is actionable. Advani 2026 reaches a compatible
conclusion (lightweight detectors beat LLM judges for false success) from a
different population.

Benchmark/artifact: LOW. The bundle of 620 scored trajectories with executed
gold and seven critics' scores at three lengths is a modest artifact; not a
benchmark.

## 3. Closest novelty threats

1. SWE-RM (Shum et al., Dec 2025). Establishes an execution-free RM for SWE
   trajectories, evaluated on discrimination, calibration and BoN test-time
   scaling on Verified. Overlap: RM scoring of SWE trajectories, BoN. Does not
   test: auditing the selected population with a second, disjoint signal, or
   disagreement. Gap defensible — but SWE-RM is also the obvious *selector and
   auditor* this study did not run; its absence is the first reviewer question.
2. Locating Hidden Failures / Scout (Rahman et al., Sep 2026). A 4B trained
   verifier selects among candidate SWE/computer-use trajectories, beats
   frontier LMs at finding failures. Overlap: verifier-selected trajectories,
   failure taxonomy. Does not test: residual value of a second signal after
   selection. Threat: shows a *trained* auditor works where off-the-shelf
   panels fail; the paper must position itself as "off-the-shelf disagreement",
   not "auditing".
3. From Confident Closing to Silent Failure (Advani, Jun 2026). ~12k
   trajectories; lightweight statistical detectors beat LLM judges for
   agent-claimed false success. Overlap: the phenomenon and the structural-
   baseline finding. Does not test: critic selection or panels; population is
   agent-claimed, not critic-selected. Gap defensible but the ruler finding is
   now anticipated — and did not replicate here.
4. Nine Judges, Two Effective Votes (Kohli, May 2026); Agreement Overstates
   Evidence (Hossain et al., Sep 2026); Evaluator Ensembles Under Reward
   Hacking (Afrin & Shihab, Aug 2026). Together they establish that judge
   panels have few effective votes and that common-mode error is not
   identifiable from internal scores. Overlap: the theoretical reason
   disagreement should fail. Do not test: SWE trajectories, executed gold,
   post-selection. The paper is an instance of their prediction; cite them as
   such and claim the measurement, not the insight.
5. Blind to the Pivotal Vote / Unanimity Without Persuasion (Shu, Aug 2026).
   Code-correctness judge panels; verification helps only on margin-pivotal
   cases; debate erases disagreement. Overlap: disagreement on code. Does not
   test: selected agent trajectories. Implication: E7-style routing by raw
   disagreement is the wrong design; margin-stratified analysis is missing
   from this paper and would be cheap.
6. Inference-Time Reward Hacking (Khalaf et al. 2025); From Curiosity to
   Caution (Yu et al., ICLR 2026). BoN over-optimization and pessimistic
   fixes. Overlap: the winner's-curse premise. Do not test: post-hoc audit
   signals. Gap defensible.
7. Last Step Matters (Li et al., Aug 2026); Agentic Uncertainty Reveals
   Agentic Overconfidence (Kaddour et al., Feb 2026); Fail-Fast Restart-Smart
   (Wang et al., Aug 2026). Agent-side uncertainty and prefix-trained failure
   monitors. Overlap: failure prediction on SWE/agent trajectories. Do not
   test: external critic panels after selection. Fail-Fast's 0.6B monitor at
   5% FPR is a stronger baseline than anything in this paper and should be
   named as the trained alternative.
8. AgentLens (Microsoft, May 2026). 10.7% of *passing* SWE trajectories are
   "lucky passes". Overlap: quality beyond pass/fail. Threatens the gold, not
   the estimand: some g=1 rows are not genuine successes.
9. Are "Solved Issues" Really Solved? (Wang, Pradel, Liu 2025); Auditing
   Reward Hackability (Rajan, Jun 2026). 7.8% of accepted patches fail
   developer tests; 25–28% of Verified/R2E tasks have weak coverage. Threat:
   label noise in the positive class; failure rates are lower bounds and
   AUROCs are attenuated. Must be stated.

Not threats, flagged: 2502.10325 is Choudhury's AgentPRM framework note (not
Xi et al.'s AgentPRM, which is 2511.08325); 2605.02964 is Thaman's Reward
Hacking Benchmark (mislabeled "resolve and identify" — irrelevant to the
estimand); the ICLR proceedings hash resolves to From Curiosity to Caution
(same as 2604.04648); OpenReview Tf4lRAOGkj = "On the Robustness of Reward
Models for Language Model Alignment" (BSR regularization; tangential);
OKWlVPHeW1 = SRM virtual-agent step-wise reward benchmark (tangential);
PFR4E8583W = Agent Error Taxonomy/Benchmark (ALFWorld/GAIA/WebShop; adjacent);
GazlTYxZss = Who&When failure attribution (adjacent); 0xpakqqTbe = RExBench
(irrelevant). GBKtBQj52q and FbDuesI6tD could not be resolved (bot-check on
both the site and the API); do not cite them until resolved. Harnessing
Disagreement (McDonnell et al.) is tabular ML on intrusion/readmission data;
cite only for the term "correlated agreement blindness". DiscoUQ is agent
ensembles on QA benchmarks; adjacent, not a threat. Layla is a 2023 chat
fine-tune: exclude entirely.

## 4. What the negative result supports

Supported (findings):
- Disagreement among off-the-shelf reward models (1–32B, chat- and code-
  trained) and one LLM family, on one window of 4k–16k tokens or the max over
  pruned windows, does not rank executed failures of gpt-5-mini's critic-
  selected SWE trajectories above ~0.6 on either benchmark. Upper bounds:
  M1 ≤ 0.71 (CI), M3 ≤ +0.22.
- Critic scores are test–retest stable and content-driven; the failure is not
  noise.
- Patch size is not a portable predictor.
- Selection with a 4B chat RM is real (index spread) but not shown to enrich
  failures (no unselected control).

Plausible mechanisms (not shown): shared blind spots suppress disagreement
(consistent with Kohli/Hossain/Afrin); evidence representation dominates
nominal diversity (consistent with the window results but confounded with
critic identity and length); selection shifts the verifier-evaluation
distribution (consistent with range restriction, corr(c,u)=0.11, but E2 was
not run).

Speculation: "disagreement universally fails"; "structural features provide
stronger warning signals" (dead after replication); any claim about trained
auditors or frontier-panel diversity.

Too small to determine: anything at the level of ±0.05 AUROC; equivalence to
chance; ordering among single critics.

## 5. Statistical audit

- Effective n: Verified 34 rows/9 repos (ICC 0.18 → ~15 effective);
  Multilingual 77/37 (ICC not reported — report it; likely small). Pooled
  ~110 rows, 47 positives, ~46 repositories.
- Positives: 12 + 35. Single-AUROC 95% half-widths ≈ 0.25 and ≈ 0.14.
- Clustering: handled by repo-clustered bootstrap (scores fixed, repos
  resampled) and within-repo permutation. Adequate for the paired
  comparison; the bootstrap is anti-conservative under small cluster counts
  (9 repos) — the paper already reports the null-calibrated power check;
  state the coverage caveat.
- Repeated configurations: ~25 analyses on the same 12 Verified positives;
  no multiplicity correction; the paper says so. The replication is the
  correction. Report a family-wise statement: of k pre-registered
  predictions, j passed on Verified, 0 on Multilingual.
- AUROC only: add AUPRC (prevalence 0.35/0.45 makes it informative) and a
  calibration/decision curve for the one signal that matters operationally
  (the best judge), per Vickers & Elkin.
- Out-of-fold: LORO incremental model is right; the registered difficulty
  control was a near-oracle (0.95) — the paper declares it; keep the
  structure-only variant as primary and say why.
- Overfitting: 5–7 features on 34 rows is at the edge even LORO; on the
  pooled 110 it is fine — pool.
- Dependence among trajectories: candidates of one instance are not
  independent, but only one per instance enters the population; fine.
- Selection-induced shift: real, unquantified. The missing analysis is
  AUROC(u) on a *random* candidate per instance (the same critics have
  already scored all candidates at 4k; gold for one random candidate per
  instance costs ~$5 on Modal). This single addition turns "conditional on
  selection" from a design choice into a measured contrast.
- Equivalence: the study supports failure-to-reject and an upper bound, not
  equivalence. Say "no evidence of, and an upper bound of".
- More trajectories from the same repositories: no. More repositories: yes,
  but the Multilingual run already did that; the next lever is the selector.

Smallest defensible analysis set for the final paper: (1) pooled and per-
benchmark M1–M3 for the primary protocol; (2) u_max with within-repo
permutation, both benchmarks; (3) best single RM and best judge, both
benchmarks, with CIs; (4) lines-added, both benchmarks; (5) LORO incremental
model, pooled, structure-only base; (6) test–retest; (7) the unselected-
candidate contrast (new). Everything else to an appendix table.

## 6. Publication potential

AAAI main (negative-results-friendly but wants generality): not competitive as
is. Objections: one agent; selector is a 4B chat RM no one deploys; SWE-RM/
Scout not run as selector or auditor; no unselected control; E2 not run;
title overclaims. Would change: a deployed-grade selector, a trained auditor
row, the unselected contrast, second agent. The negative result would be
sufficient *with* those; replication alone is not.

NeurIPS/ICML/ICLR main: not competitive. A null on an estimand the field
already expects to be null, with off-the-shelf small critics, is a
workshop paper at these venues regardless of rigor.

Findings-style (ACL Findings / EMNLP Findings / TMLR): plausible now,
strengthened considerably by the unselected contrast and a SWE-RM row. TMLR
in particular values pre-registration and honest scoping.

Agent-evaluation workshop (NeurIPS/ICLR workshops on agents, eval,
reliability): competitive now. The instrument-failure catalogue and the
pre-registration trail are assets there.

Software-engineering venue (ICSE/FSE NIER, EMSE, ASE industry/short): plausible
as an empirical study; reviewers will ask for patch-level qualitative
analysis of the 47 failures and for AgentLens-style quality labels.

## 7. Recommended framing

Primary: C, stated as a protocol and instantiated: "auditors must be
evaluated on the population the selector commits, against the selector's own
score, with executed gold". Supporting: A scoped to what was tested
(off-the-shelf critic disagreement carries no residual signal on selected
SWE trajectories, two benchmarks), and F (instrument validation) as a
methods section, not a headline. Drop B (not shown), D (confounded), E (did
not replicate).

Claim (one sentence): On critic-selected SWE-agent trajectories with executed
gold, disagreement among disjoint off-the-shelf critics carries no failure
signal beyond the selector's own score, across 50 repositories, two
benchmarks, three input lengths and two critic classes.

Title: "Auditing the Winner: Critic Disagreement Does Not Recover Silent
Failures in Selected SWE-Agent Trajectories".

Abstract (≈200 words): When an agent commits the best of n attempts under a
critic, the committed trajectory is where the critic's confidence and its
errors coincide. A common proposal is to audit it with a disjoint panel of
critics and escalate on disagreement. We pre-registered and ran this test
with executed gold on SWE-bench Verified (41 instances, 9 repositories) and
SWE-bench Multilingual (114 instances, 41 repositories): best-of-4 selection
by a reward model, a checkpoint-disjoint panel, and the official test harness.
Silent failures — committed, agent-declared-complete trajectories that fail
hidden tests — were 35% and 45% of the population. Panel disagreement ranked
them at AUROC 0.51 and 0.57, indistinguishable from the selecting critic's own
score and within its interval of chance, and this held across 4k–16k
contexts, pruned overlapping windows, seven reward models from 1B to 32B
including code-trained ones, two re-selection regimes, and a single-family
panel of LLM judges. Point estimates that cleared a structural baseline on
Verified (windowed disagreement 0.69; patch size 0.67; one 1B critic 0.71) all
fell to chance on 41 repositories. Test–retest shows the critics are stable
(r ≥ 0.98) and content-driven: the failure is in what disagreement measures.
We argue auditors must be evaluated on the selected population against the
selector, report an upper bound on residual value, and document the
instrument failures that would otherwise have produced a positive result.

Contribution bullets:
- A selection-conditional evaluation protocol for audit signals — selected
  population, selector baseline, executed gold, pre-registered bars — with
  the first two-benchmark instantiation on SWE-agent trajectories.
- A replicated null: off-the-shelf critic disagreement adds ≤ +0.22 AUROC
  (95% upper bound) over the selecting critic across 50 repositories, seven
  reward models, three lengths and one judge family; Verified-only positives
  do not travel.
- Diagnostics and instrument record: test–retest stability of long-window
  reward-model scores; the repository-clustering ceiling on effective n for
  SWE-bench Verified; six instrument failures that each produced a plausible
  false finding.

Limitations paragraph: One agent (gpt-5-mini, 40 steps) and one selector
family; the selector is a 4B chat reward model rather than a deployed-grade
verifier such as SWE-RM, so the selected population is that of a weak
selector. No unselected-candidate control isolates the effect of selection
itself. Panel members are ≤8B chat reward models plus 7B/32B code-trained
ones; LLM judges are a single vendor. Gold is the harness verdict, which
accepts some incorrect patches and rejects some correct ones (Wang et al.
2025; Rajan 2026), so failure rates are lower bounds and AUROCs attenuated.
Forty-seven positives bound single AUROCs to roughly ±0.14; ~25 analyses on
the Verified positives are uncorrected and the replication is the
correction. The proposal's diversity-vs-error-correlation experiment was not
run. The generation cap was exceeded by $3.30 for the reason declared.

"What is genuinely new here?" The measurement, not the idea: nobody had put a
disjoint audit panel on the population a critic actually committed, with
executed gold, structural baselines and pre-registered bars, and asked what
it adds over the selector — and then re-asked on 41 repositories when the
first answer flattered the panel. The answer, an upper bound of about +0.2
AUROC, is the contribution; the correlated-judge literature predicted it and
this is the first executed-gold instance on agent trajectories.

## 8. Decisive follow-up (smallest that changes the venue)

Goal: turn "off-the-shelf disagreement fails" into "here is what selection-
conditional auditing looks like with contemporary components", ~$150 and two
weeks, no new method.

- Repositories/failures: keep both benchmarks; add SWE-bench Multimodal or a
  second Multilingual draw to reach ≥ 80 positives across ≥ 60 repositories.
- Agents: two (gpt-5-mini; one open model, e.g. Qwen3-Coder via mini-swe-
  agent) — different failure structure, same pipeline.
- Selectors: three regimes on the same candidate pools: the 4B chat RM
  (current), SWE-RM 30B-A3B, and a frontier judge. This is the axis that
  makes C testable: does residual audit value depend on selector strength?
- Critic families as auditors: the current panel; SWE-RM as a single auditor;
  Scout if released; two cross-vendor frontier judges. Report single-auditor
  and disagreement rows side by side.
- Evidence views: 4k prefix, full 16k, pruned windows (u_max) — keep; drop
  8k and tail.
- Structural baselines: lines added, step count, files touched; and the
  agent's own final confidence (Kaddour et al.).
- Unselected control (the one cheap addition that could change a scientific
  conclusion): gold for one random candidate per instance; AUROC of each
  signal on random vs selected rows. This is the direct test of the
  principle.
- Calibration/selective review: E7 as pre-registered — precision/recall at
  5/10/20% review budgets, decision curves, for the best signal; margin-
  stratified (pivotal-vote) analysis of whether disagreement helps on
  close-call selections only.
- Pre-registration: bars as now; primary = ΔAUROC(best auditor − selector)
  by selector regime, pooled, repo-clustered; secondary = disagreement rows.
- Matrix: 2 agents × ~230 instances × k=4 ≈ 1,800 trajectories (~$70);
  gold ~460 + 230 random (~$20 Modal); scoring on Modal (~$30).

Replicate the null again? No — done. Formalize the principle? Yes, briefly,
as the protocol (§7), not as theory. Build an evidence-aware auditor? Only if
the SWE-RM/Scout rows show residual value; otherwise it is a different
paper. Margin-conditioned routing? As an analysis, yes; as a method, no.

## 9. Claim–evidence map

- Claim: selection concentrates silent failures. Evidence: 35%/45% failure
  rates among committed rows; index spread. Closest: Khalaf 2025; Yu et al.
  2026. Threat: no unselected control. Safe wording: "a third to a half of
  committed trajectories are silent failures". Needed: random-candidate gold.
- Claim: panel disagreement carries no residual signal. Evidence: M1 0.51/
  0.57, M3 −0.11/+0.005, CIs; ladder. Closest: Kohli 2026, Hossain 2026,
  Afrin & Shihab 2026. Threat: weak selector, small critics. Safe wording:
  "off-the-shelf critic disagreement, upper bound +0.22". Needed: SWE-RM row.
- Claim: nothing beyond structure out of fold. Evidence: LORO +u −0.03, +c
  +0.02 (Verified). Closest: Advani 2026. Threat: 34 rows; ruler itself did
  not replicate. Safe wording: "on Verified"; pool with Multilingual.
- Claim: critics are stable, content-driven. Evidence: r_fmt 0.98–0.99,
  r_shift 0.86–0.95, r_adj 0.2–0.56. Closest: Lost in the Middle (position
  effects) — different. Threat: none serious. Safe wording: as is.
- Claim: single critics carry signal at length. Evidence: Verified 0.71/
  0.72–0.75; Multilingual 0.58/0.66. Threat: did not replicate for RMs;
  judges shrank. Safe wording: "a frontier judge reading the diff is the best
  single signal, ~0.66–0.75 depending on benchmark".
- Claim: repository clustering caps effective n on Verified. Evidence: ICC
  0.033–0.053 from six public runs; design-effect arithmetic. Closest: none
  located. Threat: ICC estimated on resolution, not silent failure. Safe
  wording: as is, with that caveat.
- Claim: instrument failures produce false findings. Evidence: six documented
  cases with oracle controls. Closest: Rajan 2026; AgentLens. Safe wording:
  as is; keep it a methods section.
- Claim: gold is trustworthy. Evidence: oracle pairs pass. Threat: Wang et al.
  2025; Rajan 2026; AgentLens lucky passes. Needed: one paragraph and, if
  cheap, developer-test re-grading of the 47 positives.

## 10. Blunt verdict

1. Broad idea novel? No. Predicted by the correlated-judge and BoN reward-
   hacking literature; several 2026 papers state the mechanism.
2. Exact study novel? Yes, as a measurement: disjoint audit panel on the
   committed population, executed gold, structural baselines, pre-registered,
   replicated across benchmarks. Not novel as an insight.
3. Relevant in 2026? Yes — escalation triggers for coding agents are being
   built now, and the closest trained-auditor papers (Scout, Fail-Fast) make
   "what does a second signal add after selection" a live question.
4. Current evidence publishable? Yes at a workshop or Findings/TMLR level;
   not at a main track without the selector axis and the unselected control.
5. Venue level supported now: agent-evaluation workshop (strong);
   Findings/TMLR (plausible); AAAI main (weak); NeurIPS-class main (no).
6. Single most valuable addition: run SWE-RM (or a frontier judge) as the
   selector on the existing candidate pools and grade one random candidate
   per instance — it converts the paper's principle into a measured contrast
   and answers the first two reviewer objections for under $30.
7. Continue, reframe, expand, or abandon? Reframe now (title, abstract,
   demote Verified-only positives, add the unselected control), submit to a
   workshop or TMLR; expand along the selector axis only if a main-track
   submission is the goal. Do not abandon: the negative result is clean and
   the record is unusually honest, which is the asset. Do not pivot to a
   method.

## Artifacts A–C

AirLLM: compute, not science. It streams decoder layers for *generation* and
does not expose sequence-classification/value heads; reward-model scoring
would need custom wiring, and layer streaming at 16k tokens would make a
32B critic take minutes per row. Modal already solved the compute problem
for ~$15. Mention only in a reproducibility appendix if at all.

Layla: a 2023 Llama-2-7B ShareGPT chat model. Exclude.

Laya: a 421M ModernBERT decision head trained for calibrated probabilities on
short text; not evaluated on code or agent traces; context far below a
trajectory. As an auditor over raw trajectories it is a category error. As a
calibrated P(fail) head over a feature vector (structure + critic scores +
patch summary) it is a proper-scoring-rule classifier — legitimate, but that
is a method paper on a different question ("can a trained head detect silent
failures?"), which Fail-Fast and Scout already pursue with more data. It does
not change the scientific conclusion of this paper; leave it out.
