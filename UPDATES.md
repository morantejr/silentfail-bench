# Research log — SilentFail-Bench

One entry per stage. Each records what was done, what was learned, and what
changed in the design as a result. Detail lives in the linked documents; this
file is the spine. Newest at the bottom.

---

## Stage 0 · Proposal and freezes — 2026-09-22

**Done.** Proposal revised for gold, leakage, pilot and pressure
(`SilentFail_AAAI_Proposal.pdf`). Design frozen in
[`silentfail_bench/FREEZES.md`](silentfail_bench/FREEZES.md): whole-trajectory
`c`, best-of-*n* selection, disjoint selection critic vs audit panel,
trajectory-level gold, repository-level held-out split, primary estimand
ΔAUROC = AUROC(M_c+u) − AUROC(M_c).

**Learned.** Four implementation defects would have corrupted any run
([`DEFECTS.md`](silentfail_bench/DEFECTS.md)): the power function reported
~0.40 "power" under the true null; hyphenated repo names were truncated,
breaking the repo split; splits used builtin `hash()` and were not
reproducible; IRLS diverged under separation. All fixed with regression tests
(suite 5 → 18).

**Changed.** Power is now a null-calibrated, repo-clustered bootstrap.
The 0.03 MDE placeholder is not viable: ~1,900 trajectories for ΔAUROC≈0.045,
before clustering.

---

## Stage 0 · Free-compute pilots on HumanEval+ (pilots 1–3b) — 2026-09-22/23

**Done.** No container runtime on the host, so SWE-bench gold was
impossible; HumanEval+ substituted (real executed gold, multi-step agent,
sandboxed). Pilot 1: 40 tasks, 2 LLM judges. Pilot 2: same, 4 true reward
models on Apple Silicon. Pilot 3: selection pressure n=16. Pilot 3b: real
reward-model selection critic over the same 640 candidates.

**Learned.**
- The oracle must be validated: a missing `numpy` under `-S` scored *every*
  trajectory as a failure — a fake 100% failure rate caught only because the
  reference solutions were run through the harness first.
- A coarse LLM-judge critic (values {5,6,8}) silently turns best-of-16 into
  first-draw: 39/40 committed trajectories identical to n=4. Selection had
  never happened in pilots 1–3. **`c_sel` must be a continuous reward model.**
- HumanEval+ yields ~8% silent failures regardless of pressure: 3 positives
  per 40 tasks. Not enough to test anything.
- A "Goodhart signature" reported from pilot 2 (critic below chance) was an
  artifact of the coarse judge and was withdrawn (`MENTOR_NOTE.md` §10).
- Reward-model throughput: ≤3B models score in ~1 s on Metal; an 8B pages
  and takes 134 s. Panels are viable locally at small sizes.

**Changed.** Pre-registration introduced
([`PREREGISTRATION.md`](silentfail_bench/PREREGISTRATION.md)): visible-pass
population, positive class, metrics, predictions, and the rule that <10
positives yields no conclusion in either direction.

---

## Stage 0 · Repo clustering and profile B infrastructure — 2026-09-23

**Done.** SWE-bench Verified stood up on the laptop without admin rights:
Colima + Lima + Docker CLI as standalone binaries, Rosetta for the
x86_64-only images, `swebench` 5.x under a uv-managed Python 3.12,
mini-swe-agent as the scaffold ([`PROFILE_B_SETUP.md`](silentfail_bench/PROFILE_B_SETUP.md)).
Gold validated with three oracle controls (gold patch resolves; empty patch
short-circuits; no-op patch runs tests and fails).

**Learned.**
- SWE-bench Verified is 500 instances over **12 repos**, django 46%. The ICC
  measured from six public submissions on this scaffold is 0.033–0.053 →
  effective n ≈ 64–95, not 500. More trajectories do not help; repos are the
  scarce resource (`MENTOR_NOTE.md` §8).
- Local 7B and 14B agents cannot drive the scaffold to a submission (the 14B
  finds the fix site and narrates it 22 times). Generation is
  capability-bound. gpt-5-mini resolves the smoke instance for $0.02.

**Changed.** Generation moves to the API (gpt-5-mini, $0.25/trajectory hard
cap); selection, audit and gold stay local. Two power routes presented to the
mentor without a recommendation: re-scope the MDE, or fund n≈1,000–2,000.

---

## Stage 0 · SWE-bench pilot, 41 instances, real gold — 2026-09-24

**Done.** 50 instances × 4 candidates launched unattended; generation
stalled at 41 when macOS swap exhausted the host disk (declared deviation).
Reward-model selection and 4-model audit, then the official harness on the
34 usable committed patches: **22 resolved, 12 unresolved, 0 errors.**
$3.65 total. Analysis pre-registered before any gold existed
([`PREREGISTRATION_SWEBENCH_PILOT.md`](silentfail_bench/PREREGISTRATION_SWEBENCH_PILOT.md)),
executed by
[`scripts/prereg_readout_swebench.py`](silentfail_bench/scripts/prereg_readout_swebench.py).

**Learned** (`MENTOR_NOTE.md` §12).
- **Supply solved:** 35% silent-failure rate among committed trajectories;
  ~146 positives projected at 500 instances. GO on the pre-registered rule.
- **First inference-eligible test of H1 (12 positives): not supported.**
  AUROC(u) = 0.51, AUROC(−c) = 0.62, difference −0.11 (CI [−0.37, +0.22]).
  Panel disagreement is at chance; the selecting critic's own score carries
  modest signal. The m=3 robustness panel agrees.
- The Goodhart signature is absent with a real critic: `c` ranks its
  mistakes slightly below its successes, not above.
- ICC by repo 0.18 on 9 repos — unstable, but above the public estimate.
- A third manufactured uniform rate was caught (0/41 from image-pull failures
  through the SDK) before it could be mistaken for a finding.

**Bounding caveat.** Every critic scored a 4,096-token *prefix* of ~40k-char
trajectories — never the patch. Fair between `c` and `u`; blind to the thing
graded. An exploratory rescoring on the final diff + last 3,000 tokens was
pre-declared before gold and is reported in the next entry.

**Changed.** The confirmatory design must fix its scoring input and confront
the repo bound before any larger run; "GO" answers powering, not design.

---

## Stage 0 · Exploratory rescoring with the patch in view — 2026-09-24

**Done.** Pre-declared before gold: same 34 trajectories and 12 positives,
same critics, input = final diff + last 3,000 tokens instead of the
4,096-token prefix (`MENTOR_NOTE.md` §13).

**Learned.**
- The null is **not** a truncation artifact. With the patch in view,
  AUROC(u) fell to **0.46** and AUROC(−c) to **0.53** — both at or below
  chance. The critic's prefix score and window score are uncorrelated (0.03).
- The prefix's modest 0.62 was probably a *difficulty* signal from the
  problem statement, not a judgment of the solution.
- **Chat-preference reward models carry ~no information about patch
  correctness on SWE-bench trajectories**, in either window.

**Changed.** H1 is closed *for this class of critic on this task*. The
confirmatory run of this design is not worth its $40. The open question moves
to critics that actually score code — execution-feedback or process reward
models — which is a different instrument. That is the decision for the
mentor: pivot the critic class, or write the negative result.

---

## Stage 0 · Amendment 2 — GPU rescoring, the design as written — 2026-09-24

**Done.** Six reward models on Kaggle's free 2×T4, patch in view (4k
fallback declared), including the proposal's named Skywork-8B critic for
the first time. Re-selection by that critic; 20 changed patches graded by the
harness. Exploratory covariate baseline (`MENTOR_NOTE.md` §14).

**Learned.**
- Audit-only: every chat RM at or below chance on the 12 silent failures;
  panel `u` 0.49. The primary critic's prefix and patch scores correlate at
  0.18.
- **The proposal's own design** — 8B named critic selecting, disjoint panel
  auditing, real gold — gives `u` = 0.445, `−c` = 0.602, M3 = −0.157
  (CI [−0.56, +0.14]). Not supported.
- **A ruler beats the critics:** lines added (0.665) and step count (0.657)
  outrank every reward model tested.
- **AceCodeRM-7B (code-trained, execution-feedback preferences): 0.405** —
  below chance on the 12 silent failures. A2-P1 fails. Adding it to the
  panel leaves `u` at 0.473.

**Changed.** The negative result now holds across input window, model size,
critic identity, selection regime, **and critic class** — the code-trained
critic fails too. The confirmatory run of this design is not warranted. Open:
multi-window `u_max` (Amendment 3, pre-registered, running on the L4).

---

## L4 · Amendments 2 (true 16k), 3 (windows), 4 (reviewer-driven) — 2026-09-24

**Done.** Algoverse L4 (24 GB): all seven critics at a true 16k and 8k on all
164 candidates; Amendment 3 pruned overlapping windows on the 34 committed
rows; Amendment 4 pre-registered from a skeptical-reviewer pass and run:
(4a) incremental value beyond structure, (4b) window test–retest, (4c) LLM
judges (OpenAI, $1.96), (4d) 8k ladder point. Draft v0.2. Detail:
`MENTOR_NOTE.md` §15.

**Learned.**
- **Amendment 3 passed all three predictions**: `u_max` 0.686 > 0.665. Then
  bounded: permutation p 0.04 free / 0.21 within repo; jackknife 0.657–0.731;
  median two windows after pruning.
- Longer context helps single critics (1B RM 0.712 at 16k), not
  disagreement (panel 0.439 at 16k).
- LLM judges with the patch in view reach 0.72–0.75; their disagreement
  0.53; seeing the agent's own test runs does not help.
- Test–retest r 0.98–0.99: **the "critics hold no judgment" mechanism is
  withdrawn.** Critics are stable and content-driven; disagreement is the
  wrong statistic.
- No critic adds out-of-fold value beyond patch size (+u −0.03, +c +0.02).
  The registered difficulty control was a near-oracle (0.95) and is
  declared a defect.

**Changed.** The paper's claim narrows from "no detector" to "disagreement is
not the signal": critics carry a moderate signal individually; panel
disagreement on a single window carries none; windowed `u_max` is a marginal,
underpowered positive. Next decisive step is a wide-repository replication
(Amendment 4e), not more Verified trajectories.

---

## Modal · 32B critic, remaining gold, replication launched — 2026-09-25

**Done.** Modal replaces Colima and the L4: AceCodeRM-32B in fp16 on an
A100-80GB; official gold for the 9 re-selected patches through a runner that
mirrors the harness (oracle pair passed); Amendment 4e revised to SWE-bench
Multilingual (41 repositories) and pre-registered; generation started.

**Learned.** 32B code-trained critic: 0.557 (A2-P1 fails again). The named
8B critic re-selecting at 16k: `u` 0.483, its own score 0.736 — the
dissociation once more.

**Changed.** Nothing in the claim; two loose ends closed. The replication
decides between "null survives" and "narrower positive".

---

## Replication · Amendment 4e on SWE-bench Multilingual — 2026-09-25

**Done.** 456 trajectories, 114 instances, 41 repositories, nine languages;
agent sandboxes and official gold on Modal (3 oracle pairs passed); seven
critics at 4k, 16k and windowed; the pre-registered R-P1..R-P3 readout.
Draft v0.3 (`paper/main.pdf`, 12 pages, 8 figures).

**Learned.** 45.5% silent failures on 77 usable rows (35 positives, 37
repos). Panel `u` 0.565 [0.43, 0.71] vs the critic's 0.560 — **the null
replicates** (R-P1). `u_max` 0.541 [0.40, 0.68], within-repo p 0.53 — **the
windowed positive does not** (R-P2). The lines-added ruler fell to 0.508:
a Verified regularity, not a law. Every reward model 0.36–0.58; LLM judges
0.58–0.66 with disagreement 0.514.

**Changed.** End state fixed: **null survives**. The pivot is closed; the
paper is the negative result, now on 53 repositories and 47 positives
combined. Remaining work is writing: citations, mentor read, AAAI cut.

---

## Review and Tier 1 · draft v0.4 — 2026-09-25

**Done.** Skeptical-reviewer assessment with 63 verified references
(`silentfail_bench/REVIEW_skeptical_2026-09-25.md`). Tier 1 of its
recommendations: retitle and rescope; related work with 64 citations;
trained-auditor paragraph; limitations from the review; and the
pre-registered **unselected-candidate control** (Amendment 4f), graded on
Modal for both benchmarks.

**Learned.** With a 4B chat reward model selecting, the committed candidate
fails no more often than one passed over (−3 / −4 points; CIs include 0):
**best-of-4 did not concentrate failure.** On Verified the selector's own
score ranks failures at 0.81 among unselected candidates but 0.61 among
committed ones — selection consumed the critic's signal, the
selection-conditional gap measured directly. On Multilingual the critic has
no signal in either arm. Disagreement is at chance off-selection as well.

**Changed.** The paper claims a silent-failure *rate*, not *concentration*;
its protocol claim (evaluate auditors on the selected population) is now
backed by a measured 0.2 gap; the reviewer's "weak selector" objection is
now a stated fact. End state unchanged: null survives. Next decision is the
mentor's: submit (workshop / TMLR) or run the selector axis (Tier 2).

---

## Tier 1.5 · draft v0.5 — 2026-09-25

**Done.** The proposal's registered confirmatory endpoint computed by
leave-one-repository-out on 47 pooled positives; operating points at review
budgets; alternative disagreement statistics; Multilingual table; control
figure; six instrument failures enumerated; intro discipline; title in
failure-to-reject form.

**Learned.** ΔAUROC(M_{c+u} − M_c) = **−0.03 [−0.10, +0.04]** pooled: adding
disagreement to the selector's score lowers out-of-fold AUROC. No
alternative u statistic exceeds 0.60. At a 20% review budget u catches 25%
/ 23% of failures; the selector's own score 33% / 29%.

**Changed.** The upper bound on disagreement's residual value is +0.04 on
the registered endpoint (was +0.22, per-protocol). Nothing else.

---

## Verification pass · draft v0.6 — 2026-09-25

**Done.** Two checks requested by external reads, no new data.

**Learned.**
- **Leak fix verified.** Stored `u` reproduces exactly from training-only
  rank norms (max |diff| 0.0); the leaky variant differs by 0.023 and would
  have given 0.523 not 0.508. Every reported `u` is post-fix.
- **The endpoint's bound was estimator-dependent.** Four ways of computing
  ΔAUROC(M_c+u − M_c) pooled: −0.119, −0.030, −0.014, **+0.013**. Signs
  disagree; all intervals include zero; loosest upper bound **+0.12**.

**Changed.** The paper quotes +0.12, not +0.04, and reports all four
estimators instead of one. The conclusion now says "no *replicable* signal"
and names the Verified u_max = 0.686 exception explicitly. Weaker claim,
estimator-independent.

---

## Next entry (pending): mentor decision — submit v0.6 or run Amendment 5 (selector axis, declared, not run)
