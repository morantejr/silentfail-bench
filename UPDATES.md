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

## Next entry (pending): exploratory tail-window rescoring
