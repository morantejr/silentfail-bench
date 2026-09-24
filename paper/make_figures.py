#!/usr/bin/env python3
"""All paper figures from committed data. Run from repo root: python3 paper/make_figures.py"""
import json, sys
from pathlib import Path
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
D = Path("silentfail_bench/data/swebench_pilot50"); OUT = Path("paper/figures")
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False, "figure.dpi": 150})
BLUE, RED, GRAY, GOLD = "#2b6cb0", "#c0392b", "#7f8c8d", "#b7791f"

# ---------- Fig 1: the ladder of nulls ----------
rows = [  # (label, u, negc, [ci_lo, ci_hi] of u - (-c) if known)
 ("Primary: prefix (4k), 4 chat RMs", 0.508, 0.621, (-0.373, 0.219)),
 ("Submission window (diff + 3k tail)", 0.458, 0.530, (-0.35, 0.133)),
 ("Patch in view (4k), 4 chat RMs [Kaggle]", 0.494, 0.621, (-0.276, 0.111)),
 ("+ AceCodeRM as 5th auditor", 0.473, 0.621, (-0.296, 0.104)),
 ("Proposal's 8B critic selecting, disjoint panel", 0.445, 0.602, (-0.564, 0.143)),
 ("Code-trained critic selecting, 5-RM panel", 0.542, 0.472, (-0.208, 0.452)),
]
fig, ax = plt.subplots(figsize=(6.4, 3.0))
y = np.arange(len(rows))[::-1]
for yi, (lab, u, nc, ci) in zip(y, rows):
    ax.plot([nc], [yi + 0.15], "s", color=GRAY, ms=6, zorder=3)
    ax.plot([u], [yi - 0.15], "o", color=BLUE, ms=6, zorder=3)
ax.axvline(0.5, color="k", lw=0.8, ls=":"); ax.axvline(0.665, color=GOLD, lw=1.6)
ax.text(0.667, len(rows) - 0.55, "the ruler:\nlines added\nin the patch\n(0.665)", color=GOLD, fontsize=7.5, va="top")
ax.set_yticks(y); ax.set_yticklabels([r[0] for r in rows], fontsize=7.5)
ax.set_xlim(0.35, 0.75); ax.set_xlabel("AUROC for silent failure (12 positives / 22 successes)")
ax.plot([], [], "o", color=BLUE, label="panel disagreement $u$"); ax.plot([], [], "s", color=GRAY, label="selecting critic's own score $-c$")
ax.legend(loc="lower right", fontsize=7.5, frameon=False); fig.tight_layout(); fig.savefig(OUT / "fig_ladder.pdf"); plt.close(fig)

# ---------- Fig 2: critic instability across input windows ----------
tail = [json.loads(l) for l in (D / "pilot_free_scored_tail.jsonl").read_text().splitlines() if l.strip()]
pop = [r for r in tail if r.get("visible_pass") and not r["excluded"]]
k4 = json.loads((D / "gpu_scores/scores_gpu_skywork_skywork_reward_v2_qwen3_4b.json").read_text())["scores"]
cp = np.array([r["c"] for r in pop]); ct = np.array([r["c_tail"] for r in pop])
ck = np.array([k4[f"{r['instance_id']}#{r['selected_candidate_index']}"]["score"] for r in pop])
g = np.array([r["g"] for r in pop])
fig, axs = plt.subplots(1, 2, figsize=(6.4, 2.7))
for ax, yv, name in ((axs[0], ct, "submission window (diff + 3k tail)"), (axs[1], ck, "patch in view, 4k (Kaggle)")):
    for gg, col, lab in ((1, BLUE, "success"), (0, RED, "silent failure")):
        ax.scatter(cp[g == gg], yv[g == gg], s=22, color=col, alpha=0.8, label=lab, edgecolor="none")
    r = np.corrcoef(cp, yv)[0, 1]
    ax.set_title(f"same critic, r = {r:.2f}", fontsize=9); ax.set_xlabel("score on the prefix"); ax.set_ylabel(f"score on the {name}", fontsize=8)
axs[0].legend(fontsize=7.5, frameon=False); fig.suptitle("Skywork-Reward-V2-Qwen3-4B on the same 34 trajectories", fontsize=9, y=1.0)
fig.tight_layout(); fig.savefig(OUT / "fig_instability.pdf"); plt.close(fig)

# ---------- Fig 3: the ruler ----------
cov = json.loads((D / "covariate_baseline.json").read_text())
bars = [("lines added (patch)", 0.665, GOLD), ("step count", 0.657, GOLD), ("patch chars", 0.619, GOLD),
        ("best chat RM (1B, 1 of 6)", 0.636, GRAY), ("primary critic $c$ (prefix)", 0.621, GRAY), ("8B named critic", 0.534, GRAY),
        ("AceCodeRM-7B (code-trained)", 0.405, GRAY), ("panel $u$ (prefix)", 0.508, BLUE), ("panel $u$ (patch in view)", 0.494, BLUE)]
bars.sort(key=lambda b: b[1])
fig, ax = plt.subplots(figsize=(6.4, 2.9))
ax.barh([b[0] for b in bars], [b[1] for b in bars], color=[b[2] for b in bars], height=0.65)
ax.axvline(0.5, color="k", lw=0.8, ls=":"); ax.set_xlim(0.3, 0.75); ax.set_xlabel("AUROC for silent failure (same 34 trajectories, 12 positives)")
for i, b in enumerate(bars): ax.text(b[1] + 0.005, i, f"{b[1]:.3f}", va="center", fontsize=7.5)
ax.tick_params(axis="y", labelsize=8); fig.tight_layout(); fig.savefig(OUT / "fig_ruler.pdf"); plt.close(fig)

# ---------- Fig 4: the pilot ----------
prim = [json.loads(l) for l in (D / "pilot_free_scored.jsonl").read_text().splitlines() if l.strip()]
rd = json.loads((D / "prereg_readout.json").read_text())
fig, axs = plt.subplots(2, 2, figsize=(6.4, 4.6))
ax = axs[0, 0]; h = rd["watch_items"]["selected_index_histogram"]
ax.bar([str(k) for k in h], list(h.values()), color=BLUE); ax.set_title("best-of-4 actually selected", fontsize=9); ax.set_xlabel("chosen candidate index"); ax.set_ylabel("instances")
ax = axs[0, 1]; stages = [("candidates", 164), ("submitted", 138), ("usable committed", 34), ("resolved", 22), ("silent failures", 12)]
ax.barh([s[0] for s in stages][::-1], [s[1] for s in stages][::-1], color=[GRAY, GRAY, GRAY, BLUE, RED][::-1]); ax.set_title("from 164 candidates to 12 positives", fontsize=9)
for i, (n, v) in enumerate(stages[::-1]): ax.text(v + 2, i, str(v), va="center", fontsize=8)
ax.set_xlim(0, 190); ax.tick_params(axis="y", labelsize=8)
ax = axs[1, 0]; pr = rd["M0d_icc_by_repo"]["per_repo_failure_rate"]; usable = [r for r in prim if not r["excluded"]]
cnt = {k: sum(1 for r in usable if r["repo"] == k) for k in pr}
ks = sorted(pr, key=lambda k: -pr[k]); ax.bar([k.split("__")[1] for k in ks], [pr[k] for k in ks], color=RED, alpha=0.85)
for i, k in enumerate(ks): ax.text(i, pr[k] + 0.03, f"n={cnt[k]}", ha="center", fontsize=6.5)
ax.set_ylim(0, 1.2); ax.set_ylabel("failure rate"); ax.set_title(f"per-repo failure rate (ICC = {rd['M0d_icc_by_repo']['icc']:.2f}, 9 repos)", fontsize=9); ax.tick_params(axis="x", rotation=45, labelsize=7)
ax = axs[1, 1]; icc = np.linspace(0, 0.2, 200); m_eff = 129.2; deff = 1 + (m_eff - 1) * icc
ax.plot(icc, 500 / deff, color=BLUE); ax.axvspan(0.033, 0.053, color=GOLD, alpha=0.25, label="public submissions (6 models)"); ax.axvline(0.18, color=RED, ls="--", lw=1, label="this pilot, 9 repos")
ax.set_xlabel("intraclass correlation of failure by repo"); ax.set_ylabel("effective n (of 500)"); ax.set_title("SWE-bench Verified: 12 repos, django 46%", fontsize=9); ax.legend(fontsize=7, frameon=False)
fig.tight_layout(); fig.savefig(OUT / "fig_pilot.pdf"); plt.close(fig)

# ---------- Fig 5: design schematic ----------
fig, ax = plt.subplots(figsize=(6.4, 2.2)); ax.axis("off")
boxes = [(0.02, "SWE-bench\nVerified task"), (0.20, "agent\n(gpt-5-mini)\n× n = 4"), (0.40, "selection critic $c$\n(reward model)\nargmax → commit"),
         (0.62, "disjoint audit\npanel, m = 4\n→ $u$ = disagreement"), (0.82, "gold $g$\nofficial\ntest harness")]
for x, t in boxes:
    ax.add_patch(plt.Rectangle((x, 0.3), 0.16, 0.45, fc="#eef3f8", ec=BLUE, lw=1.2)); ax.text(x + 0.08, 0.525, t, ha="center", va="center", fontsize=7.5)
for x in (0.18, 0.36, 0.56, 0.78): ax.annotate("", xy=(x + 0.02, 0.525), xytext=(x, 0.525), arrowprops=dict(arrowstyle="->", lw=1.2, color="k"))
ax.text(0.5, 0.08, "estimand: does $u$ rank silent failures ($g{=}0$ among committed, agent-declared-done trajectories) beyond $c$?", ha="center", fontsize=8)
ax.set_xlim(0, 1); ax.set_ylim(0, 1); fig.tight_layout(); fig.savefig(OUT / "fig_design.pdf"); plt.close(fig)
print("figures:", sorted(p.name for p in OUT.glob("*.pdf")))
