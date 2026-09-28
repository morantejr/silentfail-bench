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
# Consistent meaning of colour across every figure in the paper:
#   BLUE = panel disagreement (the signal under test)   GRAY = a single critic acting alone
#   GOLD = a structural baseline needing no model        RED  = silent failures / distant comparisons
from matplotlib.lines import Line2D
def legend_key(ax, items, **kw):
    """items: list of (marker, colour, label); marker 'o'/'s'/'D'/'^' or 'bar'."""
    h = [Line2D([], [], marker=(None if m == "bar" else m), color=c, lw=(7 if m == "bar" else 0),
                markersize=6, markerfacecolor=c, markeredgecolor="none", label=l) for m, c, l in items]
    kw.setdefault("fontsize", 7); kw.setdefault("frameon", False); kw.setdefault("handlelength", 1.6)
    return ax.legend(handles=h, **kw)
def chance(ax, x=True, at=0.5, label="chance"):
    (ax.axvline if x else ax.axhline)(at, color="k", lw=0.8, ls=":", zorder=0)
    if x: ax.annotate(label, xy=(at, 1.0), xycoords=("data", "axes fraction"), xytext=(0, 2),
                      textcoords="offset points", ha="center", fontsize=6.5, color="k")

# ---------- Fig 1: the ladder of nulls ----------
rows = [  # (label, u, negc, [ci_lo, ci_hi] of u - (-c)); must match Table 1 / tab:ladder
 ("Primary: prefix (4k), 4 chat RMs", 0.508, 0.621, (-0.373, 0.219)),
 ("  robustness, m=3 family-disjoint", 0.545, 0.621, (-0.39, 0.23)),
 ("Submission window (diff + 3k tail)", 0.458, 0.530, (-0.35, 0.133)),
 ("Patch in view (4k), 4 chat RMs [Kaggle]", 0.494, 0.621, (-0.276, 0.111)),
 ("+ AceCodeRM as 5th auditor", 0.473, 0.621, (-0.296, 0.104)),
 ("Proposal's 8B critic selecting, disjoint panel", 0.455, 0.602, (-0.486, 0.26)),
 ("  ... the same at a true 16k", 0.483, 0.736, (-0.468, 0.283)),
 ("Code-trained critic selecting, 5-RM panel", 0.542, 0.472, (-0.208, 0.452)),
 ("  the code-trained critic at 16k (33 rows)", 0.607, 0.463, (-0.11, 0.63)),
 ("Full trajectory, true 16k (L4), 4 chat RMs", 0.439, 0.621, (-0.384, 0.171)),
 ("LLM judges (gpt-5 family), patch in view", 0.528, 0.621, (-0.30, 0.10)),
 ("  execution-aware (+ last 6k tok)", 0.640, 0.621, (-0.28, 0.16)),
 ("Pruned overlapping windows, $u_{\\max}$ (L4)", 0.686, 0.621, (-0.11, 0.333)),
]
fig, ax = plt.subplots(figsize=(6.8, 4.55))
y = np.arange(len(rows))[::-1]
for yi, (lab, u, nc, ci) in zip(y, rows):
    ax.plot([nc], [yi + 0.12], "s", color=GRAY, ms=5, zorder=3)
    ax.plot([u], [yi - 0.12], "o", color=BLUE, ms=5, zorder=3)
chance(ax)
ax.axvline(0.665, color=GOLD, lw=1.6)
ax.annotate("ruler: lines added (0.665)", xy=(0.665, -1.05), xytext=(0.668, -1.05), color=GOLD, fontsize=6.5, va="center")
ax.set_yticks(y); ax.set_yticklabels([r[0] for r in rows], fontsize=6.2)
ax.set_ylim(-1.5, len(rows) - 0.25)
ax.set_xlim(0.35, 0.80); ax.set_xlabel("AUROC for silent failure (primary: 12/34; re-selection rows 11–12)", fontsize=8)
legend_key(ax, [("o", BLUE, "panel disagreement $u$ (the signal under test)"), ("s", GRAY, "selecting critic's own score $-c$")],
    loc="lower center", bbox_to_anchor=(0.5, -0.20), ncol=2, fontsize=6.5)
fig.tight_layout(); fig.savefig(OUT / "fig_ladder.pdf", bbox_inches="tight"); plt.close(fig)

# ---------- Fig 2: critic instability across input windows ----------
tail = [json.loads(l) for l in (D / "pilot_free_scored_tail.jsonl").read_text().splitlines() if l.strip()]
pop = [r for r in tail if r.get("visible_pass") and not r["excluded"]]
k4 = json.loads((D / "gpu_scores/scores_gpu_skywork_skywork_reward_v2_qwen3_4b.json").read_text())["scores"]
cp = np.array([r["c"] for r in pop]); ct = np.array([r["c_tail"] for r in pop])
ck = np.array([k4[f"{r['instance_id']}#{r['selected_candidate_index']}"]["score"] for r in pop])
g = np.array([r["g"] for r in pop])
fig, axs = plt.subplots(1, 2, figsize=(6.4, 3.0))
for ax, yv, name in ((axs[0], ct, "submission window (diff + 3k tail)"), (axs[1], ck, "patch in view, 4k (Kaggle)")):
    for gg, col, lab in ((1, BLUE, "success"), (0, RED, "silent failure")):
        ax.scatter(cp[g == gg], yv[g == gg], s=22, color=col, alpha=0.8, label=lab, edgecolor="none")
    r = np.corrcoef(cp, yv)[0, 1]
    ax.set_title(f"same critic, r = {r:.2f}", fontsize=9); ax.set_xlabel("score on the prefix"); ax.set_ylabel(f"score on the {name}", fontsize=8)
legend_key(axs[0], [("o", BLUE, "test suite passed"), ("o", RED, "silent failure")], loc="upper center", bbox_to_anchor=(1.12, -0.26), ncol=2, fontsize=7.5); fig.suptitle("Skywork-Reward-V2-Qwen3-4B on the same 34 trajectories", fontsize=9, y=1.0)
fig.tight_layout(); fig.savefig(OUT / "fig_instability.pdf", bbox_inches="tight"); plt.close(fig)

# ---------- Fig 3: the ruler ----------
cov = json.loads((D / "covariate_baseline.json").read_text())
bars = [("lines added (patch)", 0.665, GOLD), ("step count", 0.657, GOLD), ("patch chars", 0.619, GOLD),
        ("best chat RM (1B, 1 of 6)", 0.636, GRAY), ("primary critic $c$ (prefix)", 0.621, GRAY), ("8B named critic", 0.534, GRAY),
        ("AceCodeRM-7B (code-trained)", 0.405, GRAY), ("panel $u$ (prefix)", 0.508, BLUE), ("panel $u$ (patch in view)", 0.494, BLUE),
        ("1B chat RM, full 16k (1 of 7)", 0.712, GRAY), ("panel $u_{\\max}$, pruned windows", 0.686, BLUE)]
bars.sort(key=lambda b: b[1])
fig, ax = plt.subplots(figsize=(6.4, 3.3))
ax.barh([b[0] for b in bars], [b[1] for b in bars], color=[b[2] for b in bars], height=0.65)
chance(ax); ax.set_xlim(0.3, 0.82); ax.set_xlabel("AUROC for silent failure (same 34 trajectories, 12 positives)")
for i, b in enumerate(bars): ax.text(b[1] + 0.005, i, f"{b[1]:.3f}", va="center", fontsize=7.5)
ax.tick_params(axis="y", labelsize=8)
legend_key(ax, [("bar", GOLD, "structural baseline (no model)"), ("bar", GRAY, "one critic on its own"), ("bar", BLUE, "panel disagreement")],
    loc="lower right", ncol=1)
fig.tight_layout(); fig.savefig(OUT / "fig_ruler.pdf", bbox_inches="tight"); plt.close(fig)

# ---------- Fig 4: the pilot ----------
prim = [json.loads(l) for l in (D / "pilot_free_scored.jsonl").read_text().splitlines() if l.strip()]
rd = json.loads((D / "prereg_readout.json").read_text())
fig, axs = plt.subplots(2, 2, figsize=(6.6, 5.2))
ax = axs[0, 0]; h = rd["watch_items"]["selected_index_histogram"]
ax.bar([str(k) for k in h], list(h.values()), color=BLUE); ax.set_title("selection really happened", fontsize=8.5)
ax.set_xlabel("which of the 4 candidates was chosen", fontsize=8); ax.set_ylabel("instances", fontsize=8)
ax = axs[0, 1]; stages = [("candidates", 164), ("submitted", 138), ("usable committed", 34), ("resolved", 22), ("silent failures", 12)]
ax.barh([s[0] for s in stages][::-1], [s[1] for s in stages][::-1], color=[GRAY, GRAY, GRAY, BLUE, RED][::-1]); ax.set_title("164 candidates become 12 failures to study", fontsize=8.5)
for i, (n, v) in enumerate(stages[::-1]): ax.text(v + 2, i, str(v), va="center", fontsize=8)
ax.set_xlim(0, 190); ax.tick_params(axis="y", labelsize=8)
ax = axs[1, 0]; pr = rd["M0d_icc_by_repo"]["per_repo_failure_rate"]; usable = [r for r in prim if not r["excluded"]]
cnt = {k: sum(1 for r in usable if r["repo"] == k) for k in pr}
ks = sorted(pr, key=lambda k: -pr[k]); short = {"scikit-learn": "sklearn", "matplotlib": "matplotlib", "sphinx-doc": "sphinx", "pydata": "xarray", "pytest-dev": "pytest", "pylint-dev": "pylint"}
lbl = [short.get(k.split("__")[0], k.split("__")[1]) for k in ks]
ax.bar(range(len(ks)), [pr[k] for k in ks], color=RED, alpha=0.85)
for i, k in enumerate(ks): ax.text(i, pr[k] + 0.04, f"n={cnt[k]}", ha="center", fontsize=6)
ax.set_xticks(range(len(ks))); ax.set_xticklabels(lbl, fontsize=6.5, rotation=40, ha="right")
ax.set_ylim(0, 1.25); ax.set_ylabel("failure rate", fontsize=8); ax.set_title(f"failure rate by repository (ICC {rd['M0d_icc_by_repo']['icc']:.2f}, 9 repos)", fontsize=8.5)
ax = axs[1, 1]; icc = np.linspace(0, 0.2, 200); m_eff = 129.2; deff = 1 + (m_eff - 1) * icc
ax.plot(icc, 500 / deff, color=BLUE); ax.axvspan(0.033, 0.053, color=GOLD, alpha=0.25, label="public submissions (6 models)"); ax.axvline(0.18, color=RED, ls="--", lw=1, label="this pilot, 9 repos")
ax.set_xlabel("intraclass correlation of failure by repo", fontsize=8); ax.set_ylabel("effective n (of 500)", fontsize=8)
ax.set_title("why 500 instances are not 500 samples", fontsize=8.5); ax.legend(fontsize=6.5, frameon=False, loc="upper right")
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

# ---------- Fig 6: windowed auditing (Amendment 3) ----------
a3 = json.loads((D / "prereg_readout_amendment3.json").read_text())
fig, axs = plt.subplots(1, 2, figsize=(6.6, 3.4), gridspec_kw={"width_ratios": [1, 1.3]})
ax = axs[0]
items = [("$u_{\\max}$", a3["u_max"]["AUROC"], BLUE), ("$u_{\\mathrm{mean}}$", a3["u_mean"]["AUROC"], BLUE), ("$u_{\\mathrm{last}}$", a3["u_last"]["AUROC"], BLUE),
         ("8B critic\nalone", a3["named_critic_mean_window"]["AUROC_negc"], GRAY), ("$u$ on\nprefix", 0.508, "#a9c4e0")]
ax.bar([i[0] for i in items], [i[1] for i in items], color=[i[2] for i in items], width=0.65)
jk = a3["u_max_robustness"]["jackknife_min_max"]; ax.plot([0, 0], jk, color="k", lw=1.2); ax.plot([-0.12, 0.12], [jk[0]] * 2, color="k", lw=1); ax.plot([-0.12, 0.12], [jk[1]] * 2, color="k", lw=1)
ax.axhline(0.665, color=GOLD, lw=1.6); chance(ax, x=False); ax.set_ylim(0.4, 0.82); ax.set_ylabel("AUROC for silent failure", fontsize=8)
ax.text(-0.45, 0.672, "ruler", color=GOLD, fontsize=7, ha="left", va="bottom")
ax.text(-0.45, 0.505, "chance", color="k", fontsize=6.5, ha="left", va="bottom")
ax.set_title("aggregating disagreement over windows", fontsize=8.5)
ax.set_xlabel(f"permutation $p$ = {a3['u_max']['p_perm']:.2f} free, {a3['u_max']['p_perm_within_repo']:.2f} within repo", fontsize=7)
ax.set_xticks(range(len(items))); ax.set_xticklabels([i[0] for i in items], fontsize=7, rotation=20, ha="right")
ax = axs[1]; pc = a3["per_critic_window_aggregates_EXPLORATORY"]
names = {"ray2333_grm_gemma2_2b_rewardmodel_ft": "GRM-2B", "ray2333_grm_llama3_2_3b_rewardmodel_ft": "GRM-3B", "skywork_skywork_reward_v2_llama_3_2_1b": "Skywork-1B",
         "skywork_skywork_reward_v2_qwen3_1_7b": "Skywork-1.7B", "skywork_skywork_reward_v2_qwen3_4b": "Skywork-4B (selector)", "skywork_skywork_reward_v2_llama_3_1_8b": "Skywork-8B (named)", "tiger_lab_acecoderm_7b": "AceCodeRM-7B"}
order = [k for k in names if k in pc]; yy = np.arange(len(order))[::-1]
for yi, k in zip(yy, order):
    ax.plot([pc[k]["AUROC_negmean"]], [yi], "o", color=GRAY, ms=5); ax.plot([pc[k]["AUROC_neglast"]], [yi], "D", color=BLUE, ms=5); ax.plot([pc[k]["AUROC_negmin"]], [yi], "^", color=RED, ms=5)
ax.set_yticks(yy); ax.set_yticklabels([names[k] for k in order], fontsize=7.5); chance(ax); ax.axvline(0.665, color=GOLD, lw=1.6)
ax.set_xlim(0.28, 0.88); ax.set_ylim(-1.5, len(order) - 0.4)
ax.set_xlabel("each critic alone, aggregated over its windows (exploratory)", fontsize=7.5)
legend_key(ax, [("o", GRAY, "mean window"), ("D", BLUE, "last window"), ("^", RED, "lowest window")],
    loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=3)
ax.set_ylim(-0.6, len(order) - 0.4)
fig.tight_layout(); fig.savefig(OUT / "fig_windows.pdf", bbox_inches="tight"); plt.close(fig)

# ---------- Fig 7: test-retest stability (Amendment 4b) ----------
st = json.loads((D / "prereg_readout_amendment4b.json").read_text())
order = [k for k in names if k in st["critics"]]; labels = [names[k] for k in order] + ["panel $u$ (per window)"]
vals = [st["critics"][k] for k in order] + [st["panel_u"]]
fig, ax = plt.subplots(figsize=(6.6, 3.7)); x = np.arange(len(labels)); w = 0.2
for j, (key, lab, col) in enumerate((("r_fmt", "same window, reformatted", BLUE), ("r_shift", "window shifted 128 tokens", "#6fa3d8"), ("r_adj", "adjacent windows (25% overlap)", GRAY), ("r_far", "first vs last window", RED))):
    ax.bar(x + (j - 1.5) * w, [v[key] for v in vals], width=w, color=col, label=lab)
ax.axhline(0, color="k", lw=0.6); ax.axhline(0.8, color=GRAY, lw=0.7, ls=":")
ax.annotate("r = 0.8", xy=(len(labels) - 0.4, 0.81), fontsize=6, color=GRAY, va="bottom", ha="right")
ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=6.8, rotation=28, ha="right"); ax.set_ylabel("Spearman r between the two scorings", fontsize=8); ax.set_ylim(-0.55, 1.12)
ax.legend(fontsize=6.5, frameon=False, ncol=2, loc="upper center", bbox_to_anchor=(0.5, -0.32))
ax.set_title("same content, re-asked (blues) vs. different content (grey, red)", fontsize=9)
fig.tight_layout(); fig.savefig(OUT / "fig_stability.pdf", bbox_inches="tight"); plt.close(fig)

# ---------- Fig 8: replication on 41 repositories (Amendment 4e) ----------
RP = Path("silentfail_bench/data/replication_multilingual")
rr = json.loads((RP / "prereg_readout_replication.json").read_text()); rw = json.loads((RP / "readout_windows_secondary.json").read_text())
ver = {"panel $u$ (prefix 4k)": (0.508, (0.373, 0.643)), "panel $u_{\\max}$ (windows)": (0.686, (0.657, 0.731)), "lines added (ruler)": (0.665, None),
       "best single RM, 16k": (0.712, None), "best LLM judge, patch in view": (0.752, None)}
ml = {"panel $u$ (prefix 4k)": (rr["primary_prefix_u"]["M1_AUROC_u"], tuple(rr["primary_prefix_u"]["CI95_repo_clustered"])),
      "panel $u_{\\max}$ (windows)": (rr["windowed_u_max"]["AUROC"], tuple(rr["windowed_u_max"]["CI95_repo_clustered"])),
      "lines added (ruler)": (rr["ruler_lines_added"]["AUROC"], tuple(rr["ruler_lines_added"]["CI95_repo_clustered"])),
      "best single RM, 16k": (max(rr["single_critics_full16k"].values()), None)}
jr = RP / "prereg_readout_replication_judges.json"
if jr.exists():
    jd = json.loads(jr.read_text()); ml["best LLM judge, patch in view"] = (jd["conditions"]["J1"]["best_single"]["AUROC"], None)
labels = list(ver); yy = np.arange(len(labels))[::-1]
fig, ax = plt.subplots(figsize=(6.6, 3.4))
for yi, lab in zip(yy, labels):
    v, vci = ver[lab]; ax.plot([v], [yi + 0.16], "o", color=GRAY, ms=6)
    if lab in ml:
        m, mci = ml[lab]; ax.plot([m], [yi - 0.16], "o", color=BLUE, ms=6)
        if mci: ax.plot(list(mci), [yi - 0.16] * 2, color=BLUE, lw=1.2)
chance(ax); ax.axvline(0.665, color=GOLD, lw=1.2, ls="--")
ax.annotate("pilot's structural baseline", xy=(0.665, -0.95), xytext=(0.672, -0.95), color=GOLD, fontsize=6.5, va="center")
ax.set_yticks(yy); ax.set_yticklabels(labels, fontsize=8); ax.set_xlim(0.3, 0.85); ax.set_ylim(-1.7, len(labels) - 0.4)
ax.set_xlabel("AUROC for silent failure")
legend_key(ax, [("o", GRAY, "pilot: Verified, 9 repos, 12 failures"), ("o", BLUE, f"replication: Multilingual, {rr['n_repos']} repos, {rr['silent_failures']} failures (bars = 95% CI)")],
    loc="lower center", bbox_to_anchor=(0.5, -0.34), ncol=1)
ax.set_title("the same measurements on three times the repositories", fontsize=9, loc="left", pad=12)
fig.tight_layout(); fig.savefig(OUT / "fig_replication.pdf", bbox_inches="tight"); plt.close(fig)

# ---------- Fig 9: the unselected-candidate control (Amendment 4f) ----------
c4f = json.loads((Path("silentfail_bench/data/random_arm/prereg_readout_amendment4f.json")).read_text())
fig, axs = plt.subplots(1, 2, figsize=(6.4, 2.6))
for ax, key, title in ((axs[0], "failure_rate", "silent-failure rate"), (axs[1], "AUROC_negc", "selector's own score, AUROC for failure")):
    x = np.arange(2); w = 0.34
    rnd = [c4f[b][f"{key}_random"] for b in ("verified", "multilingual")]
    com = [c4f[b][f"{key}_committed"] if key == "failure_rate" else c4f[b]["AUROC_negc_committed_same_rows"] for b in ("verified", "multilingual")]
    ax.bar(x - w/2, rnd, w, color=GRAY, label="candidate the critic passed over"); ax.bar(x + w/2, com, w, color=BLUE, label="candidate the critic committed")
    for i in range(2):
        ax.text(x[i] - w/2, rnd[i] + 0.01, f"{rnd[i]:.2f}", ha="center", fontsize=7); ax.text(x[i] + w/2, com[i] + 0.01, f"{com[i]:.2f}", ha="center", fontsize=7)
    ax.set_xticks(x); ax.set_xticklabels([f"Verified\n(n={c4f['verified']['n_paired']} paired)", f"Multilingual\n(n={c4f['multilingual']['n_paired']} paired)"], fontsize=8)
    ax.set_title(title, fontsize=9); ax.set_ylim(0, 1.0 if key == "AUROC_negc" else 0.65)
    if key == "AUROC_negc":
        ax.axhline(0.5, color="k", lw=0.8, ls=":")
        ax.annotate("chance", xy=(1.45, 0.515), fontsize=6.5, va="bottom", ha="right", color="k")
axs[0].legend(fontsize=6.5, frameon=False, loc="upper left")
fig.tight_layout(); fig.savefig(OUT / "fig_control.pdf"); plt.close(fig)
print("figures:", sorted(p.name for p in OUT.glob("*.pdf")))
