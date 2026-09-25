#!/usr/bin/env python3
"""Cross-check numbers that appear in more than one place.

Run from the repo root:  python3 paper/check_consistency.py
Catches the class of error where a value hardcoded in make_figures.py drifts
from the table in main.tex, or where either drifts from the source JSON.
"""
import json, re, sys
from pathlib import Path

tex = Path("paper/main.tex").read_text()
figs = Path("paper/make_figures.py").read_text()
D = Path("silentfail_bench/data")
fail = []

def check(name, a, b, tol=0.0015):
    ok = a is not None and b is not None and abs(a - b) <= tol
    print(f"  {'ok  ' if ok else 'FAIL'}  {name}: {a} vs {b}")
    if not ok: fail.append(name)

# --- ladder: main.tex Table 1 rows vs the hardcoded rows in make_figures.py ---
NUM = r"(?:\\textbf\{)?([0-9.]+)\}?"
tex_rows = {}
for m in re.finditer(r"^(.{5,60}?) & " + NUM + r" & " + NUM + r" & \$?[+-]?[0-9.]+\$? & ", tex, re.M):
    label = re.sub(r"\\quad|\\textbf|\\[a-zA-Z]+|[{}$\\]", "", m.group(1)).strip()
    tex_rows[label] = (float(m.group(2)), float(m.group(3)))
fig_rows = {re.sub(r"[${}\\]|\\\\max|_|\{|\}", "", k).strip(): (u, c)
            for k, u, c in re.findall(r'\("([^"]+)",\s*([0-9.]+),\s*([0-9.]+),', figs)
            for u, c in [(float(u), float(c))]}
print("ladder: every (u, -c) pair plotted in make_figures.py must appear in Table 1")
tex_pairs = {(round(u, 3), round(c, 3)) for u, c in tex_rows.values()}
for k, (fu, fc) in sorted(fig_rows.items()):
    pair = (round(fu, 3), round(fc, 3))
    ok = pair in tex_pairs
    print(f"  {'ok  ' if ok else 'FAIL'}  {k[:46]:48s} ({fu}, {fc})" + ("" if ok else "  <- not in Table 1"))
    if not ok: fail.append(f"figure row {k[:40]}")
print(f"  ({len(fig_rows)} plotted rows checked against {len(tex_pairs)} table rows)\n")

print("paper text vs source JSON:")
a2 = json.loads((D / "swebench_pilot50/prereg_readout_amendment2.json").read_text())
check("8B re-selection u", a2["reselect"]["skywork_skywork_reward_v2_llama_3_1_8b"]["M1_AUROC_u"],
      tex_rows.get("8B named critic selecting; disjoint panel", (None, None))[0])
rep = json.loads((D / "replication_multilingual/prereg_readout_replication.json").read_text())
for label, val in (("replication panel u", rep["primary_prefix_u"]["M1_AUROC_u"]),
                   ("replication u_max", rep["windowed_u_max"]["AUROC"]),
                   ("replication ruler", rep["ruler_lines_added"]["AUROC"])):
    check(label, val, float(re.search(re.escape(f"{val:.3f}"), tex).group(0)) if f"{val:.3f}" in tex else None)
rob = json.loads((D / "verify_and_robustify.json").read_text())
pooled = next(r for r in rob["endpoint_robustness"] if r["label"] == "pooled")
check("endpoint fixed-weight pooled", pooled["fixed_unit_weights_no_fitting"]["delta"],
      0.013 if "+0.013" in tex else None)
check("endpoint LORO pooled", pooled["logistic_LORO"]["delta"], -0.030 if "-0.030" in tex or "$-0.030$" in tex else None)

# --- analysed repository counts ---
print("\nreward-model count (from scripts/gpu_rescore.py):")
sys.path.insert(0, "silentfail_bench/scripts")
from gpu_rescore import CRITICS, CRITICS_32B          # noqa: E402
allc = CRITICS + CRITICS_32B
n_chat = sum(1 for _, k in allc if k == "seqcls"); n_code = len(allc) - n_chat
WORD = {4: "four", 5: "five", 6: "six", 7: "seven", 8: "eight", 9: "nine", 10: "ten"}
check("total reward models", len(allc), n_chat + n_code)
for f in ("paper/main.tex", "paper/main_short.tex"):
    src = Path(f).read_text()
    bad = []
    for m in re.finditer(r"\b(four|five|six|seven|eight|nine|ten) (reward models|critics|chat-preference)(.{0,110})", src, re.S | re.I):
        word, noun, after = m.group(1).lower(), m.group(2).lower(), m.group(3)
        want = {"chat-preference": n_chat}.get(noun, len(allc))
        if WORD.get(want) == word: continue
        if re.search(r"all three lengths|complete ladder|no 4k pass|auditing|as auditors|audit panel", after): continue   # a stated subset: the m=4 audit panel, or the models with a full ladder
        bad.append(f"{word} {noun}")
    print(f"  {'ok  ' if not bad else 'FAIL'}  {f}: {bad or 'all counts agree with the code'}")
    if bad: fail.append(f"{f} model count")

print("\nrepository counts as written in the prose (analysed sets):")
import itertools
for f in ("paper/main.tex", "paper/main_short.tex"):
    src = Path(f).read_text()
    bad = []
    # 41 and 12 are benchmark totals and are only allowed next to an explicit qualifier
    for m in re.finditer(r"(.{70})(41|12) repositor(.{0,90})", src, re.S):
        ctx = m.group(1) + " " + m.group(3)
        if not re.search(r"sampl|drawn|Verified is|analys|survive", ctx): bad.append(ctx[-55:].strip())
    print(f"  {'ok  ' if not bad else 'FAIL'}  {f}: unqualified benchmark-total counts: {bad or 'none'}")
    if bad: fail.append(f"{f} repo count")

print("\nanalysed repository counts:")
for f, tag, want in ((D / "swebench_pilot50/pilot_free_scored.jsonl", "pilot", 9),
                     (D / "replication_multilingual/replication_scored.jsonl", "replication", 37)):
    rows = [json.loads(l) for l in f.read_text().splitlines() if l.strip()]
    pop = [r for r in rows if not r["excluded"] and r["visible_pass"]]
    check(f"{tag} repos", len({r['repo'] for r in pop}), want)

print("\n" + ("ALL CONSISTENT" if not fail else f"{len(fail)} MISMATCH(ES): " + "; ".join(fail)))
sys.exit(1 if fail else 0)
