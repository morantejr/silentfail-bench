#!/usr/bin/env python3
"""Cross-check numbers that appear in more than one place.

Run from the repo root:  python3 paper/check_consistency.py
Catches the class of error where a value hardcoded in make_figures.py drifts
from the table in main.tex, or where either drifts from the source JSON.
"""
import json, re, sys
from decimal import Decimal, ROUND_HALF_UP
import numpy as np
from pathlib import Path

tex = Path("paper/main.tex").read_text()
figs = Path("paper/make_figures.py").read_text()
D = Path("silentfail_bench/data")
fail = []

def _pdf_pages(path):
    """Read a PDF's pages and CLOSE the file. The bite harness runs this checker ~230 times
    in one process; an unclosed reader per run exhausts the descriptor limit and the
    checker crashes mid-sweep, which the harness cannot tell from a caught perturbation."""
    import pypdf
    with open(path, "rb") as fh:
        return [pg.extract_text() or "" for pg in pypdf.PdfReader(fh).pages]

def check(name, a, b, tol=0.0015):
    ok = a is not None and b is not None and abs(a - b) <= tol
    print(f"  {'ok  ' if ok else 'FAIL'}  {name}: {a} vs {b}")
    if not ok: fail.append(name)

# --- ladder: main.tex tab:ladder rows vs the hardcoded rows in make_figures.py ---
NUM = r"(?:\\textbf\{)?([0-9.]+)\}?"
# Scope to the ladder table only. Matching every tabular in the paper falsely
# treats endpoint / budget tables as missing figure rows.
_ladder_anchor = tex.find("label{tab:ladder}")
_ladder_chunk = tex[_ladder_anchor - 4500:_ladder_anchor] if _ladder_anchor >= 0 else ""
_tab_start = _ladder_chunk.rfind("\\begin{tabular}")
ladder_tex = _ladder_chunk[_tab_start:] if _tab_start >= 0 else ""
tex_rows = {}
for m in re.finditer(r"^(.{5,80}?) & " + NUM + r" & " + NUM + r" & \$?[+-]?[0-9.]+\$? & ", ladder_tex, re.M):
    label = re.sub(r"\\quad|\\textbf|\\[a-zA-Z]+|[{}$\\]", "", m.group(1)).strip()
    if not label or label.lower().startswith("configuration"):
        continue
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
print(f"  ({len(fig_rows)} plotted rows checked against {len(tex_pairs)} ladder-table rows)")
print("ladder: every tab:ladder (u, -c) pair must also be plotted (the reverse direction)")
fig_pairs = {(round(u, 3), round(c, 3)) for u, c in fig_rows.values()}
for lab, (tu, tc) in sorted(tex_rows.items()):
    pair = (round(tu, 3), round(tc, 3))
    ok = pair in fig_pairs
    print(f"  {'ok  ' if ok else 'FAIL'}  {lab[:46]:48s} ({tu}, {tc})" + ("" if ok else "  <- not plotted"))
    if not ok: fail.append(f"unplotted table row {lab[:40]}")
print()

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
WORD = {1: "one", 2: "two", 3: "three", 4: "four", 5: "five", 6: "six", 7: "seven", 8: "eight", 9: "nine", 10: "ten"}
check("total reward models", len(allc), n_chat + n_code)
for f in ("paper/main.tex", "paper/main_short.tex"):
    src = Path(f).read_text()
    bad = []
    for m in re.finditer(r"\b(four|five|six|seven|eight|nine|ten) (reward models|critics|chat-preference)(.{0,110})", src, re.S | re.I):
        word, noun, after = m.group(1).lower(), m.group(2).lower(), m.group(3)
        want = {"chat-preference": n_chat}.get(noun, len(allc))
        if WORD.get(want) == word: continue
        if re.search(r"all three lengths|complete ladder|no 4k pass|auditing|as auditors|audit panel|scored on this axis", after): continue   # a stated subset: the m=4 audit panel, or the Amendment 13 axis panel   # a stated subset: the m=4 audit panel, or the models with a full ladder
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

# --- Amendment 8: failure taxonomy and lucky-pass bound vs their source JSON ---
print("\nfailure taxonomy (E5) vs data/readout_failure_taxonomy.json:")
tax = json.loads((D / "readout_failure_taxonomy.json").read_text())
ROWS = [("No progress on the bug, broke nothing", "no progress on the bug, broke nothing"),
        ("Broke something else, bug unfixed", "broke something else, bug unfixed"),
        ("Fixed the bug, broke something else", "fixed the bug, broke something else"),
        ("Partial fix: some target tests green", "partial fix"),
        ("Patch did not apply", "patch did not apply")]
for tex_label, json_key in ROWS:
    m = re.search(re.escape(tex_label) + r" & (\d+) & (\d+) & (\d+) \\\\", tex)
    if not m:
        print(f"  FAIL  taxonomy row missing from Table: {tex_label}"); fail.append(f"taxonomy row {json_key}"); continue
    tv, tm, tp = (int(x) for x in m.groups())
    jv = tax["verified"]["classes"].get(json_key, 0); jm = tax["multilingual"]["classes"].get(json_key, 0)
    jp = tax["pooled_classes"].get(json_key, 0)
    ok = (tv, tm, tp) == (jv, jm, jp) and tv + tm == tp
    print(f"  {'ok  ' if ok else 'FAIL'}  {json_key[:40]:42s} table ({tv},{tm},{tp}) vs json ({jv},{jm},{jp})")
    if not ok: fail.append(f"taxonomy {json_key}")
check("taxonomy total failures", tax["pooled_n_failures"], 47 if "47 silent failures" in tex else None)
# the one class above the inference floor, quoted in the prose
mlc = tax["multilingual"]["per_class"]["no progress on the bug, broke nothing"]
check("above-floor class AUROC_u", mlc["AUROC_u"], 0.534 if "0.534" in tex else None)
print(f"  {'ok  ' if not mlc['below_inference_floor'] else 'FAIL'}  the quoted class is above the inference floor (n={mlc['n']})")
if mlc["below_inference_floor"]: fail.append("quoted taxonomy class below floor")
# guard-test counts must match the analysed populations
gv, gm = tax["verified"]["guard_tests"], tax["multilingual"]["guard_tests"]
for txt, got in ((f"{gv['pop_with_single_f2p_test']} of {tax['verified']['n_analysed']} on Verified", gv),
                 (f"{gm['pop_with_single_f2p_test']} of {tax['multilingual']['n_analysed']} on Multilingual", gm)):
    ok = txt in tex
    print(f"  {'ok  ' if ok else 'FAIL'}  guard-test phrase present: '{txt}'")
    if not ok: fail.append(f"guard-test phrase '{txt}'")
tot = gv["pop_with_single_f2p_test"] + gm["pop_with_single_f2p_test"]
totn = tax["verified"]["n_analysed"] + tax["multilingual"]["n_analysed"]
ok = f"{tot} of {totn} analysed rows" in tex
print(f"  {'ok  ' if ok else 'FAIL'}  pooled guard-test count: '{tot} of {totn} analysed rows'")
if not ok: fail.append("pooled guard-test count")

print("\nlucky-pass bound vs data/readout_luckypass.json:")
lp = json.loads((D / "readout_luckypass.json").read_text())
for tag, bench in (("Verified", "verified"), ("Multilingual", "multilingual")):
    d = lp[bench]["among_passes"]["disjoint_from_reference_patch"]
    ok = f"{d['n']} of {d['of']} passes on {tag}" in tex or f"{d['n']} of {d['of']} on {tag}" in tex
    print(f"  {'ok  ' if ok else 'FAIL'}  disjoint passes on {tag}: {d['n']} of {d['of']}")
    if not ok: fail.append(f"disjoint passes {tag}")
P = lp["pooled"]
def phrase(name, want):
    """Anchor to the exact sentence, not a bare number: these values also occur in the tables."""
    ok = want in tex
    print(f"  {'ok  ' if ok else 'FAIL'}  {name}: {'found' if ok else 'MISSING'} -> {want[:74]}")
    if not ok: fail.append(f"luckypass {name}")

def money(x): return f"{x:+.3f}".replace("+", "+").replace("-", "-")
rate = P["suspect_rate_of_passes"] * 100
phrase("pooled suspect rate", f"pooled {rate:.1f}\\%")
# The two rates in the detailed lucky-pass paragraph must agree. The version
# history now mentions 6.2% too, so stop before the sensitivity paragraph.
_lp0 = tex.index("label{sec:luckypass}")
_lp1 = tex.index("The endpoint survives the worst case.", _lp0)
_sec = tex[_lp0:_lp1]
n_rate = _sec.count(f"{rate:.1f}\\%")
ok = n_rate == 2 and r"\textbf{1.6\%}" in _sec
print(f"  {'ok  ' if ok else 'FAIL'}  suspect rates in the detailed paragraph: heuristic {n_rate} mentions, inspection 1.6%")
if not ok: fail.append("luckypass suspect rate restated")

df = lp["multilingual"]["among_failures_for_contrast"]["disjoint_from_reference_patch"]
phrase("disjoint-failure contrast", f"{df['n']} of {df['of']} Multilingual failures are disjoint")

AL, DR, RL = P["as_labelled"], P["drop_suspect_passes"], P["relabel_suspect_as_failure"]
def m(v): return f"${v:+.3f}$".replace("+", "+")
lo, hi = DR["logistic_LORO"]["CI95"]
phrase("drop LORO with interval",
       f"from {m(AL['logistic_LORO']['delta'])} to {m(DR['logistic_LORO']['delta'])} [{m(lo)}, {m(hi)}]")
phrase("drop unfitted estimator",
       f"from {m(AL['fixed_unit_weights']['delta'])} to {m(DR['fixed_unit_weights']['delta'])}")
rl, ru = RL["logistic_LORO"]["delta"], RL["fixed_unit_weights"]["delta"]
print(f"  {'ok  ' if abs(rl - ru) < 1e-9 else 'FAIL'}  relabel deltas coincide, as the prose claims ({rl} vs {ru})")
if abs(rl - ru) >= 1e-9: fail.append("luckypass relabel 'under both'")
phrase("relabel delta", f"gives {m(rl)} under both")
phrase("AUROC_u under all three scenarios",
       f"falls from {AL['raw_AUROC_u']:.3f} to {DR['raw_AUROC_u']:.3f} and to {RL['raw_AUROC_u']:.3f}")
# the estimator must still reproduce the paper's published endpoint on the unmodified labels
rb = next(r for r in rob["endpoint_robustness"] if r["label"] == "pooled")
check("luckypass reproduces published LORO", AL["logistic_LORO"]["delta"], rb["logistic_LORO"]["delta"])
check("luckypass reproduces published unit", AL["fixed_unit_weights"]["delta"], rb["fixed_unit_weights_no_fitting"]["delta"])

# --- S-P1's second condition and the Amendment 5 narration ---
print("\nS-P1 second condition vs data/readout_sp1b.json:")
sp = json.loads((D / "readout_sp1b.json").read_text())
short = Path("paper/main_short.tex").read_text()
for bench, tag in (("verified", "Verified"), ("multilingual", "Multilingual")):
    r = sp[bench]
    want = f"{r['committed_failure_rate']:.3f} against {r['random_failure_rate']:.3f} on {tag}"
    ok = want in tex
    print(f"  {'ok  ' if ok else 'FAIL'}  paired rates on {tag}: {'found' if ok else 'MISSING'} -> {want}")
    if not ok: fail.append(f"S-P1b rates {tag}")
    lo, hi = r["paired_difference_CI"]
    want2 = f"${r['paired_difference']:+.3f}$ [${lo:+.2f}$, ${hi:+.2f}$]".replace("$+0.00$", "$0.00$")
    ok2 = want2 in tex
    print(f"  {'ok  ' if ok2 else 'FAIL'}  paired difference on {tag}: {'found' if ok2 else 'MISSING'} -> {want2}")
    if not ok2: fail.append(f"S-P1b difference {tag}")
    # the registered condition must actually be met for the paper to say S-P1 passes
    print(f"  {'ok  ' if r['S_P1b_committed_below_random'] else 'FAIL'}  S-P1b met on {tag} (paper claims S-P1 passes on both)")
    if not r["S_P1b_committed_below_random"]: fail.append(f"S-P1b not met {tag}")
    # the paper claims the reverse discordance never happens
    print(f"  {'ok  ' if r['discordant']['committed_fails_random_passes'] == 0 else 'FAIL'}  zero reverse-discordant pairs on {tag}")
    if r["discordant"]["committed_fails_random_passes"]: fail.append(f"reverse discordance {tag}")

print("\nAmendment 5 must not be narrated as a clean null:")
banned = [("S-P3 does not hold", "the registered criterion was met, then overruled"),
          ("null now holds inside the regime", "the interval contains the registered +0.05"),
          ("deliberately not run", "Amendment 5 was run")]
for phrase, why in banned:
    hits = [f for f, src in (("main.tex", tex), ("main_short.tex", short)) if phrase in src]
    print(f"  {'ok  ' if not hits else 'FAIL'}  '{phrase}' absent ({why}){'' if not hits else ' -> still in ' + ', '.join(hits)}")
    if hits: fail.append(f"banned phrase '{phrase}'")
# anchored, because the bare word occurs in several places including this appendix's own history row
required = [("main.tex", tex, "It is inconclusive as to equivalence"),
            ("main.tex", tex, "\\emph{consistent with} the null and does not establish it"),
            ("main_short.tex", short, "\\emph{consistent with} the null and does not establish it")]
for f, src, phrase in required:
    ok = phrase in src
    print(f"  {'ok  ' if ok else 'FAIL'}  {f}: {'found' if ok else 'MISSING'} -> {phrase[:62]}")
    if not ok: fail.append(f"{f} missing '{phrase[:34]}'")

print("\njudge claims vs data/readout_judges_gpt6.json:")
jg = json.loads((D / "readout_judges_gpt6.json").read_text())
best = {}
for bench in ("verified", "multilingual"):
    sj = jg[bench]["single_judges"]
    best[bench] = max(sj, key=lambda k: sj[k][0])
ok = best["verified"] == best["multilingual"] == "gpt-6-sol"
print(f"  {'ok  ' if ok else 'FAIL'}  best single judge is gpt-6-sol on both ({best})")
if not ok: fail.append("best judge identity")
ok = "best ranker is\n\\texttt{gpt-6-sol}" in tex or "best ranker is \\texttt{gpt-6-sol}" in tex
print(f"  {'ok  ' if ok else 'FAIL'}  paper names gpt-6-sol as the best ranker")
if not ok: fail.append("paper names wrong best ranker")
ok = "strongest single detector in the whole study" not in tex
print(f"  {'ok  ' if ok else 'FAIL'}  the 'two-thirds' claim no longer says 'whole study' (gpt-6-sol reaches {jg['multilingual']['single_judges']['gpt-6-sol'][0]})")
if not ok: fail.append("two-thirds contradiction")
# calibration: the paper must not call the best ranker the most overconfident
_batch2 = json.loads((D / "readout_pi_batch2.json").read_text())
ece = {b: {k: v["ECE"] for k, v in _batch2[b]["CAL_judge_calibration"]["judges"].items()} for b in ("verified", "multilingual")}
worst_ml = max(ece["multilingual"], key=ece["multilingual"].get)
ok = worst_ml == "gpt-5-mini" and "largest error belongs to \\texttt{gpt-5-mini}" in tex
print(f"  {'ok  ' if ok else 'FAIL'}  worst-calibrated on the replication is {worst_ml}, and the paper says so")
if not ok: fail.append("calibration attribution")

# --- Amendment 5 on the CORRECTED population, and the audit findings ---
print("\nAmendment 5, corrected population vs data/selector_axis/readout_amendment5_corrected.json:")
a5 = json.loads((D / "selector_axis/readout_amendment5_corrected.json").read_text())
cm = a5["corrected"]["multilingual"]; cv = a5["corrected"]["verified"]
pm = a5["as_published"]["multilingual"]
def want(phrase, label):
    ok = phrase in tex
    print(f"  {'ok  ' if ok else 'FAIL'}  {label}: {'found' if ok else 'MISSING'} -> {phrase[:66]}")
    if not ok: fail.append(f"A5 {label}")
want(f"({cm['n']} rows, {cm['failures']} failures, {cm['repositories']} repositories)", "corrected population")
want(f"$\\mathrm{{AUROC}}(u) = {cm['AUROC_u']:.3f}$", "corrected AUROC_u")
want(f"0.706 to {cm['committed_AUROC_detect']:.3f}, or {cm['S_P2_gap']:.3f}", "corrected S-P2 gap")
e = cm["estimators"]
want(f"${e['logistic_LORO']['delta']:+.3f}$ from the leave-one-repository-out logistic", "corrected LORO")
# this value appears in both the abstract and the body; require every occurrence to be the right one
_uw = f"${e['fixed_unit_weights']['delta']:+.3f}$ [${e['fixed_unit_weights']['CI95'][0]:+.2f}$, ${e['fixed_unit_weights']['CI95'][1]:+.2f}$]"
_n = tex.count(_uw)
print(f"  {'ok  ' if _n == 2 else 'FAIL'}  corrected unfitted stated identically in abstract and body (found {_n}, want 2) -> {_uw}")
if _n != 2: fail.append("corrected unfitted occurrences")
want(f"reaches $+{cm['upper_bound_vs_margin']['upper']:.3f}$", "corrected upper bound")
# verdicts the paper asserts must match the computed booleans
for label, got, claim in (("S-P2 fails on Multilingual", cm["S_P2_passes"], "S-P2 fails on both benchmarks"),
                          ("S-P2 fails on Verified", cv["S_P2_passes"], "S-P2 fails on both benchmarks"),
                          ("S-P3 cond1 not met", cm["S_P3_cond1_u_above_0.60"], "S-P3 fails on its first condition")):
    ok = (got is False) and claim in tex
    print(f"  {'ok  ' if ok else 'FAIL'}  {label}: computed={got}, paper says '{claim[:38]}'")
    if not ok: fail.append(f"A5 verdict {label}")
# the three estimators must actually agree in sign, as the paper claims
signs = {np.sign(e[k]["delta"]) for k in e} if False else {(1 if e[k]["delta"] > 0 else -1) for k in e}
ok = ("agree in sign" not in tex) or len(signs) <= 2
print(f"  {'ok  ' if ok else 'FAIL'}  estimator deltas: {[e[k]['delta'] for k in e]}")
# the published LORO anomaly the paper says dissolves
want(f"+{pm['estimators']['logistic_LORO']['delta']:.3f}$ [$+0.21$, $+0.52$]", "published anomaly quoted")
ok = pm["estimators"]["logistic_LORO"]["baseline_degenerate"] and not cm["estimators"]["logistic_LORO"]["baseline_degenerate"]
print(f"  {'ok  ' if ok else 'FAIL'}  baseline degenerate before ({pm['estimators']['logistic_LORO']['baseline_AUROC']}) and sound after ({cm['estimators']['logistic_LORO']['baseline_AUROC']})")
if not ok: fail.append("A5 baseline degeneracy claim")
for banned in ("S-P2 passes on Multilingual", "the selection-conditional gap, now measured on a selector that has signal to lose"):
    hits = [f for f, src in (("main.tex", tex), ("main_short.tex", short)) if banned in src]
    print(f"  {'ok  ' if not hits else 'FAIL'}  withdrawn claim absent: '{banned[:44]}'")
    if hits: fail.append(f"withdrawn claim '{banned[:30]}'")

print("\naudit findings: registered items that had been unreported")
pr = json.loads((D / "swebench_pilot50/prereg_readout.json").read_text())
gg = pr["go_no_go"]
ok = gg["decision"] in tex and "146 silent failures projected" in tex
print(f"  {'ok  ' if ok else 'FAIL'}  go/no-go decision {gg['decision']} reported with its projection")
if not ok: fail.append("go/no-go unreported")
ok = pr["registered_m4"]["P1_M1_gt_0p5"] and "P1 passes as written" in tex
print(f"  {'ok  ' if ok else 'FAIL'}  P1 cleared its bar ({pr['registered_m4']['P1_M1_gt_0p5']}) and the paper discloses it")
if not ok: fail.append("P1 disclosure")
a2_16k = json.loads((D / "swebench_pilot50/prereg_readout_amendment2_16k_with32b.json").read_text())
r16 = a2_16k["reselect"]["tiger_lab_acecoderm_7b"]
row = f"& {r16['M1_AUROC_u']:.3f} & {r16['M2_AUROC_negc']:.3f} & ${r16['M3']:+.3f}$ & $[{r16['CI95'][0]:+.2f}, {r16['CI95'][1]:+.2f}]$"
ok = row in tex
print(f"  {'ok  ' if ok else 'FAIL'}  carried-forward 16k code-trained row present -> {row[:54]}")
if not ok: fail.append("16k code-trained row")
# the widest sweep bound must be the max CI upper over the ladder rows actually printed
# the sweep bound is the widest upper bound among the LADDER TABLE's own rows, not any interval in the prose
ladder = tex[tex.index("label{tab:ladder}") - 4000:tex.index("label{tab:ladder}")] if "label{tab:ladder}" in tex else ""
uppers = [float(hi) for lo, hi in re.findall(r"\$\[([-+][0-9.]+), ([-+][0-9.]+)\]\$", ladder)]
mx = max(uppers) if uppers else None
ok = mx is not None and f"the widest reaches $+{mx:.2f}$" in tex and f"widest interval in the sweep reaches $+{mx:.2f}$" in tex
print(f"  {'ok  ' if ok else 'FAIL'}  widest ladder-table upper bound is +{mx} over {len(uppers)} rows, and both bound sentences quote it")
if not ok: fail.append("widest sweep bound")
f4 = json.loads((D / "random_arm/prereg_readout_amendment4f.json").read_text())
fp2 = (f4["verified"]["F_P2_selector_sees_more_on_random"], f4["multilingual"]["F_P2_selector_sees_more_on_random"])
ok = fp2 == (True, False) and "passed on Verified and failed on Multilingual" in tex
print(f"  {'ok  ' if ok else 'FAIL'}  F-P2 verdicts {fp2} stated with the split")
if not ok: fail.append("F-P2 split")
a4c = json.loads((D / "swebench_pilot50/prereg_readout_amendment4c.json").read_text())
ok = a4c["predictions"]["A4c-P1_best_single_J1_gt_bar"] and "registered narrowing we decline" in tex
print(f"  {'ok  ' if ok else 'FAIL'}  A4c-P1 passed and its registered narrowing is addressed")
if not ok: fail.append("A4c narrowing")

print("\ncounts that drift when the study grows:")
_jg = json.loads((D / "readout_judges_gpt6.json").read_text())
_nj = len(_jg["multilingual"]["single_judges"])
for f, src in (("main.tex", tex), ("main_short.tex", short)):
    ok = f"{WORD.get(_nj, _nj)} LLM judges" in src
    print(f"  {'ok  ' if ok else 'FAIL'}  {f}: judge count states {WORD.get(_nj)} ({_nj} scored in the data)")
    if not ok: fail.append(f"{f} judge count")
# the instrument-failure list length must equal the number the prose claims
m = re.search(r"We record (\w+) instrument failures", tex)
listed = len(re.findall(r"\\item", tex[tex.index("We record"):tex.index("\\end{enumerate}", tex.index("We record"))]))
ok = m and WORD.get(listed) == m.group(1)
print(f"  {'ok  ' if ok else 'FAIL'}  main.tex: prose says '{m.group(1) if m else '?'}' instrument failures, list has {listed}")
if not ok: fail.append("instrument failure count")
# the best single judge's two benchmark values must both be quoted where the superlative is made
_bv = max(_jg["verified"]["single_judges"].values(), key=lambda v: v[0])[0]
_bm = max(_jg["multilingual"]["single_judges"].values(), key=lambda v: v[0])[0]
ok = f"{_bm:.3f} [0.68, 0.87] on the replication and {_bv:.3f}" in tex
print(f"  {'ok  ' if ok else 'FAIL'}  best-detector superlative quotes both benchmarks ({_bm}, {_bv})")
if not ok: fail.append("best detector both benchmarks")
# the taxonomy's below-floor range must span the actual min and max
_tx = json.loads((D / "readout_failure_taxonomy.json").read_text())
_bf = [d["AUROC_u"] for b in ("verified", "multilingual") for d in _tx[b]["per_class"].values() if d["below_inference_floor"]]
_ns = [d["n"] for b in ("verified", "multilingual") for d in _tx[b]["per_class"].values() if d["below_inference_floor"]]
ok = f"range from {min(_bf):.3f} to {max(_bf):.3f}" in tex and f"hold {WORD.get(min(_ns), min(_ns))} to {WORD.get(max(_ns), max(_ns))} positives" in tex
print(f"  {'ok  ' if ok else 'FAIL'}  below-floor classes: AUROC {min(_bf)}-{max(_bf)}, n {min(_ns)}-{max(_ns)}, both quoted")
if not ok: fail.append("taxonomy below-floor range")

print("\naudit: caption / headline / superseded JSON traps:")
# fig:ladder used to claim no circle crosses the ruler while plotting u_max=0.686
ok = "No circle crosses it" not in tex and "Exactly one circle crosses it" in tex and "0.686" in tex
print(f"  {'ok  ' if ok else 'FAIL'}  fig:ladder caption acknowledges the plotted u_max crossing")
if not ok: fail.append("fig:ladder caption")
# The squares half of the same caption was never pinned, and said "Two squares do" while only
# one -c in the plotted set clears the ruler. Derive both counts from make_figures.py.
_ruler = json.loads((D / "swebench_pilot50/covariate_baseline.json").read_text())["single_covariate_AUROC"]["patch_added_lines"]
_plotted = [(float(u), float(c)) for _, u, c in re.findall(r'\("([^"]+)",\s*([0-9.]+),\s*([0-9.]+),', figs)]
_circ = [u for u, c in _plotted if u > _ruler]
_sqr = [c for u, c in _plotted if c > _ruler]
WORDN = {1: "Exactly one", 2: "Two", 3: "Three"}
ok = len(_circ) == 1 and len(_sqr) == 1 \
    and f"{WORDN[len(_sqr)]} square does" in tex and "Two squares do" not in tex \
    and f"{max(_sqr):.3f}" in tex
print(f"  {'ok  ' if ok else 'FAIL'}  fig:ladder squares: {len(_sqr)} of {len(_plotted)} plotted "
      f"$-c$ exceed the {_ruler} ruler (max {max(_sqr)}), and the caption says so")
if not ok: fail.append("fig:ladder squares count")
ok = "It does not work." not in tex and "It does not work." not in short
print(f"  {'ok  ' if ok else 'FAIL'}  banned absolute 'It does not work.' absent from both papers")
if not ok: fail.append("absolute does-not-work")
ok = "about two-thirds" not in short
print(f"  {'ok  ' if ok else 'FAIL'}  short paper no longer says 'about two-thirds' for the best detector")
if not ok: fail.append("short two-thirds")
ok = "inconclusive rather than null" not in short
print(f"  {'ok  ' if ok else 'FAIL'}  short limitations no longer use the pre-correction A5 'inconclusive' gloss")
if not ok: fail.append("short stale A5 gloss")
_tier = json.loads((D / "readout_tier15.json").read_text())
_D = _tier["D_simultaneous_bound"]
_w = _D["widest_upper_in_sweep"]["upper"]
# Compare the rendering, not a constant: the readout holds the source value and the
# paper quotes it to two decimals. Hardcoding 0.63 here let the script and the
# committed artifact disagree, because only the artifact was corrected.
ok = _w == a2_16k["reselect"]["tiger_lab_acecoderm_7b"]["CI95"][1] and f"${_w:+.2f}$" in tex
print(f"  {'ok  ' if ok else 'FAIL'}  tier15 widest_upper_in_sweep {_w} comes from its named source and the paper quotes {_w:+.2f}")
if not ok: fail.append("tier15 widest upper")
# The per-protocol bound is also derived now; check both ends against their readouts.
ok = _D["primary_protocol_M3_upper"]["verified"] == pr["registered_m4"]["M3_CI95_repo_clustered"][1] \
    and _D["primary_protocol_M3_upper"]["multilingual"] == rep["primary_prefix_u"]["M3_CI95"][1] \
    and f"own upper interval is ${max(_D['primary_protocol_M3_upper'].values()):+.2f}$" in tex
print(f"  {'ok  ' if ok else 'FAIL'}  tier15 per-protocol uppers {_D['primary_protocol_M3_upper']} match the primary readouts and the paper")
if not ok: fail.append("tier15 per-protocol upper")
# mixed-sign corrected A5 deltas must not be narrated as agreeing in sign
# (do not match the legitimate phrase "disagree in sign")
ok = re.search(r"(?<!dis)agree in sign", tex) is None and re.search(r"(?<!dis)agree in sign", short) is None
print(f"  {'ok  ' if ok else 'FAIL'}  banned false 'agree in sign' for mixed A5 deltas")
if not ok: fail.append("agree in sign")
ok = "six instrument failures" not in short and "seven instrument failures" in short
print(f"  {'ok  ' if ok else 'FAIL'}  short paper instrument-failure count is seven")
if not ok: fail.append("short instrument count")
ok = (r"\textbf{P2} (Goodhart)" in tex or "P2 (Goodhart)" in tex) and "A2-P1" in tex \
     and "fails at 0.621" in tex and "fails at 0.405" in tex
print(f"  {'ok  ' if ok else 'FAIL'}  named P2 and A2-P1 failures disclosed with source values")
if not ok: fail.append("P2/A2-P1 disclosure")
ok = "Nothing that rose above the ruler on Verified rose above chance" not in tex
print(f"  {'ok  ' if ok else 'FAIL'}  banned overclaimed replication superlative")
if not ok: fail.append("replication superlative")

print("\naudit: statistical validity (R-P1 narration, draft version, instrument record):")
_sci = tex.split(r"\section{How the claim changed under scrutiny}")[0]
ok = "(the null replicates)" not in _sci and "(the null replicates)" not in short
print(f"  {'ok  ' if ok else 'FAIL'}  banned R-P1 headline '(the null replicates)' outside history appendix")
if not ok: fail.append("R-P1 null-replicates headline")
ok = "registered null-replication rule" in _sci and "upper bound 0.71" in _sci and "not equivalence to chance" in _sci
print(f"  {'ok  ' if ok else 'FAIL'}  long paper R-P1 states met-as-written + upper bound 0.71 + not equivalence")
if not ok: fail.append("R-P1 long narration")
ok = "Registered rule met" in short and "upper bound 0.71" in short and "Amendment~7a" in short
print(f"  {'ok  ' if ok else 'FAIL'}  short paper R-P1 matches failure-to-reject reading")
if not ok: fail.append("R-P1 short narration")
ok = re.search(r"\\textbf\{R-P1\}[^\n]*\\emph\{Passed\.\}", short) is None
print(f"  {'ok  ' if ok else 'FAIL'}  short paper does not mark R-P1 as Passed")
if not ok: fail.append("R-P1 short Passed")
# the claim-change table is a longtable since v0.9.12: tabularx cannot break across pages
_hist = tex[tex.index("label{sec:history}") : tex.index("\\end{longtable}", tex.index("label{sec:history}"))]
_ver_nums = [int(x) for x in re.findall(r"v0\.9\.(\d+)", _hist)]
_title_m = re.search(r"Working draft v0\.9\.(\d+)", tex)
_max_ver = max(_ver_nums) if _ver_nums else None
ok = _title_m and _max_ver is not None and int(_title_m.group(1)) == _max_ver
print(f"  {'ok  ' if ok else 'FAIL'}  title draft v0.9.{_title_m.group(1) if _title_m else '?'} matches changelog max v0.9.{_max_ver}")
if not ok: fail.append("draft version drift")
ok = "registered audit" in tex and "registered audit" in short \
    and "rather than by inspection of the numbers" not in tex and "rather than by inspecting the numbers" not in short
print(f"  {'ok  ' if ok else 'FAIL'}  instrument-failure prose admits registered audits (item 7)")
if not ok: fail.append("instrument audit prose")

print("\naudit: code-path traps (E3 fold alignment, Fig8 Verified JSON, A5 script gate):")
mf = Path(__file__).resolve().parent.joinpath("make_figures.py").read_text()
ok = 'ver = {"panel $u$ (prefix 4k)": (0.508' not in mf and '_pr["registered_m4"]["M1_AUROC_u"]' in mf
print(f"  {'ok  ' if ok else 'FAIL'}  make_figures.py loads Fig8 Verified points from committed JSON")
if not ok: fail.append("Fig8 Verified hardcodes")
_fig8 = {
    "panel u": json.loads((D / "swebench_pilot50/prereg_readout.json").read_text())["registered_m4"]["M1_AUROC_u"],
    "umax": json.loads((D / "swebench_pilot50/prereg_readout_amendment3.json").read_text())["u_max"]["AUROC"],
    "lines": json.loads((D / "swebench_pilot50/covariate_baseline.json").read_text())["single_covariate_AUROC"]["patch_added_lines"],
    "judge": json.loads((D / "swebench_pilot50/prereg_readout_amendment4c.json").read_text())["conditions"]["J1"]["best_single"]["AUROC"],
}
_a16f = json.loads((D / "swebench_pilot50/prereg_readout_amendment2_16k_with32b.json").read_text())
_fig8["rm16"] = max(v["AUROC_negc_full"] for v in _a16f["audit_only"].values() if isinstance(v, dict) and "AUROC_negc_full" in v)
# Anchor: primary AUROC(u) sentence + ruler/judge values already required elsewhere; refuse silent JSON drift.
_anchor_u = f"$\\mathrm{{AUROC}}(u)=0.508$" in tex.replace(" ", "") or "AUROC$(u)=0.508$" in tex.replace(" ", "") \
    or "gave $\\mathrm{AUROC}(u)=0.508$" in tex or "gave $\\mathrm{AUROC}(u) = 0.508$" in tex \
    or "Primary: prefix (4k), 4 chat RMs & 0.508" in tex
ok = _anchor_u and abs(_fig8["panel u"] - 0.508) < 1e-9 and abs(_fig8["umax"] - 0.686) < 1e-9 \
    and abs(_fig8["lines"] - 0.665) < 1e-9 and abs(_fig8["rm16"] - 0.712) < 1e-9 and abs(_fig8["judge"] - 0.752) < 1e-9
print(f"  {'ok  ' if ok else 'FAIL'}  Fig8 Verified JSON set matches paper anchors {_fig8}")
if not ok: fail.append("Fig8 Verified JSON drift")
_b2 = json.loads((D / "readout_pi_batch2.json").read_text())
_e3v = _b2["verified"]["E3_selection_pressure"]
# Fold-aligned value must stay in JSON. The claim-change appendix may name the
# pre-fix 0.444 as history; the scientific body and short paper must not.
_body = tex.split(r"\section{How the claim changed under scrutiny}")[0]
ok = abs(_e3v["best_of_4"]["AUROC_u"] - 0.462) < 1e-9 and "0.444" not in _body and "0.444" not in short
print(f"  {'ok  ' if ok else 'FAIL'}  E3 Verified best_of_4 AUROC_u is {_e3v['best_of_4']['AUROC_u']} (fold-aligned); stale 0.444 absent from body/short")
if not ok: fail.append("E3 fold-aligned AUROC")
# Inspect only the ECE binding line (not this assertion's own string literal).
_ece_line = next(ln for ln in Path(__file__).read_text().splitlines() if "CAL_judge_calibration" in ln and "ECE" in ln)
ok = "jg and" not in _ece_line and "_batch2[b]" in _ece_line
print(f"  {'ok  ' if ok else 'FAIL'}  ECE loader reads batch2 without `jg and` short-circuit")
if not ok: fail.append("ECE jg-and short-circuit")
import subprocess
_a5 = subprocess.run(
    [sys.executable, str(Path(__file__).resolve().parents[1] / "silentfail_bench/scripts/prereg_readout_amendment5.py")],
    capture_output=True, text=True,
)
ok = _a5.returncode == 2 and "DEFECTIVE" in (_a5.stderr + _a5.stdout)
print(f"  {'ok  ' if ok else 'FAIL'}  defective A5 script refuses without --write-defective-archive (exit {_a5.returncode})")
if not ok: fail.append("A5 defective script gate")

print("\naudit: Amendment 7 readiness must be executable, not a dry-run claim:")
_root = Path(__file__).resolve().parents[1]
_a7 = (_root / "silentfail_bench/scripts/readout_amendment7.py").read_text()
_a7test = (_root / "silentfail_bench/scripts/test_readout_amendment7.py").read_text()
_pre = (_root / "silentfail_bench/scripts/preflight_amendment7.py").read_text()
_prep = (_root / "silentfail_bench/scripts/prepare_amendment7.py").read_text()
_runbook = (_root / "silentfail_bench/RESUME_AMENDMENT7.md").read_text()
_prereg = (_root / "silentfail_bench/PREREGISTRATION_SWEBENCH_PILOT.md").read_text()
_mentor = (_root / "silentfail_bench/MENTOR_NOTE.md").read_text()
_readme = (_root / "README.md").read_text()
ok = "production_readout(a)" in _a7 and "readout_amendment7.json" in _a7 \
    and "replication_scored.jsonl" in _a7 and 'rows_f = out_dir / "rows.jsonl"' not in _a7
print(f"  {'ok  ' if ok else 'FAIL'}  A7 production path computes and writes JSON; stale rows.jsonl load absent")
if not ok: fail.append("A7 production no-op")
ok = "configs/agent_axis_instances_multilingual.txt" in _pre \
    and "configs/replication_instances_multilingual.txt" not in _pre
print(f"  {'ok  ' if ok else 'FAIL'}  A7 preflight validates paused IDs against pinned 114-instance set")
if not ok: fail.append("A7 preflight wrong set")
ok = "a7-random-v1" in _prep and "expected_keys" in _prep and "range(4)" in _prep
print(f"  {'ok  ' if ok else 'FAIL'}  A7 preparation freezes random arm and exact 114x4 candidate set")
if not ok: fail.append("A7 random/gold preparation")
ok = "a7-production-test:" in _a7test and "gold incomplete" in _a7test \
    and "must_not_exist.json" in _a7test
print(f"  {'ok  ' if ok else 'FAIL'}  A7 production fixture exercises joins and bites partial gold")
if not ok: fail.append("A7 production fixture")
for phrase, label in (
    ("--all-candidates", "mandatory gpt-5 all-candidate scoring"),
    ("a7_agent_axis_cand${C}", "four candidate-indexed gold runs"),
    ("scripts/readout_amendment7.py", "production readout command"),
    ("scripts/test_readout_amendment7.py", "production fixture command"),
):
    ok = phrase in _runbook
    print(f"  {'ok  ' if ok else 'FAIL'}  A7 runbook contains {label}")
    if not ok: fail.append(f"A7 runbook {label}")
ok = "# Amendment 7b" in _prereg and "from $12 to\n**$22**" in _prereg \
    and "no paid\noverage is authorized" in _prereg
print(f"  {'ok  ' if ok else 'FAIL'}  A7b records corrected $22 free-credit cap before gold")
if not ok: fail.append("A7b budget correction")
ok = "agree in sign for the first time" not in _mentor \
    and "+0.012, −0.024 and −0.018" in _mentor
print(f"  {'ok  ' if ok else 'FAIL'}  mentor note reports corrected A5 mixed signs")
if not ok: fail.append("mentor A5 signs")
# Derive the version from the paper's own title block, which is already checked
# against the changelog. Hardcoding it here meant three places to hand-sync.
# Do not hardcode the body length here: the README states it, and the measured
# page-budget check below requires the built PDF to agree with that statement.
_body_claim = re.search(r"9-page submission version \(body within (six|seven)\)", _readme)
ok = f"draft v0.9.{_title_m.group(1)}" in _readme and _body_claim is not None \
    and f"Nine-page submission version; body within {_body_claim.group(1)} pages" in _readme \
    and "7-page cut" not in _readme and "cut to 7 pages" not in _readme
print(f"  {'ok  ' if ok else 'FAIL'}  README status/version and short-paper length are current")
if not ok: fail.append("README status")
ok = "selected_4b[i] = pick_4b(choices)" in _a7 \
    and "return max(choices, key=lambda pair: pair[0])[1]" in _a7 \
    and "assert pick_4b([(2.0, c) for c in range(4)]) == 0" in _a7test \
    and "# Amendment 7c — selector tie rule" in _prereg
print(f"  {'ok  ' if ok else 'FAIL'}  A7 4B selection keeps the original first-candidate score-tie rule")
if not ok: fail.append("A7 4B selector tie rule")
_pooled = next(x for x in json.loads((D / "verify_and_robustify.json").read_text())["endpoint_robustness"]
               if x["label"] == "pooled")
_upper = _pooled["fixed_unit_weights_no_fitting"]["CI95"][1]
ok = _upper > 0.05 and "An upper bound on what disagreement can be worth, not a demonstration that it is worthless." in tex \
    and "A replicated lack of detectable gain" in short \
    and "lack of detectable gain replicates" in _readme \
    and "carries no information about its hidden failure" not in tex
print(f"  {'ok  ' if ok else 'FAIL'}  pooled upper bound {_upper:+.3f} above actionable +0.05 is described as non-detection")
if not ok: fail.append("null interpretation exceeds pooled precision")
# --- values the paper quotes that no assertion pinned, found by paper/bite_test.py ---
# Each of these was read by this checker but only in a threshold, a max, or an
# argmax, so the source value could drift to anything and every check stayed
# green while the paper kept quoting the old number.
print("\nquoted values a bite test showed were unpinned:")

def need(s_, label):
    ok_ = s_ in tex
    print(f"  {'ok  ' if ok_ else 'FAIL'}  {label}: {'found' if ok_ else 'MISSING'} -> {s_[:66]}")
    if not ok_: fail.append(label)

def r2(x):
    """Two decimals, rounding halves away from zero: the convention the paper uses.

    Python's format() rounds the binary double, so the Verified unfitted upper
    bound 0.245 renders as +0.24 while both the table and the prose print +0.25.
    Comparing against format() would have reported the paper as wrong.
    """
    return f"{Decimal(str(x)).quantize(Decimal('0.01'), rounding=ROUND_HALF_UP):+.2f}"

# The loosest pooled upper bound is the most repeated number in the paper. The
# only check on it was `_upper > 0.05`, which passes for any drifted value.
_full = ("logistic_LORO", "logistic_5fold_clustered", "fixed_unit_weights_no_fitting")
_pu = {k: _pooled[k]["CI95"][1] for k in _full}
_loose = max(_pu, key=_pu.get)
ok = _loose == "fixed_unit_weights_no_fitting"
print(f"  {'ok  ' if ok else 'FAIL'}  loosest of the three full-coverage pooled uppers is the unfitted estimator {_pu}")
if not ok: fail.append("loosest pooled estimator identity")
_ub, _lb = r2(_pu[_loose]), r2(_pooled[_loose]["CI95"][0])
need(f"the loosest pooled upper bound is ${_ub}$", "loosest bound: abstract, body, caption")
need(f"loosest pooled upper bound of ${_ub}$", "loosest bound: plain terms and contributions")
need(f"${_ub}$ pooled across the three full-coverage estimators", "loosest bound: limitations")
need(f"below ${_ub}$ on the pooled population", "loosest bound: conclusion")
need(f"pooled bound ${_ub}$ AUROC", "loosest bound: summary table")
need(f"[${_lb}$, ${_ub}$] pooled", "pooled unfitted interval in prose")
need(f"$\\mathbf{{[{_lb}, {_ub}]}}$", "pooled row of tab:endpoint")

# Every full-coverage row of tab:endpoint, built from JSON. Only the pooled
# unfitted row and one pooled delta were pinned before, so the other rows'
# intervals could drift silently.
_EST = (("logistic_LORO", "leave-one-repo-out"),
        ("logistic_5fold_clustered", "5 clustered folds"),
        ("fixed_unit_weights_no_fitting", "fixed unit weights"))
_rows_ok = 0
for _rec in rob["endpoint_robustness"]:
    for _k, _nm in _EST:
        _d = _rec[_k]
        if _rec["label"] == "pooled" and _k == "fixed_unit_weights_no_fitting":
            continue          # the bold pooled row, pinned above with \mathbf
        # the unfitted estimator names its two arms differently
        _b = _d["AUROC_Mc"] if "AUROC_Mc" in _d else _d["AUROC_negc"]
        _c2 = _d["AUROC_Mcu"] if "AUROC_Mcu" in _d else _d["AUROC_negc_plus_u"]
        _row = (f"& {_b:.3f} & {_c2:.3f} & ${_d['delta']:+.3f}$ "
                f"& $[{r2(_d['CI95'][0])}, {r2(_d['CI95'][1])}]$")
        if _row in tex:
            _rows_ok += 1
        else:
            print(f"  FAIL  tab:endpoint row missing for {_rec['label']}/{_nm} -> {_row}")
            fail.append(f"endpoint row {_rec['label']}/{_k}")
print(f"  {'ok  ' if _rows_ok == 8 else 'FAIL'}  {_rows_ok} of 8 non-bold full-coverage tab:endpoint rows match their JSON")
# The LORO pooled bound is also quoted in the limitations paragraph.
need(f"${r2(_pooled['logistic_LORO']['CI95'][1])}$ pooled for the leave-one-repository-out logistic",
     "LORO pooled bound in limitations")

# The inference floor the taxonomy derives its flags from is itself a registered
# quantity the paper states in words.
# Individual judge AUROCs. Only the best judge's identity and value were pinned;
# the paper quotes five of them by name.
for _b, _j, _ctx in (("multilingual", "gpt-5", ""), ("multilingual", "gpt-5-mini", ""),
                     ("multilingual", "gpt-5-nano", ""),
                     ("verified", "gpt-5", "reached "), ("verified", "gpt-5-mini", "")):
    _v = jg[_b]["single_judges"][_j][0]
    need(f"\\texttt{{{_j}}} {_ctx}{_v:.3f}", f"{_b} judge {_j} = {_v:.3f}")
_g5 = jg["multilingual"]["single_judges"]["gpt-5"]
need(f"above chance at {_g5[0]:.3f} [{_g5[1][0]:.2f}, {_g5[1][1]:.2f}]", "gpt-5 replication value with its interval")

# u_max on the replication, quoted twice, previously unpinned.
need(f"$u_{{\\max}}$ {rep['windowed_u_max']['AUROC']:.3f}", "windowed u_max on the replication")

# Amendment 5 corrected: the second estimator and both baselines are quoted.
need(f"${cm['estimators']['logistic_5fold']['delta']:+.3f}$ from five clustered folds", "corrected 5-fold estimator")
need(f"with a baseline of {cm['estimators']['logistic_LORO']['baseline_AUROC']:.3f} against a raw",
     "corrected LORO baseline")
need(f"collapsed to {pm['estimators']['logistic_LORO']['baseline_AUROC']:.3f}", "as-published degenerate baseline")

# Internal consistency of the taxonomy readout. The per-class count is stored
# twice; only the copy the table reads was checked. The floor flag is a function
# of the count and the readout's own declared floor, so derive it rather than
# trusting it.
_fl = tax["inference_floor"]
need(f"inference floor of {WORD.get(_fl, _fl)} positives", "pre-registered inference floor")
for _b in ("verified", "multilingual"):
    for _k, _d in tax[_b]["per_class"].items():
        ok = tax[_b]["classes"].get(_k, 0) == _d["n"] and _d["below_inference_floor"] == (_d["n"] < _fl)
        if not ok:
            print(f"  FAIL  {_b} taxonomy class '{_k}': classes={tax[_b]['classes'].get(_k)} "
                  f"per_class.n={_d['n']} floor_flag={_d['below_inference_floor']} (floor {_fl})")
            fail.append(f"taxonomy internal consistency {_b}/{_k}")
print(f"  ok    taxonomy per-class counts agree across both encodings and every floor flag "
      f"follows from n < {_fl}" if not [f for f in fail if f.startswith('taxonomy internal')] else "")

# --- Amendment 10 outcome: every gate number must come from its readout ---
print("\nAmendment 10 capability gates vs data/pilot_deepseek_a10/readout_amendment10.json:")
_a10 = json.loads((D / "pilot_deepseek_a10/readout_amendment10.json").read_text())
_g1, _g2, _g3 = _a10["P10_1"], _a10["P10_2"], _a10["P10_3"]
# The decision is the load-bearing part: a passing gate set would mean the axis ran.
ok = _a10["all_gates_pass"] is False and _g1["passes"] is False \
    and _g2["passes"] is True and _g3["passes"] is True
print(f"  {'ok  ' if ok else 'FAIL'}  gate verdicts are (P10-1 {_g1['passes']}, P10-2 {_g2['passes']}, "
      f"P10-3 {_g3['passes']}), all_gates_pass {_a10['all_gates_pass']}")
if not ok: fail.append("A10 gate verdicts")
# the one failing gate, stated identically in both papers
for _nm, _src in (("main.tex", tex), ("main_short.tex", short)):
    _n = _src.count(f"{_g1['submitted']} of {_a10['instances']}")
    ok = _n >= 1 and f"{int(_g1['bar'] * 100)}\\%" in _src
    print(f"  {'ok  ' if ok else 'FAIL'}  {_nm} quotes {_g1['submitted']} of {_a10['instances']} "
          f"against the {int(_g1['bar'] * 100)}% bar ({_n} mention(s))")
    if not ok: fail.append(f"{_nm} A10 submit count")
_resolved = _a10["instances"] - _g2["failures"]
for _nm, _src in (("main.tex", tex), ("main_short.tex", short)):
    for _want, _lab in ((f"{_resolved} of {_a10['instances']} resolved", "resolved count"),
                        (f"failure rate of {_g2['failure_rate']:.3f} against", "failure rate"),
                        (f"{_g2['gpt5mini_baseline']['failure_rate']:.3f}", "gpt-5-mini baseline"),
                        (f"\\${_g3['per_trajectory_usd']:.4f} per trajectory", "cost per trajectory")):
        ok = _want in _src or (_lab == "failure rate" and f"failure rate of {_g2['failure_rate']:.3f}" in _src)
        print(f"  {'ok  ' if ok else 'FAIL'}  {_nm} {_lab}: {'found' if ok else 'MISSING'} -> {_want}")
        if not ok: fail.append(f"{_nm} A10 {_lab}")
# the mechanism the run falsified must not survive anywhere
_banned = "ability to hold a text-driven action format"
ok = _banned not in tex and _banned not in short
print(f"  {'ok  ' if ok else 'FAIL'}  the format-drift mechanism the gold run falsified is absent from both papers")
if not ok: fail.append("A10 stale format-drift mechanism")
# the step-budget mechanism, and the refusal to renegotiate the budget
for _nm, _src in (("main.tex", tex), ("main_short.tex", short)):
    ok = "exhausted the step budget" in _src or "exhausted the 40-step budget" in _src
    print(f"  {'ok  ' if ok else 'FAIL'}  {_nm} states the step-budget mechanism")
    if not ok: fail.append(f"{_nm} A10 mechanism")

# --- the short paper's page budget, measured rather than asserted ---
# The README claimed "body under six pages" and nothing verified it; the body in
# fact fills six, and an edit on 1 Oct silently pushed the conclusion onto page 7.
# This is a submission gate, so it fails closed if it cannot be measured.
print("\nshort-paper page budget, measured from the built PDF:")
_pdf = Path("paper/main_short.pdf")
try:
    import pypdf
    _pages = _pdf_pages(_pdf)
except Exception as _e:                      # missing dep or missing/corrupt PDF
    print(f"  FAIL  cannot measure main_short.pdf ({type(_e).__name__}: {_e}). "
          f"Install pypdf or rebuild the PDF; an unverifiable page budget is not a passing one.")
    fail.append("short paper page budget unmeasurable")
else:
    _concl = next((i for i, t_ in enumerate(_pages, 1) if "Conclusion" in t_), None)
    _refs = next((i for i, t_ in enumerate(_pages, 1) if "References" in t_), None)
    # The 1 Oct corrections (per-benchmark heterogeneity, F-P1/F-P3, the A4c-P1 overrule and
    # the self-evaluation disclosure) pushed the body from six pages to seven. Cutting a
    # registered disclosure to hold a round number would be the wrong trade, so the bar is
    # now whatever the README claims, and the two must not drift apart. The hard page limit
    # lives on main_agentverify.pdf, which faces a real one.
    _claimed = 7 if "body within seven" in _readme else (6 if "body within six" in _readme else None)
    ok = (_concl is not None and _refs is not None and _refs >= _concl
          and _claimed is not None and _concl <= _claimed)
    print(f"  {'ok  ' if ok else 'FAIL'}  {len(_pages)} pages; conclusion starts p{_concl}, "
          f"references p{_refs}; README claims the body closes by p{_claimed}")
    if not ok: fail.append("short paper body length disagrees with the README")

# --- the anonymous IEEE submission must be generated, current, and clean ---
print("\nAGENTVERIFY submission (paper/main_agentverify.tex, generated):")
_av = subprocess.run([sys.executable, str(Path(__file__).resolve().parent / "make_agentverify.py"), "--check"],
                     capture_output=True, text=True)
ok = _av.returncode == 0
print(f"  {'ok  ' if ok else 'FAIL'}  generated file matches main_short.tex ({_av.stdout.strip() or _av.stderr.strip()})")
if not ok: fail.append("agentverify stale")
_avt = (Path(__file__).resolve().parent / "main_agentverify.tex").read_text()
# double-blind: nothing identifying may survive the preamble swap
_ids = [t for t in ("Morante", "Algoverse", "morantejr", "github.com") if t in _avt]
print(f"  {'ok  ' if not _ids else 'FAIL'}  no identifying token in the anonymous source"
      + (f" (found {_ids})" if _ids else ""))
if _ids: fail.append("agentverify not anonymous")
# the venue's limit is 8 pages INCLUDING references
_avpdf = Path(__file__).resolve().parent / "main_agentverify.pdf"
try:
    import pypdf
    _avp = _pdf_pages(_avpdf)
    _n = len(_avp)
    _anon = not any(t in " ".join(_avp).lower()
                    for t in ("morante", "algoverse", "morantejr"))
    ok = _n <= 8 and _anon
    print(f"  {'ok  ' if ok else 'FAIL'}  built PDF is {_n} pages including references (limit 8), "
          f"author-anonymous {_anon}")
    if not ok: fail.append("agentverify page limit or anonymity")
except Exception as _e:
    print(f"  FAIL  cannot measure main_agentverify.pdf ({type(_e).__name__}: {_e}); "
          f"an unverifiable page limit is not a passing one")
    fail.append("agentverify pdf unmeasurable")

# --- the short paper must carry the long paper's corrections, not just avoid its banned phrases ---
# Before 1 Oct the short-paper coverage here was almost entirely a blocklist of strings a
# previous audit had found. Positive assertions ran against main.tex only, so every
# correction the long paper received could be, and was, absent from the submission cut:
# a false figure caption, an unscoped replication superlative, four undisclosed registered
# predictions and a misattributed panel size all shipped into the anonymous PDF.
print("\nshort paper carries the long paper's corrections:")
_shortsrc = {"main_short.tex": short, "main_agentverify.tex": _avt}

def both(pred, label, want=True):
    bad = [n for n, src in _shortsrc.items() if (pred in src) is not want]
    ok_ = not bad
    print(f"  {'ok  ' if ok_ else 'FAIL'}  {label}" + ("" if ok_ else f" -> wrong in {', '.join(bad)}"))
    if not ok_: fail.append(f"short: {label}")

# 1. the figure caption must state the crossing it plots, in every version
both("Exactly one circle crosses it", "fig:ladder caption admits the plotted crossing")
both("No circle crosses it", "the false 'No circle crosses it' is gone", want=False)
# 2. the replication superlative must stay scoped: judges did not fall to chance
both("all fell to chance", "no unscoped 'all fell to chance'", want=False)
both("no configuration produced a replicable signal", "no unscoped 'no configuration' claim", want=False)
# 3. registered predictions the cut dropped
for _lab in ("F-P1", "F-P3", "A4c-P1"):
    both(_lab, f"{_lab} disclosed")
both("self-evaluation", "the gpt-5-mini self-evaluation confound disclosed")
# 4. the panel behind 0.494 and 0.439 is the registered four-member one
both("four-member panel", "0.494 / 0.439 attributed to the four-member panel")
both("six chat RMs from 1B to 8B with the patch in view (panel", "no six-member panel misreading", want=False)
# 5. the fragility sentence must match the table it sits under
both("eight to thirty-six clusters", "no per-benchmark cluster range under a pooled-only table", want=False)
# 6. F-P1 failed, so no concentration conclusion
both("did not concentrate failure", "no concentration conclusion from a failed prediction", want=False)
both("did not detect increased failure under selection", "the failure-to-detect wording is used")
# 7. a standalone submission may not point at documents it does not carry
for _ph in ("in the appendix", "the long version", "extended version of this paper"):
    both(_ph, f"no dangling pointer to '{_ph}'", want=False)
# 8. the pooled bound must not hide the per-benchmark sign disagreement
both("disagree in sign", "per-benchmark sign disagreement disclosed")
# 9/10. the corrected A5 population triple, and the best detector on both benchmarks
both(f"{a5['corrected']['multilingual']['repositories']} repositories", "corrected A5 population names its repositories")
both(f"{jg['verified']['single_judges']['gpt-6-sol'][0]:.3f}", "best-detector superlative quotes the pilot too")

# --- the symposium abstract is a third artifact; pin it to the same sources ---
# main_short.tex drifted from main.tex and shipped four false claims into a submission.
# A new standalone artifact gets its numbers asserted on day one rather than later.
print("\nsymposium abstract (paper/main_symposium.tex):")
_sym = (Path(__file__).resolve().parent / "main_symposium.tex").read_text()
_symchecks = [
    (f"{_pooled['logistic_LORO']['delta']:+.3f}".replace("+", "$+").replace("-", "$-") + "$", "pooled LORO delta"),
    (f"the loosest pooled upper bound is ${r2(_pu[_loose])}$", "loosest pooled bound"),
    (f"{_g1['submitted']} of {_a10['instances']} against an 80\\% bar", "A10 submit count and bar"),
    (f"{_a10['instances'] - _g2['failures']} of {_a10['instances']} resolved", "A10 resolved count"),
    (f"failure rate of {_g2['failure_rate']:.3f} against", "A10 failure rate"),
    (f"\\${_g3['per_trajectory_usd']:.4f} per trajectory", "A10 cost"),
    (f"{jg['multilingual']['single_judges']['gpt-6-sol'][0]:.3f}", "best detector, replication"),
    (f"{jg['verified']['single_judges']['gpt-6-sol'][0]:.3f}", "best detector, pilot"),
    ("A bound, not an absence", "bound-not-absence framing"),
    ("exhausted the step budget", "A10 step-budget mechanism"),
]
for _want, _lab in _symchecks:
    ok = _want in _sym
    print(f"  {'ok  ' if ok else 'FAIL'}  {_lab}: {'found' if ok else 'MISSING'} -> {_want[:52]}")
    if not ok: fail.append(f"symposium {_lab}")
# the symposium venue is not double-blind, so the author block must be present
ok = "Joseph Morante" in _sym and "Algoverse" in _sym
print(f"  {'ok  ' if ok else 'FAIL'}  not anonymised (this venue is not double-blind)")
if not ok: fail.append("symposium author block")
# and it must stay inside two pages
try:
    import pypdf as _pp
    _n = len(_pdf_pages(Path(__file__).resolve().parent / "main_symposium.pdf"))
    ok = _n <= 2
    print(f"  {'ok  ' if ok else 'FAIL'}  built PDF is {_n} page(s); the CFP's abstract option allows 2")
    if not ok: fail.append("symposium page limit")
except Exception as _e:
    print(f"  FAIL  cannot measure main_symposium.pdf ({type(_e).__name__})")
    fail.append("symposium pdf unmeasurable")

# --- the scaffold attribution, which the mentor's first objection lands on ---
# Amendment 10 declared that a both-models gate failure is "a finding about the scaffold
# and not about the models". That sentence lived only in the pre-registration while the
# papers reported the gate failure, leaving the obvious objection unanswered in print.
print("\nscaffold attribution for the Amendment 10 gate failure:")
_resolved = _a10["instances"] - _g2["failures"]
for _nm, _src in (("main.tex", tex), ("main_short.tex", short),
                  ("main_agentverify.tex", _avt),
                  ("main_symposium.tex", (Path(__file__).resolve().parent / "main_symposium.tex").read_text())):
    ok = "this scaffold" in _src and f"{_resolved} of the {_g1['submitted']}" in _src
    print(f"  {'ok  ' if ok else 'FAIL'}  {_nm} attributes the bar to this scaffold and gives the "
          f"{_resolved}-of-{_g1['submitted']} submitted-and-resolved figure")
    if not ok: fail.append(f"{_nm} scaffold attribution")
# the mentor note must carry the gate table, not a recollection of it
_mn = _mentor
ok = all(x in _mn for x in (f"{_g1['submitted']}/{_a10['instances']} = {_g1['rate']}",
                            f"{_g2['failure_rate']}, gap {_g2['gap']}",
                            f"${_g3['per_trajectory_usd']}",
                            f"${_g3['measured_total_usd']}"))
print(f"  {'ok  ' if ok else 'FAIL'}  MENTOR_NOTE section 26 reports the gate numbers from the readout")
if not ok: fail.append("mentor note A10 gate numbers")

# --- Amendment 11: every artifact carries the 100-step result, from its readout ---
print("\nAmendment 11 (100-step budget) vs data/readout_amendment11.json:")
_a11 = json.loads((D / "readout_amendment11.json").read_text())
_q1, _q2, _q3 = _a11["P11_1"], _a11["P11_2"], _a11["P11_3"]
_ds, _gm = _q2["deepseek"], _q2["gpt5mini_at_same_budget"]
_mx = _a11["budget_was_binding_at_40"]["deepseek_steps_at_100"]["max"]
ok = _a11["all_gates_pass"] is True and _q1["passes"] and _q2["passes"] and _q3["passes"] \
    and _q2["passes_A"] and _q2["passes_B"] and _q2["classification_post_hoc"] is True
print(f"  {'ok  ' if ok else 'FAIL'}  all three gates pass, P11-2 under both classifications, reclassification flagged post hoc")
if not ok: fail.append("A11 gate verdicts")
# the budget must be provably non-binding at 100, or the result is not what the papers say
ok = _a11["budget_was_binding_at_40"]["deepseek_steps_at_100"]["hit_ceiling"] == 0 and _mx < 100
print(f"  {'ok  ' if ok else 'FAIL'}  no DeepSeek trajectory hit the 100-step ceiling (max {_mx})")
if not ok: fail.append("A11 ceiling")
# The gaps are derived quantities: pin them to their own inputs, not to a threshold.
# A threshold alone let 0.458 drift to 0.508 unnoticed in the bite test -- the same
# shape as the +0.12 bound guarded only by `> 0.05` (DEFECTS #15).
_gA = round(_gm["failure_rate_A"] - _ds["failure_rate_A"], 3)
_gB = round(_gm["failure_rate_B"] - _ds["failure_rate_B"], 3)
ok = abs(_q2["gap_A_amendment_10b_exclusion"] - _gA) < 1e-9 and abs(_q2["gap_B_agent_crashes_as_failures"] - _gB) < 1e-9 \
    and _gA >= 0.10 and _gB >= 0.10
print(f"  {'ok  ' if ok else 'FAIL'}  gaps recompute from the rates ({_gA} / {_gB}) and both clear 0.10")
if not ok: fail.append("A11 gap")
_sym = (Path(__file__).resolve().parent / "main_symposium.tex").read_text()
for _nm, _src in (("main.tex", tex), ("main_short.tex", short), ("main_agentverify.tex", _avt), ("main_symposium.tex", _sym)):
    for _want, _lab in ((f"{_q1['submitted']} of {_a11['instances']}", "submit count"),
                        (f"({_q1['rate']:.3f}", "submit rate"),
                        (f"longest {_mx} steps", "longest run"),
                        (f"{_ds['failure_rate_B']:.3f}", "DeepSeek failure rate"),
                        (f"{_gm['failure_rate_B']:.3f}", "comparator failure rate, crashes counted"),
                        (f"{_gm['failure_rate_A']:.3f}", "comparator failure rate, crashes excluded"),
                        ("post hoc", "post-hoc budget declared")):
        ok = _want in _src
        print(f"  {'ok  ' if ok else 'FAIL'}  {_nm} {_lab}: {'found' if ok else 'MISSING'} -> {_want}")
        if not ok: fail.append(f"{_nm} A11 {_lab}")
# Amendment 10's recorded 40-step verdict must survive: a new measurement does not rewrite the old one
ok = f"{_g1['submitted']} of {_a10['instances']}" in tex and "32 of 41" in short and "at 40 steps" in tex.lower() or "40-step" in tex
print(f"  {'ok  ' if ok else 'FAIL'}  the 40-step Amendment 10 result is still reported alongside the 100-step one")
if not ok: fail.append("A10 verdict erased by A11")

# --- Amendment 12: GLM, not evaluable as registered, reported descriptively everywhere ---
print("\nAmendment 12 (GLM under DeepSeek's settings) vs data/readout_amendment12.json:")
_a12 = json.loads((D / "readout_amendment12.json").read_text())
_r1, _r2, _r3 = _a12["P12_1"], _a12["P12_2"], _a12["P12_3"]
_cut = len(_r1["binding_limit"]["cost_limit_0.25_cut"])
ok = _r1["evaluable_as_registered"] is False and _cut > 0 and len(_r1["binding_limit"]["step_limit_cut"]) == 0 \
    and _r3["passes"] is False and _a12["all_gates_pass"] is False
print(f"  {'ok  ' if ok else 'FAIL'}  P12-1 not evaluable ({_cut} cost-cap cuts, 0 step-cap cuts), P12-3 fails, no gate set passes")
if not ok: fail.append("A12 verdicts")
ok = _r2["glm"]["resolved"] == _r1["submitted"]
print(f"  {'ok  ' if ok else 'FAIL'}  every submitted GLM patch resolved ({_r2['glm']['resolved']} of {_r1['submitted']}), as the papers claim")
if not ok: fail.append("A12 resolved-equals-submitted")
_sym12 = (Path(__file__).resolve().parent / "main_symposium.tex").read_text()
for _nm, _src in (("main.tex", tex), ("main_short.tex", short), ("main_agentverify.tex", _avt), ("main_symposium.tex", _sym12)):
    for _want, _lab in ((f"submitted on {_r1['submitted']} of {_a12['instances']}", "GLM submit count"),
                        (f"{_cut} of {_a12['instances']} runs were cut by a \\$0.25", "cost-cap cut count"),
                        (f"\\${_r3['per_trajectory_usd']:.2f}", "GLM cost per trajectory"),
                        ("budget question is unanswered for GLM", "not-evaluable framing"),
                        ("disqualified on cost at any budget", "disqualification stated")):
        ok = _want in _src
        print(f"  {'ok  ' if ok else 'FAIL'}  {_nm} {_lab}: {'found' if ok else 'MISSING'} -> {_want[:48]}")
        if not ok: fail.append(f"{_nm} A12 {_lab}")
    ok = "was not taken to gold" not in _src and "reached 13 of 41 without gold" not in _src
    print(f"  {'ok  ' if ok else 'FAIL'}  {_nm} no stale 'not taken to gold' claim")
    if not ok: fail.append(f"{_nm} A12 stale gold claim")

_a7_real = subprocess.run(
    [sys.executable, str(_root / "silentfail_bench/scripts/readout_amendment7.py")],
    capture_output=True, text=True,
)
ok = _a7_real.returncode == 2 and "no partial result was written" in (_a7_real.stdout + _a7_real.stderr) \
    and not (_root / "silentfail_bench/data/agent_axis/readout_amendment7.json").exists()
print(f"  {'ok  ' if ok else 'FAIL'}  incomplete A7 production inputs fail closed (exit {_a7_real.returncode})")
if not ok: fail.append("A7 partial-input gate")

# ---------------------------------------------------------------------------------------------
print("\ncandidate diversity, per-attempt wording, positioning, failure-to-reject:")
_b1 = json.loads((D / "readout_pi_batch1.json").read_text())
_u2 = lambda x: r2(x).lstrip("+")
_cv, _cm = _b1["verified"]["CD_candidate_diversity"], _b1["multilingual"]["CD_candidate_diversity"]
check("candidate-diversity overlap sentence derives", f"mean pairwise overlap of changed lines was {_u2(_cv['mean_pairwise_patch_overlap'])} on Verified and {_u2(_cm['mean_pairwise_patch_overlap'])} on Multilingual" in tex, True)
check("candidate-diversity near-identical sentence derives", f"occurred in {int(round(_cv['frac_instances_with_near_identical_patches_overlap_gt_0.9']*100))}\\% and {int(round(_cm['frac_instances_with_near_identical_patches_overlap_gt_0.9']*100))}\\% of instances" in tex, True)
check("candidate-diversity selector-spread sentence derives", f"mean standard deviation of {_u2(_cv['mean_selector_score_spread'])} and {_u2(_cm['mean_selector_score_spread'])}" in tex, True)
check("conclusion no longer says the replication estimates 'are at chance'", "and they are at chance." in tex, False)
for _n, _t in (("main.tex", tex), ("main_short.tex", Path("paper/main_short.tex").read_text()), ("main_agentverify.tex", Path("paper/main_agentverify.tex").read_text()), ("main_symposium.tex", Path("paper/main_symposium.tex").read_text()), ("README.md", Path("README.md").read_text())):
    check(f"Amendment 13 comparison stated per attempt in {_n}", ("four attempts per instance" in _t) or ("budget per attempt" in _t), True)
check("positioning sentence present", "measures the residual value of a second signal on the trajectory a selector commits" in tex, True)
def _abstract(txt):
    m = re.search(r"\\begin\{abstract\}(.*?)\\end\{abstract\}", txt, re.S); return m.group(1) if m else ""
# The abstract quotes AUROC(u) at three decimals, the stored precision. A two-decimal
# rendering was a double rounding (0.5646 -> 0.565 -> 0.57; independent recomputation, 11 Oct).
_u_pair = f"{pr['registered_m4']['M1_AUROC_u']:.3f} and {rep['primary_prefix_u']['M1_AUROC_u']:.3f}"
check("the long paper's abstract keeps the three numbers the mentor asked for", all(x in _abstract(tex) for x in ("35\\% and 45\\%", _u_pair, "$+0.12$")), True)
check("the long paper's abstract is short (under 220 words)", len(_abstract(tex).split()) < 220, True)
check("denominators explicit: long paper's summary paragraph; the cuts' abstracts say 'of the committed trajectories'", ("of the committed population (visible-pass, non-empty, gradable)" in tex) and all("of the committed trajectories were silent failures" in _abstract(t) for t in (tex, Path("paper/main_short.tex").read_text(), Path("paper/main_agentverify.tex").read_text())), True)
check("every cut's abstract keeps the three numbers", all(all(x in _abstract(t) for x in ("35\\% and 45\\%", _u_pair, "$+0.12$")) for t in (tex, Path("paper/main_short.tex").read_text(), Path("paper/main_agentverify.tex").read_text())), True)
check("the title states the bound, not an absence", ("An Upper Bound on What Critic Disagreement Adds" in tex) and ("Adds No Detectable" not in tex), True)
# The 6 Oct retitle reached only main.tex; the cut and the IEEE version kept the old title
# for a week (defect #25). The cut must carry the same title and the generators derive it.
check("the nine-page cut carries the retitled paper", "An Upper Bound on What Critic Disagreement Adds" in Path("paper/main_short.tex").read_text(), True)
check("the IEEE version carries the retitled paper", "An Upper Bound on What Critic Disagreement Adds" in Path("paper/main_agentverify.tex").read_text(), True)
check("no version keeps the pre-retitle wording", any("Evaluate Failure Detectors on the" in Path(p).read_text() for p in ("paper/main.tex","paper/main_short.tex","paper/main_agentverify.tex","paper/main_symposium.tex")), False)

print("\nmechanism sentences (critic-by-trajectory interaction; E2 floor) vs data/readout_critic_demeaning.json:")
_cd = json.loads((D / "readout_critic_demeaning.json").read_text())["benchmarks"]
_res = [p["rank_scale"]["share_residual"] for b in _cd.values() for p in b["panels"].values()]
_trj = [p["rank_scale"]["share_trajectory"] for b in _cd.values() for p in b["panels"].values()]
_crt = [p["rank_scale"]["share_critic"] for b in _cd.values() for p in b["panels"].values()]
check("rank-scale critic share under 1% everywhere", max(_crt) < 0.01, True)
# the paper prints the RANGE of the trajectory and residual shares, so interior values cannot be pinned by the sentence;
# the three shares of each decomposition sum to one, which pins every one of them.
for _bn, _b in _cd.items():
    for _pn, _p in _b["panels"].items():
        _rs = _p["rank_scale"]
        check(f"rank-scale shares sum to one ({_bn}/{_pn})", _rs["share_critic"] + _rs["share_trajectory"] + _rs["share_residual"], 1.0, tol=0.002)
_pct = lambda x: int(Decimal(str(x * 100)).quantize(Decimal("1"), rounding=ROUND_HALF_UP))   # half away from zero, not banker's
_short_tex = Path("paper/main_short.tex").read_text(); _av_tex = Path("paper/main_agentverify.tex").read_text()
check("interaction-share sentence derives (long)", f"the remaining {_pct(min(_res))}--{_pct(max(_res))}\\% is residual critic-by-trajectory variation" in tex, True)
check("trajectory-share sentence derives (long)", f"a shared trajectory level accounts for {_pct(min(_trj))}--{_pct(max(_trj))}\\%" in tex, True)
check("mechanism shares derive (short and AGENTVERIFY)", all(f"{_pct(min(_trj))}--{_pct(max(_trj))}\\% is shared trajectory level and {_pct(min(_res))}--{_pct(max(_res))}\\% is residual critic-by-trajectory variation" in t for t in (_short_tex, _av_tex)), True)
check("'dominated by between-critic heterogeneity' gone from every cut", any("dominated by between-critic heterogeneity" in t for t in (tex, _short_tex, _av_tex)), False)
check("'no decorrelated panel to be had' gone", "no decorrelated panel to be had" in tex, False)
check("'the axis was not launched' is qualified everywhere", any(re.search(r"so the axis was not launched", t) for t in (tex, _short_tex, _av_tex, Path("paper/main_symposium.tex").read_text())), False)
_ev, _em = _cd["verified"]["E2_error_correlation"], _cd["multilingual"]["E2_error_correlation"]
check("E2 observed band still reproduces the printed 0.71-0.89 / 0.74-0.84", (r2(_ev["observed_range"][0]), r2(_ev["observed_range"][1]), r2(_em["observed_range"][0]), r2(_em["observed_range"][1])) == ("+0.71", "+0.89", "+0.74", "+0.84"), True)
check("E2 floor sentence derives", f"{_ev['independence_floor_range'][0]:.2f}--{_ev['independence_floor_range'][1]:.2f} (Verified) and {_em['independence_floor_range'][0]:.2f}--{_em['independence_floor_range'][1]:.2f} (Multilingual)" in tex, True)
check("E2 permutation sentence derives", f"({_ev['label_permutation_range_mean'][1]:.2f} and {_em['label_permutation_range_mean'][1]:.2f} on average over 200 permutations)" in tex, True)
check("E2 floor is below the observed upper edge (the band is not all floor)", _ev["independence_floor_range"][1] < _ev["observed_range"][1] - 0.1, True)

print("\nlearned-combiner sentence (DEFECTS #23) vs data/readout_pi_batch1.json:")
_b1 = json.loads((D / "readout_pi_batch1.json").read_text())
_V, _M = _b1["verified"]["LC_learned_combiner"], _b1["multilingual"]["LC_learned_combiner"]
_u2 = lambda x: r2(x).lstrip("+")   # unsigned two decimals, half away from zero
check("DEFECTS #23: published fixed-weight raw-panel row is the pass direction (1 - detect)", _V["4 raw panel scores"]["fixed_weights_oof"] + _V["4 raw panel scores"]["fixed_weights_oof_detect"], 1.0)
check("DEFECTS #23: c-alone detect matches raw AUROC(-c) within 0.01", abs(_V["c alone"]["fixed_weights_oof_detect"] - 0.621) < 0.01, True)
_lc_sent = (f"ranks failures at {_u2(_V['4 raw panel scores']['fixed_weights_oof_detect'])} on Verified and {_u2(_M['4 raw panel scores']['fixed_weights_oof_detect'])} on Multilingual "
            f"against {_u2(_V['u alone (published)']['fixed_weights_oof'])} and {_u2(_M['u alone (published)']['fixed_weights_oof'])} for disagreement")
check("learned-combiner sentence derives from the JSON", _lc_sent in tex, True)
check("learned-combiner logistic figures derive", f"({_u2(_V['4 raw panel scores']['logistic_oof'])} and {_u2(_M['4 raw panel scores']['logistic_oof'])} out of fold)" in tex, True)
check("learned-combiner: the inverted figures are named as the error", f"in the pass direction ({_u2(_V['4 raw panel scores']['fixed_weights_oof'])} and {_u2(_M['4 raw panel scores']['fixed_weights_oof'])})" in tex, True)
check("learned-combiner: 'not carrying much at all' is gone", "not carrying much at all" in tex, False)

print("\nAmendment 14 (gpt-6 judges as selectors) vs its readouts:")
_a14 = {J: json.loads((D / f"agent_axis_deepseek/readout_amendment14_{J}.json").read_text()) for J in ("gpt-6-sol", "gpt-6-luna")}
_g14 = {J: _a14[J]["selector_gate_A7a_2"] for J in _a14}
for J in _a14:
    check(f"A14 {J} gate fails", _g14[J]["passes"], False)
    check(f"A14 {J} pool AUROC below the bar", _g14[J]["pool_AUROC_detect"] < 0.65, True)
    check(f"A14 {J} verdict descriptive only", _a14[J]["confirmatory_verdict"]["branch"] == "DESCRIPTIVE ONLY", True)
_s14 = f"\\texttt{{gpt-6-sol}} reaches {_g14['gpt-6-sol']['pool_AUROC_detect']:.3f} and \\texttt{{gpt-6-luna}} {_g14['gpt-6-luna']['pool_AUROC_detect']:.3f} (Amendment 14)"
_r14 = f"({_g14['gpt-6-sol']['committed_failure_rate_unconditional']:.3f} and {_g14['gpt-6-luna']['committed_failure_rate_unconditional']:.3f} against {_g14['gpt-6-sol']['random_failure_rate_unconditional']:.3f})"
for _n, _t in (("main.tex", tex), ("main_short.tex", Path("paper/main_short.tex").read_text()), ("main_agentverify.tex", Path("paper/main_agentverify.tex").read_text())):
    check(f"A14 sentence derives in {_n}", (_s14 in _t) and (_r14 in _t), True)
check("A14 symposium figure derives", f"\\texttt{{gpt-6-sol}}, reaches {_g14['gpt-6-sol']['pool_AUROC_detect']:.3f} as a selector" in Path("paper/main_symposium.tex").read_text(), True)
# Judges return integer scores and tie at the top; the readout keeps the first attempt.
# The paper states the rule and the pick rate under the other one (independent recomputation, 11 Oct).
_tie = f"tie at the top on {_g14['gpt-6-sol']['tied_top_instances']} and {_g14['gpt-6-luna']['tied_top_instances']} instances"
_tie2 = f"fail at the random rate ({_g14['gpt-6-sol']['committed_failure_rate_unconditional_last_attempt_tie']:.3f}) and \\texttt{{gpt-6-luna}}'s at {_g14['gpt-6-luna']['committed_failure_rate_unconditional_last_attempt_tie']:.3f}"
for _n, _t in (("main.tex", tex), ("main_short.tex", Path("paper/main_short.tex").read_text())):
    check(f"A14 tie-rule clause derives in {_n}", (_tie in _t) and (_tie2 in _t), True)
check("A14 gpt-6-sol last-attempt tie rate equals the random rate (so 'worse than random' is rule-dependent)", _g14['gpt-6-sol']['committed_failure_rate_unconditional_last_attempt_tie'], _g14['gpt-6-sol']['random_failure_rate_unconditional'])
check("A14 gpt-6-luna stays above random under the last-attempt rule", _g14['gpt-6-luna']['committed_failure_rate_unconditional_last_attempt_tie'] > _g14['gpt-6-luna']['random_failure_rate_unconditional'], True)
check("A14 readouts record the tie rule", all(_g14[J]['tie_rule'] == "first attempt wins a tied top score" for J in _g14), True)
# A table row that ends in "\\bottomrule" (two backslashes) prints the word instead of the rule; it shipped in
# main.pdf from 6 to 11 Oct. Scan every built PDF's text for leaked rule commands.
for _pdfn in ("main.pdf", "main_short.pdf", "main_symposium.pdf", "main_agentverify.pdf"):
    try:
        _leak = [w for w in ("bottomrule", "midrule", "toprule") if any(w in t_ for t_ in _pdf_pages(Path("paper") / _pdfn))]
    except Exception as _e:
        _leak = [f"unreadable: {type(_e).__name__}"]
    check(f"no leaked rule command printed in {_pdfn}", _leak == [], True)
check("A14 README figures derive", f"gpt-6-sol {_g14['gpt-6-sol']['pool_AUROC_detect']:.3f}, gpt-6-luna {_g14['gpt-6-luna']['pool_AUROC_detect']:.3f}" in Path("README.md").read_text(), True)

print("\nAmendment 13 (agent axis on DeepSeek) vs its readouts:")
_a13 = json.loads((D / "agent_axis_deepseek/readout_amendment7_deepseek.json").read_text())
_a13p = json.loads((D / "agent_axis_deepseek/readout_amendment13_panel.json").read_text())
_cap = _a13["capability_A7_P1"]; _gate = _a13["selector_gate_A7a_2"]; _arms = _a13["arms"]
_new_rate = _cap["new_agent"]["unconditional_failure_rate_band"][0]
_old_rate = _cap["old_agent"]["failure_rate_B_agent_crashes_as_failures"]
check("A13 capability branch is PASS", _cap["branch"] == "PASS", True)
check("A13 comparator both ways agree", _old_rate, _cap["old_agent"]["failure_rate_A_amendment_10b_exclusion"])
check("A13 selector gate fails", _gate["passes"], False)
check("A13 verdict is DESCRIPTIVE ONLY", _a13["confirmatory_verdict"]["branch"] == "DESCRIPTIVE ONLY", True)
check("A13 axis panel is five reward models (score files)", _a13p["reward_models"]["n"], 5)
check("A13 no critic above the bar", _a13p["any_critic_above_bar"], False)
check("A13 judge pool AUROC equals the gate's", _a13p["critics"]["gpt5_judge_J1"]["AUROC_detect"], _gate["pool_AUROC_detect"])
_gap = round(_old_rate - _new_rate, 3)
_rm_lo, _rm_hi = _a13p["reward_models"]["min_AUROC_detect"], _a13p["reward_models"]["max_AUROC_detect"]
_sj, _s4 = _arms["S_judge"], _arms["S_4b"]
_ej, _e4 = _sj["estimators"]["fixed_unit_weights_PRIMARY"], _s4["estimators"]["fixed_unit_weights_PRIMARY"]
_prose_main = Path("paper/main.tex").read_text(); _prose_short = Path("paper/main_short.tex").read_text(); _prose_av = Path("paper/main_agentverify.tex").read_text()
_prose_symp = Path("paper/main_symposium.tex").read_text(); _readme13 = Path("README.md").read_text(); _mentor13 = Path("silentfail_bench/MENTOR_NOTE.md").read_text()
_prereg13 = Path("silentfail_bench/PREREGISTRATION_SWEBENCH_PILOT.md").read_text()
def _fmt3(x): return f"{x:.3f}"
def _signed(x): return ("+" if x >= 0 else "-") + f"{abs(x):.3f}"
_s_rates = f"{_fmt3(_new_rate)} against {_fmt3(_old_rate)}"
for name, txt in (("main.tex", _prose_main), ("main_short.tex", _prose_short), ("main_agentverify.tex", _prose_av), ("main_symposium.tex", _prose_symp), ("README.md", _readme13), ("MENTOR_NOTE.md", _mentor13)):
    check(f"A13 rates sentence in {name}", _s_rates in txt, True)
    check(f"A13 judge pool AUROC in {name}", _fmt3(_gate["pool_AUROC_detect"]) in txt, True)
for name, txt in (("main.tex", _prose_main), ("main_short.tex", _prose_short), ("main_agentverify.tex", _prose_av), ("main_symposium.tex", _prose_symp), ("README.md", _readme13), ("MENTOR_NOTE.md", _mentor13), ("PREREGISTRATION", _prereg13)):
    check(f"A13 reward-model range in {name}", (_fmt3(_rm_lo) in txt) and (_fmt3(_rm_hi) in txt), True)
check("A13 gap in main.tex", f"a gap of {_fmt3(_gap)}" in _prose_main, True)
check("A13 gap in PREREGISTRATION", f"gap is **{_fmt3(_gap)}**" in _prereg13, True)
_s_floor = f"{_sj['failures']} failures (judge arm, $n={_sj['n']}$) and {_s4['failures']} (4B arm, $n={_s4['n']}$)"
check("A13 floor sentence in main.tex", _s_floor in _prose_main, True)
check("A13 floor pair in main_short/agentverify", (f"{_sj['failures']} and {_s4['failures']} failures" in _prose_short) and (f"{_sj['failures']} and {_s4['failures']} failures" in _prose_av), True)
def _delta_tex(e): return f"${_signed(e['delta'])}$ [${_signed(e['CI95'][0])}$, ${_signed(e['CI95'][1])}$]"
check("A13 judge-arm primary delta in main.tex", _delta_tex(_ej) in _prose_main, True)
check("A13 4B-arm primary delta in main.tex", _delta_tex(_e4) in _prose_main, True)
check("A13 judge-arm primary delta in main_short", _delta_tex(_ej) in _prose_short, True)
check("A13 descriptive-only wording, both cuts", ("descriptive only" in _prose_short) and ("descriptive only" in _prose_av) and ("descriptive only" in _prose_symp), True)
check("A13: no artifact still calls the axis unanswered", any(("registered and unanswered" in t) or ("registered and \\emph{unanswered}" in t) for t in (_prose_short, _prose_av, _prose_symp)), False)
def _abstract(txt):
    m = re.search(r"\\begin\{abstract\}(.*?)\\end\{abstract\}", txt, re.S); return m.group(1) if m else ""
def _summary(txt):
    """The long paper's former abstract now opens Section 4 as 'Summary of results' (mentor review, 6 Oct); its
    abstract keeps three numbers. The A13 figures are therefore pinned in that paragraph for main.tex."""
    i = txt.find("\\paragraph{Summary of results.}")
    return txt[i: txt.find("\\subsection", i)] if i >= 0 else ""
def _a13_item(txt):
    """The cuts now carry the short abstract too (6 Oct); their A13 figures are pinned in the agent-axis limitations item."""
    i = txt.find("A capable open-weight agent: the axis ran")
    return txt[i: txt.find("\\item", i + 10)] if i >= 0 else ""
for name, txt, _ab in (("main.tex (summary paragraph)", _prose_main, _summary(_prose_main)), ("main_short.tex (agent-axis item)", _prose_short, _a13_item(_prose_short)), ("main_agentverify.tex (agent-axis item)", _prose_av, _a13_item(_prose_av))):
    check(f"A13 critic collapse stated in the abstract of {name} (rates)", (_s_rates.replace(" against ", " of 456 attempts against ") in _ab) or (_s_rates in _ab and "attempts" in _ab), True)
    check(f"A13 critic collapse stated in the abstract of {name} (judge AUROC)", _fmt3(_gate["pool_AUROC_detect"]) in _ab, True)
    check(f"A13 critic collapse stated in the abstract of {name} (RM range)", (_fmt3(_rm_lo) in _ab) and (_fmt3(_rm_hi) in _ab), True)
_orc_n = 0; _orc_bad = 0
for _c in range(4):
    for _d in json.loads((D / f"gold_modal/a7_agent_axis_cand{_c}/report.json").read_text())["details"]:
        _m = _d.get("model_name_or_path", "")
        if _m.startswith("ORACLE"):
            _orc_n += 1
            if (_m.endswith("reference_patch") and not _d.get("resolved")) or (_m.endswith("empty_patch") and _d.get("resolved")): _orc_bad += 1
check("A13 oracle rows across the four gold reports (only cand0 ran with --oracle 3)", _orc_n, 6)
check("A13 oracle anomalies", _orc_bad, 0)
check("A13 oracle sentence in main.tex derives", f"{_orc_n - _orc_bad} of {_orc_n} oracle checks" in _prose_main, True)
_stale = ("remains the open axis", "remains untested", "that is the next experiment", "is the open axis", "more expensive, not cheaper", "every critic is at chance", "every critic at chance", "fails first", "it was 0.671")
def _body_no_history(txt):
    return re.sub(r"\\begin\{longtable\}.*?\\end\{longtable\}", "", txt, flags=re.S)
for name, txt in (("main.tex", _body_no_history(_prose_main)), ("main_short.tex", _prose_short), ("main_agentverify.tex", _prose_av), ("main_symposium.tex", _prose_symp), ("README.md", _readme13), ("MENTOR_NOTE.md", _mentor13)):
    _hits = [ph for ph in _stale if ph in txt]
    check(f"A13 no stale or reaching phrase in {name} {_hits}", len(_hits), 0)
for name, txt in (("main.tex", _prose_main), ("main_short.tex", _prose_short), ("main_agentverify.tex", _prose_av)):
    check(f"A13 abstract of {name} says descriptive only", "descriptive only" in _abstract(txt), True)
check("A13 upper bounds exceed +0.05 (so 'descriptive only' is not an equivalence claim)", (_ej["CI95"][1] > 0.05) and (_e4["CI95"][1] > 0.05), True)

print("\n" + ("ALL CONSISTENT" if not fail else f"{len(fail)} MISMATCH(ES): " + "; ".join(fail)))
sys.exit(1 if fail else 0)

