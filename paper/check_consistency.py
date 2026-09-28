#!/usr/bin/env python3
"""Cross-check numbers that appear in more than one place.

Run from the repo root:  python3 paper/check_consistency.py
Catches the class of error where a value hardcoded in make_figures.py drifts
from the table in main.tex, or where either drifts from the source JSON.
"""
import json, re, sys
import numpy as np
from pathlib import Path

tex = Path("paper/main.tex").read_text()
figs = Path("paper/make_figures.py").read_text()
D = Path("silentfail_bench/data")
fail = []

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
# the same rate is restated later in the paragraph; both occurrences must agree
n_rate = tex.count(f"{rate:.1f}\\%")
print(f"  {'ok  ' if n_rate == 2 else 'FAIL'}  suspect rate stated consistently in both places (found {n_rate}, want 2)")
if n_rate != 2: fail.append("luckypass suspect rate restated")

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
_w = _tier["D_simultaneous_bound"]["widest_upper_in_sweep"]["upper"]
ok = abs(_w - 0.63) < 1e-9 and "+0.63" in tex
print(f"  {'ok  ' if ok else 'FAIL'}  tier15 widest_upper_in_sweep is {_w}, paper quotes +0.63")
if not ok: fail.append("tier15 widest upper")
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

print("\n" + ("ALL CONSISTENT" if not fail else f"{len(fail)} MISMATCH(ES): " + "; ".join(fail)))
sys.exit(1 if fail else 0)
