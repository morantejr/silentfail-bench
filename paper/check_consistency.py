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
    want2 = f"${r['paired_difference']:+.3f}$ [${lo:+.2f}$, ${hi:+.2f}$]".replace("+0.00", "0.00")
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
required = [("main.tex", tex, "It is inconclusive."),
            ("main.tex", tex, "that cell is \\emph{inconclusive} rather than null"),
            ("main.tex", tex, "\\emph{consistent with} the null and does not establish it"),
            ("main_short.tex", short, "The verdict is \\textbf{inconclusive}"),
            ("main_short.tex", short, "consistent with the null and does not establish it")]
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
ece = {b: {k: v["ECE"] for k, v in jg and json.loads((D / "readout_pi_batch2.json").read_text())[b]["CAL_judge_calibration"]["judges"].items()} for b in ("verified", "multilingual")}
worst_ml = max(ece["multilingual"], key=ece["multilingual"].get)
ok = worst_ml == "gpt-5-mini" and "largest error belongs to \\texttt{gpt-5-mini}" in tex
print(f"  {'ok  ' if ok else 'FAIL'}  worst-calibrated on the replication is {worst_ml}, and the paper says so")
if not ok: fail.append("calibration attribution")

print("\n" + ("ALL CONSISTENT" if not fail else f"{len(fail)} MISMATCH(ES): " + "; ".join(fail)))
sys.exit(1 if fail else 0)
