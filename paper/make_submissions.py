#!/usr/bin/env python3
"""Generate one venue-formatted submission PDF per target workshop from main_short.tex.
Never hand-edit the outputs: every version is derived mechanically from the cut, so a
correction to main_short.tex reaches all of them on the next run (the lesson of defects
#16 and #25, where hand-maintained or hard-coded copies drifted from their source).

Formats, read from each venue's call on 13 Oct 2026:
  AAAI-27 workshops  aaai2027.sty, letter, two-column, no page numbers, Times via the
                     style file; [submission] option for double-blind; 8 pages excluding
                     references (Deployable AI: 7). secnumdepth raised to 2 so \\ref to
                     sections resolves (the kit allows 1 or 2).
  ICSE-27 workshops  \\documentclass[10pt,conference]{IEEEtran}; AGENT: 8 pages including
                     references, single-anonymous (named); LLM4Code: 4-8 pages including
                     references, double-anonymous.
  SANER-27 track     IEEEtran 10pt conference; 10 pages + 2 for references; double-anonymous.
Run:  python3 paper/make_submissions.py            (writes paper/submissions/*.tex and .pdf)
      python3 paper/make_submissions.py --out DIR  (also copies named PDFs into DIR)
"""
from __future__ import annotations
import argparse, os, re, shutil, subprocess, sys
from pathlib import Path
HERE = Path(__file__).resolve().parent
SRC = HERE / "main_short.tex"
OUTDIR = HERE / "submissions"
AUTHOR_NAME, AUTHOR_AFFIL, AUTHOR_EMAIL = "Joseph Morante", "Algoverse AI Research", "morantejoseph1@gmail.com"
SHORT_TITLE = "Auditing-the-Winner"
VENUES = [
    dict(slug="AAAI27-AgenticSE", style="aaai", anon=True, limit=8, kind="excl_refs",
         venue="AAAI-27 Workshop on Agentic Software Engineering (double-blind, non-archival; confirmed on the workshop site 20 Oct)"),
    dict(slug="AAAI27-TrustworthyAgentEvolution", style="aaai", anon=True, limit=8, kind="excl_refs",
         venue="AAAI-27 Workshop on Trustworthy Evolution of AI Agents (double-blind)"),
    dict(slug="AAAI27-WMAC", style="aaai", anon=True, limit=8, kind="excl_refs",
         venue="AAAI-27 WMAC: Measuring Collective Intelligence in Agentic Systems (double-blind, non-archival)"),
    dict(slug="AAAI27-DeployableAI", style="aaai", anon=True, limit=7, kind="excl_refs",
         venue="AAAI-27 Deployable AI Workshop (double-blind, non-archival, 7 pages)"),
    dict(slug="ICSE27-AGENT", style="ieee", anon=False, limit=8, kind="incl_refs",
         venue="AGENT 2027 at ICSE: Agentic Engineering (single-anonymous, 8 pages including references)"),
    dict(slug="ICSE27-LLM4Code", style="ieee", anon=True, limit=8, kind="incl_refs",
         venue="LLM4Code 2027 at ICSE (double-anonymous, 4-8 pages including references)"),
    dict(slug="SANER27-AgenticAI4SE", style="ieee", anon=True, limit=10, kind="excl_refs",
         venue="SANER 2027 Agentic AI4SE track (double-anonymous, 10 pages + 2 for references)"),
]
COMMON = r"\newcommand{\gzero}{g{=}0}\newcommand{\DeltaAUROC}{\Delta\mathrm{AUROC}}"
IEEE_PRE = r"""\documentclass[10pt,conference]{IEEEtran}
\IEEEoverridecommandlockouts
\usepackage{amsmath,amssymb,booktabs,enumitem,tabularx,array,graphicx,xcolor}
\usepackage{url}
\usepackage[numbers,sort&compress]{natbib}
\usepackage{stfloats}
\usepackage[hidelinks]{hyperref}
""" + COMMON + "\n"
AAAI_PRE = r"""\documentclass[letterpaper]{article} % DO NOT CHANGE THIS
\usepackage[@@OPT@@]{aaai2027}  % DO NOT CHANGE THIS
\usepackage[hyphens]{url}  % DO NOT CHANGE THIS
\usepackage{graphicx} % DO NOT CHANGE THIS
\urlstyle{rm} % DO NOT CHANGE THIS
\def\UrlFont{\rm}  % DO NOT CHANGE THIS
\usepackage{natbib}  % DO NOT CHANGE THIS AND DO NOT ADD ANY OPTIONS TO IT
\usepackage{caption} % DO NOT CHANGE THIS AND DO NOT ADD ANY OPTIONS TO IT
\frenchspacing  % DO NOT CHANGE THIS
\usepackage{booktabs}
\usepackage{amsmath,amssymb,enumitem,tabularx,array,xcolor}
\pdfinfo{
/TemplateVersion (2027.1)
}
\setcounter{secnumdepth}{2} % the body cross-references numbered sections
""" + COMMON + "\n"


def source_title(src: str, style: str = "aaai") -> str:
    m = re.search(r"\\title\{\\textbf\{(.*?)\}\}", src, re.S)
    if not m:
        raise SystemExit("ABORT: no \\title{\\textbf{...}} in main_short.tex")
    t = re.sub(r"\\\\\[[^\]]*\]", r"\\\\", m.group(1))
    return re.sub(r"\\\\\s*", " ", t).strip() if style == "ieee" else t


def body_of(src: str, style: str) -> str:
    body = src[src.index(r"\begin{document}\maketitle") + len(r"\begin{document}\maketitle"):]
    body = re.sub(r"\\begin\{(figure|table)\}\[t\]", r"\\begin{\1*}[t]", body)
    body = re.sub(r"\\end\{(figure|table)\}", r"\\end{\1*}", body)
    body = re.sub(r"\\paragraph\{([^}]*)\}(\\label\{[^}]*\})",
                  lambda m: r"\subsection{" + m.group(1).rstrip(".") + "}" + m.group(2), body)
    if style == "ieee":
        body = re.sub(r"\\paragraph\{([^}]*?)\.\}", r"\\paragraph{\1}", body)
        body = body.replace(r"\bibliographystyle{plainnat}", r"\bibliographystyle{IEEEtran}")
    else:
        body = body.replace(r"\bibliographystyle{plainnat}", r"\bibliographystyle{aaai2027}")
    return body


def render(v: dict, src: str) -> str:
    title = source_title(src, v["style"])
    head = f"% GENERATED FILE -- do not edit. Produced by paper/make_submissions.py from paper/main_short.tex.\n% Target: {v['venue']}\n"
    if v["style"] == "ieee":
        if v["anon"]:
            author = r"\author{\IEEEauthorblockN{Anonymous submission}\IEEEauthorblockA{Paper under double-anonymous review}}"
        else:
            author = (r"\author{\IEEEauthorblockN{" + AUTHOR_NAME + r"}\IEEEauthorblockA{" + AUTHOR_AFFIL +
                      r"\\ " + AUTHOR_EMAIL + "}}")
        pre = IEEE_PRE
    else:
        pre = AAAI_PRE.replace("@@OPT@@", "submission" if v["anon"] else "")
        if v["anon"]:
            author = "\\author{Anonymous submission}\n\\affiliations{Paper under double-blind review}"
        else:
            author = "\\author{" + AUTHOR_NAME + "}\n\\affiliations{" + AUTHOR_AFFIL + "\\\\ " + AUTHOR_EMAIL + "}"
    out = head + pre + "\n\\title{" + title + "}\n" + author + "\n\n\\begin{document}\\maketitle\n" + body_of(src, v["style"])
    if v["anon"]:
        for token in ("Morante", "Algoverse", "morantejr", "github.com", AUTHOR_EMAIL):
            if token in out:
                raise SystemExit(f"ABORT: '{token}' survives into the anonymous {v['slug']} version")
    return out


def build(slug: str) -> Path:
    env = dict(os.environ, TEXINPUTS=f".:{OUTDIR}:", BIBINPUTS=f"{HERE}:", BSTINPUTS=f"{HERE}:{OUTDIR}:")
    tex = OUTDIR / f"{slug}.tex"
    def latex():
        r = subprocess.run(["pdflatex", "-interaction=nonstopmode", "-halt-on-error",
                            f"-output-directory={OUTDIR}", str(tex)], cwd=HERE, env=env, capture_output=True, text=True)
        if r.returncode:
            log = (OUTDIR / f"{slug}.log").read_text(errors="ignore")
            i = log.find("\n!"); raise SystemExit(f"LaTeX failed for {slug}:\n{log[i:i+600]}")
    latex()
    subprocess.run(["bibtex", slug], cwd=OUTDIR, env=env, capture_output=True)
    latex(); latex()
    return OUTDIR / f"{slug}.pdf"


def measure(pdf: Path) -> tuple[int, int | None, int]:
    import fitz
    d = fitz.open(pdf); refpage = None
    for i, p in enumerate(d, 1):
        if re.search(r"^\s*References\s*$", p.get_text(), re.I | re.M):
            refpage = i; break
    log = pdf.with_suffix(".log").read_text(errors="ignore")
    return d.page_count, refpage, log.count("Overfull")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="", help="directory to receive the named PDF copies")
    a = ap.parse_args()
    OUTDIR.mkdir(exist_ok=True)
    src = SRC.read_text()
    rows = []
    for v in VENUES:
        (OUTDIR / f"{v['slug']}.tex").write_text(render(v, src))
        pdf = build(v["slug"])
        n, ref, over = measure(pdf)
        if v["kind"] == "incl_refs":
            ok = n <= v["limit"]; detail = f"{n} pages incl. refs (limit {v['limit']})"
        else:
            body = (ref - 1) if ref else n
            ok = ref is not None and ref <= v["limit"] + 1 and n <= v["limit"] + 2
            detail = f"references start p{ref}, {n} pages total (limit {v['limit']} + refs)"
        rows.append((v["slug"], "OK " if ok else "OVER", detail, over))
        if a.out:
            dest = Path(a.out).expanduser(); dest.mkdir(parents=True, exist_ok=True)
            shutil.copy(pdf, dest / f"Morante_{SHORT_TITLE}_{v['slug']}.pdf")
    for slug, ok, detail, over in rows:
        print(f"  {ok}  {slug:36s} {detail}; {over} overfull")
    return 0 if all(r[1] == "OK " for r in rows) else 1


if __name__ == "__main__":
    sys.exit(main())
