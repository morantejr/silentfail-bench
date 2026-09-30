#!/usr/bin/env python3
"""Bite-test paper/check_consistency.py: prove each assertion can actually fail.

A consistency check that cannot fail is decoration. This repository has twice
shipped checks that passed on deliberately corrupted input, because they searched
for a bare number that also occurs elsewhere in the paper. This harness measures
the property directly instead of trusting it.

Method. Run the checker once with an instrumented JSON loader to record every
scalar it reads, and from where. Then, for each scalar, run the checker again
with that one value perturbed, and require a non-zero exit. Nothing on disk is
ever modified: the perturbation is injected into the loader, so an interrupted
run cannot leave a corrupted readout behind.

Perturbation size matters and is the reason this file exists rather than a shell
loop. The paper prints three decimals, and the readouts store values already
rounded to three decimals, so a perturbation below 0.0005 changes no printed
digit and no honest checker could catch it. An earlier ad-hoc harness used such
a perturbation and reported four false failures. Floats here move by at least
0.05, which changes every rendering the paper uses (3dp, 2dp, and 1dp percent).

Usage:
    python3 paper/bite_test.py                     # full sweep: ~195 runs, ~30 min
    python3 paper/bite_test.py --coverage          # list what the checker reads, then stop
    python3 paper/bite_test.py --file luckypass    # substring filter on the source

Do not edit the checker or either paper while a sweep runs. The sweep re-executes
the checker from disk on every iteration, so an edit changes what is being
measured, and a transient crash in the checker reads as "caught". The files are
fingerprinted and the run aborts if any of them moved.
"""
import argparse, contextlib, hashlib, io, json, os, runpy, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CHECKER = ROOT / "paper/check_consistency.py"
EXEMPT_FILE = Path(__file__).with_name("bite_test_exemptions.json")

# ---------------------------------------------------------------- instrumentation

class _Tracker:
    """Wraps decoded JSON so every scalar read is recorded by file and key path.

    Values can sit inside lists as well as dicts -- several readouts store
    [auroc, lo, hi] -- so both containers are wrapped. A dict-only tracker
    reports those files as entirely unread.
    """

    def __init__(self):
        self.accessed = {}
        self.lastread = (None, None)   # (exact text object, file) from the last .json read
        self.target = None       # (file, path) to perturb
        self.newval = None
        self.hit = False

    def mark(self, path, file, v):
        if isinstance(v, (dict, list)):
            return v
        self.accessed.setdefault(file, set()).add(path)
        if self.target == (file, path):
            self.hit = True
            return self.newval
        return v

    def wrap(self, o, path, file):
        T, L = self._td(), self._tl()
        if isinstance(o, dict):
            return T([(k, self.wrap(v, f"{path}/{k}", file)) for k, v in o.items()], path, file)
        if isinstance(o, list):
            return L([self.wrap(v, f"{path}/{i}", file) for i, v in enumerate(o)], path, file)
        return o

    def _td(self):
        tr = self
        class TD(dict):
            def __init__(s, pairs, path, file):
                super().__init__(pairs); s._p, s._f = path, file
            def __getitem__(s, k):
                return tr.mark(f"{s._p}/{k}", s._f, super().__getitem__(k))
            def get(s, k, default=None):
                return tr.mark(f"{s._p}/{k}", s._f, dict.__getitem__(s, k)) if k in s else default
            def values(s):
                return [tr.mark(f"{s._p}/{k}", s._f, v) for k, v in dict.items(s)]
            def items(s):
                return [(k, tr.mark(f"{s._p}/{k}", s._f, v)) for k, v in dict.items(s)]
        return TD

    def _tl(self):
        tr = self
        class TL(list):
            def __init__(s, items, path, file):
                super().__init__(items); s._p, s._f = path, file
            def __getitem__(s, i):
                v = list.__getitem__(s, i)
                return v if isinstance(i, slice) else tr.mark(f"{s._p}/{i}", s._f, v)
            def __iter__(s):
                for i, v in enumerate(list.__iter__(s)):
                    yield tr.mark(f"{s._p}/{i}", s._f, v)
        return TL


def run_checker(tracker):
    """Execute the checker in-process. Returns (exit_code, crashed)."""
    real_loads, real_rt = json.loads, Path.read_text

    def loads(s, *a, **k):
        o = real_loads(s, *a, **k)
        # Attribute by object identity, not by "the last .json path seen". The
        # checker also parses .jsonl files a line at a time; a last-path scheme
        # files those row keys under an unrelated readout and then reports them
        # as missing targets.
        text, file = tracker.lastread
        if text is not s:
            return o
        return tracker.wrap(o, "", file)

    def read_text(self, *a, **k):
        out = real_rt(self, *a, **k)
        if str(self).endswith(".json"):
            p = Path(self)
            try:
                rel = str(p.resolve().relative_to(ROOT))
            except ValueError:
                rel = str(p)
            tracker.lastread = (out, rel)
        return out

    json.loads, Path.read_text = loads, read_text
    argv, code, crashed = sys.argv[:], 0, False
    try:
        sys.argv = ["check_consistency.py"]
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            try:
                runpy.run_path(str(CHECKER), run_name="__main__")
            except SystemExit as e:
                code = e.code if isinstance(e.code, int) else 1
            except Exception:
                code, crashed = 1, True
    finally:
        json.loads, Path.read_text, sys.argv = real_loads, real_rt, argv
    return code, crashed


# ---------------------------------------------------------------- perturbation

def perturb(v):
    """A value that must change at least one printed digit, or None to skip."""
    if isinstance(v, bool):
        return not v
    if isinstance(v, int):
        return v + 7
    if isinstance(v, float):
        # 0.05 moves 3dp, 2dp and 1dp-percent renderings alike.
        return round(v + 0.05, 6) if abs(v) < 5 else round(v * 1.2, 6)
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--coverage", action="store_true", help="list what the checker reads and stop")
    ap.add_argument("--file", default="", help="only perturb sources whose path contains this")
    args = ap.parse_args()

    os.chdir(ROOT)
    # The sweep re-executes the checker from disk hundreds of times. Editing the
    # checker or either paper while it runs silently changes what is being
    # measured: a transient crash in the checker returns non-zero and is scored
    # as "caught". One sweep was invalidated exactly this way, so the inputs are
    # fingerprinted and re-checked at the end.
    # Everything the checker reads, not only the Python and the papers: it also
    # asserts against these records, so editing one mid-sweep moves the target too.
    WATCH = ["paper/check_consistency.py", "paper/main.tex", "paper/main_short.tex",
             "paper/make_figures.py", "paper/bite_test_exemptions.json",
             "README.md", "silentfail_bench/PREREGISTRATION_SWEBENCH_PILOT.md",
             "silentfail_bench/MENTOR_NOTE.md", "silentfail_bench/RESUME_AMENDMENT7.md"]

    def fingerprint():
        h = {}
        for f in WATCH:
            fp = ROOT / f
            h[f] = hashlib.sha256(fp.read_bytes()).hexdigest()[:16] if fp.exists() else None
        return h

    start_fp = fingerprint()
    tr = _Tracker()
    code, crashed = run_checker(tr)
    if code != 0 or crashed:
        print(f"ABORT: the checker does not pass on the clean tree (exit {code}, crashed={crashed}).")
        print("Bite-testing a red checker measures nothing. Fix the checker first.")
        return 2
    baseline = {f: sorted(p) for f, p in tr.accessed.items()}
    leaves = [(f, p) for f in sorted(baseline) for p in baseline[f]]
    print(f"clean tree: exit 0; {len(leaves)} scalars read from {len(baseline)} sources")
    if args.coverage:
        for f in sorted(baseline):
            print(f"\n{f}  ({len(baseline[f])})")
            for p in baseline[f]:
                print(f"    {p}")
        return 0

    exempt = json.loads(EXEMPT_FILE.read_text()) if EXEMPT_FILE.exists() else {}
    todo = [(f, p) for f, p in leaves if args.file in f]
    caught, uncaught, skipped = [], [], []
    for i, (f, p) in enumerate(todo, 1):
        raw = json.loads((ROOT / f).read_text())
        cur = raw
        try:
            for k in [x for x in p.split("/") if x]:
                cur = cur[int(k)] if isinstance(cur, list) else cur[k]
        except (KeyError, IndexError, ValueError):
            print(f"  MISSING TARGET  {f}{p}")
            uncaught.append((f, p, "target not found"))
            continue
        new = perturb(cur)
        if new is None:
            skipped.append((f, p))
            continue
        t = _Tracker()
        t.target, t.newval = (f, p), new
        rc, crash = run_checker(t)
        if not t.hit:
            # The scalar was read on the clean run but not on this one: an earlier
            # assertion short-circuited. Report rather than silently pass.
            uncaught.append((f, p, "not reached on perturbed run"))
            verdict = "NOT REACHED"
        elif rc != 0:
            caught.append((f, p))
            verdict = "caught" + (" (crash)" if crash else "")
        else:
            uncaught.append((f, p, "checker still exits 0"))
            verdict = "NOT CAUGHT"
        print(f"  [{i:3d}/{len(todo)}] {verdict:16s} {f.split('/')[-1]}{p}  {cur} -> {new}")

    end_fp = fingerprint()
    if end_fp != start_fp:
        changed = [f for f in WATCH if start_fp.get(f) != end_fp.get(f)]
        print(f"\nABORT: {', '.join(changed)} changed while the sweep was running.")
        print("Every verdict above was measured against a moving target and none of them count.")
        print("Re-run on a tree nobody is editing.")
        return 2

    print(f"\n{len(caught)} caught, {len(uncaught)} not caught, {len(skipped)} non-numeric skipped,"
          f" of {len(todo)} scalars")
    unexpected = [(f, p, why) for f, p, why in uncaught if p not in exempt.get(f, {})]
    for f, p, why in uncaught:
        tag = "exempt" if p in exempt.get(f, {}) else "UNEXPECTED"
        note = exempt.get(f, {}).get(p, why)
        print(f"  {tag:10s} {f.split('/')[-1]}{p}: {note}")
    needed = {(f, p) for f, p, _ in uncaught}
    stale = [(f, p) for f, paths in exempt.items() for p in paths if (f, p) not in needed]
    for f, p in stale:
        print(f"  STALE      {f.split('/')[-1]}{p}: now pinned; remove this exemption")
    if stale:
        print(f"\n{len(stale)} exemption(s) no longer needed. A stale exemption can hide a "
              f"later regression, so delete it from paper/bite_test_exemptions.json.")
        return 1
    if unexpected:
        print(f"\n{len(unexpected)} scalar(s) the paper reads but no assertion pins.")
        print("Either add an assertion, or record why it needs none in paper/bite_test_exemptions.json.")
        return 1
    print("\nEVERY PINNED SCALAR BITES")
    return 0


if __name__ == "__main__":
    sys.exit(main())
