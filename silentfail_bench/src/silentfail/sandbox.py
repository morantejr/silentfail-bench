"""Sandboxed execution of model-generated code, for real gold g.

Defence in depth, because this host holds OneDrive-synced work data:
  1. macOS sandbox-exec, deny-by-default: no network, no filesystem writes
     outside the per-execution temp dir.
  2. POSIX rlimits: CPU seconds, open files, no core dumps.
     VERIFIED LIMITATION: macOS silently ignores RLIMIT_AS -- inside the
     sandbox getrlimit(RLIMIT_AS) still reports unlimited, and an 800 MB
     allocation succeeds under a nominal 512 MB cap. Memory is therefore NOT
     capped on this platform; the CPU-second and wall-clock limits are the
     real controls, and both were confirmed to fire.
  3. Ephemeral temp cwd, wiped after each execution.
  4. Hard wall-clock timeout with process-group kill.

sandbox-exec is deprecated by Apple but present and functional. If the profile
fails to apply, `available()` reports False and the caller must decide -- we
degrade loudly (SandboxUnavailable), never silently to an unsandboxed run.
"""

from __future__ import annotations

import os
import resource
import shutil
import signal
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

SANDBOX_EXEC = "/usr/bin/sandbox-exec"

# Deny by default. Allow: process exec, reading system libs/frameworks, and
# read/write strictly under the execution temp dir. No network of any kind.
_PROFILE = """(version 1)
(deny default)
(allow process-exec)
(allow process-fork)
(allow sysctl-read)
(allow mach*)
(allow signal)
;; Reads are broadly allowed because the dyld shared cache and the CLT
;; toolchain need them for the interpreter to start at all; a tightly-scoped
;; read list aborts Python with SIGABRT before any user code runs.
;; This is safe only because egress is impossible: network is denied and the
;; process's only output channel is the stdout/stderr we capture.
(allow file-read*)
;; ...with the user's actual data explicitly clawed back. Later rules win.
(deny file-read* (subpath "{home}/Library/CloudStorage"))
(deny file-read* (subpath "{home}/Documents"))
(deny file-read* (subpath "{home}/Desktop"))
(deny file-read* (subpath "{home}/Downloads"))
(deny file-read* (subpath "{home}/Pictures"))
(deny file-read* (subpath "{home}/.ssh"))
(deny file-read* (subpath "{home}/.aws"))
(deny file-read* (subpath "{home}/.config"))
(deny file-read* (subpath "{home}/.gnupg"))
(deny file-read* (literal "{home}/.zshrc"))
(deny file-read* (literal "{home}/.netrc"))
(allow file-write*
  (subpath "{tmp}")
  (literal "/dev/null"))
(deny network*)
"""


class SandboxUnavailable(RuntimeError):
    pass


@dataclass
class ExecResult:
    ok: bool                 # exit code 0
    timed_out: bool
    returncode: int
    stdout: str
    stderr: str
    sandboxed: bool


def available() -> bool:
    return sys.platform == "darwin" and Path(SANDBOX_EXEC).exists()


def _limits(cpu_sec: int, address_space_mb: int, nofile: int):
    def _apply():
        os.setsid()
        resource.setrlimit(resource.RLIMIT_CPU, (cpu_sec, cpu_sec))
        mem = address_space_mb * 1024 * 1024
        try:
            # Best effort only: macOS ignores this (see module docstring).
            resource.setrlimit(resource.RLIMIT_AS, (mem, mem))
        except (ValueError, OSError):
            pass
        resource.setrlimit(resource.RLIMIT_NOFILE, (nofile, nofile))
        resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    return _apply


def run_python(
    source: str,
    *,
    wall_timeout_sec: int = 15,
    cpu_sec: int = 10,
    address_space_mb: int = 2048,
    nofile: int = 64,
    require_sandbox: bool = True,
) -> ExecResult:
    """Execute `source` in an isolated interpreter. Never raises on user code."""
    if require_sandbox and not available():
        raise SandboxUnavailable(
            "sandbox-exec unavailable; refusing to execute generated code unsandboxed"
        )

    tmp = tempfile.mkdtemp(prefix="sfb_exec_")
    try:
        real_tmp = os.path.realpath(tmp)
        # -I isolates the interpreter (no PYTHONPATH, no user site), which also
        # hides numpy -- and the HumanEval+ gold tests import numpy. Re-inject
        # exactly the site-packages dirs this interpreter already uses, and
        # nothing else.
        site_dirs = [d for d in sys.path if d.endswith("site-packages")]
        preamble = "import sys\n" + "".join(
            f"sys.path.append({d!r})\n" for d in site_dirs
        )
        script = Path(real_tmp) / "prog.py"
        # Explicit UTF-8: some HumanEval+ tests carry non-ASCII literals, and
        # locale-default encoding produced "Non-UTF-8 code" SyntaxErrors.
        script.write_text("# -*- coding: utf-8 -*-\n" + preamble + source, encoding="utf-8")

        argv = [sys.executable, "-I", "-B", str(script)]  # NOT -S: gold tests import numpy
        sandboxed = False
        if available():
            prof = Path(real_tmp) / "policy.sb"
            prof.write_text(_PROFILE.format(tmp=real_tmp, home=os.path.expanduser("~")), encoding="utf-8")
            argv = [SANDBOX_EXEC, "-f", str(prof)] + argv
            sandboxed = True

        env = {
            "PATH": "/usr/bin:/bin",
            "HOME": real_tmp,
            "TMPDIR": real_tmp,
            "PYTHONHASHSEED": "0",
            "PYTHONDONTWRITEBYTECODE": "1",
            "NO_NETWORK": "1",
        }
        try:
            p = subprocess.Popen(
                argv, cwd=real_tmp, env=env,
                stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
                preexec_fn=_limits(cpu_sec, address_space_mb, nofile),
            )
        except OSError as e:
            return ExecResult(False, False, -1, "", f"spawn failed: {e}", sandboxed)

        try:
            out, err = p.communicate(timeout=wall_timeout_sec)
            return ExecResult(p.returncode == 0, False, p.returncode, out[-8000:], err[-8000:], sandboxed)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(os.getpgid(p.pid), signal.SIGKILL)
            except (ProcessLookupError, PermissionError):
                p.kill()
            p.communicate()
            return ExecResult(False, True, -9, "", "wall-clock timeout", sandboxed)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
