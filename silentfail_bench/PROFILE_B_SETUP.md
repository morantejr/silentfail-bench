# Profile B — real SWE-bench Verified gold on Apple Silicon, no sudo

Validated 2026-09-22 on M4 Pro / 24 GB / macOS 25.4, with Homebrew broken
(`/opt/homebrew` not user-owned, so `brew install` exits 0 and installs
nothing; the fix needs `sudo chown`).

## Install (all standalone, no admin rights)

```bash
mkdir -p ~/.local/bin ~/.local/lima

# colima
curl -sL -o ~/.local/bin/colima \
  https://github.com/abiosoft/colima/releases/download/v0.10.3/colima-Darwin-arm64
chmod +x ~/.local/bin/colima

# lima — colima needs BOTH `lima` and `limactl`; a limactl-only symlink
# fails with "lima not found, run 'brew install lima'"
curl -sL https://github.com/lima-vm/lima/releases/download/v2.2.0/lima-2.2.0-Darwin-arm64.tar.gz \
  | tar xz -C ~/.local/lima

# docker CLI
curl -sL https://download.docker.com/mac/static/stable/aarch64/docker-27.3.1.tgz \
  | tar xz -C /tmp && mv /tmp/docker/docker ~/.local/bin/docker

export PATH="$HOME/.local/lima/bin:$HOME/.local/bin:$PATH"

# swebench >= 5 uses PEP-604 annotations at runtime; host Python 3.9 cannot
# import it. uv installs a standalone CPython without sudo.
curl -sLSf https://astral.sh/uv/install.sh | sh
uv venv --python 3.12 .venv-swebench
uv pip install --python .venv-swebench/bin/python swebench datasets
```

## Start the VM

```bash
colima start --vm-type vz --vz-rosetta --cpu 6 --memory 12 --disk 40
export DOCKER_HOST="unix://$HOME/.colima/default/docker.sock"
export DOCKER_DEFAULT_PLATFORM=linux/amd64
```

`--vz-rosetta` is what makes x86_64 containers usable; verify with
`docker run --rm --platform linux/amd64 alpine uname -m` → `x86_64`.

**Requested `--disk 40`, got 19 GB.** Check `colima ssh -- df -h /` before
scaling; at 4.23 GB per instance image that is ~4 instances resident.

## Two traps

1. **Stale credential helper.** A removed Docker Desktop can leave
   `"credsStore": "desktop"` in `~/.docker/config.json`, which fails every pull
   with `StoreError`. Remove the key (check `auths` is empty first).
2. **x86_64-only images.** SWE-bench instance images have no arm64 manifest,
   and the harness pulls without specifying a platform, so the client asks for
   `linux/arm64/v8` and gets *"no matching manifest"*. Pre-pull explicitly:
   ```bash
   docker pull --platform linux/amd64 swebench/sweb.eval.x86_64.<instance>
   ```

## Validation — always run the oracle controls first

If the known-correct answer does not resolve, the oracle is broken and every
downstream number is fiction. This caught a fatal bug in profile A, where a
missing numpy made *every* trajectory score g=0 — a 100% failure rate that
looked like a finding.

The no-op control matters as much as the gold one: an empty patch is
short-circuited as `empty_patch` without running anything, so only a
**non-empty patch that applies cleanly and fixes nothing** proves the harness
actually executes tests and can fail them.

```bash
.venv-swebench/bin/python -m swebench.harness.run_evaluation \
  -d SWE-bench/SWE-bench_Verified -p gold -id goldctl \
  --max_workers 1 -i pallets__flask-5014
```

| control | expected | observed |
|---|---|---|
| gold patch | resolved | **resolved 1/1, 33 s** ✅ |
| empty patch | not resolved | classified `empty_patch`, short-circuited ✅ |
| non-empty no-op patch | not resolved, tests actually run | **completed 1, unresolved 1, 27 s** ✅ |

## Measured cost

| item | value |
|---|---|
| instance image | **4.23 GB** |
| image pull | 42 s |
| evaluation (image cached) | 33 s |
| per instance, cold | ~75 s |

A 40-instance pilot is roughly an hour of wall clock, but ~170 GB of pulls if
env layers are not shared — and only ~4 images fit at once in a 19 GB VM, so
the loop must pull → evaluate → prune.

## Read this before scaling

SWE-bench Verified is 500 instances over **12 repos**, django alone being 46%.
Because the primary estimand clusters on repo, effective n is ~67–140, not 500.
See `MENTOR_NOTE.md` §8 — that limit binds harder than n and more trajectories
do not relieve it.


## Agent smoke run — 2026-09-23

`mini-swe-agent 2.4.6` → Ollama `qwen2.5-coder:7b` → x86 container under
Rosetta, on `pallets__flask-5014`. Three attempts; each failure was a config
trap, now fixed in `configs/mini_swe_ollama.yaml`:

| attempt | died at | cause | fix |
|---|---|---|---|
| 1 | 57 s, 1 API call | litellm cost tracker has no price for a local model and raises | `cost_tracking: ignore_errors` |
| 2 | 3 API calls | default config needs native tool calls; 7B via Ollama emits none → `RepeatedFormatError` | `swebench_backticks.yaml` + `model_class: litellm_textbased` |
| 3 | elapsed=109s, 30 calls | **pipeline works** — real commands executed in the container — but the model looped (`3/30` distinct commands) and hit `step_limit=30` with **no patch** | not a config problem |

**Verdict.** The profile-B pipeline is validated end to end: image resolution
on arm64, model routing, text-based tool calling, command execution under
emulation, trajectory capture. What is *not* viable is a 7B coder as the
SWE-bench agent: it navigates in circles and submits nothing, and an empty
patch is an excluded row, so the yield is zero usable trajectories. That is
consistent with the public leaderboard, where every mini-swe-agent entry
uses a frontier model. **Generation is the bottleneck, and on this task it is
a model-capability bottleneck, not a compute one.**

Timing for sizing: ~109s for one 30-step candidate at 7B; a
capable model would use fewer, longer steps.

## OpenRouter free tier — 2026-09-23

Generation via `qwen/qwen3.8-27b:free` and `z-ai/glm-5.2:free` both returned
**HTTP 429 on the very first request**, from two different providers. A 429 on
request #1 is not the per-minute window (some calls would succeed); it is the
**key's daily free-request cap** (~50/day without purchased credits) already
consumed. mini-swe-agent backs off 4→8→16→…60 s for 10 attempts, then aborts.

Consequences for a $0 run:
- One 40-step trajectory needs ~40 requests: it fits the no-credit daily cap
  only on a fresh day, and a multi-instance pilot does not fit at all.
- Free models cost $0 per request regardless; what the cap gates is *count*.
  OpenRouter raises the free-model cap to ~1000/day once an account has
  ≥ $10 of lifetime credits, which remain unspent on `:free` models — but
  buying them is a real $10 outlay, not a free action.
- The transport itself is fine: auth accepted, model ids valid, the text-based
  parser is the one validated with the 7B. Only the quota is blocking.

## Local model ladder on one instance — 2026-09-23

Same instance (`pallets__flask-5014`), same text-based config, same 30-step
limit. The correct fix is a one-line `ValueError` when a Blueprint name
contains a dot.

| model | elapsed | distinct commands / 30 | what it did | patch |
|---|---|---|---|---|
| qwen2.5-coder:7b | 109 s | 3 | `cd src && ls` / `cd flask && ls`, alternating | none |
| qwen2.5-coder:14b | ~4 min | 9 | grep'd `Blueprint`, **found the exact validation site**, then `echo`'d its own conclusion 22× | none |

The 14B is qualitatively different — it located the fix — and identically
useless for the pipeline, because an empty patch is an excluded row. The
failure moved from *navigation* to *acting*: it narrated the answer instead
of editing the file. That is a well-known small-model agent failure and not
one a step limit or a prompt nudge should be trusted to solve at pilot scale.

**Conclusion of the ladder:** local models up to 14B cannot drive this
scaffold to a submission on this host. Generation for the confirmatory run
needs a frontier-class model; on a $0 budget that means the free OpenRouter
tier, gated by its daily request cap (see above).

## GPT-5-mini smoke run — RESOLVED (2026-09-23)

Same instance, native tool calls (`swebench.yaml`), `gpt-5-mini` via OpenAI.
**23 API calls, $0.0201, real patch, passes the official gold tests.** The
fix is the empty-blueprint-name `ValueError` in `src/flask/blueprints.py`,
the same file the gold patch touches. Full profile-B pipeline validated with
a capable model; pilot repriced at ~$4 for 200 trajectories.

Two traps, both fixed in the orchestrator:

1. **`swebench-single` ends on an interactive prompt** ("Agent wants to
   finish… Enter to quit") that `-y` does not cover; the run sat on it for
   90 minutes with `info.submission` unwritten. Batch mode (`mini-extra
   swebench --filter '^id$'`) is non-interactive and writes `exit_status` and
   `submission` itself. `--exit-immediately` exists for the single runner.
2. **With native tool calls the diff lives in `role: "tool"` messages**, not
   `role: "user"`. Trajectory text and any recovery must include them.

## Orchestrator validated — 2026-09-23 18:33

`scripts/run_swebench_pilot.py --stage generate --k 1` on the cached flask
instance, through the batch runner: `exit_status: Submitted`, 1,219-char
patch to `src/flask/blueprints.py`, **$0.014, 101 s, 31 messages.** This is
the exact code path the pilot runs.

One more trap fixed on the way: the orchestrator had been injecting the
Ollama endpoint into the agent's environment from the profile-A days, which
sent `gpt-5-mini` to `localhost:11434` (NotFoundError). The environment now
carries only Docker plumbing; the endpoint comes from the mini-config.

**Pilot launch plan.** Generation first (`--stage generate`, API-bound,
~6 h, ~$3–4, ≤ $50 under the $0.25/trajectory cap), then scoring + gold as a
separate stage so the 8 GB selection RM never shares memory with the
container VM. Every trajectory is checkpointed; the split is free.

## Two interpreters — 2026-09-23 20:35 (caught by dry run, before the overnight chain reached it)

The orchestrator runs under `.venv-swebench` (CPython 3.12, because
`swebench>=5` needs it), but torch/transformers had only ever been installed
into the system Python 3.9. Every reward-model run so far had been launched
with `python3` directly, so `--stage rm` inside the orchestrator was the
first time `rm_backend` was imported from the venv — `ModuleNotFoundError:
torch`. Fixed by installing the torch stack into the venv; **one interpreter
for the whole pipeline** is the rule from here.

Why it matters: the unattended chain would have hit this at ~3:30 AM, logged
"RM stage FAILED", and stopped. The dry run of never-executed stages on a
one-candidate directory is cheap insurance and should precede every
unattended run.
