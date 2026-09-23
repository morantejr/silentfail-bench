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
