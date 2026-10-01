"""Run mini-swe-agent's `swebench` batch CLI with sandboxes that close (DEFECTS #21).

Same arguments as `mini-extra swebench ...`. The only difference: before the CLI
starts, the registry name `swerex_modal` is pointed at SwerexModalClosingEnvironment,
so the runner's image selection (which matches that literal name) still applies and
every sandbox is terminated when its trajectory releases the environment.

    python -m envs.run_swebench_closing --subset multilingual --split test ...
"""
import runpy

from minisweagent.environments import _ENVIRONMENT_MAPPING

_ENVIRONMENT_MAPPING["swerex_modal"] = "envs.swerex_modal_closing.SwerexModalClosingEnvironment"

if __name__ == "__main__":
    runpy.run_module("minisweagent.run.benchmarks.swebench", run_name="__main__", alter_sys=True)
