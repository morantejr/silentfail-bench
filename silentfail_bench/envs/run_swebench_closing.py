"""Run mini-swe-agent's `swebench` batch CLI with sandboxes that close (DEFECTS #21).

Same arguments as `mini-extra swebench ...`. The only difference: before the CLI
starts, the registry name `swerex_modal` is pointed at SwerexModalClosingEnvironment,
so the runner's image selection (which matches that literal name) still applies and
every sandbox is terminated when its trajectory releases the environment.

    python -m envs.run_swebench_closing --subset multilingual --split test ...
"""
import gc

from minisweagent.environments import _ENVIRONMENT_MAPPING
import minisweagent.run.benchmarks.swebench as _swebench

_ENVIRONMENT_MAPPING["swerex_modal"] = "envs.swerex_modal_closing.SwerexModalClosingEnvironment"

# The closing environment terminates its sandbox when the environment object is collected. On
# the text-based model path that happens by refcount the moment process_instance returns; on
# the litellm (native tool-call) path the environment sits in a reference cycle and waits for
# the cyclic collector, so sandboxes closed late and in bursts (3 Oct 2026, the gpt-5-mini
# comparator: 15-19 live containers for 6 workers). Collect after every instance.
_orig_process_instance = _swebench.process_instance


def _process_instance_then_collect(*args, **kwargs):
    try:
        return _orig_process_instance(*args, **kwargs)
    finally:
        gc.collect()


_swebench.process_instance = _process_instance_then_collect

if __name__ == "__main__":
    # Call the CLI object on the imported module rather than re-executing the module as __main__:
    # runpy would build a fresh module whose process_instance is the unwrapped original (the first
    # version of this launcher did that, and the gc.collect() wrapper silently never ran).
    import sys
    sys.argv[0] = "mini-extra swebench"
    _swebench.app()
