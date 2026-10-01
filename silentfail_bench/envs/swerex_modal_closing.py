"""A swerex_modal environment that actually terminates its Modal sandbox.

Two upstream defects stack (DEFECTS.md #21). mini-swe-agent 2.4.6's batch runner
(run/benchmarks/programbench.py, process_instance) never calls env.stop() after a
trajectory; and swerex's ModalDeployment.stop() only terminates the sandbox when
``poll()`` says it has ALREADY exited, so a live sandbox is never killed. Every
sandbox therefore lives to its deployment_timeout. On 1 Oct 2026 the Amendment 13
run accumulated 100 idle sandboxes, hit Modal's 100-container workspace cap, and
the next 14 instances died in SandboxTimeoutError before any model call.

This subclass terminates the sandbox the moment the environment object is released
(the agent drops it after save()), and again at interpreter exit for anything still
alive. It changes nothing about what the agent sees: same image, same commands,
same timeouts. ``stop()`` is also made effective for callers that do use it.

Applied by scripts/run_replication_modal.py as a ``-c environment.environment_class``
override so the registered YAMLs stay byte-identical; the trajectory's
info.config.environment_type records which class ran.
"""
from __future__ import annotations

import logging
import weakref

from minisweagent.environments.extra.swerex_modal import SwerexModalEnvironment

log = logging.getLogger("swerex_modal_closing")


def _terminate(deployment) -> None:
    sb = getattr(deployment, "_sandbox", None)
    if sb is None:
        return
    try:
        sb.terminate()            # synchronous Modal API; safe from a worker thread without a loop
        log.info("terminated sandbox %s", getattr(sb, "object_id", "?"))
    except Exception as e:        # a failed terminate must never break a finished trajectory
        log.warning("sandbox terminate failed: %s", e)
    finally:
        deployment._sandbox = None


class SwerexModalClosingEnvironment(SwerexModalEnvironment):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Holds the deployment, not self, so it cannot keep the environment alive.
        self._finalizer = weakref.finalize(self, _terminate, self.deployment)

    def stop(self):
        self._finalizer()
