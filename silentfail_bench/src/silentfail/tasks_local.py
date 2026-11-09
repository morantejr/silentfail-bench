"""Profile-A task family: HumanEval+ with a visible/held-out test separation.

Gold g is REAL executed unit tests, not a hash. The separation matters:
  visible_test : base HumanEval asserts. The agent may run these between steps.
  gold_test    : HumanEval+ extended tests. Never shown to the agent, never
                 shown to any critic. Defines g.

This is the EvalPlus design intent -- base tests are known-weak, so passing
them while failing `plus` is exactly the silent failure the project studies.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from silentfail.identifiers import stable_hash

DATA = Path(__file__).resolve().parents[2] / "data" / "humaneval" / "humaneval_plus_merged.json"


@dataclass
class Task:
    task_id: str
    entry_point: str
    prompt: str
    visible_test: str
    gold_test: str
    reference_solution: str = ""   # EvalPlus canonical; control only, never shown

    @property
    def group_key(self) -> str:
        """Split unit for profile A. NOT a repo -- see anchor_free.yaml."""
        return self.task_id


def load_tasks() -> list[Task]:
    return [Task(**r) for r in json.loads(DATA.read_text())]


def sample_tasks(n: int, seed: int = 20260922) -> list[Task]:
    tasks = load_tasks()
    tasks.sort(key=lambda t: stable_hash(f"{seed}:{t.task_id}"))
    return tasks[:n]


def harness_source(solution: str, test_src: str, entry_point: str) -> str:
    """Assemble a self-contained script: solution + tests + check(entry_point)."""
    return f"{solution}\n\n{test_src}\n\ncheck({entry_point})\nprint('__ALL_TESTS_PASSED__')\n"
