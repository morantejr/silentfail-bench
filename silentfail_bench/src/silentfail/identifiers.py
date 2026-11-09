"""Stable identifiers and splits.

Defect fixes (2026-09-22):
  #1 repo_from_instance() mangled any repo whose name contains a hyphen
     ('scikit-learn__scikit-learn-10297' -> 'scikit'), corrupting the
     repo-level grouping that the held-out design depends on.
  #2 builtin hash() is salted per process (PYTHONHASHSEED), so split
     assignment was not reproducible across runs.
"""

from __future__ import annotations

import hashlib
import re

# SWE-bench instance ids are '<org>__<repo>-<issue_number>'. Only the FINAL
# '-<digits>' is the issue number; every earlier hyphen belongs to the repo.
_SWEBENCH_ID = re.compile(r"^(?P<repo>.+?)-(?P<issue>\d+)$")


def repo_from_instance(instance_id: str) -> str:
    """Group key for an instance id. Hyphen-safe."""
    m = _SWEBENCH_ID.match(instance_id)
    if m:
        return m.group("repo")
    return instance_id


def stable_hash(key: str) -> int:
    """Process-stable 64-bit hash. Replaces builtin hash() for any decision
    that must reproduce across runs (splits, deterministic mock gold)."""
    return int.from_bytes(hashlib.sha256(key.encode()).digest()[:8], "big")


def stable_bucket(key: str, buckets: int = 10, *, salt: str = "") -> int:
    return stable_hash(f"{salt}:{key}") % buckets


def assign_split(group_key: str, *, train_frac: float = 0.7, salt: str = "20260922") -> str:
    """Deterministic group-level train/test assignment.

    Grouping is by `group_key` (repo for SWE-bench, problem_id for the local-exec
    profile), so every row sharing a key lands on the same side of the split.
    """
    return "train" if stable_bucket(group_key, 1000, salt=salt) < int(train_frac * 1000) else "test"
