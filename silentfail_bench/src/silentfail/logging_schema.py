"""Trajectory log schema helpers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

SCHEMA_VERSION = "0.1.0"

REQUIRED_FIELDS = (
    "trajectory_id",
    "instance_id",
    "repo",
    "split",
    "seed",
    "n_candidates",
    "selection_rule",
    "c",
    "c_sel_id",
    "audit_scores",
    "u",
    "g",
    "length_chars",
    "step_count",
    "mean_agent_entropy",
    "excluded",
    "agent_git_commit",
    "schema_version",
)


def schema_path() -> Path:
    return Path(__file__).resolve().parents[2] / "schema" / "trajectory_log.schema.json"


def load_schema() -> dict[str, Any]:
    return json.loads(schema_path().read_text())


def validate_row(row: dict[str, Any]) -> list[str]:
    """Lightweight validation without requiring jsonschema package."""
    errors: list[str] = []
    for key in REQUIRED_FIELDS:
        if key not in row:
            errors.append(f"missing required field: {key}")
    if row.get("schema_version") != SCHEMA_VERSION:
        errors.append(f"schema_version must be {SCHEMA_VERSION}")
    if "g" in row and row["g"] not in (0, 1):
        errors.append("g must be 0 or 1 (trajectory success)")
    if "excluded" in row and not isinstance(row["excluded"], bool):
        errors.append("excluded must be bool")
    if row.get("selection_rule") not in (None, "best_of_n"):
        errors.append("selection_rule must be best_of_n")
    return errors


def make_row(**kwargs: Any) -> dict[str, Any]:
    row = {"schema_version": SCHEMA_VERSION, **kwargs}
    errs = validate_row(row)
    if errs:
        raise ValueError("; ".join(errs))
    return row
