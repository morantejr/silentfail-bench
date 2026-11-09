"""Unit tests for schema, scoring, combine, mock pilot."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from silentfail.combine import auroc, delta_auroc, fit_models
from silentfail.logging_schema import make_row, validate_row
from silentfail.pilot import go_no_go, pilot_summary, simulate_power
from silentfail.runner import run_pilot_mock
from silentfail.scoring import fit_rank_norms, uncertainty_std


ROOT = Path(__file__).resolve().parents[1]


def test_make_row_ok():
    row = make_row(
        trajectory_id="t1",
        instance_id="org__repo-1",
        repo="org__repo",
        split="train",
        seed=0,
        n_candidates=16,
        selection_rule="best_of_n",
        c=1.2,
        c_sel_id="sel",
        audit_scores={"a": 0.1, "b": 0.2},
        u=0.05,
        g=0,
        length_chars=10,
        step_count=3,
        mean_agent_entropy=None,
        excluded=False,
        agent_git_commit="abc",
    )
    assert validate_row(row) == []


def test_rank_norm_and_u():
    fits = fit_rank_norms({"a": [0.0, 1.0, 2.0], "b": [0.0, 1.0, 2.0]})
    assert 0.0 <= fits["a"].transform(1.0) <= 1.0
    assert uncertainty_std({"a": 0.1, "b": 0.9}) > 0


def test_delta_auroc_runs():
    rng = np.random.default_rng(0)
    n = 80
    c = rng.normal(size=n)
    u = c + rng.normal(scale=0.5, size=n)
    g = (c + 0.5 * u + rng.normal(scale=0.5, size=n) > 0).astype(int)
    ctrl = rng.normal(size=(n, 3))
    models = fit_models(c[:50], u[:50], ctrl[:50], g[:50])
    metrics = delta_auroc(models, c[50:], u[50:], ctrl[50:], g[50:])
    assert "delta_AUROC" in metrics
    assert auroc(np.array([1, 0, 1, 0]), np.array([0.9, 0.1, 0.8, 0.2])) > 0.5


def test_mock_pilot_and_gonogo(tmp_path: Path):
    out = tmp_path / "pilot.jsonl"
    scored = run_pilot_mock(ROOT / "configs" / "anchor.yaml", out, n=40)
    rows = [json.loads(l) for l in scored.read_text().splitlines() if l.strip()]
    assert len(rows) == 40
    summary = pilot_summary(rows)
    assert summary["n_usable"] > 0
    decision = go_no_go(summary, min_failures=1)
    assert decision["decision"] in {"GO", "NO_GO"}


def test_power_sim():
    sim = simulate_power(0.4, 0.05, 100, n_sims=20, seed=1)
    assert 0.0 <= sim["power_delta_gt_0"] <= 1.0
