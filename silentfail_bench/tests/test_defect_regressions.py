"""Regressions for the four Stage-0 defects found 2026-09-22."""

from __future__ import annotations

import subprocess
import sys

import numpy as np
import pytest

from silentfail.combine import delta_auroc, fit_models
from silentfail.identifiers import assign_split, repo_from_instance, stable_hash
from silentfail.stats import auroc_fast, cluster_bootstrap_delta_auroc, simulate_power


# --- defect #1: hyphenated repo names were truncated ------------------------
@pytest.mark.parametrize("instance_id,expected", [
    ("astropy__astropy-12907", "astropy__astropy"),
    ("scikit-learn__scikit-learn-10297", "scikit-learn__scikit-learn"),
    ("sphinx-doc__sphinx-8721", "sphinx-doc__sphinx"),
    ("pylint-dev__pylint-6506", "pylint-dev__pylint"),
    ("django__django-11099", "django__django"),
])
def test_repo_grouping_survives_hyphens(instance_id, expected):
    assert repo_from_instance(instance_id) == expected


# --- defect #2: builtin hash() is per-process salted ------------------------
def test_split_is_reproducible_across_processes():
    code = (
        "import sys; sys.path.insert(0,'src');"
        "from silentfail.identifiers import assign_split;"
        "print([assign_split(r) for r in "
        "['django__django','scikit-learn__scikit-learn','psf__requests','sympy__sympy']])"
    )
    outs = {
        subprocess.run([sys.executable, "-c", code], capture_output=True, text=True).stdout.strip()
        for _ in range(3)
    }
    assert len(outs) == 1, f"split assignment differs across processes: {outs}"


def test_split_respects_train_fraction():
    keys = [f"org__repo{i}" for i in range(2000)]
    frac = sum(1 for k in keys if assign_split(k) == "train") / len(keys)
    assert 0.66 < frac < 0.74


# --- defect #3: IRLS diverged under separation, yielding NaN ----------------
def test_irls_survives_separation_and_huge_control_scale():
    n = 80
    c = np.linspace(-3, 3, n)
    u = np.linspace(-1, 1, n)
    g = (c > 0).astype(int)                       # perfectly separable
    ctrl = np.column_stack([np.abs(c) * 1e5, np.arange(n)])
    idx = np.argsort(stable_hash("x") % 7 + np.arange(n) % 3)   # mix classes into both splits
    c, u, g, ctrl = c[idx], u[idx], g[idx], ctrl[idx]
    m = fit_models(c[:55], u[:55], ctrl[:55], g[:55])
    for model in m.values():
        assert np.all(np.isfinite(model.coef)) and np.isfinite(model.intercept)
    r = delta_auroc(m, c[55:], u[55:], ctrl[55:], g[55:])
    assert np.isfinite(r["AUROC_M_c"])


# --- defect #4: "power" was ~0.4 under the true null ------------------------
def test_delta_test_is_calibrated_under_the_null():
    r = simulate_power(0.65, 0.0, 120, 1, n_sims=120, n_boot=250, seed=11)
    assert r["n_sims_valid"] > 0
    # Must be near or below alpha -- the old statistic scored 0.39-0.44 here.
    assert r["rejection_rate"] <= 0.10, f"not calibrated: {r['rejection_rate']}"


def test_power_rises_with_a_real_effect():
    null = simulate_power(0.65, 0.0, 160, 1, n_sims=80, n_boot=200, seed=3)
    real = simulate_power(0.65, 1.5, 160, 1, n_sims=80, n_boot=200, seed=3)
    assert real["rejection_rate"] > null["rejection_rate"] + 0.3


def test_skipped_sims_are_not_counted_as_failures():
    r = simulate_power(0.65, 0.0, 120, 1, n_sims=60, n_boot=150, seed=2)
    assert r["n_sims_valid"] + r["n_sims_skipped"] == r["n_sims_requested"]


def test_auroc_fast_matches_reference_with_ties():
    from silentfail.combine import auroc
    rng = np.random.default_rng(0)
    for _ in range(10):
        y = rng.integers(0, 2, 150)
        s = np.round(rng.normal(size=150), 1)      # forces ties
        assert abs(auroc(y, s) - auroc_fast(y, s)) < 1e-9


def test_cluster_bootstrap_respects_groups():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 120)
    s = rng.normal(size=120)
    groups = np.repeat(np.arange(12), 10)
    t = cluster_bootstrap_delta_auroc(y, s, s + 1e-9 * rng.normal(size=120), groups, n_boot=200)
    assert t.n_groups == 12
    assert t.ci_lo <= t.delta <= t.ci_hi or np.isnan(t.ci_lo)
