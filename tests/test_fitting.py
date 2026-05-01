"""Tests for the trend-fitting pipeline.

We don't test that any particular family wins on real data; we just verify
that fits are computed correctly and that selection picks the family that fits
synthetic data drawn from a known process.
"""
from __future__ import annotations

import numpy as np

from privscale.analysis.fitting import fit_candidates, select_best


def test_constant_data_prefers_constant():
    d = np.array([100, 200, 400, 800, 1600], dtype=float)
    y = np.full_like(d, 0.5) + np.random.default_rng(0).normal(0, 0.001, size=len(d))
    fits = fit_candidates(d, y)
    best = select_best(fits)
    assert best.name == "constant"


def test_linear_data_prefers_linear():
    rng = np.random.default_rng(0)
    d = np.array([100, 200, 400, 800, 1600, 3200], dtype=float)
    y = 0.1 + 1e-4 * d + rng.normal(0, 0.005, size=len(d))
    fits = fit_candidates(d, y)
    best = select_best(fits)
    assert best.name in ("linear", "d_over_logd")  # close families on small ranges


def test_d_over_logd_data_prefers_that_family():
    rng = np.random.default_rng(0)
    d = np.array([200, 500, 1000, 2000, 4000, 8000, 16000], dtype=float)
    y = 0.5 + 0.001 * d / np.log(d) + rng.normal(0, 0.01, size=len(d))
    fits = fit_candidates(d, y)
    best = select_best(fits)
    assert best.name == "d_over_logd"


def test_r2_in_range():
    d = np.array([10, 20, 40, 80, 160], dtype=float)
    y = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
    fits = fit_candidates(d, y)
    for f in fits.values():
        assert -0.1 <= f.r2 <= 1.001
