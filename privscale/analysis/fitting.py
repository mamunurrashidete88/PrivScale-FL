"""Candidate trend families for scaling-law analysis.

Fits three families against (d_i, y_i) pairs:
  - constant:     y = a
  - linear:       y = a + b*d
  - d_over_logd:  y = a + b * d / log(d)

Selection uses AIC computed from the residual sum of squares under a Gaussian
noise assumption; we also return R^2 for reporting. No family is privileged in
the fit pipeline; the selected model is simply the one with the best AIC.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class FitResult:
    name: str
    coefs: np.ndarray
    predictions: np.ndarray
    rss: float
    r2: float
    aic: float
    n_params: int


def _fit_ols(X: np.ndarray, y: np.ndarray, name: str) -> FitResult:
    coefs, *_ = np.linalg.lstsq(X, y, rcond=None)
    pred = X @ coefs
    resid = y - pred
    rss = float(resid @ resid)
    ss_tot = float(np.var(y) * len(y))
    r2 = 1.0 - rss / ss_tot if ss_tot > 0 else 0.0
    n = len(y)
    k = X.shape[1]
    # AICc for Gaussian residuals (small-sample-corrected; reduces to AIC as n grows).
    # Adding a tiny floor on rss/n prevents log(0) for perfectly-fit data.
    aic = n * np.log(rss / max(1, n) + 1e-24) + 2 * k
    if n - k - 1 > 0:
        aic += 2 * k * (k + 1) / (n - k - 1)
    return FitResult(name=name, coefs=coefs, predictions=pred, rss=rss, r2=r2, aic=float(aic), n_params=k)


def fit_candidates(d: np.ndarray, y: np.ndarray) -> dict[str, FitResult]:
    d = np.asarray(d, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    log_d = np.log(np.maximum(d, 2.0))

    fits: dict[str, FitResult] = {}
    # Constant: y = a
    X0 = np.ones((len(d), 1))
    fits["constant"] = _fit_ols(X0, y, "constant")
    # Linear: y = a + b*d
    X1 = np.column_stack([np.ones_like(d), d])
    fits["linear"] = _fit_ols(X1, y, "linear")
    # d / log d: y = a + b * d / log(d)
    X2 = np.column_stack([np.ones_like(d), d / log_d])
    fits["d_over_logd"] = _fit_ols(X2, y, "d_over_logd")
    return fits


def select_best(fits: dict[str, FitResult]) -> FitResult:
    return min(fits.values(), key=lambda f: f.aic)
