"""Rényi differential privacy accounting for the Gaussian mechanism.

Implements the subsampled Gaussian RDP accountant of Mironov et al. (2017)
and Wang, Balle, Kasiviswanathan (2019), restricted to Poisson subsampling.
All formulas follow the presentation in Mironov (2017) and the standard
dp-accounting/tensorflow-privacy implementations.

We keep dependencies minimal: a single file with pure-Python functions.

References
----------
- Mironov, I. (2017). Rényi Differential Privacy. CSF.
- Wang, Y.-X., Balle, B., Kasiviswanathan, S. (2019). Subsampled Rényi DP.
- Mironov, Talwar, Zhang (2019). Rényi Divergence of Gaussian mixtures.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy import special


_DEFAULT_ORDERS = tuple(
    list(range(2, 64)) + [128, 256, 512]
)


def _log_add(a: float, b: float) -> float:
    if a == -np.inf:
        return b
    if b == -np.inf:
        return a
    if a > b:
        return a + math.log1p(math.exp(b - a))
    return b + math.log1p(math.exp(a - b))


def _log_sub(a: float, b: float) -> float:
    if a < b:
        raise ValueError("log_sub requires a >= b")
    if a == b:
        return -np.inf
    # log(exp(a) - exp(b)) = a + log(1 - exp(b-a))
    return a + math.log1p(-math.exp(b - a))


def _log_comb(n: int, k: int) -> float:
    return (
        special.gammaln(n + 1)
        - special.gammaln(k + 1)
        - special.gammaln(n - k + 1)
    )


def _compute_log_a_int(q: float, sigma: float, alpha: int) -> float:
    """Log of A_alpha for integer alpha, per Mironov (2017) Eq. (4)-(5)."""
    log_a = -np.inf
    for i in range(alpha + 1):
        log_coef_i = (
            _log_comb(alpha, i)
            + i * math.log(q)
            + (alpha - i) * math.log(1.0 - q)
        )
        s = log_coef_i + (i * i - i) / (2.0 * sigma * sigma)
        log_a = _log_add(log_a, s)
    return log_a


def _compute_log_a_frac(q: float, sigma: float, alpha: float) -> float:
    """Log of A_alpha for non-integer alpha (Mironov, Talwar, Zhang 2019)."""
    log_a0, log_a1 = -np.inf, -np.inf
    i = 0
    z0 = sigma * sigma * math.log(1.0 / q - 1.0) + 0.5
    while True:
        coef = special.binom(alpha, i)
        log_coef = math.log(abs(coef))
        j = alpha - i
        log_t0 = log_coef + i * math.log(q) + j * math.log(1.0 - q)
        log_t1 = log_coef + j * math.log(q) + i * math.log(1.0 - q)
        log_e0 = math.log(0.5) + _log_erfc((i - z0) / (math.sqrt(2) * sigma))
        log_e1 = math.log(0.5) + _log_erfc((z0 - j) / (math.sqrt(2) * sigma))
        log_s0 = log_t0 + (i * i - i) / (2 * sigma * sigma) + log_e0
        log_s1 = log_t1 + (j * j - j) / (2 * sigma * sigma) + log_e1
        if coef > 0:
            log_a0 = _log_add(log_a0, log_s0)
            log_a1 = _log_add(log_a1, log_s1)
        else:
            log_a0 = _log_sub(log_a0, log_s0) if log_a0 > log_s0 else log_a0
            log_a1 = _log_sub(log_a1, log_s1) if log_a1 > log_s1 else log_a1
        i += 1
        if max(log_s0, log_s1) < -30:
            break
        if i > 1000:
            break
    return _log_add(log_a0, log_a1)


def _log_erfc(x: float) -> float:
    try:
        return math.log(special.erfc(x))
    except ValueError:
        # very negative x: erfc ≈ 2; very positive: use asymptotic
        if x < -5:
            return math.log(2.0)
        return math.log(special.erfcx(x)) - x * x


def rdp_for_subsampled_gaussian(q: float, sigma: float, alpha: float) -> float:
    """RDP at order alpha for a single application of the subsampled Gaussian
    mechanism with sampling rate q and noise multiplier sigma.
    """
    if q == 0:
        return 0.0
    if q == 1.0:
        return alpha / (2.0 * sigma * sigma)
    if float(alpha).is_integer():
        log_a = _compute_log_a_int(q, sigma, int(alpha))
    else:
        log_a = _compute_log_a_frac(q, sigma, alpha)
    return log_a / (alpha - 1.0)


def compute_rdp(
    q: float, sigma: float, steps: int, orders: tuple = _DEFAULT_ORDERS
) -> np.ndarray:
    """RDP vector across orders for `steps` applications."""
    out = np.zeros(len(orders), dtype=np.float64)
    for i, a in enumerate(orders):
        out[i] = steps * rdp_for_subsampled_gaussian(q, sigma, float(a))
    return out


def rdp_to_epsilon(
    rdp: np.ndarray, delta: float, orders: tuple = _DEFAULT_ORDERS
) -> tuple[float, float]:
    """Convert an RDP vector to an (eps, best_order) pair at target delta.

    Uses the Mironov (2017) conversion: eps = rdp_alpha + log(1/delta) / (alpha-1).
    Returns the minimum over orders.
    """
    best_eps = float("inf")
    best_alpha = float(orders[0])
    for a, r in zip(orders, rdp):
        eps = r + math.log(1.0 / delta) / (a - 1.0)
        if eps < best_eps:
            best_eps = eps
            best_alpha = float(a)
    return best_eps, best_alpha


@dataclass
class RDPAccountant:
    """Accumulates RDP across rounds for a single (q, sigma) configuration.

    If q or sigma changes across rounds, call `step(q_t, sigma_t)` per round
    instead of `accumulate(steps)`.
    """
    orders: tuple = _DEFAULT_ORDERS

    def __post_init__(self):
        self.rdp = np.zeros(len(self.orders), dtype=np.float64)

    def step(self, q: float, sigma: float, n: int = 1) -> None:
        for i, a in enumerate(self.orders):
            self.rdp[i] += n * rdp_for_subsampled_gaussian(q, sigma, float(a))

    def get_epsilon(self, delta: float) -> tuple[float, float]:
        return rdp_to_epsilon(self.rdp, delta=delta, orders=self.orders)


def find_sigma_for_epsilon(
    target_epsilon: float,
    delta: float,
    q: float,
    steps: int,
    sigma_lo: float = 0.3,
    sigma_hi: float = 100.0,
    tol: float = 1e-3,
    orders: tuple = _DEFAULT_ORDERS,
) -> float:
    """Bisection search for the smallest sigma achieving (epsilon, delta) after
    `steps` applications at sampling rate `q`.
    """
    def eps_at(sigma: float) -> float:
        rdp = compute_rdp(q, sigma, steps, orders)
        return rdp_to_epsilon(rdp, delta, orders)[0]

    # Ensure sigma_hi is large enough
    while eps_at(sigma_hi) > target_epsilon:
        sigma_hi *= 2
        if sigma_hi > 1e6:
            raise RuntimeError("Could not find a sigma upper bound for target epsilon")
    # And sigma_lo small enough
    while eps_at(sigma_lo) < target_epsilon:
        sigma_lo /= 2
        if sigma_lo < 1e-4:
            # Already tiny enough; target is very loose
            return sigma_lo

    while sigma_hi - sigma_lo > tol:
        mid = 0.5 * (sigma_lo + sigma_hi)
        if eps_at(mid) > target_epsilon:
            sigma_lo = mid
        else:
            sigma_hi = mid
    return sigma_hi
