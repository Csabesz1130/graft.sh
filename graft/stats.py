"""Exact binomial statistics for certificates. Standard library only."""
from __future__ import annotations

import math


def _binom_cdf(k: int, n: int, p: float) -> float:
    """P(X <= k) for X ~ Binomial(n, p)."""
    if k < 0:
        return 0.0
    if k >= n:
        return 1.0
    if p <= 0.0:
        return 1.0
    if p >= 1.0:
        return 0.0
    logs = [math.lgamma(n + 1) - math.lgamma(i + 1) - math.lgamma(n - i + 1)
            + i * math.log(p) + (n - i) * math.log1p(-p) for i in range(k + 1)]
    top = max(logs)
    return min(1.0, math.exp(top) * sum(math.exp(v - top) for v in logs))


def lower_bound(passed: int, total: int, confidence: float = 0.95) -> float:
    """One-sided Clopper-Pearson lower bound on the true pass rate.

    The largest p for which seeing `passed` or more passes out of `total` still has probability
    at most 1 - confidence. With every case passing this is (1 - confidence) ** (1 / total).
    """
    if total <= 0 or passed <= 0:
        return 0.0
    alpha = 1.0 - confidence
    if passed == total:
        return alpha ** (1.0 / total)
    lo, hi = 0.0, passed / total
    for _ in range(80):
        mid = (lo + hi) / 2
        # P(X >= passed | p = mid) grows with p; the bound is where it reaches alpha
        if 1.0 - _binom_cdf(passed - 1, total, mid) > alpha:
            hi = mid
        else:
            lo = mid
    return lo


def sign_test(wins: int, losses: int) -> float:
    """Two-sided exact sign test (McNemar's exact test) for paired outcomes.

    `wins` are cases only the new model passes, `losses` cases only the old one passes.
    Returns the p-value of the difference under "both models are equally good".
    """
    n = wins + losses
    if n == 0:
        return 1.0
    return min(1.0, 2.0 * _binom_cdf(min(wins, losses), n, 0.5))
