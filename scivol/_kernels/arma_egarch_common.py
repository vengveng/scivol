from __future__ import annotations

import math

import numpy as np
from numpy.typing import NDArray

NORMAL_NLL_CONSTANT = 0.5 * math.log(2.0 * math.pi)


def arma_egarch_n_eff(n: int, p_ar: int, q_ma: int, p_arch: int, q_egarch: int) -> int:
    return n - max(p_ar, q_ma, p_arch, q_egarch, 1)


def arma_egarch_total_nll(
    avg_nll: float,
    n: int,
    p_ar: int,
    q_ma: int,
    p_arch: int,
    q_egarch: int,
    *,
    dist: str,
) -> float:
    n_eff = arma_egarch_n_eff(n, p_ar, q_ma, p_arch, q_egarch)
    if n_eff <= 0 or not math.isfinite(avg_nll):
        return float(avg_nll)

    total = float(avg_nll) * n_eff
    if dist == "normal":
        total += NORMAL_NLL_CONSTANT * n_eff
    return total


def arma_egarch_total_derivative(
    value: NDArray[np.float64],
    n: int,
    p_ar: int,
    q_ma: int,
    p_arch: int,
    q_egarch: int,
) -> NDArray[np.float64]:
    n_eff = arma_egarch_n_eff(n, p_ar, q_ma, p_arch, q_egarch)
    if n_eff <= 0:
        return value.copy()
    return value * float(n_eff)
