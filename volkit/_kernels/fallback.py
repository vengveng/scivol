"""
Extremely slow but dependency-free reference implementations.
They ensure the high-level estimator keeps working when the native
extension is not yet compiled.
"""

from __future__ import annotations

import numpy as np


# ------------------------------------------------------------------ #
# special: GARCH(1,1) + Normal
# ------------------------------------------------------------------ #
def garch11_normal_py(y: np.ndarray, theta: np.ndarray) -> float:
    omega, alpha, beta = theta
    n = y.size
    h = np.empty(n)
    h[0] = np.var(y)
    for t in range(1, n):
        h[t] = omega + alpha * y[t - 1] ** 2 + beta * h[t - 1]
    ll = -0.5 * (np.log(2 * np.pi) + np.log(h) + y**2 / h).sum()
    return ll


# ------------------------------------------------------------------ #
# general GARCH(p,q) + Normal            (Student-t omitted for brevity)
# ------------------------------------------------------------------ #
def garch_pq_normal_py(
    y: np.ndarray,
    theta: np.ndarray,
    spec,  # CompositeSpec, but typed loosely to avoid circular imports
) -> float:
    # read p, q from the volatility component
    from ..components import GARCH
    from ..roles import Role

    vol = spec.get_component(Role.VOLATILITY)
    assert isinstance(vol, GARCH)
    p, q = vol.p, vol.q

    omega = theta[0]
    alpha = theta[1 : 1 + p]
    beta = theta[1 + p : 1 + p + q]

    n = y.size
    h = np.empty(n)
    h[0] = np.var(y)

    for t in range(1, n):
        term_a = sum(alpha[i] * y[t - 1 - i] ** 2 for i in range(min(p, t)))
        term_b = sum(beta[j] * h[t - 1 - j] for j in range(min(q, t)))
        h[t] = omega + term_a + term_b

    ll = -0.5 * (np.log(2 * np.pi) + np.log(h) + y**2 / h).sum()
    return ll