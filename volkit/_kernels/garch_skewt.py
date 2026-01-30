"""
GARCH(p,q) + Skewed Student-t likelihood (numerical derivatives).

UID handled:  "GARCH(p,q)+SkewT"

Uses the Hansen (1994) parameterization for the skewed Student-t distribution.
Since no analytical derivatives are available, numerical optimization is used.
"""

from __future__ import annotations
import re
import time
from typing import Dict, Tuple

import numpy as np
from numpy.typing import NDArray

from .. import _core
from ..components.vol import GARCH
from ..components.density import SkewT
from ..spec.composite import CompositeSpec
from ..result import EstimationResult
from .routine import Routine

# Import skewt likelihood from reference implementation
import sys
from pathlib import Path

# Add parent directory to path for importing likelihoods
_root = Path(__file__).resolve().parents[2]
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

from likelihoods import skewt_loglik

# ------------------------------------------------------------------ #
# cache (p,q) → Routine
# ------------------------------------------------------------------ #
_CACHE: Dict[Tuple[int, int], Routine] = {}

# ------------------------------------------------------------------ #
# helper
# ------------------------------------------------------------------ #
def _as_cptr(a: NDArray[np.float64]) -> int:
    return np.ascontiguousarray(a, np.float64).ctypes.data


def _garch_variance(
    theta_garch: NDArray[np.float64],
    resid2: NDArray[np.float64],
    sigma2: NDArray[np.float64],
    p: int,
    q: int,
) -> None:
    """Compute GARCH variance using C extension (modifies sigma2 in-place)."""
    n = len(resid2)
    if p == 1 and q == 1:
        _core._garch_variance_11(
            _as_cptr(theta_garch),
            _as_cptr(resid2),
            _as_cptr(sigma2),
            n
        )
    else:
        _core._garch_variance_pq(
            _as_cptr(theta_garch),
            _as_cptr(resid2),
            _as_cptr(sigma2),
            n, p, q
        )


# ------------------------------------------------------------------ #
# builder
# ------------------------------------------------------------------ #
def _build(p: int, q: int) -> Routine:
    uid = f"GARCH({p},{q})+SkewT"

    vol = GARCH(p, q)
    dens = SkewT()
    spec = CompositeSpec(vol, dens)

    n_garch = vol.n_params  # 1 + p + q
    n_dist = dens.n_params  # 2 (nu, lam)
    n_total = n_garch + n_dist

    # -------------------------------------------------------------------------
    def fit(
        resid: NDArray[np.float64],
        solver: str = "nelder-mead",
        verbose: bool = False,
        **_
    ) -> EstimationResult:

        from scipy.optimize import minimize, LinearConstraint

        t_start = time.perf_counter()

        n = resid.size
        resid2 = resid ** 2
        sigma2 = np.zeros(n, dtype=np.float64)
        sigma2[0] = np.mean(resid2)

        def objective(theta: NDArray[np.float64]) -> float:
            """Negative log-likelihood for GARCH + Skew-t."""
            theta_garch = theta[:n_garch]
            nu, lam = theta[n_garch], theta[n_garch + 1]

            # Compute GARCH variance using C extension
            _garch_variance(theta_garch, resid2, sigma2, p, q)

            # Compute skew-t log-likelihood using Python function
            ll = skewt_loglik(resid, sigma2, nu, lam)
            return -ll / n  # Return negative for minimization, scaled

        # Initial values and bounds
        start = np.concatenate((vol.default_start(resid), dens.default_start(resid)))
        bounds = vol.bounds() + dens.bounds()

        # Stationarity constraint: sum(alpha) + sum(beta) < 1
        # A @ theta gives: alpha_1 + ... + alpha_p + beta_1 + ... + beta_q
        A = np.array([[0] + [1] * p + [1] * q + [0, 0]])  # Skip omega and dist params
        lc = LinearConstraint(A, lb=1e-12, ub=1.0 - 1e-8)

        if solver.lower() == "nelder-mead":
            res = minimize(
                objective,
                start,
                method="Nelder-Mead",
                bounds=bounds,
                tol=1e-12,
                options={"maxfev": 50000, "disp": verbose}
            )
        elif solver.lower() == "slsqp":
            res = minimize(
                objective,
                start,
                method="SLSQP",
                bounds=bounds,
                constraints=lc,
                tol=1e-12,
                options={"disp": verbose, "ftol": 1e-12, "maxiter": 5000}
            )
        elif solver.lower() in ("trust-constr", "trust"):
            res = minimize(
                objective,
                start,
                method="trust-constr",
                bounds=bounds,
                constraints=lc,
                tol=1e-12,
                options={"disp": verbose, "xtol": 1e-6, "maxiter": 5000}
            )
        else:
            raise ValueError(f"Unknown solver '{solver}'")

        # Convert back to total negative log-likelihood
        res.fun = res.fun * n

        t_elapsed = time.perf_counter() - t_start

        # Unpack parameters into components
        vol.unpack(res.x[:n_garch])
        dens.unpack(res.x[n_garch:])

        # Compute final sigma2 for storage
        _garch_variance(res.x[:n_garch], resid2, sigma2, p, q)

        return EstimationResult(spec, res, resid, sigma2=sigma2.copy(), time_elapsed=t_elapsed)

    # -------------------------------------------------------------------------
    return Routine(
        uid=uid,
        fit=fit,
        n_params=n_total,
        start=lambda y: np.concatenate((vol.default_start(y), dens.default_start(y))),
        bounds=lambda: vol.bounds() + dens.bounds(),
    )


# ------------------------------------------------------------------ #
# public hook for the central registry
# ------------------------------------------------------------------ #
_UID_RE = re.compile(r"GARCH\((\d+),(\d+)\)\+SkewT$")


def get_routine(uid: str) -> Routine:
    m = _UID_RE.match(uid)
    if not m:
        raise RuntimeError(f"garch_skewt cannot handle uid '{uid}'")
    p, q = map(int, m.groups())
    return _CACHE.setdefault((p, q), _build(p, q))
