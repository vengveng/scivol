"""
volkit._kernels  -  run-time kernel registry

Public API
----------
get_special_kernel(uid: str)
get_general_kernel()
"""

from __future__ import annotations

import importlib
import warnings
from typing import Callable, Dict, Optional

import numpy as np

from ..spec import CompositeSpec
from ..components import GARCH, StudentT
from ..roles import Role

# ------------------------------------------------------------------ #
# Internal storage
# ------------------------------------------------------------------ #
_SPECIAL: Dict[str, Callable[[np.ndarray, np.ndarray], float]] = {}
_GENERAL: Optional[Callable[[np.ndarray, np.ndarray, CompositeSpec], float]] = None

def _register_special(uid: str, func: Callable[[np.ndarray, np.ndarray], float]):
    _SPECIAL[uid] = func

def _set_general(func: Callable[[np.ndarray, np.ndarray, CompositeSpec], float]):
    global _GENERAL
    _GENERAL = func

# ------------------------------------------------------------------ #
# 1.  Attempt to import compiled extension
# ------------------------------------------------------------------ #
try:
    _c = importlib.import_module("volkit._core")  # the binary .so / .pyd
    HAVE_CORE = True
except ModuleNotFoundError:
    _c = None
    HAVE_CORE = False
    warnings.warn(
        "volkit compiled core not found - using slow pure-Python kernels.",
        RuntimeWarning,
    )

# ------------------------------------------------------------------ #
# 2.  Register kernels
# ------------------------------------------------------------------ #
if HAVE_CORE:
    # -------- special: GARCH(1,1)+Normal ---------------------------
    if hasattr(_c, "_special_garch_oo_normal"):
        def _garch11_normal(y: np.ndarray, theta: np.ndarray) -> float:
            return _c._special_garch_oo_normal(y, theta)

        _register_special("GARCH(1,1)+Normal", _garch11_normal)

    # (add more one-off kernels here as soon as the C layer exports them)

    # -------- general GARCH(p,q)+density ---------------------------
    if {
        "_garch_variance_pq",
        "_normal_likelihood",
        "_any_studentt_likelihood",
    } <= set(dir(_c)):

        def _general(y: np.ndarray, theta: np.ndarray, spec: CompositeSpec) -> float:
            """
            Generic (potentially slow) bridge:

            1) call the C routine _garch_variance_pq to obtain σ²_t
            2) call the matching density likelihood
            """
            vol = spec.get_component(Role.VOLATILITY)
            dens = spec.get_component(Role.DENSITY)
            assert isinstance(vol, GARCH)
            p, q = vol.p, vol.q
            n = y.size

            # --- allocate contiguous buffers -----------------------
            resid2 = np.ascontiguousarray(y**2, dtype=np.float64)
            sigma2 = np.empty_like(resid2)

            # --- step 1: conditional variance ----------------------
            _c._garch_variance_pq(
                theta.ctypes.data,
                resid2.ctypes.data,
                sigma2.ctypes.data,
                n,
                p,
                q,
            )

            # --- step 2: log-likelihood ----------------------------
            if isinstance(dens, StudentT):
                # df is last element of theta by construction
                nu = float(theta[-1])
                llh = _c._any_studentt_likelihood(
                    sigma2.ctypes.data, resid2.ctypes.data, n, nu
                )
            else:  # Normal default
                llh = _c._normal_likelihood(
                    sigma2.ctypes.data, resid2.ctypes.data, n
                )

            return llh

        _set_general(_general)

# ------------------------------------------------------------------ #
# 3.  Pure-Python fall-backs
# ------------------------------------------------------------------ #
if _GENERAL is None:
    from .fallback import (  # do the import only if needed
        garch11_normal_py,
        garch_pq_normal_py,
    )

    _register_special("GARCH(1,1)+Normal", garch11_normal_py)
    _set_general(garch_pq_normal_py)

# ------------------------------------------------------------------ #
# 4.  Public accessors
# ------------------------------------------------------------------ #
def get_special_kernel(
    uid: str,
) -> Optional[Callable[[np.ndarray, np.ndarray], float]]:
    return _SPECIAL.get(uid)


def get_general_kernel() -> Optional[
    Callable[[np.ndarray, np.ndarray, CompositeSpec], float]
]:
    return _GENERAL