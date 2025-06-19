"""
Stub file for the compiled extension **volkit._core**

Every pointer parameter is typed as `_IntPtr`, a union of:

  • `int`                       ─ raw address, e.g. `array.ctypes.data`
  • `ctypes.c_void_p`           ─ generic void pointer
  • `ctypes._Pointer[Any]`      ─ any typed ctypes pointer

Editors and type-checkers use this file only; at runtime CPython loads the
matching _core.cpython-*.so.
"""

from typing import Any, Union
import ctypes

_IntPtr = Union[int, ctypes.c_void_p, ctypes._Pointer[Any]]
_Size    = Union[int, ctypes.c_size_t]

# ── General GARCH(p,q) helpers ────────────────────────────────────────────
def _garch_variance_pq(
    theta_ptr  : _IntPtr,
    resid2_ptr : _IntPtr,
    sigma2_ptr : _IntPtr,
    n: _Size,
    p: _Size,
    q: _Size,
) -> None: ...

def _normal_likelihood(
    sigma2_ptr : _IntPtr,
    resid2_ptr : _IntPtr,
    n: _Size,
) -> float: ...

def _general_garch_pq_std_err_robust(
    resid2_ptr : _IntPtr,
    sigma2_ptr : _IntPtr,
    OPG_ptr    : _IntPtr,
    HESS_ptr   : _IntPtr,
    n: _Size,
    p: _Size,
    q: _Size,
) -> None: ...

# ── Special GARCH(1,1) helpers ────────────────────────────────────────────
def _special_garch_oo_normal(
    theta_ptr  : _IntPtr,
    resid2_ptr : _IntPtr,
    sigma2_ptr : _IntPtr,
    n: _Size,
) -> float: ...

def _special_garch_oo_normal_variance(
    theta_ptr  : _IntPtr,
    resid2_ptr : _IntPtr,
    sigma2_ptr : _IntPtr,
    n: _Size,
) -> None: ...

def _special_garch_11_std_err_robust(
    resid2_ptr : _IntPtr,
    sigma2_ptr : _IntPtr,
    OPG_ptr    : _IntPtr,
    HESS_ptr   : _IntPtr,
    n: _Size,
) -> None: ...

# ── Student-t likelihood ─────────────────────────────────────────────────
def _any_studentt_likelihood(
    sigma2_ptr : _IntPtr,
    r2os2_ptr  : _IntPtr,
    n: _Size,
    nu: float,
) -> float: ...

# Nothing is meant for star-import; keep top-level clean
__all__: list[str] = []