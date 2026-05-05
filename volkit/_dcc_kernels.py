"""
volkit/_dcc_kernels.py
======================
Thin Python wrappers around the C DCC Gaussian kernels.

All computation is in C (including internal memory management).
This module only:
  1. Pre-allocates numpy output buffers (grad, hess, scores, nll)
  2. Passes pointers to C via  array.ctypes.data
  3. Returns results from the pre-allocated buffers
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from typing import Tuple

from . import _core


def _ptr(arr: NDArray[np.float64]) -> int:
    """Return ctypes data pointer as int for passing to C."""
    return arr.ctypes.data


def _c(arr: NDArray[np.float64]) -> NDArray[np.float64]:
    """Ensure C-contiguous float64."""
    return np.ascontiguousarray(arr, dtype=np.float64)


# ─────────────────────────────────────────────────────────────────────────────
# NLL only
# ─────────────────────────────────────────────────────────────────────────────

def dcc_nll(
    theta: NDArray[np.float64],
    eps: NDArray[np.float64],
    Qbar: NDArray[np.float64],
    p: int, q: int,
) -> float:
    """DCC(p,q) Gaussian NLL (average over T).  Dispatches to _11 when applicable."""
    T, N = eps.shape
    theta_c = _c(theta.ravel())
    eps_c = _c(eps)
    Qbar_c = _c(Qbar.ravel())

    if p == 1 and q == 1:
        return _core._dcc_nll_11_gaussian(
            _ptr(theta_c), _ptr(eps_c), _ptr(Qbar_c), T, N)
    return _core._dcc_nll_pq_gaussian(
        _ptr(theta_c), _ptr(eps_c), _ptr(Qbar_c), T, N, p, q)


# ─────────────────────────────────────────────────────────────────────────────
# NLL + Gradient  (+ optional per-obs scores)
# ─────────────────────────────────────────────────────────────────────────────

def dcc_nll_grad(
    theta: NDArray[np.float64],
    eps: NDArray[np.float64],
    Qbar: NDArray[np.float64],
    p: int, q: int,
    return_scores: bool = False,
) -> Tuple[float, NDArray[np.float64], ...]:
    """DCC(p,q) NLL + gradient.  Returns (nll, grad) or (nll, grad, scores)."""
    T, N = eps.shape
    K = p + q
    theta_c = _c(theta.ravel())
    eps_c = _c(eps)
    Qbar_c = _c(Qbar.ravel())

    grad = np.empty(K, dtype=np.float64)
    nll_buf = np.empty(1, dtype=np.float64)
    scores = np.empty(T * K, dtype=np.float64) if return_scores else None
    scores_ptr = _ptr(scores) if scores is not None else 0

    if p == 1 and q == 1:
        _core._dcc_nll_grad_11_gaussian(
            _ptr(theta_c), _ptr(eps_c), _ptr(Qbar_c),
            _ptr(grad), _ptr(nll_buf), scores_ptr, T, N)
    else:
        _core._dcc_nll_grad_pq_gaussian(
            _ptr(theta_c), _ptr(eps_c), _ptr(Qbar_c),
            _ptr(grad), _ptr(nll_buf), scores_ptr, T, N, p, q)

    nll = float(nll_buf[0])
    if return_scores:
        return nll, grad, scores.reshape(T, K)  # type: ignore[union-attr]
    return nll, grad


# ─────────────────────────────────────────────────────────────────────────────
# NLL + Gradient + Hessian  (+ optional per-obs scores)
# ─────────────────────────────────────────────────────────────────────────────

def dcc_nll_grad_hess(
    theta: NDArray[np.float64],
    eps: NDArray[np.float64],
    Qbar: NDArray[np.float64],
    p: int, q: int,
    return_scores: bool = False,
) -> Tuple[float, NDArray[np.float64], NDArray[np.float64], ...]:
    """DCC(p,q) NLL + gradient + Hessian.  Returns (nll, grad, hess[, scores])."""
    T, N = eps.shape
    K = p + q
    theta_c = _c(theta.ravel())
    eps_c = _c(eps)
    Qbar_c = _c(Qbar.ravel())

    grad = np.empty(K, dtype=np.float64)
    hess = np.empty(K * K, dtype=np.float64)
    nll_buf = np.empty(1, dtype=np.float64)
    scores = np.empty(T * K, dtype=np.float64) if return_scores else None
    scores_ptr = _ptr(scores) if scores is not None else 0

    if p == 1 and q == 1:
        _core._dcc_nll_grad_hess_11_gaussian(
            _ptr(theta_c), _ptr(eps_c), _ptr(Qbar_c),
            _ptr(grad), _ptr(hess), _ptr(nll_buf), scores_ptr, T, N)
    else:
        _core._dcc_nll_grad_hess_pq_gaussian(
            _ptr(theta_c), _ptr(eps_c), _ptr(Qbar_c),
            _ptr(grad), _ptr(hess), _ptr(nll_buf), scores_ptr,
            T, N, p, q)

    nll = float(nll_buf[0])
    hess_mat = hess.reshape(K, K)
    if return_scores:
        return nll, grad, hess_mat, scores.reshape(T, K)  # type: ignore[union-attr]
    return nll, grad, hess_mat


# ─────────────────────────────────────────────────────────────────────────────
# Utilities
# ─────────────────────────────────────────────────────────────────────────────

def compute_qbar(eps: NDArray[np.float64]) -> NDArray[np.float64]:
    """Unconditional second-moment matrix:  Qbar = (1/T) Σ eps_t eps_t'."""
    return (eps.T @ eps) / eps.shape[0]
