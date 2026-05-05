from __future__ import annotations

import re

import numpy as np
from numpy.typing import NDArray

import volkit._core as _c

from .ad_oracle import garch_value_grad_hess


def _as_ptr(a: NDArray[np.float64]) -> int:
    """Return a C-contiguous double* address."""
    return np.ascontiguousarray(a, np.float64).ctypes.data


def check_routine(
    routine,
    data: NDArray[np.float64],
    *,
    rtol: float = 1e-8,
    atol: float = 1e-10,
) -> None:
    """
    AD-oracle validation for any compiled GARCH Routine.

    Parameters
    ----------
    routine : volkit._kernels.base.Routine
        The object returned by ``get_routine("GARCH(p,q)+…")``.
    data : np.ndarray
        1-D residual series.
    rtol, atol : float
        Tolerances passed to ``np.allclose``.
    """
    theta0 = routine.start(data).astype(np.float64, copy=True)
    uid = routine.uid
    dist = "studentt" if "StudentT" in uid else "normal"
    p, q = map(int, re.search(r"GARCH\((\d+),(\d+)\)", uid).groups())  # type: ignore[union-attr]

    family = "_studentt" if dist == "studentt" else "_normal"
    try:
        g_c = getattr(_c, f"_garch_ll_grad_{p}{q}{family}")
        h_c = getattr(_c, f"_garch_ll_hess_{p}{q}{family}")
        extra: tuple[int, ...] = ()
    except AttributeError:
        g_c = getattr(_c, f"_garch_ll_grad_pq{family}")
        h_c = getattr(_c, f"_garch_ll_hess_pq{family}")
        extra = (p, q)

    resid2 = np.ascontiguousarray(data * data, dtype=np.float64)
    sigma = np.empty_like(resid2)
    sigma[0] = resid2.mean() or 1e-6

    grad_an = np.empty_like(theta0)
    g_c(_as_ptr(theta0), _as_ptr(resid2), _as_ptr(sigma), _as_ptr(grad_an), resid2.size, *extra)

    sigma[0] = resid2.mean() or 1e-6
    hess_an = np.empty((theta0.size, theta0.size), dtype=np.float64)
    h_c(_as_ptr(theta0), _as_ptr(resid2), _as_ptr(sigma), _as_ptr(hess_an), resid2.size, *extra)

    _, grad_ref, hess_ref = garch_value_grad_hess(theta0, resid2, p, q, dist=dist)

    abs_grad_err = np.abs(grad_ref - grad_an)
    abs_hess_err = np.abs(hess_ref - hess_an)
    eps = 1e-20
    grad_err = abs_grad_err / (np.abs(grad_an) + eps) * 100.0
    hess_err = abs_hess_err / (np.abs(hess_an) + eps) * 100.0

    _fmt = {"float_kind": lambda x: f"{x:12.6f}"}
    with np.printoptions(suppress=True, formatter=_fmt, linewidth=120):  # type: ignore[arg-type]
        print(f"Routine: {uid}")
        print(f"Sample size: {data.size}")
        print("=" * 80)
        print("GRADIENT COMPARISON")
        print("=" * 80)
        print(f"Gradient (Analytical):\n {grad_an}")
        print(f"Gradient (AD Oracle):\n {grad_ref}")
        print(f"Gradient rel-error [%]:\n {grad_err}\n")
        print("=" * 80)
        print("HESSIAN COMPARISON")
        print("=" * 80)
        print(f"Hessian (Analytical):\n{hess_an}\n")
        print(f"Hessian (AD Oracle):\n{hess_ref}\n")
        print(f"Hessian rel-error [%]:\n{hess_err}\n")
        print("-" * 80)
        print("EXTRA DIAGNOSTICS")
        print("-" * 80)
        print(f"Condition number of analytical Hessian:\n {np.linalg.cond(hess_an):.2e}")
        print(f"Eigenvalues of analytical Hessian:\n {np.sort(np.linalg.eigvalsh(hess_an))}")

    print("\n" + "=" * 80)
    print("SUMMARY STATISTICS")
    print("=" * 80)
    print(f"Gradient max rel-error: {grad_err.max():.2e} %")
    print(f"Hessian max rel-error:  {hess_err.max():.2e} %")

    gradient_ok = np.allclose(grad_ref, grad_an, rtol=rtol, atol=atol)
    hessian_ok = np.allclose(hess_ref, hess_an, rtol=rtol, atol=atol)

    if not gradient_ok:
        idx = np.unravel_index(abs_grad_err.argmax(), abs_grad_err.shape)
        raise AssertionError(f"Gradient mismatch at {idx}: rel Δ={grad_err.max():.3e}%")

    if not hessian_ok:
        idx = np.unravel_index(abs_hess_err.argmax(), abs_hess_err.shape)
        raise AssertionError(f"Hessian mismatch at {idx}: rel Δ={hess_err.max():.3e}%")

    print("\n✓ All derivatives validated successfully!")
    print(f"  Gradient: max rel-error = {grad_err.max():.2e} %")
    print(f"  Hessian:  max rel-error = {hess_err.max():.2e} %")