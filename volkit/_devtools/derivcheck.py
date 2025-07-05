# volkit/_devtools/derivcheck.py
from __future__ import annotations
import re
from typing import Callable

import numpy as np
import numdifftools as nd
from numpy.typing import NDArray

import volkit._core as _c


# ───────────────────────── helpers ──────────────────────────────────────────
def _as_ptr(a: NDArray[np.float64]) -> int:
    """Return a C-contiguous double* address."""
    return np.ascontiguousarray(a, np.float64).ctypes.data


def _finite_step(theta: NDArray[np.float64],
                 lo: NDArray[np.float64],
                 hi: NDArray[np.float64],
                 base_rel: float = 1e-4) -> NDArray[np.float64]:
    """Vector of per-parameter step sizes that stay inside [lo, hi]."""
    h = np.maximum(np.abs(theta), 1.0) * base_rel
    h = np.minimum(h, (hi - theta) / 3.1)      # leave margin
    h = np.minimum(h, (theta - lo) / 3.1)
    h[h == 0] = base_rel                       # for free params
    return h


# ───────────────────────── main check routine ──────────────────────────────
def check_routine(
    routine,
    data: NDArray[np.float64],
    *,
    rtol: float = 1e-8,
    atol: float = 1e-10,
) -> None:
    """
    Finite-difference validation for any compiled volkit Routine.

    Parameters
    ----------
    routine : volkit._kernels.base.Routine
        The object returned by ``get_routine("GARCH(p,q)+…")``.
    data : np.ndarray
        1-D residual series.
    rtol, atol : float
        Tolerances passed to ``np.allclose``.
    """
    # ── pick an interior θ₀ -------------------------------------------------
    theta0 = routine.start(data).astype(float, copy=True)
    bds = routine.bounds()
    lo = np.array([b[0] if b[0] is not None else -np.inf for b in bds])
    hi = np.array([b[1] if b[1] is not None else +np.inf for b in bds])
    theta0 = np.where(np.isfinite(lo + hi), 0.5 * (lo + hi), theta0)
    h_vec = _finite_step(theta0, lo, hi)

    # ── choose underlying C symbols ----------------------------------------
    uid = routine.uid
    family = "_studentt" if "StudentT" in uid else "_normal"
    p, q = map(int, re.search(r"GARCH\((\d+),(\d+)\)", uid).groups()) # type: ignore

    try:                       # specialised (1,1)
        f_c = getattr(_c, f"_garch_ll_{p}{q}{family}")
        g_c = getattr(_c, f"_garch_ll_grad_{p}{q}{family}")
        h_c = getattr(_c, f"_garch_ll_hess_{p}{q}{family}")
        extra: tuple[int, ...] = ()
    except AttributeError:     # generic (p,q)
        f_c = getattr(_c, f"_garch_ll_pq{family}")
        g_c = getattr(_c, f"_garch_ll_grad_pq{family}")
        h_c = getattr(_c, f"_garch_ll_hess_pq{family}")
        extra = (p, q)

    # ── allocate shared buffers --------------------------------------------
    eps2 = data * data
    sigma = np.empty_like(eps2)
    eps2_ptr, n = _as_ptr(eps2), data.size

    # ── python wrappers to call C ------------------------------------------
    def _ensure_sigma_seed() -> None:
        sigma[0] = eps2.mean() or 1e-6   # positive seed

    def obj(th: NDArray[np.float64]) -> float:
        _ensure_sigma_seed()
        return f_c(_as_ptr(th), eps2_ptr, _as_ptr(sigma), n, *extra)

    def grad_an(th: NDArray[np.float64]) -> NDArray[np.float64]:
        buf = np.empty_like(th)
        _ensure_sigma_seed()
        g_c(_as_ptr(th), eps2_ptr, _as_ptr(sigma), buf.ctypes.data, n, *extra)
        return buf

    def hess_an(th: NDArray[np.float64]) -> NDArray[np.float64]:
        k = th.size
        buf = np.empty((k, k))
        _ensure_sigma_seed()
        h_c(_as_ptr(th), eps2_ptr, _as_ptr(sigma), buf.ctypes.data, n, *extra)
        return buf

    # ── numerical derivatives (8-point central) ----------------------------
    step_scale = 10.0
    step_dict = {"step": h_vec * step_scale, "method": "central"}
    grad_fd = nd.Gradient(obj, **step_dict)(theta0)
    hess_direct = nd.Hessian(obj, **step_dict)(theta0)      # 2nd-diff FD Hessian
    hess_jacobian = nd.Jacobian(grad_an, **step_dict)(theta0)  # Jacobian of analytical gradient

    # ── compute all values -------------------------------------------------
    grad_fd_val = grad_fd
    grad_an_val = grad_an(theta0)
    hess_an_val = hess_an(theta0)
    hess_direct_val = hess_direct
    hess_jacobian_val = hess_jacobian

    # ── compute errors -----------------------------------------------------
    abs_grad_err = np.abs(grad_fd_val - grad_an_val)
    abs_hess_direct_err = np.abs(hess_direct_val - hess_an_val)
    abs_hess_jacobian_err = np.abs(hess_jacobian_val - hess_an_val)
    abs_hess_methods_err = np.abs(hess_direct_val - hess_jacobian_val) # type: ignore

    # relative error (%) – avoid divide-by-zero with a tiny floor
    eps = 1e-20
    grad_err = abs_grad_err / (np.abs(grad_an_val) + eps) * 100.0
    hess_direct_err = abs_hess_direct_err / (np.abs(hess_an_val) + eps) * 100.0
    hess_jacobian_err = abs_hess_jacobian_err / (np.abs(hess_an_val) + eps) * 100.0
    hess_methods_err = abs_hess_methods_err / (np.abs(hess_an_val) + eps) * 100.0

    # ── display results ----------------------------------------------------
    _fmt = {'float_kind': lambda x: f"{x:12.6f}"}

    with np.printoptions(suppress=True, formatter=_fmt, linewidth=120):  # type: ignore
        print(f"Routine: {routine.uid}")
        print(f"Sample size: {data.size}")

        print("=" * 80)
        print("GRADIENT COMPARISON")
        print("=" * 80)
        print(f"Gradient (Analytical):\n {grad_an_val}")
        print(f"Gradient (Numerical):\n {grad_fd_val}")
        print(f"Gradient rel-error [%]:\n {grad_err}\n")

        print("=" * 80)
        print("HESSIAN COMPARISON")
        print("=" * 80)
        print(f"Hessian (Analytical):\n{hess_an_val}\n")
        
        print(f"Hessian (2nd-diff FD):\n{hess_direct_val}\n")
        
        print(f"Hessian (Jacobian-FD):\n{hess_jacobian_val}\n")

        print("-" * 80)
        print("HESSIAN ERROR ANALYSIS")
        print("-" * 80)
        print(f"2nd-diff FD vs Analytical - rel-error [%]:\n{hess_direct_err}\n")
        
        print(f"Jacobian-FD vs Analytical - rel-error [%]:\n{hess_jacobian_err}\n")
        
        print(f"2nd-diff FD vs Jacobian-FD - rel-error [%]:\n{hess_methods_err}\n")

        print("-" * 80)
        print("RATIO ANALYSIS")
        print("-" * 80)
        print(f"2nd-diff FD / Analytical:\n{hess_direct_val / (hess_an_val + eps)}\n")
        
        print(f"Jacobian-FD / Analytical:\n{hess_jacobian_val / (hess_an_val + eps)}\n")
        
        print(f"2nd-diff FD / Jacobian-FD:\n{hess_direct_val / (hess_jacobian_val + eps)}") # type: ignore

        print("-" * 80)
        print("EXTRA DIAGNOSTICS")
        print("-" * 80)

        print(f"Condition number of analytical Hessian:\n {np.linalg.cond(hess_an_val):.2e}")
        print(f"Per-element sign matrix (an vs fd):\n{np.sign(hess_an_val) != np.sign(hess_direct_val)}")
        idx = np.unravel_index(abs_hess_methods_err.argmax(), abs_hess_methods_err.shape)
        print(f"Worst discrepancy at index:\n {idx} with rel-error {hess_methods_err.max():.2e} %")
        print(f"Eigenvalues of analytical Hessian:\n {np.sort(np.linalg.eigvalsh(hess_an_val))}")
        print(f"Step sizes used for FD:\n {h_vec * step_scale}")

    # ── summary statistics -------------------------------------------------
    print("\n" + "=" * 80)
    print("SUMMARY STATISTICS")
    print("=" * 80)
    print(f"Gradient max rel-error:           {grad_err.max():.2e} %")
    print(f"Hessian (direct) max rel-error:   {hess_direct_err.max():.2e} %")
    print(f"Hessian (Jacobian) max rel-error: {hess_jacobian_err.max():.2e} %")
    print(f"Hessian methods difference:       {hess_methods_err.max():.2e} %")

    # ── validation checks --------------------------------------------------
    gradient_ok = np.allclose(grad_fd_val, grad_an_val, rtol=rtol, atol=np.inf)
    hess_direct_ok = np.allclose(hess_direct_val, hess_an_val, rtol=rtol * 10, atol=np.inf)
    hess_jacobian_ok = np.allclose(hess_jacobian_val, hess_an_val, rtol=rtol * 10, atol=np.inf)

    if not gradient_ok:
        idx = np.unravel_index(abs_grad_err.argmax(), abs_grad_err.shape)
        raise AssertionError(f"Gradient mismatch at {idx}: "
                             f"rel Δ={grad_err.max():.3e}%")

    if not hess_direct_ok:
        idx = np.unravel_index(abs_hess_direct_err.argmax(), abs_hess_direct_err.shape)
        raise AssertionError(f"Direct Hessian mismatch at {idx}: "
                             f"rel Δ={hess_direct_err.max():.3e}%")

    if not hess_jacobian_ok:
        idx = np.unravel_index(abs_hess_jacobian_err.argmax(), abs_hess_jacobian_err.shape)
        raise AssertionError(f"Jacobian Hessian mismatch at {idx}: "
                             f"rel Δ={hess_jacobian_err.max():.3e}%")

    print("\n✓ All derivatives validated successfully!")
    print(f"  Gradient:              max rel-error = {grad_err.max():.2e} %")
    print(f"  Hessian (2nd-diff FD): max rel-error = {hess_direct_err.max():.2e} %")
    print(f"  Hessian (Jacobian-FD): max rel-error = {hess_jacobian_err.max():.2e} %")