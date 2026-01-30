"""
Test C extension vs Python reference implementation.

This module provides detailed comparisons between:
1. C extension functions in volkit._core
2. Python reference functions in likelihoods.py

The goal is to identify numerical discrepancies, particularly in:
- Log-likelihood computation
- Gradient computation
- Hessian computation
- OPG (Outer Product of Gradients) computation

Usage:
    python tests/test_c_vs_python.py           # Run all comparisons
    python tests/test_c_vs_python.py --verbose # Detailed output
    pytest tests/test_c_vs_python.py -v        # Via pytest
"""

from __future__ import annotations

import sys
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
from numpy.typing import NDArray

# Add root to path for imports
_root = Path(__file__).resolve().parents[1]
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

import volkit._core as _c
from likelihoods import (
    garch11_normal_hessian,
    garch11_normal_gradient,
    garch11_normal_robust_se,
    garch11_studentt_hessian,
    garch11_studentt_gradient,
)


# =============================================================================
# HELPERS
# =============================================================================

def _as_cptr(a: NDArray[np.float64]) -> int:
    """Return a C-contiguous float64 array's data pointer as int."""
    return np.ascontiguousarray(a, np.float64).ctypes.data


@dataclass
class ComparisonResult:
    """Result of comparing C vs Python computation."""
    name: str
    c_value: NDArray[np.float64]
    py_value: NDArray[np.float64]
    abs_diff: NDArray[np.float64]
    rel_diff: NDArray[np.float64]
    max_abs_diff: float
    max_rel_diff: float
    passed: bool
    
    def __str__(self) -> str:
        status = "PASS" if self.passed else "FAIL"
        return (f"{self.name}: [{status}] "
                f"max_abs={self.max_abs_diff:.2e}, max_rel={self.max_rel_diff:.2e}")


def compare(
    name: str, 
    c_val: NDArray[np.float64], 
    py_val: NDArray[np.float64],
    rtol: float = 1e-6,
    atol: float = 1e-10,
) -> ComparisonResult:
    """Compare two arrays and return detailed comparison result."""
    c_val = np.atleast_1d(c_val)
    py_val = np.atleast_1d(py_val)
    
    abs_diff = np.abs(c_val - py_val)
    # Avoid division by zero
    denom = np.maximum(np.abs(py_val), 1e-20)
    rel_diff = abs_diff / denom
    
    max_abs = float(np.max(abs_diff))
    max_rel = float(np.max(rel_diff))
    
    passed = np.allclose(c_val, py_val, rtol=rtol, atol=atol)
    
    return ComparisonResult(
        name=name,
        c_value=c_val,
        py_value=py_val,
        abs_diff=abs_diff,
        rel_diff=rel_diff,
        max_abs_diff=max_abs,
        max_rel_diff=max_rel,
        passed=passed,
    )


def generate_test_data(
    n: int = 500,
    seed: int = 42,
) -> Tuple[NDArray[np.float64], NDArray[np.float64], float]:
    """
    Generate test data for GARCH estimation.
    
    Returns:
        resid: Residual series
        resid2: Squared residuals
        sigma2_init: Initial variance
    """
    np.random.seed(seed)
    
    # Simulate GARCH(1,1) process
    omega, alpha, beta = 1e-6, 0.05, 0.93
    
    sigma2 = np.empty(n)
    resid = np.empty(n)
    
    sigma2[0] = omega / (1 - alpha - beta)  # unconditional variance
    resid[0] = np.sqrt(sigma2[0]) * np.random.randn()
    
    for t in range(1, n):
        sigma2[t] = omega + alpha * resid[t-1]**2 + beta * sigma2[t-1]
        resid[t] = np.sqrt(sigma2[t]) * np.random.randn()
    
    resid2 = resid ** 2
    sigma2_init = np.mean(resid2)
    
    return resid, resid2, sigma2_init


# =============================================================================
# C EXTENSION WRAPPERS
# =============================================================================

def c_garch11_normal_ll(
    params: NDArray[np.float64],
    resid2: NDArray[np.float64],
    sigma2_init: float,
) -> Tuple[float, NDArray[np.float64]]:
    """Compute GARCH(1,1) Normal log-likelihood using C extension."""
    n = len(resid2)
    sigma2 = np.empty(n, dtype=np.float64)
    sigma2[0] = sigma2_init
    
    nll = _c._garch_ll_11_normal(
        _as_cptr(params),
        _as_cptr(resid2),
        _as_cptr(sigma2),
        n,
    )
    
    return nll, sigma2


def c_garch11_normal_gradient(
    params: NDArray[np.float64],
    resid2: NDArray[np.float64],
    sigma2_init: float,
) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Compute GARCH(1,1) Normal gradient using C extension."""
    n = len(resid2)
    sigma2 = np.empty(n, dtype=np.float64)
    sigma2[0] = sigma2_init
    grad = np.empty(3, dtype=np.float64)
    
    _c._garch_ll_grad_11_normal(
        _as_cptr(params),
        _as_cptr(resid2),
        _as_cptr(sigma2),
        _as_cptr(grad),
        n,
    )
    
    return grad, sigma2


def c_garch11_normal_hessian(
    params: NDArray[np.float64],
    resid2: NDArray[np.float64],
    sigma2_init: float,
) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Compute GARCH(1,1) Normal Hessian using C extension."""
    n = len(resid2)
    sigma2 = np.empty(n, dtype=np.float64)
    sigma2[0] = sigma2_init
    hess = np.empty((3, 3), dtype=np.float64)
    
    _c._garch_ll_hess_11_normal(
        _as_cptr(params),
        _as_cptr(resid2),
        _as_cptr(sigma2),
        _as_cptr(hess),
        n,
    )
    
    return hess, sigma2


def c_garch11_opg_hess(
    params: NDArray[np.float64],
    resid2: NDArray[np.float64],
    sigma2: NDArray[np.float64],
) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Compute GARCH(1,1) OPG and Hessian using C extension (errors_garch.c)."""
    n = len(resid2)
    opg = np.empty((3, 3), dtype=np.float64)
    hess = np.empty((3, 3), dtype=np.float64)
    
    _c._garch_opg_hess_11(
        _as_cptr(params),
        _as_cptr(resid2),
        _as_cptr(sigma2),
        _as_cptr(opg),
        _as_cptr(hess),
        n,
    )
    
    return opg, hess


def c_garch11_studentt_gradient(
    params: NDArray[np.float64],
    resid2: NDArray[np.float64],
    sigma2_init: float,
) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Compute GARCH(1,1) Student-t gradient using C extension."""
    n = len(resid2)
    sigma2 = np.empty(n, dtype=np.float64)
    sigma2[0] = sigma2_init
    grad = np.empty(4, dtype=np.float64)
    
    _c._garch_ll_grad_11_studentt(
        _as_cptr(params),
        _as_cptr(resid2),
        _as_cptr(sigma2),
        _as_cptr(grad),
        n,
    )
    
    return grad, sigma2


def c_garch11_studentt_hessian(
    params: NDArray[np.float64],
    resid2: NDArray[np.float64],
    sigma2_init: float,
) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Compute GARCH(1,1) Student-t Hessian using C extension."""
    n = len(resid2)
    sigma2 = np.empty(n, dtype=np.float64)
    sigma2[0] = sigma2_init
    hess = np.empty((4, 4), dtype=np.float64)
    
    _c._garch_ll_hess_11_studentt(
        _as_cptr(params),
        _as_cptr(resid2),
        _as_cptr(sigma2),
        _as_cptr(hess),
        n,
    )
    
    return hess, sigma2


# =============================================================================
# NUMERICAL DERIVATIVES (FINITE DIFFERENCES)
# =============================================================================

def numerical_gradient(
    objective: callable,
    params: NDArray[np.float64],
    eps: float = 1e-7,
) -> NDArray[np.float64]:
    """Compute gradient via central finite differences."""
    k = len(params)
    grad = np.empty(k)
    
    for i in range(k):
        params_plus = params.copy()
        params_minus = params.copy()
        params_plus[i] += eps
        params_minus[i] -= eps
        
        grad[i] = (objective(params_plus) - objective(params_minus)) / (2 * eps)
    
    return grad


def numerical_hessian(
    objective: callable,
    params: NDArray[np.float64],
    eps: float = 1e-5,
) -> NDArray[np.float64]:
    """Compute Hessian via central finite differences."""
    k = len(params)
    hess = np.empty((k, k))
    f0 = objective(params)
    
    for i in range(k):
        for j in range(k):
            params_pp = params.copy()
            params_pm = params.copy()
            params_mp = params.copy()
            params_mm = params.copy()
            
            params_pp[i] += eps
            params_pp[j] += eps
            
            params_pm[i] += eps
            params_pm[j] -= eps
            
            params_mp[i] -= eps
            params_mp[j] += eps
            
            params_mm[i] -= eps
            params_mm[j] -= eps
            
            hess[i, j] = (
                objective(params_pp) 
                - objective(params_pm) 
                - objective(params_mp) 
                + objective(params_mm)
            ) / (4 * eps * eps)
    
    return hess


# =============================================================================
# COMPARISON FUNCTIONS
# =============================================================================

def compare_garch11_normal(
    params: NDArray[np.float64],
    resid2: NDArray[np.float64],
    sigma2_init: float,
    verbose: bool = True,
) -> dict:
    """
    Compare GARCH(1,1) + Normal: C vs Python vs Numerical.
    """
    results = {}
    
    print("\n" + "=" * 70)
    print("GARCH(1,1) + NORMAL: C vs Python vs Numerical")
    print("=" * 70)
    print(f"Parameters: omega={params[0]:.2e}, alpha={params[1]:.4f}, beta={params[2]:.4f}")
    print(f"Sample size: {len(resid2)}")
    print()
    
    # -- Gradient comparison --
    print("-" * 70)
    print("GRADIENT COMPARISON")
    print("-" * 70)
    
    grad_c, sigma2_c = c_garch11_normal_gradient(params, resid2, sigma2_init)
    py_result = garch11_normal_gradient(params, resid2, sigma2_init)
    grad_py = py_result.grad
    
    # Numerical gradient using C objective
    def c_obj(p):
        nll, _ = c_garch11_normal_ll(p, resid2, sigma2_init)
        return nll
    grad_num = numerical_gradient(c_obj, params)
    
    results['grad_c_vs_py'] = compare("Gradient C vs Python", grad_c, grad_py)
    results['grad_c_vs_num'] = compare("Gradient C vs Numerical", grad_c, grad_num)
    results['grad_py_vs_num'] = compare("Gradient Python vs Numerical", grad_py, grad_num)
    
    if verbose:
        print(f"  C gradient:         {grad_c}")
        print(f"  Python gradient:    {grad_py}")
        print(f"  Numerical gradient: {grad_num}")
        print()
    
    print(f"  {results['grad_c_vs_py']}")
    print(f"  {results['grad_c_vs_num']}")
    print(f"  {results['grad_py_vs_num']}")
    
    # -- Hessian comparison --
    print()
    print("-" * 70)
    print("HESSIAN COMPARISON")
    print("-" * 70)
    
    hess_c, _ = c_garch11_normal_hessian(params, resid2, sigma2_init)
    py_hess_result = garch11_normal_hessian(params, resid2, sigma2_init)
    hess_py = py_hess_result.hess
    
    hess_num = numerical_hessian(c_obj, params)
    
    results['hess_c_vs_py'] = compare("Hessian C vs Python", hess_c, hess_py)
    results['hess_c_vs_num'] = compare("Hessian C vs Numerical", hess_c, hess_num)
    results['hess_py_vs_num'] = compare("Hessian Python vs Numerical", hess_py, hess_num)
    
    if verbose:
        print("  C Hessian:")
        print(hess_c)
        print()
        print("  Python Hessian:")
        print(hess_py)
        print()
        print("  Numerical Hessian:")
        print(hess_num)
        print()
    
    print(f"  {results['hess_c_vs_py']}")
    print(f"  {results['hess_c_vs_num']}")
    print(f"  {results['hess_py_vs_num']}")
    
    # -- OPG comparison (critical - suspected issue) --
    print()
    print("-" * 70)
    print("OPG COMPARISON (errors_garch.c vs likelihoods.py)")
    print("-" * 70)
    
    # Need sigma2 computed at the same params
    _, sigma2 = c_garch11_normal_ll(params, resid2, sigma2_init)
    
    opg_c, hess_approx_c = c_garch11_opg_hess(params, resid2, sigma2)
    
    # Python robust_se computes OPG correctly (but doesn't scale by 1/n)
    py_robust = garch11_normal_robust_se(params, resid2, sigma2_init)
    opg_py = py_robust.opg
    n = len(resid2)
    
    # C scales by 1/n, so we need to scale Python's OPG for comparison
    results['opg_c_vs_py'] = compare("OPG C vs Python/n", opg_c, opg_py / n, rtol=0.01, atol=1e-6)
    results['opg_hess_c_vs_py'] = compare("OPG-Hess C vs Python-Hess/n", hess_approx_c, py_robust.hess / n, rtol=0.01, atol=1e-6)
    
    if verbose:
        print("  C OPG (from errors_garch.c):")
        print(opg_c)
        print()
        print("  Python OPG (from likelihoods.py):")
        print(opg_py)
        print()
        print("  C OPG / Python OPG (ratio):")
        with np.errstate(divide='ignore', invalid='ignore'):
            ratio = opg_c / opg_py
            ratio = np.where(np.isfinite(ratio), ratio, np.nan)
        print(ratio)
        print()
    
    print(f"  {results['opg_c_vs_py']}")
    print(f"  {results['opg_hess_c_vs_py']}")
    
    return results


def compare_garch11_studentt(
    params: NDArray[np.float64],
    resid2: NDArray[np.float64],
    sigma2_init: float,
    verbose: bool = True,
) -> dict:
    """
    Compare GARCH(1,1) + Student-t: C vs Python vs Numerical.
    """
    results = {}
    
    print("\n" + "=" * 70)
    print("GARCH(1,1) + STUDENT-T: C vs Python vs Numerical")
    print("=" * 70)
    print(f"Parameters: omega={params[0]:.2e}, alpha={params[1]:.4f}, "
          f"beta={params[2]:.4f}, nu={params[3]:.2f}")
    print(f"Sample size: {len(resid2)}")
    print()
    
    # -- Gradient comparison --
    print("-" * 70)
    print("GRADIENT COMPARISON")
    print("-" * 70)
    
    grad_c, sigma2_c = c_garch11_studentt_gradient(params, resid2, sigma2_init)
    py_result = garch11_studentt_gradient(params, resid2, sigma2_init)
    grad_py = py_result.grad
    
    # Numerical gradient using C objective
    def c_obj(p):
        n = len(resid2)
        sigma2 = np.empty(n, dtype=np.float64)
        sigma2[0] = sigma2_init
        nll = _c._garch_ll_11_studentt(
            _as_cptr(p),
            _as_cptr(resid2),
            _as_cptr(sigma2),
            n,
        )
        return nll
    
    grad_num = numerical_gradient(c_obj, params)
    
    results['grad_c_vs_py'] = compare("Gradient C vs Python", grad_c, grad_py)
    results['grad_c_vs_num'] = compare("Gradient C vs Numerical", grad_c, grad_num)
    results['grad_py_vs_num'] = compare("Gradient Python vs Numerical", grad_py, grad_num)
    
    if verbose:
        print(f"  C gradient:         {grad_c}")
        print(f"  Python gradient:    {grad_py}")
        print(f"  Numerical gradient: {grad_num}")
        print()
    
    print(f"  {results['grad_c_vs_py']}")
    print(f"  {results['grad_c_vs_num']}")
    print(f"  {results['grad_py_vs_num']}")
    
    # -- Hessian comparison --
    print()
    print("-" * 70)
    print("HESSIAN COMPARISON")
    print("-" * 70)
    
    hess_c, _ = c_garch11_studentt_hessian(params, resid2, sigma2_init)
    py_hess_result = garch11_studentt_hessian(params, resid2, sigma2_init)
    hess_py = py_hess_result.hess
    
    hess_num = numerical_hessian(c_obj, params)
    
    results['hess_c_vs_py'] = compare("Hessian C vs Python", hess_c, hess_py)
    results['hess_c_vs_num'] = compare("Hessian C vs Numerical", hess_c, hess_num)
    results['hess_py_vs_num'] = compare("Hessian Python vs Numerical", hess_py, hess_num)
    
    if verbose:
        print("  C Hessian:")
        print(hess_c)
        print()
        print("  Python Hessian:")
        print(hess_py)
        print()
        print("  Numerical Hessian:")
        print(hess_num)
        print()
    
    print(f"  {results['hess_c_vs_py']}")
    print(f"  {results['hess_c_vs_num']}")
    print(f"  {results['hess_py_vs_num']}")
    
    return results


def diagnose_opg_issue(
    params: NDArray[np.float64],
    resid2: NDArray[np.float64],
    sigma2_init: float,
) -> None:
    """
    Detailed diagnosis of the OPG discrepancy.
    
    The C code in errors_garch.c computes:
        grad[0] = 1.0
        grad[1] = residuals2[t-1]
        grad[2] = sigma2[t-1]
    
    But the correct recursive derivative is:
        d[0] = 1.0 + beta * d_prev[0]
        d[1] = resid2[t-1] + beta * d_prev[1]
        d[2] = sigma2[t-1] + beta * d_prev[2]
    """
    print("\n" + "=" * 70)
    print("OPG ISSUE DIAGNOSIS")
    print("=" * 70)
    
    omega, alpha, beta = params[0], params[1], params[2]
    n = len(resid2)
    
    # Compute sigma2
    sigma2 = np.empty(n, dtype=np.float64)
    sigma2[0] = sigma2_init
    for t in range(1, n):
        sigma2[t] = omega + alpha * resid2[t-1] + beta * sigma2[t-1]
    
    # C-style derivatives (NO recursion - what errors_garch.c does)
    grad_c_style = np.zeros((n, 3))
    grad_c_style[0, 0] = 1.0
    for t in range(1, n):
        grad_c_style[t, 0] = 1.0
        grad_c_style[t, 1] = resid2[t-1]
        grad_c_style[t, 2] = sigma2[t-1]
    
    # Python-style derivatives (WITH recursion - correct)
    grad_py_style = np.zeros((n, 3))
    d_prev = np.zeros(3)
    for t in range(1, n):
        d_curr = np.array([
            1.0 + beta * d_prev[0],
            resid2[t-1] + beta * d_prev[1],
            sigma2[t-1] + beta * d_prev[2],
        ])
        grad_py_style[t] = d_curr
        d_prev = d_curr
    
    # Compare at a few time points
    print("\nPer-observation derivatives ∂σ²_t/∂θ:")
    print("-" * 70)
    print(f"{'t':>5} {'C-style[0]':>12} {'Py-style[0]':>12} {'C-style[1]':>12} {'Py-style[1]':>12} {'C-style[2]':>12} {'Py-style[2]':>12}")
    print("-" * 70)
    
    for t in [1, 5, 10, 50, 100, min(200, n-1)]:
        if t < n:
            c = grad_c_style[t]
            p = grad_py_style[t]
            print(f"{t:5d} {c[0]:12.4e} {p[0]:12.4e} {c[1]:12.4e} {p[1]:12.4e} {c[2]:12.4e} {p[2]:12.4e}")
    
    # Compute OPG both ways
    opg_c_style = np.zeros((3, 3))
    opg_py_style = np.zeros((3, 3))
    
    for t in range(n):
        inv_s2 = 1.0 / sigma2[t]
        res_os = resid2[t] * inv_s2
        c_grad_t = 0.5 * (1.0 - res_os) * inv_s2
        
        # Score at t
        score_c = c_grad_t * grad_c_style[t]
        score_py = c_grad_t * grad_py_style[t]
        
        opg_c_style += np.outer(score_c, score_c)
        opg_py_style += np.outer(score_py, score_py)
    
    print("\n\nOPG with C-style (non-recursive) derivatives:")
    print(opg_c_style)
    
    print("\nOPG with Python-style (recursive) derivatives:")
    print(opg_py_style)
    
    print("\nRatio (C-style / Python-style):")
    with np.errstate(divide='ignore', invalid='ignore'):
        ratio = opg_c_style / opg_py_style
        ratio = np.where(np.isfinite(ratio), ratio, np.nan)
    print(ratio)
    
    print("\n\nActual C function _garch_opg_hess_11 output (scaled by 1/n):")
    opg_actual_c, hess_actual_c = c_garch11_opg_hess(params, resid2, sigma2)
    print(opg_actual_c)
    
    print("\nPython-style OPG / n (to match C scaling):")
    print(opg_py_style / n)
    
    print("\nConclusion:")
    if np.allclose(opg_actual_c, opg_py_style / n, rtol=0.01):
        print("  -> FIXED! The C function now uses RECURSIVE derivatives correctly")
    elif np.allclose(opg_actual_c, opg_c_style / n, rtol=0.01):
        print("  -> The C function uses NON-RECURSIVE derivatives (bug still present)")
    else:
        print("  -> The C function behavior is different from both implementations")


# =============================================================================
# PYTEST TESTS
# =============================================================================

def test_garch11_normal_gradient():
    """Test GARCH(1,1) Normal gradient: C vs Python."""
    resid, resid2, sigma2_init = generate_test_data()
    params = np.array([1e-6, 0.05, 0.93])
    
    grad_c, _ = c_garch11_normal_gradient(params, resid2, sigma2_init)
    py_result = garch11_normal_gradient(params, resid2, sigma2_init)
    
    np.testing.assert_allclose(grad_c, py_result.grad, rtol=1e-6)


def test_garch11_normal_hessian():
    """Test GARCH(1,1) Normal Hessian: C vs Python."""
    resid, resid2, sigma2_init = generate_test_data()
    params = np.array([1e-6, 0.05, 0.93])
    
    hess_c, _ = c_garch11_normal_hessian(params, resid2, sigma2_init)
    py_result = garch11_normal_hessian(params, resid2, sigma2_init)
    
    np.testing.assert_allclose(hess_c, py_result.hess, rtol=1e-6)


def test_garch11_studentt_gradient():
    """Test GARCH(1,1) Student-t gradient: C vs Python."""
    resid, resid2, sigma2_init = generate_test_data()
    params = np.array([1e-6, 0.05, 0.93, 8.0])
    
    grad_c, _ = c_garch11_studentt_gradient(params, resid2, sigma2_init)
    py_result = garch11_studentt_gradient(params, resid2, sigma2_init)
    
    np.testing.assert_allclose(grad_c, py_result.grad, rtol=1e-5)


def test_garch11_studentt_hessian():
    """Test GARCH(1,1) Student-t Hessian: C vs Python."""
    resid, resid2, sigma2_init = generate_test_data()
    params = np.array([1e-6, 0.05, 0.93, 8.0])
    
    hess_c, _ = c_garch11_studentt_hessian(params, resid2, sigma2_init)
    py_result = garch11_studentt_hessian(params, resid2, sigma2_init)
    
    np.testing.assert_allclose(hess_c, py_result.hess, rtol=1e-4)


# =============================================================================
# MAIN
# =============================================================================

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Compare C vs Python GARCH implementations")
    parser.add_argument("--verbose", "-v", action="store_true", help="Show detailed output")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--n", type=int, default=500, help="Sample size")
    args = parser.parse_args()
    
    # Generate test data
    resid, resid2, sigma2_init = generate_test_data(n=args.n, seed=args.seed)
    
    # Test parameters
    params_normal = np.array([1e-6, 0.05, 0.93])
    params_studentt = np.array([1e-6, 0.05, 0.93, 8.0])
    
    # Run comparisons
    results_normal = compare_garch11_normal(params_normal, resid2, sigma2_init, verbose=args.verbose)
    results_studentt = compare_garch11_studentt(params_studentt, resid2, sigma2_init, verbose=args.verbose)
    
    # Diagnose OPG issue
    diagnose_opg_issue(params_normal, resid2, sigma2_init)
    
    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    
    all_passed = True
    for name, result in {**results_normal, **results_studentt}.items():
        status = "PASS" if result.passed else "FAIL"
        if not result.passed:
            all_passed = False
        print(f"  {name}: [{status}] max_rel={result.max_rel_diff:.2e}")
    
    print()
    if all_passed:
        print("All comparisons PASSED")
    else:
        print("Some comparisons FAILED - check detailed output above")
    
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
