"""
Symbolic Derivative Verification for Volkit Kernels
====================================================

Developer tool for verifying analytical derivatives using SymPy.

Key Design Principles (from the decomposition approach):
1. Don't expand the whole likelihood symbolically
2. Write kernels as scalar building blocks + forward sensitivity recursions
3. Each time step contributes via outer products and second-sensitivity terms

This module provides:
- Symbolic definitions of likelihood building blocks
- Scalar partial derivatives (∂ℓ/∂e, ∂ℓ/∂h, ∂²ℓ/∂e², etc.)
- Verification against numerical finite differences
- CSE (common subexpression elimination) for compact code generation

Supported distributions:
- Normal
- Student-t
- Hansen (1994) Skew-t

Future: AR mean equations, AR-GARCH coupling
"""

from __future__ import annotations
import numpy as np
from typing import Dict, Tuple, Callable, List
import sympy as sp
from sympy import (
    Symbol, symbols, sqrt, log, exp, pi, gamma, Abs, sign,
    diff, simplify, cse, lambdify, Piecewise, Function
)
from sympy.stats import density
from dataclasses import dataclass


# =============================================================================
# SYMBOLIC BUILDING BLOCKS
# =============================================================================

@dataclass
class ScalarPartials:
    """Container for scalar partial derivatives of log-likelihood."""
    ell_e: sp.Expr      # ∂ℓ/∂e
    ell_h: sp.Expr      # ∂ℓ/∂h
    ell_ee: sp.Expr     # ∂²ℓ/∂e²
    ell_hh: sp.Expr     # ∂²ℓ/∂h²
    ell_eh: sp.Expr     # ∂²ℓ/∂e∂h
    
    # For skew-t, we also need z-based partials
    ell_z: sp.Expr = None       # ∂ℓ/∂z (standardized residual)
    ell_zz: sp.Expr = None      # ∂²ℓ/∂z²
    ell_nu: sp.Expr = None      # ∂ℓ/∂ν
    ell_lam: sp.Expr = None     # ∂ℓ/∂λ


def normal_scalar_partials() -> Tuple[sp.Expr, ScalarPartials]:
    """
    Normal distribution scalar building blocks.
    
    Per-t log-likelihood (up to constant):
        ℓ_t = -½ log(h) - ½ e²/h
    
    Returns
    -------
    ell : symbolic log-likelihood
    partials : ScalarPartials with first and second derivatives
    """
    e, h = symbols('e h', real=True, positive=True)
    
    # Log-likelihood (ignoring constant -½ log(2π))
    ell = -sp.Rational(1, 2) * log(h) - sp.Rational(1, 2) * e**2 / h
    
    # First derivatives
    ell_e = diff(ell, e)  # -e/h
    ell_h = diff(ell, h)  # -½(1/h - e²/h²) = -½(h - e²)/h²
    
    # Second derivatives
    ell_ee = diff(ell_e, e)  # -1/h
    ell_hh = diff(ell_h, h)  # ½(1/h² - 2e²/h³)
    ell_eh = diff(ell_e, h)  # e/h²
    
    return ell, ScalarPartials(
        ell_e=simplify(ell_e),
        ell_h=simplify(ell_h),
        ell_ee=simplify(ell_ee),
        ell_hh=simplify(ell_hh),
        ell_eh=simplify(ell_eh),
    )


def studentt_scalar_partials() -> Tuple[sp.Expr, ScalarPartials]:
    """
    Student-t distribution scalar building blocks.
    
    Per-t log-likelihood:
        ℓ_t = log(c) - ½ log(h) - ½(ν+1) log(1 + e²/(h(ν-2)))
    
    where c = Γ((ν+1)/2) / (Γ(ν/2) √(π(ν-2)))
    """
    e, h, nu = symbols('e h nu', real=True, positive=True)
    
    # Constant (log of normalizing constant)
    c_log = sp.loggamma((nu + 1) / 2) - sp.loggamma(nu / 2) - sp.Rational(1, 2) * log(pi * (nu - 2))
    
    # Standardized squared residual
    r2_scaled = e**2 / (h * (nu - 2))
    
    # Log-likelihood
    ell = c_log - sp.Rational(1, 2) * log(h) - sp.Rational(1, 2) * (nu + 1) * log(1 + r2_scaled)
    
    # First derivatives wrt e and h
    ell_e = diff(ell, e)
    ell_h = diff(ell, h)
    
    # Second derivatives
    ell_ee = diff(ell_e, e)
    ell_hh = diff(ell_h, h)
    ell_eh = diff(ell_e, h)
    
    return ell, ScalarPartials(
        ell_e=simplify(ell_e),
        ell_h=simplify(ell_h),
        ell_ee=simplify(ell_ee),
        ell_hh=simplify(ell_hh),
        ell_eh=simplify(ell_eh),
    )


def hansen_skewt_scalar_partials() -> Tuple[sp.Expr, ScalarPartials, Dict[str, sp.Expr]]:
    """
    Hansen (1994) Skew-t distribution scalar building blocks.
    
    Uses standardized residual z = e/√h and piecewise scaling.
    
    Key intermediates:
        u = b*z + a
        s = (1-λ) if z < -a/b else (1+λ)
        D = (ν-2)*s² + u²
    
    Log density part:
        log g(z|ν,λ) = log(b) + log(c) - ½(ν+1) log(D/(ν-2))
    
    Returns
    -------
    ell : symbolic log-likelihood  
    partials : ScalarPartials
    intermediates : dict of intermediate expressions for code generation
    """
    z, nu, lam = symbols('z nu lam', real=True)
    
    # Hansen constants
    c_log = sp.loggamma((nu + 1) / 2) - sp.loggamma(nu / 2) - sp.Rational(1, 2) * log(pi * (nu - 2))
    c_exp = exp(c_log)
    a = 4 * lam * c_exp * (nu - 2) / (nu - 1)
    b = sqrt(1 + 3 * lam**2 - a**2)
    
    # Key intermediates
    u = b * z + a
    
    # For symbolic analysis, use Piecewise for s
    # s = 1 - λ*sign(u) = (1-λ) if u < 0 else (1+λ)
    s = Piecewise((1 - lam, u < 0), (1 + lam, True))
    
    # For cleaner derivatives, we'll work with the "positive side" (u >= 0)
    # and note that the structure is symmetric
    s_pos = 1 + lam
    s_neg = 1 - lam
    
    # D = (ν-2)*s² + u²
    D_pos = (nu - 2) * s_pos**2 + u**2
    D_neg = (nu - 2) * s_neg**2 + u**2
    
    # Log-likelihood (positive side, typical case)
    # ℓ = log(b) + c_log - ½(ν+1) log(D/(ν-2))
    #   = log(b) + c_log - ½(ν+1) log(D) + ½(ν+1) log(ν-2)
    ell_pos = log(b) + c_log - sp.Rational(1, 2) * (nu + 1) * log(D_pos / (nu - 2))
    
    # =====================================
    # Derivatives wrt z (the key partials)
    # =====================================
    
    # For u >= 0 (positive side):
    # ∂ℓ/∂z = -(ν+1) * b * u / D
    ell_z_pos = diff(ell_pos, z)
    
    # ∂²ℓ/∂z² = -(ν+1) * b² * ((ν-2)*s² - u²) / D²
    ell_zz_pos = diff(ell_z_pos, z)
    
    # =====================================
    # Chain rule: z = e/√h
    # =====================================
    # ∂z/∂e = 1/√h
    # ∂z/∂h = -e/(2h^(3/2)) = -z/(2h)
    # ∂²z/∂e² = 0
    # ∂²z/∂h² = 3e/(4h^(5/2)) = 3z/(4h²)
    # ∂²z/∂e∂h = -1/(2h^(3/2))
    
    e, h = symbols('e h', real=True, positive=True)
    z_expr = e / sqrt(h)
    
    dz_de = diff(z_expr, e)   # 1/√h
    dz_dh = diff(z_expr, h)   # -e/(2h^(3/2))
    d2z_de2 = diff(dz_de, e)  # 0
    d2z_dh2 = diff(dz_dh, h)  # 3e/(4h^(5/2))
    d2z_dedh = diff(dz_de, h) # -1/(2h^(3/2))
    
    # Full partials via chain rule:
    # ∂ℓ/∂e = (∂ℓ/∂z)(∂z/∂e)
    # ∂ℓ/∂h = (∂ℓ/∂z)(∂z/∂h) - ½/h  (the -½/h from the Jacobian -½log(h))
    
    # Substitute z_expr into the z-derivatives
    ell_z_sub = ell_z_pos.subs(z, z_expr)
    ell_zz_sub = ell_zz_pos.subs(z, z_expr)
    
    # Include the -½log(h) term from the Jacobian
    ell_h_scale = -sp.Rational(1, 2) / h
    
    ell_e = ell_z_sub * dz_de
    ell_h = ell_z_sub * dz_dh + ell_h_scale
    
    # Second derivatives (chain rule for second derivatives)
    ell_ee = ell_zz_sub * dz_de**2 + ell_z_sub * d2z_de2
    ell_hh = ell_zz_sub * dz_dh**2 + ell_z_sub * d2z_dh2 + sp.Rational(1, 2) / h**2
    ell_eh = ell_zz_sub * dz_de * dz_dh + ell_z_sub * d2z_dedh
    
    # Intermediates for code generation
    intermediates = {
        'a': a,
        'b': b,
        'c_log': c_log,
        'u': u,
        'D_pos': D_pos,
        'D_neg': D_neg,
        'dz_de': dz_de,
        'dz_dh': dz_dh,
    }
    
    return ell_pos, ScalarPartials(
        ell_e=simplify(ell_e),
        ell_h=simplify(ell_h),
        ell_ee=simplify(ell_ee),
        ell_hh=simplify(ell_hh),
        ell_eh=simplify(ell_eh),
        ell_z=simplify(ell_z_pos),
        ell_zz=simplify(ell_zz_pos),
    ), intermediates


# =============================================================================
# GARCH VARIANCE SENSITIVITIES
# =============================================================================

def garch_variance_sensitivities(p: int = 1, q: int = 1) -> Dict[str, sp.Expr]:
    """
    Symbolic GARCH(p,q) variance sensitivities.
    
    h_t = ω + Σᵢ αᵢ e²_{t-i} + Σⱼ βⱼ h_{t-j}
    
    First derivative recursion:
        ∂h_t/∂θ_k = 𝟙[θ_k=ω] 
                    + Σᵢ (𝟙[θ_k=αᵢ] r_{t-i} + αᵢ ∂r_{t-i}/∂θ_k)
                    + Σⱼ (𝟙[θ_k=βⱼ] h_{t-j} + βⱼ ∂h_{t-j}/∂θ_k)
    
    where r = e² and ∂r/∂θ = 2e ∂e/∂θ
    """
    omega = Symbol('omega', positive=True)
    alphas = [Symbol(f'alpha_{i}', positive=True) for i in range(1, q + 1)]
    betas = [Symbol(f'beta_{j}', positive=True) for j in range(1, p + 1)]
    
    # Lagged values (symbolic)
    e_lags = [Symbol(f'e_{{t-{i}}}', real=True) for i in range(1, q + 1)]
    h_lags = [Symbol(f'h_{{t-{j}}}', positive=True) for j in range(1, p + 1)]
    
    # Lagged sensitivities
    de_lags = [[Symbol(f'de_{{t-{i}}}/d{k}', real=True) 
                for k in ['omega'] + [f'alpha_{j}' for j in range(1, q+1)] + [f'beta_{j}' for j in range(1, p+1)]]
               for i in range(1, q + 1)]
    dh_lags = [[Symbol(f'dh_{{t-{j}}}/d{k}', real=True)
                for k in ['omega'] + [f'alpha_{j}' for j in range(1, q+1)] + [f'beta_{j}' for j in range(1, p+1)]]
               for j in range(1, p + 1)]
    
    # GARCH variance
    h_t = omega + sum(alphas[i] * e_lags[i]**2 for i in range(q)) + sum(betas[j] * h_lags[j] for j in range(p))
    
    return {
        'h_t': h_t,
        'omega': omega,
        'alphas': alphas,
        'betas': betas,
        'e_lags': e_lags,
        'h_lags': h_lags,
    }


def garch_11_sensitivities_symbolic():
    """
    Symbolic GARCH(1,1) variance sensitivities for verification.
    
    h_t = ω + α e²_{t-1} + β h_{t-1}
    
    Returns symbolic expressions for:
    - dh/dω, dh/dα, dh/dβ
    - d²h/dω², d²h/dα², d²h/dβ², d²h/dωdα, etc.
    """
    omega, alpha, beta = symbols('omega alpha beta', positive=True)
    e_lag, h_lag = symbols('e_lag h_lag', real=True, positive=True)
    
    # Lagged sensitivities (from previous step)
    dh_lag_domega = Symbol('dh_lag_domega', real=True)
    dh_lag_dalpha = Symbol('dh_lag_dalpha', real=True)
    dh_lag_dbeta = Symbol('dh_lag_dbeta', real=True)
    
    de_lag_domega = Symbol('de_lag_domega', real=True)
    de_lag_dalpha = Symbol('de_lag_dalpha', real=True)
    de_lag_dbeta = Symbol('de_lag_dbeta', real=True)
    
    # For pure GARCH (no mean model), de/dθ = 0 for GARCH params
    # But for AR-GARCH, these will be non-zero
    
    # GARCH variance
    r_lag = e_lag**2  # squared residual
    h_t = omega + alpha * r_lag + beta * h_lag
    
    # First sensitivities (recursion)
    # ∂h_t/∂ω = 1 + β ∂h_{t-1}/∂ω + α ∂r_{t-1}/∂ω
    # For pure GARCH: ∂r/∂ω = 2e ∂e/∂ω = 0
    dh_domega = 1 + beta * dh_lag_domega + alpha * 2 * e_lag * de_lag_domega
    dh_dalpha = r_lag + beta * dh_lag_dalpha + alpha * 2 * e_lag * de_lag_dalpha
    dh_dbeta = h_lag + beta * dh_lag_dbeta + alpha * 2 * e_lag * de_lag_dbeta
    
    return {
        'h_t': h_t,
        'dh_domega': dh_domega,
        'dh_dalpha': dh_dalpha,
        'dh_dbeta': dh_dbeta,
        'params': (omega, alpha, beta),
        'lags': (e_lag, h_lag),
        'lag_sensitivities': {
            'dh_lag': (dh_lag_domega, dh_lag_dalpha, dh_lag_dbeta),
            'de_lag': (de_lag_domega, de_lag_dalpha, de_lag_dbeta),
        }
    }


# =============================================================================
# AR MEAN SENSITIVITIES (for future AR-GARCH)
# =============================================================================

def ar_mean_sensitivities(P: int = 1) -> Dict[str, sp.Expr]:
    """
    Symbolic AR(P) mean sensitivities.
    
    μ_t = c + Σᵢ φᵢ y_{t-i}
    e_t = y_t - μ_t
    
    First sensitivities (linear AR → simple):
        ∂e_t/∂c = -1
        ∂e_t/∂φᵢ = -y_{t-i}
        ∂e_t/∂(GARCH params) = 0
    
    Second sensitivities:
        ∂²e_t/∂θ∂θ' = 0  (linear mean!)
    
    This is the key simplification: d²e = 0 for linear AR.
    """
    c = Symbol('c', real=True)
    phis = [Symbol(f'phi_{i}', real=True) for i in range(1, P + 1)]
    y_lags = [Symbol(f'y_{{t-{i}}}', real=True) for i in range(1, P + 1)]
    y_t = Symbol('y_t', real=True)
    
    # Mean
    mu_t = c + sum(phis[i] * y_lags[i] for i in range(P))
    
    # Residual
    e_t = y_t - mu_t
    
    # First sensitivities
    de_dc = diff(e_t, c)  # -1
    de_dphi = [diff(e_t, phis[i]) for i in range(P)]  # -y_{t-i}
    
    # Second sensitivities (all zero for linear AR!)
    d2e = {
        'd2e_dc2': diff(de_dc, c),
        'd2e_dcphi': [diff(de_dc, phis[i]) for i in range(P)],
        'd2e_dphi2': [[diff(de_dphi[i], phis[j]) for j in range(P)] for i in range(P)],
    }
    
    return {
        'mu_t': mu_t,
        'e_t': e_t,
        'de_dc': de_dc,
        'de_dphi': de_dphi,
        'd2e': d2e,  # All zeros!
        'params': (c, phis),
    }


# =============================================================================
# CODE GENERATION WITH CSE
# =============================================================================

def generate_c_code(partials: ScalarPartials, dist_name: str) -> str:
    """
    Generate C code for scalar partials using CSE.
    
    This produces compact, readable C code without expression blowup.
    """
    from sympy.printing.c import C99CodePrinter
    
    expressions = [
        ('ell_e', partials.ell_e),
        ('ell_h', partials.ell_h),
        ('ell_ee', partials.ell_ee),
        ('ell_hh', partials.ell_hh),
        ('ell_eh', partials.ell_eh),
    ]
    
    if partials.ell_z is not None:
        expressions.extend([
            ('ell_z', partials.ell_z),
            ('ell_zz', partials.ell_zz),
        ])
    
    # Apply CSE
    replacements, reduced = cse([expr for _, expr in expressions])
    
    printer = C99CodePrinter()
    
    lines = [f"// {dist_name} scalar partials (generated by symbolic_derivatives.py)"]
    lines.append("")
    
    # CSE temporaries
    for var, expr in replacements:
        lines.append(f"const double {var} = {printer.doprint(expr)};")
    
    lines.append("")
    
    # Final expressions
    for (name, _), reduced_expr in zip(expressions, reduced):
        lines.append(f"const double {name} = {printer.doprint(reduced_expr)};")
    
    return "\n".join(lines)


def generate_numpy_lambdas(partials: ScalarPartials) -> Dict[str, Callable]:
    """
    Generate NumPy-compatible lambda functions for numerical evaluation.
    """
    e, h = symbols('e h', real=True, positive=True)
    
    return {
        'ell_e': lambdify((e, h), partials.ell_e, 'numpy'),
        'ell_h': lambdify((e, h), partials.ell_h, 'numpy'),
        'ell_ee': lambdify((e, h), partials.ell_ee, 'numpy'),
        'ell_hh': lambdify((e, h), partials.ell_hh, 'numpy'),
        'ell_eh': lambdify((e, h), partials.ell_eh, 'numpy'),
    }


# =============================================================================
# NUMERICAL VERIFICATION
# =============================================================================

def verify_partials_numerically(
    ell_func: Callable,
    partials_funcs: Dict[str, Callable],
    e_val: float = 0.5,
    h_val: float = 1.0,
    eps: float = 1e-7,
) -> Dict[str, Tuple[float, float, float]]:
    """
    Verify symbolic partials against finite differences.
    
    Returns dict of (analytical, numerical, relative_error) for each partial.
    """
    results = {}
    
    # First derivatives
    ell_0 = ell_func(e_val, h_val)
    
    # ∂ℓ/∂e via finite diff
    ell_e_num = (ell_func(e_val + eps, h_val) - ell_func(e_val - eps, h_val)) / (2 * eps)
    ell_e_ana = partials_funcs['ell_e'](e_val, h_val)
    results['ell_e'] = (ell_e_ana, ell_e_num, abs(ell_e_ana - ell_e_num) / (abs(ell_e_ana) + 1e-10))
    
    # ∂ℓ/∂h via finite diff
    ell_h_num = (ell_func(e_val, h_val + eps) - ell_func(e_val, h_val - eps)) / (2 * eps)
    ell_h_ana = partials_funcs['ell_h'](e_val, h_val)
    results['ell_h'] = (ell_h_ana, ell_h_num, abs(ell_h_ana - ell_h_num) / (abs(ell_h_ana) + 1e-10))
    
    # Second derivatives
    eps2 = 1e-5
    
    # ∂²ℓ/∂e²
    ell_ee_num = (ell_func(e_val + eps2, h_val) - 2*ell_0 + ell_func(e_val - eps2, h_val)) / eps2**2
    ell_ee_ana = partials_funcs['ell_ee'](e_val, h_val)
    results['ell_ee'] = (ell_ee_ana, ell_ee_num, abs(ell_ee_ana - ell_ee_num) / (abs(ell_ee_ana) + 1e-10))
    
    # ∂²ℓ/∂h²
    ell_hh_num = (ell_func(e_val, h_val + eps2) - 2*ell_0 + ell_func(e_val, h_val - eps2)) / eps2**2
    ell_hh_ana = partials_funcs['ell_hh'](e_val, h_val)
    results['ell_hh'] = (ell_hh_ana, ell_hh_num, abs(ell_hh_ana - ell_hh_num) / (abs(ell_hh_ana) + 1e-10))
    
    # ∂²ℓ/∂e∂h (mixed)
    ell_eh_num = (ell_func(e_val + eps2, h_val + eps2) 
                  - ell_func(e_val + eps2, h_val - eps2)
                  - ell_func(e_val - eps2, h_val + eps2)
                  + ell_func(e_val - eps2, h_val - eps2)) / (4 * eps2**2)
    ell_eh_ana = partials_funcs['ell_eh'](e_val, h_val)
    results['ell_eh'] = (ell_eh_ana, ell_eh_num, abs(ell_eh_ana - ell_eh_num) / (abs(ell_eh_ana) + 1e-10))
    
    return results


# =============================================================================
# GRADIENT/HESSIAN ASSEMBLY FORMULAS
# =============================================================================

def print_assembly_formulas():
    """
    Print the outer-product assembly formulas for gradient and Hessian.
    
    This is the core "kernel writing" pattern.
    """
    print("""
GRADIENT/HESSIAN ASSEMBLY FORMULAS
==================================

For any distribution, the gradient and Hessian are assembled from:
- Scalar partials: ℓ_e, ℓ_h, ℓ_ee, ℓ_hh, ℓ_eh
- State sensitivities: de (K×1), dh (K×1), d²e (K×K), d²h (K×K)

GRADIENT (K×1):
    ∇_θ ℓ_t = ℓ_e · de + ℓ_h · dh

HESSIAN (K×K):
    ∇²_θ ℓ_t = ℓ_ee · (de)(de)ᵀ
             + ℓ_hh · (dh)(dh)ᵀ
             + ℓ_eh · [(de)(dh)ᵀ + (dh)(de)ᵀ]
             + ℓ_e · d²e
             + ℓ_h · d²h

For LINEAR AR mean: d²e = 0, so:
    ∇²_θ ℓ_t = ℓ_ee · (de)(de)ᵀ
             + ℓ_hh · (dh)(dh)ᵀ
             + ℓ_eh · [(de)(dh)ᵀ + (dh)(de)ᵀ]
             + ℓ_h · d²h

This is compact and numerically stable!

For SKEW-T, use z = e/√h and:
    ∇_θ ℓ_t = ℓ_z · dz + ℓ_h^(scale) · dh + (ν,λ terms)
    
where:
    dz = (1/√h) · de - (z/2h) · dh
    """)


# =============================================================================
# MAIN: RUN VERIFICATION
# =============================================================================

def run_normal_verification():
    """Quick verification of Normal distribution partials."""
    print("NORMAL DISTRIBUTION")
    print("-" * 50)
    
    ell, partials = normal_scalar_partials()
    print(f"  ℓ = {ell}")
    print(f"  ∂ℓ/∂e = {partials.ell_e}")
    print(f"  ∂ℓ/∂h = {partials.ell_h}")
    print(f"  ∂²ℓ/∂e² = {partials.ell_ee}")
    print(f"  ∂²ℓ/∂h² = {partials.ell_hh}")
    print(f"  ∂²ℓ/∂e∂h = {partials.ell_eh}")
    
    # Numerical verification
    e, h = symbols('e h', real=True, positive=True)
    ell_func = lambdify((e, h), ell, 'numpy')
    partials_funcs = generate_numpy_lambdas(partials)
    
    print("\nNumerical verification:")
    results = verify_partials_numerically(ell_func, partials_funcs)
    for name, (ana, num, err) in results.items():
        status = "✓" if err < 1e-5 else "✗"
        print(f"  {name}: ana={ana:+.6f}, num={num:+.6f}, err={err:.2e} {status}")
    
    return ell, partials


def run_garch_verification():
    """Verify GARCH sensitivity formulas."""
    print("\nGARCH(1,1) VARIANCE SENSITIVITIES")
    print("-" * 50)
    
    sens = garch_11_sensitivities_symbolic()
    print(f"  h_t = {sens['h_t']}")
    print(f"  ∂h_t/∂ω = {sens['dh_domega']}")
    print(f"  ∂h_t/∂α = {sens['dh_dalpha']}")
    print(f"  ∂h_t/∂β = {sens['dh_dbeta']}")
    
    return sens


def run_ar_verification():
    """Verify AR sensitivity formulas."""
    print("\nAR(1) MEAN SENSITIVITIES")
    print("-" * 50)
    
    sens = ar_mean_sensitivities(P=1)
    print(f"  μ_t = {sens['mu_t']}")
    print(f"  e_t = {sens['e_t']}")
    print(f"  ∂e_t/∂c = {sens['de_dc']}")
    print(f"  ∂e_t/∂φ₁ = {sens['de_dphi'][0]}")
    print(f"  ∂²e_t/∂c² = {sens['d2e']['d2e_dc2']} (zero for linear AR!)")
    
    return sens


if __name__ == "__main__":
    import sys
    
    print("=" * 60)
    print("SYMBOLIC DERIVATIVE VERIFICATION")
    print("=" * 60)
    print()
    
    # Quick mode by default
    if len(sys.argv) > 1 and sys.argv[1] == '--full':
        # Full verification (slow - includes Student-t, Skew-t)
        print("Running FULL verification (this may take a while)...")
        print()
        
        run_normal_verification()
        
        print("\n" + "=" * 50)
        print("STUDENT-T (computing...)")
        ell_t, partials_t = studentt_scalar_partials()
        print(f"  ∂ℓ/∂e = {partials_t.ell_e}")
        
        print("\n" + "=" * 50)
        print("HANSEN SKEW-T (computing...)")
        ell_s, partials_s, inter = hansen_skewt_scalar_partials()
        print(f"  intermediates: a, b computed")
        
        run_garch_verification()
        run_ar_verification()
        
    else:
        # Quick verification (just Normal + GARCH + AR)
        run_normal_verification()
        run_garch_verification()
        run_ar_verification()
    
    print()
    print_assembly_formulas()
    
    print("\n" + "=" * 60)
    print("VERIFICATION COMPLETE")
    print("=" * 60)
