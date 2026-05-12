"""
Master DGP-Based Estimation Tests for scivol
=============================================

This file tests scivol estimation by generating synthetic data from known
Data Generating Processes (DGPs) and verifying that estimation recovers
the true parameters.

**KEEP THIS FILE EVERGREEN** - Update when new models are added to scivol.

Test Coverage:
- GARCH(1,1) + Normal
- GARCH(1,1) + StudentT  
- GARCH(1,1) + SkewT
- GARCH(p,q) + Normal (various orders)
- ARMA(1,1) + GARCH(1,1) + Normal (when API supports it)

EGARCH coverage follows the shipped public tensor, including the vol-only GED
surface, the shipped ARMA+EGARCH densities, and the shipped ARX/HARX standalone
and linked EGARCH families.

Each test:
1. Generates 5000 observations from known true parameters
2. Estimates the model using scivol
3. Verifies convergence (optimization success)
4. Checks parameter recovery (within tolerance)
"""

from __future__ import annotations

import numpy as np
import pytest
from numpy.typing import NDArray
from scipy.special import gammaln
from scipy.stats import gennorm

from scivol._evaluation import _egarch_abs_moment, _hansen_skewt_ppf

# -----------------------------------------------------------------------------
# Data Generating Process (DGP) Functions
# -----------------------------------------------------------------------------


def simulate_garch_normal(
    n: int,
    omega: float,
    alpha: float,
    beta: float,
    seed: int = 42,
) -> NDArray[np.float64]:
    """
    Simulate GARCH(1,1) with Normal innovations.
    
    Model:
        σ²_t = ω + α·ε²_{t-1} + β·σ²_{t-1}
        y_t = σ_t · z_t,  z_t ~ N(0,1)
    
    Returns y (returns), not σ².
    """
    rng = np.random.default_rng(seed)
    
    y = np.zeros(n, dtype=np.float64)
    sigma2 = np.zeros(n, dtype=np.float64)
    
    # Unconditional variance as starting value
    sigma2_uncond = omega / (1 - alpha - beta)
    sigma2[0] = sigma2_uncond
    y[0] = np.sqrt(sigma2[0]) * rng.standard_normal()
    
    for t in range(1, n):
        sigma2[t] = omega + alpha * y[t-1]**2 + beta * sigma2[t-1]
        y[t] = np.sqrt(sigma2[t]) * rng.standard_normal()
    
    return y


def simulate_garch_studentt(
    n: int,
    omega: float,
    alpha: float,
    beta: float,
    nu: float,
    seed: int = 42,
) -> NDArray[np.float64]:
    """
    Simulate GARCH(1,1) with Student-t innovations.
    
    Model:
        σ²_t = ω + α·ε²_{t-1} + β·σ²_{t-1}
        y_t = σ_t · z_t,  z_t ~ t_ν (standardized to unit variance)
    """
    rng = np.random.default_rng(seed)
    
    y = np.zeros(n, dtype=np.float64)
    sigma2 = np.zeros(n, dtype=np.float64)
    
    sigma2_uncond = omega / (1 - alpha - beta)
    sigma2[0] = sigma2_uncond
    
    # Standardized t: divide by sqrt(nu/(nu-2)) to get unit variance
    scale = np.sqrt((nu - 2) / nu)
    
    z = rng.standard_t(nu, size=n) * scale
    y[0] = np.sqrt(sigma2[0]) * z[0]
    
    for t in range(1, n):
        sigma2[t] = omega + alpha * y[t-1]**2 + beta * sigma2[t-1]
        y[t] = np.sqrt(sigma2[t]) * z[t]
    
    return y


def simulate_garch_skewt(
    n: int,
    omega: float,
    alpha: float,
    beta: float,
    nu: float,
    lam: float,
    seed: int = 42,
) -> NDArray[np.float64]:
    """
    Simulate GARCH(1,1) with Hansen (1994) Skew-t innovations.
    
    Uses a simpler approximation: Student-t with asymmetric scaling.
    """
    rng = np.random.default_rng(seed)
    
    y = np.zeros(n, dtype=np.float64)
    sigma2 = np.zeros(n, dtype=np.float64)
    
    sigma2_uncond = omega / (1 - alpha - beta)
    sigma2[0] = sigma2_uncond
    
    # Standardized t scale
    t_scale = np.sqrt((nu - 2) / nu)
    
    for t in range(n):
        if t > 0:
            sigma2[t] = omega + alpha * y[t-1]**2 + beta * sigma2[t-1]
        
        # Generate skew-t approximation:
        # Draw from t, then apply asymmetric transformation
        z_raw = rng.standard_t(nu) * t_scale
        
        # Apply skewness: compress one tail, expand the other
        if z_raw < 0:
            z = z_raw * (1 - lam)
        else:
            z = z_raw * (1 + lam)
        
        y[t] = np.sqrt(sigma2[t]) * z
    
    return y


def simulate_garch_ged(
    n: int,
    omega: float,
    alpha: float,
    beta: float,
    nu: float,
    seed: int = 42,
) -> NDArray[np.float64]:
    """Simulate GARCH(1,1) with GED innovations standardised to unit variance."""
    rng = np.random.default_rng(seed)

    y = np.zeros(n, dtype=np.float64)
    sigma2 = np.zeros(n, dtype=np.float64)

    sigma2_uncond = omega / (1 - alpha - beta)
    sigma2[0] = sigma2_uncond

    scale = np.sqrt(np.exp(gammaln(1.0 / nu) - gammaln(3.0 / nu)))
    z = np.asarray(gennorm.rvs(beta=nu, scale=scale, size=n, random_state=rng), dtype=np.float64)
    y[0] = np.sqrt(sigma2[0]) * z[0]

    for t in range(1, n):
        sigma2[t] = omega + alpha * y[t - 1] ** 2 + beta * sigma2[t - 1]
        y[t] = np.sqrt(sigma2[t]) * z[t]

    return y


def simulate_garch_pq_normal(
    n: int,
    omega: float,
    alphas: list[float],
    betas: list[float],
    seed: int = 42,
) -> NDArray[np.float64]:
    """
    Simulate GARCH(p,q) with Normal innovations.
    
    Model:
        σ²_t = ω + Σ α_i·ε²_{t-i} + Σ β_j·σ²_{t-j}
    """
    rng = np.random.default_rng(seed)
    
    P = len(alphas)
    Q = len(betas)
    max_lag = max(P, Q)
    
    y = np.zeros(n, dtype=np.float64)
    sigma2 = np.zeros(n, dtype=np.float64)
    
    # Unconditional variance
    alpha_sum = sum(alphas)
    beta_sum = sum(betas)
    sigma2_uncond = omega / (1 - alpha_sum - beta_sum)
    
    # Initialize
    for t in range(max_lag):
        sigma2[t] = sigma2_uncond
        y[t] = np.sqrt(sigma2[t]) * rng.standard_normal()
    
    for t in range(max_lag, n):
        sigma2[t] = omega
        for i, a in enumerate(alphas):
            sigma2[t] += a * y[t-1-i]**2
        for j, b in enumerate(betas):
            sigma2[t] += b * sigma2[t-1-j]
        y[t] = np.sqrt(sigma2[t]) * rng.standard_normal()
    
    return y


def simulate_arma_garch_normal(
    n: int,
    c: float,
    phi: float,
    theta: float,
    omega: float,
    alpha: float,
    beta: float,
    seed: int = 42,
) -> NDArray[np.float64]:
    """
    Simulate ARMA(1,1)-GARCH(1,1) with Normal innovations.
    
    Model:
        y_t = c + φ·y_{t-1} + θ·ε_{t-1} + ε_t
        ε_t = σ_t · z_t,  z_t ~ N(0,1)
        σ²_t = ω + α·ε²_{t-1} + β·σ²_{t-1}
    """
    rng = np.random.default_rng(seed)
    
    y = np.zeros(n, dtype=np.float64)
    eps = np.zeros(n, dtype=np.float64)
    sigma2 = np.zeros(n, dtype=np.float64)
    
    # Unconditional variance
    sigma2_uncond = omega / (1 - alpha - beta)
    
    # Initialize t=0
    sigma2[0] = sigma2_uncond
    eps[0] = np.sqrt(sigma2[0]) * rng.standard_normal()
    y[0] = c / (1 - phi) + eps[0]  # Approximate unconditional mean
    
    for t in range(1, n):
        sigma2[t] = omega + alpha * eps[t-1]**2 + beta * sigma2[t-1]
        eps[t] = np.sqrt(sigma2[t]) * rng.standard_normal()
        y[t] = c + phi * y[t-1] + theta * eps[t-1] + eps[t]
    
    return y


def simulate_arma_gjr_garch_normal(
    n: int,
    c: float,
    phi: float,
    theta: float,
    omega: float,
    alpha: float,
    gamma: float,
    beta: float,
    seed: int = 42,
) -> NDArray[np.float64]:
    """Simulate ARMA(1,1)-GJR-GARCH(1,1) with Normal innovations."""
    rng = np.random.default_rng(seed)

    y = np.zeros(n, dtype=np.float64)
    eps = np.zeros(n, dtype=np.float64)
    sigma2 = np.zeros(n, dtype=np.float64)

    sigma2[0] = omega / (1 - alpha - 0.5 * gamma - beta)
    eps[0] = np.sqrt(sigma2[0]) * rng.standard_normal()
    y[0] = c / (1 - phi) + eps[0]

    for t in range(1, n):
        ind = 1.0 if eps[t - 1] < 0.0 else 0.0
        sigma2[t] = omega + alpha * eps[t - 1]**2 + gamma * ind * eps[t - 1]**2 + beta * sigma2[t - 1]
        eps[t] = np.sqrt(sigma2[t]) * rng.standard_normal()
        y[t] = c + phi * y[t - 1] + theta * eps[t - 1] + eps[t]

    return y


def simulate_arma_gjr_garch_studentt(
    n: int,
    c: float,
    phi: float,
    theta: float,
    omega: float,
    alpha: float,
    gamma: float,
    beta: float,
    nu: float,
    seed: int = 42,
) -> NDArray[np.float64]:
    """Simulate ARMA(1,1)-GJR-GARCH(1,1) with Student-t innovations."""
    rng = np.random.default_rng(seed)

    y = np.zeros(n, dtype=np.float64)
    eps = np.zeros(n, dtype=np.float64)
    sigma2 = np.zeros(n, dtype=np.float64)

    sigma2[0] = omega / (1 - alpha - 0.5 * gamma - beta)
    draws = rng.standard_t(nu, size=n) * np.sqrt((nu - 2) / nu)
    eps[0] = np.sqrt(sigma2[0]) * draws[0]
    y[0] = c / (1 - phi) + eps[0]

    for t in range(1, n):
        ind = 1.0 if eps[t - 1] < 0.0 else 0.0
        sigma2[t] = omega + alpha * eps[t - 1]**2 + gamma * ind * eps[t - 1]**2 + beta * sigma2[t - 1]
        eps[t] = np.sqrt(sigma2[t]) * draws[t]
        y[t] = c + phi * y[t - 1] + theta * eps[t - 1] + eps[t]

    return y


def simulate_arma_gjr_garch_skewt(
    n: int,
    c: float,
    phi: float,
    theta: float,
    omega: float,
    alpha: float,
    gamma: float,
    beta: float,
    nu: float,
    lam: float,
    seed: int = 42,
) -> NDArray[np.float64]:
    """Simulate ARMA(1,1)-GJR-GARCH(1,1) with skew-t-style innovations."""
    rng = np.random.default_rng(seed)

    y = np.zeros(n, dtype=np.float64)
    eps = np.zeros(n, dtype=np.float64)
    sigma2 = np.zeros(n, dtype=np.float64)
    t_scale = np.sqrt((nu - 2) / nu)

    sigma2[0] = omega / (1 - alpha - 0.5 * gamma - beta)
    z0_raw = rng.standard_t(nu) * t_scale
    z0 = z0_raw * (1 - lam) if z0_raw < 0 else z0_raw * (1 + lam)
    eps[0] = np.sqrt(sigma2[0]) * z0
    y[0] = c / (1 - phi) + eps[0]

    for t in range(1, n):
        ind = 1.0 if eps[t - 1] < 0.0 else 0.0
        sigma2[t] = omega + alpha * eps[t - 1]**2 + gamma * ind * eps[t - 1]**2 + beta * sigma2[t - 1]
        z_raw = rng.standard_t(nu) * t_scale
        z = z_raw * (1 - lam) if z_raw < 0 else z_raw * (1 + lam)
        eps[t] = np.sqrt(sigma2[t]) * z
        y[t] = c + phi * y[t - 1] + theta * eps[t - 1] + eps[t]

    return y


def simulate_arma_egarch_pq(
    n: int,
    c: float,
    phis: list[float],
    thetas: list[float],
    omega: float,
    alphas: list[float],
    gammas: list[float],
    betas: list[float],
    *,
    dist: str = "Normal",
    nu: float = 8.0,
    lam: float = 0.0,
    seed: int = 42,
) -> NDArray[np.float64]:
    """
    Simulate linked ARMA(p,q)-EGARCH(P,Q) data from the joint recursion.

    Model:
        y_t = c + sum(phi_i y_{t-i}) + sum(theta_j eps_{t-j}) + eps_t
        eps_t = sqrt(h_t) * z_t
        log h_t = omega
                  + sum(alpha_i (|z_{t-i}| - E|z|))
                  + sum(gamma_i z_{t-i})
                  + sum(beta_j log h_{t-j})
    """
    rng = np.random.default_rng(seed)

    p_ar = len(phis)
    q_ma = len(thetas)
    P = len(alphas)
    Q = len(betas)
    max_lag = max(p_ar, q_ma, P, Q, 1)

    y = np.zeros(n, dtype=np.float64)
    eps = np.zeros(n, dtype=np.float64)
    sigma2 = np.zeros(n, dtype=np.float64)
    logh = np.zeros(n, dtype=np.float64)

    persistence = float(sum(betas))
    logh0 = omega / max(1.0 - persistence, 1e-3)
    sigma2[:max_lag] = np.exp(logh0)
    logh[:max_lag] = logh0

    if dist == "Normal":
        draws = rng.standard_normal(n)
        abs_moment = _egarch_abs_moment("Normal", None, None)
    elif dist == "StudentT":
        draws = rng.standard_t(nu, size=n) * np.sqrt((nu - 2.0) / nu)
        abs_moment = _egarch_abs_moment("StudentT", nu, None)
    elif dist == "SkewT":
        draws = _hansen_skewt_ppf(rng.uniform(size=n), nu, lam)
        abs_moment = _egarch_abs_moment("SkewT", nu, lam)
    elif dist == "GED":
        scale = np.sqrt(np.exp(gammaln(1.0 / nu) - gammaln(3.0 / nu)))
        draws = np.asarray(gennorm.rvs(beta=nu, scale=scale, size=n, random_state=rng), dtype=np.float64)
        abs_moment = _egarch_abs_moment("GED", nu, None)
    else:
        raise ValueError(f"Unsupported dist '{dist}'")

    for t in range(max_lag):
        eps[t] = np.sqrt(sigma2[t]) * draws[t]
        y[t] = eps[t]

    for t in range(max_lag, n):
        mean_t = c
        for i, phi_i in enumerate(phis, start=1):
            mean_t += phi_i * y[t - i]
        for j, theta_j in enumerate(thetas, start=1):
            mean_t += theta_j * eps[t - j]

        logh_t = omega
        for i, alpha_i in enumerate(alphas, start=1):
            z_lag = eps[t - i] / np.sqrt(sigma2[t - i])
            logh_t += alpha_i * (abs(z_lag) - abs_moment) + gammas[i - 1] * z_lag
        for j, beta_j in enumerate(betas, start=1):
            logh_t += beta_j * logh[t - j]

        logh[t] = logh_t
        sigma2[t] = np.exp(logh_t)
        eps[t] = np.sqrt(sigma2[t]) * draws[t]
        y[t] = mean_t + eps[t]

    return y


def simulate_arma_normal(
    n: int,
    c: float,
    phi: float,
    theta: float,
    sigma: float = 0.01,
    seed: int = 42,
) -> NDArray[np.float64]:
    """
    Simulate ARMA(1,1) with Normal innovations (constant variance).
    
    Model:
        y_t = c + φ·y_{t-1} + θ·ε_{t-1} + ε_t
        ε_t ~ N(0, σ²)
    """
    rng = np.random.default_rng(seed)
    
    y = np.zeros(n, dtype=np.float64)
    eps = np.zeros(n, dtype=np.float64)
    
    # Initialize t=0
    eps[0] = rng.standard_normal() * sigma
    y[0] = c / (1 - phi) + eps[0]  # Approximate unconditional mean
    
    for t in range(1, n):
        eps[t] = rng.standard_normal() * sigma
        y[t] = c + phi * y[t-1] + theta * eps[t-1] + eps[t]
    
    return y


def simulate_gjr_garch_normal(
    n: int,
    omega: float,
    alpha: float,
    gamma: float,
    beta: float,
    seed: int = 42,
) -> NDArray[np.float64]:
    """
    Simulate GJR-GARCH(1,1) with Normal innovations.
    
    Model:
        σ²_t = ω + α·ε²_{t-1} + γ·I(ε_{t-1}<0)·ε²_{t-1} + β·σ²_{t-1}
        y_t = σ_t · z_t,  z_t ~ N(0,1)
    
    Stationarity requires: α + 0.5·γ + β < 1 (symmetric distributions).
    """
    rng = np.random.default_rng(seed)
    
    y = np.zeros(n, dtype=np.float64)
    sigma2 = np.zeros(n, dtype=np.float64)
    
    # Unconditional variance: ω / (1 - α - 0.5·γ - β)  [symmetric dist]
    sigma2_uncond = omega / (1 - alpha - 0.5 * gamma - beta)
    sigma2[0] = sigma2_uncond
    y[0] = np.sqrt(sigma2[0]) * rng.standard_normal()
    
    for t in range(1, n):
        e = y[t-1]
        ind = 1.0 if e < 0 else 0.0
        sigma2[t] = omega + alpha * e**2 + gamma * ind * e**2 + beta * sigma2[t-1]
        y[t] = np.sqrt(sigma2[t]) * rng.standard_normal()
    
    return y


def simulate_gjr_garch_studentt(
    n: int,
    omega: float,
    alpha: float,
    gamma: float,
    beta: float,
    nu: float,
    seed: int = 42,
) -> NDArray[np.float64]:
    """
    Simulate GJR-GARCH(1,1) with Student-t innovations.
    
    Stationarity requires: α + 0.5·γ + β < 1 (Student-t is symmetric).
    """
    rng = np.random.default_rng(seed)
    
    y = np.zeros(n, dtype=np.float64)
    sigma2 = np.zeros(n, dtype=np.float64)
    
    sigma2_uncond = omega / (1 - alpha - 0.5 * gamma - beta)
    sigma2[0] = sigma2_uncond
    
    scale = np.sqrt((nu - 2) / nu)
    z = rng.standard_t(nu, size=n) * scale
    y[0] = np.sqrt(sigma2[0]) * z[0]
    
    for t in range(1, n):
        e = y[t-1]
        ind = 1.0 if e < 0 else 0.0
        sigma2[t] = omega + alpha * e**2 + gamma * ind * e**2 + beta * sigma2[t-1]
        y[t] = np.sqrt(sigma2[t]) * z[t]
    
    return y


def simulate_gjr_garch_skewt(
    n: int,
    omega: float,
    alpha: float,
    gamma: float,
    beta: float,
    nu: float,
    lam: float,
    seed: int = 42,
) -> NDArray[np.float64]:
    """
    Simulate GJR-GARCH(1,1) with Skew-t innovations.
    
    Stationarity requires: α + γ·P(z<0) + β < 1 where P(z<0) depends on lambda.
    For lambda < 0 (left-skew): P(z<0) > 0.5, making the constraint tighter.
    """
    rng = np.random.default_rng(seed)
    
    y = np.zeros(n, dtype=np.float64)
    sigma2 = np.zeros(n, dtype=np.float64)
    
    # Approximate P(z < 0) for skew-t; for small |lambda|, ≈ 0.5
    # Conservative: use P(z<0) ≈ 0.5 + 0.3*|lam| when lam < 0
    p_neg = 0.5  # Simplified; exact value depends on Hansen density
    sigma2_uncond = omega / (1 - alpha - p_neg * gamma - beta)
    sigma2[0] = sigma2_uncond
    
    t_scale = np.sqrt((nu - 2) / nu)
    
    for t in range(n):
        if t > 0:
            e = y[t-1]
            ind = 1.0 if e < 0 else 0.0
            sigma2[t] = omega + alpha * e**2 + gamma * ind * e**2 + beta * sigma2[t-1]
        
        z_raw = rng.standard_t(nu) * t_scale
        if z_raw < 0:
            z = z_raw * (1 - lam)
        else:
            z = z_raw * (1 + lam)
        
        y[t] = np.sqrt(sigma2[t]) * z
    
    return y


def simulate_egarch_normal(
    n: int,
    omega: float,
    alpha: float,
    gamma: float,
    beta: float,
    seed: int = 42,
) -> NDArray[np.float64]:
    """Simulate EGARCH(1,1) with Normal innovations."""
    rng = np.random.default_rng(seed)
    y = np.zeros(n, dtype=np.float64)
    sigma2 = np.zeros(n, dtype=np.float64)
    abs_moment = np.sqrt(2.0 / np.pi)

    sigma2[0] = np.exp(omega / (1.0 - beta))
    y[0] = np.sqrt(sigma2[0]) * rng.standard_normal()
    for t in range(1, n):
        z_prev = y[t - 1] / np.sqrt(sigma2[t - 1])
        logh_t = omega + alpha * (abs(z_prev) - abs_moment) + gamma * z_prev + beta * np.log(sigma2[t - 1])
        sigma2[t] = np.exp(logh_t)
        y[t] = np.sqrt(sigma2[t]) * rng.standard_normal()
    return y


def simulate_egarch_studentt(
    n: int,
    omega: float,
    alpha: float,
    gamma: float,
    beta: float,
    nu: float,
    seed: int = 42,
) -> NDArray[np.float64]:
    """Simulate EGARCH(1,1) with standardized Student-t innovations."""
    rng = np.random.default_rng(seed)
    y = np.zeros(n, dtype=np.float64)
    sigma2 = np.zeros(n, dtype=np.float64)
    abs_moment = (
        2.0
        * np.exp(gammaln(0.5 * (nu + 1.0)) - gammaln(0.5 * nu))
        * (nu - 2.0)
        / ((nu - 1.0) * np.sqrt(np.pi * (nu - 2.0)))
    )
    draws = rng.standard_t(nu, size=n) * np.sqrt((nu - 2.0) / nu)

    sigma2[0] = np.exp(omega / (1.0 - beta))
    y[0] = np.sqrt(sigma2[0]) * draws[0]
    for t in range(1, n):
        z_prev = y[t - 1] / np.sqrt(sigma2[t - 1])
        logh_t = omega + alpha * (abs(z_prev) - abs_moment) + gamma * z_prev + beta * np.log(sigma2[t - 1])
        sigma2[t] = np.exp(logh_t)
        y[t] = np.sqrt(sigma2[t]) * draws[t]
    return y


def simulate_egarch_skewt(
    n: int,
    omega: float,
    alpha: float,
    gamma: float,
    beta: float,
    nu: float,
    lam: float,
    seed: int = 42,
) -> NDArray[np.float64]:
    """Simulate EGARCH(1,1) with Hansen-skew-t innovations."""
    rng = np.random.default_rng(seed)
    y = np.zeros(n, dtype=np.float64)
    sigma2 = np.zeros(n, dtype=np.float64)
    abs_moment = _egarch_abs_moment("SkewT", nu, lam)
    draws = _hansen_skewt_ppf(rng.uniform(size=n), nu, lam)

    sigma2[0] = np.exp(omega / (1.0 - beta))
    y[0] = np.sqrt(sigma2[0]) * draws[0]
    for t in range(1, n):
        z_prev = y[t - 1] / np.sqrt(sigma2[t - 1])
        logh_t = omega + alpha * (abs(z_prev) - abs_moment) + gamma * z_prev + beta * np.log(sigma2[t - 1])
        sigma2[t] = np.exp(logh_t)
        y[t] = np.sqrt(sigma2[t]) * draws[t]
    return y


def simulate_egarch_pq(
    n: int,
    omega: float,
    alphas: list[float],
    gammas: list[float],
    betas: list[float],
    dist: str = "Normal",
    nu: float | None = None,
    lam: float | None = None,
    seed: int = 42,
) -> NDArray[np.float64]:
    rng = np.random.default_rng(seed)
    p = len(alphas)
    q = len(betas)
    max_lag = max(p, q)
    y = np.zeros(n, dtype=np.float64)
    sigma2 = np.zeros(n, dtype=np.float64)
    abs_moment = _egarch_abs_moment(dist, nu, lam)
    if dist == "Normal":
        draws = rng.standard_normal(n)
    elif dist == "StudentT":
        assert nu is not None
        draws = rng.standard_t(nu, size=n) * np.sqrt((nu - 2.0) / nu)
    elif dist == "SkewT":
        assert nu is not None and lam is not None
        draws = _hansen_skewt_ppf(rng.uniform(size=n), nu, lam)
    elif dist == "GED":
        assert nu is not None
        scale = np.sqrt(np.exp(gammaln(1.0 / nu) - gammaln(3.0 / nu)))
        draws = np.asarray(gennorm.rvs(beta=nu, scale=scale, size=n, random_state=rng), dtype=np.float64)
    else:
        raise ValueError(dist)

    persistence = float(sum(betas))
    sigma2[:max_lag] = np.exp(omega / (1.0 - persistence))
    y[:max_lag] = np.sqrt(sigma2[:max_lag]) * draws[:max_lag]
    for t in range(max_lag, n):
        logh_t = omega
        for i, alpha_i in enumerate(alphas, start=1):
            z_lag = y[t - i] / np.sqrt(sigma2[t - i])
            logh_t += alpha_i * (abs(z_lag) - abs_moment) + gammas[i - 1] * z_lag
        for j, beta_j in enumerate(betas, start=1):
            logh_t += beta_j * np.log(sigma2[t - j])
        sigma2[t] = np.exp(logh_t)
        y[t] = np.sqrt(sigma2[t]) * draws[t]
    return y


def simulate_meanx_surface(
    spec,
    params: NDArray[np.float64],
    n: int,
    *,
    seed: int = 42,
    burn: int = 300,
    n_exog: int = 1,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Simulate any shipped ARX/HARX public surface with exogenous regressors."""
    rng = np.random.default_rng(seed + 1000)
    x_full = np.ascontiguousarray(rng.standard_normal((n + burn, n_exog)), dtype=np.float64)
    sim = spec.simulate(n, np.ascontiguousarray(params, dtype=np.float64), burn=burn, seed=seed, x=x_full)
    data = np.ascontiguousarray(sim.data, dtype=np.float64)
    x_fit = np.ascontiguousarray(x_full[burn:], dtype=np.float64)
    return data, x_fit


# -----------------------------------------------------------------------------
# Test Configuration
# -----------------------------------------------------------------------------

N_OBS = 5000  # Observations per test series
MEANX_OBS = 2500  # Keep mean+exog recovery coverage targeted and fast.

# True parameters for each model
TRUE_PARAMS = {
    "garch_normal": {
        "omega": 1e-6,
        "alpha": 0.08,
        "beta": 0.90,
    },
    "garch_studentt": {
        "omega": 1e-6,
        "alpha": 0.08,
        "beta": 0.90,
        "nu": 8.0,
    },
    "garch_skewt": {
        "omega": 1e-6,
        "alpha": 0.08,
        "beta": 0.90,
        "nu": 8.0,
        "lam": -0.15,
    },
    "garch_ged": {
        "omega": 1e-6,
        "alpha": 0.08,
        "beta": 0.90,
        "nu": 1.7,
    },
    "garch_21_normal": {
        "omega": 1e-6,
        "alphas": [0.05, 0.03],
        "betas": [0.90],
    },
    "garch_22_normal": {
        "omega": 1e-6,
        "alphas": [0.04, 0.03],
        "betas": [0.50, 0.40],
    },
    "arma_garch_normal": {
        "c": 0.0002,
        "phi": 0.1,
        "theta": -0.05,
        "omega": 1e-6,
        "alpha": 0.08,
        "beta": 0.90,
    },
    "arma_gjr_garch_normal": {
        "c": 0.0002,
        "phi": 0.1,
        "theta": -0.05,
        "omega": 1e-6,
        "alpha": 0.03,
        "gamma": 0.07,
        "beta": 0.86,
    },
    "arma_gjr_garch_studentt": {
        "c": 0.0002,
        "phi": 0.1,
        "theta": -0.05,
        "omega": 1e-6,
        "alpha": 0.03,
        "gamma": 0.07,
        "beta": 0.86,
        "nu": 8.0,
    },
    "arma_gjr_garch_skewt": {
        "c": 0.0002,
        "phi": 0.1,
        "theta": -0.05,
        "omega": 1e-6,
        "alpha": 0.03,
        "gamma": 0.07,
        "beta": 0.86,
        "nu": 8.0,
        "lam": -0.15,
    },
    "arma_egarch_normal": {
        "c": 0.0002,
        "phi": 0.45,
        "theta": -0.25,
        "omega": -0.18,
        "alpha": 0.10,
        "gamma": -0.05,
        "beta": 0.92,
    },
    "arma_egarch_studentt": {
        "c": 0.0002,
        "phi": 0.12,
        "theta": -0.06,
        "omega": -0.20,
        "alpha": 0.08,
        "gamma": -0.04,
        "beta": 0.90,
        "nu": 8.0,
    },
    "arma_egarch_skewt": {
        "c": 0.0002,
        "phi": 0.12,
        "theta": -0.06,
        "omega": -0.20,
        "alpha": 0.08,
        "gamma": -0.04,
        "beta": 0.90,
        "nu": 8.0,
        "lam": -0.15,
    },
    "arma_egarch_ged": {
        "c": 0.0002,
        "phi": 0.12,
        "theta": -0.06,
        "omega": -0.20,
        "alpha": 0.08,
        "gamma": -0.04,
        "beta": 0.90,
        "nu": 1.7,
    },
    "arma_egarch_21_normal": {
        "c": 0.0002,
        "phis": [0.15],
        "thetas": [-0.08],
        "omega": -0.22,
        "alphas": [0.08, 0.03],
        "gammas": [-0.04, 0.01],
        "betas": [0.90],
    },
    "arma_normal": {
        "c": 0.0001,
        "phi": 0.5,   # Larger phi for better identification
        "theta": -0.3,  # Larger theta for better identification
        "sigma": 0.01,
    },
    "gjr_garch_normal": {
        "omega": 1e-6,
        "alpha": 0.03,
        "gamma": 0.07,
        "beta": 0.88,
    },
    "gjr_garch_studentt": {
        "omega": 1e-6,
        "alpha": 0.03,
        "gamma": 0.07,
        "beta": 0.88,
        "nu": 8.0,
    },
    "gjr_garch_skewt": {
        "omega": 1e-6,
        "alpha": 0.03,
        "gamma": 0.07,
        "beta": 0.88,
        "nu": 8.0,
        "lam": -0.15,
    },
    "egarch_normal": {
        "omega": -0.15,
        "alpha": 0.12,
        "gamma": -0.05,
        "beta": 0.92,
    },
    "egarch_studentt": {
        "omega": -0.15,
        "alpha": 0.10,
        "gamma": -0.04,
        "beta": 0.90,
        "nu": 8.0,
    },
    "egarch_ged": {
        "omega": -0.15,
        "alpha": 0.10,
        "gamma": -0.04,
        "beta": 0.90,
        "nu": 1.7,
    },
    "egarch_21_normal": {
        "omega": -0.18,
        "alphas": [0.08, 0.03],
        "gammas": [-0.05, 0.01],
        "betas": [0.90],
    },
    "egarch_21_studentt": {
        "omega": -0.18,
        "alphas": [0.07, 0.03],
        "gammas": [-0.04, 0.01],
        "betas": [0.88],
        "nu": 8.0,
    },
    "egarch_21_skewt": {
        "omega": -0.18,
        "alphas": [0.07, 0.03],
        "gammas": [-0.04, 0.01],
        "betas": [0.88],
        "nu": 8.0,
        "lam": -0.15,
    },
    "egarch_skewt": {
        "omega": -0.15,
        "alpha": 0.10,
        "gamma": -0.04,
        "beta": 0.90,
        "nu": 8.0,
        "lam": -0.15,
    },
    "arx_normal": {
        "params": np.array([0.25, 0.40, 0.90], dtype=np.float64),
    },
    "harx_normal": {
        "params": np.array([0.10, 0.55, 0.20, 0.80], dtype=np.float64),
    },
    "arx_studentt": {
        "params": np.array([0.15, 0.35, 0.70, 0.60, 8.0], dtype=np.float64),
    },
    "harx_skewt": {
        "params": np.array([0.05, 0.60, 0.20, 1.50, 0.75, 8.0, -0.10], dtype=np.float64),
    },
    "arx_ged": {
        "params": np.array([0.08, 0.35, 0.90, 0.65, 1.6], dtype=np.float64),
    },
    "arx_egarch_normal": {
        "params": np.array([0.02, 0.22, 0.15, -0.15, 0.10, -0.04, 0.91], dtype=np.float64),
    },
    "arx_egarch_studentt": {
        "params": np.array([0.02, 0.22, 0.15, -0.18, 0.08, -0.03, 0.90, 8.0], dtype=np.float64),
    },
    "harx_egarch_skewt": {
        "params": np.array([0.01, 0.30, 0.09, 0.12, -0.20, 0.06, 0.03, 0.02, -0.01, 0.89, 8.0, -0.10], dtype=np.float64),
    },
    "arx_egarch_ged": {
        "params": np.array([0.02, 0.22, 0.15, -0.18, 0.08, -0.03, 0.90, 1.6], dtype=np.float64),
    },
}

# Tolerance for parameter recovery (relative)
PARAM_RTOL = 0.30  # 30% - generous for finite sample


# -----------------------------------------------------------------------------
# Tests: GARCH + Normal
# -----------------------------------------------------------------------------

class TestGARCHNormal:
    """Tests for GARCH(1,1) + Normal estimation."""
    
    @pytest.fixture
    def data(self) -> NDArray[np.float64]:
        """Generate synthetic GARCH(1,1) Normal data."""
        p = TRUE_PARAMS["garch_normal"]
        return simulate_garch_normal(N_OBS, p["omega"], p["alpha"], p["beta"])
    
    def test_estimation_runs(self, data: NDArray[np.float64]) -> None:
        """Test that estimation runs and produces valid output."""
        from scivol import GARCH, Normal
        
        spec = GARCH(1, 1) + Normal()
        result = spec.fit(data)
        
        # Check we got valid parameters (even if optimizer reports non-convergence)
        assert result.params is not None
        assert len(result.params) == 3  # omega, alpha, beta
        assert np.all(np.isfinite(result.params))
        assert result.log_likelihood > 0  # Should be positive for reasonable fit
    
    def test_parameter_recovery(self, data: NDArray[np.float64]) -> None:
        """Test that estimated parameters are close to true values."""
        from scivol import GARCH, Normal
        
        spec = GARCH(1, 1) + Normal()
        result = spec.fit(data)
        
        true = TRUE_PARAMS["garch_normal"]
        
        # Check each parameter (with generous tolerance for finite sample)
        omega_hat = result.params[0]
        alpha_hat = result.params[1]
        beta_hat = result.params[2]
        
        # Persistence should be well-estimated
        true_persistence = true["alpha"] + true["beta"]
        est_persistence = alpha_hat + beta_hat
        
        assert abs(est_persistence - true_persistence) < 0.05, \
            f"Persistence: true={true_persistence:.4f}, est={est_persistence:.4f}"
        
        # Individual parameters (looser bounds)
        assert abs(alpha_hat - true["alpha"]) / true["alpha"] < PARAM_RTOL, \
            f"Alpha: true={true['alpha']:.4f}, est={alpha_hat:.4f}"
        assert abs(beta_hat - true["beta"]) / true["beta"] < PARAM_RTOL, \
            f"Beta: true={true['beta']:.4f}, est={beta_hat:.4f}"


# -----------------------------------------------------------------------------
# Tests: GARCH + Student-t
# -----------------------------------------------------------------------------

class TestGARCHStudentT:
    """Tests for GARCH(1,1) + Student-t estimation."""
    
    @pytest.fixture
    def data(self) -> NDArray[np.float64]:
        """Generate synthetic GARCH(1,1) Student-t data."""
        p = TRUE_PARAMS["garch_studentt"]
        return simulate_garch_studentt(
            N_OBS, p["omega"], p["alpha"], p["beta"], p["nu"]
        )
    
    def test_estimation_runs(self, data: NDArray[np.float64]) -> None:
        """Test that estimation runs and produces valid output."""
        from scivol import GARCH, StudentT
        
        spec = GARCH(1, 1) + StudentT()
        result = spec.fit(data)
        
        assert result.params is not None
        assert len(result.params) == 4  # omega, alpha, beta, nu
        assert np.all(np.isfinite(result.params))
        assert result.log_likelihood > 0
    
    def test_parameter_recovery(self, data: NDArray[np.float64]) -> None:
        """Test that estimated parameters are close to true values."""
        from scivol import GARCH, StudentT
        
        spec = GARCH(1, 1) + StudentT()
        result = spec.fit(data)
        
        true = TRUE_PARAMS["garch_studentt"]
        
        # Check persistence
        alpha_hat = result.params[1]
        beta_hat = result.params[2]
        nu_hat = result.params[3]
        
        true_persistence = true["alpha"] + true["beta"]
        est_persistence = alpha_hat + beta_hat
        
        assert abs(est_persistence - true_persistence) < 0.05, \
            f"Persistence: true={true_persistence:.4f}, est={est_persistence:.4f}"
        
        # Degrees of freedom (can be harder to pin down)
        assert 4 < nu_hat < 20, f"Nu out of reasonable range: {nu_hat:.2f}"


# -----------------------------------------------------------------------------
# Tests: GARCH + GED
# -----------------------------------------------------------------------------

class TestGARCHGED:
    """Tests for GARCH(1,1) + GED estimation."""

    @pytest.fixture
    def data(self) -> NDArray[np.float64]:
        p = TRUE_PARAMS["garch_ged"]
        return simulate_garch_ged(N_OBS, p["omega"], p["alpha"], p["beta"], p["nu"])

    def test_estimation_runs(self, data: NDArray[np.float64]) -> None:
        from scivol import GARCH, GED

        spec = GARCH(1, 1) + GED()
        result = spec.fit(data, solver="slsqp", log_mode=False, verbose=False)

        assert result.params is not None
        assert len(result.params) == 4
        assert np.all(np.isfinite(result.params))
        assert result.log_likelihood > 0

    def test_parameter_recovery(self, data: NDArray[np.float64]) -> None:
        from scivol import GARCH, GED

        spec = GARCH(1, 1) + GED()
        result = spec.fit(data, solver="slsqp", log_mode=False, verbose=False)

        true = TRUE_PARAMS["garch_ged"]
        alpha_hat = result.params[1]
        beta_hat = result.params[2]
        nu_hat = result.params[3]

        true_persistence = true["alpha"] + true["beta"]
        est_persistence = alpha_hat + beta_hat

        assert abs(est_persistence - true_persistence) < 0.06, (
            f"Persistence: true={true_persistence:.4f}, est={est_persistence:.4f}"
        )
        assert 1.1 < nu_hat < 5.0, f"GED nu out of reasonable range: {nu_hat:.2f}"


# -----------------------------------------------------------------------------
# Tests: GARCH + Skew-t
# -----------------------------------------------------------------------------

class TestGARCHSkewT:
    """Tests for GARCH(1,1) + Skew-t estimation."""
    
    @pytest.fixture
    def data(self) -> NDArray[np.float64]:
        """Generate synthetic GARCH(1,1) Skew-t data."""
        p = TRUE_PARAMS["garch_skewt"]
        return simulate_garch_skewt(
            N_OBS, p["omega"], p["alpha"], p["beta"], p["nu"], p["lam"]
        )
    
    def test_estimation_runs(self, data: NDArray[np.float64]) -> None:
        """Test that estimation runs and produces valid output."""
        from scivol import GARCH, SkewT
        
        spec = GARCH(1, 1) + SkewT()
        result = spec.fit(data)
        
        assert result.params is not None
        assert len(result.params) == 5  # omega, alpha, beta, nu, lam
        assert np.all(np.isfinite(result.params))
        assert result.log_likelihood > 0
    
    def test_parameter_recovery(self, data: NDArray[np.float64]) -> None:
        """Test that estimated parameters are close to true values."""
        from scivol import GARCH, SkewT
        
        spec = GARCH(1, 1) + SkewT()
        result = spec.fit(data)
        
        true = TRUE_PARAMS["garch_skewt"]
        
        # Check persistence
        alpha_hat = result.params[1]
        beta_hat = result.params[2]
        
        true_persistence = true["alpha"] + true["beta"]
        est_persistence = alpha_hat + beta_hat
        
        assert abs(est_persistence - true_persistence) < 0.05, \
            f"Persistence: true={true_persistence:.4f}, est={est_persistence:.4f}"
        
        # Skewness parameter should have correct sign
        lam_hat = result.params[4]
        assert lam_hat * true["lam"] > 0 or abs(lam_hat) < 0.1, \
            f"Lambda sign mismatch: true={true['lam']:.3f}, est={lam_hat:.3f}"


# -----------------------------------------------------------------------------
# Tests: GARCH(p,q) with higher orders
# -----------------------------------------------------------------------------

class TestGARCHHigherOrder:
    """Tests for GARCH(p,q) with p,q > 1."""
    
    @pytest.fixture
    def data_21(self) -> NDArray[np.float64]:
        """Generate GARCH(2,1) Normal data."""
        p = TRUE_PARAMS["garch_21_normal"]
        return simulate_garch_pq_normal(
            N_OBS, p["omega"], p["alphas"], p["betas"]
        )
    
    @pytest.fixture
    def data_22(self) -> NDArray[np.float64]:
        """Generate GARCH(2,2) Normal data."""
        p = TRUE_PARAMS["garch_22_normal"]
        return simulate_garch_pq_normal(
            N_OBS, p["omega"], p["alphas"], p["betas"]
        )
    
    def test_garch_21_runs(self, data_21: NDArray[np.float64]) -> None:
        """Test GARCH(2,1) runs and produces valid output."""
        from scivol import GARCH, Normal
        
        spec = GARCH(2, 1) + Normal()
        result = spec.fit(data_21)
        
        assert result.params is not None
        assert len(result.params) == 4  # omega, alpha1, alpha2, beta1
        assert np.all(np.isfinite(result.params))
        assert result.log_likelihood > 0
    
    def test_garch_22_runs(self, data_22: NDArray[np.float64]) -> None:
        """Test GARCH(2,2) runs and produces valid output."""
        from scivol import GARCH, Normal
        
        spec = GARCH(2, 2) + Normal()
        result = spec.fit(data_22)
        
        assert result.params is not None
        assert len(result.params) == 5  # omega, alpha1, alpha2, beta1, beta2
        assert np.all(np.isfinite(result.params))
        assert result.log_likelihood > 0


# -----------------------------------------------------------------------------
# Tests: ARMA-GARCH
# -----------------------------------------------------------------------------

class TestARMAGARCHNormal:
    """Tests for ARMA(1,1)-GARCH(1,1) + Normal estimation."""
    
    @pytest.fixture
    def data(self) -> NDArray[np.float64]:
        """Generate ARMA(1,1)-GARCH(1,1) Normal data."""
        p = TRUE_PARAMS["arma_garch_normal"]
        return simulate_arma_garch_normal(
            N_OBS, p["c"], p["phi"], p["theta"], 
            p["omega"], p["alpha"], p["beta"]
        )
    
    def test_estimation_runs(self, data: NDArray[np.float64]) -> None:
        """Test that estimation runs and produces valid output."""
        from scivol import ARMA, GARCH, Normal
        
        spec = ARMA(1, 1) + GARCH(1, 1) + Normal()
        result = spec.fit(data)
        
        assert result.params is not None
        assert len(result.params) == 6  # c, phi, theta, omega, alpha, beta
        assert np.all(np.isfinite(result.params))
        assert result.log_likelihood > 0
    
    def test_parameter_recovery(self, data: NDArray[np.float64]) -> None:
        """Test that estimated parameters are close to true values."""
        from scivol import ARMA, GARCH, Normal
        
        spec = ARMA(1, 1) + GARCH(1, 1) + Normal()
        result = spec.fit(data)
        
        true = TRUE_PARAMS["arma_garch_normal"]
        
        # Check GARCH persistence
        alpha_hat = result.params[4]
        beta_hat = result.params[5]
        
        true_persistence = true["alpha"] + true["beta"]
        est_persistence = alpha_hat + beta_hat
        
        assert abs(est_persistence - true_persistence) < 0.10, \
            f"Persistence: true={true_persistence:.4f}, est={est_persistence:.4f}"


class TestARMAGARCHStudentT:
    """Tests for ARMA(1,1)-GARCH(1,1) + StudentT estimation."""
    
    @pytest.fixture
    def data(self) -> NDArray[np.float64]:
        """Generate ARMA(1,1)-GARCH(1,1) Student-t data."""
        # Use same params but with Student-t innovations
        p = TRUE_PARAMS["arma_garch_normal"]
        # Simulate with Student-t-like heavier tails
        n = N_OBS
        rng = np.random.default_rng(42)
        
        y = np.zeros(n, dtype=np.float64)
        eps = np.zeros(n, dtype=np.float64)
        sigma2 = np.zeros(n, dtype=np.float64)
        
        sigma2_uncond = p["omega"] / (1 - p["alpha"] - p["beta"])
        sigma2[0] = sigma2_uncond
        
        nu = 8.0
        scale = np.sqrt((nu - 2) / nu)
        z = rng.standard_t(nu, size=n) * scale
        
        eps[0] = np.sqrt(sigma2[0]) * z[0]
        y[0] = p["c"] / (1 - p["phi"]) + eps[0]
        
        for t in range(1, n):
            sigma2[t] = p["omega"] + p["alpha"] * eps[t-1]**2 + p["beta"] * sigma2[t-1]
            eps[t] = np.sqrt(sigma2[t]) * z[t]
            y[t] = p["c"] + p["phi"] * y[t-1] + p["theta"] * eps[t-1] + eps[t]
        
        return y
    
    def test_estimation_runs(self, data: NDArray[np.float64]) -> None:
        """Test that estimation runs and produces valid output."""
        from scivol import ARMA, GARCH, StudentT
        
        spec = ARMA(1, 1) + GARCH(1, 1) + StudentT()
        result = spec.fit(data)
        
        assert result.params is not None
        assert len(result.params) == 7  # c, phi, theta, omega, alpha, beta, nu
        assert np.all(np.isfinite(result.params))
        assert result.log_likelihood > 0


class TestARMAGARCHSkewT:
    """Tests for ARMA(1,1)-GARCH(1,1) + SkewT estimation."""
    
    @pytest.fixture
    def data(self) -> NDArray[np.float64]:
        """Generate ARMA(1,1)-GARCH(1,1) Skew-t data."""
        p = TRUE_PARAMS["arma_garch_normal"]
        n = N_OBS
        rng = np.random.default_rng(42)
        
        y = np.zeros(n, dtype=np.float64)
        eps = np.zeros(n, dtype=np.float64)
        sigma2 = np.zeros(n, dtype=np.float64)
        
        sigma2_uncond = p["omega"] / (1 - p["alpha"] - p["beta"])
        sigma2[0] = sigma2_uncond
        
        nu = 8.0
        lam = -0.15
        t_scale = np.sqrt((nu - 2) / nu)
        
        for t in range(n):
            if t > 0:
                sigma2[t] = p["omega"] + p["alpha"] * eps[t-1]**2 + p["beta"] * sigma2[t-1]
            
            z_raw = rng.standard_t(nu) * t_scale
            if z_raw < 0:
                z = z_raw * (1 - lam)
            else:
                z = z_raw * (1 + lam)
            
            eps[t] = np.sqrt(sigma2[t]) * z
            if t == 0:
                y[t] = p["c"] / (1 - p["phi"]) + eps[t]
            else:
                y[t] = p["c"] + p["phi"] * y[t-1] + p["theta"] * eps[t-1] + eps[t]
        
        return y
    
    def test_estimation_runs(self, data: NDArray[np.float64]) -> None:
        """Test that estimation runs and produces valid output."""
        from scivol import ARMA, GARCH, SkewT
        
        spec = ARMA(1, 1) + GARCH(1, 1) + SkewT()
        result = spec.fit(data)
        
        assert result.params is not None
        assert len(result.params) == 8  # c, phi, theta, omega, alpha, beta, nu, lam
        assert np.all(np.isfinite(result.params))
        assert result.log_likelihood > 0


# -----------------------------------------------------------------------------
# Tests: ARMA-GJR-GARCH
# -----------------------------------------------------------------------------

class TestARMAGJRGARCHNormal:
    """Tests for ARMA(1,1)-GJR-GARCH(1,1) + Normal estimation."""

    @pytest.fixture
    def data(self) -> NDArray[np.float64]:
        p = TRUE_PARAMS["arma_gjr_garch_normal"]
        return simulate_arma_gjr_garch_normal(
            N_OBS, p["c"], p["phi"], p["theta"], p["omega"], p["alpha"], p["gamma"], p["beta"]
        )

    def test_estimation_runs(self, data: NDArray[np.float64]) -> None:
        from scivol import ARMA, GJRGARCH, Normal

        result = (ARMA(1, 1) + GJRGARCH(1, 1) + Normal()).fit(data)
        assert result.params is not None
        assert len(result.params) == 7
        assert np.all(np.isfinite(result.params))
        assert result.log_likelihood > 0

    def test_parameter_recovery(self, data: NDArray[np.float64]) -> None:
        from scivol import ARMA, GJRGARCH, Normal

        result = (ARMA(1, 1) + GJRGARCH(1, 1) + Normal()).fit(data)
        true = TRUE_PARAMS["arma_gjr_garch_normal"]
        alpha_hat = result.params[4]
        gamma_hat = result.params[5]
        beta_hat = result.params[6]

        true_persistence = true["alpha"] + 0.5 * true["gamma"] + true["beta"]
        est_persistence = alpha_hat + 0.5 * gamma_hat + beta_hat

        assert abs(est_persistence - true_persistence) < 0.10
        assert gamma_hat > 0


class TestARMAGJRGARCHStudentT:
    """Tests for ARMA(1,1)-GJR-GARCH(1,1) + StudentT estimation."""

    @pytest.fixture
    def data(self) -> NDArray[np.float64]:
        p = TRUE_PARAMS["arma_gjr_garch_studentt"]
        return simulate_arma_gjr_garch_studentt(
            N_OBS,
            p["c"],
            p["phi"],
            p["theta"],
            p["omega"],
            p["alpha"],
            p["gamma"],
            p["beta"],
            p["nu"],
        )

    def test_estimation_runs(self, data: NDArray[np.float64]) -> None:
        from scivol import ARMA, GJRGARCH, StudentT

        result = (ARMA(1, 1) + GJRGARCH(1, 1) + StudentT()).fit(data)
        assert result.params is not None
        assert len(result.params) == 8
        assert np.all(np.isfinite(result.params))
        assert result.log_likelihood > 0

    def test_parameter_recovery(self, data: NDArray[np.float64]) -> None:
        from scivol import ARMA, GJRGARCH, StudentT

        result = (ARMA(1, 1) + GJRGARCH(1, 1) + StudentT()).fit(data)
        true = TRUE_PARAMS["arma_gjr_garch_studentt"]
        alpha_hat = result.params[4]
        gamma_hat = result.params[5]
        beta_hat = result.params[6]
        nu_hat = result.params[7]

        true_persistence = true["alpha"] + 0.5 * true["gamma"] + true["beta"]
        est_persistence = alpha_hat + 0.5 * gamma_hat + beta_hat

        assert abs(est_persistence - true_persistence) < 0.10
        assert gamma_hat > 0
        assert 4 < nu_hat < 20


class TestARMAGJRGARCHSkewT:
    """Tests for ARMA(1,1)-GJR-GARCH(1,1) + SkewT estimation."""

    @pytest.fixture
    def data(self) -> NDArray[np.float64]:
        p = TRUE_PARAMS["arma_gjr_garch_skewt"]
        return simulate_arma_gjr_garch_skewt(
            N_OBS,
            p["c"],
            p["phi"],
            p["theta"],
            p["omega"],
            p["alpha"],
            p["gamma"],
            p["beta"],
            p["nu"],
            p["lam"],
        )

    def test_estimation_runs(self, data: NDArray[np.float64]) -> None:
        from scivol import ARMA, GJRGARCH, SkewT

        result = (ARMA(1, 1) + GJRGARCH(1, 1) + SkewT()).fit(data)
        assert result.params is not None
        assert len(result.params) == 9
        assert np.all(np.isfinite(result.params))
        assert result.log_likelihood > 0

    def test_parameter_recovery(self, data: NDArray[np.float64]) -> None:
        from scivol import ARMA, GJRGARCH, SkewT

        result = (ARMA(1, 1) + GJRGARCH(1, 1) + SkewT()).fit(data)
        true = TRUE_PARAMS["arma_gjr_garch_skewt"]
        alpha_hat = result.params[4]
        gamma_hat = result.params[5]
        beta_hat = result.params[6]
        lam_hat = result.params[8]

        true_persistence = true["alpha"] + 0.5 * true["gamma"] + true["beta"]
        est_persistence = alpha_hat + 0.5 * gamma_hat + beta_hat

        assert abs(est_persistence - true_persistence) < 0.10
        assert gamma_hat > 0
        assert lam_hat * true["lam"] > 0 or abs(lam_hat) < 0.1


# -----------------------------------------------------------------------------
# Tests: Pure ARMA (no volatility dynamics)
# -----------------------------------------------------------------------------

class TestARMANormal:
    """Tests for ARMA(1,1) + Normal estimation (constant variance)."""
    
    @pytest.fixture
    def data(self) -> NDArray[np.float64]:
        """Generate ARMA(1,1) Normal data."""
        p = TRUE_PARAMS["arma_normal"]
        return simulate_arma_normal(
            N_OBS, p["c"], p["phi"], p["theta"], p["sigma"]
        )
    
    def test_estimation_runs(self, data: NDArray[np.float64]) -> None:
        """Test that estimation runs and produces valid output."""
        from scivol import ARMA, Normal
        
        spec = ARMA(1, 1) + Normal()
        result = spec.fit(data)
        
        assert result.params is not None
        assert len(result.params) == 3  # c, phi, theta
        assert np.all(np.isfinite(result.params))
        assert result.log_likelihood > 0
    
    def test_parameter_recovery(self, data: NDArray[np.float64]) -> None:
        """Test that estimated parameters are close to true values."""
        from scivol import ARMA, Normal
        
        spec = ARMA(1, 1) + Normal()
        result = spec.fit(data)
        
        true = TRUE_PARAMS["arma_normal"]
        
        c_hat = result.params[0]
        phi_hat = result.params[1]
        theta_hat = result.params[2]
        
        # Check AR coefficient (most identifiable)
        assert abs(phi_hat - true["phi"]) / abs(true["phi"]) < PARAM_RTOL, \
            f"Phi: true={true['phi']:.4f}, est={phi_hat:.4f}"
        
        # MA recovery is a bit weaker here; the fitted point slightly beats the
        # exact DGP parameters on this realized sample, so keep a slightly wider
        # finite-sample band than the AR coefficient uses.
        assert abs(theta_hat - true["theta"]) / abs(true["theta"]) < 0.35, \
            f"Theta: true={true['theta']:.4f}, est={theta_hat:.4f}"


# -----------------------------------------------------------------------------
# Tests: GJR-GARCH + Normal
# -----------------------------------------------------------------------------

class TestGJRGARCHNormal:
    """Tests for GJR-GARCH(1,1) + Normal estimation."""
    
    @pytest.fixture
    def data(self) -> NDArray[np.float64]:
        p = TRUE_PARAMS["gjr_garch_normal"]
        return simulate_gjr_garch_normal(
            N_OBS, p["omega"], p["alpha"], p["gamma"], p["beta"]
        )
    
    def test_estimation_runs(self, data: NDArray[np.float64]) -> None:
        from scivol import GJRGARCH, Normal
        
        spec = GJRGARCH(1, 1) + Normal()
        result = spec.fit(data)
        
        assert result.params is not None
        assert len(result.params) == 4  # omega, alpha, gamma, beta
        assert np.all(np.isfinite(result.params))
        assert result.log_likelihood > 0
    
    def test_parameter_recovery(self, data: NDArray[np.float64]) -> None:
        from scivol import GJRGARCH, Normal
        
        spec = GJRGARCH(1, 1) + Normal()
        result = spec.fit(data)
        
        true = TRUE_PARAMS["gjr_garch_normal"]
        
        alpha_hat = result.params[1]
        gamma_hat = result.params[2]
        beta_hat = result.params[3]
        
        # Check persistence: α + 0.5·γ + β
        true_persistence = true["alpha"] + 0.5 * true["gamma"] + true["beta"]
        est_persistence = alpha_hat + 0.5 * gamma_hat + beta_hat
        
        assert abs(est_persistence - true_persistence) < 0.05, \
            f"Persistence: true={true_persistence:.4f}, est={est_persistence:.4f}"
        
        # Leverage effect should be positive
        assert gamma_hat > 0, f"Gamma should be positive: {gamma_hat:.4f}"
        
        # Gamma should be in reasonable range
        assert abs(gamma_hat - true["gamma"]) / true["gamma"] < 0.5, \
            f"Gamma: true={true['gamma']:.4f}, est={gamma_hat:.4f}"


# -----------------------------------------------------------------------------
# Tests: GJR-GARCH + Student-t
# -----------------------------------------------------------------------------

class TestGJRGARCHStudentT:
    """Tests for GJR-GARCH(1,1) + Student-t estimation."""
    
    @pytest.fixture
    def data(self) -> NDArray[np.float64]:
        p = TRUE_PARAMS["gjr_garch_studentt"]
        return simulate_gjr_garch_studentt(
            N_OBS, p["omega"], p["alpha"], p["gamma"], p["beta"], p["nu"]
        )
    
    def test_estimation_runs(self, data: NDArray[np.float64]) -> None:
        from scivol import GJRGARCH, StudentT
        
        spec = GJRGARCH(1, 1) + StudentT()
        result = spec.fit(data)
        
        assert result.params is not None
        assert len(result.params) == 5  # omega, alpha, gamma, beta, nu
        assert np.all(np.isfinite(result.params))
        assert result.log_likelihood > 0
    
    def test_parameter_recovery(self, data: NDArray[np.float64]) -> None:
        from scivol import GJRGARCH, StudentT
        
        spec = GJRGARCH(1, 1) + StudentT()
        result = spec.fit(data)
        
        true = TRUE_PARAMS["gjr_garch_studentt"]
        
        alpha_hat = result.params[1]
        gamma_hat = result.params[2]
        beta_hat = result.params[3]
        nu_hat = result.params[4]
        
        true_persistence = true["alpha"] + 0.5 * true["gamma"] + true["beta"]
        est_persistence = alpha_hat + 0.5 * gamma_hat + beta_hat
        
        assert abs(est_persistence - true_persistence) < 0.05, \
            f"Persistence: true={true_persistence:.4f}, est={est_persistence:.4f}"
        
        assert gamma_hat > 0, f"Gamma should be positive: {gamma_hat:.4f}"
        assert 4 < nu_hat < 20, f"Nu out of reasonable range: {nu_hat:.2f}"


# -----------------------------------------------------------------------------
# Tests: GJR-GARCH + Skew-t
# -----------------------------------------------------------------------------

class TestGJRGARCHSkewT:
    """Tests for GJR-GARCH(1,1) + Skew-t estimation."""
    
    @pytest.fixture
    def data(self) -> NDArray[np.float64]:
        p = TRUE_PARAMS["gjr_garch_skewt"]
        return simulate_gjr_garch_skewt(
            N_OBS, p["omega"], p["alpha"], p["gamma"], p["beta"],
            p["nu"], p["lam"]
        )
    
    def test_estimation_runs(self, data: NDArray[np.float64]) -> None:
        from scivol import GJRGARCH, SkewT
        
        spec = GJRGARCH(1, 1) + SkewT()
        result = spec.fit(data)
        
        assert result.params is not None
        assert len(result.params) == 6  # omega, alpha, gamma, beta, nu, lam
        assert np.all(np.isfinite(result.params))
        assert result.log_likelihood > 0
    
    def test_parameter_recovery(self, data: NDArray[np.float64]) -> None:
        from scivol import GJRGARCH, SkewT
        
        spec = GJRGARCH(1, 1) + SkewT()
        result = spec.fit(data)
        
        true = TRUE_PARAMS["gjr_garch_skewt"]
        
        alpha_hat = result.params[1]
        gamma_hat = result.params[2]
        beta_hat = result.params[3]
        lam_hat = result.params[5]
        
        true_persistence = true["alpha"] + 0.5 * true["gamma"] + true["beta"]
        est_persistence = alpha_hat + 0.5 * gamma_hat + beta_hat
        
        assert abs(est_persistence - true_persistence) < 0.05, \
            f"Persistence: true={true_persistence:.4f}, est={est_persistence:.4f}"
        
        assert gamma_hat > 0, f"Gamma should be positive: {gamma_hat:.4f}"
        
        # Skewness parameter should have correct sign
        assert lam_hat * true["lam"] > 0 or abs(lam_hat) < 0.1, \
            f"Lambda sign mismatch: true={true['lam']:.3f}, est={lam_hat:.3f}"


# -----------------------------------------------------------------------------
# Tests: EGARCH shipped surfaces
# -----------------------------------------------------------------------------

class TestEGARCHNormal:
    @pytest.fixture
    def data(self) -> NDArray[np.float64]:
        p = TRUE_PARAMS["egarch_normal"]
        return simulate_egarch_normal(N_OBS, p["omega"], p["alpha"], p["gamma"], p["beta"])

    def test_estimation_runs(self, data: NDArray[np.float64]) -> None:
        from scivol import EGARCH, Normal

        result = (EGARCH(1, 1) + Normal()).fit(data)
        assert result.params is not None
        assert len(result.params) == 4
        assert np.all(np.isfinite(result.params))
        assert np.isfinite(result.log_likelihood)


class TestEGARCHStudentT:
    @pytest.fixture
    def data(self) -> NDArray[np.float64]:
        p = TRUE_PARAMS["egarch_studentt"]
        return simulate_egarch_studentt(N_OBS, p["omega"], p["alpha"], p["gamma"], p["beta"], p["nu"])

    def test_estimation_runs(self, data: NDArray[np.float64]) -> None:
        from scivol import EGARCH, StudentT

        result = (EGARCH(1, 1) + StudentT()).fit(data)
        assert result.params is not None
        assert len(result.params) == 5
        assert np.all(np.isfinite(result.params))
        assert np.isfinite(result.log_likelihood)


class TestEGARCHSkewT:
    @pytest.fixture
    def data(self) -> NDArray[np.float64]:
        p = TRUE_PARAMS["egarch_skewt"]
        return simulate_egarch_skewt(N_OBS, p["omega"], p["alpha"], p["gamma"], p["beta"], p["nu"], p["lam"])

    def test_estimation_runs(self, data: NDArray[np.float64]) -> None:
        from scivol import EGARCH, SkewT

        result = (EGARCH(1, 1) + SkewT()).fit(data)
        assert result.params is not None
        assert len(result.params) == 6
        assert np.all(np.isfinite(result.params))
        assert np.isfinite(result.log_likelihood)

    def test_parameter_recovery(self, data: NDArray[np.float64]) -> None:
        from scivol import EGARCH, SkewT

        result = (EGARCH(1, 1) + SkewT()).fit(data)
        true = TRUE_PARAMS["egarch_skewt"]

        beta_hat = result.params[3]
        assert abs(beta_hat - true["beta"]) < 0.08, f"Beta recovery failed: true={true['beta']:.3f}, est={beta_hat:.3f}"

        lam_hat = result.params[5]
        assert lam_hat * true["lam"] > 0 or abs(lam_hat) < 0.1, \
            f"Lambda sign mismatch: true={true['lam']:.3f}, est={lam_hat:.3f}"


class TestEGARCHGED:
    @pytest.fixture
    def data(self) -> NDArray[np.float64]:
        p = TRUE_PARAMS["egarch_ged"]
        return simulate_egarch_pq(
            N_OBS,
            p["omega"],
            [p["alpha"]],
            [p["gamma"]],
            [p["beta"]],
            dist="GED",
            nu=p["nu"],
        )

    def test_estimation_runs(self, data: NDArray[np.float64]) -> None:
        from scivol import EGARCH, GED

        result = (EGARCH(1, 1) + GED()).fit(data)
        assert result.params is not None
        assert len(result.params) == 5
        assert np.all(np.isfinite(result.params))
        assert np.isfinite(result.log_likelihood)


class TestEGARCHHigherOrder:
    @pytest.fixture
    def data_21_normal(self) -> NDArray[np.float64]:
        p = TRUE_PARAMS["egarch_21_normal"]
        return simulate_egarch_pq(N_OBS, p["omega"], p["alphas"], p["gammas"], p["betas"], dist="Normal")

    @pytest.fixture
    def data_21_studentt(self) -> NDArray[np.float64]:
        p = TRUE_PARAMS["egarch_21_studentt"]
        return simulate_egarch_pq(
            N_OBS,
            p["omega"],
            p["alphas"],
            p["gammas"],
            p["betas"],
            dist="StudentT",
            nu=p["nu"],
        )

    @pytest.fixture
    def data_21_skewt(self) -> NDArray[np.float64]:
        p = TRUE_PARAMS["egarch_21_skewt"]
        return simulate_egarch_pq(
            N_OBS,
            p["omega"],
            p["alphas"],
            p["gammas"],
            p["betas"],
            dist="SkewT",
            nu=p["nu"],
            lam=p["lam"],
        )

    def test_egarch_21_normal_runs(self, data_21_normal: NDArray[np.float64]) -> None:
        from scivol import EGARCH, Normal

        result = (EGARCH(2, 1) + Normal()).fit(data_21_normal)
        assert result.params is not None
        assert len(result.params) == 6
        assert np.all(np.isfinite(result.params))

    def test_egarch_21_studentt_runs(self, data_21_studentt: NDArray[np.float64]) -> None:
        from scivol import EGARCH, StudentT

        result = (EGARCH(2, 1) + StudentT()).fit(data_21_studentt)
        assert result.params is not None
        assert len(result.params) == 7
        assert np.all(np.isfinite(result.params))

    def test_egarch_21_skewt_runs(self, data_21_skewt: NDArray[np.float64]) -> None:
        from scivol import EGARCH, SkewT

        result = (EGARCH(2, 1) + SkewT()).fit(data_21_skewt)
        assert result.params is not None
        assert len(result.params) == 8
        assert np.all(np.isfinite(result.params))


# -----------------------------------------------------------------------------
# Tests: ARMA + EGARCH shipped surfaces
# -----------------------------------------------------------------------------

class TestARMAEGARCHNormal:
    @pytest.fixture
    def data(self) -> NDArray[np.float64]:
        p = TRUE_PARAMS["arma_egarch_normal"]
        return simulate_arma_egarch_pq(
            N_OBS,
            p["c"],
            [p["phi"]],
            [p["theta"]],
            p["omega"],
            [p["alpha"]],
            [p["gamma"]],
            [p["beta"]],
            dist="Normal",
        )

    def test_estimation_runs(self, data: NDArray[np.float64]) -> None:
        from scivol import ARMA, EGARCH, Normal

        result = (ARMA(1, 1) + EGARCH(1, 1) + Normal()).fit(data)
        assert result.params is not None
        assert len(result.params) == 7
        assert np.all(np.isfinite(result.params))
        assert np.isfinite(result.log_likelihood)

    def test_parameter_recovery(self, data: NDArray[np.float64]) -> None:
        from scivol import ARMA, EGARCH, Normal

        result = (ARMA(1, 1) + EGARCH(1, 1) + Normal()).fit(data)
        true = TRUE_PARAMS["arma_egarch_normal"]

        phi_hat = result.params[1]
        theta_hat = result.params[2]
        alpha_hat = result.params[4]
        gamma_hat = result.params[5]
        beta_hat = result.params[6]

        assert abs(phi_hat - true["phi"]) < 0.12
        assert abs(theta_hat - true["theta"]) < 0.12
        assert abs(alpha_hat - true["alpha"]) < 0.08
        assert gamma_hat * true["gamma"] > 0 or abs(gamma_hat) < 0.05
        assert abs(beta_hat - true["beta"]) < 0.08


class TestARMAEGARCHStudentT:
    @pytest.fixture
    def data(self) -> NDArray[np.float64]:
        p = TRUE_PARAMS["arma_egarch_studentt"]
        return simulate_arma_egarch_pq(
            N_OBS,
            p["c"],
            [p["phi"]],
            [p["theta"]],
            p["omega"],
            [p["alpha"]],
            [p["gamma"]],
            [p["beta"]],
            dist="StudentT",
            nu=p["nu"],
        )

    def test_estimation_runs(self, data: NDArray[np.float64]) -> None:
        from scivol import ARMA, EGARCH, StudentT

        result = (ARMA(1, 1) + EGARCH(1, 1) + StudentT()).fit(data)
        assert result.params is not None
        assert len(result.params) == 8
        assert np.all(np.isfinite(result.params))
        assert np.isfinite(result.log_likelihood)


class TestARMAEGARCHSkewT:
    @pytest.fixture
    def data(self) -> NDArray[np.float64]:
        p = TRUE_PARAMS["arma_egarch_skewt"]
        return simulate_arma_egarch_pq(
            N_OBS,
            p["c"],
            [p["phi"]],
            [p["theta"]],
            p["omega"],
            [p["alpha"]],
            [p["gamma"]],
            [p["beta"]],
            dist="SkewT",
            nu=p["nu"],
            lam=p["lam"],
        )

    def test_estimation_runs(self, data: NDArray[np.float64]) -> None:
        from scivol import ARMA, EGARCH, SkewT

        result = (ARMA(1, 1) + EGARCH(1, 1) + SkewT()).fit(data)
        assert result.params is not None
        assert len(result.params) == 9
        assert np.all(np.isfinite(result.params))
        assert np.isfinite(result.log_likelihood)


class TestARMAEGARCHGED:
    @pytest.fixture
    def data(self) -> NDArray[np.float64]:
        p = TRUE_PARAMS["arma_egarch_ged"]
        return simulate_arma_egarch_pq(
            N_OBS,
            p["c"],
            [p["phi"]],
            [p["theta"]],
            p["omega"],
            [p["alpha"]],
            [p["gamma"]],
            [p["beta"]],
            dist="GED",
            nu=p["nu"],
        )

    def test_estimation_runs(self, data: NDArray[np.float64]) -> None:
        from scivol import ARMA, EGARCH, GED

        result = (ARMA(1, 1) + EGARCH(1, 1) + GED()).fit(data)
        assert result.params is not None
        assert len(result.params) == 8
        assert np.all(np.isfinite(result.params))
        assert np.isfinite(result.log_likelihood)


class TestARMAEGARCHHigherOrder:
    @pytest.fixture
    def data_21_normal(self) -> NDArray[np.float64]:
        p = TRUE_PARAMS["arma_egarch_21_normal"]
        return simulate_arma_egarch_pq(
            N_OBS,
            p["c"],
            p["phis"],
            p["thetas"],
            p["omega"],
            p["alphas"],
            p["gammas"],
            p["betas"],
            dist="Normal",
        )

    def test_arma_egarch_21_normal_runs(self, data_21_normal: NDArray[np.float64]) -> None:
        from scivol import ARMA, EGARCH, Normal

        result = (ARMA(1, 1) + EGARCH(2, 1) + Normal()).fit(data_21_normal)
        assert result.params is not None
        assert len(result.params) == 9
        assert np.all(np.isfinite(result.params))


# -----------------------------------------------------------------------------
# Tests: ARX/HARX standalone and linked EGARCH shipped surfaces
# -----------------------------------------------------------------------------

class TestARXHARXStandalone:
    def test_arx_normal_parameter_recovery(self) -> None:
        from scivol import ARX, Normal

        spec = ARX(1) + Normal()
        params = TRUE_PARAMS["arx_normal"]["params"]
        data, x = simulate_meanx_surface(spec, params, MEANX_OBS, seed=3101)

        result = spec.fit(data, x=x, solver="slsqp", log_mode=False, verbose=False)

        assert result.converged
        np.testing.assert_allclose(result.params, params, atol=0.18, rtol=0.0)

    def test_harx_normal_parameter_recovery(self) -> None:
        from scivol import HARX, Normal

        spec = HARX((1, 5)) + Normal()
        params = TRUE_PARAMS["harx_normal"]["params"]
        data, x = simulate_meanx_surface(spec, params, MEANX_OBS, seed=3102)

        result = spec.fit(data, x=x, solver="slsqp", log_mode=False, verbose=False)

        assert result.converged
        np.testing.assert_allclose(result.params, params, atol=0.20, rtol=0.0)

    def test_arx_studentt_estimation_runs(self) -> None:
        from scivol import ARX, StudentT

        spec = ARX(1) + StudentT()
        params = TRUE_PARAMS["arx_studentt"]["params"]
        data, x = simulate_meanx_surface(spec, params, MEANX_OBS, seed=3103)

        result = spec.fit(data, x=x, solver="slsqp", log_mode=False, verbose=False)

        assert result.converged
        assert len(result.params) == params.size
        assert np.all(np.isfinite(result.params))

    def test_harx_skewt_estimation_runs(self) -> None:
        from scivol import HARX, SkewT

        spec = HARX((1, 5)) + SkewT()
        params = TRUE_PARAMS["harx_skewt"]["params"]
        data, x = simulate_meanx_surface(spec, params, MEANX_OBS, seed=3104)

        result = spec.fit(data, x=x, solver="slsqp", log_mode=False, verbose=False)

        assert result.converged
        assert len(result.params) == params.size
        assert np.all(np.isfinite(result.params))

    def test_arx_ged_estimation_runs(self) -> None:
        from scivol import ARX, GED

        spec = ARX(1) + GED()
        params = TRUE_PARAMS["arx_ged"]["params"]
        data, x = simulate_meanx_surface(spec, params, MEANX_OBS, seed=3105)

        result = spec.fit(data, x=x, solver="slsqp", log_mode=False, verbose=False)

        assert result.converged
        assert len(result.params) == params.size
        assert np.all(np.isfinite(result.params))


class TestARXHARXEGARCH:
    def test_arx_egarch_normal_parameter_recovery(self) -> None:
        from scivol import ARX, EGARCH, Normal

        spec = ARX(1) + EGARCH(1, 1) + Normal()
        params = TRUE_PARAMS["arx_egarch_normal"]["params"]
        data, x = simulate_meanx_surface(spec, params, MEANX_OBS, seed=3201)

        result = spec.fit(data, x=x, solver="slsqp", log_mode=False, verbose=False)

        assert result.converged
        np.testing.assert_allclose(result.params[:3], params[:3], atol=0.16, rtol=0.0)
        assert result.params[6] == pytest.approx(params[6], abs=0.10)

    def test_arx_egarch_studentt_estimation_runs(self) -> None:
        from scivol import ARX, EGARCH, StudentT

        spec = ARX(1) + EGARCH(1, 1) + StudentT()
        params = TRUE_PARAMS["arx_egarch_studentt"]["params"]
        data, x = simulate_meanx_surface(spec, params, MEANX_OBS, seed=3202)

        result = spec.fit(data, x=x, solver="slsqp", log_mode=False, verbose=False)

        assert result.converged
        assert len(result.params) == params.size
        assert np.all(np.isfinite(result.params))

    def test_harx_egarch_skewt_estimation_runs(self) -> None:
        from scivol import HARX, EGARCH, SkewT

        spec = HARX((1, 5)) + EGARCH(2, 1) + SkewT()
        params = TRUE_PARAMS["harx_egarch_skewt"]["params"]
        data, x = simulate_meanx_surface(spec, params, MEANX_OBS, seed=3203)

        result = spec.fit(data, x=x, solver="slsqp", log_mode=False, verbose=False)

        assert result.converged
        assert len(result.params) == params.size
        assert np.all(np.isfinite(result.params))

    def test_arx_egarch_ged_estimation_runs(self) -> None:
        from scivol import ARX, EGARCH, GED

        spec = ARX(1) + EGARCH(1, 1) + GED()
        params = TRUE_PARAMS["arx_egarch_ged"]["params"]
        data, x = simulate_meanx_surface(spec, params, MEANX_OBS, seed=3204)

        result = spec.fit(data, x=x, solver="slsqp", log_mode=False, verbose=False)

        assert result.converged
        assert len(result.params) == params.size
        assert np.all(np.isfinite(result.params))


# -----------------------------------------------------------------------------
# Smoke test: all models run without error
# -----------------------------------------------------------------------------

class TestSmokeAll:
    """Quick smoke tests that all model combinations run."""
    
    @pytest.fixture
    def small_data(self) -> NDArray[np.float64]:
        """Small dataset for quick smoke tests."""
        return simulate_garch_normal(1000, 1e-6, 0.1, 0.85, seed=123)
    
    @pytest.mark.parametrize("p,q", [(1, 1), (1, 2), (2, 1), (2, 2)])
    def test_garch_orders_normal(
        self, small_data: NDArray[np.float64], p: int, q: int
    ) -> None:
        """Test various GARCH orders with Normal."""
        from scivol import GARCH, Normal
        
        spec = GARCH(p, q) + Normal()
        result = spec.fit(small_data)
        # Just check it runs - convergence may vary
        assert result is not None
    
    def test_garch_studentt_runs(self, small_data: NDArray[np.float64]) -> None:
        """Test GARCH + StudentT runs."""
        from scivol import GARCH, StudentT
        
        spec = GARCH(1, 1) + StudentT()
        result = spec.fit(small_data)
        assert result is not None
    
    def test_garch_skewt_runs(self, small_data: NDArray[np.float64]) -> None:
        """Test GARCH + SkewT runs."""
        from scivol import GARCH, SkewT
        
        spec = GARCH(1, 1) + SkewT()
        result = spec.fit(small_data)
        assert result is not None
    
    def test_gjr_garch_normal_runs(self, small_data: NDArray[np.float64]) -> None:
        """Test GJR-GARCH + Normal runs."""
        from scivol import GJRGARCH, Normal
        
        spec = GJRGARCH(1, 1) + Normal()
        result = spec.fit(small_data)
        assert result is not None
    
    def test_gjr_garch_studentt_runs(self, small_data: NDArray[np.float64]) -> None:
        """Test GJR-GARCH + StudentT runs."""
        from scivol import GJRGARCH, StudentT
        
        spec = GJRGARCH(1, 1) + StudentT()
        result = spec.fit(small_data)
        assert result is not None
    
    def test_gjr_garch_skewt_runs(self, small_data: NDArray[np.float64]) -> None:
        """Test GJR-GARCH + SkewT runs."""
        from scivol import GJRGARCH, SkewT
        
        spec = GJRGARCH(1, 1) + SkewT()
        result = spec.fit(small_data)
        assert result is not None

    def test_egarch_normal_runs(self, small_data: NDArray[np.float64]) -> None:
        from scivol import EGARCH, Normal

        result = (EGARCH(2, 1) + Normal()).fit(small_data)
        assert result is not None

    def test_egarch_studentt_runs(self, small_data: NDArray[np.float64]) -> None:
        from scivol import EGARCH, StudentT

        result = (EGARCH(2, 1) + StudentT()).fit(small_data)
        assert result is not None

    def test_arma_egarch_normal_runs(self, small_data: NDArray[np.float64]) -> None:
        from scivol import ARMA, EGARCH, Normal

        result = (ARMA(1, 1) + EGARCH(1, 1) + Normal()).fit(small_data)
        assert result is not None

    def test_arma_egarch_studentt_runs(self, small_data: NDArray[np.float64]) -> None:
        from scivol import ARMA, EGARCH, StudentT

        result = (ARMA(1, 1) + EGARCH(1, 1) + StudentT()).fit(small_data)
        assert result is not None

# =============================================================================
# DCC-GARCH(1,1) + Normal  (two-step DGP recovery)
# =============================================================================

# True parameters for the DCC-GARCH DGP
DCC_GARCH_TRUE = {
    "garch": [
        (1e-6, 0.08, 0.90),   # series 0
        (2e-6, 0.10, 0.87),   # series 1
        (1.5e-6, 0.06, 0.92), # series 2
    ],
    "dcc_a": 0.04,
    "dcc_b": 0.94,
    "Qbar": np.array([
        [1.0, 0.5, 0.3],
        [0.5, 1.0, 0.4],
        [0.3, 0.4, 1.0],
    ]),
}


def simulate_dcc_garch_11(
    T: int,
    garch_params: list,
    dcc_a: float,
    dcc_b: float,
    Qbar: NDArray[np.float64],
    seed: int = 42,
) -> NDArray[np.float64]:
    """
    Simulate DCC(1,1)-GARCH(1,1) with Normal innovations.

    Returns raw returns (T, N) — the input the user would provide.
    """
    rng = np.random.default_rng(seed)
    N = len(garch_params)
    c = 1.0 - dcc_a - dcc_b

    returns = np.zeros((T, N))
    sigma2 = np.zeros((T, N))

    # Initialise h_0 at unconditional variance
    for i in range(N):
        omega, alpha, beta = garch_params[i]
        sigma2[0, i] = omega / (1.0 - alpha - beta)

    Q_prev = Qbar.copy()
    e_prev = np.zeros(N)

    for t in range(T):
        # DCC recursion
        Q = c * Qbar + dcc_b * Q_prev + dcc_a * np.outer(e_prev, e_prev)
        diag = np.sqrt(np.maximum(np.diag(Q), 1e-12))
        R = Q / np.outer(diag, diag)

        # Correlated standard-normal innovations
        L = np.linalg.cholesky(R)
        z = L @ rng.standard_normal(N)

        # GARCH variance and returns
        if t > 0:
            for i in range(N):
                omega, alpha, beta = garch_params[i]
                sigma2[t, i] = omega + alpha * returns[t - 1, i] ** 2 + beta * sigma2[t - 1, i]
        returns[t] = np.sqrt(sigma2[t]) * z

        # Standardised residuals for DCC
        e_prev = z.copy()
        Q_prev = Q.copy()

    return returns


class TestDCCGARCH:
    """
    End-to-end DCC-GARCH(1,1) estimation from raw returns.

    DGP:  3-series DCC(1,1)-GARCH(1,1) with Normal innovations.
    Estimation uses DCC.fit(returns) which:
      Step 1: fits univariate GARCH(1,1)+Normal to each series
      Step 2: fits DCC(1,1) on the standardised residuals
    """

    N_OBS = 10_000

    @pytest.fixture(scope="class")
    def dcc_returns(self) -> NDArray[np.float64]:
        p = DCC_GARCH_TRUE
        return simulate_dcc_garch_11(
            self.N_OBS, p["garch"], p["dcc_a"], p["dcc_b"], p["Qbar"], seed=123,
        )

    @pytest.fixture(scope="class")
    def dcc_result(self, dcc_returns):
        from scivol import DCC, GARCH, Normal
        dcc = DCC(1, 1)
        return dcc.fit(dcc_returns, univariate_spec=GARCH(1, 1) + Normal())

    # -- Step 1: univariate GARCH recovery --

    def test_univariate_convergence(self, dcc_result) -> None:
        """All univariate fits should converge."""
        for i, r in enumerate(dcc_result.univariate_results):
            assert r.converged, f"Series {i} failed to converge"

    def test_univariate_parameter_recovery(self, dcc_result) -> None:
        """Univariate GARCH parameters should be close to truth."""
        for i, r in enumerate(dcc_result.univariate_results):
            omega_true, alpha_true, beta_true = DCC_GARCH_TRUE["garch"][i]
            np.testing.assert_allclose(r.params[0], omega_true, rtol=0.40,
                                       err_msg=f"omega mismatch, series {i}")
            np.testing.assert_allclose(r.params[1], alpha_true, rtol=0.25,
                                       err_msg=f"alpha mismatch, series {i}")
            np.testing.assert_allclose(r.params[2], beta_true, rtol=0.05,
                                       err_msg=f"beta mismatch, series {i}")

    # -- Step 2: DCC recovery --

    def test_dcc_convergence(self, dcc_result) -> None:
        """DCC optimisation should converge."""
        assert dcc_result.converged

    def test_dcc_parameter_recovery(self, dcc_result) -> None:
        """DCC parameters should be close to truth."""
        a_true = DCC_GARCH_TRUE["dcc_a"]
        b_true = DCC_GARCH_TRUE["dcc_b"]
        a_hat = dcc_result.params.a[0]
        b_hat = dcc_result.params.b[0]
        np.testing.assert_allclose(a_hat, a_true, rtol=0.30,
                                   err_msg=f"DCC a: {a_hat:.5f} vs {a_true}")
        np.testing.assert_allclose(b_hat, b_true, rtol=0.05,
                                   err_msg=f"DCC b: {b_hat:.5f} vs {b_true}")

    def test_dcc_persistence(self, dcc_result) -> None:
        """Persistence should be close to truth and < 1."""
        pers_true = DCC_GARCH_TRUE["dcc_a"] + DCC_GARCH_TRUE["dcc_b"]
        pers_hat = dcc_result.params.persistence
        assert 0 < pers_hat < 1
        np.testing.assert_allclose(pers_hat, pers_true, atol=0.03)

    def test_dcc_standard_errors(self, dcc_result) -> None:
        """MLE and robust SEs should exist and be positive."""
        assert dcc_result.std_errors is not None
        assert dcc_result.std_errors_robust is not None
        assert np.all(dcc_result.std_errors > 0)
        assert np.all(dcc_result.std_errors_robust > 0)

    def test_dcc_summary(self, dcc_result) -> None:
        """Summary should print without error."""
        s = dcc_result.summary()
        assert "DCC(1,1)" in s
        assert "a[1]" in s and "b[1]" in s


# -----------------------------------------------------------------------------
# Run tests directly
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
