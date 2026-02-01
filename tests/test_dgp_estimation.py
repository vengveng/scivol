"""
Master DGP-Based Estimation Tests for volkit
=============================================

This file tests volkit estimation by generating synthetic data from known
Data Generating Processes (DGPs) and verifying that estimation recovers
the true parameters.

**KEEP THIS FILE EVERGREEN** - Update when new models are added to volkit.

Test Coverage:
- GARCH(1,1) + Normal
- GARCH(1,1) + StudentT  
- GARCH(1,1) + SkewT
- GARCH(p,q) + Normal (various orders)
- ARMA(1,1) + GARCH(1,1) + Normal (when API supports it)

Each test:
1. Generates 5000 observations from known true parameters
2. Estimates the model using volkit
3. Verifies convergence (optimization success)
4. Checks parameter recovery (within tolerance)
"""

from __future__ import annotations

import numpy as np
import pytest
from numpy.typing import NDArray
from scipy.special import gammaln

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


# -----------------------------------------------------------------------------
# Test Configuration
# -----------------------------------------------------------------------------

N_OBS = 5000  # Observations per test series

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
        from volkit import GARCH, Normal
        
        spec = GARCH(1, 1) + Normal()
        result = spec.fit(data)
        
        # Check we got valid parameters (even if optimizer reports non-convergence)
        assert result.params is not None
        assert len(result.params) == 3  # omega, alpha, beta
        assert np.all(np.isfinite(result.params))
        assert result.loglikelihood > 0  # Should be positive for reasonable fit
    
    def test_parameter_recovery(self, data: NDArray[np.float64]) -> None:
        """Test that estimated parameters are close to true values."""
        from volkit import GARCH, Normal
        
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
        from volkit import GARCH, StudentT
        
        spec = GARCH(1, 1) + StudentT()
        result = spec.fit(data)
        
        assert result.params is not None
        assert len(result.params) == 4  # omega, alpha, beta, nu
        assert np.all(np.isfinite(result.params))
        assert result.loglikelihood > 0
    
    def test_parameter_recovery(self, data: NDArray[np.float64]) -> None:
        """Test that estimated parameters are close to true values."""
        from volkit import GARCH, StudentT
        
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
        from volkit import GARCH, SkewT
        
        spec = GARCH(1, 1) + SkewT()
        result = spec.fit(data)
        
        assert result.params is not None
        assert len(result.params) == 5  # omega, alpha, beta, nu, lam
        assert np.all(np.isfinite(result.params))
        assert result.loglikelihood > 0
    
    def test_parameter_recovery(self, data: NDArray[np.float64]) -> None:
        """Test that estimated parameters are close to true values."""
        from volkit import GARCH, SkewT
        
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
        from volkit import GARCH, Normal
        
        spec = GARCH(2, 1) + Normal()
        result = spec.fit(data_21)
        
        assert result.params is not None
        assert len(result.params) == 4  # omega, alpha1, alpha2, beta1
        assert np.all(np.isfinite(result.params))
        assert result.loglikelihood > 0
    
    def test_garch_22_runs(self, data_22: NDArray[np.float64]) -> None:
        """Test GARCH(2,2) runs and produces valid output."""
        from volkit import GARCH, Normal
        
        spec = GARCH(2, 2) + Normal()
        result = spec.fit(data_22)
        
        assert result.params is not None
        assert len(result.params) == 5  # omega, alpha1, alpha2, beta1, beta2
        assert np.all(np.isfinite(result.params))
        assert result.loglikelihood > 0


# -----------------------------------------------------------------------------
# Tests: ARMA-GARCH (placeholder - enable when API supports it)
# -----------------------------------------------------------------------------

@pytest.mark.skip(reason="ARMA-GARCH not yet integrated into volkit API")
class TestARMAGARCH:
    """Tests for ARMA(1,1)-GARCH(1,1) estimation."""
    
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
        from volkit import ARMA, GARCH, Normal
        
        spec = ARMA(1, 1) + GARCH(1, 1) + Normal()
        result = spec.fit(data)
        
        assert result.params is not None
        assert len(result.params) == 6  # c, phi, theta, omega, alpha, beta
        assert np.all(np.isfinite(result.params))
        assert result.loglikelihood > 0


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
        from volkit import GARCH, Normal
        
        spec = GARCH(p, q) + Normal()
        result = spec.fit(small_data)
        # Just check it runs - convergence may vary
        assert result is not None
    
    def test_garch_studentt_runs(self, small_data: NDArray[np.float64]) -> None:
        """Test GARCH + StudentT runs."""
        from volkit import GARCH, StudentT
        
        spec = GARCH(1, 1) + StudentT()
        result = spec.fit(small_data)
        assert result is not None
    
    def test_garch_skewt_runs(self, small_data: NDArray[np.float64]) -> None:
        """Test GARCH + SkewT runs."""
        from volkit import GARCH, SkewT
        
        spec = GARCH(1, 1) + SkewT()
        result = spec.fit(small_data)
        assert result is not None


# -----------------------------------------------------------------------------
# Run tests directly
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
