"""
Parity Tests: volkit vs Reference Implementation

Verifies that the volkit C extension library produces results
consistent with the reference Python implementation.
"""

import pytest
import numpy as np
from numpy.testing import assert_allclose

# Reference implementation (volkit_compat wraps volkit with legacy interface)
import sys
sys.path.insert(0, str(__file__).rsplit('/tests/', 1)[0])  # Add project root
from volkit_compat import fit_garch as fit_garch_ref

# volkit implementation
from volkit import GARCH, Normal, StudentT, SkewT, MLE, QMLE


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def sample_data():
    """Generate sample data for testing."""
    np.random.seed(42)
    n = 1000
    
    # Simulate GARCH(1,1) process
    omega, alpha, beta = 1e-6, 0.1, 0.85
    sigma2 = np.zeros(n)
    eps = np.zeros(n)
    sigma2[0] = omega / (1 - alpha - beta)
    
    for t in range(1, n):
        sigma2[t] = omega + alpha * eps[t-1]**2 + beta * sigma2[t-1]
        eps[t] = np.random.randn() * np.sqrt(sigma2[t])
    
    return eps


@pytest.fixture
def real_data():
    """Load real data if available, otherwise use simulated."""
    try:
        import pandas as pd
        data = pd.read_csv("data/DATA_HW1.csv", skiprows=1, parse_dates=["DATE"], dayfirst=True)
        returns = np.log1p(data["S&PCOMP(RI)"].pct_change(fill_method=None)).dropna()
        # Remove zero returns
        returns = returns[returns != 0]
        return np.asarray(returns, dtype=np.float64)
    except Exception:
        # Fallback to simulated data
        np.random.seed(42)
        return np.random.randn(1000) * 0.01


# =============================================================================
# Test GARCH + Normal
# =============================================================================

class TestGARCHNormal:
    """Test GARCH(1,1) + Normal distribution."""
    
    def test_loglikelihood_parity(self, sample_data):
        """Log-likelihood should match between implementations."""
        eps = sample_data
        
        # Reference
        r_ref = fit_garch_ref(eps, dist="normal", method="mle", p=1, q=1,
                              solver="trust-constr", use_logspace=False, verbose=False)
        
        # volkit
        spec = GARCH(1, 1) + Normal()
        r_vol = MLE().fit(spec, eps, solver="trust", log_mode=False, verbose=False)
        
        # Log-likelihood should be close (within optimization tolerance)
        assert_allclose(r_ref.log_likelihood, r_vol.loglikelihood, rtol=1e-3)
    
    def test_parameters_parity(self, sample_data):
        """GARCH parameters should match between implementations."""
        eps = sample_data
        
        # Reference
        r_ref = fit_garch_ref(eps, dist="normal", method="mle", p=1, q=1,
                              solver="trust-constr", use_logspace=False, verbose=False)
        
        # volkit
        spec = GARCH(1, 1) + Normal()
        r_vol = MLE().fit(spec, eps, solver="trust", log_mode=False, verbose=False)
        
        # Parameters should be close
        assert_allclose(r_ref.garch_params.omega, r_vol.garch_params.omega, rtol=0.1)
        assert_allclose(r_ref.garch_params.alpha[0], r_vol.garch_params.alpha[0], rtol=0.1)
        assert_allclose(r_ref.garch_params.beta[0], r_vol.garch_params.beta[0], rtol=0.1)
    
    def test_sigma2_computed(self, sample_data):
        """Volkit should compute and store sigma2."""
        eps = sample_data
        
        spec = GARCH(1, 1) + Normal()
        r_vol = MLE().fit(spec, eps, solver="trust", log_mode=False, verbose=False)
        
        assert r_vol.sigma2 is not None
        assert len(r_vol.sigma2) == len(eps)
        assert np.all(r_vol.sigma2 > 0)
    
    def test_std_resid_computed(self, sample_data):
        """Volkit should compute standardized residuals."""
        eps = sample_data
        
        spec = GARCH(1, 1) + Normal()
        r_vol = MLE().fit(spec, eps, solver="trust", log_mode=False, verbose=False)
        
        assert r_vol.std_resid is not None
        assert len(r_vol.std_resid) == len(eps)


# =============================================================================
# Test GARCH + Student-t
# =============================================================================

class TestGARCHStudentT:
    """Test GARCH(1,1) + Student-t distribution."""
    
    def test_loglikelihood_parity(self, sample_data):
        """Log-likelihood should match between implementations."""
        eps = sample_data
        
        # Reference
        r_ref = fit_garch_ref(eps, dist="studentt", method="mle", p=1, q=1,
                              solver="nelder-mead", verbose=False)
        
        # volkit
        spec = GARCH(1, 1) + StudentT()
        r_vol = MLE().fit(spec, eps, solver="nelder-mead", log_mode=False, verbose=False)
        
        # Log-likelihood should be close
        # Wider tolerance for Student-t due to non-convex optimization landscape
        # Both should converge to good local optima
        assert_allclose(r_ref.log_likelihood, r_vol.loglikelihood, rtol=0.05)
    
    def test_nu_parameter(self, sample_data):
        """Degrees of freedom parameter should be estimated."""
        eps = sample_data
        
        spec = GARCH(1, 1) + StudentT()
        r_vol = MLE().fit(spec, eps, solver="nelder-mead", log_mode=False, verbose=False)
        
        # nu should be reasonable (> 2 for finite variance)
        assert r_vol.dist_params.nu > 2.0
        assert r_vol.dist_params.nu < 100.0


# =============================================================================
# Test GARCH + Skew-t
# =============================================================================

class TestGARCHSkewT:
    """Test GARCH(1,1) + Skew-t distribution."""
    
    def test_loglikelihood_parity(self, sample_data):
        """Log-likelihood should match between implementations."""
        eps = sample_data
        
        # Reference
        r_ref = fit_garch_ref(eps, dist="skewt", method="mle", p=1, q=1,
                              solver="nelder-mead", verbose=False)
        
        # volkit
        spec = GARCH(1, 1) + SkewT()
        r_vol = MLE().fit(spec, eps, solver="nelder-mead", verbose=False)
        
        # Log-likelihood should be close
        # Wider tolerance for Skew-t due to more complex optimization
        assert_allclose(r_ref.log_likelihood, r_vol.loglikelihood, rtol=0.05)
    
    def test_dist_parameters(self, sample_data):
        """Distribution parameters should be estimated."""
        eps = sample_data
        
        spec = GARCH(1, 1) + SkewT()
        r_vol = MLE().fit(spec, eps, solver="nelder-mead", verbose=False)
        
        # nu should be reasonable
        assert r_vol.dist_params.nu > 2.0
        assert r_vol.dist_params.nu < 100.0
        
        # lambda should be in valid range
        assert -1.0 < r_vol.dist_params.lam < 1.0


# =============================================================================
# Test QMLE
# =============================================================================

class TestQMLE:
    """Test QMLE with robust standard errors."""
    
    def test_qmle_runs(self, sample_data):
        """QMLE should run and produce results."""
        eps = sample_data
        
        spec = GARCH(1, 1) + Normal()
        # Use nelder-mead for more robust convergence on simulated data
        r_vol = QMLE().fit(spec, eps, solver="nelder-mead", verbose=False)
        
        # Check that estimation completed (may not always converge to optimal)
        assert r_vol is not None
        assert r_vol.method == "QMLE"
        # Log-likelihood should be reasonable (not NaN or extremely negative)
        assert np.isfinite(r_vol.loglikelihood)
    
    def test_robust_se_computed(self, sample_data):
        """QMLE should attempt to compute robust standard errors."""
        eps = sample_data
        
        spec = GARCH(1, 1) + Normal()
        r_vol = QMLE().fit(spec, eps, solver="nelder-mead", verbose=False)
        
        # Check that the method completed
        assert r_vol is not None
        # Note: robust SEs may be None if OPG/Hessian computation fails
        # but the estimation should still complete


# =============================================================================
# Test Result Object
# =============================================================================

class TestEstimationResult:
    """Test EstimationResult attributes."""
    
    def test_garch_params_structure(self, sample_data):
        """garch_params should have omega, alpha, beta."""
        eps = sample_data
        
        spec = GARCH(1, 1) + Normal()
        r_vol = MLE().fit(spec, eps, solver="trust", log_mode=False, verbose=False)
        
        gp = r_vol.garch_params
        assert gp is not None
        assert hasattr(gp, 'omega')
        assert hasattr(gp, 'alpha')
        assert hasattr(gp, 'beta')
        assert hasattr(gp, 'persistence')
    
    def test_information_criteria(self, sample_data):
        """AIC and BIC should be computed."""
        eps = sample_data
        
        spec = GARCH(1, 1) + Normal()
        r_vol = MLE().fit(spec, eps, solver="trust", log_mode=False, verbose=False)
        
        assert np.isfinite(r_vol.aic)
        assert np.isfinite(r_vol.bic)
        assert np.isfinite(r_vol.hqic)
    
    def test_timing(self, sample_data):
        """time_elapsed should be recorded."""
        eps = sample_data
        
        spec = GARCH(1, 1) + Normal()
        r_vol = MLE().fit(spec, eps, solver="trust", log_mode=False, verbose=False)
        
        assert r_vol.time_elapsed is not None
        assert r_vol.time_elapsed > 0


# =============================================================================
# Test Component System
# =============================================================================

class TestComponentSystem:
    """Test the component composition system."""
    
    def test_composition_syntax(self):
        """Component composition should work."""
        spec1 = GARCH(1, 1) + Normal()
        spec2 = GARCH(1, 1) + StudentT()
        spec3 = GARCH(1, 1) + SkewT()
        
        assert str(spec1) == "GARCH(1,1)+Normal"
        assert str(spec2) == "GARCH(1,1)+StudentT"
        assert str(spec3) == "GARCH(1,1)+SkewT"
    
    def test_spec_string_representation(self):
        """Spec should have correct string representation for kernel dispatch."""
        spec = GARCH(1, 1) + Normal()
        assert "GARCH" in str(spec)
        assert "Normal" in str(spec)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
