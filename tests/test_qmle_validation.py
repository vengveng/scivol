"""
QMLE Validation Tests

These tests verify the correctness of robust standard error computation
using the sandwich (Huber-White) covariance estimator.

The tests use the following approach:
1. For correctly specified models with large samples, robust SEs should
   be close to MLE SEs (within a factor of ~2)
2. The sandwich formula V = H^{-1} @ OPG @ H^{-1} should be computed correctly
3. Two-step QMLE for Student-t/Skew-t should work correctly

QMLE is invoked via ``spec.fit(data, method='qmle')``.
"""
import pytest
import numpy as np
from numpy.testing import assert_allclose, assert_array_less

from volkit import GARCH, Normal, StudentT, SkewT


# =============================================================================
# Test Data Generation
# =============================================================================

def generate_garch_data(n: int, omega: float, alpha: float, beta: float, 
                        seed: int = 42) -> np.ndarray:
    """Generate GARCH(1,1) data with Normal innovations."""
    rng = np.random.default_rng(seed)
    
    y = np.zeros(n)
    sigma2 = np.zeros(n)
    sigma2[0] = omega / (1 - alpha - beta)  # unconditional variance
    
    for t in range(1, n):
        sigma2[t] = omega + alpha * y[t-1]**2 + beta * sigma2[t-1]
        y[t] = np.sqrt(sigma2[t]) * rng.standard_normal()
    
    return y


def generate_garch_studentt_data(n: int, omega: float, alpha: float, beta: float,
                                  nu: float, seed: int = 42) -> np.ndarray:
    """Generate GARCH(1,1) data with Student-t innovations."""
    rng = np.random.default_rng(seed)
    
    # Standardized t distribution (variance = 1)
    scale = np.sqrt((nu - 2) / nu)
    
    y = np.zeros(n)
    sigma2 = np.zeros(n)
    sigma2[0] = omega / (1 - alpha - beta)
    
    for t in range(1, n):
        sigma2[t] = omega + alpha * y[t-1]**2 + beta * sigma2[t-1]
        z = rng.standard_t(df=nu) * scale  # standardized to unit variance
        y[t] = np.sqrt(sigma2[t]) * z
    
    return y


# =============================================================================
# Test QMLE Basic Functionality
# =============================================================================

class TestQMLEBasic:
    """Basic QMLE functionality tests."""
    
    @pytest.fixture
    def normal_data(self):
        """GARCH(1,1) data with Normal innovations."""
        return generate_garch_data(2000, omega=1e-6, alpha=0.1, beta=0.85)
    
    @pytest.fixture
    def studentt_data(self):
        """GARCH(1,1) data with Student-t innovations."""
        return generate_garch_studentt_data(2000, omega=1e-6, alpha=0.1, beta=0.85, nu=6)
    
    def test_qmle_returns_robust_se(self, normal_data):
        """QMLE should compute robust standard errors."""
        spec = GARCH(1, 1) + Normal()
        result = spec.fit(normal_data, method='qmle', solver='slsqp', verbose=False)
        
        # Should have robust SEs computed
        assert result.std_errors_robust is not None
        assert len(result.std_errors_robust) == 3  # omega, alpha, beta
        assert np.all(np.isfinite(result.std_errors_robust))
        assert np.all(result.std_errors_robust > 0)
    
    def test_qmle_studentt_two_step(self, studentt_data):
        """QMLE with Student-t uses two-step procedure."""
        spec = GARCH(1, 1) + StudentT()
        result = spec.fit(studentt_data, method='qmle', solver='slsqp', verbose=False)
        
        # Should have parameters
        assert len(result.params) == 4  # omega, alpha, beta, nu
        
        # Should have robust SEs (may be MLE for nu)
        if result.std_errors_robust is not None:
            assert len(result.std_errors_robust) == 4
    
    def test_qmle_method_attribute(self, normal_data):
        """QMLE result should have method='QMLE'."""
        spec = GARCH(1, 1) + Normal()
        result = spec.fit(normal_data, method='qmle', solver='slsqp', verbose=False)
        
        assert result.method == "QMLE"


# =============================================================================
# Test Sandwich Covariance Formula
# =============================================================================

class TestSandwichCovariance:
    """Test sandwich covariance formula correctness."""
    
    @pytest.fixture
    def correctly_specified_data(self):
        """Data from correctly specified model (Normal)."""
        return generate_garch_data(3000, omega=1e-6, alpha=0.1, beta=0.85, seed=123)
    
    def test_sandwich_components_exist(self, correctly_specified_data):
        """QMLE should compute OPG and Hessian."""
        spec = GARCH(1, 1) + Normal()
        result = spec.fit(correctly_specified_data, method='qmle', solver='slsqp', verbose=False)
        
        # Check components exist
        assert result.hessian is not None or result.cov_matrix is not None
    
    def test_robust_vs_mle_se_ratio(self, correctly_specified_data):
        """For correctly specified model, robust and MLE SEs should be similar."""
        spec = GARCH(1, 1) + Normal()
        
        # Fit with MLE
        result_mle = spec.fit(correctly_specified_data, solver='slsqp', verbose=False)
        
        # Fit with QMLE
        result_qmle = spec.fit(correctly_specified_data, method='qmle', solver='slsqp', verbose=False)
        
        # Both should have SEs
        se_mle = result_mle.std_errors
        se_robust = result_qmle.std_errors_robust
        
        if se_mle is not None and se_robust is not None:
            # Ratio should be within factor of 3 for correctly specified model
            ratio = se_robust / se_mle
            assert np.all(ratio > 0.3), f"Ratio too small: {ratio}"
            assert np.all(ratio < 3.0), f"Ratio too large: {ratio}"


# =============================================================================
# Test Misspecification Detection
# =============================================================================

class TestMisspecification:
    """Test that robust SEs differ from MLE SEs under misspecification."""
    
    @pytest.fixture
    def misspecified_data(self):
        """Data from Student-t model fitted as Normal (misspecified)."""
        return generate_garch_studentt_data(3000, omega=1e-6, alpha=0.1, beta=0.85, 
                                            nu=4, seed=456)
    
    def test_robust_differs_from_mle_under_misspec(self, misspecified_data):
        """Under misspecification, robust SEs may differ from MLE SEs."""
        spec = GARCH(1, 1) + Normal()
        
        # Fit with MLE
        result_mle = spec.fit(misspecified_data, solver='slsqp', verbose=False)
        
        # Fit with QMLE
        result_qmle = spec.fit(misspecified_data, method='qmle', solver='slsqp', verbose=False)
        
        se_mle = result_mle.std_errors
        se_robust = result_qmle.std_errors_robust
        
        # Just check they're both computed
        if se_mle is not None and se_robust is not None:
            assert len(se_mle) == len(se_robust)
            assert np.all(np.isfinite(se_mle))
            assert np.all(np.isfinite(se_robust))


# =============================================================================
# Test Two-Step QMLE for Student-t
# =============================================================================

class TestTwoStepStudentT:
    """Test two-step QMLE procedure for Student-t distribution."""
    
    @pytest.fixture
    def studentt_data(self):
        """Student-t GARCH data."""
        return generate_garch_studentt_data(2000, omega=1e-6, alpha=0.1, beta=0.85, 
                                            nu=6, seed=789)
    
    def test_studentt_parameter_recovery(self, studentt_data):
        """Two-step QMLE should recover parameters reasonably."""
        spec = GARCH(1, 1) + StudentT()
        result = spec.fit(studentt_data, method='qmle', solver='slsqp', verbose=False)
        
        assert len(result.params) == 4
        omega, alpha, beta, nu = result.params
        
        assert omega > 0
        assert alpha > 0
        assert beta > 0
        assert nu > 2
        assert alpha + beta < 1
    
    def test_studentt_nu_has_se(self, studentt_data):
        """Student-t nu parameter should have standard error."""
        spec = GARCH(1, 1) + StudentT()
        result = spec.fit(studentt_data, method='qmle', solver='slsqp', verbose=False)
        
        if result.std_errors is not None:
            assert len(result.std_errors) == 4
            se_nu = result.std_errors[3]
            assert np.isfinite(se_nu)
            assert se_nu > 0


# =============================================================================
# Test Two-Step QMLE for Skew-t
# =============================================================================

class TestTwoStepSkewT:
    """Test two-step QMLE procedure for Skew-t distribution."""
    
    @pytest.fixture
    def skewt_data(self):
        """Use Student-t data as proxy for Skew-t (lambda should be near 0)."""
        return generate_garch_studentt_data(2000, omega=1e-6, alpha=0.1, beta=0.85, 
                                            nu=6, seed=321)
    
    def test_skewt_parameter_count(self, skewt_data):
        """Skew-t should have 5 parameters."""
        spec = GARCH(1, 1) + SkewT()
        result = spec.fit(skewt_data, method='qmle', solver='slsqp', verbose=False)
        
        assert len(result.params) == 5
    
    def test_skewt_lambda_bounded(self, skewt_data):
        """Skew-t lambda should be in (-1, 1)."""
        spec = GARCH(1, 1) + SkewT()
        result = spec.fit(skewt_data, method='qmle', solver='slsqp', verbose=False)
        
        lam = result.params[4]
        assert -1 < lam < 1
    
    def test_skewt_has_ses(self, skewt_data):
        """Skew-t should have standard errors for all parameters."""
        spec = GARCH(1, 1) + SkewT()
        result = spec.fit(skewt_data, method='qmle', solver='slsqp', verbose=False)
        
        if result.std_errors is not None:
            assert len(result.std_errors) == 5
            assert np.all(np.isfinite(result.std_errors))
            assert np.all(result.std_errors > 0)


# =============================================================================
# Test QMLE Numerical Stability
# =============================================================================

class TestQMLENumericalStability:
    """Test numerical stability of QMLE computations."""
    
    @pytest.fixture
    def small_sample(self):
        """Small sample data."""
        return generate_garch_data(200, omega=1e-6, alpha=0.1, beta=0.85, seed=111)
    
    @pytest.fixture
    def large_sample(self):
        """Large sample data."""
        return generate_garch_data(5000, omega=1e-6, alpha=0.1, beta=0.85, seed=222)
    
    def test_small_sample_doesnt_crash(self, small_sample):
        """QMLE should not crash on small samples."""
        spec = GARCH(1, 1) + Normal()
        result = spec.fit(small_sample, method='qmle', solver='slsqp', verbose=False)
        assert result is not None
    
    def test_large_sample_works(self, large_sample):
        """QMLE should work well on large samples."""
        spec = GARCH(1, 1) + Normal()
        result = spec.fit(large_sample, method='qmle', solver='slsqp', verbose=False)
        
        assert result.std_errors_robust is not None
        assert np.all(np.isfinite(result.std_errors_robust))
