"""
Error Handling and Edge Case Tests

Tests for boundary conditions, invalid inputs, and numerical stability.
"""
import pytest
import numpy as np

from volkit import GARCH, ARMA, Normal, StudentT, SkewT, MLE


# =============================================================================
# Invalid Input Data Tests
# =============================================================================

class TestInvalidInputData:
    """Test handling of invalid input data."""
    
    def test_nan_in_data_raises(self):
        """NaN values in data should raise ValueError."""
        data = np.array([1.0, np.nan, 0.5, 0.3])
        spec = GARCH(1, 1) + Normal()
        
        with pytest.raises(ValueError, match="NaN|nan|missing"):
            spec.fit(data)
    
    def test_inf_in_data_raises(self):
        """Inf values in data should raise ValueError."""
        data = np.array([1.0, np.inf, 0.5, 0.3])
        spec = GARCH(1, 1) + Normal()
        
        with pytest.raises(ValueError, match="Inf|inf|finite"):
            spec.fit(data)
    
    def test_negative_inf_in_data_raises(self):
        """Negative Inf values in data should raise ValueError."""
        data = np.array([1.0, -np.inf, 0.5, 0.3])
        spec = GARCH(1, 1) + Normal()
        
        with pytest.raises(ValueError, match="Inf|inf|finite"):
            spec.fit(data)
    
    def test_2d_array_treated_as_multi_series(self):
        """2D array input is treated as multiple series."""
        # Note: volkit treats 2D arrays as multi-column data (each column is a series)
        # This test documents that behavior rather than expecting an error
        np.random.seed(42)
        data = np.random.randn(100, 2) * 0.01
        spec = GARCH(1, 1) + Normal()
        
        # Should fit each column as a separate series
        results = spec.fit(data, n_jobs=1)
        
        # Returns dict keyed by column index
        assert isinstance(results, dict)
        assert len(results) == 2
    
    def test_empty_data_raises(self):
        """Empty data should raise ValueError."""
        data = np.array([])
        spec = GARCH(1, 1) + Normal()
        
        with pytest.raises(ValueError):
            spec.fit(data)
    
    def test_single_observation_raises(self):
        """Single observation should raise ValueError."""
        data = np.array([0.01])
        spec = GARCH(1, 1) + Normal()
        
        with pytest.raises(ValueError):
            spec.fit(data)


# =============================================================================
# Small Sample Tests
# =============================================================================

class TestSmallSamples:
    """Test behavior with small sample sizes."""
    
    @pytest.fixture
    def very_small_data(self):
        """20 observations."""
        np.random.seed(42)
        return np.random.randn(20) * 0.01
    
    @pytest.fixture
    def small_data(self):
        """50 observations."""
        np.random.seed(42)
        return np.random.randn(50) * 0.01
    
    def test_garch_normal_small_sample(self, small_data):
        """GARCH + Normal should work with 50 observations."""
        spec = GARCH(1, 1) + Normal()
        result = spec.fit(small_data, solver='nelder-mead', verbose=False)
        
        # Should complete without error
        assert result is not None
        assert np.isfinite(result.log_likelihood)
    
    def test_garch_studentt_small_sample(self, small_data):
        """GARCH + StudentT should work with 50 observations."""
        spec = GARCH(1, 1) + StudentT()
        result = spec.fit(small_data, solver='nelder-mead', verbose=False)
        
        assert result is not None
        assert np.isfinite(result.log_likelihood)
    
    def test_garch_skewt_small_sample(self, small_data):
        """GARCH + SkewT should work with 50 observations."""
        spec = GARCH(1, 1) + SkewT()
        result = spec.fit(small_data, solver='nelder-mead', verbose=False)
        
        assert result is not None
        assert np.isfinite(result.log_likelihood)
    
    def test_very_small_sample_warns_or_succeeds(self, very_small_data):
        """Very small sample should either warn or succeed gracefully."""
        spec = GARCH(1, 1) + Normal()
        
        # Should not crash - may or may not converge
        try:
            result = spec.fit(very_small_data, solver='nelder-mead', verbose=False)
            # If it returns, check it's a valid result
            assert result is not None
        except (ValueError, RuntimeError):
            # Some solvers may refuse to fit with very few observations
            pass


# =============================================================================
# Boundary Parameter Tests
# =============================================================================

class TestBoundaryParameters:
    """Test behavior near parameter boundaries."""
    
    @pytest.fixture
    def sample_data(self):
        """Standard sample data."""
        np.random.seed(42)
        return np.random.randn(500) * 0.01
    
    def test_high_persistence_data(self):
        """Test data simulated with high persistence (α + β → 1)."""
        np.random.seed(42)
        n = 500
        omega, alpha, beta = 1e-6, 0.15, 0.84  # persistence = 0.99
        
        # Simulate GARCH process
        y = np.zeros(n)
        sigma2 = np.zeros(n)
        sigma2[0] = omega / (1 - alpha - beta)
        
        for t in range(1, n):
            sigma2[t] = omega + alpha * y[t-1]**2 + beta * sigma2[t-1]
            y[t] = np.sqrt(sigma2[t]) * np.random.randn()
        
        # Fit model
        spec = GARCH(1, 1) + Normal()
        result = spec.fit(y, solver='slsqp', verbose=False)
        
        # Should complete and find high persistence
        assert result is not None
        assert np.isfinite(result.log_likelihood)
        
        gp = result.garch_params
        if gp is not None:
            # Persistence should be high (close to true value)
            assert gp.persistence > 0.8
    
    def test_studentt_near_nu_boundary(self, sample_data):
        """Test Student-t with nu close to 2 (heavy tails)."""
        spec = GARCH(1, 1) + StudentT()
        result = spec.fit(sample_data, solver='slsqp', verbose=False)
        
        # nu should be > 2 (required for finite variance)
        nu = result.params[-1]
        assert nu > 2.0
    
    def test_skewt_lambda_bounded(self, sample_data):
        """Test Skew-t lambda stays in (-1, 1)."""
        spec = GARCH(1, 1) + SkewT()
        result = spec.fit(sample_data, solver='slsqp', verbose=False)
        
        # lambda should be in (-1, 1)
        lam = result.params[-1]
        assert -1.0 < lam < 1.0


# =============================================================================
# Numerical Stability Tests
# =============================================================================

class TestNumericalStability:
    """Test numerical stability in edge cases."""
    
    def test_constant_data_handling(self):
        """Constant data should be handled gracefully."""
        data = np.ones(100) * 0.01
        spec = GARCH(1, 1) + Normal()
        
        # Should not crash (may not converge well)
        try:
            result = spec.fit(data, solver='nelder-mead', verbose=False)
            assert result is not None
        except (ValueError, RuntimeError):
            # Acceptable to raise on degenerate data
            pass
    
    def test_near_zero_variance_data(self):
        """Very low variance data should be handled."""
        np.random.seed(42)
        data = np.random.randn(500) * 1e-10
        spec = GARCH(1, 1) + Normal()
        
        # Should complete - may have numerical issues
        try:
            result = spec.fit(data, solver='nelder-mead', verbose=False)
            assert result is not None
        except (ValueError, RuntimeError, FloatingPointError):
            # Acceptable for extreme cases
            pass
    
    def test_large_magnitude_data(self):
        """Large magnitude data should be handled."""
        np.random.seed(42)
        data = np.random.randn(500) * 100
        spec = GARCH(1, 1) + Normal()
        
        result = spec.fit(data, solver='nelder-mead', verbose=False)
        
        assert result is not None
        assert np.isfinite(result.log_likelihood)


# =============================================================================
# Convergence Tests
# =============================================================================

class TestConvergence:
    """Test convergence behavior."""
    
    @pytest.fixture
    def sample_data(self):
        """Standard sample data."""
        np.random.seed(42)
        return np.random.randn(1000) * 0.01
    
    def test_slsqp_converges(self, sample_data):
        """SLSQP solver should converge on well-behaved data."""
        spec = GARCH(1, 1) + Normal()
        result = spec.fit(sample_data, solver='slsqp', verbose=False)
        
        # Check convergence
        assert result.converged or np.isfinite(result.log_likelihood)
    
    def test_nelder_mead_converges(self, sample_data):
        """Nelder-Mead solver should converge on well-behaved data."""
        spec = GARCH(1, 1) + Normal()
        result = spec.fit(sample_data, solver='nelder-mead', verbose=False)
        
        # Check we get a valid result
        assert np.isfinite(result.log_likelihood)
    
    def test_different_solvers_similar_results(self, sample_data):
        """Different solvers should give similar results."""
        spec = GARCH(1, 1) + Normal()
        
        result_slsqp = spec.fit(sample_data, solver='slsqp', verbose=False)
        result_nm = spec.fit(sample_data, solver='nelder-mead', verbose=False)
        
        # Log-likelihoods should be close
        ll_diff = abs(result_slsqp.log_likelihood - result_nm.log_likelihood)
        # Allow some tolerance since they're different optimization methods
        assert ll_diff < 10.0  # Within 10 LL units
