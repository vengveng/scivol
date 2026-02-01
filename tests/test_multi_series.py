"""Tests for multi-series parallel fitting."""
import pytest
import numpy as np

# Check if pandas is available
try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False

from volkit import GARCH, Normal, StudentT, AutoDensity


@pytest.mark.skipif(not HAS_PANDAS, reason="pandas not installed")
class TestMultiSeriesFitting:
    """Test fitting multiple series in parallel."""
    
    @pytest.fixture
    def multi_series_df(self):
        dates = pd.date_range('2020-01-01', periods=500, freq='D')
        np.random.seed(42)
        return pd.DataFrame({
            'SPY': np.random.randn(500) * 0.01,
            'AAPL': np.random.randn(500) * 0.015,
            'GOOGL': np.random.randn(500) * 0.012,
        }, index=dates)
    
    def test_returns_dict(self, multi_series_df):
        """Multi-column DataFrame should return dict of results."""
        spec = GARCH(1, 1) + Normal()
        results = spec.fit(multi_series_df, n_jobs=1)
        
        assert isinstance(results, dict)
        assert set(results.keys()) == {'SPY', 'AAPL', 'GOOGL'}
    
    def test_each_result_valid(self, multi_series_df):
        """Each result in dict should be valid EstimationResult."""
        spec = GARCH(1, 1) + Normal()
        results = spec.fit(multi_series_df, n_jobs=1)
        
        for name, result in results.items():
            # Check we got a valid result with sigma2
            assert result.sigma2 is not None, f"No sigma2 for {name}"
            assert len(result.sigma2) == 500
            # Check log-likelihood is finite (optimizer may report success=False but still converge)
            assert np.isfinite(result.loglikelihood), f"Invalid LL for {name}"
    
    def test_index_preserved_multi(self, multi_series_df):
        """Each series result should have correct index."""
        spec = GARCH(1, 1) + Normal()
        results = spec.fit(multi_series_df, n_jobs=1)
        
        for name, result in results.items():
            pd.testing.assert_index_equal(
                result.sigma2.index, multi_series_df.index
            )
    
    def test_series_names(self, multi_series_df):
        """Each result should have correct series name."""
        spec = GARCH(1, 1) + Normal()
        results = spec.fit(multi_series_df, n_jobs=1)
        
        for name in ['SPY', 'AAPL', 'GOOGL']:
            assert results[name].series_name == name
    
    def test_n_jobs_one_sequential(self, multi_series_df):
        """n_jobs=1 should work (sequential fallback)."""
        spec = GARCH(1, 1) + Normal()
        results = spec.fit(multi_series_df, n_jobs=1)
        
        assert isinstance(results, dict)
        assert len(results) == 3


@pytest.mark.skipif(not HAS_PANDAS, reason="pandas not installed") 
class TestMultiSeriesParallel:
    """Test parallel execution with multiple workers."""
    
    @pytest.fixture
    def two_series_df(self):
        dates = pd.date_range('2020-01-01', periods=500, freq='D')
        np.random.seed(42)
        return pd.DataFrame({
            'A': np.random.randn(500) * 0.01,
            'B': np.random.randn(500) * 0.01,
        }, index=dates)
    
    def test_parallel_execution(self, two_series_df):
        """Parallel fitting should produce valid results."""
        spec = GARCH(1, 1) + Normal()
        results = spec.fit(two_series_df, n_jobs=2)
        
        assert isinstance(results, dict)
        assert len(results) == 2
        for name, result in results.items():
            # Check we got valid results (sigma2 exists and LL is finite)
            assert result.sigma2 is not None
            assert np.isfinite(result.loglikelihood)


class TestMultiSeriesNumpy:
    """Test multi-series with numpy arrays."""
    
    def test_2d_numpy_array(self):
        """2D numpy array should be treated as multi-series."""
        np.random.seed(42)
        data = np.random.randn(500, 3) * 0.01
        
        spec = GARCH(1, 1) + Normal()
        results = spec.fit(data, n_jobs=1)
        
        assert isinstance(results, dict)
        # Column names should be strings of integers
        assert '0' in results
        assert '1' in results
        assert '2' in results
    
    def test_2d_numpy_no_index(self):
        """2D numpy array results should have no index."""
        np.random.seed(42)
        data = np.random.randn(500, 2) * 0.01
        
        spec = GARCH(1, 1) + Normal()
        results = spec.fit(data, n_jobs=1)
        
        for name, result in results.items():
            assert result.index is None
            # sigma2 should be numpy array
            assert isinstance(result.sigma2, np.ndarray)


@pytest.mark.skipif(not HAS_PANDAS, reason="pandas not installed")
class TestMultiSeriesAuto:
    """Test auto-selection with multiple series."""
    
    @pytest.fixture
    def two_series_df(self):
        dates = pd.date_range('2020-01-01', periods=500, freq='D')
        np.random.seed(42)
        return pd.DataFrame({
            'A': np.random.randn(500) * 0.01,
            'B': np.random.randn(500) * 0.01,
        }, index=dates)
    
    def test_auto_with_multi_series(self, two_series_df):
        """Auto selection should work with multiple series."""
        spec = GARCH(auto={'max_p': 2, 'max_q': 1}) + Normal()
        results = spec.fit(two_series_df, n_jobs=1)
        
        assert isinstance(results, dict)
        assert len(results) == 2
        
        # Both should have results with selection info
        for name, result in results.items():
            assert hasattr(result, '_selection_candidates')
