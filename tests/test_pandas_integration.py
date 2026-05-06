"""Tests for pandas input/output integration."""
import pytest
import numpy as np

# Check if pandas is available
try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False

from scivol import GARCH, Normal, StudentT


@pytest.mark.skipif(not HAS_PANDAS, reason="pandas not installed")
class TestPandasSeries:
    """Test pandas Series input handling."""
    
    @pytest.fixture
    def sample_series(self):
        dates = pd.date_range('2020-01-01', periods=500, freq='D')
        rng = np.random.default_rng(42)
        return pd.Series(rng.standard_normal(500) * 0.01, index=dates, name='returns')
    
    def test_series_index_preserved(self, sample_series):
        """sigma2 and std_resid should have same index as input."""
        spec = GARCH(1, 1) + Normal()
        result = spec.fit(sample_series, n_jobs=1)
        
        assert isinstance(result.sigma2, pd.Series)
        pd.testing.assert_index_equal(result.sigma2.index, sample_series.index)
    
    def test_series_name_in_output(self, sample_series):
        """Output series should include original name."""
        spec = GARCH(1, 1) + Normal()
        result = spec.fit(sample_series, n_jobs=1)
        
        assert 'returns' in result.sigma2.name
    
    def test_std_resid_pandas(self, sample_series):
        """std_resid should also return pandas."""
        spec = GARCH(1, 1) + Normal()
        result = spec.fit(sample_series, n_jobs=1)
        
        assert isinstance(result.std_resid, pd.Series)
        pd.testing.assert_index_equal(result.std_resid.index, sample_series.index)
    
    def test_index_property(self, sample_series):
        """result.index should return the original index."""
        spec = GARCH(1, 1) + Normal()
        result = spec.fit(sample_series, n_jobs=1)
        
        assert result.index is not None
        pd.testing.assert_index_equal(result.index, sample_series.index)
    
    def test_series_name_property(self, sample_series):
        """result.series_name should return the original name."""
        spec = GARCH(1, 1) + Normal()
        result = spec.fit(sample_series, n_jobs=1)
        
        assert result.series_name == 'returns'


@pytest.mark.skipif(not HAS_PANDAS, reason="pandas not installed")
class TestPandasDataFrame:
    """Test pandas DataFrame input handling."""
    
    @pytest.fixture
    def single_col_df(self):
        dates = pd.date_range('2020-01-01', periods=500, freq='D')
        rng = np.random.default_rng(42)
        return pd.DataFrame({'SPY': rng.standard_normal(500) * 0.01}, index=dates)
    
    def test_single_column_df(self, single_col_df):
        """Single-column DataFrame should work like Series."""
        spec = GARCH(1, 1) + Normal()
        result = spec.fit(single_col_df, n_jobs=1)
        
        # Should return single result, not dict
        assert hasattr(result, 'sigma2')
        assert isinstance(result.sigma2, pd.Series)
    
    def test_single_column_df_index_preserved(self, single_col_df):
        """Single-column DataFrame should preserve index."""
        spec = GARCH(1, 1) + Normal()
        result = spec.fit(single_col_df, n_jobs=1)
        
        pd.testing.assert_index_equal(result.sigma2.index, single_col_df.index)
    
    def test_single_column_df_name(self, single_col_df):
        """Single-column DataFrame should use column name."""
        spec = GARCH(1, 1) + Normal()
        result = spec.fit(single_col_df, n_jobs=1)
        
        assert result.series_name == 'SPY'
        assert 'SPY' in result.sigma2.name


class TestNumpyInput:
    """Test that numpy input still works and returns numpy."""
    
    def test_numpy_returns_numpy(self):
        """numpy input should return numpy arrays."""
        rng = np.random.default_rng(42)
        data = rng.standard_normal(500) * 0.01
        
        spec = GARCH(1, 1) + Normal()
        result = spec.fit(data, n_jobs=1)
        
        assert isinstance(result.sigma2, np.ndarray)
        # Should not be a pandas Series
        if HAS_PANDAS:
            assert not isinstance(result.sigma2, pd.Series)
    
    def test_numpy_no_index(self):
        """numpy input should have None index."""
        rng = np.random.default_rng(42)
        data = rng.standard_normal(500) * 0.01
        
        spec = GARCH(1, 1) + Normal()
        result = spec.fit(data, n_jobs=1)
        
        assert result.index is None
        assert result.series_name is None


@pytest.mark.skipif(not HAS_PANDAS, reason="pandas not installed")
class TestDateTimeIndex:
    """Test different pandas index types."""
    
    def test_datetime_index(self):
        """DatetimeIndex should be preserved."""
        dates = pd.date_range('2020-01-01', periods=500, freq='D')
        rng = np.random.default_rng(42)
        series = pd.Series(rng.standard_normal(500) * 0.01, index=dates)
        
        spec = GARCH(1, 1) + Normal()
        result = spec.fit(series, n_jobs=1)
        
        assert isinstance(result.sigma2.index, pd.DatetimeIndex)
    
    def test_int_index(self):
        """Integer index should be preserved."""
        rng = np.random.default_rng(42)
        series = pd.Series(rng.standard_normal(500) * 0.01, index=range(100, 600))
        
        spec = GARCH(1, 1) + Normal()
        result = spec.fit(series, n_jobs=1)
        
        pd.testing.assert_index_equal(result.sigma2.index, series.index)
    
    def test_string_index(self):
        """String index should be preserved."""
        rng = np.random.default_rng(42)
        index = [f"obs_{i}" for i in range(500)]
        series = pd.Series(rng.standard_normal(500) * 0.01, index=index)
        
        spec = GARCH(1, 1) + Normal()
        result = spec.fit(series, n_jobs=1)
        
        pd.testing.assert_index_equal(result.sigma2.index, series.index)
