"""Tests for multi-series parallel fitting.

These tests are slow (4+ minutes) because they fit multiple models in parallel.
Run with: pytest tests/test_multi_series.py --run-slow
"""
import pytest
import numpy as np

# Mark all tests in this module as slow
pytestmark = pytest.mark.slow

# Check if pandas is available
try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False

from scivol import GARCH, Normal, StudentT


@pytest.mark.skipif(not HAS_PANDAS, reason="pandas not installed")
class TestMultiSeriesFitting:
    """Test fitting multiple series in parallel."""

    @pytest.fixture
    def multi_series_df(self):
        dates = pd.date_range('2020-01-01', periods=500, freq='D')
        rng = np.random.default_rng(42)
        return pd.DataFrame({
            'SPY': rng.standard_normal(500) * 0.01,
            'AAPL': rng.standard_normal(500) * 0.015,
            'GOOGL': rng.standard_normal(500) * 0.012,
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
            assert np.isfinite(result.log_likelihood), f"Invalid LL for {name}"

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

    def test_explicit_hold_back_preserves_effective_index(self, multi_series_df):
        spec = GARCH(1, 1) + Normal()
        results = spec.fit(multi_series_df, n_jobs=1, hold_back=7)

        for result in results.values():
            pd.testing.assert_index_equal(result.sigma2.index, multi_series_df.index[7:])
            assert result.fit_info.effective_hold_back == 7

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
        rng = np.random.default_rng(42)
        return pd.DataFrame({
            'A': rng.standard_normal(500) * 0.01,
            'B': rng.standard_normal(500) * 0.01,
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
            assert np.isfinite(result.log_likelihood)


class TestMultiSeriesNumpy:
    """Test multi-series with numpy arrays."""

    def test_2d_numpy_array(self):
        """2D numpy array should be treated as multi-series."""
        rng = np.random.default_rng(42)
        data = rng.standard_normal((500, 3)) * 0.01

        spec = GARCH(1, 1) + Normal()
        results = spec.fit(data, n_jobs=1)

        assert isinstance(results, dict)
        # Column names should be strings of integers
        assert '0' in results
        assert '1' in results
        assert '2' in results

    def test_2d_numpy_no_index(self):
        """2D numpy array results should have no index."""
        rng = np.random.default_rng(42)
        data = rng.standard_normal((500, 2)) * 0.01

        spec = GARCH(1, 1) + Normal()
        results = spec.fit(data, n_jobs=1)

        for name, result in results.items():
            assert result.index is None
            # sigma2 should be numpy array
            assert isinstance(result.sigma2, np.ndarray)
