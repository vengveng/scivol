"""Tests for parallel auto-selection.

These tests are slow (4+ minutes) because they fit many models for auto-selection.
Run with: pytest tests/test_parallel_auto.py --run-slow
"""
import pytest
import numpy as np
from scivol import GARCH, Normal, AutoDensity

# Mark all tests in this module as slow
pytestmark = pytest.mark.slow


class TestParallelAutoSelection:
    """Test parallel candidate fitting in auto-selection."""
    
    @pytest.fixture
    def sample_data(self):
        rng = np.random.default_rng(42)
        return rng.standard_normal(500) * 0.01
    
    def test_sequential_auto_works(self, sample_data):
        """Auto-selection should work with n_jobs=1."""
        spec = GARCH(auto={'max_p': 2, 'max_q': 2}) + Normal()
        result = spec.fit(sample_data, n_jobs=1)
        
        # Check we got valid results
        assert result.sigma2 is not None
        assert np.isfinite(result.log_likelihood)
        assert hasattr(result, '_selection_candidates')
    
    def test_parallel_auto_works(self, sample_data):
        """Auto-selection should work with n_jobs > 1."""
        spec = GARCH(auto={'max_p': 2, 'max_q': 2}) + Normal()
        result = spec.fit(sample_data, n_jobs=2)
        
        # Check we got valid results
        assert result.sigma2 is not None
        assert np.isfinite(result.log_likelihood)
        assert hasattr(result, '_selection_candidates')
    
    def test_parallel_same_result_as_sequential(self, sample_data):
        """Parallel and sequential should select same best model."""
        spec = GARCH(auto={'max_p': 2, 'max_q': 2}) + Normal()
        
        result_seq = spec.fit(sample_data, n_jobs=1)
        result_par = spec.fit(sample_data, n_jobs=4)
        
        # Same best model selected (compare spec signatures)
        assert str(result_seq.spec) == str(result_par.spec)
    
    def test_auto_density_sequential(self, sample_data):
        """AutoDensity should work in sequential mode."""
        spec = GARCH(1, 1) + AutoDensity()
        result = spec.fit(sample_data, n_jobs=1)
        
        # Check we got valid results
        assert result.sigma2 is not None
        assert np.isfinite(result.log_likelihood)
    
    def test_candidates_count(self, sample_data):
        """Should have correct number of candidates."""
        spec = GARCH(auto={'max_p': 2, 'max_q': 2}) + Normal()
        result = spec.fit(sample_data, n_jobs=1)
        
        # 2x2 = 4 candidates
        assert len(result._selection_candidates) == 4
    
    def test_candidates_sorted_by_score(self, sample_data):
        """Candidates should be sorted by score (ascending)."""
        spec = GARCH(auto={'max_p': 2, 'max_q': 2}) + Normal()
        result = spec.fit(sample_data, n_jobs=1)
        
        candidates = result._selection_candidates
        scores = [c.score for c in candidates]
        assert scores == sorted(scores)


class TestEdgeCases:
    """Test edge cases and error handling."""
    
    def test_n_jobs_exceeds_tasks(self):
        """n_jobs > number of tasks should still work."""
        rng = np.random.default_rng(42)
        data = rng.standard_normal(500) * 0.01
        
        # Only 1 candidate (fixed spec)
        spec = GARCH(1, 1) + Normal()
        result = spec.fit(data, n_jobs=8)
        
        # Check we got valid results
        assert result.sigma2 is not None
        assert np.isfinite(result.log_likelihood)
    
    def test_default_n_jobs(self):
        """Default n_jobs (None) should work."""
        rng = np.random.default_rng(42)
        data = rng.standard_normal(500) * 0.01
        
        spec = GARCH(1, 1) + Normal()
        # n_jobs=None (default)
        result = spec.fit(data)
        
        # Check we got valid results
        assert result.sigma2 is not None
        assert np.isfinite(result.log_likelihood)
    
    def test_small_search_space(self):
        """Small search space (<=2 candidates) should use sequential."""
        rng = np.random.default_rng(42)
        data = rng.standard_normal(500) * 0.01
        
        spec = GARCH(auto={'max_p': 1, 'max_q': 2}) + Normal()
        result = spec.fit(data, n_jobs=4)
        
        # Should have 2 candidates (1x2)
        assert len(result._selection_candidates) == 2
        # Check we got valid results
        assert result.sigma2 is not None
        assert np.isfinite(result.log_likelihood)
    
    def test_verbose_parallel(self, capsys):
        """Verbose mode should print progress in parallel mode."""
        rng = np.random.default_rng(42)
        data = rng.standard_normal(500) * 0.01
        
        spec = GARCH(auto={'max_p': 2, 'max_q': 2}) + Normal()
        result = spec.fit(data, n_jobs=2, verbose_selection=True)
        
        captured = capsys.readouterr()
        assert "Auto-selecting" in captured.out
        assert "Best model" in captured.out
    
    def test_verbose_sequential(self, capsys):
        """Verbose mode should print progress in sequential mode."""
        rng = np.random.default_rng(42)
        data = rng.standard_normal(500) * 0.01
        
        spec = GARCH(auto={'max_p': 2, 'max_q': 1}) + Normal()
        result = spec.fit(data, n_jobs=1, verbose_selection=True)
        
        captured = capsys.readouterr()
        assert "Auto-selecting" in captured.out


class TestDifferentDistributions:
    """Test auto-selection with different distributions."""
    
    @pytest.fixture
    def sample_data(self):
        rng = np.random.default_rng(42)
        return rng.standard_normal(500) * 0.01
    
    def test_auto_density_candidates(self, sample_data):
        """AutoDensity should try all specified distributions."""
        spec = GARCH(1, 1) + AutoDensity(candidates=['Normal', 'StudentT'])
        result = spec.fit(sample_data, n_jobs=1)
        
        # Should have 2 candidates (1 GARCH spec x 2 densities)
        assert len(result._selection_candidates) == 2
    
    def test_full_auto(self, sample_data):
        """Full auto (lags + density) should work."""
        spec = GARCH(auto={'max_p': 2, 'max_q': 1}) + AutoDensity(candidates=['Normal', 'StudentT'])
        result = spec.fit(sample_data, n_jobs=1)
        
        # 2 p values x 1 q value x 2 densities = 4 candidates
        assert len(result._selection_candidates) == 4
