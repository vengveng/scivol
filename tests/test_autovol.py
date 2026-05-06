"""Tests for AutoVol component and vol-type-aware auto-selection.

These tests cover:
- AutoVol component construction and properties
- Composition with density components
- Candidate generation (cross-product of vol types × (p,q) ranges)
- Normalization of vol_candidates to (vol_type, p, q) tuples
- End-to-end fitting with AutoVol (kept fast via minimal search grids)

All tests use small data and restricted search grids so they complete quickly
(no @pytest.mark.slow needed).
"""
import pytest
import numpy as np
from scivol import (
    GARCH,
    GJRGARCH,
    AutoVol,
    AutoDensity,
    Normal,
    StudentT,
    SkewT,
    CompositeSpec,
    Role,
)


# ── Shared fixture ──────────────────────────────────────────────────
@pytest.fixture
def sample_data():
    """Small dataset for fast fitting tests."""
    rng = np.random.default_rng(42)
    return rng.standard_normal(400) * 0.01


# ====================================================================
# 1. Component construction
# ====================================================================
class TestAutoVolConstruction:
    """Unit tests for AutoVol component properties (no fitting)."""

    def test_default_construction(self):
        av = AutoVol()
        assert av.signature == "AutoVol"
        assert av._is_auto is True
        assert av.candidates == ['GARCH', 'GJRGARCH']
        assert av.max_p == 3
        assert av.max_q == 3
        assert av.role is Role.VOLATILITY

    def test_custom_candidates(self):
        av = AutoVol(candidates=['GJRGARCH'])
        assert av.candidates == ['GJRGARCH']

    def test_custom_ranges(self):
        av = AutoVol(max_p=2, max_q=1)
        assert av.max_p == 2
        assert av.max_q == 1

    def test_placeholder_methods(self):
        """AutoVol is a placeholder — n_params, bounds, etc. are empty."""
        av = AutoVol()
        assert av.n_params == 0
        assert av.bounds() == []
        assert len(av.default_start(np.array([]))) == 0
        assert av.pack({}).shape == (0,)
        assert av.unpack(np.array([])) == {}


# ====================================================================
# 2. Candidate generation
# ====================================================================
class TestCandidateGeneration:
    """Test that get_candidates() produces the correct cross-product."""

    def test_default_candidates(self):
        av = AutoVol()
        cands = av.get_candidates()
        # 2 vol types × 3 p values × 3 q values = 18
        assert len(cands) == 18
        # All tuples are (str, int, int)
        for vol_type, p, q in cands:
            assert isinstance(vol_type, str)
            assert isinstance(p, int)
            assert isinstance(q, int)

    def test_single_type_candidates(self):
        av = AutoVol(candidates=['GARCH'], max_p=2, max_q=2)
        cands = av.get_candidates()
        # 1 × 2 × 2 = 4
        assert len(cands) == 4
        assert all(vt == 'GARCH' for vt, _, _ in cands)

    def test_restricted_range(self):
        av = AutoVol(candidates=['GARCH', 'GJRGARCH'], max_p=1, max_q=1)
        cands = av.get_candidates()
        # 2 × 1 × 1 = 2
        assert len(cands) == 2
        assert cands == [('GARCH', 1, 1), ('GJRGARCH', 1, 1)]

    def test_order_is_type_then_p_then_q(self):
        av = AutoVol(candidates=['GARCH', 'GJRGARCH'], max_p=2, max_q=2)
        cands = av.get_candidates()
        # First block should be all GARCH, then all GJRGARCH
        garch_cands = [(vt, p, q) for vt, p, q in cands if vt == 'GARCH']
        gjr_cands = [(vt, p, q) for vt, p, q in cands if vt == 'GJRGARCH']
        assert len(garch_cands) == 4
        assert len(gjr_cands) == 4
        # Within GARCH block: p varies in outer loop, q in inner
        assert garch_cands[0] == ('GARCH', 1, 1)
        assert garch_cands[1] == ('GARCH', 1, 2)
        assert garch_cands[2] == ('GARCH', 2, 1)
        assert garch_cands[3] == ('GARCH', 2, 2)


# ====================================================================
# 3. Composition
# ====================================================================
class TestAutoVolComposition:
    """Test that AutoVol composes correctly with density components."""

    def test_autovol_plus_normal(self):
        spec = AutoVol() + Normal()
        assert str(spec) == "AutoVol+Normal"
        assert spec.get_component(Role.VOLATILITY).signature == "AutoVol"
        assert spec.get_component(Role.DENSITY).signature == "Normal"

    def test_autovol_auto_injects_normal(self):
        """Bare AutoVol should auto-inject Normal density."""
        av = AutoVol()
        spec = av.spec
        assert spec.get_component(Role.DENSITY) is not None
        assert spec.get_component(Role.DENSITY).signature == "Normal"

    def test_autovol_plus_studentt(self):
        spec = AutoVol() + StudentT()
        assert "AutoVol" in str(spec)
        assert "StudentT" in str(spec)

    def test_autovol_plus_autodensity(self):
        spec = AutoVol() + AutoDensity()
        assert str(spec) == "AutoVol+AutoDensity"

    def test_autovol_replaces_garch_raises(self):
        """Cannot combine AutoVol with a concrete GARCH (duplicate VOLATILITY role)."""
        with pytest.raises(ValueError, match="Multiple components with role"):
            AutoVol() + GARCH(1, 1)

    def test_autovol_has_auto_flag(self):
        spec = AutoVol() + Normal()
        # _has_auto_components should detect AutoVol
        from scivol._mixins import FitsMixin
        # CompositeSpec inherits FitsMixin
        assert spec._has_auto_components()


# ====================================================================
# 4. Normalization of vol candidates
# ====================================================================
class TestVolCandidateNormalization:
    """Test that _get_vol_candidates() returns normalized (vol_type, p, q) tuples
    for AutoVol, GARCH(auto=True), and GJRGARCH(auto=True)."""

    def test_autovol_returns_typed_tuples(self):
        spec = AutoVol(candidates=['GARCH', 'GJRGARCH'], max_p=1, max_q=1) + Normal()
        cands = spec._get_vol_candidates()
        assert cands == [('GARCH', 1, 1), ('GJRGARCH', 1, 1)]

    def test_garch_auto_normalizes(self):
        spec = GARCH(auto={'max_p': 2, 'max_q': 1}) + Normal()
        cands = spec._get_vol_candidates()
        expected = [('GARCH', 1, 1), ('GARCH', 2, 1)]
        assert cands == expected

    def test_gjrgarch_auto_normalizes(self):
        spec = GJRGARCH(auto={'max_p': 2, 'max_q': 1}) + Normal()
        cands = spec._get_vol_candidates()
        expected = [('GJRGARCH', 1, 1), ('GJRGARCH', 2, 1)]
        assert cands == expected

    def test_fixed_garch_normalizes(self):
        spec = GARCH(1, 1) + Normal()
        cands = spec._get_vol_candidates()
        assert cands == [('GARCH', 1, 1)]

    def test_fixed_gjrgarch_normalizes(self):
        spec = GJRGARCH(2, 1) + Normal()
        cands = spec._get_vol_candidates()
        assert cands == [('GJRGARCH', 2, 1)]


# ====================================================================
# 5. End-to-end fitting with AutoVol (fast, minimal grids)
# ====================================================================
class TestAutoVolFitting:
    """Integration tests: AutoVol actually selects and fits a model.

    Kept fast by using max_p=1, max_q=1 so only 2 candidates
    (GARCH(1,1) + GJRGARCH(1,1)) are fitted.
    """

    def test_autovol_fits_and_selects(self, sample_data):
        """AutoVol with minimal grid should fit and return a valid result."""
        spec = AutoVol(max_p=1, max_q=1) + Normal()
        result = spec.fit(sample_data, n_jobs=1)

        assert result.sigma2 is not None
        assert np.isfinite(result.log_likelihood)
        assert hasattr(result, '_selection_candidates')

    def test_autovol_candidate_count(self, sample_data):
        """Number of candidates should match the search grid."""
        spec = AutoVol(max_p=1, max_q=1) + Normal()
        result = spec.fit(sample_data, n_jobs=1)

        # 2 vol types × 1 × 1 = 2 candidates
        assert len(result._selection_candidates) == 2

    def test_autovol_candidates_sorted(self, sample_data):
        """Selection candidates should be sorted by score (best first)."""
        spec = AutoVol(max_p=1, max_q=1) + Normal()
        result = spec.fit(sample_data, n_jobs=1)

        scores = [c.score for c in result._selection_candidates]
        assert scores == sorted(scores)

    def test_autovol_selected_spec_is_concrete(self, sample_data):
        """The winning spec should be a concrete model, not AutoVol."""
        spec = AutoVol(max_p=1, max_q=1) + Normal()
        result = spec.fit(sample_data, n_jobs=1)

        sig = str(result.spec)
        assert "AutoVol" not in sig
        assert "GARCH" in sig  # Either "GARCH(1,1)" or "GJR-GARCH(1,1)"

    def test_autovol_single_candidate_garch_only(self, sample_data):
        """AutoVol with one type and one order produces one candidate."""
        spec = AutoVol(candidates=['GARCH'], max_p=1, max_q=1) + Normal()
        result = spec.fit(sample_data, n_jobs=1)

        assert len(result._selection_candidates) == 1
        assert "GARCH(1,1)" in str(result.spec)

    def test_autovol_single_candidate_gjrgarch_only(self, sample_data):
        """AutoVol with only GJRGARCH."""
        spec = AutoVol(candidates=['GJRGARCH'], max_p=1, max_q=1) + Normal()
        result = spec.fit(sample_data, n_jobs=1)

        assert len(result._selection_candidates) == 1
        assert "GJR-GARCH(1,1)" in str(result.spec)

    def test_autovol_with_studentt(self, sample_data):
        """AutoVol with a non-Normal fixed density."""
        spec = AutoVol(max_p=1, max_q=1) + StudentT()
        result = spec.fit(sample_data, n_jobs=1)

        assert np.isfinite(result.log_likelihood)
        assert "StudentT" in str(result.spec)
        assert len(result._selection_candidates) == 2


# ====================================================================
# 6. AutoVol + AutoDensity combined
# ====================================================================
class TestAutoVolWithAutoDensity:
    """Test the full-auto case: AutoVol() + AutoDensity()."""

    def test_combined_candidate_count(self, sample_data):
        """Full auto with minimal grid: 2 vol types × 1 pq × 2 densities = 4."""
        spec = (
            AutoVol(max_p=1, max_q=1)
            + AutoDensity(candidates=['Normal', 'StudentT'])
        )
        result = spec.fit(sample_data, n_jobs=1)

        assert len(result._selection_candidates) == 4

    def test_combined_selects_concrete(self, sample_data):
        """Result spec should have concrete vol and density."""
        spec = (
            AutoVol(max_p=1, max_q=1)
            + AutoDensity(candidates=['Normal', 'StudentT'])
        )
        result = spec.fit(sample_data, n_jobs=1)

        sig = str(result.spec)
        assert "AutoVol" not in sig
        assert "AutoDensity" not in sig
        assert "GARCH" in sig  # GARCH or GJR-GARCH

    def test_combined_has_valid_result(self, sample_data):
        spec = (
            AutoVol(max_p=1, max_q=1)
            + AutoDensity(candidates=['Normal', 'StudentT'])
        )
        result = spec.fit(sample_data, n_jobs=1)

        assert result.sigma2 is not None
        assert np.isfinite(result.log_likelihood)
        assert np.isfinite(result.aic)


# ====================================================================
# 7. Backward compatibility: existing GARCH(auto=True) still works
# ====================================================================
class TestBackwardCompatibility:
    """Verify that the (vol_type, p, q) refactor didn't break existing
    GARCH(auto=...) and GJRGARCH(auto=...) auto-selection."""

    def test_garch_auto_still_fits(self, sample_data):
        spec = GARCH(auto={'max_p': 2, 'max_q': 1}) + Normal()
        result = spec.fit(sample_data, n_jobs=1)

        assert np.isfinite(result.log_likelihood)
        assert len(result._selection_candidates) == 2
        assert "GARCH" in str(result.spec)

    def test_gjrgarch_auto_still_fits(self, sample_data):
        spec = GJRGARCH(auto={'max_p': 1, 'max_q': 1}) + Normal()
        result = spec.fit(sample_data, n_jobs=1)

        assert np.isfinite(result.log_likelihood)
        assert len(result._selection_candidates) == 1
        assert "GJR-GARCH" in str(result.spec)

    def test_garch_auto_with_autodensity(self, sample_data):
        spec = GARCH(auto={'max_p': 1, 'max_q': 1}) + AutoDensity(candidates=['Normal', 'StudentT'])
        result = spec.fit(sample_data, n_jobs=1)

        # 1 GARCH order × 2 densities = 2
        assert len(result._selection_candidates) == 2
        assert np.isfinite(result.log_likelihood)
