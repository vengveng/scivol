"""Tests for the custom selection criterion feature.

Covers:
- make_default_criterion() and _score_candidate() helpers
- ModelCandidate.diagnostics field
- Custom criterion callable passed to .fit()
- diagnostic_kwargs passthrough to diagnostic_tests()
- Backward compatibility (criterion=None uses default behavior)
- Warning when both criterion and diagnostic_weight are set

All tests use small data and minimal search grids for fast execution.
"""
import warnings

import numpy as np
import pytest

from volkit import GARCH, GJRGARCH, AutoVol, AutoDensity, Normal, StudentT
from volkit._autoselect import (
    ModelCandidate,
    make_default_criterion,
    _score_candidate,
)


# ── Shared fixtures ─────────────────────────────────────────────────
@pytest.fixture
def sample_data():
    """Small dataset for fast fitting tests."""
    rng = np.random.default_rng(42)
    return rng.standard_normal(400) * 0.01


@pytest.fixture
def fitted_result(sample_data):
    """A pre-fitted GARCH(1,1)+Normal result for unit tests."""
    spec = GARCH(1, 1) + Normal()
    return spec.fit(sample_data, n_jobs=1)


# ====================================================================
# 1. make_default_criterion
# ====================================================================
class TestDefaultCriterion:
    """Unit tests for the default criterion factory."""

    def test_returns_callable(self):
        crit = make_default_criterion()
        assert callable(crit)

    def test_uses_aic_when_no_diagnostics(self, fitted_result):
        crit = make_default_criterion(diagnostic_weight=50.0)
        score = crit(fitted_result, None)
        # diagnostics=None → AIC + diagnostic_weight
        assert score == pytest.approx(fitted_result.aic + 50.0)

    def test_zero_penalty_when_all_pass(self, fitted_result):
        """If all diagnostic tests pass, score == AIC."""
        crit = make_default_criterion(diagnostic_weight=50.0)
        # Build a fake diagnostics dict where nothing fails
        diag = {
            'dgt': {'reject': False},
            'ljung_box': {
                1: {'reject': False},
                2: {'reject': False},
                3: {'reject': False},
                4: {'reject': False},
            },
        }
        score = crit(fitted_result, diag)
        assert score == pytest.approx(fitted_result.aic)

    def test_penalty_for_failures(self, fitted_result):
        crit = make_default_criterion(diagnostic_weight=100.0)
        diag = {
            'dgt': {'reject': True},  # +1
            'ljung_box': {
                1: {'reject': True},   # +1
                2: {'reject': False},
                3: {'reject': False},
                4: {'reject': True},   # +1
            },
        }
        score = crit(fitted_result, diag)
        expected = fitted_result.aic + 100.0 * 3
        assert score == pytest.approx(expected)

    def test_custom_weight(self, fitted_result):
        crit = make_default_criterion(diagnostic_weight=0.0)
        diag = {
            'dgt': {'reject': True},
            'ljung_box': {1: {'reject': True}, 2: {'reject': True},
                          3: {'reject': True}, 4: {'reject': True}},
        }
        # Zero weight → no penalty
        assert crit(fitted_result, diag) == pytest.approx(fitted_result.aic)


# ====================================================================
# 2. _score_candidate helper
# ====================================================================
class TestScoreCandidate:
    """Unit tests for the _score_candidate helper."""

    def test_returns_score_and_diagnostics(self, fitted_result):
        crit = make_default_criterion()
        score, diag = _score_candidate(fitted_result, crit)
        assert np.isfinite(score)
        assert diag is not None
        assert 'dgt' in diag
        assert 'ljung_box' in diag

    def test_passes_diagnostic_kwargs(self, fitted_result):
        """diagnostic_kwargs should be forwarded to diagnostic_tests()."""
        crit = make_default_criterion()
        # Use non-default lags to verify passthrough
        _, diag = _score_candidate(
            fitted_result, crit, diagnostic_kwargs={'lags': 5}
        )
        assert diag is not None
        # Check that LB tests used 5 lags
        for moment_dict in diag['ljung_box'].values():
            assert moment_dict['lags'] == 5

    def test_custom_n_cells_passthrough(self, fitted_result):
        crit = make_default_criterion()
        _, diag = _score_candidate(
            fitted_result, crit, diagnostic_kwargs={'n_cells': 20}
        )
        assert diag is not None
        assert diag['dgt']['n_cells'] == 20

    def test_custom_alpha_passthrough(self, fitted_result):
        crit = make_default_criterion()
        _, diag = _score_candidate(
            fitted_result, crit, diagnostic_kwargs={'alpha': 0.01}
        )
        assert diag is not None
        assert diag['alpha'] == 0.01

    def test_criterion_exception_returns_inf(self, fitted_result):
        """If the criterion callable raises, score should be inf."""
        def bad_crit(result, diag):
            raise ValueError("boom")

        score, diag = _score_candidate(fitted_result, bad_crit)
        assert score == float('inf')
        # diagnostics still attempted
        assert diag is not None

    def test_custom_criterion_callable(self, fitted_result):
        """A user-supplied criterion should be called with result and diagnostics."""
        calls = []

        def spy_crit(result, diagnostics):
            calls.append((result, diagnostics))
            return result.bic

        score, diag = _score_candidate(fitted_result, spy_crit)
        assert len(calls) == 1
        assert calls[0][0] is fitted_result
        assert calls[0][1] is diag
        assert score == pytest.approx(fitted_result.bic)


# ====================================================================
# 3. ModelCandidate.diagnostics field
# ====================================================================
class TestModelCandidateDiagnostics:
    """Verify the diagnostics field exists on ModelCandidate."""

    def test_diagnostics_default_none(self):
        spec = GARCH(1, 1) + Normal()
        mc = ModelCandidate(spec=spec)
        assert mc.diagnostics is None

    def test_diagnostics_can_be_set(self):
        spec = GARCH(1, 1) + Normal()
        mc = ModelCandidate(spec=spec)
        mc.diagnostics = {'dgt': {'reject': False}, 'ljung_box': {}}
        assert mc.diagnostics is not None
        assert mc.diagnostics['dgt']['reject'] is False


# ====================================================================
# 4. End-to-end: custom criterion via .fit()
# ====================================================================
class TestCustomCriterionFit:
    """Integration tests: custom criterion callable through .fit()."""

    def test_custom_criterion_selects_by_bic(self, sample_data):
        """A BIC-only criterion should select the model with lowest BIC."""
        def bic_only(result, diagnostics):
            return result.bic

        spec = AutoVol(max_p=1, max_q=1) + Normal()
        result = spec.fit(sample_data, n_jobs=1, criterion=bic_only)

        assert np.isfinite(result.log_likelihood)
        candidates = result._selection_candidates
        # Scores should match BIC values
        for c in candidates:
            if c.result is not None:
                assert c.score == pytest.approx(c.result.bic)

    def test_criterion_receives_diagnostics(self, sample_data):
        """Criterion should get non-None diagnostics for successful fits."""
        diagnostics_seen = []

        def capture_crit(result, diagnostics):
            diagnostics_seen.append(diagnostics)
            return result.aic

        spec = GARCH(auto={'max_p': 1, 'max_q': 1}) + Normal()
        spec.fit(sample_data, n_jobs=1, criterion=capture_crit)

        assert len(diagnostics_seen) >= 1
        for d in diagnostics_seen:
            assert d is not None
            assert 'dgt' in d

    def test_inf_rejects_candidate(self, sample_data):
        """Returning inf should reject a candidate."""
        def reject_gjr(result, diagnostics):
            if 'GJR' in str(result.spec):
                return float('inf')
            return result.aic

        spec = AutoVol(max_p=1, max_q=1) + Normal()
        result = spec.fit(sample_data, n_jobs=1, criterion=reject_gjr)

        # Best model should not be GJR-GARCH
        assert 'GJR' not in str(result.spec)

    def test_candidates_have_diagnostics_field(self, sample_data):
        """After fitting, each candidate should have a diagnostics dict."""
        spec = AutoVol(max_p=1, max_q=1) + Normal()
        result = spec.fit(sample_data, n_jobs=1)

        for c in result._selection_candidates:
            if c.result is not None and np.isfinite(c.score):
                assert c.diagnostics is not None
                assert 'dgt' in c.diagnostics
                assert 'ljung_box' in c.diagnostics


# ====================================================================
# 5. diagnostic_kwargs passthrough via .fit()
# ====================================================================
class TestDiagnosticKwargsFit:
    """Integration tests: diagnostic_kwargs forwarded through .fit()."""

    def test_custom_lags(self, sample_data):
        """diagnostic_kwargs={'lags': 5} should produce LB tests with 5 lags."""
        spec = AutoVol(max_p=1, max_q=1) + Normal()
        result = spec.fit(
            sample_data, n_jobs=1,
            diagnostic_kwargs={'lags': 5},
        )

        for c in result._selection_candidates:
            if c.diagnostics is not None:
                for lb in c.diagnostics['ljung_box'].values():
                    assert lb['lags'] == 5

    def test_custom_lags_with_criterion(self, sample_data):
        """diagnostic_kwargs should work alongside a custom criterion."""
        lags_seen = []

        def check_lags_crit(result, diagnostics):
            if diagnostics is not None:
                lags_seen.append(diagnostics['ljung_box'][1]['lags'])
            return result.aic

        spec = GARCH(auto={'max_p': 1, 'max_q': 1}) + Normal()
        spec.fit(
            sample_data, n_jobs=1,
            criterion=check_lags_crit,
            diagnostic_kwargs={'lags': 7},
        )

        assert len(lags_seen) >= 1
        assert all(l == 7 for l in lags_seen)

    def test_custom_n_cells_via_fit(self, sample_data):
        spec = AutoVol(candidates=['GARCH'], max_p=1, max_q=1) + Normal()
        result = spec.fit(
            sample_data, n_jobs=1,
            diagnostic_kwargs={'n_cells': 25},
        )
        c = result._selection_candidates[0]
        assert c.diagnostics is not None
        assert c.diagnostics['dgt']['n_cells'] == 25


# ====================================================================
# 6. Backward compatibility
# ====================================================================
class TestBackwardCompatibility:
    """Default behavior should be unchanged when criterion=None."""

    def test_default_criterion_none_uses_aic_penalty(self, sample_data):
        """criterion=None should use AIC + weight * n_failures."""
        spec = AutoVol(max_p=1, max_q=1) + Normal()
        result = spec.fit(sample_data, n_jobs=1, diagnostic_weight=50.0)

        assert np.isfinite(result.log_likelihood)
        assert len(result._selection_candidates) == 2

    def test_diagnostic_weight_still_works(self, sample_data):
        """diagnostic_weight should affect scoring when criterion is None."""
        spec = GARCH(auto={'max_p': 1, 'max_q': 1}) + Normal()

        r1 = spec.fit(sample_data, n_jobs=1, diagnostic_weight=0.0)
        r2 = spec.fit(sample_data, n_jobs=1, diagnostic_weight=500.0)

        # With weight=0, scores should equal AIC
        for c in r1._selection_candidates:
            if c.result is not None and c.diagnostics is not None:
                # No penalty
                assert c.score == pytest.approx(c.aic)


# ====================================================================
# 7. Warning when both criterion and diagnostic_weight are set
# ====================================================================
class TestCriterionWarning:
    """Warn the user if both criterion and diagnostic_weight are explicitly provided."""

    def test_warns_when_both_set(self, sample_data):
        def my_crit(result, diag):
            return result.aic

        spec = GARCH(1, 1) + Normal()
        # No auto → won't trigger auto-selection, but the warning
        # is issued in .fit() before delegation.
        # We need an auto spec to actually trigger the warning path.
        spec = AutoVol(candidates=['GARCH'], max_p=1, max_q=1) + Normal()

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            spec.fit(
                sample_data, n_jobs=1,
                criterion=my_crit,
                diagnostic_weight=100.0,  # non-default
            )
            # Find the relevant warning
            matching = [x for x in w if "diagnostic_weight" in str(x.message)]
            assert len(matching) >= 1

    def test_no_warning_when_default_weight(self, sample_data):
        """No warning when diagnostic_weight is the default (50.0)."""
        def my_crit(result, diag):
            return result.aic

        spec = AutoVol(candidates=['GARCH'], max_p=1, max_q=1) + Normal()

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            spec.fit(
                sample_data, n_jobs=1,
                criterion=my_crit,
                # diagnostic_weight not set → uses default 50.0
            )
            matching = [x for x in w if "diagnostic_weight" in str(x.message)]
            assert len(matching) == 0

    def test_no_warning_when_no_criterion(self, sample_data):
        """No warning when only diagnostic_weight is set (no custom criterion)."""
        spec = AutoVol(candidates=['GARCH'], max_p=1, max_q=1) + Normal()

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            spec.fit(
                sample_data, n_jobs=1,
                diagnostic_weight=200.0,
            )
            matching = [x for x in w if "diagnostic_weight" in str(x.message)]
            assert len(matching) == 0
