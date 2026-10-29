"""
Tests for the `.fit()` convenience method injected by FitsMixin.

We patch the kernel routine at run-time so the test does not depend on
SciPy, the optimisation routine or the C kernels.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest

from scivol import GARCH, ARMA, Role, CompositeSpec
from scivol.result import FitMetadata


class DummyResult:  # simple stand-in for EstimationResult
    def __init__(self, spec: CompositeSpec, data_len: int, kwargs):
        self.spec = spec
        self.n_obs = data_len
        self.data = np.zeros(data_len)
        self.kwargs = kwargs
        self.fit_info = FitMetadata()
        self._index = None
        self._name = None


# ------------------------------------------------------------------ #
# helper fixtures
# ------------------------------------------------------------------ #
FIXED = {"x", "hold_back", "scale", "method", "index", "name"}


@pytest.fixture
def monkeypatched_routine(monkeypatch):
    """Replace the native fit with a stub that records its call and returns a DummyResult."""
    calls = {}

    def fake_fit(spec, y, **kw):
        calls["spec"] = spec
        calls["y"] = y
        calls["kw"] = {k: v for k, v in kw.items() if k not in FIXED}
        calls["method"] = kw["method"]
        return DummyResult(spec, len(y), kw)

    monkeypatch.setattr("scivol._fit.fit_spec", fake_fit)
    return calls


@pytest.fixture
def sample_data():
    return np.arange(20.0)


# ------------------------------------------------------------------ #
# 1. default method path (method='mle')
# ------------------------------------------------------------------ #
def test_fit_default_method(monkeypatched_routine, sample_data):
    model = GARCH(1, 1)

    res = model.fit(sample_data)  # uses default MLE path

    # The monkey-patched routine should have been called
    np.testing.assert_array_equal(monkeypatched_routine["y"], sample_data)
    assert monkeypatched_routine["kw"] == {} and monkeypatched_routine["method"] == "mle"
    assert isinstance(res, DummyResult)
    assert res.n_obs == len(sample_data)


# ------------------------------------------------------------------ #
# 2. kwargs pass through to the kernel routine
# ------------------------------------------------------------------ #
def test_fit_passes_kwargs(monkeypatched_routine, sample_data):
    model = GARCH(1, 1)

    res = model.fit(sample_data, foo="bar")

    assert monkeypatched_routine["kw"] == {"foo": "bar"}
    assert isinstance(res, DummyResult)


def test_small_sample_warning_is_recorded_not_emitted(monkeypatched_routine):
    model = GARCH(1, 1)
    data = np.arange(10.0)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        res = model.fit(data)

    runtime_warnings = [w for w in caught if issubclass(w.category, RuntimeWarning)]
    assert runtime_warnings == []
    assert res.fit_info.warnings == ("10 obs for 3 parameters; recommend >15.",)


# ------------------------------------------------------------------ #
# 3. .fit() works on a CompositeSpec directly
# ------------------------------------------------------------------ #
def test_fit_on_composite_spec(monkeypatched_routine, sample_data):
    spec = GARCH(1, 1) + ARMA(1, 0)  # CompositeSpec

    res = spec.fit(sample_data, answer=42)

    assert monkeypatched_routine["kw"] == {"answer": 42}
    assert isinstance(res, DummyResult)

    # CompositeSpec has a volatility component
    assert spec.get_component(Role.VOLATILITY) is not None


# ------------------------------------------------------------------ #
# 4. method='qmle' dispatches to the QMLE path
# ------------------------------------------------------------------ #
def test_fit_qmle_dispatch(monkeypatched_routine, sample_data):
    """method='qmle' reaches the native fit as its estimator."""
    model = GARCH(1, 1)
    res = model.fit(sample_data, method="qmle")

    assert monkeypatched_routine["spec"] == model.spec
    assert monkeypatched_routine["method"] == "qmle"
    assert isinstance(res, DummyResult)


# ------------------------------------------------------------------ #
# 5. method='mle' does not touch the QMLE path
# ------------------------------------------------------------------ #
def test_fit_mle_no_qmle(monkeypatched_routine, sample_data):
    """method='mle' should use the kernel routine, never QMLE."""
    model = GARCH(1, 1)

    res = model.fit(sample_data, method="mle")

    # Kernel routine was called
    np.testing.assert_array_equal(monkeypatched_routine["y"], sample_data)
    assert isinstance(res, DummyResult)
