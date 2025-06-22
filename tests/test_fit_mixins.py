"""
Tests for the `.fit()` convenience method injected by FitsMixin.

We patch volkit.estimators.mle.MLE.fit at run-time so the test
does not depend on SciPy, the optimisation routine or the C kernels.
"""

from __future__ import annotations

import numpy as np
import pytest

from volkit import GARCH, ARMA, Role, CompositeSpec
from volkit.estimators import MLE


class DummyResult:  # simple stand-in for EstimationResult
    def __init__(self, spec: CompositeSpec, data_len: int, kwargs):
        self.spec = spec
        self.n_obs = data_len
        self.kwargs = kwargs


# ------------------------------------------------------------------ #
# helper fixtures
# ------------------------------------------------------------------ #
@pytest.fixture
def monkeypatched_mle(monkeypatch):
    """
    Replace MLE.fit with a stub that records its call and returns
    a DummyResult instance.
    """
    calls = {}

    def fake_fit(self, spec, data, **kw):
        calls["self"] = self
        calls["spec"] = spec
        calls["data"] = data
        calls["kw"] = kw
        return DummyResult(spec, len(data), kw)

    monkeypatch.setattr(MLE, "fit", fake_fit, raising=True)
    return calls


@pytest.fixture
def sample_data():
    return np.arange(20.0)


# ------------------------------------------------------------------ #
# 1. default estimator path (estimator=None)
# ------------------------------------------------------------------ #
def test_fit_default_estimator(monkeypatched_mle, sample_data):
    model = GARCH(1, 1)

    res = model.fit(sample_data)  # uses default MLE

    # The monkey-patched function should have been called once
    assert monkeypatched_mle["spec"] == model.spec
    assert monkeypatched_mle["data"] is sample_data
    assert monkeypatched_mle["kw"] == {}
    assert isinstance(res, DummyResult)
    assert res.n_obs == len(sample_data)


# ------------------------------------------------------------------ #
# 2. estimator instance passed in
# ------------------------------------------------------------------ #
def test_fit_with_estimator_instance(monkeypatch, sample_data):
    calls = {}

    def fake_fit(self, spec, data, **kw):
        calls["self"] = self
        calls["spec"] = spec
        calls["data"] = data
        calls["kw"] = kw
        return DummyResult(spec, len(data), kw)

    est = MLE(max_iter=99)
    monkeypatch.setattr(MLE, "fit", fake_fit, raising=True)

    model = GARCH(1, 1)
    res = model.fit(sample_data, estimator=est, foo="bar")

    assert calls["self"] is est                         # same instance
    assert calls["spec"] == model.spec
    assert calls["kw"] == {"foo": "bar"}
    assert isinstance(res, DummyResult)


# ------------------------------------------------------------------ #
# 3. estimator class / factory passed in
# ------------------------------------------------------------------ #
def test_fit_with_estimator_class(monkeypatch, sample_data):
    ctor_called = {}
    calls = {}

    class DummyEstimator(MLE):
        def __init__(self):
            ctor_called["was_called"] = True

        def fit(self, spec, data, **kw):
            calls["spec"] = spec
            calls["data"] = data
            calls["kw"] = kw
            return DummyResult(spec, len(data), kw)

    spec = ARMA(1, 1) + GARCH(1, 1)

    res = spec.fit(sample_data, estimator=DummyEstimator, spam=123)

    assert ctor_called["was_called"]
    assert calls["spec"] == spec
    assert calls["kw"] == {"spam": 123}
    assert isinstance(res, DummyResult)


# ------------------------------------------------------------------ #
# 4. .fit() also works when called on a CompositeSpec directly
# ------------------------------------------------------------------ #
def test_fit_on_composite_spec(monkeypatched_mle, sample_data):
    spec = GARCH(1, 1) + ARMA(1, 0)  # CompositeSpec

    res = spec.fit(sample_data, answer=42)

    assert monkeypatched_mle["spec"] is spec
    assert monkeypatched_mle["kw"] == {"answer": 42}
    assert isinstance(res, DummyResult)

    # mix-in promised that spec attribute is itself
    assert spec.get_component(Role.VOLATILITY) is not None