"""
Tests for the `.fit()` convenience method injected by FitsMixin.

We patch the kernel routine at run-time so the test does not depend on
SciPy, the optimisation routine or the C kernels.
"""

from __future__ import annotations

import numpy as np
import pytest

from scivol import GARCH, ARMA, Role, CompositeSpec
from scivol._kernels.routine import Routine


class DummyResult:  # simple stand-in for EstimationResult
    def __init__(self, spec: CompositeSpec, data_len: int, kwargs):
        self.spec = spec
        self.n_obs = data_len
        self.kwargs = kwargs
        self._index = None
        self._name = None


# ------------------------------------------------------------------ #
# helper fixtures
# ------------------------------------------------------------------ #
@pytest.fixture
def monkeypatched_routine(monkeypatch):
    """
    Replace the kernel routine with a stub that records its call and
    returns a DummyResult instance.
    """
    calls = {}

    def fake_fit(y, **kw):
        calls["y"] = y
        calls["kw"] = kw
        return DummyResult("GARCH(1,1)+Normal", len(y), kw)

    import scivol._kernels as _k
    dummy = Routine(uid="GARCH(1,1)+Normal", n_params=3, fit=fake_fit)
    monkeypatch.setitem(_k._ROUTINES, "GARCH(1,1)+Normal", dummy)

    # Also add ARMA combos
    for arma_uid in ["ARMA(1,1)+GARCH(1,1)+Normal", "ARMA(1,0)+GARCH(1,1)+Normal"]:
        def _make_arma_fit(uid):
            def fake_fit_arma(y, **kw):
                calls["y"] = y
                calls["kw"] = kw
                return DummyResult(uid, len(y), kw)
            return fake_fit_arma

        dummy_arma = Routine(uid=arma_uid, n_params=6, fit=_make_arma_fit(arma_uid))
        monkeypatch.setitem(_k._ROUTINES, arma_uid, dummy_arma)

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
    assert monkeypatched_routine["y"] is sample_data
    assert monkeypatched_routine["kw"] == {}
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
def test_fit_qmle_dispatch(monkeypatch, sample_data):
    """method='qmle' should call fit_qmle, not the kernel routine."""
    calls = {}

    def fake_fit_qmle(spec, data, **kw):
        calls["spec"] = spec
        calls["data"] = data
        calls["kw"] = kw
        return DummyResult(spec, len(data), kw)

    monkeypatch.setattr("scivol._qmle.fit_qmle", fake_fit_qmle)

    model = GARCH(1, 1)
    res = model.fit(sample_data, method="qmle", solver="slsqp")

    assert calls["spec"] == model.spec
    assert calls["data"] is sample_data
    assert calls["kw"] == {"solver": "slsqp"}
    assert isinstance(res, DummyResult)


# ------------------------------------------------------------------ #
# 5. method='mle' does not touch the QMLE path
# ------------------------------------------------------------------ #
def test_fit_mle_no_qmle(monkeypatched_routine, sample_data):
    """method='mle' should use the kernel routine, never QMLE."""
    model = GARCH(1, 1)

    res = model.fit(sample_data, method="mle")

    # Kernel routine was called
    assert monkeypatched_routine["y"] is sample_data
    assert isinstance(res, DummyResult)
