# test/test_mle_core.py
from __future__ import annotations

from types import SimpleNamespace
from typing import Dict

import numpy as np
import pytest

from volkit import ARMA, GARCH
from volkit.estimators import MLE
from volkit.result import EstimationResult
from volkit._kernels.routine import Routine


# ------------------------------------------------------------------ #
# helper: build a dummy Routine object
# ------------------------------------------------------------------ #
def _make_dummy(uid: str, calls: Dict[str, dict]) -> Routine:
    def _fit(y, **kw):
        calls["uid"] = uid
        calls["y"] = y
        calls["kw"] = kw
        # simple stand-in that looks like EstimationResult
        return SimpleNamespace(spec=uid, n_obs=len(y))

    return Routine(uid=uid, n_params=1, fit=_fit)


# ------------------------------------------------------------------ #
# fixture: patch the registry for two UIDs
# ------------------------------------------------------------------ #
@pytest.fixture
def dummy_routines(monkeypatch):
    calls: Dict[str, dict] = {}

    import volkit._kernels as _k
    monkeypatch.setitem(
        _k._ROUTINES, "GARCH(1,1)+Normal",
        _make_dummy("GARCH(1,1)+Normal", calls),
    )
    monkeypatch.setitem(
        _k._ROUTINES, "ARMA(1,1)+GARCH(1,1)+Normal",
        _make_dummy("ARMA(1,1)+GARCH(1,1)+Normal", calls),
    )
    return calls


# ------------------------------------------------------------------ #
# 1. Dispatcher uses the routine registered for a Component spec
# ------------------------------------------------------------------ #
def test_mle_component_spec(dummy_routines):
    rng = np.random.default_rng(42)
    data = rng.standard_normal(30)
    model = GARCH(1, 1)

    res = MLE().fit(model, data, foo="bar")

    assert res.n_obs == 30
    assert dummy_routines["uid"] == "GARCH(1,1)+Normal"
    assert np.all(dummy_routines["y"] == data)
    assert dummy_routines["kw"] == {
        "foo": "bar",
        "solver": "trust",
        "log_mode": True,
        "verbose": False,
    }


# ------------------------------------------------------------------ #
# 2. Dispatcher works for a CompositeSpec
# ------------------------------------------------------------------ #
def test_mle_composite_spec(dummy_routines):
    rng = np.random.default_rng(42)
    data = rng.standard_normal(20)
    spec = ARMA(1, 1) + GARCH(1, 1)

    res = MLE().fit(spec, data)

    assert isinstance(res, SimpleNamespace)  # dummy result
    assert dummy_routines["uid"] == "ARMA(1,1)+GARCH(1,1)+Normal"


# ------------------------------------------------------------------ #
# 3. Data-validation still raises on bad input
# ------------------------------------------------------------------ #
@pytest.mark.parametrize(
    "bad",
    [
        np.empty((2, 4)),                # 2-D
        np.array([np.nan, 0.0]),         # NaN
        np.array([np.inf, 1.0]),         # Inf
        np.array([0.0]),                 # too short
    ],
)
def test_validate_data_raises(bad):
    with pytest.raises(ValueError):
        MLE()._validate_data(bad)