# test/test_mle_core.py
from __future__ import annotations

from types import SimpleNamespace
from typing import Dict

import numpy as np
import pytest

from volkit import GARCH, ARMA, Role
from volkit.estimators import MLE
from volkit.result import EstimationResult
from volkit._kernels.routine import Routine


# ------------------------------------------------------------------ #
# helpers
# ------------------------------------------------------------------ #
def _dummy_result(spec, data_len, **kw):
    """Minimal EstimationResult stand-in."""
    return SimpleNamespace(spec=spec, n_obs=data_len, extra=kw)


@pytest.fixture
def dummy_routine(monkeypatch):
    """
    Register a fake Routine for two UIDs:
      - 'GARCH(1,1)+Normal'
      - 'ARMA(1,1)+GARCH(1,1)+Normal'
    The Routine.fit records its call and returns a DummyResult.
    """
    calls: Dict[str, dict] = {}

    def _make(uid):
        def _fit(y, **kw):
            calls["uid"] = uid
            calls["y"] = y
            calls["kw"] = kw
            # In a real routine you would build EstimationResult; for the
            # dispatcher test a dummy object is enough.
            return _dummy_result(uid, len(y), **kw)

        return Routine(uid=uid, n_params=3, fit=_fit)

    import volkit._kernels as _k
    monkeypatch.setitem(_k._ROUTINES, "GARCH(1,1)+Normal", _make("GARCH(1,1)+Normal"))
    monkeypatch.setitem(
        _k._ROUTINES,
        "ARMA(1,1)+GARCH(1,1)+Normal",
        _make("ARMA(1,1)+GARCH(1,1)+Normal"),
    )
    return calls


# ------------------------------------------------------------------ #
# 1. Dispatcher picks the correct routine
# ------------------------------------------------------------------ #
def test_mle_dispatches_to_routine(dummy_routine):
    data = np.random.standard_normal(30)
    model = GARCH(1, 1)

    mle = MLE()
    res = mle.fit(model, data, spam="eggs")   # kwargs forwarded

    assert mle.last_result is res
    assert dummy_routine["uid"] == "GARCH(1,1)+Normal"
    assert np.all(dummy_routine["y"] == data)
    assert dummy_routine["kw"] == {"spam": "eggs"}


# ------------------------------------------------------------------ #
# 2. Works when a CompositeSpec is supplied
# ------------------------------------------------------------------ #
def test_mle_composite_spec(dummy_routine):
    data = np.random.standard_normal(20)
    spec = ARMA(1, 1) + GARCH(1, 1)           # CompositeSpec

    res = MLE().fit(spec, data)

    assert res.n_obs == 20
    assert dummy_routine["uid"] == "ARMA(1,1)+GARCH(1,1)+Normal"


# ------------------------------------------------------------------ #
# 3. Data-validation still raises
# ------------------------------------------------------------------ #
@pytest.mark.parametrize(
    "bad",
    [
        np.empty((2, 3)),                     # 2-D
        np.array([np.nan, 0.0, 1.0]),         # NaN
        np.array([np.inf, 0.0, 1.0]),         # Inf
        np.array([0.0]),                      # too short
    ],
)
def test_validate_data_raises(bad):
    with pytest.raises(ValueError):
        MLE()._validate_data(bad)