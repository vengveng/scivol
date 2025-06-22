from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from volkit import ARMA, GARCH, Role
from volkit.estimators import MLE
from volkit.result import EstimationResult
from volkit.components import Component
from volkit.spec import CompositeSpec


# ------------------------------------------------------------------ #
# helpers
# ------------------------------------------------------------------ #
@pytest.fixture
def dummy_kernel(monkeypatch):
    """
    Provide a kernel that just returns -123.  It lets us test the control
    flow without touching the true likelihood implementation.
    """
    def k(data: np.ndarray, params: np.ndarray, _spec: CompositeSpec | None = None):
        return -123.0

    # by default treat as "general"; individual tests can override
    monkeypatch.setattr("volkit.estimators.mle.get_general_kernel", lambda: k)
    monkeypatch.setattr("volkit.estimators.mle.get_special_kernel", lambda uid: None)
    return k


@pytest.fixture
def fake_minimize(monkeypatch):
    """
    Replace scipy.optimize.minimize with a stub that returns a static
    result object but still records the arguments.
    """
    calls = {}

    def _minimize(obj, x0, **kw):
        calls["x0"] = x0
        calls["kw"] = kw
        # produce a SciPy-like namespace
        k = SimpleNamespace(
            x=x0 * 0.0 + 0.5,     # dummy fitted parameters
            fun=obj(x0),          # objective at start
            success=True,
            nit=5,
            message="ok",
        )
        return k

    monkeypatch.setattr("volkit.estimators.mle.minimize", _minimize)
    return calls


# ------------------------------------------------------------------ #
# 1. _get_kernel selection logic
# ------------------------------------------------------------------ #
def test_get_kernel_prefers_special(monkeypatch, dummy_kernel):
    mle = MLE()

    # case A: special available
    monkeypatch.setattr(
        "volkit.estimators.mle.get_special_kernel", lambda uid: lambda d, p: -1.0
    )
    spec = CompositeSpec(GARCH(1, 1))
    k = mle._get_kernel(spec)
    assert k([], np.zeros(1)) == -1.0     # special chosen

    # case B: no special → fall back to general
    monkeypatch.setattr("volkit.estimators.mle.get_special_kernel", lambda uid: None)
    k2 = mle._get_kernel(spec)
    assert k2 is not k                     # different object
    assert k2([], np.zeros(1)) == -123.0   # value from dummy_kernel


# ------------------------------------------------------------------ #
# 2. End-to-end fit path (joint)
# ------------------------------------------------------------------ #
def test_mle_fit_returns_estimation_result(
    dummy_kernel, fake_minimize, monkeypatch
):
    data = np.random.standard_normal(30)
    # Component, not CompositeSpec → wrapper path exercised
    model = GARCH(1, 1)

    mle = MLE(max_iter=200, tol=1e-6)
    res = mle.fit(model, data)            # uses patched kernel/minimize

    # result object
    assert isinstance(res, EstimationResult)
    assert res.spec == model.spec
    assert res.n_obs == len(data)
    # estimator stored
    assert mle.last_result is res
    # components unpacked
    assert isinstance(model.fitted_params, dict)

    # minimize was called with correct start size
    p = model.n_params + 0  # + density params (0 for Normal)
    assert fake_minimize["x0"].shape[0] == p
    assert fake_minimize["kw"]["method"] == "Nelder-Mead"


# ------------------------------------------------------------------ #
# 3. Data validation guards
# ------------------------------------------------------------------ #
@pytest.mark.parametrize(
    "bad",
    [
        np.empty((2, 3)),                     # 2-D
        np.array([np.nan, 0.0, 1.0]),         # NaN
        np.array([np.inf, 0.0, 1.0]),         # inf
        np.array([0.0]),                      # too short (<2)
    ],
)
def test_validate_data_raises(bad):
    mle = MLE()
    with pytest.raises(ValueError):
        mle._validate_data(bad)