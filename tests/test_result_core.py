"""
Tests for volkit.result.EstimationResult

These checks are *pure-Python* - they do **not** rely on the C kernels
or SciPy.  They only assume that:

* components expose .unpack() to set fitted_params
* EstimationResult follows the public API documented in the design note
"""

from __future__ import annotations

import re
from types import SimpleNamespace

import numpy as np
import pytest

from volkit import ARMA, GARCH, StudentT, Role, CompositeSpec
from volkit.result import EstimationResult


# ------------------------------------------------------------------ #
# helpers
# ------------------------------------------------------------------ #
class DummyOpt(SimpleNamespace):
    """
    A minimal object that quacks like scipy.optimize.OptimizeResult
    """

    def __init__(
        self,
        x: np.ndarray,
        fun: float,
        success: bool = True,
        nit: int = 7,
        message: str = "converged",
    ) -> None:
        super().__init__(x=x, fun=fun, success=success, nit=nit, message=message)


def _build_fitted_spec() -> CompositeSpec:
    """
    ARMA(1,1) mean + GARCH(1,1) vol + Student-t density,
    each with fake fitted parameters.
    """
    arma = ARMA(1, 1)
    garch = GARCH(1, 1)
    t_dist = StudentT()

    # inject mock parameter estimates
    arma.unpack(np.array([0.05, 0.20, -0.10]))          # const, ar1, ma1
    garch.unpack(np.array([0.01, 0.08, 0.90]))          # ω, α1, β1
    t_dist.unpack(np.array([8.0]))                      # df

    return arma + garch + t_dist


# ------------------------------------------------------------------ #
# 1. Scalar & shortcut properties
# ------------------------------------------------------------------ #
def test_scalar_properties():
    rng = np.random.default_rng(42)
    data = rng.standard_normal(250)

    spec = _build_fitted_spec()
    k = spec.total_params
    theta = np.linspace(0.1, 1.0, k)

    opt = DummyOpt(x=theta, fun=123.456)  # negative log-lik

    res = EstimationResult(spec, opt, data)

    # optimisation proxy
    assert res.params is opt.x
    assert res.niter == opt.nit
    assert res.success is True
    assert res.convergence_message == "converged"

    # log-likelihood sign convention
    assert res.loglikelihood == pytest.approx(-opt.fun)

    # information criteria
    expected_aic = 2 * k + 2 * opt.fun
    expected_bic = k * np.log(len(data)) + 2 * opt.fun
    expected_hqic = 2 * k * np.log(np.log(len(data))) + 2 * opt.fun

    assert res.aic == pytest.approx(expected_aic)
    assert res.bic == pytest.approx(expected_bic)
    assert res.hqic == pytest.approx(expected_hqic)

    # component shorthands
    assert res.vol.role is Role.VOLATILITY
    assert res.mean.role is Role.MEAN
    assert res.density.role is Role.DENSITY
    assert res.get_component(Role.MEAN) is res.mean
    assert res.has_component(Role.DENSITY) is True


# ------------------------------------------------------------------ #
# 2. to_dict output
# ------------------------------------------------------------------ #
def test_to_dict_structure():
    data = np.ones(50)
    spec = _build_fitted_spec()
    opt = DummyOpt(x=np.arange(spec.total_params), fun=10.0)
    res = EstimationResult(spec, opt, data)

    d = res.to_dict()

    # high-level keys
    assert {
        "model",
        "loglikelihood",
        "aic",
        "bic",
        "hqic",
        "n_obs",
        "n_params",
        "converged",
        "iterations",
        "parameters",
    } <= d.keys()

    # parameters sub-dict contains one entry per component
    assert len(d["parameters"]) == len(spec.components)
    for comp in spec.components:
        assert comp.signature in d["parameters"]
        assert d["parameters"][comp.signature] == comp.fitted_params


# ------------------------------------------------------------------ #
# 3. summary() produces human-readable text
# ------------------------------------------------------------------ #
def test_summary_prints(capsys):
    data = np.zeros(30)
    spec = _build_fitted_spec()
    opt = DummyOpt(x=np.zeros(spec.total_params), fun=0.0)
    res = EstimationResult(spec, opt, data)

    # should not raise and should write to stdout
    res.summary()
    captured = capsys.readouterr().out

    # very loose sanity checks
    assert str(spec) in captured
    assert re.search(r"Log-?Likelihood", captured, flags=re.I)
    assert "GARCH Persistence" in captured