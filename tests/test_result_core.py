"""
Tests for scivol.result.EstimationResult

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

from scivol import ARMA, GARCH, StudentT, Role, CompositeSpec
from scivol.result import EstimationResult


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
    assert res.n_iter == opt.nit
    assert res.converged is True
    assert res.convergence_message == "converged"

    # log-likelihood sign convention
    assert res.log_likelihood == pytest.approx(-opt.fun)

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

    # high-level keys (using standardized names)
    assert {
        "model",
        "log_likelihood",
        "aic",
        "bic",
        "hqic",
        "n_obs",
        "n_params",
        "converged",
        "n_iter",
        "parameters",
    } <= d.keys()

    # parameters sub-dict contains one entry per component
    assert len(d["parameters"]) == len(spec.components)
    for comp in spec.components:
        assert comp.signature in d["parameters"]
        assert d["parameters"][comp.signature] == comp.fitted_params


def test_component_params_are_snapshotted():
    data = np.ones(50)
    spec = _build_fitted_spec()
    opt = DummyOpt(x=np.arange(spec.total_params), fun=10.0)
    res = EstimationResult(spec, opt, data)

    gp_before = res.garch_params
    ap_before = res.arma_params
    dp_before = res.dist_params
    assert gp_before is not None
    assert ap_before is not None

    # Simulate later routine reuse mutating the original component instances.
    assert res.vol is not None
    assert res.mean is not None
    assert res.density is not None
    res.vol.unpack(np.array([0.02, 0.15, 0.70]))
    res.mean.unpack(np.array([0.10, -0.30, 0.40]))
    res.density.unpack(np.array([12.0]))

    gp_after = res.garch_params
    ap_after = res.arma_params
    dp_after = res.dist_params
    assert gp_after is not None
    assert ap_after is not None

    np.testing.assert_allclose(gp_after.to_array(), gp_before.to_array())
    np.testing.assert_allclose(ap_after.to_array(), ap_before.to_array())
    assert dp_after.nu == dp_before.nu

    d = res.to_dict()
    assert d["parameters"]["ARMA(1,1)"]["const"] == pytest.approx(0.05)
    assert d["parameters"]["GARCH(1,1)"]["omega"] == pytest.approx(0.01)
    assert d["dist_params"]["nu"] == pytest.approx(8.0)


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
    assert "Persistence" in captured  # Check for persistence info


# ------------------------------------------------------------------ #
# 4. forecast() method
# ------------------------------------------------------------------ #
def test_forecast_returns_dict():
    """forecast() should return dict with variance and volatility."""
    rng = np.random.default_rng(99)
    data = rng.standard_normal(100) * 0.01
    spec = _build_fitted_spec()
    # Create params with reasonable GARCH values
    params = np.array([1e-6, 0.1, 0.85, 8.0, 0.0, 0.0, 0.0])
    opt = DummyOpt(x=params, fun=10.0)
    res = EstimationResult(spec, opt, data)
    
    # Need to set sigma2 for forecast to work
    res._sigma2 = rng.random(len(data)) * 1e-4
    
    fc = res.forecast(horizon=5)
    
    assert isinstance(fc, dict)
    assert 'variance' in fc
    assert 'volatility' in fc
    assert len(fc['variance']) == 5
    assert len(fc['volatility']) == 5
    # Volatility should be sqrt of variance
    np.testing.assert_allclose(fc['volatility'], np.sqrt(fc['variance']))


def test_forecast_convergence():
    """Long-horizon forecast should converge to unconditional variance."""
    # Use simulated GARCH data for proper convergence test
    from scivol import GARCH, Normal
    
    rng = np.random.default_rng(42)
    n = 500
    omega, alpha, beta = 1e-6, 0.1, 0.85  # Known parameters
    
    # Simulate GARCH(1,1) process
    y = np.zeros(n)
    sigma2 = np.zeros(n)
    sigma2[0] = omega / (1 - alpha - beta)  # unconditional variance
    
    for t in range(1, n):
        sigma2[t] = omega + alpha * y[t-1]**2 + beta * sigma2[t-1]
        y[t] = np.sqrt(sigma2[t]) * rng.standard_normal()
    
    spec = GARCH(1, 1) + Normal()
    res = spec.fit(y, solver='slsqp', verbose=False)
    
    # Skip if no GARCH params
    gp = res.garch_params
    if gp is None:
        pytest.skip("No GARCH params available")
    
    # Skip if model is non-stationary (persistence >= 1)
    if gp.persistence >= 1.0:
        pytest.skip(f"Model non-stationary: persistence = {gp.persistence}")
    
    # Long horizon forecast
    fc = res.forecast(horizon=200)
    
    # Should converge to unconditional variance
    unconditional_var = gp.unconditional_variance
    
    # Check last forecast is close to unconditional (within 30%)
    # Loose tolerance due to estimation error
    rel_diff = abs(fc['variance'][-1] - unconditional_var) / unconditional_var
    assert rel_diff < 0.3, f"Forecast didn't converge: rel_diff = {rel_diff}"