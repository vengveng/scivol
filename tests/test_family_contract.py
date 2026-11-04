"""The contract every volatility family and density must meet.

The cases come from the native enums, so a new family or density is tested
here as soon as it is registered, before any test of its own. Exact
derivatives are validated separately against the JAX oracles; the
finite-difference check below is only a coarse smoke test of the wiring.
"""

from __future__ import annotations

import itertools

import numpy as np
import pytest

import scivol
from scivol import _scivolapi as api
from scivol._fit import native_spec, spec_from_native
from scivol.components.density import DensityComponent
from scivol.components.vol import VolatilityComponent

VOLATILITIES = list(api.VolatilityKind.__members__)
DENSITIES = list(api.DensityKind.__members__)
STATIONARITIES = list(api.Stationarity.__members__)
MEANS = ("none", "arma", "arx")

rng = np.random.default_rng(20261101)
Y = 0.9 * rng.standard_t(7, 400) / np.sqrt(1.4)
X = rng.standard_normal((400, 1))


def subclasses(cls):
    out = set()
    for sub in cls.__subclasses__():
        out |= {sub} | subclasses(sub)
    return out


def native(mean: str, volatility: str, density: str, stationarity: str = "covariance", p: int = 1, q: int = 1):
    spec = api.ModelSpec()
    if mean == "arma":
        spec.mean.constant, spec.mean.ar, spec.mean.ma = True, 1, 1
    elif mean == "arx":
        spec.mean.constant, spec.mean.ar, spec.mean.exog = True, 1, 1
    spec.volatility = getattr(api.VolatilityKind, volatility)
    spec.p, spec.q = (0, 0) if volatility == "constant" else (p, q)
    spec.stationarity = getattr(api.Stationarity, stationarity)
    spec.density = getattr(api.DensityKind, density)
    return spec


def model_or_skip(spec) -> api.Model:
    try:
        return api.Model(spec)
    except ValueError as exc:  # a declaration the family does not admit
        pytest.skip(str(exc))


def regressors(spec):
    return X if spec.mean.exog else None


def value(model, y, theta, x=None):
    out = api.Jet(model, y, x).theta(theta, api.VALUE)
    assert out["status"] == "ok"
    return out["value"]


CASES = [(m, v, d, s) for v, d in itertools.product(VOLATILITIES, DENSITIES)
         for s in (["covariance"] if v == "constant" else STATIONARITIES) for m in MEANS]


def test_every_native_kind_has_a_python_component():
    volatility = {c.kind for c in subclasses(VolatilityComponent) if hasattr(c, "kind")}
    density = {c.kind for c in subclasses(DensityComponent) if hasattr(c, "kind")}
    assert volatility == set(VOLATILITIES) - {"constant"}
    assert density == set(DENSITIES)
    for cls in subclasses(VolatilityComponent) | subclasses(DensityComponent):
        assert getattr(scivol, cls.__name__) is cls


@pytest.mark.parametrize("mean,volatility,density,stationarity", CASES)
def test_model_declares_a_consistent_domain(mean, volatility, density, stationarity):
    spec = native(mean, volatility, density, stationarity)
    model = model_or_skip(spec)
    x = regressors(spec)
    d = model.dimension
    names = list(model.parameter_names)
    assert len(names) == d and len(set(names)) == d

    lower, upper = model.bounds()
    assert len(lower) == len(upper) == d
    rows = model.constraint_names
    assert len(rows) == len(model.open_constraints())

    theta = np.asarray(model.start(Y, x))
    assert model.feasible(theta), "the start lies in the domain"
    assert np.all(~(theta < lower)) and np.all(~(theta > upper))
    values, jacobian = model.constraints(theta)
    assert values.shape == (len(rows),) and jacobian.shape == (len(rows), d)
    assert np.all(values >= 0.0)

    z = model.to_z(theta)
    assert z is not None
    np.testing.assert_allclose(model.to_theta(z), theta, rtol=1e-10, atol=1e-12)

    assert np.isfinite(model.persistence(theta))
    variance = model.unconditional_variance(theta)
    assert np.isnan(variance) or variance > 0.0
    caps = model.capabilities
    assert caps.likelihood_can_have_gradient_kinks == caps.residual_kink_decomposition
    assert model.constant_variance == (volatility == "constant")

    # The Python components round-trip through the native description.
    assert native_spec(spec_from_native(spec), spec.mean.exog) == spec


@pytest.mark.parametrize("mean,volatility,density,stationarity", CASES)
def test_likelihood_is_equivariant_to_the_data_scale(mean, volatility, density, stationarity):
    model = model_or_skip(native(mean, volatility, density, stationarity))
    x = regressors(model.spec)
    theta = np.asarray(model.start(Y, x))
    factor = 3.0
    scaled = np.asarray(model.rescale(theta, factor))
    n = api.Jet(model, Y, x).theta(theta, api.PATHS)["residual"].size
    assert value(model, factor * Y, scaled, x) == pytest.approx(value(model, Y, theta, x) + n * np.log(factor),
                                                                rel=1e-11)


@pytest.mark.parametrize("volatility,density", list(itertools.product(VOLATILITIES, DENSITIES)))
def test_derivatives_are_wired_to_the_likelihood(volatility, density):
    model = model_or_skip(native("arx", volatility, density))
    theta = np.asarray(model.start(Y, X))
    jet = api.Jet(model, Y, X)
    out = jet.theta(theta, api.VALUE | api.GRADIENT | api.HESSIAN)
    assert out["status"] == "ok"
    step = 1e-6 * np.maximum(np.abs(theta), 1e-2)
    for k in range(model.dimension):
        up, down = theta.copy(), theta.copy()
        up[k] += step[k]
        down[k] -= step[k]
        g_up, g_down = (jet.theta(t, api.VALUE | api.GRADIENT) for t in (up, down))
        fd = (g_up["value"] - g_down["value"]) / (2 * step[k])
        assert out["gradient"][k] == pytest.approx(fd, rel=1e-5, abs=1e-5 * len(Y)), model.parameter_names[k]
        column = (g_up["gradient"] - g_down["gradient"]) / (2 * step[k])
        np.testing.assert_allclose(out["hessian"][:, k], column, rtol=1e-4, atol=1e-4 * len(Y))


@pytest.mark.parametrize("volatility,density", list(itertools.product(VOLATILITIES, DENSITIES)))
def test_restrictions_embed_exactly(volatility, density):
    parent = model_or_skip(native("arx", volatility, density, p=2 if volatility != "constant" else 1))
    first = max(parent.spec.mean.conditioning, 1)
    plan = api.ComparisonPlan(Y, first, X, None)
    for r in parent.restrictions():
        child = api.Model(r.child)
        theta = np.asarray(child.start(plan))
        lifted = r.embed(theta)
        if lifted is None:  # the child's start does not lift into the parent's domain
            continue
        assert r.on_face(lifted)
        np.testing.assert_allclose(r.map_start(lifted), theta, rtol=0, atol=0)
        f_parent = api.Jet(parent, plan).theta(lifted, api.VALUE)["value"]
        f_child = api.Jet(child, plan).theta(theta, api.VALUE)["value"]
        assert f_parent == pytest.approx(f_child, rel=1e-12), r.name


@pytest.mark.parametrize("volatility,density", list(itertools.product(VOLATILITIES, DENSITIES)))
def test_forecasts_and_simulations_run_at_the_start(volatility, density):
    model = model_or_skip(native("none", volatility, density))
    theta = np.asarray(model.start(Y, None))
    horizon = 4
    exact = min(horizon, model.capabilities.exact_variance_horizon)
    z = np.random.default_rng(1).standard_normal((50, horizon))
    try:
        out = api.forecast(model, theta, Y, horizon, None, None, z)
    except ValueError as exc:  # a forecast the density's moments do not admit (EGARCH with polynomial tails)
        assert "does not exist" in str(exc)
    else:
        assert np.all(np.isfinite(out["residual_variance"][:exact])) and np.all(out["residual_variance"] > 0)
    simulated = api.simulate_series(model, theta, z, None)
    assert np.all(np.isfinite(simulated["y"]))


@pytest.mark.parametrize("density", DENSITIES)
def test_density_is_standardized(density):
    model = api.Model(native("none", "garch", density))
    shape = np.asarray(model.start(Y, None))[3:]
    law = model.innovation_density(np.r_[np.asarray(model.start(Y, None))[:3], shape])
    u = (np.arange(200_000) + 0.5) / 200_000
    z = np.asarray(api.quantiles(law.density, law.shape, u, False))
    assert np.mean(z) == pytest.approx(0.0, abs=5e-3)
    assert np.mean(z * z) == pytest.approx(1.0, rel=2e-2)
    lower, upper = api.distribution(law.density, law.shape, np.ascontiguousarray(z[::997]))
    np.testing.assert_allclose(lower, u[::997], rtol=1e-8, atol=1e-12)


@pytest.mark.parametrize("density", DENSITIES)
def test_density_functions_and_moments_agree(density):
    """The density integrates to one; its moments, distribution and quantile
    functions are those of that density."""
    from scipy.integrate import quad

    kind = getattr(api.DensityKind, density)
    shape = [d["start"] for d in api.density_shapes(kind)]
    if kind == api.DensityKind.skew_t:
        shape[1] = -0.2  # an asymmetric point
    f = lambda z: float(np.exp(api.log_density(kind, shape, np.array([z]))[0]))
    for k in range(5):
        integral = quad(lambda z: z**k * f(z), -np.inf, np.inf, limit=400, epsabs=1e-11)[0]
        expected = 1.0 if k == 0 else api.standardized_moment(kind, shape, k)
        assert integral == pytest.approx(expected, rel=1e-7, abs=1e-8), k
    z = np.array([-1.3, 0.2, 2.1])
    lower, _ = api.distribution(kind, shape, z)
    assert lower[0] == pytest.approx(quad(f, -np.inf, -1.3)[0], rel=1e-8)
    np.testing.assert_allclose(api.quantiles(kind, shape, np.ascontiguousarray(lower), False), z, rtol=1e-9)
