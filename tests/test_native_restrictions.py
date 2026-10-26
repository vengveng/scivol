"""Compiled children, common conditioning, LR geometry, and independent fits."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from itertools import combinations
from math import comb

import numpy as np
import pytest
from numpy.testing import assert_allclose
from scipy.stats import chi2

from scivol import ARX, GARCH
from scivol import _scivolapi as api
from scivol._fit import fit_spec_on_plan


def model(volatility="garch", p=1, q=1, density="normal", *, ar=0, ma=0,
          constant=False, har=(), exog=0):
    s = api.ModelSpec()
    s.volatility = getattr(api.VolatilityKind, volatility)
    s.p, s.q = p, q
    s.density = getattr(api.DensityKind, density)
    s.mean.constant, s.mean.ar, s.mean.ma = constant, ar, ma
    s.mean.har, s.mean.exog = list(har), exog
    return api.Model(s)


def edge(parent, name):
    return next(r for r in parent.restrictions() if r.name == name)


def garch_data(n=2000, seed=17):
    rng = np.random.default_rng(seed)
    y = np.empty(n + 500)
    h, e = 1.0, 0.0
    for t, z in enumerate(rng.standard_normal(len(y))):
        h = 0.1 + 0.15 * e * e + 0.7 * h
        e = np.sqrt(h) * z
        y[t] = e
    return np.ascontiguousarray(y[500:])


@pytest.mark.parametrize("kind", ["garch", "gjr", "egarch", "aparch", "constant"])
@pytest.mark.parametrize("density", ["normal", "student_t", "skew_t", "ged"])
def test_children_are_exact_likelihood_restrictions(kind, density):
    rng = np.random.default_rng(32)
    y, x = rng.normal(size=100), rng.normal(size=(100, 2))
    p = q = 0 if kind == "constant" else 2
    parent = model(kind, p, q, density, ar=2, ma=1, constant=True, har=(1, 5), exog=2)
    plan = api.ComparisonPlan(y, 10, x, variance_seed=1.73)
    for r in parent.restrictions():
        child = api.Model(r.child)
        theta = child.start(plan)
        lifted = r.embed(theta)
        assert lifted is not None, r.name
        assert r.on_face(lifted)
        assert_allclose(r.map_start(lifted), theta, rtol=0, atol=0)
        pjet, cjet = api.Jet(parent, plan), api.Jet(child, plan)
        po = pjet.theta(lifted, api.VALUE | api.GRADIENT | api.HESSIAN | api.PATHS)
        co = cjet.theta(theta, api.VALUE | api.GRADIENT | api.HESSIAN | api.PATHS)
        assert po["status"] == co["status"] == "ok"
        assert_allclose(po["value"], co["value"], rtol=2e-13)
        e = np.asarray(r.embedding).reshape(parent.dimension, child.dimension)
        assert_allclose(e.T @ po["gradient"], co["gradient"], rtol=2e-11, atol=2e-11)
        assert_allclose(e.T @ po["hessian"] @ e, co["hessian"], rtol=2e-10, atol=2e-10)
        assert_allclose(po["variance"], co["variance"], rtol=2e-13)
        assert pjet.first_scored == cjet.first_scored == 10
        assert pjet.variance_seed == cjet.variance_seed == 1.73


def test_common_plan_seed_rows_and_exogenous_prefix():
    y = np.random.default_rng(5).normal(size=100)
    x = np.random.default_rng(6).normal(size=(100, 2))
    plan = api.ComparisonPlan(y, 12, x)
    a, b = model(ar=2, exog=2), model(ar=0, exog=1)
    assert api.Jet(a, plan).variance_seed == api.Jet(b, plan).variance_seed
    assert api.Jet(a, plan).first_scored == api.Jet(b, plan).first_scored == 12
    assert_allclose(plan.variance_seed, np.mean(y * y))
    owned = plan.y
    owned[:] = 100
    assert_allclose(plan.y, y)
    with pytest.raises(ValueError):
        api.Jet(model(ar=8), api.ComparisonPlan(y, 2))
    with pytest.raises(ValueError):
        api.ComparisonPlan(y, 12, x[:-1])
    fitted = fit_spec_on_plan(ARX(0) + GARCH(1, 1), plan, exog=1)
    assert fitted.scored == 88 and fitted.observations == 100
    cv = model("constant", 0, 0)
    out = api.Jet(cv, plan).theta(np.array([2.0]), api.PATHS)
    assert_allclose(out["variance"], 2.0)


def test_gjr_sign_loading_geometry_and_unidentified_edges():
    parent = model("gjr", 2, 1)
    shock = edge(parent, "volatility.shock")
    a = np.array(shock.directions).reshape(2, -1)
    assert_allclose(a, [[0, 0, 1, 0, 0, 0], [0, 0, 1, 0, 1, 0]])
    assert shock.types == [api.RestrictionType.closed_boundary] * 2
    assert edge(model(), "volatility.shock").identification == api.Identification.unidentified_under_null
    assert edge(model("aparch", 2), "volatility.shock").identification == api.Identification.unidentified_under_null
    assert edge(model("egarch", 2), "volatility.shock").types == [api.RestrictionType.interior_equality] * 2
    assert api.Model(edge(model(), "volatility.beta").child).dimension == 2
    assert [r.name for r in model("aparch").restrictions()] == ["volatility.shock", "volatility.beta"]
    assert [r.name for r in model(density="skew_t").restrictions()].count("density.symmetry") == 1
    assert len(model(density="generalized_hyperbolic").restrictions()) == 2


@pytest.mark.parametrize("m", range(1, 9))
def test_independent_chibar_weights(m):
    out = api.chi_bar_weights(np.eye(m))
    assert out.status == api.LRStatus.ok
    assert_allclose(out.weights, [comb(m, k) / 2**m for k in range(m + 1)], atol=1e-13)


@pytest.mark.parametrize("rho", [-0.8, -0.2, 0.0, 0.6, 0.95])
def test_two_boundary_closed_form(rho):
    v = np.array([[1.0, rho], [rho, 1.0]])
    out = api.chi_bar_weights(np.ascontiguousarray(np.linalg.inv(v)))
    assert out.status == api.LRStatus.ok
    assert_allclose(out.weights, [0.25 - np.arcsin(rho) / (2*np.pi), 0.5,
                                 0.25 + np.arcsin(rho) / (2*np.pi)], atol=1e-13)


@pytest.mark.parametrize("df", range(1, 33))
def test_native_chi_square_tails(df):
    for t in [0, 1e-10, 0.1, 2.705543454095404, 10, 50, 1000]:
        assert_allclose(api.chi_square_survival(t, df), chi2.sf(t, df), atol=3e-14, rtol=2e-12)


def test_three_boundary_formula_and_cubature_budget():
    v = np.array([[2.0, .3, -.4], [.3, 1.0, .2], [-.4, .2, 1.5]])
    j = np.ascontiguousarray(np.linalg.inv(v))
    def h(a):
        r = a / np.sqrt(np.outer(np.diag(a), np.diag(a)))
        return 1/8 + sum(np.arcsin(r[i, k]) for i, k in combinations(range(3), 2)) / (4*np.pi)
    out = api.chi_bar_weights(j)
    assert_allclose(out.weights, [h(j), .5-h(v), .5-h(j), h(v)], atol=1e-13)
    j4 = np.eye(4) + .2 * np.ones((4, 4))
    first = api.chi_bar_weights(j4, maximum_cells=64)
    second = api.chi_bar_weights(j4, maximum_cells=64)
    assert first.status == second.status == api.LRStatus.unresolved
    assert first.weights == second.weights
    assert first.numerical_error > 1e-6
    assert api.chi_bar_weights(np.ones((2, 2))).status == api.LRStatus.singular_information


def test_lr_interior_boundary_and_plan_mismatch():
    y = garch_data()
    plan = api.ComparisonPlan(y, 4)
    parent = model("gjr", density="student_t")
    r = edge(parent, "volatility.symmetry")
    child = api.Model(r.child)
    cf = api.fit(child, plan)
    options = api.FitOptions()
    options.start = list(r.embed(cf.theta))
    pf = api.fit(parent, plan, options)
    assert pf.success and cf.success
    info = api.Jet(parent, plan).theta(r.embed(cf.theta), api.HESSIAN)["hessian"]
    out = api.restriction_test(parent, child, r, pf, cf, info)
    if cf.theta[-1] == 0:
        assert out.status == api.LRStatus.retained_boundary
    else:
        assert out.status == api.LRStatus.ok, out.message
        assert out.interior_df == 1
        assert_allclose(out.p_value, chi2.sf(out.statistic, 1), atol=1e-13)
    mismatch = api.fit(child, api.ComparisonPlan(y, 4))
    assert api.restriction_test(parent, child, r, pf, mismatch, info).status == api.LRStatus.incompatible
    assert api.restrict_fit(parent, pf, r, api.ComparisonPlan(y, 4)).status == api.RestrictFitStatus.incompatible
    parent = model("garch", 1, 1)
    r = edge(parent, "volatility.beta")
    child = api.Model(r.child)
    pf, cf = api.fit(parent, plan), api.fit(child, plan)
    info = api.Jet(parent, plan).theta(r.embed(cf.theta), api.HESSIAN)["hessian"]
    out = api.restriction_test(parent, child, r, pf, cf, info)
    assert out.status == api.LRStatus.ok, out.message
    assert_allclose(out.weights, [.5, .5])
    assert_allclose(out.p_value, .5 * chi2.sf(out.statistic, 1), atol=1e-13)


def test_concurrent_native_fits_match_serial_with_separate_instances():
    y = garch_data(500)
    plan = api.ComparisonPlan(y, 6)
    factories = [lambda: model(ar=1, constant=True), lambda: model("gjr"),
                 lambda: model(har=(1, 5), constant=True), lambda: model("egarch", q=0)]
    def run(factory):
        fitted = api.fit(factory(), plan)
        return fitted.theta.copy(), fitted.log_likelihood, fitted.success
    serial = [run(factory) for factory in factories]
    with ThreadPoolExecutor(max_workers=4) as pool:
        concurrent = list(pool.map(run, factories))
    for a, b in zip(serial, concurrent):
        assert_allclose(a[0], b[0], rtol=0, atol=0)
        assert a[1:] == b[1:]


@pytest.mark.slow
@pytest.mark.parametrize("m", [2, 3, 4])
def test_weights_against_independent_gaussian_projection(m):
    rng = np.random.default_rng(2026 + m)
    raw = rng.normal(size=(m, m))
    j = raw.T @ raw + 3*np.eye(m)
    out = api.chi_bar_weights(j, maximum_cells=4096)
    draws = rng.multivariate_normal(np.zeros(m), np.linalg.inv(j), size=300_000)
    counts = np.zeros(m + 1)
    # Enumerate active sets of the quadratic-program projection independently.
    assigned = np.zeros(len(draws), dtype=bool)
    for k in range(m + 1):
        for active in combinations(range(m), k):
            s = np.array(active, dtype=int)
            r = np.array([i for i in range(m) if i not in active], dtype=int)
            projected = np.zeros_like(draws)
            if len(s):
                projected[:, s] = draws[:, s]
                if len(r):
                    projected[:, s] += draws[:, r] @ np.linalg.solve(j[np.ix_(s, s)], j[np.ix_(s, r)]).T
            grad = (projected - draws) @ j
            valid = (~assigned) & np.all(projected[:, s] > 0, axis=1) & np.all(grad[:, r] >= 0, axis=1)
            counts[k] += valid.sum()
            assigned |= valid
    assert assigned.all()
    assert_allclose(out.weights, counts / len(draws), atol=.004 + out.numerical_error)


@pytest.mark.slow
def test_gjr_symmetry_lr_small_size_simulation():
    rejections, valid = 0, 0
    for seed in range(100):
        y = garch_data(1500, 1000 + seed)
        plan = api.ComparisonPlan(y, 3)
        parent = model("gjr")
        r = edge(parent, "volatility.symmetry")
        child = api.Model(r.child)
        cf = api.fit(child, plan)
        options = api.FitOptions()
        options.start = list(r.embed(cf.theta))
        pf = api.fit(parent, plan, options)
        info = api.Jet(parent, plan).theta(r.embed(cf.theta), api.HESSIAN)["hessian"]
        out = api.restriction_test(parent, child, r, pf, cf, info, level=.05)
        if out.status == api.LRStatus.ok:
            valid += 1
            rejections += out.p_value < .05
    assert valid >= 90
    print(f"GJR symmetry size: {rejections}/{valid} at nominal 5%")
    assert 1 <= rejections <= .12 * valid


def test_exact_face_child_is_finalized_without_optimizer():
    plan = api.ComparisonPlan(np.random.default_rng(2).normal(size=600), 4)
    parent = model("constant", 0, 0, "student_t")
    pf = api.fit(parent, plan)
    r = edge(parent, "density.normal")
    assert pf.success and pf.theta[-1] == 0.0
    reused = api.restrict_fit(parent, pf, r, plan)
    assert reused.status == api.RestrictFitStatus.reused
    cf = reused.result
    assert cf.success and cf.stationary
    assert cf.kkt_residual <= cf.kkt_tolerance
    assert [phase.name for phase in cf.phases] == ["restriction:projection"]
    assert cf.phases[0].iterations == 0
    assert_allclose(r.embed(cf.theta), pf.theta, atol=0, rtol=0)
    info = api.Jet(parent, plan).theta(r.embed(cf.theta), api.HESSIAN)["hessian"]
    out = api.restriction_test(parent, api.Model(r.child), r, pf, cf, info)
    assert out.status == api.LRStatus.ok
    assert out.statistic == 0.0 and out.p_value == 1.0
    fresh = api.fit(api.Model(r.child), plan)
    assert_allclose(cf.theta, fresh.theta, rtol=2e-5)
    assert_allclose(cf.log_likelihood, pf.log_likelihood, atol=2e-12)
    shifted = pf.theta.copy()
    shifted[-1] = 1e-15
    assert not r.on_face(shifted)


def test_warm_start_is_a_new_optimization_and_domain_maps_can_fail():
    y = garch_data(1200)
    plan = api.ComparisonPlan(y, 3)
    parent = model("gjr")
    r = edge(parent, "volatility.symmetry")
    pf = api.fit(parent, plan)
    projected = r.map_start(pf.theta)
    assert projected is not None
    options = api.FitOptions()
    options.start = list(projected)
    cf = api.fit(api.Model(r.child), plan, options)
    assert cf.success
    assert sum(phase.iterations for phase in cf.phases) > 0
    # Removing negative leverage can increase persistence outside the child domain.
    assert r.map_start(np.array([.1, .8, -.6, .4])) is None
    assert r.embed(np.array([.1, .8, .4])) is None


def test_compiled_model_and_edge_spec_queries_are_copies():
    parent = model("gjr", p=2)
    before = parent.parameter_names
    queried = parent.spec
    queried.p = 8
    assert parent.spec.p == 2 and parent.parameter_names == before
    r = edge(parent, "volatility.symmetry")
    queried = r.child
    queried.p = 8
    assert r.child.p == 2
    assert edge(model("garch", 1, 0), "volatility.shock").identification == api.Identification.conditional
