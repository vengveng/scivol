"""scivolapi GARCH(p,q) with non-Normal densities: AD oracle, masks, charts, reductions."""

from __future__ import annotations

import numpy as np
import pytest

api = pytest.importorskip("scivol._scivolapi")
from tests._oracle import (  # noqa: E402
    _jax_modules,
    garch_value_grad_hess,
    gh_surface_value_grad_hess,
)

ORACLE_NAME = {"student_t": "studentt", "skew_t": "skewt", "ged": "ged"}
GH_ORACLE = {  # the GH oracle evaluates log K by quadrature, accurate to ~1e-13
    "generalized_hyperbolic": {},
    "nig": {"fixed_lambda": -0.5},
    "hyperbolic": {"fixed_lambda": 1.0},
    "variance_gamma": {"variance_gamma": True},
}
SHAPES = {  # shape points per density, including near-limit values
    "student_t": [[1/8.0], [1/2.3], [1/150.0]],
    "skew_t": [[1/7.0, -0.3], [1/3.1, 0.6], [1/60.0, 0.05]],
    "ged": [[1.4], [0.7], [3.5]],
    "generalized_hyperbolic": [[-0.3, 1.4, -0.7], [0.5, 3.0, 2.5], [0.2, 0.4, 0.3]],
    "nig": [[-0.2, 1.1], [0.6, 4.0]],
    "hyperbolic": [[0.4, 0.6], [-0.5, 2.0]],
    "variance_gamma": [[-0.3, 2.5], [0.6, 1.8]],
}
DENSITIES = list(SHAPES)


def oracle(density: str, theta: np.ndarray, y: np.ndarray, p: int, q: int):
    """(value, gradient, Hessian, tolerance) from the independent AD oracle."""
    if density in ORACLE_NAME:
        return (*garch_value_grad_hess(theta, y * y, p, q, ORACLE_NAME[density], resid=y), 1e-12)
    return (*gh_surface_value_grad_hess(theta, y, p_vol=p, q_vol=q, initial_variance=float(np.mean(y * y)),
                                        **GH_ORACLE[density]), 1e-11)
ORDERS = [(1, 1), (2, 1), (1, 2)]


def model_for(density: str, p: int = 1, q: int = 1) -> "api.Model":
    spec = api.ModelSpec()
    spec.p, spec.q = p, q
    spec.density = getattr(api.DensityKind, density)
    return api.Model(spec)


@pytest.fixture(scope="module")
def series() -> np.ndarray:
    rng = np.random.default_rng(5)
    return rng.standard_t(5, 1200) * 0.9


def point(p: int, q: int, shape) -> np.ndarray:
    return np.array([0.08] + [0.10 / p] * p + [0.80 / q] * q + list(shape))


def assert_close(actual, expected, rtol):
    scale = max(1.0, float(np.max(np.abs(expected))))
    assert float(np.max(np.abs(np.asarray(actual) - expected))) <= rtol * scale


def check_against_oracle(series, density, p, q, shapes):
    model = model_for(density, p, q)
    jet = api.Jet(model, series)
    for shape in shapes:
        theta = point(p, q, shape)
        value, gradient, hessian, tolerance = oracle(density, theta, series, p, q)
        full = jet.theta(theta, api.VALUE | api.GRADIENT | api.HESSIAN)
        assert full["status"] == "ok", shape
        assert full["value"] == pytest.approx(value, rel=tolerance), shape
        assert_close(full["gradient"], gradient, tolerance)
        assert_close(full["hessian"], hessian, tolerance)


@pytest.mark.parametrize("density", list(ORACLE_NAME))
@pytest.mark.parametrize("p,q", ORDERS)
def test_theta_jet_matches_ad_oracle(series, density, p, q):
    check_against_oracle(series, density, p, q, SHAPES[density])


# The GH oracle recompiles a quadrature per call (5-15 s): one point per member
# routinely, the full matrix with --run-slow.
@pytest.mark.parametrize("density", list(GH_ORACLE))
def test_gh_family_jet_matches_ad_oracle(series, density):
    check_against_oracle(series[:150], density, 1, 1, SHAPES[density][:1])


@pytest.mark.parametrize("density,shape", [
    ("variance_gamma", [-0.2, 3.0]),
    ("generalized_hyperbolic", [0.3, 2.01, 1.0]),
])
def test_variable_order_at_half_integer_matches_ad_oracle(series, density, shape):
    check_against_oracle(series[:150], density, 1, 1, [shape])
    jet = api.Jet(model_for(density), series)
    theta = point(1, 1, shape)
    full = jet.theta(theta, 7)
    for mask in (1, 2, 3, 4, 5, 6):
        part = jet.theta(theta, mask)
        for key, bit in (("value", 1), ("gradient", 2), ("hessian", 4)):
            if mask & bit:
                assert np.array_equal(part[key], full[key])


@pytest.mark.slow
@pytest.mark.parametrize("density", list(GH_ORACLE))
@pytest.mark.parametrize("p,q", ORDERS)
def test_gh_family_jet_matches_ad_oracle_everywhere(series, density, p, q):
    check_against_oracle(series[:300], density, p, q, SHAPES[density])


@pytest.mark.parametrize("density", DENSITIES)
def test_every_sparse_request_equals_the_full_request(series, density):
    model = model_for(density, 2, 1)
    jet = api.Jet(model, series)
    theta = point(2, 1, SHAPES[density][0])
    full = jet.theta(theta, 15)
    for mask in range(1, 16):
        part = jet.theta(theta, mask)
        assert part["status"] == "ok"
        for key, bit in (("value", 1), ("gradient", 2), ("hessian", 4), ("variance", 8)):
            if mask & bit:
                assert np.array_equal(part[key], full[key])


def jax_theta(jnp, jax, model, z):
    """Independent JAX statement of the charts declared by the model's bounds."""
    lower, upper = model.bounds()
    names = model.parameter_names
    coefficients = [i for i, n in enumerate(names) if n.startswith(("alpha", "beta"))]
    k = len(coefficients)
    limit = 1.0 - 8.0 * (k + 1) * np.finfo(float).eps
    e = jnp.exp(z[coefficients[0]:coefficients[-1] + 1])
    simplex = limit * e / (1.0 + jnp.sum(e))
    out = []
    for i in range(len(names)):
        if i in coefficients:
            out.append(simplex[i - coefficients[0]])
        elif np.isfinite(upper[i]):
            out.append((np.nextafter(lower[i], upper[i]) + (upper[i]-np.nextafter(lower[i], upper[i])) * jax.nn.sigmoid(z[i])) if lower[i] != -upper[i] else upper[i] * jnp.tanh(z[i]))
        elif np.isfinite(lower[i]):
            out.append(lower[i] + jnp.exp(z[i]))
        else:
            out.append(z[i])
    return jnp.stack(out)


@pytest.mark.parametrize("density", DENSITIES)
def test_z_jet_includes_chart_curvature(series, density):
    jax, jnp, _ = _jax_modules()
    model = model_for(density, 1, 2)
    jet = api.Jet(model, series)
    theta = point(1, 2, SHAPES[density][0])
    z = model.to_z(theta)
    assert np.allclose(model.to_theta(z), theta, rtol=1e-13, atol=0.0)

    full_theta = jet.theta(theta, api.GRADIENT | api.HESSIAN)
    jac = np.asarray(jax.jacfwd(lambda v: jax_theta(jnp, jax, model, v))(jnp.asarray(z)))
    curv = np.asarray(jax.hessian(
        lambda v: jnp.dot(jnp.asarray(full_theta["gradient"]), jax_theta(jnp, jax, model, v)))(jnp.asarray(z)))
    expected_g = jac.T @ full_theta["gradient"]
    expected_h = jac.T @ full_theta["hessian"] @ jac + curv

    full_z = jet.z(z, api.GRADIENT | api.HESSIAN)
    assert full_z["status"] == "ok"
    assert_close(full_z["gradient"], expected_g, 1e-12)
    assert_close(full_z["hessian"], expected_h, 1e-11)


def test_skew_t_at_zero_skew_is_student_t(series):
    skew, student = api.Jet(model_for("skew_t"), series), api.Jet(model_for("student_t"), series)
    a = skew.theta(point(1, 1, [1/6.5, 0.0]), 7)
    b = student.theta(point(1, 1, [1/6.5]), 7)
    assert a["value"] == pytest.approx(b["value"], rel=1e-14)
    assert_close(a["gradient"][:4], b["gradient"], 1e-12)
    assert_close(a["hessian"][:4, :4], b["hessian"], 1e-11)


def test_ged_with_shape_two_is_normal(series):
    ged, normal = api.Jet(model_for("ged"), series), api.Jet(model_for("normal"), series)
    a = ged.theta(point(1, 1, [2.0]), 7)
    b = normal.theta(point(1, 1, []), 7)
    assert a["value"] == pytest.approx(b["value"], rel=1e-14)
    assert_close(a["gradient"][:3], b["gradient"], 1e-12)
    assert_close(a["hessian"][:3, :3], b["hessian"], 1e-11)


@pytest.mark.parametrize("twice_lambda,member", [(-1, "nig"), (2, "hyperbolic")])
def test_gh_with_fixed_index_is_its_member(series, twice_lambda, member):
    gh, fixed = api.Jet(model_for("generalized_hyperbolic"), series), api.Jet(model_for(member), series)
    a = gh.theta(point(1, 1, [0.35, 1.7, 0.5 * twice_lambda]), 7)
    b = fixed.theta(point(1, 1, [0.35, 1.7]), 7)
    assert a["value"] == pytest.approx(b["value"], rel=1e-14)
    assert_close(a["gradient"][:5], b["gradient"], 1e-12)
    assert_close(a["hessian"][:5, :5], b["hessian"], 1e-12)


def test_variance_gamma_derivatives_exist_at_its_centre(series):
    """rho = 0 puts the centre at x = 0: the jet is continuous there (lambda > 3/2)."""
    theta = point(1, 1, [0.0, 2.2])
    exact, near = series.copy(), series.copy()
    exact[::9] = 0.0
    near[::9] = 1e-300
    at = api.Jet(model_for("variance_gamma"), exact).theta(theta, 7)
    beside = api.Jet(model_for("variance_gamma"), near).theta(theta, 7)
    assert at["status"] == beside["status"] == "ok"
    assert at["value"] == pytest.approx(beside["value"], rel=1e-14)
    assert_close(at["gradient"], beside["gradient"], 1e-12)
    assert_close(at["hessian"], beside["hessian"], 1e-12)


def test_ged_derivatives_exist_at_exact_zero_returns(series):
    data = series.copy()
    data[::7] = 0.0
    jet = api.Jet(model_for("ged"), data)
    theta = point(1, 1, [0.8])
    value, gradient, hessian = garch_value_grad_hess(theta, data * data, 1, 1, "ged")
    out = jet.theta(theta, 7)
    assert out["status"] == "ok"
    assert out["value"] == pytest.approx(value, rel=1e-12)
    assert_close(out["gradient"], gradient, 1e-10)
    assert_close(out["hessian"], hessian, 1e-9)


def mp_gh_terms(x, rho, zeta, lam):
    """Standardized GH negative log density summed over x, in the current mpmath precision."""
    import mpmath as mp

    k = mp.besselk
    kappa = k(lam + 1, zeta) / k(lam, zeta) / zeta
    kappa_next = k(lam + 2, zeta) / k(lam + 1, zeta) / zeta
    rho2 = 1 - rho * rho
    alpha = mp.sqrt(zeta**2 * kappa * (1 + rho**2 * zeta**2 * (kappa_next - kappa) / rho2) / rho2)
    beta, delta = alpha * rho, zeta / (alpha * mp.sqrt(rho2))
    mu, gamma = -beta * delta**2 * kappa, alpha * mp.sqrt(rho2)
    total = 0
    for value in x:
        c = mp.mpf(value) - mu
        r = mp.sqrt(delta**2 + c**2)
        total -= (lam * mp.log(gamma / delta) - mp.log(2 * mp.pi) / 2 - mp.log(k(lam, zeta)) + beta * c
                  + mp.log(k(lam - mp.mpf(1) / 2, alpha * r)) - (mp.mpf(1) / 2 - lam) * (mp.log(r) - mp.log(alpha)))
    return total


def mp_gh_nll(x, rho, zeta, lam):
    """mp_gh_terms in 80-digit arithmetic."""
    import mpmath as mp

    with mp.workdps(80):
        return float(mp_gh_terms(x, mp.mpf(rho), mp.mpf(zeta), mp.mpf(lam)))


# zeta -> 0 is a regular limit for lambda > 0 (Variance-Gamma); lambda = -1 puts the
# order lambda + 1 on zero. The reference differentiates the 60-digit likelihood.
@pytest.mark.parametrize("member,shape", [
    ("hyperbolic", [0.3, 1e-9]), ("generalized_hyperbolic", [0.3, 1e-9, 1.7]),
    ("generalized_hyperbolic", [-0.4, 0.3, -1.0]), ("nig", [0.3, 1e-9]),
])
def test_gh_family_shape_derivatives_are_exact_at_small_zeta(member, shape):
    import mpmath as mp

    data = np.random.default_rng(1).standard_t(5, 12) * 0.9
    data[::3] = 0.0
    theta = point(1, 1, shape)
    out = api.Jet(model_for(member), data).theta(theta, api.VALUE | api.GRADIENT | api.HESSIAN | api.PATHS)
    assert out["status"] == "ok"
    fixed = {"hyperbolic": 1.0, "nig": -0.5}.get(member)
    with mp.workdps(60):
        x = [mp.mpf(v) / mp.sqrt(mp.mpf(h)) for v, h in zip(data, out["variance"])]
        s0 = [mp.mpf(v) for v in shape]
        step = [mp.mpf(10) ** -15 * (v if i == 1 else 1) for i, v in enumerate(s0)]

        def f(*s):
            return mp_gh_terms(x, s[0], s[1], mp.mpf(fixed) if fixed is not None else s[2])

        def along(i, j=None):
            def g(a, b=None):
                return f(*[a if m == i else b if m == j else s0[m] for m in range(len(s0))])
            return g

        n = len(shape)
        gradient = [float(mp.diff(along(i), s0[i], h=step[i])) for i in range(n)]
        hessian = [[float(mp.diff(along(i), s0[i], 2, h=step[i]) if i == j
                          else mp.diff(along(i, j), (s0[i], s0[j]), (1, 1), h=min(step[i], step[j])))
                    for j in range(n)] for i in range(n)]
    d = len(theta)
    assert_close(np.asarray(out["gradient"])[3:], np.array(gradient), 1e-12)
    assert_close(np.asarray(out["hessian"]).reshape(d, d)[3:, 3:], np.array(hessian), 1e-12)


@pytest.mark.parametrize("member,lam", [("nig", -0.5), ("hyperbolic", 1.0)])
@pytest.mark.parametrize("rho,zeta", [(1 - 1e-14, 998.0), (-(1 - 1e-12), 50.0), (0.5, 1e4)])
def test_gh_family_is_accurate_at_extreme_skew_and_concentration(member, lam, rho, zeta):
    rng = np.random.default_rng(1)
    data = (rng.gamma(4.0, size=40) - 4.0) / 2.0
    out = api.Jet(model_for(member), data).theta(point(1, 1, [rho, zeta]), api.VALUE | api.PATHS)
    assert out["status"] == "ok"
    h = out["variance"]
    expected = mp_gh_nll(data / np.sqrt(h), rho, zeta, lam) + 0.5 * np.sum(np.log(h))
    assert out["value"] == pytest.approx(expected, rel=1e-11)


@pytest.mark.parametrize("density,bad", [
    ("student_t", [0.5]), ("skew_t", [0.2, 1.0]), ("ged", [0.0]),
    ("generalized_hyperbolic", [1.0, 1.0, 0.5]), ("nig", [0.2, 0.0]), ("variance_gamma", [0.1, 1.5]),
])
def test_shapes_outside_the_domain_are_not_evaluable(series, density, bad):
    jet = api.Jet(model_for(density), series)
    assert jet.theta(point(1, 1, bad), api.VALUE)["status"] == "not_evaluable"


# --------------------------------------------------------------------------- #
# Complete fits through the public API
# --------------------------------------------------------------------------- #

from scivol import (  # noqa: E402
    GARCH, GED, GeneralizedHyperbolic, Hyperbolic, NormalInverseGaussian, SkewT, StudentT, VarianceGamma,
)

COMPONENTS = {
    "student_t": StudentT, "skew_t": SkewT, "ged": GED, "generalized_hyperbolic": GeneralizedHyperbolic,
    "nig": NormalInverseGaussian, "hyperbolic": Hyperbolic, "variance_gamma": VarianceGamma,
}


def simulate(n: int, innovations: np.ndarray) -> np.ndarray:
    y, h = np.empty(n), 0.05 / 0.02
    for t in range(n):
        y[t] = np.sqrt(h) * innovations[t]
        h = 0.05 + 0.08 * y[t] * y[t] + 0.90 * h
    return y


@pytest.fixture(scope="module")
def t_data() -> np.ndarray:
    rng = np.random.default_rng(1)
    return simulate(4000, rng.standard_t(5, 4000) * np.sqrt(3.0 / 5.0))


@pytest.fixture(scope="module")
def hyperbolic_data() -> np.ndarray:
    from scipy.stats import genhyperbolic

    from tests._samplers import standardized_gh_parameters

    z = genhyperbolic.rvs(*standardized_gh_parameters(-0.25, 2.5, 1.0), size=4000, random_state=2)
    return simulate(4000, z)


# Frozen maxima on the same data and likelihood.
PRODUCTION_LOG_LIKELIHOOD = {
    "density_student_t": -7068.881540527609,
    "density_skew_t": -7068.871131226216,
    "density_ged": -7104.880821423252,
    "density_generalized_hyperbolic": -7031.970718976294,
    "density_nig": -7076.930666437685,
    "density_hyperbolic": -7085.301819616398,
    "density_variance_gamma": -7091.9755165130555,
}


@pytest.mark.parametrize("density", list(COMPONENTS))
def test_fit_reaches_the_production_optimum(t_data, hyperbolic_data, density):
    # On Student-t data the GH optimum is its zeta -> 0 limit, a boundary point.
    data = hyperbolic_data if density == "generalized_hyperbolic" else t_data
    result = (GARCH(1, 1) + COMPONENTS[density]()).fit(data)
    assert result.success and result.fit_info.inference_status == "available"
    assert result.log_likelihood >= PRODUCTION_LOG_LIKELIHOOD["density_" + density] - 1e-6


@pytest.mark.parametrize("coordinates", ["automatic", "theta"])
def test_likelihood_rising_toward_an_open_bound_is_not_a_maximum(coordinates):
    # On t(3) innovations the VG likelihood rises toward lambda = 3/2, which the
    # domain excludes: however close the optimizer stops, no maximum exists.
    rng = np.random.default_rng(1)
    z = rng.standard_t(3.0, 2000) * np.sqrt(1.0 / 3.0)
    y, h = np.empty(2000), 1.0
    for t in range(2000):
        y[t] = np.sqrt(h) * z[t]
        h = 0.1 + 0.05 * y[t] ** 2 + 0.85 * h
    result = (GARCH(1, 1) + VarianceGamma()).fit(y, coordinates=coordinates)
    assert result.params[-1] > 1.5 and result.params[-1] - 1.5 < 1e-6
    assert not result.success and result.fit_info.inference_status == "at_open_bound"


def test_student_t_fit_recovers_the_tail_index(t_data):
    result = (GARCH(1, 1) + StudentT()).fit(t_data)
    assert abs(result.params[3] - 0.2) < 4.0 * result.std_errors[3]
    assert abs(result.params[1] - 0.08) < 4.0 * result.std_errors[1]


@pytest.mark.parametrize("coordinates", ["automatic", "theta"])
@pytest.mark.parametrize("iid", [False, True], ids=["gjr", "iid"])
def test_gaussian_tail_optimum_is_the_closed_eta_face(coordinates, iid, capsys):
    from scivol import GJRGARCH, Normal

    z = np.random.default_rng(31).normal(size=3000)
    y, variance = np.empty(len(z)), 1.0
    for t in range(len(z)):
        y[t] = z[t] * np.sqrt(variance)
        variance = 0.05 + 0.08*y[t]**2 + 0.82*variance + 0.10*y[t]**2*(y[t] < 0)
    if iid:
        y = np.random.default_rng(2).normal(size=2500)
    normal = (GJRGARCH(1, 1) + Normal()).fit(y, coordinates=coordinates)
    student = (GJRGARCH(1, 1) + StudentT()).fit(y, coordinates=coordinates)
    assert normal.success and student.success
    assert student.params[-1] == 0.0
    # On iid data the first automatic pass can end on the flat GARCH ridge at
    # the excluded omega = 0 endpoint; the theta restart then ends at the face.
    assert student.fit_info.optimizer_phases[-1].startswith("theta:newton")
    # Equal models; on the flat iid ridge the two certified optima agree to the
    # summation roundoff of F (about 1e-13 relative).
    assert student.log_likelihood == pytest.approx(normal.log_likelihood, rel=0.0, abs=1e-9)
    assert student.dist_params.nu == np.inf
    student.summary()
    summary = capsys.readouterr().out
    assert "eta=" in summary and "nu=inf" in summary

    skew = (GJRGARCH(1, 1) + SkewT()).fit(y, coordinates=coordinates)
    assert skew.success and skew.params[-2] == 0.0
    # Hansen's Normal limit retains lambda. Check its two-piece Normal density.
    spec = api.ModelSpec()
    spec.volatility, spec.density = api.VolatilityKind.gjr, api.DensityKind.skew_t
    model = api.Model(spec)
    out = api.Jet(model, y).theta(np.ascontiguousarray(skew.params), api.VALUE | api.PATHS)
    lam = skew.params[-1]
    a = 4*lam/np.sqrt(2*np.pi)
    b = np.sqrt(1+3*lam*lam-a*a)
    standardized = y/np.sqrt(out["variance"])
    shifted = b*standardized+a
    u = shifted/(1+lam*np.where(shifted >= 0, 1., -1.))
    expected = np.sum(np.log(b)-np.log(2*np.pi)/2-u*u/2-np.log(out["variance"])/2)
    assert skew.log_likelihood == pytest.approx(expected, rel=0.0, abs=2e-10)


@pytest.mark.parametrize("density", ["student_t", "skew_t"])
@pytest.mark.parametrize("eta", [0., 1e-300, 1e-16, 1e-10, 1e-6, 1e-3, .1, .3, .49])
def test_eta_limit_jet_matches_float64_ad(density, eta):
    data = np.array([0., -10., -3., -.1, 1e-6, .2, 1., 4., 20.])
    shape = [eta] if density == "student_t" else [eta, .2]
    check_against_oracle(data, density, 1, 1, [shape])


@pytest.mark.parametrize("density", ["student_t", "skew_t"])
def test_closed_tail_domain_and_saturated_interval_chart(density):
    model = model_for(density)
    data = np.array([-.2,.1,2.,-.7])
    theta = np.asarray(model.start(data))
    tail = model.parameter_names.index("eta")
    lower, upper = model.bounds()
    assert lower[tail] == 0.0 and upper[tail] == np.nextafter(.5,0.)
    assert model.capabilities.chart_covers_domain
    theta[tail] = 0.0
    assert model.feasible(theta) and model.to_z(theta) is None
    innovation = model.innovation_density(theta)
    assert innovation.named["eta"] == 0.0 and "nu" not in innovation.named
    assert innovation.derived == {"nu": np.inf}
    for value in [-1e-300,.5,np.inf,np.nan]:
        theta[tail] = value
        assert not model.feasible(theta)
    z = model.to_z(np.asarray(model.start(data)))
    jet = api.Jet(model,data)
    for value in [-1e308,-1000.,-100.,0.,100.,1000.,1e308]:
        z[tail] = value
        point = model.to_theta(z)
        assert 0.0 < point[tail] < .5 and model.feasible(point)
        result = jet.z(z,7)
        assert result["status"] == "ok"
        assert np.all(np.isfinite(result["gradient"])) and np.all(np.isfinite(result["hessian"]))
