"""scivolapi DCC: correlation jet against a JAX oracle, fits, results."""

from __future__ import annotations

import numpy as np
import pytest

api = pytest.importorskip("scivol._scivolapi")
from tests._oracle import _jax_modules, _jax_student_log_normalizer, _jax_student_log_power  # noqa: E402


def correlation_model(p=1, q=1, density="normal"):
    spec = api.CorrelationSpec()
    spec.p, spec.q = p, q
    spec.density = getattr(api.CorrelationDensity, density)
    return api.CorrelationModel(spec)


def simulate(T, N, a=0.04, b=0.93, seed=0, eta=0.0):
    rng = np.random.default_rng(seed)
    root = np.linalg.cholesky(0.4 * np.ones((N, N)) + 0.6 * np.eye(N))
    target = root @ root.T
    q = target.copy()
    eps, previous = np.zeros((T, N)), np.zeros(N)
    for t in range(T):
        if t:
            q = (1 - a - b) * target + a * np.outer(previous, previous) + b * q
        d = 1 / np.sqrt(np.diag(q))
        z = rng.standard_normal(N)
        if eta > 0.0:
            z *= np.sqrt((1-2*eta) / rng.gamma(0.5/eta, 2*eta))
        previous = np.linalg.cholesky(q * np.outer(d, d)) @ z
        eps[t] = previous
    return eps


def quantile_reference(u, eta):
    """Independent 70-digit incomplete-beta CDF and implicit derivatives."""
    import mpmath as mp
    from scipy.special import ndtri, stdtrit

    z = float(ndtri(u))
    if eta < 1e-8:
        a = z * (z * z - 3) / 4
        b = z * (5 * z**4 - 8 * z * z - 69) / 96
        c = z * (3 * z**6 - z**4 - 95 * z * z - 267) / 384
        return z + eta * (a + eta * (b + eta * c)), a + eta * (2 * b + 3 * eta * c), 2 * b + 6 * eta * c
    if u == 0.5:
        return 0.0, 0.0, 0.0
    with mp.workdps(70):
        e, probability = mp.mpf(eta), mp.mpf(u)

        def cdf(x, h):
            y = h * x * x / (1 - 2 * h + h * x * x)
            return (1 + mp.sign(x) * mp.betainc(mp.mpf('.5'), 1 / (2 * h), 0, y, regularized=True)) / 2

        def logpdf(x, h):
            return (mp.loggamma((1 / h + 1) / 2) - mp.loggamma(1 / (2 * h))
                    - mp.log(mp.pi * (1 / h - 2)) / 2
                    - (1 / h + 1) * mp.log1p(x * x / (1 / h - 2)) / 2)

        initial = float(stdtrit(1 / eta, u) * np.sqrt(1 - 2 * eta))
        x = mp.findroot(lambda v: cdf(v, e) - probability, (initial * .999, initial * 1.001))
        g = mp.exp(logpdf(x, e))
        first = -mp.diff(lambda h: cdf(x, h), e) / g
        second = (-mp.diff(lambda h: cdf(x, h), e, 2) / g
                  - 2 * mp.diff(lambda h: logpdf(x, h), e) * first
                  - mp.diff(lambda v: logpdf(v, e), x) * first * first)
        return float(x), float(first), float(second)


def oracle(theta, eps, target, p, q, density, probabilities=None, row_output=False):
    """JAX recursion with an independently computed local quantile jet."""
    from scipy.special import ndtr, ndtri

    jax, jnp, jsp = _jax_modules()
    T, N = eps.shape
    u = ndtr(eps) if probabilities is None else probabilities
    normal = eps if probabilities is None else ndtri(u)
    if density == "student_t":
        reference = np.array([quantile_reference(v, theta[-1]) for v in u.ravel()]).reshape(T, N, 3)

    def objective(par):
        a, b = par[:p], par[p:p + q]
        if density == "student_t":
            delta = par[-1] - theta[-1]
            e = reference[..., 0] + reference[..., 1] * delta + .5 * reference[..., 2] * delta * delta
        else:
            e = jnp.asarray(normal)
        qbar = e.T @ e / T if target is None else jnp.asarray(target)
        history, contributions = [qbar], []
        for t in range(T):
            if t:
                qt = (1 - jnp.sum(a) - jnp.sum(b)) * qbar
                for i in range(min(p, t)):
                    qt = qt + a[i] * jnp.outer(e[t - 1 - i], e[t - 1 - i])
                for j in range(min(q, t)):
                    qt = qt + b[j] * history[t - 1 - j]
                history.append(qt)
            qt = history[t]
            d = 1 / jnp.sqrt(jnp.diag(qt))
            r = qt * jnp.outer(d, d)
            logdet = jnp.linalg.slogdet(r)[1]
            quad = e[t] @ jnp.linalg.solve(r, e[t])
            if density == "normal":
                term = 0.5 * (logdet + quad - e[t] @ e[t])
            else:
                eta = par[-1]
                rho = _jax_student_log_normalizer(jnp, jsp, eta) + jnp.log(2 * jnp.pi) / 2 + jnp.log1p(-2 * eta) / 2
                c = (N - N % 2) * rho - sum(jnp.log1p((2 * j + N % 2) * eta) for j in range(N // 2))
                term = c + 0.5 * logdet + _jax_student_log_power(jnp, eta, quad, shift=N + 2) - jnp.sum(_jax_student_log_power(jnp, eta, e[t]**2))
            contributions.append(term)
        return jnp.stack(contributions)

    point = jnp.asarray(theta)
    total = lambda v: jnp.sum(objective(v))
    if row_output:
        return np.asarray(objective(point)), np.asarray(jax.jacfwd(objective)(point)), np.asarray(jax.hessian(total)(point))
    return float(total(point)), np.asarray(jax.grad(total)(point)), np.asarray(jax.hessian(total)(point))


@pytest.mark.parametrize("p,q,density,theta", [
    (1, 1, "normal", [0.05, 0.9]),
    (2, 1, "student_t", [0.03, 0.02, 0.9, 1/7.0]),
    (1, 2, "normal", [0.05, 0.5, 0.4]),
])
def test_correlation_jet_matches_the_oracle(p, q, density, theta):
    eps = simulate(40, 3, seed=1)
    model = correlation_model(p, q, density)
    jet = api.CorrelationJet(model, eps)
    target_theta = np.array(theta)
    if density == "student_t":
        target_theta[-1] = 0.0
    target = np.asarray(jet.target(target_theta))
    np.testing.assert_allclose(target, eps.T @ eps / len(eps), rtol=1e-14)
    theta = np.array(theta)
    out = jet.theta(theta, api.VALUE | api.GRADIENT | api.HESSIAN | api.SCORES | api.TERMS)
    value, gradient, hessian = oracle(theta, eps, None, p, q, density)
    assert out["value"] == pytest.approx(value, rel=1e-11)
    np.testing.assert_allclose(out["gradient"], gradient, rtol=1e-9, atol=1e-10)
    np.testing.assert_allclose(out["hessian"], hessian, rtol=1e-9, atol=1e-9)
    np.testing.assert_allclose(out["scores"].sum(0), out["gradient"], rtol=1e-12, atol=1e-12)
    assert out["terms"].sum() == pytest.approx(out["value"], rel=1e-13)


def test_correlation_paths_are_correlation_matrices():
    eps = simulate(200, 4, seed=2)
    model = correlation_model()
    jet = api.CorrelationJet(model, eps)
    r = jet.theta(np.array([0.05, 0.9]), api.PATHS)["correlations"]
    assert r.shape == (200, 4, 4)
    np.testing.assert_allclose(np.diagonal(r, axis1=1, axis2=2), 1.0, rtol=1e-14)
    np.testing.assert_allclose(r, np.transpose(r, (0, 2, 1)), rtol=1e-14)
    target = np.asarray(jet.target(np.array([.05, .9])))
    d = 1 / np.sqrt(np.diag(target))
    np.testing.assert_allclose(r[0], target * np.outer(d, d), rtol=1e-14)  # Q_0 is the target
    assert np.all(np.linalg.eigvalsh(r) > 0)


def test_domain_and_starts():
    model = correlation_model(1, 1, "student_t")
    assert model.parameter_names == ["a[1]", "b[1]", "eta"]
    assert model.feasible(np.array([0.05, 0.9, 0.2]))
    assert not model.feasible(np.array([0.1, 0.9, 0.2]))  # persistence 1
    assert not model.feasible(np.array([-0.01, 0.9, 0.2]))
    assert not model.feasible(np.array([0.05, 0.9, 0.5]))
    eps = simulate(300, 3)
    options = api.FitOptions()
    options.start = [0.2, 0.9, 0.2]
    with pytest.raises(ValueError, match="domain"):
        api.fit_correlation(model, eps, options)


def test_fit_recovers_the_simulation_and_is_certified():
    from scivol.dcc import DCC

    eps = simulate(2000, 5)
    native = DCC(1, 1).fit_from_residuals(eps)
    assert native.success and native.inference == "available" and native.phases == ("z:slsqp", "theta:newton")
    for estimate, se, truth in zip(native.theta, native.std_errors, (0.04, 0.93)):
        assert abs(estimate - truth) < 4.0 * se
    jet = api.CorrelationJet(native._correlation()[0], eps)
    out = jet.theta(native.theta, api.HESSIAN | api.SCORES)
    np.testing.assert_allclose(native.cov_mle, np.linalg.inv(out["hessian"]), rtol=1e-8)
    s = out["scores"]
    inverse = np.linalg.inv(out["hessian"])
    np.testing.assert_allclose(native.cov_robust, inverse @ (s.T @ s) @ inverse, rtol=1e-8)


def test_student_t_fit_recovers_the_tail():
    from scivol.dcc import DCC

    eps = simulate(3000, 4, seed=5, eta=1/6.0)
    eps /= eps.std(0)
    marginal = api.InnovationDensity(api.DensityKind.student_t, [1 / 6])
    r = DCC(1, 1, distribution="studentt").fit_from_residuals(eps, marginals=[marginal] * 4)
    assert r.success and 1/9.0 < r.params.eta < 1/4.0
    assert r.params.persistence == pytest.approx(0.97, abs=0.03)


def test_dcc_fit_with_native_marginals():
    pd = pytest.importorskip("pandas")
    from scivol import GARCH, Normal
    from scivol.result import EstimationResult
    from scivol.dcc import DCC

    eps = simulate(1500, 3, seed=7)
    frame = pd.DataFrame(0.01 * eps, index=pd.date_range("2000", periods=1500), columns=["a", "b", "c"])
    r = DCC(1, 1).fit(frame, univariate_spec=GARCH(1, 1) + Normal(), n_jobs=1)
    assert all(isinstance(m, EstimationResult) for m in r.univariate_results)
    assert r.corr("a", "b").index[0] == frame.index[0]
    assert r.forecast_correlation(4).shape == (4, 3, 3)


# Native multivariate paths: correlations, copula innovations, marginals.

@pytest.fixture(scope="module")
def dcc_fit():
    pd = pytest.importorskip("pandas")
    from scivol import GARCH, GJRGARCH, Normal, SkewT
    from scivol.dcc import DCC

    eps = simulate(1200, 3, seed=11)
    frame = pd.DataFrame(0.01 * eps, index=pd.date_range("2000", periods=1200), columns=["a", "b", "c"])
    gaussian = DCC(1, 1).fit(frame, univariate_spec=GARCH(1, 1) + Normal(), n_jobs=1)
    skewed = DCC(1, 1, distribution="studentt").fit(frame, univariate_spec=GJRGARCH(1, 1) + SkewT(),
                                                    n_jobs=1)
    return gaussian, skewed


def test_one_step_correlation_is_the_recursion(dcc_fit):
    gaussian, _ = dcc_fit
    eps = np.asarray(gaussian._eps)
    extended = np.vstack([eps, np.ones((1, gaussian.N))])  # R_{T+1} does not depend on row T+1
    jet = api.CorrelationJet(gaussian._correlation()[0], extended, gaussian.Qbar)
    r_next = jet.theta(np.asarray(gaussian.theta), api.PATHS)["correlations"][-1]
    np.testing.assert_allclose(gaussian.forecast_correlation(1)[0], r_next, rtol=1e-13)
    s = gaussian.scenarios(4, seed=1)
    np.testing.assert_allclose(s.correlation, r_next, rtol=1e-13)


def test_gaussian_scenarios_have_the_forecast_correlation(dcc_fit):
    gaussian, _ = dcc_fit
    s = gaussian.scenarios(200_000, seed=2)
    np.testing.assert_allclose(np.corrcoef(s.innovations.T), s.correlation, atol=0.01)
    np.testing.assert_allclose(s.residuals, s.innovations * np.sqrt(s.residual_variance), rtol=1e-14)
    np.testing.assert_allclose(s.data, s.mean + s.residuals, rtol=1e-14)


def test_copula_innovations_follow_each_marginal_density(dcc_fit):

    _, skewed = dcc_fit
    s = skewed.scenarios(100_000, seed=3)
    for i, marginal in enumerate(skewed.univariate_results):
        density = marginal._native()[0].innovation_density(np.asarray(marginal.params))
        z = np.sort(s.innovations[:, i])
        lower, _ = api.distribution(density.density, density.shape, np.ascontiguousarray(z))
        ks = np.max(np.abs(lower - (np.arange(1, z.size + 1) - 0.5) / z.size))
        assert ks < 1.63 / np.sqrt(z.size) * 1.5  # well inside the 1% Kolmogorov bound


def test_simulated_paths_continue_every_marginal(dcc_fit):
    gaussian, skewed = dcc_fit
    for result in (gaussian, skewed):
        path = result.simulate(50, seed=4)
        assert path.data.shape == (50, 3) and path.correlation.shape == (50, 3, 3)
        np.testing.assert_allclose(path.residuals, np.sqrt(path.sigma2) * path.innovations, rtol=1e-14)
        first = [m.forecast(1) for m in result.univariate_results]
        np.testing.assert_allclose(path.sigma2[0], [f.residual_variance[0] for f in first], rtol=1e-13)
        np.testing.assert_allclose(path.mean[0], [f.mean[0] for f in first], rtol=1e-12, atol=1e-15)
        a = result.simulate(20, seed=9)
        np.testing.assert_array_equal(a.data, result.simulate(20, seed=9).data)


def test_forecast_covariance_is_the_average_outer_product(dcc_fit):
    gaussian, _ = dcc_fit
    f = gaussian.forecast(5, simulations=20_000, seed=5)
    np.testing.assert_allclose(np.diagonal(f.residual_covariance, axis1=1, axis2=2), f.residual_variance, rtol=0.05)
    assert np.all(np.linalg.eigvalsh(f.residual_covariance) > 0)
    np.testing.assert_allclose(f.correlation[0], gaussian.forecast_correlation(1)[0], rtol=1e-11)  # a mean of equal matrices


@pytest.mark.parametrize("coordinates", ["automatic", "theta"])
def test_gaussian_copula_tail_optimum_is_eta_zero(coordinates):
    from scivol.dcc import DCC

    eps = simulate(2500, 3, seed=0)
    options = api.FitOptions()
    options.coordinates = getattr(api.Coordinates, coordinates)
    model = correlation_model(density="student_t")
    student = api.fit_correlation(model, eps, options)
    normal = api.fit_correlation(correlation_model(), eps, options)
    assert student.success and normal.success
    assert student.theta[-1] == 0.0
    assert student.phases[-1].name == "theta:newton"
    assert student.log_likelihood == pytest.approx(normal.log_likelihood, rel=0.0, abs=2e-10)
    assert model.innovation_density(np.asarray(student.theta)).derived == {"nu": np.inf}



@pytest.mark.parametrize("eta", [0., 1e-300, 1e-16, 1e-10, 1e-6, .1, .49])
def test_copula_eta_limit_matches_ad(eta):
    eps = simulate(10, 3, seed=8)
    model = correlation_model(density="student_t")
    jet = api.CorrelationJet(model, eps)
    theta = np.array([.05, .9, eta])
    out = jet.theta(theta, 7)
    value, gradient, hessian = oracle(theta, eps, None, 1, 1, "student_t")
    assert out["value"] == pytest.approx(value, abs=1e-12)
    np.testing.assert_allclose(out["gradient"], gradient, rtol=1e-11, atol=1e-11)
    np.testing.assert_allclose(out["hessian"], hessian, rtol=1e-11, atol=1e-10)


def test_copula_normal_limit_simulation_uses_no_chi_square_draws():
    eps = simulate(20, 2, seed=4)
    normals = np.random.default_rng(5).normal(size=(4, 3, 2))
    marginals = [api.InnovationDensity(api.DensityKind.student_t, [0.])] * 2
    model = correlation_model(density="student_t")
    normal = correlation_model()
    first = api.simulate_correlation(model, np.array([.05,.9,0.]), eps, marginals, normals, None, None)
    second = api.simulate_correlation(normal, np.array([.05,.9]), eps,
                                     [api.InnovationDensity(api.DensityKind.normal, [])]*2, normals, None, None)
    np.testing.assert_array_equal(first["innovation"], second["innovation"])


@pytest.mark.parametrize("density", ["normal", "student_t"])
@pytest.mark.parametrize("fixed_target", [False, True])
def test_copula_non_normal_marginals_and_sparse_requests(density, fixed_target):
    from scipy.special import ndtri
    from scivol.dcc import CCC

    eps = simulate(8, 2, seed=17)
    marginals = [api.InnovationDensity(api.DensityKind.ged, [1.4]),
                 api.InnovationDensity(api.DensityKind.skew_t, [.17, -.25])]
    u = np.column_stack([api.distribution(m.density, m.shape, np.ascontiguousarray(eps[:, i]))[0]
                         for i, m in enumerate(marginals)])
    target = np.array([[1.1, .3], [.3, .9]]) if fixed_target else None
    theta = np.array([.07, .85] + ([.12] if density == "student_t" else []))
    jet = api.CorrelationJet(correlation_model(density=density), eps, target, marginals)
    out = jet.theta(theta, 63)
    assert out["status"] == "ok"
    f, g, h = oracle(theta, eps, target, 1, 1, density, probabilities=u)
    assert out["value"] == pytest.approx(f, abs=3e-12)
    np.testing.assert_allclose(out["gradient"], g, rtol=1e-10, atol=3e-10)
    np.testing.assert_allclose(out["hessian"], h, rtol=1e-10, atol=3e-9)
    for mask in range(1, 64):
        sparse = jet.theta(theta, mask)
        cold = api.CorrelationJet(correlation_model(density=density), eps, target, marginals).theta(theta, mask)
        assert sparse["status"] == "ok" and sparse["valid"] == mask
        assert cold["status"] == "ok" and cold["valid"] == mask
        for bit, key in [(1, "value"), (2, "gradient"), (4, "hessian"), (8, "correlations"), (16, "scores"), (32, "terms")]:
            if mask & bit:
                np.testing.assert_array_equal(sparse[key], out[key])
                np.testing.assert_array_equal(cold[key], out[key])
            else:
                assert key not in sparse
    model = correlation_model(density=density)
    z = model.to_z(theta)
    transformed = jet.z(z, 7)
    for mask in range(1, 8):
        sparse = jet.z(z, mask)
        for bit, key in [(1, "value"), (2, "gradient"), (4, "hessian")]:
            if mask & bit:
                np.testing.assert_array_equal(sparse[key], transformed[key])
    if density == "normal":
        scores = ndtri(u)
        expected = target if fixed_target else scores.T @ scores / len(scores)
        np.testing.assert_allclose(jet.target(theta), expected, rtol=2e-14, atol=2e-14)
        ccc = CCC().fit_from_residuals(eps, Qbar=target, marginals=marginals)
        np.testing.assert_allclose(ccc.Qbar, expected, rtol=2e-14, atol=2e-14)
        np.testing.assert_allclose(ccc.Rt, np.broadcast_to(ccc.params.corr, ccc.Rt.shape), rtol=2e-14)


def test_student_copula_likelihood_stays_bounded_at_the_upper_face():
    eps = simulate(3000, 4, seed=5, eta=1 / 6)
    eps /= eps.std(0)
    marginal = api.InnovationDensity(api.DensityKind.student_t, [1 / 6])
    jet = api.CorrelationJet(correlation_model(density="student_t"), eps, marginals=[marginal] * 4)
    values = [jet.theta(np.array([.04, .93, eta]), api.VALUE)["value"]
              for eta in [1 / 6, .49, .4999, .49999999]]
    assert np.all(np.isfinite(values))
    assert min(values) > -2000
    assert abs(values[-1] - values[-2]) < 10


def test_unrepresentable_marginal_probabilities_are_rejected():
    eps = np.array([[40., 1.], [-40., 2.]])
    with pytest.raises(ValueError, match="probability rounded to zero or one"):
        api.CorrelationJet(correlation_model(), eps, np.eye(2))
    with pytest.raises(ValueError, match="one marginal density per series"):
        api.CorrelationJet(correlation_model(), np.ones((3, 2)), np.eye(2),
                           [api.InnovationDensity(api.DensityKind.normal)])


def test_gaussian_normal_marginals_preserve_head_values_bitwise():
    eps = np.array([[1., .7], [-.3, .4], [.6, -1.1], [1.3, .2], [-.8, .9], [.1, -.4]])
    out = api.CorrelationJet(correlation_model(), eps).theta(np.array([.04, .9]), 7)
    # Offline output from e61fbb4, compiled with the same IEEE-safe flags.
    assert out["value"] == float.fromhex('-0x1.01af2e798f468p-5')
    np.testing.assert_array_equal(out["gradient"], [float.fromhex(v) for v in
                                  ['0x1.820e3ce7e2d9ap+0', '0x1.5a89399a9e286p-4']])
    np.testing.assert_array_equal(out["hessian"].ravel(), [float.fromhex(v) for v in
                                  ['-0x1.93e1c41114508p+0', '0x1.114b6b343ef09p+1',
                                   '0x1.114b6b343ef09p+1', '0x1.02eb21336e829p-2']])


@pytest.mark.parametrize("density", ["normal", "student_t"])
def test_simulation_recurses_on_copula_scores(density):
    eps = simulate(12, 2, seed=13)
    model = correlation_model(density=density)
    theta = np.array([.07, .85] + ([.12] if density == "student_t" else []))
    marginals = [api.InnovationDensity(api.DensityKind.ged, [1.4]),
                 api.InnovationDensity(api.DensityKind.skew_t, [.17, -.25])]
    target = np.array([[1.1, .3], [.3, .9]])
    normals = np.random.default_rng(3).normal(size=(1, 5, 2))
    chi2 = np.ones((1, 5))
    path = api.simulate_correlation(model, theta, eps, marginals, normals, chi2, target)
    extended = np.ascontiguousarray(np.vstack([eps, path["innovation"][0]]))
    filtered = api.CorrelationJet(model, extended, target, marginals).theta(theta, api.PATHS)
    np.testing.assert_allclose(path["correlation"][0], filtered["correlations"][-5:], rtol=2e-11, atol=2e-12)


def test_copula_z_derivatives_include_quantile_and_target_curvature():
    jax, jnp, _ = _jax_modules()
    model = correlation_model(2, 1, "student_t")
    eps = simulate(8, 2, seed=29)
    theta = np.array([.03, .04, .85, .12])
    z = model.to_z(theta)
    f, g, h = oracle(theta, eps, None, 2, 1, "student_t")
    upper = np.nextafter(.5, 0.)
    limit = np.nextafter(1., 0.)

    def chart(v):
        weights = jnp.exp(v[:3])
        coefficients = limit * weights / (1 + jnp.sum(weights))
        tail = upper / (1 + jnp.exp(-v[3]))
        return jnp.concatenate([coefficients, jnp.array([tail])])

    jacobian = np.asarray(jax.jacfwd(chart)(jnp.asarray(z)))
    curvature = np.asarray(jax.hessian(lambda v: jnp.dot(jnp.asarray(g), chart(v)))(jnp.asarray(z)))
    out = api.CorrelationJet(model, eps).z(z, 7)
    assert out["value"] == pytest.approx(f, abs=3e-12)
    np.testing.assert_allclose(out["gradient"], jacobian.T @ g, rtol=2e-10, atol=2e-11)
    np.testing.assert_allclose(out["hessian"], jacobian.T @ h @ jacobian + curvature, rtol=2e-9, atol=2e-10)


def test_student_target_is_checked_at_the_evaluated_eta():
    z = np.array([-.9, -.4, .1, .6, 1.2])
    eps = np.column_stack([z, 2 * z])
    jet = api.CorrelationJet(correlation_model(density="student_t"), eps)
    out = jet.theta(np.array([.05, .9, .2]), 7)
    assert out["status"] == "ok"
    assert np.linalg.det(np.asarray(jet.target(np.array([.05, .9, .2])))) > 0


def test_matching_student_marginals_preserve_the_shape_derivatives():
    eps = simulate(8, 2, seed=23, eta=.1)
    marginal = api.InnovationDensity(api.DensityKind.student_t, [.1])
    u = np.column_stack([api.distribution(marginal.density, marginal.shape,
                                        np.ascontiguousarray(eps[:, i]))[0] for i in range(2)])
    theta = np.array([.04, .9, .1])
    jet = api.CorrelationJet(correlation_model(density="student_t"), eps, marginals=[marginal] * 2)
    out = jet.theta(theta, 7)
    f, g, h = oracle(theta, eps, None, 1, 1, "student_t", probabilities=u)
    assert out["value"] == pytest.approx(f, abs=3e-12)
    np.testing.assert_allclose(jet.target(theta), eps.T @ eps / len(eps), rtol=2e-14, atol=2e-14)
    np.testing.assert_allclose(out["gradient"], g, rtol=2e-10, atol=2e-10)
    np.testing.assert_allclose(out["hessian"], h, rtol=2e-10, atol=2e-9)


def test_target_is_a_function_of_theta_before_and_after_evaluation():
    eps = simulate(300, 3, seed=41)
    model = correlation_model(density="student_t")
    jet = api.CorrelationJet(model, eps)
    theta = np.array([.04, .9, .12])
    expected = jet.target(theta)
    for eta in [.121, .4999, 0., 1e-16, .3]:
        assert jet.theta(np.array([.04, .9, eta]), 7)["status"] == "ok"
        np.testing.assert_allclose(jet.target(theta), expected, rtol=2e-13, atol=2e-14)
    with pytest.raises(ValueError, match="target"):
        jet.target(np.array([.04, .9, .5]))
