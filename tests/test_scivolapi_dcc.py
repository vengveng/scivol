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


def oracle(theta, eps, target, p, q, density):
    """F, G, H of the DCC correlation likelihood by JAX (Q_0 = target; absent pre-sample lags)."""
    jax, jnp, jsp = _jax_modules()
    T, N = eps.shape
    e = jnp.asarray(eps)

    def objective(par):
        a, b = par[:p], par[p:p + q]
        history, total = [jnp.asarray(target)], 0.0
        for t in range(T):
            if t:
                qt = (1 - jnp.sum(a) - jnp.sum(b)) * target
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
                total = total + 0.5 * (logdet + quad - e[t] @ e[t])
            else:
                eta = par[-1]
                rho = _jax_student_log_normalizer(jnp,jsp,eta) + jnp.log(2*jnp.pi)/2 + jnp.log1p(-2*eta)/2
                c = (N-N%2)*rho - sum(jnp.log1p((2*j+N%2)*eta) for j in range(N//2))
                total = total + c + 0.5*logdet + _jax_student_log_power(jnp,eta,quad,shift=N+2) - jnp.sum(_jax_student_log_power(jnp,eta,e[t]**2))

        return total

    point = jnp.asarray(theta)
    return float(objective(point)), np.asarray(jax.grad(objective)(point)), np.asarray(jax.hessian(objective)(point))


@pytest.mark.parametrize("p,q,density,theta", [
    (1, 1, "normal", [0.05, 0.9]),
    (2, 1, "student_t", [0.03, 0.02, 0.9, 1/7.0]),
    (1, 2, "normal", [0.05, 0.5, 0.4]),
])
def test_correlation_jet_matches_the_oracle(p, q, density, theta):
    eps = simulate(40, 3, seed=1)
    model = correlation_model(p, q, density)
    jet = api.CorrelationJet(model, eps)
    target = np.asarray(jet.target)
    np.testing.assert_allclose(target, eps.T @ eps / len(eps), rtol=1e-14)
    theta = np.array(theta)
    out = jet.theta(theta, api.VALUE | api.GRADIENT | api.HESSIAN | api.SCORES | api.TERMS)
    value, gradient, hessian = oracle(theta, eps, target, p, q, density)
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
    target = np.asarray(jet.target)
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
    r = DCC(1, 1, distribution="studentt").fit_from_residuals(eps)
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
    value, gradient, hessian = oracle(theta, eps, np.asarray(jet.target), 1, 1, "student_t")
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
