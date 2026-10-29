"""Native post-estimation statistics against independent reference calculations."""

from __future__ import annotations

import ast
from pathlib import Path
import subprocess
import sys

import mpmath as mp
import numpy as np
import pytest
from scipy import stats
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.regression.linear_model import OLS
from statsmodels.stats.sandwich_covariance import cov_hac

from scivol import ARMA, GARCH, GJRGARCH, Normal, SkewT, StudentT
from scivol._fit import native_spec
from scivol import _scivolapi as api
from tests._oracle import _jax_modules
from tests.test_scivolapi_mean import mean_spec, mean_point, model_for, row_terms, volatility_point


@pytest.mark.parametrize("df", [0.25, 1, 2, 5, 25, 300, 2000])
def test_reference_distribution_tails(df):
    for x in [-12.0, -0.1, 0.0, 0.1, 3.0, 12.0, 30.0]:
        assert api.student_survival(x, df) == pytest.approx(stats.t.sf(x, df), rel=2e-11, abs=1e-300)
    for x in [0.0, 0.01, 1.0, df, df + 10 * np.sqrt(2 * df), 700.0]:
        assert api.chi_square_survival(x, df) == pytest.approx(stats.chi2.sf(x, df), rel=2e-11, abs=1e-300)


def test_far_tails_against_mpmath_and_explicit_range_failure():
    with mp.workdps(90):
        for x in [8, 20, 35, 37]:
            expected = float(mp.erfc(mp.mpf(x) / mp.sqrt(2)) / 2)
            assert api.normal_survival(x) == pytest.approx(expected, rel=3e-13, abs=0)
        for x, df in [(1000.0, 5.0), (1000.0, 25.0), (1400.0, 2.0)]:
            expected = float(mp.gammainc(mp.mpf(df) / 2, mp.mpf(x) / 2, mp.inf, regularized=True))
            assert api.chi_square_survival(x, df) == pytest.approx(expected, rel=3e-12, abs=1e-320)
        x, df = mp.mpf('1e30'), mp.mpf(8)
        expected = float(mp.betainc(df / 2, mp.mpf('.5'), 0, df / (df + x*x), regularized=True) / 2)
        assert api.student_survival(float(x), float(df)) == pytest.approx(expected, rel=5e-13, abs=0)
    with pytest.raises((ValueError, OverflowError)):
        api.normal_survival(np.nan)
    with pytest.raises((ValueError, OverflowError)):
        api.normal_survival(50.0)
    with pytest.raises((ValueError, OverflowError)):
        api.chi_square_survival(1450.0, 2.0)


@pytest.mark.parametrize("df_adjustment", [0, 2, 9])
def test_portmanteau_against_statsmodels(df_adjustment):
    series = np.random.default_rng(63).normal(size=700)
    reference = acorr_ljungbox(series, lags=[8], model_df=df_adjustment, boxpierce=True, return_df=True).iloc[0]
    test = api.portmanteau(series, 8, df_adjustment)
    if df_adjustment < 8:
        assert test["q_stat"] == pytest.approx(reference.lb_stat, rel=1e-13)
        assert test["p_value"] == pytest.approx(reference.lb_pvalue, rel=1e-13)
        assert test["box_pierce"]["statistic"] == pytest.approx(reference.bp_stat, rel=1e-13)
        assert test["box_pierce"]["p_value"] == pytest.approx(reference.bp_pvalue, rel=1e-13)
    else:
        assert test["status"] == "not_estimable"
        assert np.isnan(test["p_value"])


def test_cell_counts_and_sign_bias_reference():
    rng = np.random.default_rng(81)
    pit = np.r_[0.0, rng.uniform(size=800), 1.0]
    counts = np.histogram(pit, np.linspace(0, 1, 21))[0]
    reference = stats.chisquare(counts)
    actual = api.dgt(pit, 20)
    assert actual["statistic"] == pytest.approx(reference.statistic, rel=1e-14)
    assert actual["p_value"] == pytest.approx(reference.pvalue, rel=1e-14)
    e = rng.normal(size=800)
    z = e * np.sqrt(1 + 0.3 * np.r_[0, e[:-1] < 0])
    response = z[1:] ** 2
    indicator = (e[:-1] < 0).astype(float)
    regressors = np.column_stack([indicator, indicator * e[:-1], (1 - indicator) * e[:-1]])
    actual = api.sign_bias(z, e)
    for column, key in enumerate(["sign", "negative_size", "positive_size"]):
        fitted = OLS(response, np.column_stack([np.ones(len(response)), regressors[:, column]])).fit()
        assert actual[key]["statistic"] == pytest.approx(fitted.tvalues[1], rel=2e-12)
        assert actual[key]["p_value"] == pytest.approx(fitted.pvalues[1], rel=2e-12)
    fitted = OLS(response, np.column_stack([np.ones(len(response)), regressors])).fit()
    assert actual["joint"]["statistic"] == pytest.approx(len(response) * fitted.rsquared, rel=2e-12)


def test_underflow_is_explicit_and_does_not_hide_rejection():
    test = api.dgt(np.full(2000, .5), 40)
    assert test['status'] == 'probability_underflow'
    assert np.isnan(test['p_value'])
    assert np.isfinite(test['statistic']) and test['log_p_value'] < -1000
    assert test['reject']


def test_berkowitz_conditional_likelihood_reference():
    rng = np.random.default_rng(731)
    scores = rng.normal(size=500)
    for t in range(1, len(scores)):
        scores[t] += 0.14 + 0.23 * scores[t-1]
    pit = stats.norm.cdf(scores)
    transformed = stats.norm.ppf(pit)
    fitted = OLS(transformed[1:], np.column_stack([np.ones(len(scores)-1), transformed[:-1]])).fit()
    alternative = np.sum(stats.norm.logpdf(fitted.resid, scale=np.sqrt(np.mean(fitted.resid**2))))
    null = np.sum(stats.norm.logpdf(transformed[1:]))
    test = api.berkowitz(pit)
    assert test["statistic"] == pytest.approx(2*(alternative-null), rel=5e-13)
    assert test["p_value"] == pytest.approx(stats.chi2.sf(2*(alternative-null), 3), rel=5e-13)
    assert test["ar1"] == pytest.approx(fitted.params[1], rel=1e-13)
    with pytest.raises(ValueError, match="strictly"):
        api.berkowitz(np.r_[0.0, pit])


@pytest.mark.parametrize("horizon,bandwidth", [(1, -1), (4, -1), (2, 9), (1, 0)])
def test_dm_hac_hln_reference(horizon, bandwidth):
    difference = np.random.default_rng(52).normal(loc=0.05, size=400)
    for t in range(1, len(difference)):
        difference[t] += 0.6*difference[t-1]
    n = len(difference)
    lag = max(int(n**(1/3)), horizon-1) if bandwidth == -1 else bandwidth
    centered = difference - difference.mean()
    variance = centered @ centered / n
    for k in range(1, lag+1):
        variance += 2*(1-k/(lag+1)) * (centered[k:] @ centered[:-k]) / n
    statistic = difference.mean()/np.sqrt(variance/n) * np.sqrt((n+1-2*horizon+horizon*(horizon-1)/n)/n)
    actual = api.diebold_mariano(difference, np.zeros(n), horizon, bandwidth)
    fitted = OLS(difference, np.ones((n, 1))).fit()
    assert actual["long_run_variance"] / n == pytest.approx(cov_hac(fitted, nlags=lag, use_correction=False)[0, 0], rel=2e-12)
    assert actual["statistic"] == pytest.approx(statistic, rel=1e-13)
    assert actual["p_value"] == pytest.approx(2*stats.t.sf(abs(statistic), n-1), rel=2e-12)
    assert actual["long_run_variance"] == pytest.approx(variance, rel=1e-13)
    with pytest.raises(ValueError, match="finite"):
        api.diebold_mariano(np.r_[difference[:-1], np.nan], np.zeros(n))


@pytest.mark.parametrize("volatility,p,q,density,with_mean", [
    ("garch", 1, 1, "normal", False), ("garch", 1, 2, "normal", True),
    ("garch", 2, 1, "student_t", True), ("gjr", 2, 1, "skew_t", True),
    ("egarch", 1, 1, "normal", True), ("egarch", 2, 2, "student_t", True),
    ("egarch", 1, 1, "skew_t", True), ("aparch", 1, 2, "ged", True),
    ("constant", 0, 0, "student_t", True),
    ("garch", 1, 1, "generalized_hyperbolic", True), ("garch", 2, 1, "nig", False),
    ("gjr", 1, 1, "hyperbolic", True), ("garch", 1, 2, "variance_gamma", True),
])
def test_path_sensitivities_against_jax(volatility, p, q, density, with_mean):
    jax, jnp, _ = _jax_modules()
    rng = np.random.default_rng(91)
    y, x = rng.normal(size=85), rng.normal(size=(85, 1))
    mean = mean_spec(constant=with_mean, ar=int(with_mean), exog=int(with_mean), ma=int(with_mean))
    model = model_for(mean, volatility, density, p, q)
    theta = np.array(mean_point(mean) + volatility_point(volatility, p, q, density))
    jet = api.Jet(model, y, x if with_mean else None)
    function = row_terms(y, x, mean, volatility, p, q, density, return_paths=True)
    reference = np.asarray(jax.jacfwd(function)(jnp.asarray(theta)))
    mask = api.LOG_VARIANCE_SENSITIVITIES | api.RESIDUAL_SENSITIVITIES
    actual = jet.theta(theta, mask)
    assert actual["valid"] == mask
    np.testing.assert_allclose(actual["log_variance_sensitivities"], reference[0], rtol=3e-10, atol=3e-11)
    np.testing.assert_allclose(actual["residual_sensitivities"], reference[1], rtol=3e-12, atol=3e-12)
    full = jet.theta(theta, 255)
    assert full["valid"] == 255
    for request in [64, 128, 192, 65, 194, 252, 255]:
        sparse = jet.theta(theta, request)
        assert sparse["valid"] == request
        for key in ("log_variance_sensitivities", "residual_sensitivities"):
            if key in sparse:
                np.testing.assert_array_equal(sparse[key], full[key])
    jax.clear_caches()


def test_z_sensitivities_and_every_sparse_request():
    jax, jnp, _ = _jax_modules()
    mean = mean_spec(ar=1, ma=1)
    y = np.random.default_rng(701).normal(size=100)
    x = np.empty((len(y), 0))
    model = model_for(mean, "garch", "normal")
    theta = np.array([0.05, 0.3, 0.25, 0.08, 0.1, 0.8])
    z = model.to_z(theta)
    bound = np.nextafter(1.0, 0.0)
    def chart(u):
        persistence = (1 - 8*3*np.finfo(float).eps)*jax.nn.softmax(jnp.r_[u[4:6], 0.0])[:-1]
        return jnp.r_[u[0], bound*jnp.tanh(u[1]), -bound*jnp.tanh(u[2]),
                      np.finfo(float).tiny+jnp.exp(u[3]), persistence]
    np.testing.assert_allclose(chart(jnp.asarray(z)), theta, rtol=1e-13)
    function = row_terms(y, x, mean, "garch", 1, 1, "normal", return_paths=True)
    reference = np.asarray(jax.jacfwd(lambda u: function(chart(u)))(jnp.asarray(z)))
    jet = api.Jet(model, y)
    full = jet.z(z, 255 ^ api.SCORES)
    np.testing.assert_allclose(full["log_variance_sensitivities"], reference[0], rtol=2e-11, atol=2e-12)
    np.testing.assert_allclose(full["residual_sensitivities"], reference[1], rtol=2e-11, atol=2e-12)
    for request in range(1, 256):
        if request & api.SCORES:
            continue
        sparse = jet.z(z, request)
        assert sparse["valid"] == request
        for key in sparse.keys() - {"valid", "status"}:
            np.testing.assert_array_equal(sparse[key], full[key])


def test_long_near_boundary_sensitivities_and_kink_status():
    jax, jnp, _ = _jax_modules()
    mean = mean_spec(ar=1, ma=1)
    y = np.random.default_rng(827).normal(size=3000)
    theta = np.array([.05, .98, .99, 1e-8, .00001, .9999])
    model = model_for(mean, 'garch', 'normal')
    reference = np.asarray(jax.jacfwd(row_terms(y, np.empty((len(y), 0)), mean, 'garch', 1, 1,
                                              'normal', return_paths=True))(jnp.asarray(theta)))
    output = api.Jet(model, y).theta(theta, api.LOG_VARIANCE_SENSITIVITIES | api.RESIDUAL_SENSITIVITIES)
    np.testing.assert_allclose(output['log_variance_sensitivities'], reference[0], rtol=4e-10, atol=2e-8)
    np.testing.assert_allclose(output['residual_sensitivities'], reference[1], rtol=4e-10, atol=2e-8)
    # An absolute-value variance recursion has no mean derivative at e_t=0.
    mean = mean_spec()
    model = model_for(mean, 'egarch', 'normal')
    kink_theta = np.array([y[30], .02, .15, -.08, .9])
    output = api.Jet(model, y[:80]).theta(kink_theta, api.LOG_VARIANCE_SENSITIVITIES | api.RESIDUAL_SENSITIVITIES)
    assert not output['valid'] & api.LOG_VARIANCE_SENSITIVITIES
    assert output['valid'] & api.RESIDUAL_SENSITIVITIES


def test_status_enum_keeps_public_string_statuses():
    status = api.TestStatus.ok
    assert status.name == "ok"
    test = api.portmanteau(np.random.default_rng(71).normal(size=100), 5)
    assert isinstance(test['status'], str)
    assert test['status'] == status.name
    assert api.TestStatus[test['status']] == status


def test_package_does_not_import_scipy():
    for path in Path('scivol').rglob('*.py'):
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                assert not any(alias.name.startswith('scipy') for alias in node.names), path
            if isinstance(node, ast.ImportFrom):
                assert not (node.module or '').startswith('scipy'), path
    script = '''
import importlib.abc
import sys
class BlockScipy(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == 'scipy' or fullname.startswith('scipy.'):
            raise ImportError('SciPy is unavailable in this runtime check')
sys.meta_path.insert(0, BlockScipy())
import numpy as np
from scivol import GARCH, Normal
from scivol._scivolapi import diebold_mariano
result = (GARCH() + Normal()).fit(np.random.default_rng(53).normal(size=350))
result.diagnostic_tests(print_results=False)
result.summary()
assert np.isfinite(result.aic)
diebold_mariano(np.arange(100, dtype=float), np.zeros(100))
'''
    subprocess.run([sys.executable, '-c', script], check=True, capture_output=True, text=True)


def test_result_diagnostics_preserve_public_keys_and_add_adequacy_tests():
    y = np.random.default_rng(39).normal(size=600)
    result = (ARMA(1, 0) + GARCH() + Normal()).fit(y)
    tests = result.diagnostic_tests(lags=8, print_results=False)
    assert tests['n_obs'] == len(y)-1
    assert tests['df_adjustment'] == 1
    assert tests['residual_ljung_box']['df'] == 7
    assert tests['ljung_box'][1]['df'] == 8
    assert {'dgt', 'ljung_box', 'pit', 'li_mak', 'berkowitz', 'sign_bias'} <= tests.keys()
    assert tests['dgt']['df'] == 39
    assert 'not calibrated' in tests['pit_calibration']


def mcs_reference(losses, uniforms, length, method, alpha):
    n, models = losses.shape
    indices = np.array([[(int(row[t//length]*n) + t % length) % n for t in range(n)] for row in uniforms])
    means = losses.mean(axis=0)
    bootstrap = losses[indices].mean(axis=1) - means
    active = list(range(models))
    pvalues = np.ones(models)
    included, eliminated = None, []
    monotone, final = 0.0, 1.0
    while len(active) > 1:
        mu, boot = means[active], bootstrap[:, active]
        if method == 'max':
            mu, boot = mu-mu.mean(), boot-boot.mean(axis=1, keepdims=True)
            se = boot.std(axis=0, ddof=1)
            observed = mu/se
            simulated = (boot/se).max(axis=1)
        else:
            mu = mu[:, None]-mu[None, :]
            boot = boot[:, :, None]-boot[:, None, :]
            se = boot.std(axis=0, ddof=1)
            np.fill_diagonal(se, 1.0)
            observed = (mu/se).max(axis=1)
            simulated = np.abs(boot/se).max(axis=(1,2))
        worst = int(np.argmax(observed))
        pvalue = (1 + np.count_nonzero(simulated >= observed[worst]))/(len(uniforms)+1)
        monotone = max(monotone, pvalue)
        if included is None and monotone >= alpha:
            included, final = active.copy(), monotone
        pvalues[active[worst]] = monotone
        if included is None:
            eliminated.append(active[worst])
        active.pop(worst)
    return dict(included=active if included is None else included, eliminated=eliminated,
                p_values=pvalues, final_p_value=final)


@pytest.mark.parametrize('method', ['max', 'R'])
def test_mcs_against_independent_bootstrap_reference(method):
    rng = np.random.Generator(np.random.PCG64(82))
    losses = np.ascontiguousarray(rng.normal(size=(180, 5)) + [0, .02, .1, .4, .5])
    uniforms = rng.random((599, 30))
    reference = mcs_reference(losses, uniforms, 6, method, .1)
    actual = api.model_confidence_set(losses, uniforms, 6, .1, method)
    assert actual['included'] == reference['included']
    assert actual['eliminated'] == reference['eliminated']
    np.testing.assert_array_equal(actual['p_values'], reference['p_values'])
    assert actual['final_p_value'] == reference['final_p_value']
    repeat = api.model_confidence_set(losses, uniforms, 6, .1, method)
    np.testing.assert_array_equal(repeat['p_values'], actual['p_values'])
    scaled = api.model_confidence_set(7*losses + 100, uniforms, 6, .1, method)
    np.testing.assert_array_equal(scaled['p_values'], actual['p_values'])


@pytest.mark.parametrize('method', ['max', 'R'])
def test_mcs_degenerate_contrasts_have_exact_ordering(method):
    uniforms = np.random.default_rng(4).random((199, 20))
    losses = np.zeros((80, 3))
    losses[:, 2] = .5
    actual = api.model_confidence_set(losses, uniforms, 4, .05, method)
    assert actual['included'] == [0, 1]
    assert actual['eliminated'] == [2]
    assert actual['p_values'][0] == actual['p_values'][1] == 1.0
    assert actual['p_values'][2] < .05


def test_li_mak_joint_influence_reference_including_skew_and_qmle():
    mean = mean_spec(ar=1, ma=1)
    model = model_for(mean, 'garch', 'skew_t')
    spec = ARMA(1, 1) + GARCH() + SkewT()
    truth = np.array([.05, .2, .1, .1, .1, .7, 1/8, -.2])
    y = np.ascontiguousarray(spec.simulate(2500, truth, seed=801).data)
    jet = api.Jet(model, y)
    for estimator in [api.Estimator.mle, api.Estimator.qmle]:
        fitted = spec.fit(y, method='qmle' if estimator == api.Estimator.qmle else 'mle')
        theta = np.ascontiguousarray(fitted.params)
        point = jet.theta(theta, 255)
        z, n, lags = point['innovation'], len(point['innovation']), 5
        v = z*z-1
        moment = np.zeros((n, lags))
        derivative = np.zeros((lags, len(theta)))
        for k in range(1, lags+1):
            moment[k:, k-1] = v[k:]*v[:-k]
            derivative[k-1] = -(v[:-k, None]*point['log_variance_sensitivities'][k:]).sum(axis=0)/n
        means = moment.mean(axis=0)
        if estimator == api.Estimator.qmle:
            gaussian = model_for(mean, 'garch', 'normal')
            p = gaussian.dimension
            scores = api.Jet(gaussian, y).theta(theta[:p], api.SCORES | api.HESSIAN)
            a = point['residual_sensitivities'][:, :p] / np.sqrt(point['variance'])[:, None]
            b = point['log_variance_sensitivities'][:, :p]
            np.testing.assert_allclose(scores['scores'], z[:, None]*a - .5*v[:, None]*b, rtol=2e-11, atol=2e-11)
        else:
            p, scores = len(theta), point
        bread = scores['hessian']/n
        # Each row is one time observation, each column a residual lag.
        influence = moment - scores['scores'] @ np.linalg.solve(bread.T, derivative[:, :p].T)
        covariance = np.cov(influence, rowvar=False, ddof=0)
        m = 3  # intercept, AR and MA are jointly estimated
        omitted_mean = moment - scores['scores'][:, m:] @ np.linalg.solve(bread[m:, m:].T, derivative[:, m:p].T)
        assert np.linalg.norm(covariance-np.cov(omitted_mean, rowvar=False, ddof=0)) > 1e-6
        statistic = n * means @ np.linalg.solve(covariance, means)
        actual = api.li_mak(jet, theta, lags, estimator)
        assert actual['status'] == 'ok'
        assert actual['statistic'] == pytest.approx(statistic, rel=2e-10)
        np.testing.assert_allclose(actual['derivative'], derivative, rtol=2e-12, atol=2e-12)
        np.testing.assert_allclose(actual['covariance'], covariance/np.mean(v*v)**2, rtol=2e-10, atol=2e-10)


def test_li_mak_reports_invalid_fourth_moment():
    y = np.random.default_rng(781).normal(size=300)
    model = model_for(mean_spec(constant=False), 'garch', 'student_t')
    assert api.li_mak(api.Jet(model, y), np.array([.1, .1, .8, 1/3]), 5)['status'] == 'fourth_moment_unavailable'


@pytest.mark.slow
@pytest.mark.parametrize('joint_mean', [False, True])
def test_li_mak_monte_carlo_size(joint_mean):
    rng = np.random.Generator(np.random.PCG64(20261026 + joint_mean))
    rejections, uncorrected, usable = 0, 0, 0
    replications, n, lags = 400, 3000, 5
    spec = GARCH() + Normal()
    if joint_mean:
        spec = ARMA(1, 0) + spec
    for _ in range(replications):
        shock = rng.normal(size=n+600)
        y = np.zeros(len(shock))
        h, e = 1.0, 0.0
        for t in range(len(shock)):
            h = .1 + .2*e*e + .7*h
            e = np.sqrt(h)*shock[t]
            y[t] = (.2*y[t-1] if joint_mean and t else 0.0) + e
        result = spec.fit(y[600:], accuracy=1e-9)
        if not result.success:
            continue
        model, sample, x = result._native()
        jet = api.Jet(model, sample, x)
        test = api.li_mak(jet, np.ascontiguousarray(result.params), lags)
        if test['status'] != 'ok':
            continue
        usable += 1
        rejections += test['p_value'] < .05
        uncorrected += api.portmanteau(np.ascontiguousarray(result.std_resid)**2, lags)['box_pierce']['p_value'] < .05
    corrected_size, uncorrected_size = rejections/usable, uncorrected/usable
    print(f'Li-Mak joint_mean={joint_mean}: {rejections}/{usable}={corrected_size:.4f}; Box-Pierce={uncorrected_size:.4f}')
    assert usable >= .95*replications
    assert .02 <= corrected_size <= .085
    assert uncorrected_size < .025
    assert corrected_size-uncorrected_size >= .02


@pytest.mark.parametrize('lags', [1, 10])
def test_arch_lm_against_auxiliary_ols(lags):
    z = np.random.default_rng(250).normal(size=2000)
    squared = z*z
    design = np.column_stack([np.ones(len(z)-lags)] +
                             [squared[lags-k:-k] for k in range(1, lags+1)])
    fit = OLS(squared[lags:], design).fit()
    expected = (len(z)-lags)*fit.rsquared
    actual = api.arch_lm(z, lags)
    assert actual['status'] == 'ok' and actual['df'] == lags
    assert actual['statistic'] == pytest.approx(expected, rel=2e-11)
    assert actual['p_value'] == pytest.approx(stats.chi2.sf(expected, lags), rel=2e-11)
    assert api.arch_lm(np.ones(100), lags)['status'] == 'not_estimable'


@pytest.mark.parametrize('rows,horizons', [(500, [10, 20]), (75, [10, 15]), (40, [8])])
def test_adequacy_reuses_horizons_and_holm_required_family(rows, horizons):
    y = (GARCH()+Normal()).simulate(rows, [.1, .15, .7], seed=351).data
    fitted = GARCH().fit(y)
    model = model_for(mean_spec(constant=False), 'garch', 'normal')
    jet = api.Jet(model, y)
    theta = np.ascontiguousarray(fitted.params)
    tests = api.selection_diagnostics(jet, theta, .01)
    checks = {c['name']: c for c in tests['checks']}
    point = jet.theta(theta, api.PATHS)
    assert len(checks) == 2*len(horizons)+1
    for lag in horizons:
        for name, expected in [('residual_ljung_box', api.portmanteau(point['innovation'], lag)),
                               ('absolute_li_mak', api.absolute_li_mak(jet, theta, lag))]:
            actual = checks[f'{name}[{lag}]']
            assert actual['status'] == expected['status']
            np.testing.assert_allclose(actual['statistic'], expected['statistic'], rtol=2e-10, equal_nan=True)
    def log_p(check):
        return check['log_p_value'] if check['status'] in ('ok', 'probability_underflow') else 0.
    family = sorted(checks.values(), key=log_p)
    rejected = []
    for i, check in enumerate(family):
        if log_p(check) > np.log(.01/(len(family)-i)):
            break
        rejected.append(check['name'])
    assert {c['name'] for c in family if c['holm_reject']} == set(rejected)
    assert not {'berkowitz', 'dgt'} & checks.keys()


@pytest.mark.parametrize('kind,theta,basis', [
    ('garch', [.1, .15, 0.], np.eye(3)[:, :2]),
    ('student', [.1, .15, .7, 0.], np.eye(4)[:, :3]),
    ('gjr', [.1, 0., .1, .7], np.eye(4)[:, [0, 2, 3]]),
    ('gjr', [.1, .1, -.1, .7], np.array([[1., 0., 0.], [0., 1., 0.], [0., -1., 0.], [0., 0., 1.]])),
])
def test_li_mak_conditions_score_and_hessian_on_closed_face(kind, theta, basis):
    spec = GJRGARCH()+Normal() if kind == 'gjr' else GARCH()+(StudentT() if kind == 'student' else Normal())
    theta = np.array(theta)
    y = spec.simulate(4000, theta, seed=533).data
    model = api.Model(native_spec(spec))
    jet = api.Jet(model, y)
    point = jet.theta(theta, api.PATHS | api.SCORES | api.HESSIAN | api.LOG_VARIANCE_SENSITIVITIES)
    v = point['innovation']**2-1
    n, lags = len(v), 5
    moments = np.zeros((n, lags))
    derivative = np.zeros((lags, len(theta)))
    for k in range(1, lags+1):
        moments[k:, k-1] = v[k:]*v[:-k]
        derivative[k-1] = -(v[:-k, None]*point['log_variance_sensitivities'][k:]).sum(axis=0)/n
    means = moments.mean(axis=0)
    bread = basis.T @ point['hessian'] @ basis / n
    influence = moments - (point['scores'] @ basis) @ np.linalg.solve(bread.T, (derivative @ basis).T)
    covariance = np.cov(influence, rowvar=False, ddof=0)
    expected = n*means @ np.linalg.solve(covariance, means)
    actual = api.li_mak(jet, theta, lags)
    assert actual['status'] == 'ok'
    assert actual['statistic'] == pytest.approx(expected, rel=2e-10)
    np.testing.assert_allclose(actual['covariance'], covariance/np.mean(v*v)**2, rtol=2e-10, atol=1e-12)


@pytest.mark.parametrize('eta', [.25, 1/3, .4])
def test_absolute_adequacy_needs_no_fourth_moment(eta):
    spec = GARCH()+StudentT()
    y = spec.simulate(2000, [.1, .1, .8, eta], seed=402).data
    result = spec.fit(y)
    model, sample, x = result._native()
    jet = api.Jet(model, sample, x)
    theta = np.ascontiguousarray(result.params)
    checks = api.selection_diagnostics(jet, theta)
    required = checks['checks']
    assert {c['name'] for c in required} == {
        'residual_ljung_box[10]', 'residual_ljung_box[20]',
        'absolute_li_mak[10]', 'absolute_li_mak[20]', 'absolute_sign_bias'}
    assert all(c['status'] == 'ok' for c in required)
    assert checks['status'] != 'adequacy_unresolved'
    if theta[-1] >= .25:
        assert api.li_mak(jet, theta, 10)['status'] == 'fourth_moment_unavailable'
        assert api.absolute_li_mak(jet, theta, 10, api.Estimator.qmle)['status'] == 'fourth_moment_unavailable'


def test_rejected_in_sample_pit_reference_never_blocks_adequacy():
    from scivol import Search
    y = np.random.default_rng(402).uniform(-np.sqrt(3), np.sqrt(3), size=5000)
    result = Search(density=[Normal()], closure=False).fit(y)
    assert result.status == 'selected'
    checks = {c['name']: c for c in result.candidates[0].diagnostics['checks']}
    assert not {'berkowitz', 'dgt'} & checks.keys()
    assert all(not c['holm_reject'] for c in checks.values())
    assert result.selected.diagnostic_tests(print_results=False)['dgt']['p_value'] < .0001


def test_adequacy_preserves_infeasible_parameter_reason():
    model = model_for(mean_spec(constant=False), 'garch', 'normal')
    y = np.random.default_rng(402).normal(size=1000)
    tests = api.selection_diagnostics(api.Jet(model, y), np.array([-.1, .1, .8]))
    assert tests['status'] == 'adequacy_unresolved'
    assert all(c['status'] == 'infeasible_parameters' for c in tests['checks'])


@pytest.mark.parametrize('estimator', [api.Estimator.mle, api.Estimator.qmle])
def test_absolute_joint_score_correction_against_independent_reference(estimator):
    from scipy.integrate import quad
    from scipy.special import gammaln

    spec = ARMA(1, 1)+GARCH()+SkewT()
    y = spec.simulate(2500, [.05, .2, .1, .1, .1, .7, .125, -.3], seed=820).data
    fit = spec.fit(y, method='qmle' if estimator == api.Estimator.qmle else 'mle')
    model, sample, x = fit._native()
    jet = api.Jet(model, sample, x)
    theta = np.ascontiguousarray(fit.params)
    point = jet.theta(theta, 255)
    z, n, d = point['innovation'], len(point['innovation']), len(theta)
    eta, lam = theta[-2:]
    nu = 1/eta
    c = np.exp(gammaln((nu+1)/2)-gammaln(nu/2))/np.sqrt(np.pi*(nu-2))
    a = 4*lam*c*(nu-2)/(nu-1)
    b = np.sqrt(1+3*lam*lam-a*a)
    def density(value):
        scale = 1+lam*np.sign(b*value+a)
        return b*c*(1+((b*value+a)/scale)**2/(nu-2))**(-(nu+1)/2)
    cuts = sorted([-np.inf, -a/b, 0., np.inf])
    mu = sum(quad(lambda z: abs(z)*density(z), lo, hi, epsabs=2e-12)[0]
             for lo, hi in zip(cuts[:-1], cuts[1:]))
    below = sum(quad(density, lo, hi, epsabs=2e-12)[0]
                for lo, hi in zip(cuts[:-1], cuts[1:]) if hi <= 0)
    mean_sign, variance = 1-2*below, 1-mu*mu
    if estimator == api.Estimator.qmle:
        gaussian = model_for(mean_spec(ar=1, ma=1), 'garch', 'normal')
        p = gaussian.dimension
        score_point = api.Jet(gaussian, sample, x).theta(theta[:p], api.SCORES | api.HESSIAN)
    else:
        p, score_point = d, point
    scores = score_point['scores']
    bread = score_point['hessian']/n
    expected_derivative = (mean_sign * point['residual_sensitivities']/np.sqrt(point['variance'])[:, None]
                           - .5*mu*point['log_variance_sensitivities'])
    u = abs(z)-np.mean(abs(z))
    for sign in [False, True]:
        columns = 3 if sign else 10
        instruments = np.zeros((n, columns))
        for j in range(columns):
            first = 1 if sign else j+1
            lag = z[:-first]
            r = [lag < 0, np.minimum(lag, 0), np.maximum(lag, 0)][j] if sign else abs(lag)
            instruments[first:, j] = r-r.mean()
        moments = u[:, None]*instruments
        means = moments.mean(axis=0)
        derivative = instruments.T @ expected_derivative/n
        projection = np.linalg.solve(bread.T, derivative[:, :p].T).T
        covariance = np.cov(moments-scores@projection.T, rowvar=False, ddof=0)
        expected = n*means@np.linalg.solve(covariance, means)
        actual = api.absolute_sign_bias(jet, theta, estimator) if sign else api.absolute_li_mak(jet, theta, columns, estimator)
        assert actual['status'] == 'ok'
        assert actual['statistic'] == pytest.approx(expected, rel=2e-9)
        if not sign:
            np.testing.assert_allclose(actual['derivative'], derivative, rtol=2e-9, atol=2e-12)
            np.testing.assert_allclose(actual['covariance'], covariance/variance**2, rtol=2e-9, atol=2e-12)
            assert np.linalg.norm(derivative[:, :3]) > 1e-3


@pytest.mark.parametrize('eta', [None, 1/8, 1/3, .4])
def test_absolute_diagnostics_fast_size_check(eta):
    spec = GARCH()+(Normal() if eta is None else StudentT())
    theta = [.1, .1, .8]+([] if eta is None else [eta])
    p_values = []
    for seed in range(80):
        y = spec.simulate(2500, theta, seed=20261027+seed, burn=1000).data
        fit = spec.fit(y, accuracy=1e-9)
        assert fit.success
        model, sample, x = fit._native()
        jet, point = api.Jet(model, sample, x), np.ascontiguousarray(fit.params)
        tests = [api.absolute_li_mak(jet, point, lag) for lag in [10, 20]]
        tests.append(api.absolute_sign_bias(jet, point))
        assert all(t['status'] == 'ok' for t in tests)
        p_values.append([t['p_value'] for t in tests])
    p_values = np.array(p_values)
    # Binomial intervals acknowledge the uncertainty of this small smoke panel.
    for level in [.05, .01]:
        low, high = stats.binom.interval(.999, len(p_values), level)
        assert np.all((p_values < level).sum(axis=0) >= low)
        assert np.all((p_values < level).sum(axis=0) <= high)


def test_absolute_diagnostics_power_against_leverage_and_missing_lags():
    for dgp, theta, fitted, check in [
        (GJRGARCH()+Normal(), [.1, .03, .3, .7], GARCH(), api.absolute_sign_bias),
        (GARCH(2, 1)+Normal(), [.1, .1, .3, .45], GARCH(1, 0), api.absolute_li_mak),
    ]:
        y = dgp.simulate(5000, theta, seed=20261027, burn=1000).data
        fit = fitted.fit(y)
        assert fit.success
        model, sample, x = fit._native()
        test = check(api.Jet(model, sample, x), np.ascontiguousarray(fit.params))
        assert test['status'] == 'ok' and test['p_value'] < .001


def test_aparch_open_endpoint_is_not_singular_power_information():
    from scivol import APARCH
    spec = APARCH()+SkewT()
    y = spec.simulate(1200, [.1, .1, .4, .7, .85, .15, -.2], seed=91).data
    model = api.Model(native_spec(spec))
    theta = np.array([.1, .1, np.nextafter(1., 0.), .7, .85, .15, -.2])
    assert model.feasible(theta) and model.at_open_bound(theta)
    jet = api.Jet(model, y)
    for test in [api.li_mak(jet, theta, 10), api.absolute_li_mak(jet, theta, 10),
                 api.absolute_sign_bias(jet, theta)]:
        assert test['status'] == 'at_boundary'
    # A regular interior leverage point has no artificial face removal.
    theta[2] = .4
    assert not model.at_open_bound(theta)
    assert api.absolute_li_mak(jet, theta, 10, estimated=False)['status'] == 'ok'


@pytest.mark.parametrize('density,shape', [
    ('generalized_hyperbolic', [.2, .05, -.7]),
    ('nig', [-.6, .05]),
    ('hyperbolic', [.7, .1]),
    ('variance_gamma', [-.3, 2.5]),
])
def test_absolute_moments_exist_for_heavy_gh_interiors(density, shape):
    spec = api.ModelSpec()
    spec.density = getattr(api.DensityKind, density)
    model = api.Model(spec)
    theta = np.array([.1, .1, .7]+shape)
    mu = model.innovation_power_moment(theta, 0., 1.)
    assert 0 < mu < 1 and model.feasible(theta)
    y = np.random.default_rng(72).standard_normal(500)
    jet = api.Jet(model, y)
    assert api.absolute_li_mak(jet, theta, 20, estimated=False)['status'] == 'ok'
    assert api.absolute_sign_bias(jet, theta, estimated=False)['status'] == 'ok'


def test_known_holm_rejection_dominates_an_unavailable_check():
    from scivol import APARCH
    y = np.random.default_rng(38).normal(size=1000)
    for t in range(1, len(y)):
        y[t] += .8*y[t-1]
    model = api.Model(native_spec(APARCH()+Normal()))
    theta = np.array([.1, .1, np.nextafter(1., 0.), .7, .85])
    tests = api.selection_diagnostics(api.Jet(model, y), theta)
    assert tests['status'] == 'rejected'
    assert any(t['holm_reject'] and t['name'].startswith('residual_ljung_box[')
               for t in tests['checks'])
    assert any(t['status'] == 'at_boundary' for t in tests['checks'])
