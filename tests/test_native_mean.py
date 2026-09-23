"""Mathematical and replacement gates for standalone native mean models."""

import numpy as np
import pytest

from scivol import ARMA, ARX, HARX
from scivol._mean_fit import DENSITIES, MeanModel, fit_mean_spec, assess_solution
from scivol._devtools.mean_oracle import mean_value_grad_hess

SHAPES = [
    [],
    [7.0],
    [7.0, 0.2],
    [1.7],
    [0.12, 1.8, 0.4],
    [0.12, 1.8],
    [0.12, 1.8],
    [0.12, 2.0],
]


@pytest.mark.parametrize("fixed", [False, True])
def test_gaussian_value_retry_preserves_a_small_nonzero_residual(fixed):
    small = 2.0**-27
    data = np.ones(5)
    features = np.zeros((5, 4))
    features[:, 0] = 1.0 - small
    model = MeanModel(ARX(0, constant=False) + DENSITIES[0](), data, features, fixed=fixed)
    theta = np.array([1.0 + small, 0.0, 0.0, 0.0, 1e-32])
    expected = 0.5 * len(data) * (
        np.log(2 * np.pi) + np.log(theta[-1]) + (2.0**-54) ** 2 / theta[-1]
    )
    for request in range(1, 8, 2):
        actual = model.evaluate(theta, request)[0]
        np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-12)


def sample(mean, density, n=20, *, fixed=True):
    rng = np.random.default_rng(983)
    y = rng.normal(size=n)
    x = None if isinstance(mean, ARMA) else rng.normal(size=(n, 2))
    model = MeanModel(mean + DENSITIES[density](), y, x, fixed=fixed)
    theta = np.r_[np.full(model.width - 1, 0.03), 1.3, SHAPES[density]]
    return model, theta, x


@pytest.mark.parametrize("mean", [ARX(1), HARX([1,3])])
def test_joint_mean_preparation_does_not_allocate_a_standalone_jet(mean, monkeypatch):
    from scivol import APARCH, _native_core
    from scivol._mean_fit import PreparedMean

    standalone, _, x = sample(mean, 0)
    def unexpected(*_):
        raise AssertionError("Preparing a joint design constructed a standalone jet")
    monkeypatch.setattr(_native_core, "_mean_sizes", unexpected)
    prepared = PreparedMean(mean + APARCH(3,2), standalone.data, x)
    np.testing.assert_array_equal(prepared.features, standalone.features)
    np.testing.assert_array_equal(prepared.mean_units, standalone.units[:standalone.width-1])
    assert not hasattr(prepared, "kernel_workspace")
    assert not prepared.data.flags.writeable
    assert not prepared.features.flags.writeable


@pytest.mark.parametrize("density", range(8))
@pytest.mark.parametrize("mean", [ARMA(3, 2), ARX(2), HARX([1, 3])])
@pytest.mark.parametrize("coordinates", [False, True])
def test_complete_jet_matches_ad(mean, density, coordinates):
    model, theta, x = sample(mean, density)
    point = model.map(theta, forward=False) if coordinates else theta
    features = model.features
    if features is None:
        features = np.ones((len(model.data), model.columns))
        for lag in range(1, model.p + 1):
            features[lag:, lag] = model.data[:-lag]
    expected = mean_value_grad_hess(
        point,
        model.data,
        features,
        model.q,
        model.first,
        density,
        coordinates=coordinates,
        ar_order=model.p,
    )
    actual = model.evaluate(point, coordinates=coordinates)
    for a, b in zip(actual, expected):
        np.testing.assert_allclose(a, b, rtol=2e-6, atol=2e-7)
    if density != 0 and not coordinates:
        from scivol._evaluation import filter_spec

        state = filter_spec(
            model.spec, model.data, theta, x=x, compute_score=True, compute_hessian=True
        )
        for observed, reference in zip(
            (state.log_likelihood, state.score, state.hessian), expected
        ):
            np.testing.assert_allclose(observed, -reference, rtol=2e-6, atol=2e-7)


@pytest.mark.parametrize("density", range(8))
@pytest.mark.parametrize(
    "mean", [ARMA(1, 1), ARMA(0, 0), ARMA(0, 4), ARMA(4, 0), ARX(1)]
)
@pytest.mark.parametrize("coordinates", [False, True])
def test_sparse_masks_and_generated_dispatch(mean, density, coordinates):
    model, theta, x = sample(mean, density)
    runtime = MeanModel(model.spec, model.data, x, fixed=False)
    point = model.map(theta, forward=False) if coordinates else theta
    full = model.evaluate(point, coordinates=coordinates)
    for mask in range(1, 8):
        actual = model.evaluate(point, mask, coordinates=coordinates)
        generic = runtime.evaluate(point, mask, coordinates=coordinates)
        for index, bit in enumerate((1, 2, 4)):
            if mask & bit:
                np.testing.assert_allclose(
                    actual[index], full[index], rtol=1e-11, atol=1e-10
                )
                np.testing.assert_allclose(
                    actual[index], generic[index], rtol=1e-11, atol=1e-10
                )
            else:
                assert actual[index] is None


@pytest.mark.parametrize("density", range(8))
@pytest.mark.parametrize("mean", [ARMA(1, 1), ARX(1), HARX([1, 5, 22])])
def test_native_solution_strict_credibility(mean, density):
    model, theta, x = sample(mean, density, n=400)
    native = fit_mean_spec(model.spec, model.data, x=x)
    assert native.converged, native.convergence_message
    assert native.fit_info.kkt_residual <= 1e-6
    if native.std_errors is None:
        assert any("standard errors" in warning for warning in native.warnings)
    else:
        assert np.isfinite(native.std_errors).all()
    fixed_theta = (
        np.r_[native.params, native._sigma2[0]] if density == 0 else native.params
    )
    f, g, _ = model.evaluate(fixed_theta)
    assert assess_solution(model, fixed_theta, g)[1] <= 1e-6
    np.testing.assert_allclose(f, -native.log_likelihood, atol=1e-10)


@pytest.mark.parametrize(
    "mean",
    [ARX(0, constant=False), ARX(2, constant=False), HARX([1, 3], constant=False)],
)
def test_no_intercept_and_empty_exog(mean):
    model = MeanModel(
        mean + DENSITIES[1](), np.random.default_rng(421).normal(size=120)
    )
    point = model.initialize(None)
    np.testing.assert_allclose(
        model.map(model.map(point, forward=False), forward=True), point, atol=1e-13
    )
    fit = fit_mean_spec(model.spec, model.data)
    assert fit.converged, fit.convergence_message


def test_invalid_inputs_and_domains():
    model, theta, _ = sample(ARMA(2, 3), 1)
    for bad in (0.0, -1.0, np.nan):
        invalid = theta.copy()
        invalid[model.width - 1] = bad
        with pytest.raises(ValueError):
            model.evaluate(invalid)
    theta[-1] = 2.0
    with pytest.raises(ValueError):
        model.evaluate(theta)
    with pytest.raises(ValueError):
        MeanModel(ARMA(5, 1).spec, np.arange(4.0))


@pytest.mark.parametrize("coordinates", [False, True])
def test_long_near_invertibility_boundary(coordinates):
    model, theta, _ = sample(ARMA(0, 1), 0, n=1200)
    theta[:] = [0.01, -0.998, 1.3]
    point = model.map(theta, forward=False) if coordinates else theta
    expected = mean_value_grad_hess(
        point,
        model.data,
        np.ones((len(model.data), 1)),
        1,
        1,
        0,
        coordinates=coordinates,
    )
    actual = model.evaluate(point, coordinates=coordinates)
    for a, b in zip(actual, expected):
        np.testing.assert_allclose(a, b, rtol=2e-9, atol=2e-6)
    assert np.max(np.abs(model.evaluate(theta)[2])) > 1e6


@pytest.mark.parametrize("density", [0, 1, 2])
def test_public_fit_filter_inference(density):
    from scivol._evaluation import filter_spec

    mean = ARMA(2, 1)
    data = np.random.default_rng(823).standard_t(6, size=500)
    result = (mean + DENSITIES[density]()).fit(data)
    state = filter_spec(
        result.spec,
        result.data,
        result.params,
        compute_score=True,
        compute_hessian=True,
    )
    np.testing.assert_allclose(state.log_likelihood, result.log_likelihood, atol=1e-10)
    np.testing.assert_allclose(state.sigma2, result._sigma2, rtol=1e-12)
    np.testing.assert_allclose(state.residuals, result._resid, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(-state.hessian, result._hessian, rtol=1e-11, atol=1e-10)
    assert result.converged, result.convergence_message
    assert len(result.params) == mean.n_params + (
        0 if density == 0 else 1 + DENSITIES[density]().n_params
    )


@pytest.mark.parametrize("nu", [7.0, 128.0, 1e5, 1e10])
def test_symmetric_skew_student_reduction(nu):
    student, theta, _ = sample(ARMA(2, 1), 1)
    skew, _, _ = sample(ARMA(2, 1), 2)
    theta[-1] = nu
    a = student.evaluate(theta)
    b = skew.evaluate(np.r_[theta, 0.0])
    np.testing.assert_allclose(a[0], b[0], rtol=2e-13, atol=2e-12)
    np.testing.assert_allclose(a[1], b[1][:-1], rtol=1e-8, atol=1e-10)
    np.testing.assert_allclose(a[2], b[2][:-1, :-1], rtol=1e-8, atol=1e-10)


def test_normal_profile_counts_the_estimated_variance():
    result = ARMA(1, 0).fit(np.random.default_rng(71).normal(size=200))
    assert result.fit_info.profiled_parameters == 1
    assert result.aic == 2 * (len(result.params) + 1) - 2 * result.log_likelihood
    assert result.fit_info.likelihood_n_obs == 199


@pytest.mark.parametrize("density", range(8))
def test_coordinate_bounds_are_representable(density):
    from scivol import _native_core

    model, theta, _ = sample(ARMA(1, 1), density)
    lower, upper = np.empty(model.size), np.empty(model.size)
    assert _native_core._mean_bounds(
        model.descriptor, lower.ctypes.data, upper.ctypes.data, True
    )
    z = model.map(theta, forward=False)
    for index in range(model.width, model.size):
        for endpoint in (lower[index], upper[index]):
            if not np.isfinite(endpoint):
                continue
            point = z.copy()
            point[index] = endpoint
            mapped = model.map(point, forward=True)
            assert mapped[index] >= model.lower[index]
            assert mapped[index] <= model.upper[index]
            assert np.isfinite(mapped).all()


@pytest.mark.parametrize("mean,kind", [(ARMA(1, 1), 2), (ARX(1), 1), (ARMA(3, 2), 0)])
def test_solver_and_jet_share_dispatch_identity(mean, kind):
    from scivol import _native_core

    for density in range(8):
        model, _, x = sample(mean, density)
        assert _native_core._mean_dispatch_kind(model.descriptor) == kind
        runtime = MeanModel(model.spec, model.data, x, fixed=False)
        assert _native_core._mean_dispatch_kind(runtime.descriptor) == 0


@pytest.mark.parametrize("transformed", [False, True])
@pytest.mark.parametrize("density", range(8))
def test_independently_callable_native_formulations(transformed, density):
    model, _, x = sample(ARMA(2, 1), density, n=400)
    fit = fit_mean_spec(model.spec, model.data, x=x, log_mode=transformed)
    assert fit.converged, fit.convergence_message
    assert fit.fit_info.log_mode is transformed
    assert fit.fit_info.kkt_residual <= 1e-6


def test_iteration_exhaustion_is_finite_and_not_claimed_as_success():
    model, _, _ = sample(ARMA(2, 1), 2, n=400)
    fit = fit_mean_spec(model.spec, model.data, maxiter=1)
    assert not fit.converged
    assert np.isfinite(fit.params).all()
    assert np.isfinite(fit.log_likelihood)
    assert fit.warnings


def test_prepared_sample_owns_immutable_observations():
    data = np.random.default_rng(883).normal(size=50)
    model = MeanModel(ARMA(1, 1).spec, data)
    point = model.initialize(None)
    value = model.evaluate(point, 1)[0]
    data[:] = np.nan
    assert model.evaluate(point, 1)[0] == value
    assert not model.data.flags.writeable


def test_inactive_mean_sensitivity_does_not_invent_a_ged_singularity():
    from scivol import GED

    model = MeanModel(ARX(0, constant=False) + GED(), np.array([0.0, 1.0, -1.0]))
    point = np.array([1.0, 1.5])
    expected = mean_value_grad_hess(point, model.data, np.empty((3, 0)), 0, 0, 3)
    for actual, reference in zip(model.evaluate(point), expected):
        np.testing.assert_allclose(actual, reference, rtol=1e-8, atol=1e-9)


def test_active_mean_sensitivity_reports_ged_center_nondifferentiability():
    from scivol import GED

    model = MeanModel(ARMA(0, 0) + GED(), np.array([1.0, 0.0, 1.0, -1.0]))
    point = np.array([0.0, 1.0, 1.5])
    model.evaluate(point, 3)
    with pytest.raises(ValueError, match="differentiability"):
        model.evaluate(point, 4)


def test_normal_value_remains_representable_below_reciprocal_range():
    model = MeanModel(ARMA(0, 0).spec, np.array([0.0, 1e-161, -1e-161]))
    variance = 1e-320
    value = model.evaluate(np.array([0.0, variance]), 1)[0]
    expected = 0.5 * np.sum(
        np.log(2 * np.pi) + np.log(variance) + (model.data[1:] / np.sqrt(variance)) ** 2
    )
    np.testing.assert_allclose(value, expected, atol=1e-12, rtol=0)
    with pytest.raises(ValueError):
        model.evaluate(np.array([0.0, variance]), 4)


def test_normal_value_remains_representable_when_raw_squares_overflow():
    model = MeanModel(ARMA(0, 0).spec, np.array([0.0, 1e200, -1e200]))
    variance = 1e300
    value = model.evaluate(np.array([0.0, variance]), 1)[0]
    expected = 0.5 * np.sum(
        np.log(2 * np.pi) + np.log(variance) + (model.data[1:] / np.sqrt(variance)) ** 2
    )
    np.testing.assert_allclose(value, expected, rtol=2e-15)


def test_residual_output_may_not_overwrite_jet_parameters():
    model, theta, _ = sample(ARMA(1, 1), 1, n=5)
    with pytest.raises(ValueError, match="alias"):
        model.evaluate(theta, 1, residuals=theta)


def test_normal_value_rejects_nonfinite_residual_recursion():
    model = MeanModel(ARMA(1, 0).spec, np.array([1e200, 1e200, 1e200]))
    with pytest.raises(ValueError):
        model.evaluate(np.array([0.0, 1e200, 1.0]), 1)


def test_normal_fixed_mean_explicitly_profiles_innovation_variance():
    model, theta, _ = sample(ARMA(2, 1), 0, n=100)
    result = model.spec.fix(model.data, theta[:-1])
    expected = np.mean(result._resid[model.first :] ** 2)
    np.testing.assert_allclose(result._sigma2, expected)
    assert result.fit_info.profiled_parameters == 1
    assert any("variance is profiled" in warning for warning in result.warnings)


def test_unrepresentable_inference_is_reported_without_nonfinite_matrices():
    import warnings

    data = np.random.default_rng(8471).standard_t(6, 500) * 1e-150
    with warnings.catch_warnings(record=True) as emitted:
        warnings.simplefilter("always")
        result = (ARMA(0, 0) + DENSITIES[1]()).fit(data)
    assert not emitted
    assert result.converged
    assert result.hessian is None and result.std_errors is None
    assert any(
        "rescale the data for inference" in warning for warning in result.warnings
    )


@pytest.mark.parametrize("density", range(8))
@pytest.mark.parametrize("mean", [ARMA(2, 1), ARX(2), HARX([1, 3])])
def test_established_evaluator_at_shared_points(mean, density):
    from scivol._evaluation import (
        _filter_arma,
        _filter_arma_ged,
        _filter_meanx_normal,
        _filter_meanx_standalone,
        _filter_gh,
    )

    if isinstance(mean, ARMA) and density in (1, 2):
        pytest.skip("No established standalone ARMA Student-t or Skew-t evaluator")
    model, theta, x = sample(mean, density, n=100)
    options = dict(compute_score=True, compute_hessian=True)
    if density == 0:
        residuals = np.empty_like(model.data)
        model.evaluate(theta, 1, residuals=residuals)
        theta[-1] = np.mean(residuals[model.first :] ** 2)
        legacy = (
            _filter_arma(model.spec, model.data, theta[:-1], **options)
            if isinstance(mean, ARMA)
            else _filter_meanx_normal(
                model.spec, model.data, theta[:-1], x=x, **options
            )
        )
    elif density >= 4:
        legacy = _filter_gh(model.spec, model.data, theta, x=x, **options)
    elif isinstance(mean, ARMA):
        legacy = _filter_arma_ged(model.spec, model.data, theta, **options)
    else:
        legacy = _filter_meanx_standalone(model.spec, model.data, theta, x=x, **options)
    f, g, h = model.evaluate(theta)
    old_f, old_g, old_h = -legacy.log_likelihood, -legacy.score, -legacy.hessian
    if density == 0:
        g = g[:-1]
        h = h[:-1, :-1] - np.outer(h[:-1, -1], h[-1, :-1]) / h[-1, -1]
    np.testing.assert_allclose(f, old_f, rtol=2e-10, atol=1e-9)
    np.testing.assert_allclose(g, old_g, rtol=2e-7, atol=2e-7)
    np.testing.assert_allclose(h, old_h, rtol=2e-6, atol=2e-6)


@pytest.mark.parametrize("density", [1, 2])
def test_new_arma_entries_simulate_fix_and_forecast(density):
    spec = ARMA(2, 1) + DENSITIES[density]()
    point = np.r_[0.2, 0.35, -0.12, 0.4, 1.2, SHAPES[density]]
    simulation = spec.simulate(300, point, seed=329)
    fixed = spec.fix(simulation.data, point)
    prediction = fixed.forecast(horizon=4)
    assert len(fixed._get_param_names()) == len(point)
    assert np.isfinite(fixed.log_likelihood)
    np.testing.assert_allclose(prediction.residual_variance, 1.2)


@pytest.mark.parametrize("factor", [1e-8, 1e8])
@pytest.mark.parametrize("density", [0, 1, 2, 4, 7])
def test_exact_measurement_unit_back_transforms(factor, density):
    data = np.random.default_rng(983).standard_t(6, size=400) * np.sqrt(2 / 3)
    model = MeanModel(ARMA(2, 1) + DENSITIES[density](), data)
    x = None
    a = fit_mean_spec(model.spec, model.data, x=x)
    b = fit_mean_spec(model.spec, model.data * factor, x=x)
    units = np.ones(len(a.params))
    units[0] = factor
    if density:
        units[model.width - 1] = factor**2
    np.testing.assert_allclose(b.params / units, a.params, rtol=2e-4, atol=2e-5)
    np.testing.assert_allclose(b._sigma2 / factor**2, a._sigma2, rtol=2e-5)
    np.testing.assert_allclose(
        b.log_likelihood + model.n_eff * np.log(factor), a.log_likelihood, atol=1e-7
    )
    assert a.converged and b.converged


def test_units_do_not_change_the_likelihood_of_a_weakly_identified_shape():
    model, _, _ = sample(ARMA(2, 1), 2, n=400)
    original = fit_mean_spec(model.spec, model.data)
    scaled = fit_mean_spec(model.spec, model.data * 1e-8)
    assert original.params[-2] > 1e5 and scaled.params[-2] > 1e5
    np.testing.assert_allclose(
        scaled.log_likelihood + model.n_eff * np.log(1e-8),
        original.log_likelihood,
        atol=1e-6,
        rtol=0,
    )
    assert original.converged and scaled.converged


def test_native_mean_workspaces_are_thread_isolated():
    from concurrent.futures import ThreadPoolExecutor

    def fit(density):
        model, _, x = sample(ARX(1), density, n=400)
        result = fit_mean_spec(model.spec, model.data, x=x)
        return result.params, result.log_likelihood, result.converged

    sequential = [fit(density) for density in range(8)]
    with ThreadPoolExecutor(max_workers=4) as pool:
        parallel = list(pool.map(fit, range(8)))
    for a, b in zip(sequential, parallel):
        np.testing.assert_array_equal(a[0], b[0])
        assert a[1:] == b[1:]


def test_standalone_defaults_match_the_machine_checked_manifest():
    from scivol._devtools.surface_manifest import SURFACES

    data = np.random.default_rng(175).standard_t(6, size=400)
    for surface in SURFACES:
        if surface.volatility is not None or surface.mean is None:
            continue
        x = None if surface.mean == "ARMA" else np.zeros((len(data), 1))
        fitted = surface.build().fit(data, x=x)
        assert fitted.fit_info.engine == "native-mean-jet"
        assert fitted.fit_info.log_mode == surface.default_log_mode
