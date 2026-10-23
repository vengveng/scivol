"""Focused generic active-face math, sparse callbacks, retention and arena checks."""

from pathlib import Path
import json
import re
import shutil
import subprocess

import numpy as np
import pytest
from scipy.optimize import LinearConstraint, minimize


ROOT = Path(__file__).resolve().parents[1]
NATIVE = ROOT / "scivol" / "_native"
SOURCE = Path(__file__).with_name("native_active_face_refinement.cpp")


@pytest.fixture(scope="module")
def native(tmp_path_factory):
    compiler = shutil.which("clang++") or shutil.which("g++")
    if compiler is None:
        pytest.skip("A C++20 compiler is required for active-face contracts")
    executable = tmp_path_factory.mktemp("native_active_face") / "contracts"
    frame_flags = (
        ["-Wframe-larger-than=16384", "-Werror=frame-larger-than"]
        if "clang" in Path(compiler).name
        else ["-Werror=frame-larger-than=16384"]
    )
    command = [
        compiler, "-std=c++20", "-O3", "-ffp-contract=fast", "-funroll-loops",
        *frame_flags, "-I", str(NATIVE), "-I", str(ROOT / "scivol" / "_csrc"),
        str(SOURCE),
        *(str(NATIVE / name) for name in (
            "slsqp_translation.cpp", "portable_linalg.cpp", "portable_nnls.cpp"
        )),
        "-o", str(executable),
    ]
    completed = subprocess.run(command, capture_output=True, text=True, timeout=60)
    assert completed.returncode == 0, completed.stdout + completed.stderr
    return executable


def invoke(native, command, *values):
    tokens = []
    for value in values:
        tokens.extend(str(item) for item in np.asarray(value).ravel())
    completed = subprocess.run(
        [str(native), command], input=" ".join(tokens), capture_output=True,
        text=True, timeout=10,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    return json.loads(completed.stdout)


def refine(
    native, *, point=(0., 1., 3.), center=(-1., 2., 0.), hessian=None,
    linear=None, lower=None, upper=None, units=None, normals=None, offsets=None,
    geometry=0, failure=0, backtracks=16, corrections=4, scale=.25,
    target=1e-6, short_by=0,
):
    point, center = np.asarray(point, dtype=float), np.asarray(center, dtype=float)
    n = len(point)
    hessian = np.eye(n) if hessian is None else np.asarray(hessian, dtype=float)
    linear = np.zeros(n) if linear is None else np.asarray(linear, dtype=float)
    units = np.array([1., 2., .5]) if units is None and n == 3 else (
        np.ones(n) if units is None else np.asarray(units, dtype=float)
    )
    lower = np.r_[0., np.full(n-1, -np.inf)] if lower is None else np.asarray(lower, dtype=float)
    upper = np.full(n, np.inf) if upper is None else np.asarray(upper, dtype=float)
    normals = np.empty((0, n)) if normals is None else np.asarray(normals, dtype=float)
    m = len(normals)
    offsets = np.zeros(m) if offsets is None else np.asarray(offsets, dtype=float)
    assert hessian.shape == (n, n) and center.shape == linear.shape == units.shape == (n,)
    assert normals.shape == (m, n) and offsets.shape == (m,)
    result = invoke(
        native, "phase", n, m, geometry, failure, backtracks, corrections, short_by,
        scale, target, hessian, center, linear, units, offsets, normals, point, lower, upper,
    )
    for name in ("point", "best", "evidence"):
        result[name] = np.asarray(result[name], dtype=float)
    result["requests"] = np.asarray(result["requests"], dtype=int)
    result["constraints"] = np.asarray(result["constraints"], dtype=int)
    result["minimum_scored_slack"] = float(result["minimum_scored_slack"])
    assert result["allocations"] == 0
    assert not result["requests"][[0, 3, 5, 6, 7]].any()
    return result


def quadratic_value(point, center, hessian=None, linear=None):
    point, center = np.asarray(point), np.asarray(center)
    hessian = np.eye(len(point)) if hessian is None else np.asarray(hessian)
    linear = np.zeros(len(point)) if linear is None else np.asarray(linear)
    residual = point-center
    return 3.25 + .5*residual @ hessian @ residual + linear @ point


def independent_projection(gradient, normals, gaps):
    gradient, normals, gaps = map(np.asarray, (gradient, normals, gaps))
    if not len(gaps):
        return -gradient
    solved = minimize(
        lambda step: .5*step @ step + gradient @ step,
        np.zeros(len(gradient)), jac=lambda step: step+gradient,
        constraints=LinearConstraint(normals, -gaps, np.inf), method="SLSQP",
        options={"ftol": 1e-13, "maxiter": 100},
    )
    assert solved.success, solved.message
    return solved.x


@pytest.mark.parametrize("width", [10, 32])
def test_subnormal_certificate_bounds_exact_coupled_projection(native, width):
    from fractions import Fraction

    tiny = np.nextafter(0., 1.)
    gradient = np.full(width, tiny)
    normal = np.r_[-2., np.ones(width-1)]
    result = invoke(native, "certificate", width, 1, gradient, normal,
                    [0.], np.zeros(width), [0.])
    bound = float(result["bound"])
    # Projection of -g onto normal @ step >= 0 has its largest entry at index 0.
    exact_norm = Fraction.from_float(tiny) * Fraction(3*width-3, width+3)
    assert np.isfinite(bound)
    assert Fraction.from_float(bound) >= exact_norm


@pytest.mark.parametrize("n,active", [(1, 0), (3, 0), (5, 2), (7, 3), (5, 5)])
@pytest.mark.parametrize("curvature", ["positive", "indefinite", "singular"])
def test_direction_matches_independent_reduced_curvature(native, n, active, curvature):
    rng = np.random.default_rng(731+n+active)
    normals = rng.normal(size=(active, n))
    tangent = np.linalg.qr(normals.T, mode="complete")[0][:, active:]
    rotation = np.linalg.qr(rng.normal(size=(n, n)))[0]
    spectrum = np.linspace(.5, 3., n)
    if curvature == "indefinite":
        spectrum[0] = -1.
    elif curvature == "singular":
        spectrum[:max(1, n//2)] = 0.
    hessian = (rotation*spectrum) @ rotation.T
    gradient = rng.normal(size=n)
    eigenvalues, vectors = np.linalg.eigh(tangent.T @ hessian @ tangent)
    threshold = n*np.finfo(float).eps*np.max(np.abs(eigenvalues), initial=0.)
    selected = eigenvalues > threshold
    basis = tangent @ vectors[:, selected]
    expected = -basis @ ((basis.T @ gradient)/eigenvalues[selected])
    result = invoke(native, "direction", n, active, 0, hessian, gradient, normals)
    assert result["valid"] and result["resolved"] == selected.sum()
    actual = np.asarray(result["output"])
    np.testing.assert_allclose(actual, expected, rtol=3e-11, atol=3e-12)
    np.testing.assert_allclose(result["eigenvalues"][:n-active], eigenvalues, rtol=2e-12, atol=3e-14)
    np.testing.assert_allclose(normals @ actual, 0., atol=3e-12)
    assert gradient @ actual <= 1e-13


@pytest.mark.parametrize("rows", [0, 1, 3, 5])
@pytest.mark.parametrize("row_scale", [1e-150, 1., 1e150])
def test_normal_correction_has_independent_minimum_norm_solution(native, rows, row_scale):
    rng = np.random.default_rng(94+rows)
    normals = rng.normal(size=(rows, 5))
    discrepancy = rng.normal(size=rows)
    result = invoke(native, "correction", 5, rows, 0, discrepancy*row_scale, normals*row_scale)
    assert result["valid"]
    expected = np.linalg.lstsq(normals, discrepancy, rcond=None)[0]
    np.testing.assert_allclose(result["output"], expected, rtol=3e-12, atol=3e-12)
    np.testing.assert_allclose(normals @ result["output"], discrepancy, rtol=3e-12, atol=3e-12)


@pytest.mark.parametrize("operation", ["direction", "correction"])
@pytest.mark.parametrize("failure", ["short", "dependent", "nan", "too_many_rows"])
def test_face_algebra_rejects_invalid_work(native, operation, failure):
    normals = np.array([[1., 0., 0.], [0., 1., 0.]])
    short_by = int(failure == "short")
    if failure == "dependent":
        normals[1] = 2*normals[0]
    elif failure == "nan":
        normals[0, 0] = np.nan
    elif failure == "too_many_rows":
        normals = np.ones((4, 3))
    gradient = np.ones(len(normals) if operation == "correction" else 3)
    args = [] if operation == "correction" else [np.eye(3)]
    result = invoke(native, operation, 3, len(normals), short_by, *args, gradient, normals)
    assert not result["valid"]
    np.testing.assert_array_equal(result["output"], 717.)


@pytest.mark.parametrize("curvature", ["nan", "asymmetric"])
def test_invalid_hessian_does_not_propose_a_direction(native, curvature):
    hessian = np.eye(3)
    hessian[0, 1] = np.nan if curvature == "nan" else .5
    result = invoke(native, "direction", 3, 0, 0, hessian, np.ones(3))
    assert not result["valid"]
    np.testing.assert_array_equal(result["output"], 717.)


@pytest.mark.parametrize("indefinite", [False, True])
def test_conditioning_preserves_model_units_and_quadratic(native, indefinite):
    coordinate_units = 2.**np.array([-12., 9., 20.])
    hessian = np.array([[2., .25, -.5], [.25, -3. if indefinite else 3., .5], [-.5, .5, 1.]])
    hessian = hessian / coordinate_units[:, None] / coordinate_units
    gradient = np.array([.5, -1., 2.])/coordinate_units
    normals = np.array([[1., 2., -3.], [0., 1., 1.]])/coordinate_units
    units = np.array([.5, 4., 2.])
    result = invoke(native, "conditioning", 3, 2, hessian, gradient, normals, units)
    assert result["valid"]
    ratio = np.asarray(result["units"])/units
    h = np.asarray(result["hessian"]).reshape(3, 3)
    g = np.asarray(result["gradient"])
    a = np.asarray(result["normals"]).reshape(2, 3)
    np.testing.assert_array_equal(h, hessian*ratio[:, None]*ratio)
    np.testing.assert_array_equal(g, gradient*ratio)
    np.testing.assert_array_equal(a, normals*ratio)
    displacement = np.array([.25, -.125, .5])
    original = ratio*displacement
    np.testing.assert_allclose(g @ displacement+.5*displacement @ h @ displacement,
                               gradient @ original+.5*original @ hessian @ original, rtol=3e-15)
    np.testing.assert_array_equal(a @ displacement, normals @ original)


@pytest.mark.parametrize("failure", ["gradient", "hessian", "normal", "units", "nan"])
@pytest.mark.parametrize("overflow", [False, True])
def test_failed_conditioning_does_not_commit_partial_units(native, failure, overflow):
    hessian = np.diag([1., 2.**(-100 if overflow else 100)])
    gradient, normals, units = np.ones(2), np.ones((1, 2)), np.array([3., 5.])
    extreme = np.finfo(float).max if overflow else np.nextafter(0., 1.)
    if failure == "gradient":
        gradient[1] = extreme
    elif failure == "hessian":
        hessian[0, 1] = hessian[1, 0] = extreme
    elif failure == "normal":
        normals[0, 1] = extreme
    elif failure == "units":
        units[1] = extreme
    else:
        hessian[1, 1] = np.nan
    result = invoke(native, "conditioning", 2, 1, hessian, gradient, normals, units)
    assert not result["valid"]
    np.testing.assert_array_equal(result["units"], units)


@pytest.mark.parametrize("n,rows", [(1, 0), (2, 1), (4, 8), (7, 12)])
@pytest.mark.parametrize("active", [False, True])
def test_projection_has_independent_primal_dual_stationarity(native, n, rows, active):
    rng = np.random.default_rng(127+n+rows)
    gradient, normals = rng.normal(size=n), rng.normal(size=(rows, n))
    gaps = np.zeros(rows) if active else rng.uniform(.1, 2., rows)
    result = invoke(native, "projection", n, rows, 0, gradient, normals, gaps)
    assert result["mode"] == 0
    point, dual = np.asarray(result["point"]), np.asarray(result["dual"])
    expected = independent_projection(gradient, normals, gaps)
    np.testing.assert_allclose(point, expected, rtol=3e-10, atol=3e-10)
    slack = normals @ point+gaps
    assert np.min(slack, initial=0.) >= -2e-11 and np.all(dual >= 0.)
    np.testing.assert_allclose(point+gradient, normals.T @ dual, atol=3e-10)
    assert abs(slack @ dual) <= 3e-10
    bound = float(result["bound"])
    assert bound >= np.max(np.abs(expected), initial=0.)-3e-10
    assert bound <= np.linalg.norm(gradient)*(1+1e-12)


@pytest.mark.parametrize("scale", [1., 1e5, 1e12])
def test_projection_preserves_shifted_face_with_large_force(native, scale):
    gradient = scale*np.array([1., 2.])
    result = invoke(native, "projection", 2, 1, 0, gradient, [[1., 1.]], [.125])
    assert result["mode"] == 0
    expected = scale*np.array([.5, -.5])-.0625
    np.testing.assert_allclose(np.asarray(result["point"])/scale, expected/scale, rtol=0., atol=2e-14)
    np.testing.assert_allclose(np.asarray(result["dual"])/scale, [(3*scale-.125)/(2*scale)], rtol=0., atol=2e-14)
    assert float(result["bound"]) >= np.max(np.abs(expected))


@pytest.mark.parametrize("failure", ["short", "gap", "gradient", "normal", "lost_gap", "lost_gradient"])
def test_invalid_projection_cannot_manufacture_evidence(native, failure):
    gradient, normals, gaps = np.ones(2), np.ones((1, 2)), np.ones(1)
    if failure == "gap":
        gaps[0] = -1.
    elif failure == "gradient":
        gradient[0] = np.nan
    elif failure == "normal":
        normals[0, 0] = np.nan
    elif failure in {"lost_gap", "lost_gradient"}:
        gradient[0] = 4.
        if failure == "lost_gap":
            gaps[0] = np.nextafter(0., 1.)
        else:
            gradient[1] = np.nextafter(0., 1.)
    result = invoke(native, "projection", 2, 1, int(failure == "short"), gradient, normals, gaps)
    assert result["mode"] != 0
    np.testing.assert_array_equal(result["point"], 717.)
    np.testing.assert_array_equal(result["dual"], 617.)


@pytest.mark.parametrize("row_units", [1e-100, 1., 1e100])
def test_stationarity_preserves_tiny_nonzero_slack_and_constraint_units(native, row_units):
    gap = 2.**-40
    result = invoke(native, "stationarity", 2, 1, 0, [1., 0.], [-np.inf]*2, [np.inf]*2,
                    [[row_units], [0.]], [row_units*gap])
    assert result["projected"] and result["dimension"] == 2
    np.testing.assert_array_equal(result["gaps"], [gap])
    np.testing.assert_array_equal(result["identities"], [0])
    assert gap <= float(result["bound"]) < 1e-6


@pytest.mark.parametrize("rows", [0, 3, 7])
def test_stationarity_bound_covers_independent_full_domain_projection(native, rows):
    rng = np.random.default_rng(601+rows)
    gradient, jacobian = rng.normal(size=4), rng.normal(size=(rows, 4))
    lower, upper = -rng.uniform(.1, 2., 4), rng.uniform(.1, 2., 4)
    slack = rng.uniform(.1, 2., rows)
    result = invoke(native, "stationarity", 4, rows, 0, gradient, lower, upper, jacobian.T, slack)
    normals = np.vstack([jacobian, np.eye(4), -np.eye(4)])
    gaps = np.r_[slack, -lower, upper]
    expected = independent_projection(gradient, normals, gaps)
    assert np.max(np.abs(expected))-3e-10 <= float(result["bound"]) <= np.linalg.norm(gradient)*(1+1e-12)


@pytest.mark.parametrize("failure", ["short", "negative_slack", "infeasible_bound", "nan_jacobian", "nan_gradient"])
def test_stationarity_rejects_invalid_original_domain_evidence(native, failure):
    gradient, jacobian, slack, lower = np.ones(2), np.ones((1, 2)), np.ones(1), np.full(2, -np.inf)
    if failure == "negative_slack":
        slack[0] = -1.
    elif failure == "infeasible_bound":
        lower[0] = .1
    elif failure == "nan_jacobian":
        jacobian[0, 0] = np.nan
    elif failure == "nan_gradient":
        gradient[0] = np.nan
    result = invoke(native, "stationarity", 2, 1, int(failure == "short"), gradient, lower,
                    [np.inf]*2, jacobian.T, slack)
    assert np.isinf(float(result["bound"])) and result["dimension"] == 0


@pytest.mark.parametrize("bounded", [False, True])
def test_phase_reaches_known_quadratic_optimum_with_sparse_requests(native, bounded):
    result = refine(native, lower=None if bounded else [-np.inf]*3)
    expected = [0. if bounded else -1., 2., 0.]
    assert result["status"] == 7 and result["best_valid"] and result["best_accepted"]
    np.testing.assert_allclose(result["point"], expected, rtol=0., atol=2e-14)
    np.testing.assert_array_equal(result["best"], result["point"])
    assert result["evidence"][0] == quadratic_value(expected, [-1., 2., 0.])
    assert result["evidence"][1] <= 1e-6
    np.testing.assert_array_equal(result["requests"][[1, 2, 4]], [2, 2, 1])
    assert result["curvature"] == result["trials"] == 1


@pytest.mark.parametrize("upper_face", [False, True])
def test_shifted_bound_face_changes_the_coupled_tangent_gradient(native, upper_face):
    hessian = np.array([[1., .5, 0.], [.5, 1., 0.], [0., 0., 1.]])
    if upper_face:
        point, center, lower, upper = [.9, 1., 3.], [2., 2., 0.], [-np.inf]*3, [1., np.inf, np.inf]
        expected = [1., 2.5, 0.]
    else:
        point, center, lower, upper = [.1, 1., 3.], [-1., 2., 0.], [0., -np.inf, -np.inf], [np.inf]*3
        expected = [0., 1.5, 0.]
    result = refine(native, point=point, center=center, hessian=hessian, lower=lower, upper=upper)
    assert result["status"] == 7 and result["active"] == 1 and result["trials"] == 2
    np.testing.assert_allclose(result["point"], expected, rtol=0., atol=2e-14)
    assert result["evidence"][1] <= 1e-6


def test_full_bound_face_still_has_a_normal_step(native):
    result = refine(native, point=[.1, 3.1, 1.05], lower=[0., 3., 1.])
    assert result["status"] == 7 and result["active"] == 3 and result["resolved"] == 0
    np.testing.assert_allclose(result["point"], [0., 3., 1.], rtol=0., atol=2e-14)
    assert result["evidence"][1] <= 1e-6 and result["trials"] == 1


def test_nonleading_bound_elimination_does_not_leak_large_cross_curvature(native):
    hessian = np.array([[1., 0., 1e16], [0., 1., 0.], [1e16, 0., 1.]])
    result = refine(native, point=[0., .9, 0.], center=[0., 2., 0.], hessian=hessian,
                    linear=[-1., 0., 1.], lower=[-np.inf, -np.inf, 0.],
                    normals=[[0., -1., -3.1]], offsets=[1.], geometry=3)
    assert result["best_valid"] and result["curvature"] == 1
    np.testing.assert_allclose(result["best"], [1., .9, 0.], rtol=0., atol=2e-14)
    assert result["best"][2] == 0. and 1-result["best"][1]-3.1*result["best"][2] > 0.
    np.testing.assert_allclose(result["evidence"][2], 3.355, rtol=0., atol=3e-15)


def test_resolved_weak_curvature_is_not_erased_by_other_parameter_units(native):
    result = refine(native, center=[-1., 1e9, 0.], hessian=np.diag([1., 1e-18, 1.]))
    assert result["status"] == 7 and result["curvature"] == result["trials"] == 1
    np.testing.assert_allclose(result["point"], [0., 1e9, 0.], rtol=0., atol=2e-5)
    assert result["evidence"][0] == 3.75 and result["evidence"][1] <= 1e-6


def test_several_affine_rows_match_independent_constrained_optimum(native):
    normals = np.array([[1., 0., 0.], [0., -1., 0.], [1., 1., 0.]])
    offsets = np.array([0., 1., -.5])
    result = refine(native, point=[.1, .8, 3.], normals=normals, offsets=offsets, lower=[-np.inf]*3)
    expected = minimize(lambda x: quadratic_value(x, [-1., 2., 0.]), [.1, .8, 3.],
                        jac=lambda x: x-np.array([-1., 2., 0.]),
                        constraints=LinearConstraint(normals, -offsets, np.inf), method="SLSQP",
                        options={"ftol": 1e-13, "maxiter": 100})
    assert expected.success and result["best_valid"]
    np.testing.assert_allclose(result["best"], expected.x, atol=2e-12, rtol=0.)
    assert np.min(normals @ result["best"]+offsets) >= -2e-14
    assert result["evidence"][3] <= 1e-6


def test_tiny_strict_nonlinear_slack_is_preserved_without_padding(native):
    slack = 2.**-40
    result = refine(native, point=[slack, 0., 3.], center=[-1., 0., 0.], normals=[[0., 0., 0.]],
                    offsets=[0.], geometry=2, lower=[-np.inf]*3)
    assert result["status"] == 7 and result["best_valid"]
    np.testing.assert_array_equal(result["best"], [slack, 0., 0.])
    assert result["best"][0]-result["best"][1]**2 == slack
    assert result["minimum_scored_slack"] == slack
    assert result["evidence"][3] <= 1e-6


def test_curved_retraction_keeps_strict_feasibility_and_reduces_value(native):
    initial = np.array([0., .9, 0.])
    center = np.array([.1, 2., 0.])
    result = refine(native, point=initial, center=center, normals=[[0., 0., 0.]], offsets=[1.],
                    geometry=1, lower=[-np.inf]*3)
    assert result["status"] == 7 and result["best_valid"] and result["corrections"] > 0
    assert 1-result["best"][:2] @ result["best"][:2] > 0.
    assert result["minimum_scored_slack"] > 0.
    assert result["evidence"][2] < quadratic_value(initial, center)
    fixed_slack_optimum = 3.25+.5*(np.linalg.norm(center)-.9)**2
    assert result["evidence"][2] < fixed_slack_optimum


def test_indefinite_curvature_searches_full_domain_and_certifies_bound_optimum(native):
    hessian = np.diag([1., -1., 1.])
    result = refine(native, hessian=hessian, lower=[0., 0., -np.inf], upper=[np.inf, 2., np.inf])
    assert result["status"] == 7 and result["trials"] == 2 and result["curvature"] == 1
    np.testing.assert_allclose(result["point"], [0., 0., 0.], rtol=0., atol=2e-14)
    assert result["evidence"][1] <= 1e-6
    assert result["evidence"][0] == quadratic_value([0., 0., 0.], [-1., 2., 0.], hessian)


def test_singular_curvature_does_not_certify_an_unresolved_linear_direction(native):
    result = refine(native, hessian=np.diag([1., 0., 1.]), linear=[0., 1., 0.], lower=[-np.inf]*3, backtracks=1)
    assert result["best_valid"] and result["resolved"] == 2 and result["curvature"] == 1
    assert result["evidence"][2] < quadratic_value([0., 1., 3.], [-1., 2., 0.], np.diag([1., 0., 1.]), [0., 1., 0.])
    assert result["evidence"][3] > 1e-6


def test_best_feasible_uncertified_alternative_is_retained_unchanged(native):
    result = refine(native, hessian=np.diag([1., -1., 1.]), lower=[-np.inf]*3, backtracks=1)
    assert result["status"] == 7 and result["best_valid"] and not result["best_accepted"]
    np.testing.assert_allclose(result["point"], [-1., 1., 0.], rtol=0., atol=2e-14)
    np.testing.assert_allclose(result["best"], [-1., 0., 0.], rtol=0., atol=2e-14)
    assert result["evidence"][2] < result["evidence"][0]
    assert result["evidence"][1] > 1e-6 and result["evidence"][3] > 1e-6


def test_better_feasible_f_without_g_is_retained_but_never_accepted(native):
    result = refine(native, failure=1, backtracks=2)
    assert result["status"] == 6 and result["best_valid"] and not result["best_accepted"]
    np.testing.assert_array_equal(result["point"], [0., 1., 3.])
    np.testing.assert_allclose(result["best"], [0., 2., 0.], rtol=0., atol=2e-14)
    assert result["evidence"][2] < result["evidence"][0] and np.isinf(result["evidence"][3])
    assert result["requests"][1] == result["requests"][2] == 3


@pytest.mark.parametrize("failure", [3, 14, 15, 19, 22])
def test_invalid_or_uphill_f_trial_never_requests_trial_g(native, failure):
    result = refine(native, failure=failure, backtracks=1, normals=[[0., 0., 0.]], offsets=[1.])
    assert result["status"] == 6 and result["best_valid"]
    np.testing.assert_array_equal(result["point"], [0., 1., 3.])
    np.testing.assert_array_equal(result["best"], result["point"])
    assert result["requests"][2] == 1 and result["noninitial_gradients"] == 0
    assert result["requests"][1] == (1 if failure in {14, 22} else 2)


def test_better_f_survives_unavailable_trial_stationarity(native):
    result = refine(native, failure=18, backtracks=1, normals=[[0., 0., 0.]], offsets=[1.])
    assert result["status"] == 6 and result["best_valid"] and not result["best_accepted"]
    np.testing.assert_array_equal(result["point"], [0., 1., 3.])
    np.testing.assert_allclose(result["best"], [0., 2., 0.], rtol=0., atol=2e-14)
    assert result["evidence"][2] < result["evidence"][0] and np.isinf(result["evidence"][3])
    np.testing.assert_array_equal(result["requests"][[1, 2, 4]], [2, 2, 1])


@pytest.mark.parametrize("failure,status", [(2, 4), (4, 5), (5, 4), (6, 2), (9, 5), (10, 4), (11, 5), (12, 5), (16, 2), (17, 5), (23, 5)])
def test_unavailable_derivatives_or_constraints_keep_valid_start(native, failure, status):
    result = refine(native, failure=failure, normals=[[0., 0., 0.]], offsets=[1.])
    assert result["status"] == status and result["best_valid"] and result["trials"] == 0
    np.testing.assert_array_equal(result["point"], [0., 1., 3.])
    np.testing.assert_array_equal(result["best"], result["point"])


@pytest.mark.parametrize("failure", [7, 8, 20, 21])
def test_invalid_initial_f_or_constraints_are_not_valid_estimates(native, failure):
    result = refine(native, failure=failure, normals=[[0., 0., 0.]], offsets=[1.])
    assert result["status"] == 1 and not result["best_valid"]
    np.testing.assert_array_equal(result["point"], [0., 1., 3.])
    np.testing.assert_array_equal(result["best"], 771.)
    assert result["requests"][2] == result["requests"][4] == result["trials"] == 0


@pytest.mark.parametrize("certified", [False, True])
def test_no_unused_h_for_certified_start_or_empty_phase_budget(native, certified):
    result = refine(native, point=[0., 2., 0.] if certified else [0., 1., 3.], backtracks=0)
    assert result["status"] == (3 if certified else 6) and result["best_valid"]
    np.testing.assert_array_equal(result["point"], result["best"])
    np.testing.assert_array_equal(result["requests"][[1, 2, 4]], [1, 1, 0])
    assert result["curvature"] == result["trials"] == 0


@pytest.mark.parametrize("raw_scale", [.125, 1., 8.])
@pytest.mark.parametrize("coordinate_scale", [[1., 1., 1.], [2.**-12, 2.**9, 2.**20]])
def test_phase_respects_raw_objective_scaling_and_parameter_units(native, raw_scale, coordinate_scale):
    coordinates = np.asarray(coordinate_scale)
    hessian = raw_scale*np.eye(3)/coordinates[:, None]/coordinates
    center = coordinates*np.array([-1., 2., 0.])
    result = refine(native, point=coordinates*np.array([0., 1., 3.]), center=center,
                    hessian=hessian, units=coordinates*np.array([1., 2., .5]), scale=.25/raw_scale)
    assert result["status"] == 7
    np.testing.assert_allclose(result["best"]/coordinates, [0., 2., 0.], rtol=0., atol=3e-14)
    assert result["evidence"][3] <= 1e-6
    np.testing.assert_allclose(result["evidence"][2], 3.25+.5*raw_scale, rtol=0., atol=2e-14)
    assert result["evidence"][2] == quadratic_value(result["best"], center, hessian)


def test_short_arena_preserves_point_and_best_before_any_callback(native):
    result = refine(native, short_by=1)
    assert result["status"] == 0 and not result["best_valid"]
    np.testing.assert_array_equal(result["point"], [0., 1., 3.])
    np.testing.assert_array_equal(result["best"], 771.)
    assert not result["requests"].any() and not result["constraints"].any()
    assert result["trials"] == result["curvature"] == 0


def test_input_and_overflow_guards_before_any_callback(native):
    assert invoke(native, "guards") == {"guards": 15, "allocations": 0}


def test_single_installed_backend_and_allocation_free_shared_owners():
    headers = ["active_face_refinement.hpp", "native_active_face_phase.hpp"]
    for name in [*headers, "portable_linalg.cpp", "portable_nnls.cpp", "slsqp_translation.cpp",
                 "third_party/scipy_slsqp/slsqp.c", "third_party/scipy_slsqp/nnls.c"]:
        source = (NATIVE / name).read_text()
        source = re.sub(r"/\*.*?\*/|//[^\n]*", "", source, flags=re.DOTALL)
        assert not re.search(r"\b(?:alloca|_alloca|__builtin_alloca|malloc|calloc|realloc)\s*\(|\bnew\s+", source), name
    header = (NATIVE / headers[0]).read_text()
    assert "VLK_LOCAL_DIRECT_PROJECTION" not in header
    assert "direct_projection(" in header
    translation = (NATIVE / "slsqp_translation.cpp").read_text()
    assert translation.count('#include "third_party/scipy_slsqp/slsqp.c"') == 1
    assert "direct_projection(" in translation
