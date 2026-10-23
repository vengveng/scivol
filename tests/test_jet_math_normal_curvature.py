"""Normal range recovery and the shared frozen-scale density contraction."""

import hashlib
import json
from pathlib import Path
import shutil
import subprocess

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "tests/native_jet_math_normal_curvature.cpp"
SHAPES = {
    "normal": [], "studentt": [7.0], "skewt": [7.0, 0.2], "ged": [1.7],
    "gh": [0.1, 1.8, 0.4], "nig": [0.1, 1.8],
    "hyperbolic": [0.1, 1.8], "vg": [0.1, 2.0],
}
SENTINEL = 12345.0


@pytest.fixture(scope="module")
def standalone_rows(tmp_path_factory):
    compiler = shutil.which("clang++") or shutil.which("g++")
    if compiler is None:
        pytest.skip("A C++20 compiler is required")
    directory = tmp_path_factory.mktemp("normal_curvature")
    executable = directory / "normal_curvature"
    frame_flags = (["-Wframe-larger-than=16384", "-Werror=frame-larger-than"]
                   if "clang" in Path(compiler).name
                   else ["-Werror=frame-larger-than=16384"])
    command = [
        compiler, "-std=c++20", "-O3", "-ffp-contract=fast", "-funroll-loops",
        "-march=native", "-fno-exceptions", "-fno-rtti", *frame_flags,
        "-I", str(ROOT / "scivol/_native"),
        "-I", str(ROOT / "scivol/_csrc"),
        "-I", str(ROOT / "scivol/_native/third_party/boost"),
        "-I", str(ROOT / "scivol/_native/third_party/tlfloat"),
        str(SOURCE), str(ROOT / "scivol/_native/gh_math_translation.cpp"),
        "-o", str(executable),
    ]
    compiled = subprocess.run(command, capture_output=True, text=True, timeout=60)
    assert compiled.returncode == 0, compiled.stderr
    completed = subprocess.run([str(executable)], capture_output=True, text=True, timeout=10)
    assert completed.returncode == 0, completed.stdout + completed.stderr
    rows = [json.loads(line) for line in completed.stdout.splitlines()]
    identity = {
        "compile_command": command,
        "binary_sha256": hashlib.sha256(executable.read_bytes()).hexdigest(),
        "source_sha256": hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
    }
    (directory / "identity.json").write_text(json.dumps(identity, indent=2) + "\n")
    (directory / "rows.json").write_text(json.dumps(rows, indent=2) + "\n")
    return rows


def requested_outputs(row):
    width = len(row["gradient"])
    return row["value"], np.array(row["gradient"]), np.array(row["hessian"]).reshape(width, width)


def assert_mask_projection(row, reference, scales=None):
    value, gradient, hessian = requested_outputs(row)
    expected_value, expected_gradient, expected_hessian = reference
    request = row["request"]
    if request & 1:
        np.testing.assert_allclose(value, expected_value, rtol=0.0, atol=2e-11)
    else:
        assert value == SENTINEL
    scales = np.ones(gradient.size) if scales is None else np.asarray(scales)
    if request & 2:
        tolerance = 5e-12 * np.abs(expected_gradient) + 5e-13 / scales
        assert np.isfinite(gradient).all()
        assert np.all(np.abs(gradient - expected_gradient) <= tolerance)
    else:
        np.testing.assert_array_equal(gradient, SENTINEL)
    if request & 4:
        tolerance = (5e-13 / scales[:, None]) / scales[None, :]
        tolerance += 2e-11 * np.abs(expected_hessian)
        assert np.isfinite(hessian).all()
        assert np.all(np.abs(hessian - expected_hessian) <= tolerance)
    else:
        np.testing.assert_array_equal(hessian, SENTINEL)


@pytest.mark.parametrize("family", SHAPES)
def test_all_density_frozen_scale_cross_blocks_and_sparse_requests(standalone_rows, family):
    rows = [row for row in standalone_rows if row["kind"] == "local" and row["family"] == family]
    unit = next(row for row in rows if row["scale"] == 1.0 and row["request"] == 7)
    value, gradient, hessian = requested_outputs(unit)
    assert len(rows) == 21
    for row in rows:
        assert_mask_projection(row, (value + 0.5 * np.log(row["scale"]), gradient, hessian))


def local_density_objective(jnp, jsp, family, point):
    coordinates, shapes = point[:2], point[2:]
    residual = -0.7 + jnp.array([0.2, -0.3]) @ coordinates
    residual += 0.5 * coordinates @ jnp.array([[0.03, -0.02], [-0.02, 0.01]]) @ coordinates
    variance = 1.0 + jnp.array([0.5, 0.7]) @ coordinates
    variance += 0.5 * coordinates @ jnp.array([[0.04, 0.06], [0.06, -0.03]]) @ coordinates
    standardized = residual / jnp.sqrt(variance)
    if family == "normal":
        return 0.5 * (jnp.log(2.0 * jnp.pi) + jnp.log(variance) + standardized**2)
    if family in {"studentt", "skewt"}:
        nu = shapes[0]
        log_constant = (jsp.special.gammaln(0.5 * (nu + 1.0))
                        - jsp.special.gammaln(0.5 * nu)
                        - 0.5 * jnp.log(jnp.pi * (nu - 2.0)))
        log_scale = 0.0
        if family == "skewt":
            skew = shapes[1]
            shift = 4.0 * skew * jnp.exp(log_constant) * (nu - 2.0) / (nu - 1.0)
            scale = jnp.sqrt(1.0 + 3.0 * skew**2 - shift**2)
            centered = scale * standardized + shift
            standardized = centered / (1.0 + skew * jnp.where(centered < 0.0, -1.0, 1.0))
            log_scale = jnp.log(scale)
        return (0.5 * jnp.log(variance) - log_constant - log_scale
                + 0.5 * (nu + 1.0) * jnp.log1p(standardized**2 / (nu - 2.0)))
    nu = shapes[0]
    log_scale = 0.5 * (jsp.special.gammaln(1.0 / nu) - jsp.special.gammaln(3.0 / nu))
    log_constant = jnp.log(nu) - jnp.log(2.0) - log_scale - jsp.special.gammaln(1.0 / nu)
    return (0.5 * jnp.log(variance) - log_constant
            + jnp.exp(nu * (jnp.log(jnp.abs(standardized)) - log_scale)))


@pytest.mark.parametrize("family", ["normal", "studentt", "skewt", "ged"])
def test_frozen_density_assembly_matches_independent_ad(standalone_rows, family):
    from scivol._devtools.ad_oracle import _jax_modules, _value_grad_hess
    _, jnp, jsp = _jax_modules()
    point = np.array([0.0, 0.0, *SHAPES[family]])
    expected = _value_grad_hess(lambda theta: local_density_objective(jnp, jsp, family, theta), point)
    for row in standalone_rows:
        if row["kind"] == "local" and row["family"] == family:
            assert_mask_projection(row, (expected[0] + 0.5 * np.log(row["scale"]), *expected[1:]))


def model_parameters(row):
    p, q = row["p"], row["q"]
    leverage = "gjr" in row["family"]
    linear = row["family"].startswith("linear")
    arma = row["family"].startswith("arma")
    mean_width = 2 if linear else (3 if row["kind"] == "arma11" else 1) if arma else 0
    mean = [0.04, -0.02] if linear else ([0.04, 0.15, -0.1] if mean_width == 3 else [0.04]) if arma else []
    theta = np.array(mean + [0.2] + [0.1 / p] * p + ([0.05 / p] * p if leverage else []) + [0.6 / q] * q)
    scales = np.ones(theta.size)
    if mean_width:
        scales[:2 if linear else 1] = np.sqrt(row["scale"])
    scales[mean_width] = row["scale"]
    return theta, scales, mean_width, leverage, linear, arma


def normal_model_objective(jnp, theta, row):
    p, q = row["p"], row["q"]
    _, _, mean_width, leverage, linear, arma = model_parameters(row)
    y = jnp.array([0.8, -0.7, 0.4, -0.5, 0.3, 0.9, -0.6, 0.2])
    x = jnp.array([[1.0, value] for value in [0.2, -0.3, 0.4, -0.5, 0.6, -0.7, 0.8, -0.9]])
    omega = theta[mean_width]
    alpha = theta[mean_width + 1:mean_width + 1 + p]
    beta = theta[-q:]
    gamma = theta[mean_width + 1 + p:mean_width + 1 + 2 * p] if leverage else jnp.zeros(p)
    start = max(p, q) if arma else 0
    residuals = [jnp.array(0.0)] * start
    variances = [jnp.array(1.0)] * start
    value = jnp.array(0.0)
    for time in range(start, len(y)):
        residual = y[time]
        if linear:
            residual -= x[time] @ theta[:mean_width]
        elif arma:
            residual -= theta[0]
            if mean_width == 3:
                residual -= theta[1] * y[time - 1] + theta[2] * residuals[time - 1]
        variance = jnp.array(1.0) if time == 0 else omega
        if time:
            for lag in range(min(p, time)):
                previous = residuals[time - 1 - lag]
                variance += (alpha[lag] + gamma[lag] * (previous < 0.0)) * previous**2
            for lag in range(min(q, time)):
                variance += beta[lag] * variances[time - 1 - lag]
        residuals.append(residual)
        variances.append(variance)
        value += 0.5 * (jnp.log(2.0 * jnp.pi) + jnp.log(variance) + residual**2 / variance)
    return value


def test_complete_normal_models_match_ad_in_normalized_units(standalone_rows):
    from scivol._devtools.ad_oracle import _jax_modules, _value_grad_hess
    _, jnp, _ = _jax_modules()
    references = {}
    for row in standalone_rows:
        if row["kind"] in {"local", "seed", "zero", "constant", "ged_tail"}:
            continue
        key = row["family"], row["p"], row["q"], row["kind"]
        theta, scales, _, _, _, _ = model_parameters(row)
        if key not in references:
            references[key] = _value_grad_hess(lambda point: normal_model_objective(jnp, point, row), theta)
        value, gradient, hessian = references[key]
        scored = 8 - max(row["p"], row["q"]) if row["family"].startswith("arma") else 8
        expected = (value + 0.5 * scored * np.log(row["scale"]),
                    gradient / scales, (hessian / scales[:, None]) / scales[None, :])
        assert_mask_projection(row, expected, scales)


def test_subnormal_conditioning_seed_keeps_composed_hessian_finite(standalone_rows):
    rows = [row for row in standalone_rows if row["kind"] == "seed"]
    assert len(rows) == 7
    reference = next(row for row in rows if row["request"] == 7)
    value, gradient, hessian = requested_outputs(reference)
    np.testing.assert_allclose(value, -365.39406400563286, rtol=0.0, atol=1e-12)
    np.testing.assert_allclose(gradient[0], 35.0, rtol=2e-14, atol=0.0)
    assert np.isfinite(hessian).all()
    assert hessian[0, 0] < -300.0
    for row in rows:
        assert_mask_projection(row, (value, gradient, hessian))


def test_normal_rm01_zero_residual_curvature(standalone_rows):
    from scivol._devtools.ad_oracle import _jax_modules, _value_grad_hess
    _, jnp, _ = _jax_modules()
    unit = np.array([1.0, 0.1, 0.1])
    reference = _value_grad_hess(
        lambda theta: np.log(2.0 * np.pi) + 0.5 * jnp.log(theta[0] + theta[2]), unit)
    for row in standalone_rows:
        if row["kind"] != "zero":
            continue
        scales = np.array([row["scale"], 1.0, 1.0])
        value, gradient, hessian = reference
        expected = (value + np.log(row["scale"]), gradient / scales,
                    (hessian / scales[:, None]) / scales[None, :])
        assert_mask_projection(row, expected, scales)
        if row["request"] & 4:
            actual = requested_outputs(row)[2][2, 2]
            np.testing.assert_allclose(actual, -0.4132231404958677, rtol=3e-15, atol=0.0)


def test_frozen_constant_scale_preserves_mean_variance_cross_term(standalone_rows):
    scales = np.array([1.0, 1.0, 1e308])
    gradient = np.array([100.0, -10.0, -49.5]) / scales
    unit_hessian = np.array([[100.0, -10.0, -100.0], [-10.0, 1.0, 10.0], [-100.0, 10.0, 99.5]])
    hessian = (unit_hessian / scales[:, None]) / scales[None, :]
    value = 0.5 * (np.log(2.0 * np.pi) + np.log(1e308) + 100.0)
    rows = [row for row in standalone_rows if row["kind"] == "constant"]
    assert len(rows) == 7
    for row in rows:
        assert_mask_projection(row, (value, gradient, hessian), scales)


def test_ged_nonzero_tail_is_not_reclassified_as_zero_residual(standalone_rows):
    mp = pytest.importorskip("mpmath")
    rows = [row for row in standalone_rows if row["kind"] == "ged_tail"]
    assert len(rows) == 1
    _, gradient, hessian = requested_outputs(rows[0])
    with mp.workdps(100):
        nu, variance, residual = map(mp.mpf, [1.7, 1e200, 1e-300])
        log_scale = (mp.loggamma(1 / nu) - mp.loggamma(3 / nu)) / 2
        expected = nu * (nu - 1) * mp.exp(
            (nu - 2) * mp.log(residual) - nu * (mp.log(variance) / 2 + log_scale))
    np.testing.assert_allclose(hessian[0, 0], float(expected), rtol=3e-13, atol=0.0)
    assert np.isfinite(gradient).all() and np.isfinite(hessian).all()
    np.testing.assert_allclose(hessian[1, 1], -0.5, rtol=3e-15, atol=0.0)


def evaluate_reduction_jet(native, family, theta, data, seed, p, q, request):
    workspace_bytes = (native._garch_workspace_bytes(p, q, 0, 0, request, False)
                       if family == "garch" else native._aparch_workspace_bytes(p, q, 0, 0, request))
    workspace = np.empty(workspace_bytes + 63, dtype=np.uint8)
    address = (workspace.ctypes.data + 63) & ~63
    available = workspace.nbytes - (address - workspace.ctypes.data)
    value = np.full(1, SENTINEL)
    gradient = np.full(theta.size, SENTINEL)
    hessian = np.full((theta.size, theta.size), SENTINEL)
    outputs = (address, available, value.ctypes.data, gradient.ctypes.data, hessian.ctypes.data)
    if family == "garch":
        status, valid = native._garch_jet(
            theta.ctypes.data, data.ctypes.data, data.ctypes.data, data.size, seed,
            p, q, 0, 0, request, *outputs, False)
    else:
        status, valid = native._aparch_jet(
            theta.ctypes.data, theta.size, data.ctypes.data, data.size, seed,
            p, q, 0, 0, request, *outputs)
    assert (status, valid) == (0, request)
    return value[0], gradient, hessian


@pytest.mark.parametrize("p,q", [(1, 1), (3, 2)])
def test_fresh_extension_normal_curvature_equals_aparch_reduction(p, q):
    from scivol import _native_core as native
    scale = 1e200
    data = np.zeros(max(p, q) + 2, dtype=np.float64)
    garch = np.array([scale] + [0.1 / p] * p + [0.1 / q] * q)
    aparch = np.array([scale] + [0.1 / p] * p + [0.0] * p + [0.1 / q] * q + [2.0])
    common = np.array([0] + list(range(1, 1 + p)) + list(range(1 + 2 * p, 1 + 2 * p + q)))
    scales = np.array([scale] + [1.0] * (p + q))
    for request in range(1, 8):
        actual = evaluate_reduction_jet(native, "garch", garch, data, scale, p, q, request)
        reduced = evaluate_reduction_jet(native, "aparch", aparch, data, scale, p, q, request)
        if request & 1:
            np.testing.assert_allclose(actual[0], reduced[0], rtol=0.0, atol=3e-12)
        if request & 2:
            tolerance = 5e-12 * np.abs(reduced[1][common]) + 5e-13 / scales
            assert np.all(np.abs(actual[1] - reduced[1][common]) <= tolerance)
        if request & 4:
            reference = reduced[2][np.ix_(common, common)]
            tolerance = (5e-13 / scales[:, None]) / scales[None, :]
            tolerance += 2e-11 * np.abs(reference)
            assert np.isfinite(actual[2]).all() and np.isfinite(reference).all()
            assert np.all(np.abs(actual[2] - reference) <= tolerance)
