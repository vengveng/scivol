"""Independent AD regressions for dense mean Hessians and original-z maps."""

from __future__ import annotations

import ctypes
from fractions import Fraction
from pathlib import Path
import hashlib
import json
import os
import shlex
import shutil
import subprocess
import sys
import sysconfig

import numpy as np
import pytest


@pytest.fixture(scope="module")
def fresh_mean_bridge(tmp_path_factory):
    root = Path(__file__).resolve().parents[1]
    compiler = shlex.split(os.environ.get("CXX") or sysconfig.get_config_var("CXX") or "clang++")
    if not shutil.which(compiler[0]):
        pytest.skip("A C++20 compiler is required")
    directory = tmp_path_factory.mktemp("jet_math_mean_recovery")
    library = directory / "mean_recovery.so"
    clang = "clang" in Path(compiler[0]).name
    frame_flags = ["-Wframe-larger-than=16384", "-Werror=frame-larger-than"] if clang else ["-Werror=frame-larger-than=16384"]
    command = [*compiler, "-std=c++20", "-O3", "-ffp-contract=fast", "-funroll-loops",
               "-fno-exceptions", "-fno-rtti", *frame_flags]
    if os.environ.get("SCIVOL_PORTABLE_BUILD") != "1":
        command += ["-march=native"]
    command += ["-dynamiclib" if sys.platform == "darwin" else "-shared", "-fPIC"]
    command += ["-I", str(root / "scivol/_native"), "-I", str(root / "scivol/_csrc")]
    command += [str(root / "tests/native_jet_math_mean_recovery.cpp"), "-o", str(library)]
    owners = ["tests/native_jet_math_mean_recovery.cpp", "scivol/_native/constant_mean_jet.hpp",
              "scivol/_native/mean_map.hpp", "scivol/_native/mean_equation.hpp",
              "scivol/_native/constant_scale_density.hpp", "scivol/_native/garch_density_policy.hpp",
              "scivol/_native/garch_extended_density_policy.hpp", "scivol/_native/garch_transform_policy.hpp",
              "scivol/_native/location_scale_observation_assembly.hpp", "scivol/_native/runtime_reflection.hpp",
              "scivol/_native/fixed_differential.hpp"]
    sources = {name: hashlib.sha256((root / name).read_bytes()).hexdigest() for name in owners}
    subprocess.run(command, check=True, capture_output=True, text=True, timeout=60)
    assert sources == {name: hashlib.sha256((root / name).read_bytes()).hexdigest() for name in owners}
    identity = {"command": command, "sources": sources, "library": str(library),
                "sha256": hashlib.sha256(library.read_bytes()).hexdigest(),
                "compiler": subprocess.run([*compiler, "--version"], check=True, capture_output=True, text=True, timeout=10).stdout}
    (directory / "build_identity.json").write_text(json.dumps(identity, indent=2) + "\n")
    bridge = ctypes.CDLL(str(library))
    pointer, size = ctypes.c_void_p, ctypes.c_size_t
    bridge.jet_math_mean_recovery_map.argtypes = [ctypes.c_int, size, size, size, size, pointer, pointer, pointer, pointer]
    bridge.jet_math_mean_recovery_map.restype = ctypes.c_int
    bridge.jet_math_mean_recovery_jet.argtypes = [ctypes.c_int, ctypes.c_bool, size, size, size, size,
        size, ctypes.c_bool, ctypes.c_uint, pointer, pointer, pointer, size, pointer, pointer, pointer]
    bridge.jet_math_mean_recovery_jet.restype = ctypes.c_int
    bridge.compile_command = command
    return bridge


def map_point(bridge, density, z, columns, q=0, ar_start=0, ar_order=0):
    point = np.ascontiguousarray(z, dtype=np.float64)
    theta = np.empty_like(point)
    jacobian = np.empty((point.size, point.size))
    curvature = np.empty((point.size, point.size, point.size))
    assert bridge.jet_math_mean_recovery_map(density, columns, q, ar_start, ar_order,
        point.ctypes.data, theta.ctypes.data, jacobian.ctypes.data, curvature.ctypes.data)
    return theta, jacobian, curvature


def evaluate_point(bridge, density, point, y, x, *, q=0, first=0, fixed=False,
                   coordinates=False, ar_start=0, ar_order=0, request=7):
    point = np.ascontiguousarray(point, dtype=np.float64)
    y = np.ascontiguousarray(y, dtype=np.float64)
    x = np.ascontiguousarray(x, dtype=np.float64)
    f, g, h = np.full(1, 123.0), np.full(point.size, 456.0), np.full((point.size, point.size), 789.0)
    assert bridge.jet_math_mean_recovery_jet(density, fixed, x.shape[1], q, first, ar_start,
        ar_order, coordinates, request, point.ctypes.data, y.ctypes.data, x.ctypes.data,
        y.size, f.ctypes.data, g.ctypes.data, h.ctypes.data)
    for bit, output, sentinel in zip((1, 2, 4), (f, g, h), (123.0, 456.0, 789.0)):
        if not request & bit:
            np.testing.assert_array_equal(output, sentinel)
    return f[0], g, h


def assert_requested_outputs(actual, expected, request, *, rtol=3e-10, atol=3e-11):
    for bit, output, reference in zip((1, 2, 4), actual, expected):
        if request & bit:
            assert np.all(np.isfinite(output))
            np.testing.assert_allclose(output, reference, rtol=0. if bit == 1 else rtol,
                                       atol=min(atol, 2e-11) if bit == 1 else atol)


@pytest.fixture(scope="module")
def ad_modules():
    pytest.importorskip("jax")
    from scivol._devtools.ad_oracle import _jax_modules
    return _jax_modules()


def stable_tanh(jnp, coordinate):
    tail = jnp.exp(-2.0 * jnp.abs(coordinate))
    return jnp.where(coordinate >= 0.0, 1.0, -1.0) * (1.0 - 2.0 * tail / (1.0 + tail))


def reflection_coefficients(jnp, coordinates, sign):
    coefficients = jnp.zeros_like(coordinates)
    for stage in range(coordinates.size):
        previous = coefficients
        reflection = stable_tanh(jnp, coordinates[stage])
        for row in range(stage):
            coefficients = coefficients.at[row].set(previous[row] - reflection * previous[stage - 1 - row])
        coefficients = coefficients.at[stage].set(reflection)
    return sign * coefficients


def independent_mean_map(ad, point, density, columns, q=0, ar_start=0, ar_order=0):
    jax, jnp, _ = ad
    from scivol._devtools.ad_oracle import _native_shape_coordinates
    mean_count = columns + q
    mapped = point[:mean_count]
    mapped = mapped.at[ar_start:ar_start + ar_order].set(
        reflection_coefficients(jnp, point[ar_start:ar_start + ar_order], 1.0))
    mapped = mapped.at[columns:mean_count].set(reflection_coefficients(jnp, point[columns:mean_count], -1.0))
    names = ("normal", "studentt", "skewt", "ged", "gh")
    shape = _native_shape_coordinates(jax, jnp, point[mean_count + 1:], names[density])
    return jnp.concatenate((mapped, jax.nn.softplus(point[mean_count:mean_count + 1]), shape))


@pytest.mark.parametrize("fixed", [False, True])
@pytest.mark.parametrize("coordinates", [False, True])
def test_zero_column_skewt_dense_hessian_matches_ad_and_zero_exog(fresh_mean_bridge, ad_modules, fixed, coordinates):
    from scivol._devtools.mean_oracle import mean_value_grad_hess
    bridge = fresh_mean_bridge
    y = np.array([-0.8, -0.25, 0.4, 1.2])
    theta = np.array([1.3, 7.0, 0.2])
    z = np.array([np.log(np.expm1(theta[0])), np.log(np.expm1(theta[1] - 2.0 - 8.0*np.finfo(float).eps)), np.arctanh(theta[2])])
    point = z if coordinates else theta
    reference = mean_value_grad_hess(point, y, np.empty((y.size, 0)), 0, 0, 2, coordinates=coordinates)
    full = evaluate_point(bridge, 2, point, y, np.empty((y.size, 0)), fixed=fixed, coordinates=coordinates)
    assert_requested_outputs(full, reference, 7)
    for request in range(1, 8):
        actual = evaluate_point(bridge, 2, point, y, np.empty((y.size, 0)), fixed=fixed, coordinates=coordinates, request=request)
        control = evaluate_point(bridge, 2, np.r_[0.0, point], y, np.zeros((y.size, 1)), fixed=fixed, coordinates=coordinates, request=request)
        assert_requested_outputs(actual, full, request, rtol=3e-13, atol=3e-13)
        assert_requested_outputs(actual, (control[0], control[1][1:], control[2][1:, 1:]), request, rtol=3e-13, atol=3e-13)
        if request & 4 and not coordinates:
            np.testing.assert_array_equal(actual[2], actual[2].T)


@pytest.mark.parametrize("density,shapes", [(1, [0.3]), (2, [0.3, -0.7]), (3, [-0.5]), (4, [0.4, -0.9, 0.7])])
def test_fresh_source_shape_map_uses_original_z(fresh_mean_bridge, ad_modules, density, shapes):
    jax, jnp, _ = ad_modules
    z = np.array([0.02, 0.4, *shapes])
    theta, jacobian, curvature = map_point(fresh_mean_bridge, density, z, 1)
    mapping = lambda point: independent_mean_map(ad_modules, point, density, 1)
    expected = np.asarray(mapping(jnp.asarray(z)))
    np.testing.assert_allclose(theta, expected, rtol=3e-15, atol=3e-15)
    np.testing.assert_allclose(jacobian, jax.jacfwd(mapping)(jnp.asarray(z)), rtol=3e-13, atol=3e-14)
    np.testing.assert_allclose(curvature, jax.jacfwd(jax.jacfwd(mapping))(jnp.asarray(z)), rtol=3e-13, atol=3e-14)
    wrong_point = np.r_[z[:2], theta[2:]]
    wrong_jacobian = np.asarray(jax.jacfwd(mapping)(jnp.asarray(wrong_point)))
    assert np.max(np.abs(jacobian[2:, 2:] - wrong_jacobian[2:, 2:])) > 1e-2


@pytest.mark.parametrize("coordinate", [0.2, 17.0, 18.0, 18.5, -18.5])
@pytest.mark.parametrize("order", [1, 3])
def test_reflection_map_tail_jacobian_and_curvature(fresh_mean_bridge, ad_modules, coordinate, order):
    jax, jnp, _ = ad_modules
    columns, q = order + 1, order
    z = np.r_[0.02, coordinate, np.full(order - 1, 0.13), -coordinate, np.full(order - 1, -0.17), 0.4]
    theta, jacobian, curvature = map_point(fresh_mean_bridge, 0, z, columns, q, 1, order)
    mapping = lambda point: independent_mean_map(ad_modules, point, 0, columns, q, 1, order)
    np.testing.assert_allclose(theta, mapping(jnp.asarray(z)), rtol=3e-15, atol=3e-15)
    np.testing.assert_allclose(jacobian, jax.jacfwd(mapping)(jnp.asarray(z)), rtol=3e-13, atol=0)
    np.testing.assert_allclose(curvature, jax.jacfwd(jax.jacfwd(mapping))(jnp.asarray(z)), rtol=3e-13, atol=0)


def anchored_normal_objective(ad, z, theta, y, x, q, first, ar_start, ar_order):
    jax, jnp, _ = ad
    # Anchor at the actual mapped doubles. Fraction preserves binary input products.
    forcing = np.array([float(Fraction(float(observed)) - sum(
        Fraction(float(feature)) * Fraction(float(coefficient))
        for feature, coefficient in zip(row, theta[:x.shape[1]]))) for observed, row in zip(y[first:], x[first:])])
    mapped_anchor = independent_mean_map(ad, jnp.asarray(z), 0, x.shape[1], q, ar_start, ar_order)

    def objective(point):
        mapped = theta + (independent_mean_map(ad, point, 0, x.shape[1], q, ar_start, ar_order) - mapped_anchor)
        forcing_at_point = jnp.asarray(forcing) - jnp.asarray(x[first:]) @ (mapped[:x.shape[1]] - theta[:x.shape[1]])

        def advance(history, residual):
            residual = residual - mapped[x.shape[1]:x.shape[1] + q] @ history
            return jnp.concatenate((residual[None], history[:-1])) if q else history, residual

        _, residuals = jax.lax.scan(advance, jnp.zeros(q), forcing_at_point)
        variance = mapped[x.shape[1] + q]
        return 0.5*jnp.sum(jnp.log(2.0*jnp.pi) + jnp.log(variance) + residuals*residuals/variance)

    return objective


@pytest.mark.parametrize("fixed", [False, True])
@pytest.mark.parametrize("q,columns,ar_order", [(0, 4, 1), (1, 2, 1), (3, 4, 3)])
def test_complete_normal_original_z_tail_jet(fresh_mean_bridge, ad_modules, fixed, q, columns, ar_order):
    jax, jnp, _ = ad_modules
    x = np.zeros((8, columns))
    x[:, 0], x[:, 1] = 1.0, 2.0**27
    y = 2.0**27 + np.array([-1.0, 3.0, -5.0, 7.0, -2.0, 4.0, -6.0, 1.0])
    z = np.zeros(columns + q + 1)
    z[1], z[-1] = 18.5, np.log(np.expm1(1.0))
    if q:
        z[columns] = -18.5
    theta, _, _ = map_point(fresh_mean_bridge, 0, z, columns, q, 1, ar_order)
    objective = anchored_normal_objective(ad_modules, z, theta, y, x, q, q, 1, ar_order)
    reference = (float(objective(jnp.asarray(z))), np.asarray(jax.grad(objective)(jnp.asarray(z))), np.asarray(jax.hessian(objective)(jnp.asarray(z))))
    options = dict(q=q, first=q, fixed=fixed, coordinates=True, ar_start=1, ar_order=ar_order)
    full = evaluate_point(fresh_mean_bridge, 0, z, y, x, **options)
    assert_requested_outputs(full, reference, 7, rtol=3e-12, atol=3e-12)
    # Absolute tolerance cannot conceal vanishing but materially mis-scaled tail entries.
    np.testing.assert_allclose(full[1][1], reference[1][1], rtol=3e-12, atol=0)
    np.testing.assert_allclose(full[2][0, 1], reference[2][0, 1], rtol=3e-12, atol=0)
    for request in range(1, 8):
        actual = evaluate_point(fresh_mean_bridge, 0, z, y, x, request=request, **options)
        assert_requested_outputs(actual, full, request, rtol=3e-13, atol=3e-13)
        if request & 2:
            np.testing.assert_allclose(actual[1][1], reference[1][1], rtol=3e-12, atol=0)
        if request & 4:
            np.testing.assert_allclose(actual[2][0, 1], reference[2][0, 1], rtol=3e-12, atol=0)


@pytest.mark.parametrize("columns,q", [(4, 0), (2, 1), (4, 3)])
@pytest.mark.parametrize("coordinates", [False, True])
def test_general_order_skewt_theta_and_transformed_cross_blocks(fresh_mean_bridge, ad_modules, columns, q, coordinates):
    from scivol._devtools.mean_oracle import mean_value_grad_hess
    y = np.array([-0.8, -0.25, 0.4, 1.2, -0.6, 0.2, 0.7, -0.1])
    x = np.column_stack([np.ones(y.size), np.linspace(-0.4, 0.6, y.size), np.zeros((y.size, columns - 2))])
    point = np.r_[np.full(columns + q, 0.03), 1.3, 7.0, 0.2]
    if coordinates:
        point[columns:columns + q] *= -1.0
        point[columns + q:] = [np.log(np.expm1(1.3)), np.log(np.expm1(5.0)), np.arctanh(0.2)]
    reference = mean_value_grad_hess(point, y, x, q, q, 2, coordinates=coordinates)
    full = evaluate_point(fresh_mean_bridge, 2, point, y, x, q=q, first=q, coordinates=coordinates)
    assert_requested_outputs(full, reference, 7)
    for fixed in (False, True):
        for request in range(1, 8):
            actual = evaluate_point(fresh_mean_bridge, 2, point, y, x, q=q, first=q,
                coordinates=coordinates, fixed=fixed, request=request)
            assert_requested_outputs(actual, full, request, rtol=3e-13, atol=3e-13)


def test_native_abi_zero_column_skewt_hessian(ad_modules):
    from scivol import ARX, SkewT
    from scivol._mean_fit import MeanModel
    from scivol._devtools.mean_oracle import mean_value_grad_hess
    y, theta = np.array([-0.8, -0.25, 0.4, 1.2]), np.array([1.3, 7.0, 0.2])
    for fixed in (False, True):
        model = MeanModel(ARX(0, constant=False) + SkewT(), y, None, fixed=fixed)
        for coordinates in (False, True):
            point = model.map(theta, forward=False) if coordinates else theta
            reference = mean_value_grad_hess(point, model.data, np.empty((y.size, 0)), 0, 0, 2, coordinates=coordinates)
            for request in range(1, 8):
                assert_requested_outputs(model.evaluate(point, request, coordinates=coordinates), reference, request)


@pytest.mark.parametrize("density,shapes", [(1, [0.3]), (2, [0.3, -0.7]), (3, [-0.5]), (4, [0.4, -0.9, 0.7])])
@pytest.mark.parametrize("fixed", [False, True])
def test_native_abi_shape_map_uses_original_z(ad_modules, density, shapes, fixed):
    from scivol import ARX, _native_core
    from scivol._mean_fit import DENSITIES, MeanModel
    jax, jnp, _ = ad_modules
    model = MeanModel(ARX(0) + DENSITIES[density](), np.array([-0.8, -0.25, 0.4, 1.2]), None, fixed=fixed)
    z = np.array([0.02, 0.4, *shapes])
    theta, jacobian = np.empty_like(z), np.empty((z.size, z.size))
    arena = model.coordinate_workspace
    assert _native_core._mean_map(model.descriptor, True, z.ctypes.data, theta.ctypes.data,
        arena.ctypes.data, arena.nbytes, jacobian.ctypes.data)
    mapping = lambda point: independent_mean_map(ad_modules, point, density, 1)
    np.testing.assert_allclose(theta, mapping(jnp.asarray(z)), rtol=3e-15, atol=3e-15)
    np.testing.assert_allclose(jacobian, jax.jacfwd(mapping)(jnp.asarray(z)), rtol=3e-13, atol=3e-14)


@pytest.mark.parametrize("fixed", [False, True])
@pytest.mark.parametrize("linear", [False, True])
def test_native_abi_original_z_tail_jet(ad_modules, fixed, linear):
    from scivol import _native_core
    jax, jnp, _ = ad_modules
    columns, q, ar_order = (4, 0, 1) if linear else (2, 1, 1)
    y = 2.0**27 + np.array([-1.0, 3.0, -5.0, 7.0, -2.0, 4.0, -6.0, 1.0])
    x = np.zeros((y.size, columns))
    x[:, 0] = 1.0
    x[:, 1] = 2.0**27 if linear else np.r_[0.0, y[:-1]]
    z = np.zeros(columns + q + 1)
    z[1], z[-1] = 18.5, np.log(np.expm1(1.0))
    if q:
        z[columns] = -18.5
    descriptor = (y.size, columns, q, q, 1, ar_order, 0, linear, fixed, 0, 0, 1)
    assert _native_core._mean_dispatch_kind(descriptor) == ((1 if linear else 2) if fixed else 0)
    arena = np.empty(_native_core._mean_sizes(descriptor, 0)[1], dtype=np.uint8)
    theta = np.empty_like(z)
    assert _native_core._mean_map(descriptor, True, z.ctypes.data, theta.ctypes.data, arena.ctypes.data, arena.nbytes)
    objective = anchored_normal_objective(ad_modules, z, theta, y, x, q, q, 1, ar_order)
    reference = (float(objective(jnp.asarray(z))), np.asarray(jax.grad(objective)(jnp.asarray(z))), np.asarray(jax.hessian(objective)(jnp.asarray(z))))
    theta_value = np.empty(1)
    assert _native_core._mean_jet(descriptor, theta.ctypes.data, y.ctypes.data,
        x.ctypes.data if linear else 0, 1, False, arena.ctypes.data, arena.nbytes,
        theta_value.ctypes.data, 0, 0, 0)
    np.testing.assert_allclose(theta_value[0], reference[0], rtol=0, atol=3e-12)
    for request in range(1, 8):
        f, g, h = np.full(1, 123.0), np.full(z.size, 456.0), np.full((z.size, z.size), 789.0)
        assert _native_core._mean_jet(descriptor, z.ctypes.data, y.ctypes.data,
            x.ctypes.data if linear else 0, request, True, arena.ctypes.data, arena.nbytes,
            f.ctypes.data, g.ctypes.data, h.ctypes.data, 0)
        assert_requested_outputs((f[0], g, h), reference, request, rtol=3e-12, atol=3e-12)
        for bit, output, sentinel in zip((1, 2, 4), (f, g, h), (123.0, 456.0, 789.0)):
            if not request & bit:
                np.testing.assert_array_equal(output, sentinel)
        if request & 2:
            np.testing.assert_allclose(g[1], reference[1][1], rtol=3e-12, atol=0)
        if request & 4:
            np.testing.assert_allclose(h[0, 1], reference[2][0, 1], rtol=3e-12, atol=0)
