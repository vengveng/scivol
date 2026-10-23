"""Independent AD, fixed-layout parity, and checked power-constraint arenas."""

from __future__ import annotations

import ctypes
import hashlib
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sysconfig

import numpy as np
import pytest

from scivol._devtools.ad_oracle import _jax_aparch_power_moment, _jax_modules
from scivol._devtools.ad_oracle import (
    arma_aparch_value_grad_hess,
    native_aparch_coordinate_chain_rule_value_grad_hess,
)
from scivol._devtools.joint_mean_oracle import joint_value_grad_hess


DOUBLE_PTR = ctypes.POINTER(ctypes.c_double)
SIZE = ctypes.c_size_t
SIZE_PTR = ctypes.POINTER(SIZE)
DENSITIES = [
    ("normal", 0, []),
    ("studentt", 1, [8.0]),
    ("skewt", 2, [8.0, 0.12]),
    ("ged", 3, [1.6]),
    ("gh", 4, [0.1, 1.3, 0.4]),
    ("nig", 5, [0.1, 1.3]),
    ("hyperbolic", 6, [0.1, 1.3]),
    ("vg", 7, [0.1, 3.0]),
]
ELEMENTARY_DENSITIES = [DENSITIES[i] for i in (0, 1, 3)]


def copied_counted_headers(root, directory):
    native = directory / "scivol/_native"
    sources = list((root / "scivol/_native").glob("*.hpp"))
    sources += list((root / "scivol/_native").glob("*.h"))
    sources += list((root / "scivol/_native/third_party/scipy_slsqp").glob("*.h"))
    sources += list((root / "scivol/_csrc").glob("*.h"))
    sources += [root / "scivol/_native" / f"{name}.cpp"
                for name in ("aparch_moment_precision", "gh_math_translation")]
    sources += [root / "scivol/_csrc/gh_math.c", root / "tests/native_power_constraint_curvature.cpp"]
    hashes = {}
    for source in sources:
        target = directory / source.relative_to(root)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
        hashes[str(source.relative_to(root))] = hashlib.sha256(source.read_bytes()).hexdigest()
    header = native / "aparch_moment_policy.hpp"
    owner = header.read_text()
    entry = "    using std::log; using std::lgamma;\n"
    counter = (
        "    if constexpr (requires(Number x) {x.value;}) {\n"
        "        constexpr unsigned order=[] {\n"
        "            if constexpr (requires(Number x) {x.hessian;}) return 2u;\n"
        "            else if constexpr (requires(Number x) {x.gradient;}) return 1u;\n"
        "            else return 0u;\n"
        "        }();\n"
        "        ++::aparch_test::common_preparations[order];\n"
        "    }\n"
    )
    assert owner.count(entry) == 1
    declaration = '#include "aparch_test_counts.hpp"\n'
    instrumented = declaration + owner.replace(entry, counter + entry)
    assert instrumented.removeprefix(declaration).replace(counter, "") == owner
    header.write_text(instrumented)
    (native / "aparch_test_counts.hpp").write_text(
        "#pragma once\n#include <array>\n#include <cstddef>\n"
        "namespace aparch_test {extern std::array<std::size_t,3> common_preparations;}\n"
    )
    return native, hashes


class Config(ctypes.Structure):
    _fields_ = [
        *[(name, SIZE) for name in (
            "density", "aparch", "transformed", "columns", "p", "q",
            "first_order", "first_parameters", "first_offset",
            "second_order", "second_parameters", "second_offset",
        )],
        ("first_basis", DOUBLE_PTR), ("second_basis", DOUBLE_PTR),
        ("first_scale", ctypes.c_double), ("second_scale", ctypes.c_double),
        ("fixed", SIZE),
    ]

    @classmethod
    def make(cls, code=0, *, aparch=True, transformed=False, columns=0, p=1, q=1, fixed=0,
             first=(0, 0, 0, None, 1.0), second=(0, 0, 0, None, 1.0)):
        result = cls(code, aparch, transformed, columns, p, q)
        result.fixed = fixed
        result._bases = []
        for prefix, (order, parameters, offset, basis, scale) in (
            ("first", first), ("second", second)
        ):
            for suffix, value in zip(("order", "parameters", "offset", "scale"),
                                     (order, parameters, offset, scale)):
                setattr(result, f"{prefix}_{suffix}", value)
            if basis is not None:
                array = np.ascontiguousarray(basis, dtype=np.float64)
                assert array.shape == (order, parameters)
                result._bases.append(array)
                setattr(result, f"{prefix}_basis", array.ctypes.data_as(DOUBLE_PTR))
        return result


class JointConfig(ctypes.Structure):
    _fields_ = [(name, SIZE) for name in ("density", "columns", "fixed")]

    @property
    def mean(self):
        return self.columns or 4

    @property
    def alpha(self):
        return self.mean + 1

    @property
    def gamma(self):
        return self.alpha + 3

    @property
    def beta(self):
        return self.gamma + 3

    @property
    def delta(self):
        return self.beta + 2


def compile_counted_bridge(root, directory):
    compiler = shlex.split(os.environ.get("CXX") or sysconfig.get_config_var("CXX") or "c++")
    if not compiler or not shutil.which(compiler[0]):
        pytest.skip("C++20 compiler required")
    native, hashes = copied_counted_headers(root, directory)
    library = directory / "curvature.so"
    frame = (["-Wframe-larger-than=16384", "-Werror=frame-larger-than"]
             if "clang" in Path(compiler[0]).name or os.sys.platform == "darwin"
             else ["-Werror=frame-larger-than=16384"])
    command = [
        *compiler, "-std=c++20", "-O3", "-ffp-contract=fast", "-funroll-loops", "-march=native",
        "-fno-exceptions", "-fno-rtti",
        "-fPIC", "-shared", *frame,
        "-I", str(native), "-I", str(native.parent / "_csrc"),
        "-I", str(root / "scivol/_native/third_party/boost"),
        "-I", str(root / "scivol/_native/third_party/tlfloat"),
        str(directory / "tests/native_power_constraint_curvature.cpp"),
        str(native / "aparch_moment_precision.cpp"),
        str(native / "gh_math_translation.cpp"), "-o", str(library),
    ]
    compiled = subprocess.run(command, capture_output=True, text=True, timeout=105)
    (directory / "build.json").write_text(json.dumps({
        "command": command, "source_hashes": hashes,
        "image_sha256": hashlib.sha256(library.read_bytes()).hexdigest() if library.exists() else None,
        "compiler_stderr": compiled.stderr,
        "scope": "fresh copied-header test image with private numerical backends; no installed extension",
    }, indent=2) + "\n")
    assert compiled.returncode == 0, compiled.stderr
    assert all(hashlib.sha256((root / path).read_bytes()).hexdigest() == digest
               for path, digest in hashes.items()), "Source drift during bridge build"
    return library


@pytest.fixture(scope="module")
def bridge(tmp_path_factory):
    root = Path(__file__).resolve().parents[1]
    reuse = os.environ.get("SCIVOL_TEST_POWER_BRIDGE")
    if reuse:
        directory = Path(reuse)
        library = directory / "curvature.so"
        record = json.loads((directory / "build.json").read_text())
        assert all(hashlib.sha256((root / path).read_bytes()).hexdigest() == digest
                   for path, digest in record["source_hashes"].items()), "Stale test bridge sources"
        assert hashlib.sha256(library.read_bytes()).hexdigest() == record["image_sha256"], "Changed test image"
    else:
        directory = tmp_path_factory.mktemp("power_constraint_curvature")
        library = compile_counted_bridge(root, directory)
    loaded = ctypes.CDLL(str(library))
    loaded.curvature_test_workspace_bytes.argtypes = [ctypes.POINTER(Config)]
    loaded.curvature_test_workspace_bytes.restype = SIZE
    loaded.curvature_test_weighted_hessian.argtypes = [
        ctypes.POINTER(Config), DOUBLE_PTR, DOUBLE_PTR, DOUBLE_PTR, ctypes.c_void_p, SIZE,
    ]
    loaded.curvature_test_weighted_hessian.restype = ctypes.c_int
    loaded.curvature_test_overflow_contracts.restype = ctypes.c_int
    loaded.curvature_test_preparation_contracts.restype = ctypes.c_int
    loaded.curvature_test_first_order_only_range.restype = ctypes.c_int
    loaded.curvature_test_constraint_size.argtypes = [ctypes.POINTER(Config), ctypes.c_uint]
    loaded.curvature_test_constraint_size.restype = SIZE
    loaded.curvature_test_constraints.argtypes = [
        ctypes.POINTER(Config), ctypes.c_uint, DOUBLE_PTR, DOUBLE_PTR, DOUBLE_PTR,
        ctypes.c_void_p, SIZE, SIZE_PTR, SIZE,
    ]
    loaded.curvature_test_constraints.restype = ctypes.c_int
    loaded.curvature_test_parameter_units.argtypes = [
        ctypes.POINTER(Config), DOUBLE_PTR, DOUBLE_PTR, ctypes.c_double, DOUBLE_PTR,
    ]
    loaded.curvature_test_parameter_units.restype = ctypes.c_int
    loaded.curvature_test_arbitrary_fixed_layout.restype = ctypes.c_int
    loaded.curvature_test_reset_common_preparations.argtypes = []
    loaded.curvature_test_reset_common_preparations.restype = None
    loaded.curvature_test_common_preparations.argtypes = [SIZE_PTR]
    loaded.curvature_test_common_preparations.restype = None
    loaded.curvature_test_joint_workspace.argtypes = [ctypes.POINTER(JointConfig), *[ctypes.c_uint] * 3]
    loaded.curvature_test_joint_workspace.restype = SIZE
    loaded.curvature_test_joint_jet.argtypes = [
        ctypes.POINTER(JointConfig), ctypes.c_uint, ctypes.c_uint, DOUBLE_PTR, DOUBLE_PTR,
        DOUBLE_PTR, SIZE, ctypes.c_double, ctypes.c_void_p, SIZE, DOUBLE_PTR, DOUBLE_PTR, DOUBLE_PTR,
    ]
    loaded.curvature_test_joint_jet.restype = ctypes.c_int
    loaded.curvature_test_joint_map.argtypes = [
        ctypes.POINTER(JointConfig), ctypes.c_uint, DOUBLE_PTR, DOUBLE_PTR,
        ctypes.c_void_p, SIZE, DOUBLE_PTR, DOUBLE_PTR, DOUBLE_PTR,
    ]
    loaded.curvature_test_joint_map.restype = ctypes.c_int
    loaded._build_directory = directory
    return loaded


def evaluate(bridge, config, theta, weights, *, short=False, null_memory=False, unaligned=False):
    theta = np.ascontiguousarray(theta, dtype=np.float64)
    weights = None if weights is None else np.ascontiguousarray(weights, dtype=np.float64)
    size = bridge.curvature_test_workspace_bytes(ctypes.byref(config))
    assert size
    # Reserve an aligned caller-owned arena, with sentinel bytes on both sides.
    storage = np.full(size + 64, 0xA5, dtype=np.uint8)
    offset = 16 + (-storage.ctypes.data % 16) + unaligned
    pointer = storage.ctypes.data + offset
    output = np.full((theta.size, theta.size), 123.0)
    status = bridge.curvature_test_weighted_hessian(
        ctypes.byref(config), theta.ctypes.data_as(DOUBLE_PTR),
        None if weights is None else weights.ctypes.data_as(DOUBLE_PTR),
        output.ctypes.data_as(DOUBLE_PTR), None if null_memory else pointer, size - short,
    )
    assert status != -1, "Constraint evaluation allocated heap storage"
    assert (storage[:offset] == 0xA5).all()
    assert (storage[offset + size:] == 0xA5).all()
    if not status:
        assert (output == 123.0).all()
    return status, output


def power_theta(shape, *, columns=0, p=1, q=1, delta=1.7):
    return np.r_[np.zeros(columns), 0.08, np.linspace(0.03, 0.07, p),
                 np.linspace(-0.17, 0.15, p), np.full(q, 0.5 / q), delta, shape]


def evaluate_constraints(bridge, config, theta, *, request=2, short=False,
                         null_memory=False, unaligned=False, columns=None):
    theta = np.ascontiguousarray(theta, dtype=np.float64)
    assert bridge.curvature_test_constraint_size(ctypes.byref(config), 0) == theta.size
    rows = bridge.curvature_test_constraint_size(ctypes.byref(config), 1)
    size = bridge.curvature_test_constraint_size(ctypes.byref(config), 2)
    assert size
    storage = np.full(size + 64, 0xA5, dtype=np.uint8)
    offset = 16 + (-storage.ctypes.data % 16) + unaligned
    values = np.full(rows, 123.0)
    jacobian = np.full((rows, theta.size), 456.0, order="F")
    columns = None if columns is None else np.ascontiguousarray(columns, dtype=np.uintp)
    status = bridge.curvature_test_constraints(
        ctypes.byref(config), request, theta.ctypes.data_as(DOUBLE_PTR),
        values.ctypes.data_as(DOUBLE_PTR), jacobian.ctypes.data_as(DOUBLE_PTR),
        None if null_memory else storage.ctypes.data + offset, size - short,
        None if columns is None else columns.ctypes.data_as(SIZE_PTR),
        0 if columns is None else columns.size,
    )
    assert status != -1, "Constraint evaluation allocated heap storage"
    assert (storage[:offset] == 0xA5).all()
    assert (storage[offset + size:] == 0xA5).all()
    if request == 0 or short or null_memory:
        np.testing.assert_array_equal(jacobian, 456.0)
    if request == 1 or short or null_memory:
        np.testing.assert_array_equal(values, 123.0)
    return status, values, jacobian


FIXED_LAYOUTS = [(4, 1, 1), (7, 3, 2)]


def fixed_constraint_case(code, shape, columns, p, q, *, aparch=True, transformed=False):
    first_order = 2 if columns == 4 else 3
    config = Config.make(
        code, aparch=aparch, transformed=transformed, columns=columns, p=p, q=q,
        first=(first_order, first_order, 0, None, 1.0),
        second=(2, 2, first_order, None, -1.0),
    )
    theta = power_theta(shape, columns=columns, p=p, q=q)
    if not aparch:
        theta = np.delete(theta, columns + 1 + 2 * p + q)
    theta[:columns] = np.linspace(0.12, -0.06, columns)
    theta[columns + 1] = 0.0
    if p > 1:
        theta[columns + 2] = -0.05
    return config, theta


@pytest.mark.parametrize("name,code,shape", DENSITIES)
@pytest.mark.parametrize("columns,p,q", FIXED_LAYOUTS, ids=["C4-11", "C7-32"])
@pytest.mark.parametrize("aparch", [True, False], ids=["aparch", "egarch"])
def test_fixed_runtime_constraint_requests_and_live_curvature(
        bridge, name, code, shape, columns, p, q, aparch):
    config, theta = fixed_constraint_case(code, shape, columns, p, q, aparch=aparch)
    status, values, jacobian = evaluate_constraints(bridge, config, theta)
    assert status == 1
    assert np.isfinite(values).all() and np.isfinite(jacobian).all()
    weights = np.linspace(-0.9, 0.7, values.size)
    status, hessian = evaluate(bridge, config, theta, weights)
    assert status == 1 and np.isfinite(hessian).all()
    assert np.any(hessian != 0.0), "Weighted H must execute, not just instantiate"
    np.testing.assert_array_equal(hessian, hessian.T)
    if aparch:
        alpha = columns + 1
        gamma = alpha + p
        delta = gamma + p + q
        assert hessian[alpha, delta] != 0.0
        assert hessian[gamma, gamma] == 0.0
    else:
        beta = columns + 1 + 2 * p
        if q > 1:
            assert np.any(hessian[beta:beta + q, beta:beta + q] != 0.0)
    base = np.geomspace(0.2, 3.0, theta.size)
    units = np.full_like(theta, 789.0)
    assert bridge.curvature_test_parameter_units(
        ctypes.byref(config), theta.ctypes.data_as(DOUBLE_PTR),
        base.ctypes.data_as(DOUBLE_PTR), 2.5, units.ctypes.data_as(DOUBLE_PTR),
    ) == 1
    runtime_bytes = bridge.curvature_test_workspace_bytes(ctypes.byref(config))
    # Mode 2 clears the program's runtime layout, exposing accidental field reads.
    for mode in (0, 1, 2):
        config.fixed = mode
        assert bridge.curvature_test_workspace_bytes(ctypes.byref(config)) == runtime_bytes
        for request in (0, 1, 2):
            status, actual_values, actual_jacobian = evaluate_constraints(
                bridge, config, theta, request=request, unaligned=mode == 2)
            assert status == 1
            if request != 1:
                np.testing.assert_allclose(actual_values, values, rtol=3e-13, atol=3e-14)
            if request != 0:
                np.testing.assert_allclose(actual_jacobian, jacobian, rtol=3e-13, atol=3e-14)
        status, actual_hessian = evaluate(bridge, config, theta, weights, unaligned=mode == 2)
        assert status == 1
        np.testing.assert_allclose(actual_hessian, hessian, rtol=3e-13, atol=3e-14)
        actual_units = np.full_like(theta, 789.0)
        assert bridge.curvature_test_parameter_units(
            ctypes.byref(config), theta.ctypes.data_as(DOUBLE_PTR),
            base.ctypes.data_as(DOUBLE_PTR), 2.5, actual_units.ctypes.data_as(DOUBLE_PTR),
        ) == 1
        np.testing.assert_array_equal(actual_units, units)


@pytest.mark.parametrize("columns,p,q", FIXED_LAYOUTS, ids=["C4-11", "C7-32"])
@pytest.mark.parametrize("aparch", [True, False], ids=["aparch", "egarch"])
def test_fixed_transformed_constraints_preserve_sparse_outputs(bridge, columns, p, q, aparch):
    config, theta = fixed_constraint_case(0, [], columns, p, q, aparch=aparch, transformed=True)
    status, values, jacobian = evaluate_constraints(bridge, config, theta)
    assert status == 1
    weights = np.linspace(-0.8, 0.4, values.size)
    status, hessian = evaluate(bridge, config, theta, weights)
    assert status == 1 and np.any(hessian != 0.0)
    config.fixed = 2
    for request in (0, 1, 2):
        status, actual_values, actual_jacobian = evaluate_constraints(
            bridge, config, theta, request=request)
        assert status == 1
        if request != 1:
            np.testing.assert_array_equal(actual_values, values)
        if request != 0:
            np.testing.assert_array_equal(actual_jacobian, jacobian)
    status, actual = evaluate(bridge, config, theta, weights)
    assert status == 1
    np.testing.assert_array_equal(actual, hessian)


@pytest.mark.parametrize("columns,p,q", FIXED_LAYOUTS, ids=["C4-11", "C7-32"])
@pytest.mark.parametrize("alpha,weight,delta", [
    (0.0, 1.0, 298.0), (0.0, 1e-100, 310.0),
    (1e-200, 1e-200, 310.0), (0.04, -1e-100, 310.0),
])
def test_fixed_curvature_keeps_zero_alpha_and_scaled_range_fallback(
        bridge, columns, p, q, alpha, weight, delta):
    config = Config.make(columns=columns, p=p, q=q)
    theta = power_theta([], columns=columns, p=p, q=q, delta=delta)
    alpha_start = columns + 1
    gamma_start = alpha_start + p
    theta[alpha_start:gamma_start] = 0.0
    theta[gamma_start:gamma_start + p] = 0.0 if delta == 298.0 else 0.15
    theta[alpha_start + p - 1] = alpha
    status, reference = evaluate(bridge, config, theta, [weight])
    assert status == 1 and np.isfinite(reference).all()
    for mode in (1, 2):
        config.fixed = mode
        status, actual = evaluate(bridge, config, theta, [weight])
        assert status == 1
        np.testing.assert_allclose(actual, reference, rtol=2e-12, atol=0.0)
        assert actual[alpha_start + p - 1, -1] != 0.0
        if alpha == 0.0:
            np.testing.assert_array_equal(actual[gamma_start:, gamma_start:], 0.0)


@pytest.mark.parametrize("columns,p,q", FIXED_LAYOUTS, ids=["C4-11", "C7-32"])
@pytest.mark.parametrize("aparch", [True, False], ids=["aparch", "egarch"])
def test_fixed_constraint_arenas_and_failure_output_preservation(bridge, columns, p, q, aparch):
    config, theta = fixed_constraint_case(0, [], columns, p, q, aparch=aparch)
    config.fixed = 2
    status, values, jacobian = evaluate_constraints(bridge, config, theta)
    assert status == 1
    status, unaligned_values, unaligned_jacobian = evaluate_constraints(
        bridge, config, theta, unaligned=True)
    assert status == 1
    np.testing.assert_array_equal(unaligned_values, values)
    np.testing.assert_array_equal(unaligned_jacobian, jacobian)
    weights = np.linspace(-0.9, 0.7, values.size)
    for kwargs in ({"short": True}, {"null_memory": True}):
        assert evaluate_constraints(bridge, config, theta, **kwargs)[0] == 0
        assert evaluate(bridge, config, theta, weights, **kwargs)[0] == 0
    assert evaluate(bridge, config, theta, None)[0] == 0
    weights[0] = np.nan
    assert evaluate(bridge, config, theta, weights)[0] == 0
    theta[0] = np.inf
    assert evaluate(bridge, config, theta, np.zeros(values.size))[0] == 0
    if aparch:
        theta = power_theta([], columns=columns, p=p, q=q, delta=310.0)
        weights = np.zeros(values.size)
        weights[config.first_order * 2 + config.second_order * 2] = 1.0
        assert evaluate(bridge, config, theta, weights)[0] == 0


def test_arbitrary_fixed_tuple_preserves_typed_layout_contract(bridge):
    assert bridge.curvature_test_arbitrary_fixed_layout() == 1


def persistence_ad(theta, name, columns, p, q, weight):
    jax, jnp, jsp = _jax_modules()
    alpha = columns + 1
    gamma = alpha + p
    delta = gamma + p + q
    shapes = delta + 1
    result = np.zeros((len(theta), len(theta)))
    # Differentiate the complete weighted alpha*moment expression, including alpha=0.
    for lag in range(p):
        indices = [alpha + lag, gamma + lag, delta, *range(shapes, len(theta))]
        local = jnp.asarray(theta[indices])

        def scalar(point):
            return -weight * point[0] * _jax_aparch_power_moment(
                jnp, jsp, point[1], point[2], point[3:], name
            )

        hessian = np.asarray(jax.hessian(scalar)(local))
        assert np.isfinite(hessian).all()
        result[np.ix_(indices, indices)] += hessian
    jax.clear_caches()
    return result


@pytest.mark.parametrize("name,code,shape", DENSITIES)
def test_weighted_persistence_matches_independent_ad(bridge, name, code, shape):
    config = Config.make(code, p=3, q=2, columns=2)
    theta = power_theta(shape, columns=2, p=3, q=2)
    theta[3] = 0.0
    theta[4] = -0.05  # Curvature remains defined on finite, infeasible SQP trials.
    weights = [-1.3] + ([0.8] if name in {"studentt", "skewt"} else [])
    status, actual = evaluate(bridge, config, theta, weights)
    assert status == 1
    expected = persistence_ad(theta, name, 2, 3, 2, weights[0])
    np.testing.assert_allclose(actual, expected, rtol=2e-7, atol=1e-8)
    np.testing.assert_array_equal(actual, actual.T)
    assert np.any(actual[3, 6:])  # The zero-alpha mixed block must remain live.


def polynomial_scalar(jnp, point, order, parameters, offset, basis, scale, weights):
    if not order:
        return jnp.asarray(0.0)
    local = point[offset:offset + parameters]
    coefficients = scale * local if basis is None else jnp.asarray(basis) @ local
    result = jnp.asarray(0.0)
    for stage in range(order, 0, -1):
        reflection = coefficients[-1]
        result += weights[2 * (stage - 1)] * (1.0 - reflection)
        result += weights[2 * (stage - 1) + 1] * (1.0 + reflection)
        if stage > 1:
            coefficients = (coefficients[:-1] + reflection * coefficients[-2::-1]) / (
                1.0 - reflection**2
            )
    return result


@pytest.mark.parametrize("aparch,transformed", [(True, False), (True, True), (False, False)])
def test_composed_polynomials_match_scalar_ad(bridge, aparch, transformed):
    jax, jnp, _ = _jax_modules()
    basis = np.array([[1.0, 0.0], [0.3, 0.2], [0.0, 0.6], [-0.1, 0.4]])
    first = (4, 2, 0, basis, 1.0)
    second = (3, 3, 2, None, -1.0)
    config = Config.make(aparch=aparch, transformed=transformed, columns=5, p=3, q=4,
                         first=first, second=second)
    theta = (power_theta([], columns=5, p=3, q=4) if aparch else
             np.r_[np.zeros(5), 0.08, [0.1] * 3, [-0.04] * 3, [0.1, 0.02, -0.03, 0.04]])
    theta[:5] = [0.12, -0.08, 0.1, -0.06, 0.03]
    weights = np.linspace(-0.9, 0.7, 14)
    if not transformed:
        weights = np.r_[weights, 0.6 if aparch else np.linspace(-0.3, 0.2, 8)]

    def scalar(point):
        result = polynomial_scalar(jnp, point, *first, weights[:8])
        result += polynomial_scalar(jnp, point, *second, weights[8:14])
        if not aparch and not transformed:
            result += polynomial_scalar(jnp, point, 4, 4, 12, None, 1.0, weights[14:])
        return result

    expected = np.array(jax.hessian(scalar)(jnp.asarray(theta)), copy=True)
    polynomial_weights = weights.copy()
    if aparch and not transformed:
        polynomial_weights[14] = 0.0
    status, actual = evaluate(bridge, config, theta, polynomial_weights)
    assert status == 1
    np.testing.assert_allclose(actual, expected, rtol=2e-12, atol=2e-13)
    if aparch and not transformed:
        power_weights = np.zeros_like(weights)
        power_weights[14] = weights[14]
        status, power = evaluate(bridge, config, theta, power_weights)
        assert status == 1
        reference = persistence_ad(theta, "normal", 5, 3, 4, weights[14])
        np.testing.assert_allclose(power, reference, rtol=2e-7, atol=1e-8)
        status, combined = evaluate(bridge, config, theta, weights)
        assert status == 1
        np.testing.assert_array_equal(combined, actual + power)
    jax.clear_caches()


@pytest.mark.parametrize("name,code,shape", DENSITIES)
def test_inactive_rows_skip_unavailable_moments(bridge, name, code, shape):
    config = Config.make(code, p=4, q=3)
    theta = power_theta(shape, p=4, q=3, delta=1000.0)
    weights = [0.0] + ([1.0] if name in {"studentt", "skewt"} else [])
    status, actual = evaluate(bridge, config, theta, weights)
    assert status == 1
    np.testing.assert_array_equal(actual, 0.0)


@pytest.mark.parametrize("scale", [0.0, -3.0, 1e-200, 1e200])
def test_weight_linearity_and_coordinate_scaling(bridge, scale):
    config = Config.make(1, p=4, q=3)
    theta = power_theta([8.0], p=4, q=3)
    assert evaluate(bridge, config, theta, [1.0, 0.0])[0] == 1
    baseline = evaluate(bridge, config, theta, [1.0, 0.0])[1]
    status, actual = evaluate(bridge, config, theta, [scale, 4.0])
    assert status == 1
    np.testing.assert_allclose(actual, scale * baseline, rtol=2e-15, atol=0.0)
    units = np.geomspace(0.1, 10.0, len(theta))
    pulled_back = units[:, None] * actual * units[None, :]
    np.testing.assert_allclose(pulled_back, scale * units[:, None] * baseline * units[None, :],
                               rtol=3e-15, atol=0.0)


@pytest.mark.parametrize("alpha,weight", [(0.0, 1e-100), (1e-200, 1e-200), (0.04, -1e-100)])
def test_weighted_range_recovery_matches_stable_ad(bridge, alpha, weight):
    jax, jnp, jsp = _jax_modules()
    config = Config.make()
    theta = power_theta([], delta=310.0)
    theta[1:3] = [alpha, 0.15]

    def scalar(point):
        a, gamma, delta = point
        log_moment = (0.5 * delta * jnp.log(2.0)
                      + jsp.special.gammaln(0.5 * (delta + 1.0))
                      - 0.5 * jnp.log(jnp.pi)
                      + jsp.special.logsumexp(jnp.stack([
                          delta * jnp.log1p(-gamma), delta * jnp.log1p(gamma)])) - jnp.log(2.0))
        return -np.sign(weight) * a * jnp.exp(np.log(abs(weight)) + log_moment)

    expected = np.asarray(jax.hessian(scalar)(jnp.asarray(theta[[1, 2, 4]])))
    assert np.isfinite(expected).all()
    status, actual = evaluate(bridge, config, theta, [weight])
    assert status == 1
    np.testing.assert_allclose(actual[np.ix_([1, 2, 4], [1, 2, 4])], expected,
                               rtol=2e-12, atol=0.0)
    if alpha == 0.0:
        assert actual[1, 4] != 0.0
        assert actual[4, 4] == 0.0
    jax.clear_caches()


def test_arena_overflow_and_failure_output_contracts(bridge):
    assert bridge.curvature_test_overflow_contracts() == 1
    config = Config.make()
    theta = power_theta([])
    for kwargs in ({"short": True}, {"null_memory": True}):
        assert evaluate(bridge, config, theta, [1.0], **kwargs)[0] == 0
    assert evaluate(bridge, config, theta, None)[0] == 0
    assert evaluate(bridge, config, theta, [np.nan])[0] == 0
    theta[0] = np.inf
    assert evaluate(bridge, config, theta, [0.0])[0] == 0
    theta = power_theta([], delta=310.0)
    assert evaluate(bridge, config, theta, [1.0])[0] == 0


def test_workspace_query_covers_nested_unaligned_byte_arenas(bridge):
    config = Config.make(columns=3, first=(3, 3, 0, None, 1.0))
    theta = power_theta([], columns=3)
    theta[:3] = [0.1, -0.05, 0.03]
    weights = np.linspace(-0.3, 0.5, 7)
    status, aligned = evaluate(bridge, config, theta, weights)
    assert status == 1
    status, unaligned = evaluate(bridge, config, theta, weights, unaligned=True)
    assert status == 1
    np.testing.assert_array_equal(unaligned, aligned)


def test_moment_preparation_is_request_minimal_and_shared_across_lags(bridge):
    assert bridge.curvature_test_preparation_contracts() == 1


def test_zero_alpha_does_not_require_unrepresentable_second_moment_derivatives(bridge):
    jax, jnp, jsp = _jax_modules()
    assert bridge.curvature_test_first_order_only_range() == 1
    config = Config.make()
    theta = power_theta([], delta=298.0)
    theta[1:3] = [0.0, 0.0]

    def moment(point):
        gamma, delta = point
        return 0.5 * ((1.0 - gamma)**delta + (1.0 + gamma)**delta) * jnp.exp(
            0.5 * delta * jnp.log(2.0) + jsp.special.gammaln(0.5 * (delta + 1.0))
            - 0.5 * jnp.log(jnp.pi))

    gradient = np.asarray(jax.grad(moment)(jnp.asarray(theta[[2, 4]])))
    assert np.isfinite(gradient).all()
    status, actual = evaluate(bridge, config, theta, [1.0])
    assert status == 1
    np.testing.assert_allclose(actual[1, [2, 4]], -gradient, rtol=2e-12, atol=0.0)
    np.testing.assert_array_equal(actual[np.ix_([2, 4], [2, 4])], 0.0)
    jax.clear_caches()


@pytest.mark.parametrize("name,code,shape", [DENSITIES[i] for i in (0, 2, 3)])
def test_sublinear_power_curvature_matches_ad(bridge, name, code, shape):
    config = Config.make(code, p=2, q=3)
    theta = power_theta(shape, p=2, q=3, delta=0.7)
    theta[1] = 0.0
    weights = [0.9] + ([0.0] if name == "skewt" else [])
    status, actual = evaluate(bridge, config, theta, weights)
    assert status == 1
    reference = persistence_ad(theta, name, 0, 2, 3, weights[0])
    np.testing.assert_allclose(actual, reference, rtol=2e-7, atol=1e-8)


def test_multiplier_product_does_not_underflow_before_large_derivatives(bridge):
    jax, jnp, jsp = _jax_modules()
    config = Config.make()
    theta = power_theta([], delta=250.0)
    theta[1:3] = [1e-200, 0.0]
    weight = 1e-200

    # Differentiate the nonzero-alpha local block in log alpha coordinates,
    # then pull it back independently. The native adapter stays in theta.
    def scalar(point):
        log_alpha, gamma, delta = point
        log_moment = (0.5 * delta * jnp.log(2.0)
                      + jsp.special.gammaln(0.5 * (delta + 1.0))
                      - 0.5 * jnp.log(jnp.pi)
                      + jsp.special.logsumexp(jnp.stack([
                          delta * jnp.log1p(-gamma), delta * jnp.log1p(gamma)])) - jnp.log(2.0))
        return -jnp.exp(np.log(weight) + log_alpha + log_moment)

    local = jnp.asarray([np.log(theta[1]), theta[2], theta[4]])
    expected = np.asarray(jax.hessian(scalar)(local))[1:, 1:]
    status, actual = evaluate(bridge, config, theta, [weight])
    assert status == 1
    assert np.any(expected != 0.0)
    np.testing.assert_allclose(actual[np.ix_([2, 4], [2, 4])], expected, rtol=2e-12, atol=0.0)
    jax.clear_caches()


def test_empty_transformed_constraints_and_inactive_polynomial(bridge):
    config = Config.make(transformed=True)
    theta = power_theta([], delta=1000.0)
    status, actual = evaluate(bridge, config, theta, None)
    assert status == 1
    np.testing.assert_array_equal(actual, 0.0)
    config = Config.make(columns=2, first=(2, 2, 0, None, 1.0))
    theta = power_theta([], columns=2)
    theta[:2] = [1.0, 1.0]  # Singular inverse-Schur point, on an inactive block.
    assert evaluate(bridge, config, theta, [0.0] * 5)[0] == 1
    assert evaluate(bridge, config, theta, [1.0] * 5)[0] == 0


def common_preparations(bridge):
    result = np.zeros(3, dtype=np.uintp)
    bridge.curvature_test_common_preparations(result.ctypes.data_as(SIZE_PTR))
    return result.tolist()


def assert_request_minimal(counts, request):
    if not request & 4:
        assert counts[2] == 0, counts
    if request == 1:
        assert counts[1] == 0, counts


def assert_strict_elementary_ad(bridge, actual, target, label):
    actual, target = np.asarray(actual), np.asarray(target)
    assert actual.shape == target.shape
    assert np.isfinite(actual).all() and np.isfinite(target).all()
    error = np.abs(actual - target)
    ratio = error / (2e-11 + 3e-10 * np.abs(target))
    worst = np.unravel_index(np.argmax(ratio), ratio.shape)
    record = {"label": label, "max_absolute_error": float(np.max(error)),
              "max_tolerance_ratio": float(ratio[worst]), "worst_index": list(map(int, worst)),
              "actual_at_worst": float(actual[worst]), "target_at_worst": float(target[worst]),
              "atol": 2e-11, "rtol": 3e-10}
    with (bridge._build_directory / "strict_ad.jsonl").open("a") as output:
        output.write(json.dumps(record) + "\n")
    assert record["max_tolerance_ratio"] <= 1.0, record


def joint_coordinates(config, shape):
    mean = np.linspace(0.025, 0.1, config.mean) if config.columns else [0.04, 0.12, -0.06, 0.08]
    power = np.log(1.7 / (shape[0] - 1.7)) if config.density == 1 else np.log(np.expm1(1.7))
    shape_z = [] if not shape else [np.log(np.expm1(shape[0] - (2.0 if config.density == 1 else 0.0)))]
    return np.ascontiguousarray(np.r_[mean, -1.3, np.linspace(-2.1, -1.6, 3),
                                     np.linspace(-0.35, 0.45, 3), [-0.6, 0.1], power, shape_z])


def evaluate_joint(bridge, config, point, request=7, *, data=None, features=None,
                   transformed=False, weights=None, short=False):
    point = np.ascontiguousarray(point, dtype=np.float64)
    map_only = data is None
    k = point.size
    first = np.full(k if map_only else 1, 123.0)
    second = np.full((k, k) if map_only else k, 456.0)
    third = np.full((k, k), 789.0)
    size = bridge.curvature_test_joint_workspace(ctypes.byref(config), transformed, request, map_only)
    assert size
    storage = np.full(size + 64, 0xA5, dtype=np.uint8)
    offset = 16 + (-storage.ctypes.data % 16)
    bridge.curvature_test_reset_common_preparations()
    if map_only:
        weights = np.ascontiguousarray(np.linspace(-0.3, 0.5, k) if weights is None else weights)
        status = bridge.curvature_test_joint_map(
            ctypes.byref(config), request, point.ctypes.data_as(DOUBLE_PTR),
            weights.ctypes.data_as(DOUBLE_PTR), storage.ctypes.data + offset, size - short,
            first.ctypes.data_as(DOUBLE_PTR), second.ctypes.data_as(DOUBLE_PTR),
            third.ctypes.data_as(DOUBLE_PTR),
        )
        assert status != -1, "Map allocated heap storage"
        assert status == (0 if short else 1)
    else:
        data = np.ascontiguousarray(data, dtype=np.float64)
        features = None if features is None else np.ascontiguousarray(features, dtype=np.float64)
        status = bridge.curvature_test_joint_jet(
            ctypes.byref(config), transformed, request, point.ctypes.data_as(DOUBLE_PTR),
            data.ctypes.data_as(DOUBLE_PTR),
            None if features is None else features.ctypes.data_as(DOUBLE_PTR),
            data.size, 1.0, storage.ctypes.data + offset, size - short,
            first.ctypes.data_as(DOUBLE_PTR), second.ctypes.data_as(DOUBLE_PTR),
            third.ctypes.data_as(DOUBLE_PTR),
        )
        assert status not in (-998, -999), "Jet allocated or returned an incomplete request"
        assert (status != 0) if short else (status == 0)
    assert (storage[:offset] == 0xA5).all()
    assert (storage[offset + size:] == 0xA5).all()
    for bit, output, sentinel in zip((1, 2, 4), (first, second, third), (123.0, 456.0, 789.0)):
        if not request & bit or short:
            np.testing.assert_array_equal(output, sentinel)
        else:
            assert np.isfinite(output).all()
    counts = common_preparations(bridge)
    assert_request_minimal(counts, request)
    return (first, second, third), counts


@pytest.mark.parametrize("name,code,shape", ELEMENTARY_DENSITIES)
@pytest.mark.parametrize("columns", [0, 4, 7], ids=["ARMA2132", "C4-32", "C7-32"])
def test_elementary_joint_consumers_share_preparation_and_sparse_jets(bridge, name, code, shape, columns):
    config = JointConfig(code, columns, 0)
    z = joint_coordinates(config, shape)
    rng = np.random.default_rng(421009)
    data = rng.normal(size=10)
    features = None if not columns else np.ascontiguousarray(
        np.column_stack([np.ones(data.size), rng.normal(size=(data.size, columns - 1))]))
    mapped, _ = evaluate_joint(bridge, config, z)
    theta = mapped[0]
    # One independent full oracle per coordinate point, never per sparse mask.
    theta_ad = (joint_value_grad_hess(theta, data, features, 1.0, 3, 2, 3, code)
                if columns else arma_aparch_value_grad_hess(
                    theta, data, 2, 1, 3, 2, name, initial_variance=1.0))
    z_ad = (joint_value_grad_hess(theta, data, features, 1.0, 3, 2, 3, code, coordinates=z)
            if columns else native_aparch_coordinate_chain_rule_value_grad_hess(
                z, theta, *theta_ad, 3, 2, name, p_ar=2, q_ma=1))
    runtime_jets = {}
    for fixed in (0, 1):
        config.fixed = fixed
        for request in (7, 1, 2, 3, 4, 5, 6):
            actual, counts = evaluate_joint(bridge, config, z, request)
            assert counts == ([1, 0, 1] if request & 4 else [1, 1, 0] if request & 2 else [1, 0, 0])
            for bit, output, full in zip((1, 2, 4), actual, mapped):
                if request & bit:
                    np.testing.assert_allclose(output, full, rtol=2e-12, atol=2e-13)
        for transformed, point, expected in ((False, theta, theta_ad), (True, z, z_ad)):
            whole = None
            for request in (7, 1, 2, 3, 4, 5, 6):
                actual, _ = evaluate_joint(bridge, config, point, request, data=data,
                                           features=features, transformed=transformed)
                if whole is None:
                    whole = actual
                    if fixed:
                        for output, reference in zip(actual, runtime_jets[transformed]):
                            np.testing.assert_allclose(output, reference, rtol=2e-12, atol=2e-13)
                    else:
                        runtime_jets[transformed] = actual
                    assert np.max(np.abs(whole[2][:config.mean, config.mean:])) > 1e-4
                    assert abs(actual[0][0] - expected[0]) < 1e-9
                    for name_bit, output, control in zip(("G", "H"), actual[1:], expected[1:]):
                        assert_strict_elementary_ad(
                            bridge, output, control, f"{name}/C{columns}/fixed{fixed}/z{int(transformed)}/{name_bit}")
                for bit, output, full in zip((1, 2, 4), actual, whole):
                    if request & bit:
                        np.testing.assert_allclose(output, full, rtol=2e-12, atol=2e-13)
            evaluate_joint(bridge, config, point, 1, data=data, features=features,
                           transformed=transformed, short=True)
    _jax_modules()[0].clear_caches()


@pytest.mark.parametrize("name,code,shape", ELEMENTARY_DENSITIES)
@pytest.mark.parametrize("columns", [4, 7], ids=["C4-32", "C7-32"])
def test_elementary_complete_weighted_curvature_and_selected_columns(bridge, name, code, shape, columns):
    config, theta = fixed_constraint_case(code, shape, columns, 3, 2)
    status, values, jacobian = evaluate_constraints(bridge, config, theta)
    assert status == 1
    weights = np.linspace(-0.9, 0.7, values.size)
    persistence_row = 2 * (config.first_order + config.second_order)
    weights[persistence_row] = -1.3
    jax, jnp, jsp = _jax_modules()
    alpha, gamma = columns + 1, columns + 4
    delta = columns + 9

    def complete(point):
        result = polynomial_scalar(jnp, point, config.first_order, config.first_parameters,
                                   config.first_offset, None, config.first_scale, weights[:2 * config.first_order])
        result += polynomial_scalar(jnp, point, config.second_order, config.second_parameters,
                                    config.second_offset, None, config.second_scale,
                                    weights[2 * config.first_order:persistence_row])
        moments = jnp.stack([_jax_aparch_power_moment(jnp, jsp, point[gamma + i],
                                                    point[delta], point[delta + 1:], name) for i in range(3)])
        result += weights[persistence_row] * (1.0 - point[alpha:gamma] @ moments
                                               - jnp.sum(point[gamma + 3:delta]))
        if code == 1:
            result += weights[-1] * (point[-1] - point[delta])
        return result

    expected = np.asarray(jax.hessian(complete)(jnp.asarray(theta)))
    for mode in (0, 1, 2):
        config.fixed = mode
        bridge.curvature_test_reset_common_preparations()
        status, actual = evaluate(bridge, config, theta, weights)
        assert status == 1 and common_preparations(bridge) == [0, 1, 1]
        assert_strict_elementary_ad(bridge, actual, expected, f"{name}/C{columns}/fixed{mode}/weighted-H")
        np.testing.assert_array_equal(actual, actual.T)
        assert actual[alpha, delta] != 0.0
        assert actual[alpha, gamma] != 0.0
        if shape:
            assert actual[alpha, delta + 1] != 0.0
        np.testing.assert_array_equal(actual[gamma, gamma:], 0.0)
        for request, selection, counts in (
            (0, None, [1, 0, 0]), (2, None, [2, 1, 0]),
            (1, [alpha, gamma, delta, theta.size - 1], [1, 1, 0]),
            (1, [gamma + 3], [0, 0, 0]), (1, [], [0, 0, 0]),
        ):
            bridge.curvature_test_reset_common_preparations()
            status, selected_values, selected_jacobian = evaluate_constraints(
                bridge, config, theta, request=request, columns=selection)
            assert status == 1 and common_preparations(bridge) == counts
            if request != 1:
                np.testing.assert_allclose(selected_values, values, rtol=3e-13, atol=3e-14)
            if request != 0:
                selected = list(range(theta.size)) if selection is None else selection
                np.testing.assert_allclose(selected_jacobian[:, selected], jacobian[:, selected],
                                           rtol=3e-13, atol=3e-14)
        zero_weights = np.zeros_like(weights)
        bridge.curvature_test_reset_common_preparations()
        status, inactive = evaluate(bridge, config, theta, zero_weights)
        assert status == 1 and common_preparations(bridge) == [0, 0, 0]
        np.testing.assert_array_equal(inactive, 0.0)
    jax.clear_caches()


def normalized_student_alpha_ad(config, z, theta, lag):
    # Anchor the scalar at rounded theta, retaining the original-z gap derivatives.
    jax, jnp, jsp = _jax_modules()
    d, n = config.delta, config.delta + 1
    q0 = float(1.0 / (1.0 + np.exp(z[d])))
    keys = np.r_[np.arange(config.alpha, config.gamma), np.arange(config.beta, config.delta)]
    logits = z[keys]
    maximum = max(0.0, max(logits))
    raw = np.exp(logits - maximum)
    probability = raw / (np.exp(-maximum) + sum(raw))

    def normalized(displacement):
        g_index = config.gamma + lag
        g = theta[g_index] + jnp.tanh(z[g_index] + displacement[g_index]) - jnp.tanh(z[g_index])
        dn = jax.nn.softplus(z[n] + displacement[n]) - jax.nn.softplus(z[n])
        nu = theta[n] + dn
        log_ratio = jax.nn.softplus(z[d]) - jax.nn.softplus(z[d] + displacement[d])
        gap = theta[n] - theta[d] + q0 * dn + nu * q0 * jnp.expm1(log_ratio)
        power = nu - gap

        def log_moment(g, power, nu, gap):
            return (jnp.logaddexp(power * jnp.log1p(-g), power * jnp.log1p(g)) - jnp.log(2.0)
                    + power / 2 * jnp.log(nu - 2.0) + jsp.special.gammaln((power + 1.0) / 2)
                    + jsp.special.gammaln(1.0 + gap / 2) - jnp.log(gap) + jnp.log(2.0)
                    - 0.5 * jnp.log(jnp.pi) - jsp.special.gammaln(nu / 2))

        anchor = log_moment(theta[g_index], theta[d], theta[n], theta[n] - theta[d])
        changes = displacement[keys]
        ratio = jnp.exp(changes[lag]) / (1.0 + jnp.sum(probability * jnp.expm1(changes)))
        return ratio * jnp.exp(anchor - log_moment(g, power, nu, gap))

    zero = jnp.zeros(z.size)
    return np.asarray(jax.grad(normalized)(zero)), np.asarray(jax.hessian(normalized)(zero))


@pytest.mark.parametrize("columns", [0, 7], ids=["ARMA2132", "C7-32"])
def test_elementary_reuse_retains_original_z_student_plateau(bridge, columns):
    config = JointConfig(1, columns, 0)
    mapped_points = []
    slopes = []
    for power_z in (36.0, 36.25):
        config.fixed = 0
        z = joint_coordinates(config, [8.0])
        z[config.delta] = power_z
        mapped, _ = evaluate_joint(bridge, config, z)
        theta = mapped[0]
        mapped_points.append(theta)
        slopes.append(mapped[1][config.alpha, config.delta] / theta[config.alpha])
        for lag in range(3):
            expected_g, expected_h = normalized_student_alpha_ad(config, z, theta, lag)
            weights = np.zeros(theta.size)
            weights[config.alpha + lag] = 1.0 / theta[config.alpha + lag]
            for fixed in (0, 1):
                config.fixed = fixed
                actual, _ = evaluate_joint(bridge, config, z, weights=weights)
                np.testing.assert_array_equal(actual[0], theta)
                np.testing.assert_allclose(actual[1][config.alpha + lag] / theta[config.alpha + lag],
                                           expected_g, rtol=3e-11, atol=3e-11)
                np.testing.assert_allclose(actual[2], expected_h, rtol=3e-11, atol=3e-11)
    np.testing.assert_array_equal(mapped_points[0], mapped_points[1])
    assert slopes[0] != slopes[1], "Original-z derivatives must survive identical rounded theta"
    _jax_modules()[0].clear_caches()
