"""Independent truncated-beta MP and density-recurrence AD controls, legacy-free."""

from functools import lru_cache
import math
import os
from pathlib import Path
import subprocess

import mpmath as mp
import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[2]
FAILING_SHAPE = (2.0 + math.log1p(math.exp(0.3)), math.tanh(0.3))


@pytest.fixture(scope="module")
def moment_binary(tmp_path_factory):
    supplied = os.environ.get("SCIVOL_SKEW_MOMENT_TEST")
    if supplied:
        binary = Path(supplied).resolve(strict=True)
    else:
        pressure = Path.home() / "bin" / "mempressure"
        if pressure.exists():
            import json
            state = json.loads(subprocess.check_output([str(pressure)], text=True))
            assert state["state"] != "critical", "Do not compile under critical memory pressure"
        binary = tmp_path_factory.mktemp("skew_moment") / "test_skew_negative_moment"
        subprocess.run([
            os.environ.get("CXX", "clang++"), "-std=c++20", "-O3", "-fno-fast-math",
            "-ffp-contract=fast", "-fno-math-errno", "-Wframe-larger-than=16384",
            "-Werror=frame-larger-than", "-Werror=vla", "-Werror=alloca",
            "-I" + str(ROOT / "core" / "src"),
            str(ROOT / "tests" / "native" / "test_skew_negative_moment.cpp"),
            "-o", str(binary),
        ], check=True, capture_output=True, text=True, timeout=120)
    subprocess.run([str(binary)], check=True, capture_output=True, text=True, timeout=120)
    return binary


def native_orders(binary, point):
    output = subprocess.check_output(
        [str(binary), "--jet", *(repr(float(x)) for x in point)], text=True, timeout=120)
    rows = np.array([[float(x) for x in line.split()] for line in output.splitlines()])
    np.testing.assert_array_equal(rows[:, 0], [0, 1, 2])
    np.testing.assert_array_equal(rows[:, 1], [1, 1, 1])
    return rows[:, 2:]


def beta_negative_second(nu, skew):
    """Integrate the original Student density via DLMF 8.17.1, not native algebra."""
    if skew == 0:
        return mp.mpf("0.5")
    s = abs(skew)
    k = nu - 2
    c = mp.gamma((nu + 1) / 2) / (mp.gamma(nu / 2) * mp.sqrt(mp.pi * k))
    h = c * k / (nu - 1)
    a = 4 * s * h
    b2 = 1 + 3 * s * s - a * a
    left, right = 1 - s, 1 + s
    t = a / right
    x = t * t / (k + t * t)
    partial = [c * k ** (mp.mpf(j + 1) / 2) / 2
               * mp.betainc(mp.mpf(j + 1) / 2, (nu - j) / 2, 0, x)
               for j in range(3)]
    tail = left ** 3 / 2 + 2 * left ** 2 * a * h + left * a * a / 2
    bounded = right ** 3 * partial[2] - 2 * right ** 2 * a * partial[1]
    bounded += right * a * a * partial[0]
    result = (tail + bounded) / b2
    return result if skew > 0 else 1 - result


def mp_shape_jet(point):
    with mp.workdps(110):
        x = tuple(map(mp.mpf, point))
        g = [mp.diff(beta_negative_second, x, (1, 0)),
             mp.diff(beta_negative_second, x, (0, 1))]
        h = [mp.diff(beta_negative_second, x, (2, 0)),
             mp.diff(beta_negative_second, x, (1, 1)),
             mp.diff(beta_negative_second, x, (0, 2))]
        return np.array([beta_negative_second(*x), *g, *h], dtype=float)


MP_POINTS = [FAILING_SHAPE, (FAILING_SHAPE[0], -FAILING_SHAPE[1]),
    (2.000001, 0.99), (2.000001, -0.99), (2.01, 0.7), (3.0, -0.8),
    (7.0, 0.3), (7.0, -0.3), (7.0, 0.0), (7.0, 1e-20), (7.0, -1e-20),
    (np.nextafter(2.0, 3.0), 0.0), (np.nextafter(2.0, 3.0), 0.3),
    (np.nextafter(2.0, 3.0), np.nextafter(1.0, 0.0)),
    (np.nextafter(2.0, 3.0), -np.nextafter(1.0, 0.0)),
    (127.99, 0.4), (128.0, 0.4), (128.01, -0.4),
    (1e6, 0.8), (1e12, 0.3)]


@pytest.mark.parametrize("point", MP_POINTS)
def test_all_derivative_orders_against_incomplete_beta_mp(moment_binary, point):
    actual = native_orders(moment_binary, point)
    expected = mp_shape_jet(point)
    for order in range(3):
        width = (1, 3, 6)[order]
        np.testing.assert_allclose(actual[order, :width], expected[:width], rtol=8e-10, atol=3e-12)
        np.testing.assert_array_equal(actual[order, width:], np.zeros(6 - width))
    np.testing.assert_allclose(actual[:, 0], actual[2, 0], rtol=0, atol=3e-14)
    np.testing.assert_allclose(actual[1, :3], actual[2, :3], rtol=2e-13, atol=3e-14)
    if point[0] >= 1e6:
        # Absolute tolerances alone would hide lost Gaussian-limit shape slopes.
        np.testing.assert_allclose(actual[2, [1, 3, 4]], expected[[1, 3, 4]], rtol=8e-8, atol=0)
    if point[0] == np.nextafter(2.0, 3.0) and point[1] > 0.9:
        assert actual[2, 0] > 0
        np.testing.assert_allclose(actual[2, 0], expected[0], rtol=2e-12, atol=0)


@lru_cache(maxsize=2)
def recurrence_ad_functions(count):
    import jax
    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp
    import jax.scipy as jsp

    nodes, weights = np.polynomial.legendre.leggauss(count)
    u = jnp.asarray((nodes + 1) / 2)
    w = jnp.asarray(weights / 2)

    def moment(shape):
        nu, skew = shape
        s, k = jnp.abs(skew), nu - 2
        c = jnp.exp(jsp.special.gammaln((nu + 1) / 2) - jsp.special.gammaln(nu / 2))
        c /= jnp.sqrt(jnp.pi * k)
        a = 4 * s * c * k / (nu - 1)
        b = jnp.sqrt(1 + 3 * s * s - a * a)
        mu, scale = a / b, jnp.sqrt(k) * (1 - s) / b
        radius = (u / (1 - u)) ** 4
        z = scale * radius + mu
        density = b * c * (1 + radius * radius) ** (-(nu + 1) / 2)
        jacobian = scale * 4 * u ** 3 / (1 - u) ** 5
        lower0 = jnp.sum(w * density * jacobian)
        lower1 = jnp.sum(w * z * density * jacobian)
        # Integration by parts lowers the unbounded x^2 integral to x^0 and x^1.
        tail = (mu * (nu - 3) * lower1 + (mu * mu + scale * scale) * lower0
                + b * c * scale * scale * mu) / k
        magnitude = mu * u
        adjusted = (a - b * magnitude) / (1 + s)
        bounded_density = b * c * (1 + adjusted * adjusted / k) ** (-(nu + 1) / 2)
        bounded = jnp.sum(w * magnitude * magnitude * bounded_density * mu)
        negative = tail + bounded
        return jnp.where(skew < 0, 1 - negative, negative)

    return moment, jax.grad(moment), jax.hessian(moment)


def recurrence_ad_jet(point, count):
    import jax.numpy as jnp
    f, g, h = recurrence_ad_functions(count)
    x = jnp.asarray(point, dtype=jnp.float64)
    gradient, hessian = np.asarray(g(x)), np.asarray(h(x))
    return np.array([float(f(x)), *gradient, hessian[0, 0], hessian[0, 1], hessian[1, 1]])


@pytest.mark.parametrize("point", [FAILING_SHAPE, (FAILING_SHAPE[0], -FAILING_SHAPE[1]),
    (2.01, 0.7), (2.1, -0.9), (3.0, 0.9999), (7.0, -0.3)])
def test_independent_original_density_recurrence_ad(moment_binary, point):
    coarse, expected = recurrence_ad_jet(point, 512), recurrence_ad_jet(point, 1024)
    np.testing.assert_allclose(coarse, expected, rtol=2e-8, atol=2e-9)
    np.testing.assert_allclose(expected, mp_shape_jet(point), rtol=2e-8, atol=2e-9)
    np.testing.assert_allclose(native_orders(moment_binary, point)[2], expected, rtol=2e-8, atol=2e-9)


def mp_map(z, p, q):
    count = 2 * p + q
    eps = mp.mpf(float(np.finfo(np.float64).eps))
    limit = mp.mpf(1)
    nu = 2 + mp.log(1 + mp.exp(z[-2]))
    skew = mp.tanh(z[-1])
    moment = beta_negative_second(nu, skew)
    masses = [mp.exp(x) for x in z[1:1 + count]]
    denominator = 1 + sum(masses)
    contributions = [limit * x / denominator for x in masses]
    alpha = [x / (1 - moment) for x in contributions[:p]]
    gamma = [contributions[p + i] / moment - alpha[i] for i in range(p)]
    return [mp.log(1 + mp.exp(z[0])),
            *alpha, *gamma, *contributions[2 * p:], nu, skew]


def mp_map_jet(z, p, q, theta_gradient):
    with mp.workdps(90):
        x = tuple(map(mp.mpf, z))
        width = len(x)
        theta = mp_map(x, p, q)
        jacobian = np.empty((width, width))
        for row in range(width):
            for col in range(width):
                orders = tuple(int(i == col) for i in range(width))
                jacobian[row, col] = mp.diff(lambda *a: mp_map(a, p, q)[row], x, orders)
        weights = tuple(map(mp.mpf, theta_gradient))
        def contraction(*a):
            return sum(w * t for w, t in zip(weights, mp_map(a, p, q), strict=True))
        curvature = np.empty((width, width))
        for row in range(width):
            for col in range(row, width):
                orders = tuple(int(i == row) + int(i == col) for i in range(width))
                curvature[row, col] = curvature[col, row] = mp.diff(contraction, x, orders)
        return np.array(theta, dtype=float), jacobian, curvature


def test_original_failing_map_value_jacobian_and_curvature_mp(moment_binary):
    lines = subprocess.check_output([str(moment_binary), "--map"], text=True, timeout=120).splitlines()
    actual = [np.fromstring(line, sep=" ") for line in lines]
    z = [-1.2, -2.5, -2.5, -2.5, 0.3, 0.3]
    theta, jacobian, curvature = mp_map_jet(z, 1, 1, np.arange(1, 7))
    np.testing.assert_allclose(actual[0], theta, rtol=2e-13, atol=2e-14)
    np.testing.assert_allclose(actual[1].reshape(6, 6), jacobian, rtol=2e-10, atol=2e-12)
    np.testing.assert_allclose(actual[2].reshape(6, 6), curvature, rtol=8e-10, atol=8e-12)


@pytest.mark.parametrize("orders", [(1, 1), (3, 2)])
def test_linked_original_z_sparse_requests(native, orders):
    """Chief runs this after rebuilding both shared GJR consumers."""
    from oracles.garch import full_jet

    p, q = orders
    z = np.array([-1.2, *([-2.5] * (2 * p + q)), 0.3, 0.3])
    case = dict(model="gjr", density="skewt", p=p, q=q,
        y=np.array([-.2, .4, -1.1, .8, -.5, .1, 1.7, -.9]), h0=.7,
        parameters=z, transformed=True, first=0, stop=8, hold_back=0,
        scored_rows=tuple(range(8)))
    full = native(case, 7)
    assert full["status"] == 0, full.get("reasons")
    scalar_case = dict(case, parameters=full["theta"], transformed=False)
    f, g, h = full_jet(scalar_case)
    theta, jacobian, curvature = mp_map_jet(z, p, q, g)
    np.testing.assert_allclose(full["theta"], theta, rtol=3e-13, atol=3e-14)
    expected = (f, jacobian.T @ g, jacobian.T @ h @ jacobian + curvature)
    for mask in range(1, 8):
        actual = full if mask == 7 else native(case, mask)
        assert actual["status"] == 0 and actual["valid_mask"] == mask
        assert actual["dispatch"] == int(orders == (1, 1))
        for bit, name, reference in zip((1, 2, 4), ("f", "g", "h"), expected, strict=True):
            if mask & bit:
                np.testing.assert_allclose(actual[name], reference, rtol=3e-8, atol=3e-8)
                np.testing.assert_allclose(actual[name], full[name], rtol=2e-12, atol=2e-12)
            else:
                np.testing.assert_array_equal(np.asarray(actual[name]).view(np.uint64),
                    np.full_like(np.asarray(actual[name]), actual["sentinel"]).view(np.uint64))
