from __future__ import annotations

import ctypes
from functools import lru_cache

import numpy as np
import pytest

from scivol import (
    APARCH,
    ARMA,
    GED,
    GeneralizedHyperbolic,
    Hyperbolic,
    Normal,
    NormalInverseGaussian,
    SkewT,
    StudentT,
    VarianceGamma,
    _native_core,
)
from scivol._devtools.ad_oracle import (
    aparch_power_moment_value_grad_hess,
    aparch_value_grad_hess,
    arma_aparch_value_grad_hess,
    gh_surface_value_grad_hess,
    native_aparch_coordinate_chain_rule_value_grad_hess,
    vg_surface_value_grad_hess,
)
from scivol.roles import Role


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
VOLATILITY = np.array([0.08, 0.07, 0.15, 0.82, 1.7])
MEAN = np.array([0.03, 0.12, -0.08])


def _workspace(size: int) -> np.ndarray:
    return np.empty(max(size, 1), dtype=np.uint8)


def _theta(shape: list[float], *, joint: bool) -> np.ndarray:
    blocks = [MEAN, VOLATILITY, np.asarray(shape)] if joint else [VOLATILITY, np.asarray(shape)]
    return np.ascontiguousarray(np.concatenate(blocks), dtype=np.float64)


def _evaluate(
    parameters: np.ndarray,
    data: np.ndarray,
    density: int,
    request: int = 7,
    *,
    transformed: bool = False,
    orders: tuple[int, int, int, int] | None = None,
    initial_variance: float | None = None,
) -> tuple[int, int, float, np.ndarray, np.ndarray]:
    p_ar, q_ma, p_vol, q_vol = orders or (0, 0, 1, 1)
    joint = orders is not None
    initial = float(np.mean(data * data)) if initial_variance is None else initial_variance
    value = np.full(1, np.nan)
    gradient = np.full(parameters.size, np.nan)
    hessian = np.full((parameters.size, parameters.size), np.nan)
    if joint:
        size = _native_core._arma_aparch_workspace_bytes(
            p_ar, q_ma, p_vol, q_vol, density, int(transformed), request
        )
        workspace = _workspace(size)
        status, valid = _native_core._arma_aparch_jet(
            parameters.ctypes.data,
            parameters.size,
            data.ctypes.data,
            data.size,
            initial,
            p_ar,
            q_ma,
            p_vol,
            q_vol,
            density,
            int(transformed),
            request,
            workspace.ctypes.data,
            workspace.nbytes,
            value.ctypes.data,
            gradient.ctypes.data,
            hessian.ctypes.data,
        )
    else:
        size = _native_core._aparch_workspace_bytes(
            p_vol, q_vol, density, int(transformed), request
        )
        workspace = _workspace(size)
        status, valid = _native_core._aparch_jet(
            parameters.ctypes.data,
            parameters.size,
            data.ctypes.data,
            data.size,
            initial,
            p_vol,
            q_vol,
            density,
            int(transformed),
            request,
            workspace.ctypes.data,
            workspace.nbytes,
            value.ctypes.data,
            gradient.ctypes.data,
            hessian.ctypes.data,
        )
    return status, valid, float(value[0]), gradient, hessian


@pytest.mark.parametrize("small,gamma", [(1e-170, 0.15), (0.1, np.nextafter(1.0, 0.0))])
def test_fixed_aparch_shock_derivatives_avoid_cancelled_or_squared_small_bases(small, gamma):
    data = np.array([0.7, small, -0.4, 0.2, -0.5])
    theta = np.array([0.2, 0.07, gamma, 0.6, 1.7])
    expected = aparch_value_grad_hess(theta, data, 1, 1, dist="normal")
    for request in range(1, 8):
        status, valid, value, gradient, hessian = _evaluate(theta, data, 0, request)
        assert (status, valid) == (0, request)
        for bit, actual, reference in zip((1, 2, 4), (value, gradient, hessian), expected):
            if request & bit:
                np.testing.assert_allclose(actual, reference, rtol=2e-6, atol=2e-7)


def _map(
    theta: np.ndarray,
    density: int,
    *,
    orders: tuple[int, int, int, int] | None = None,
) -> np.ndarray:
    destination = np.empty_like(theta)
    if orders is None:
        workspace = _workspace(
            _native_core._aparch_map_workspace_bytes(1, 1, density)
        )
        _native_core._map_aparch(
            theta.ctypes.data,
            destination.ctypes.data,
            1,
            1,
            density,
            False,
            workspace.ctypes.data,
            workspace.nbytes,
        )
    else:
        p_ar, q_ma, p_vol, q_vol = orders
        workspace = _workspace(
            _native_core._arma_aparch_map_workspace_bytes(
                p_ar, q_ma, p_vol, q_vol, density
            )
        )
        _native_core._map_arma_aparch(
            theta.ctypes.data,
            destination.ctypes.data,
            p_ar,
            q_ma,
            p_vol,
            q_vol,
            density,
            False,
            workspace.ctypes.data,
            workspace.nbytes,
        )
    return destination


@pytest.mark.parametrize(("name", "code", "shape"), DENSITIES)
@pytest.mark.parametrize("delta", [0.7, 1.7])
def test_aparch_stationarity_moment_matches_ad(
    name: str, code: int, shape: list[float], delta: float,
) -> None:
    shape_array = np.asarray(shape, dtype=np.float64)
    local = np.r_[0.15, delta, shape_array]
    value = np.empty(1)
    gradient = np.empty(local.size)
    hessian = np.empty((local.size, local.size))
    status = _native_core._aparch_asymmetric_power_moment(
        shape_array.ctypes.data,
        code,
        local[0],
        local[1],
        7,
        value.ctypes.data,
        gradient.ctypes.data,
        hessian.ctypes.data,
    )
    expected = aparch_power_moment_value_grad_hess(local, name)
    assert status == 0
    np.testing.assert_allclose(value[0], expected[0], rtol=2e-9, atol=2e-11)
    np.testing.assert_allclose(gradient, expected[1], rtol=2e-8, atol=2e-10)
    np.testing.assert_allclose(hessian, expected[2], rtol=2e-7, atol=1e-8)


@pytest.mark.parametrize("gamma", [0.0, 0.25])
@pytest.mark.parametrize("delta", [0.7, 1.7, 2.0, 3.5])
def test_zero_skew_power_moment_oracle_has_finite_exact_reduction(gamma, delta):
    local = np.array([gamma, delta, 8.0, 0.0])
    reference = aparch_power_moment_value_grad_hess(local, "skewt")
    student = aparch_power_moment_value_grad_hess(local[:3], "studentt")
    for value in reference:
        assert np.isfinite(value).all()
    np.testing.assert_allclose(reference[0], student[0], rtol=2e-9, atol=2e-11)
    np.testing.assert_allclose(reference[1][:3], student[1], rtol=2e-8, atol=2e-10)
    np.testing.assert_allclose(reference[2][:3, :3], student[2], rtol=2e-7, atol=1e-8)


@pytest.mark.parametrize("delta", [0.7, 1.7, 2.0, 3.5])
def test_zero_skew_power_moment_native_matches_full_oracle(delta):
    gamma = 0.25
    local = np.array([gamma, delta, 8.0, 0.0])
    reference = aparch_power_moment_value_grad_hess(local, "skewt")
    value, gradient, hessian = np.empty(1), np.empty(4), np.empty((4, 4))
    shape = np.ascontiguousarray(local[2:])
    assert _native_core._aparch_asymmetric_power_moment(
        shape.ctypes.data, 2, gamma, delta, 7, value.ctypes.data,
        gradient.ctypes.data, hessian.ctypes.data,
    ) == 0
    for actual, expected in zip((value[0], gradient, hessian), reference):
        np.testing.assert_allclose(actual, expected, rtol=2e-7, atol=1e-8)


@pytest.mark.parametrize("power", [0.7, 1.7])
def test_zero_skew_vg_moment_derivatives_match_independent_density_integrals(power):
    mp = pytest.importorskip("mpmath")
    local = np.array([0.25, power, 0.0, 3.0])
    value, gradient, hessian = aparch_power_moment_value_grad_hess(local, "vg")
    with mp.workdps(40):
        gamma, delta = mp.mpf(0.25), mp.mpf(power)

        def density(z, rho):
            # At lambda=3, K_(5/2) gives an exact polynomial-times-exponential PDF.
            skew = mp.sqrt(2) * rho / mp.sqrt(1 - rho * rho)
            scale = mp.sqrt(3 * (1 + skew * skew))
            y = scale * z + 3 * skew
            a = mp.sqrt(2 + skew * skew)
            return (
                scale
                / 2
                * mp.exp(skew * y - a * abs(y))
                * (y * y / a**3 + 3 * abs(y) / a**4 + 3 / a**5)
            )

        def moment_derivative(order):
            def integrand(x):
                if x == 0:
                    return mp.mpf(0)
                return x**delta * (
                    (1 - gamma) ** delta
                    * mp.diff(lambda rho: density(x, rho), 0, order)
                    + (1 + gamma) ** delta
                    * mp.diff(lambda rho: density(-x, rho), 0, order)
                )

            return float(mp.quad(integrand, [0, 1, 4, 12, mp.inf]))

        reference = [moment_derivative(order) for order in (0, 1, 2)]
    np.testing.assert_allclose(
        [value, gradient[2], hessian[2, 2]],
        reference,
        rtol=2e-9,
        atol=2e-11,
    )


@pytest.mark.parametrize("rho", [-0.12, 0.0, 0.12])
def test_vg_power_moment_split_oracle_converges(rho):
    from scivol._devtools.ad_oracle import _jax_modules, _jax_vg_power_moment

    jax,jnp,jsp = _jax_modules()
    local = np.array([0.1,1.4,rho,2.0])
    references=[]
    for count in (64,128):
        def objective(theta):
            return _jax_vg_power_moment(jnp,jsp,*theta,nodes_count=count)
        point=jnp.asarray(local)
        references.append((float(objective(point)),np.asarray(jax.grad(objective)(point)),
            np.asarray(jax.hessian(objective)(point))))
    for coarse,fine in zip(*references):
        np.testing.assert_allclose(coarse,fine,rtol=2e-6,atol=2e-8)
    value,gradient,hessian=np.empty(1),np.empty(4),np.empty((4,4))
    shape=np.ascontiguousarray(local[2:])
    assert _native_core._aparch_asymmetric_power_moment(
        shape.ctypes.data,7,local[0],local[1],7,value.ctypes.data,
        gradient.ctypes.data,hessian.ctypes.data)==0
    for actual,expected in zip((value[0],gradient,hessian),references[-1]):
        np.testing.assert_allclose(actual,expected,rtol=2e-6,atol=2e-8)


@pytest.mark.parametrize(("name", "code", "shape"), DENSITIES)
def test_standalone_aparch_theta_jet_matches_ad(
    name: str, code: int, shape: list[float]
) -> None:
    data = np.random.default_rng(714).normal(size=24).astype(np.float64)
    theta = _theta(shape, joint=False)
    status, valid, value, gradient, hessian = _evaluate(theta, data, code)
    if name in {"gh", "nig", "hyperbolic", "vg"}:
        expected = (
            vg_surface_value_grad_hess(
                theta, data, volatility_kind="aparch", p_vol=1, q_vol=1
            )
            if name == "vg"
            else gh_surface_value_grad_hess(
                theta,
                data,
                volatility_kind="aparch",
                p_vol=1,
                q_vol=1,
                fixed_lambda={"nig": -0.5, "hyperbolic": 1.0}.get(name),
            )
        )
    else:
        expected = aparch_value_grad_hess(theta, data, 1, 1, name)
    assert (status, valid) == (0, 7)
    np.testing.assert_allclose(value, expected[0], rtol=3e-9, atol=3e-9)
    np.testing.assert_allclose(gradient, expected[1], rtol=4e-7, atol=4e-9)
    np.testing.assert_allclose(hessian, expected[2], rtol=4e-6, atol=4e-8)


@pytest.mark.parametrize(("name", "code", "shape"), DENSITIES)
def test_joint_arma_aparch_theta_jet_matches_ad(
    name: str, code: int, shape: list[float]
) -> None:
    data = np.random.default_rng(2718).normal(size=28).astype(np.float64)
    theta = _theta(shape, joint=True)
    actual = _evaluate(theta, data, code, orders=(1, 1, 1, 1))
    expected = arma_aparch_value_grad_hess(theta, data, 1, 1, 1, 1, name)
    assert actual[:2] == (0, 7)
    np.testing.assert_allclose(actual[2], expected[0], rtol=3e-9, atol=3e-9)
    np.testing.assert_allclose(actual[3], expected[1], rtol=4e-7, atol=4e-9)
    np.testing.assert_allclose(actual[4], expected[2], rtol=4e-6, atol=4e-8)
    assert np.max(np.abs(actual[4][:3, 3:])) > 1e-6


@pytest.mark.parametrize(("name", "code", "shape"), DENSITIES)
def test_aparch_transformed_jet_matches_independent_chart_ad(
    name: str, code: int, shape: list[float]
) -> None:
    data = np.random.default_rng(99).normal(size=14).astype(np.float64)
    theta = _theta(shape, joint=False)
    coordinates = _map(theta, code)
    base = _evaluate(theta, data, code)
    actual = _evaluate(coordinates, data, code, transformed=True)
    expected = native_aparch_coordinate_chain_rule_value_grad_hess(
        coordinates, theta, base[2], base[3], base[4], 1, 1, name
    )
    assert actual[:2] == (0, 7)
    np.testing.assert_allclose(actual[2], expected[0], rtol=3e-9, atol=3e-9)
    np.testing.assert_allclose(actual[3], expected[1], rtol=4e-7, atol=4e-9)
    np.testing.assert_allclose(actual[4], expected[2], rtol=4e-6, atol=4e-8)


@pytest.mark.parametrize(
    ("name", "code", "shape"),
    [("normal", 0, []), ("studentt", 1, [8.0]), ("gh", 4, [0.1, 1.3, 0.4])],
)
def test_joint_aparch_transformed_jet_includes_mean_curvature(
    name: str, code: int, shape: list[float]
) -> None:
    data = np.random.default_rng(100).normal(size=14).astype(np.float64)
    theta = _theta(shape, joint=True)
    orders = (1, 1, 1, 1)
    coordinates = _map(theta, code, orders=orders)
    base = _evaluate(theta, data, code, orders=orders)
    actual = _evaluate(coordinates, data, code, transformed=True, orders=orders)
    expected = native_aparch_coordinate_chain_rule_value_grad_hess(
        coordinates,
        theta,
        base[2],
        base[3],
        base[4],
        1,
        1,
        name,
        p_ar=1,
        q_ma=1,
    )
    assert actual[:2] == (0, 7)
    np.testing.assert_allclose(actual[3], expected[1], rtol=4e-7, atol=4e-9)
    np.testing.assert_allclose(actual[4], expected[2], rtol=4e-6, atol=4e-8)


@pytest.mark.parametrize(
    ("joint", "transformed", "code", "shape"),
    [
        (False, False, 0, []),
        (False, True, 4, [0.1, 1.3, 0.4]),
        (True, False, 0, []),
        (True, True, 1, [8.0]),
    ],
)
def test_every_sparse_request_is_a_projection_of_the_full_jet(
    joint: bool, transformed: bool, code: int, shape: list[float]
) -> None:
    data = np.random.default_rng(18).normal(size=18).astype(np.float64)
    orders = (1, 1, 1, 1) if joint else None
    theta = _theta(shape, joint=joint)
    point = _map(theta, code, orders=orders) if transformed else theta
    full = _evaluate(point, data, code, transformed=transformed, orders=orders)
    for request in range(1, 8):
        sparse = _evaluate(
            point,
            data,
            code,
            request,
            transformed=transformed,
            orders=orders,
        )
        assert sparse[:2] == (0, request)
        if request & 1:
            np.testing.assert_allclose(sparse[2], full[2], rtol=2e-12, atol=2e-12)
        if request & 2:
            np.testing.assert_allclose(sparse[3], full[3], rtol=2e-11, atol=2e-11)
        if request & 4:
            np.testing.assert_allclose(sparse[4], full[4], rtol=2e-10, atol=2e-10)


@lru_cache(maxsize=1)
def _runtime_library() -> ctypes.CDLL:
    library = ctypes.CDLL(_native_core.__file__)
    size = ctypes.c_size_t
    unsigned = ctypes.c_uint32
    pointer = ctypes.c_void_p
    library.vlk_runtime_aparch_jet_workspace_bytes.argtypes = [
        size, size, unsigned, unsigned, unsigned
    ]
    library.vlk_runtime_aparch_jet_workspace_bytes.restype = size
    library.vlk_runtime_aparch_jet_evaluate.argtypes = [
        pointer, size, pointer, size, ctypes.c_double, size, size,
        unsigned, unsigned, unsigned, pointer, size, pointer, pointer,
        pointer, ctypes.POINTER(unsigned),
    ]
    library.vlk_runtime_aparch_jet_evaluate.restype = ctypes.c_int32
    library.vlk_runtime_arma_aparch_jet_workspace_bytes.argtypes = [
        size, size, size, size, unsigned, unsigned, unsigned
    ]
    library.vlk_runtime_arma_aparch_jet_workspace_bytes.restype = size
    library.vlk_runtime_arma_aparch_jet_evaluate.argtypes = [
        pointer, size, pointer, size, ctypes.c_double, size, size, size, size,
        unsigned, unsigned, unsigned, pointer, size, pointer, pointer,
        pointer, ctypes.POINTER(unsigned),
    ]
    library.vlk_runtime_arma_aparch_jet_evaluate.restype = ctypes.c_int32
    return library


def _forced_runtime(
    theta: np.ndarray,
    data: np.ndarray,
    code: int,
    *,
    orders: tuple[int, int, int, int] | None,
) -> tuple[float, np.ndarray, np.ndarray]:
    library = _runtime_library()
    value = np.empty(1)
    gradient = np.empty(theta.size)
    hessian = np.empty((theta.size, theta.size))
    valid = ctypes.c_uint32(0)
    initial = float(np.mean(data * data))
    if orders is None:
        size = library.vlk_runtime_aparch_jet_workspace_bytes(1, 1, code, 0, 7)
        workspace = _workspace(size)
        status = library.vlk_runtime_aparch_jet_evaluate(
            theta.ctypes.data, theta.size, data.ctypes.data, data.size, initial,
            1, 1, code, 0, 7, workspace.ctypes.data, workspace.nbytes,
            value.ctypes.data, gradient.ctypes.data, hessian.ctypes.data,
            ctypes.byref(valid),
        )
    else:
        p_ar, q_ma, p_vol, q_vol = orders
        size = library.vlk_runtime_arma_aparch_jet_workspace_bytes(
            p_ar, q_ma, p_vol, q_vol, code, 0, 7
        )
        workspace = _workspace(size)
        status = library.vlk_runtime_arma_aparch_jet_evaluate(
            theta.ctypes.data, theta.size, data.ctypes.data, data.size, initial,
            p_ar, q_ma, p_vol, q_vol, code, 0, 7,
            workspace.ctypes.data, workspace.nbytes, value.ctypes.data,
            gradient.ctypes.data, hessian.ctypes.data, ctypes.byref(valid),
        )
    assert (status, valid.value) == (0, 7)
    return float(value[0]), gradient, hessian


@pytest.mark.parametrize(("code", "shape"), [(0, []), (4, [0.1, 1.3, 0.4])])
def test_fixed_and_runtime_aparch_specializations_are_identical(
    code: int, shape: list[float]
) -> None:
    data = np.random.default_rng(42).normal(size=20).astype(np.float64)
    standalone = _theta(shape, joint=False)
    fixed = _evaluate(standalone, data, code)[2:]
    runtime = _forced_runtime(standalone, data, code, orders=None)
    for left, right in zip(fixed, runtime):
        np.testing.assert_allclose(left, right, rtol=2e-12, atol=2e-12)

    joint = _theta(shape, joint=True)
    fixed_joint = _evaluate(joint, data, code, orders=(1, 1, 1, 1))[2:]
    runtime_joint = _forced_runtime(joint, data, code, orders=(1, 1, 1, 1))
    for left, right in zip(fixed_joint, runtime_joint):
        np.testing.assert_allclose(left, right, rtol=2e-12, atol=2e-12)


def test_arbitrary_joint_orders_match_ad() -> None:
    data = np.random.default_rng(204).normal(size=18).astype(np.float64)
    orders = (2, 1, 3, 2)
    theta = np.array([
        0.02, 0.10, -0.04, -0.08,
        0.06, 0.02, 0.015, 0.01,
        0.10, -0.12, 0.08,
        0.55, 0.18, 1.8,
    ])
    actual = _evaluate(theta, data, 0, orders=orders)
    expected = arma_aparch_value_grad_hess(theta, data, *orders, "normal")
    assert actual[:2] == (0, 7)
    np.testing.assert_allclose(actual[2], expected[0], rtol=2e-10, atol=2e-10)
    np.testing.assert_allclose(actual[3], expected[1], rtol=2e-8, atol=2e-9)
    np.testing.assert_allclose(actual[4], expected[2], rtol=2e-7, atol=2e-8)
    assert _native_core._arma_aparch_dispatch_kind(*orders, 0, 0) == 0


@pytest.mark.parametrize("p_ar,q_ma", [(0, 0), (0, 2), (3, 0)])
def test_joint_aparch_empty_reflection_blocks_have_exact_transformed_hessian(
    p_ar, q_ma
):
    data = np.random.default_rng(2087).normal(size=19)
    orders = (p_ar, q_ma, 2, 1)
    theta = np.r_[
        0.03,
        np.full(p_ar, 0.04),
        np.full(q_ma, -0.03),
        0.08,
        0.04,
        0.03,
        0.1,
        -0.1,
        0.6,
        1.7,
    ]
    coordinates = _map(theta, 0, orders=orders)
    theta_jet = _evaluate(theta, data, 0, orders=orders)
    assert theta_jet[:2] == (0, 7)
    expected = native_aparch_coordinate_chain_rule_value_grad_hess(
        coordinates, theta, *theta_jet[2:], 2, 1, "normal", p_ar=p_ar, q_ma=q_ma
    )
    for mask in (4, 5, 6, 7):
        actual = _evaluate(coordinates, data, 0, mask, transformed=True, orders=orders)
        assert actual[:2] == (0, mask)
        np.testing.assert_allclose(actual[4], expected[2], rtol=2e-7, atol=2e-8)


def test_aparch_domain_and_zero_residual_derivative_semantics() -> None:
    data = np.array([0.0, 1.0, 0.0, 1.0])
    theta = np.r_[0.0, 0.0, 0.0, 0.08, 0.07, 0.2, 0.82, 1.7]
    assert _evaluate(theta, data, 0, 1, orders=(1, 1, 1, 1), initial_variance=1.0)[:2] == (0, 1)
    assert _evaluate(theta, data, 0, 2, orders=(1, 1, 1, 1), initial_variance=1.0)[:2] == (0, 2)
    assert _evaluate(theta, data, 0, 4, orders=(1, 1, 1, 1), initial_variance=1.0)[0] == 3

    smooth = theta.copy()
    smooth[5] = 0.0
    smooth[7] = 2.0
    assert _evaluate(smooth, data, 0, 4, orders=(1, 1, 1, 1), initial_variance=1.0)[:2] == (0, 4)

    student = np.r_[VOLATILITY, 4.0]
    student[4] = 4.0
    assert _evaluate(student, data, 1, initial_variance=1.0)[0] == 3


def test_canonical_filters_match_direct_recursions() -> None:
    data = np.random.default_rng(33).normal(size=16).astype(np.float64)
    theta = _theta([], joint=False)
    initial = float(np.mean(data * data))
    workspace = _workspace(_native_core._aparch_filter_workspace_bytes(1, 1, 0))
    variance = np.empty_like(data)
    status = _native_core._aparch_filter_state(
        theta.ctypes.data, data.ctypes.data, data.size, initial,
        1, 1, 0, workspace.ctypes.data, workspace.nbytes,
        variance.ctypes.data,
    )
    power = np.empty_like(data)
    power[0] = initial ** (0.5 * theta[4])
    expected = np.empty_like(data)
    expected[0] = initial
    for time in range(1, data.size):
        shock = abs(data[time - 1]) - theta[2] * data[time - 1]
        power[time] = theta[0] + theta[1] * shock ** theta[4] + theta[3] * power[time - 1]
        expected[time] = power[time] ** (2.0 / theta[4])
    assert status == 0
    np.testing.assert_allclose(variance, expected, rtol=2e-14, atol=2e-14)


def test_aparch_exact_garch_reduction() -> None:
    data = np.random.default_rng(71).normal(size=18).astype(np.float64)
    aparch_theta = np.array([0.08, 0.07, 0.0, 0.82, 2.0])
    garch_theta = aparch_theta[[0, 1, 3]]
    actual = _evaluate(aparch_theta, data, 0)
    squared = np.ascontiguousarray(data * data)
    workspace = _workspace(_native_core._garch_workspace_bytes(1, 1, 0, 0, 7))
    value = np.empty(1)
    gradient = np.empty(3)
    hessian = np.empty((3, 3))
    status, valid = _native_core._garch_jet(
        garch_theta.ctypes.data, data.ctypes.data, squared.ctypes.data,
        data.size, float(np.mean(squared)), 1, 1, 0, 0, 7,
        workspace.ctypes.data, workspace.nbytes, value.ctypes.data,
        gradient.ctypes.data, hessian.ctypes.data,
    )
    common = np.array([0, 1, 3])
    assert (status, valid) == (0, 7)
    np.testing.assert_allclose(actual[2], value[0], rtol=2e-13, atol=2e-13)
    np.testing.assert_allclose(actual[3][common], gradient, rtol=2e-12, atol=2e-12)
    np.testing.assert_allclose(
        actual[4][np.ix_(common, common)], hessian, rtol=2e-11, atol=2e-11
    )


def test_expected_fixed_dispatches_are_reachable() -> None:
    for p in (1, 2):
        for q in (1, 2):
            assert _native_core._aparch_dispatch_kind(p, q, 0, 0) == 1
    assert _native_core._aparch_dispatch_kind(3, 2, 0, 0) == 0
    assert _native_core._arma_aparch_dispatch_kind(1, 1, 1, 1, 0, 0) == 1
    assert _native_core._arma_aparch_dispatch_kind(2, 1, 1, 1, 0, 0) == 0


def _simulate_joint_aparch_studentt() -> np.ndarray:
    rng = np.random.default_rng(91)
    observations = 1_200
    nu = 7.0
    innovations = rng.standard_t(nu, size=observations) * np.sqrt((nu - 2.0) / nu)
    residuals = np.empty(observations)
    powers = np.empty(observations)
    data = np.empty(observations)
    powers[0] = 1.0
    residuals[0] = innovations[0]
    data[0] = residuals[0]
    for index in range(1, observations):
        powers[index] = (
            0.08
            + 0.10 * (abs(residuals[index - 1]) - 0.15 * residuals[index - 1]) ** 1.6
            + 0.80 * powers[index - 1]
        )
        residuals[index] = powers[index] ** (1.0 / 1.6) * innovations[index]
        data[index] = (
            0.03 + 0.18 * data[index - 1]
            - 0.08 * residuals[index - 1] + residuals[index]
        )
    return np.ascontiguousarray(data)


def test_aparch_density_matrix_and_joint_surface_default_to_native() -> None:
    from scivol._native_fit import _uses_native_by_default

    densities = (
        Normal(), StudentT(), SkewT(), GED(),
        GeneralizedHyperbolic(), NormalInverseGaussian(),
        Hyperbolic(), VarianceGamma(),
    )
    for density in densities:
        assert _uses_native_by_default(APARCH(1, 1) + density)
        assert _uses_native_by_default(ARMA(1, 1) + APARCH(1, 1) + density)


def test_joint_aparch_default_is_credible_and_beats_matched_scipy() -> None:
    from scivol._native_fit import (
        _describe_model,
        _evaluate_theta_request,
        _initial_theta,
    )

    data = _simulate_joint_aparch_studentt()
    spec = ARMA(1, 1) + APARCH(1, 1) + StudentT()
    model = _describe_model(spec)
    start, _ = _initial_theta(spec, model, data, None)
    native = spec.fit(data, initial_params=start, maxiter=500)
    scipy = spec.fit(
        data,
        solver="slsqp",
        initial_params=start,
        maxiter=500,
    )
    native_nll = _evaluate_theta_request(
        model, data, np.asarray(native.params), request=1
    )[0]
    scipy_nll = _evaluate_theta_request(
        model, data, np.asarray(scipy.params), request=1
    )[0]

    assert native.converged
    assert native.fit_info.solver == "native-slsqp"
    assert native.fit_info.kkt_residual <= native.fit_info.kkt_tolerance
    assert native.fit_info.maximum_constraint_violation == 0.0
    assert native_nll <= scipy_nll + 1e-8
    filtered = spec.filter(data, native.params)
    np.testing.assert_allclose(filtered.residuals, native.resid)
    np.testing.assert_allclose(filtered.sigma2, native.sigma2)
    assert spec.score(data, native.params).shape == native.params.shape
    assert spec.hessian(data, native.params).shape == (
        native.params.size, native.params.size
    )
    assert native.garch_params is not None
    assert 0.0 < native.garch_params.persistence < 1.0
    assert 0.0 < spec.get_component(Role.VOLATILITY).persistence() < 1.0
