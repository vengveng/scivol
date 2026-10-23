"""Independent precision and portability checks for the production wide scalar."""

import ctypes
from pathlib import Path
import shutil
import subprocess
import sys

import mpmath as mp
import numpy as np
import pytest


_INTEGRAL_TYPES = [
    ("boolean", ctypes.c_bool, False),
    ("i8", ctypes.c_int8, True),
    ("u8", ctypes.c_uint8, False),
    ("i16", ctypes.c_int16, True),
    ("u16", ctypes.c_uint16, False),
    ("i32", ctypes.c_int32, True),
    ("u32", ctypes.c_uint32, False),
    ("i64", ctypes.c_int64, True),
    ("u64", ctypes.c_uint64, False),
    ("signed_long", ctypes.c_long, True),
    ("unsigned_long", ctypes.c_ulong, False),
    ("size_t", ctypes.c_size_t, False),
]
_WIDE_INTEGRAL_TYPES = [item for item in _INTEGRAL_TYPES if item[0] in ("i64", "u64", "size_t")]


@pytest.fixture(
    scope="module", params=[False, True], ids=["native-limbs", "portable-limbs"]
)
def backend(request, tmp_path_factory):
    compiler = shutil.which("clang++") or shutil.which("g++")
    if compiler is None or sys.platform == "win32":
        pytest.skip("Standalone precision probe requires a Unix C++20 toolchain")
    root = Path(__file__).resolve().parents[1]
    directory = tmp_path_factory.mktemp("wide_scalar")
    native = root / "scivol/_native"
    obj = directory / "probe.o"
    library = directory / ("probe.dylib" if sys.platform == "darwin" else "probe.so")
    executable = directory / "probe"
    flags = (
        ["-Wframe-larger-than=16384", "-Werror=frame-larger-than"]
        if "clang" in Path(compiler).name
        else ["-Werror=frame-larger-than=16384"]
    )
    includes = [
        native,
        root / "scivol/_csrc",
        native / "third_party/boost",
        native / "third_party/tlfloat",
    ]
    command = [
        compiler,
        "-std=c++20",
        "-O3",
        "-ffp-contract=fast",
        "-funroll-loops",
        "-march=native",
        "-fno-exceptions",
        "-fno-rtti",
        "-fPIC",
        *flags,
    ]
    if request.param:
        command.append("-DSCIVOL_WIDE_PORTABLE_LIMBS")
    for path in includes:
        command.extend(["-I", str(path)])
    subprocess.run(
        [*command, "-c", str(root / "tests/native_wide_precision.cpp"), "-o", str(obj)],
        check=True,
        capture_output=True,
        text=True,
        timeout=120,
    )
    subprocess.run([compiler, str(obj), "-o", str(executable)], check=True, timeout=30)
    # Run before any Python calls to check first-use C++ allocations as well.
    subprocess.run([str(executable)], check=True, timeout=30)
    subprocess.run(
        [
            compiler,
            "-dynamiclib" if sys.platform == "darwin" else "-shared",
            str(obj),
            "-o",
            str(library),
        ],
        check=True,
        timeout=30,
    )
    result = ctypes.CDLL(str(library))
    for name, arguments in {
        "probe_scalar_binary": [
            ctypes.c_uint,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
        ],
        "probe_scalar_fma": [ctypes.c_double] * 3 + [ctypes.c_void_p],
        "probe_portable_math": [ctypes.c_uint, ctypes.c_double, ctypes.c_void_p],
        "probe_wide_erf": [ctypes.c_void_p, ctypes.c_void_p],
        "probe_wide_exp": [ctypes.c_void_p, ctypes.c_void_p],
        "probe_wide_log": [ctypes.c_void_p, ctypes.c_void_p],
        "probe_moment": [ctypes.c_double] * 2 + [ctypes.c_void_p],
        "probe_nig_moment": [ctypes.c_uint] + [ctypes.c_void_p] * 3,
        "probe_elementary_moment": [ctypes.c_uint] * 2 + [ctypes.c_void_p] * 3,
        "probe_gh_moment": [ctypes.c_void_p] * 2,
        "probe_gh_series": [ctypes.c_void_p] * 2,
        "probe_gh_series_route": [ctypes.c_void_p],
        "probe_gh_bessel_ladder": [ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p],
        "probe_folded_normal": [ctypes.c_void_p] * 2,
        "probe_hyperbolic_moment": [ctypes.c_void_p] * 2,
        "probe_residual": [ctypes.c_void_p] * 2
        + [ctypes.c_size_t, ctypes.c_double, ctypes.c_void_p],
    }.items():
        function = getattr(result, name)
        function.argtypes, function.restype = arguments, ctypes.c_int
    result.probe_exp_polynomial.argtypes = [ctypes.c_void_p, ctypes.c_double,
                                           ctypes.c_void_p, ctypes.c_void_p]
    result.probe_exp_polynomial.restype = None
    result.probe_wide_add.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t]
    result.probe_wide_add.restype = None
    result.probe_wide_multiply_observed.argtypes = [ctypes.c_void_p] * 4 + [ctypes.c_size_t]
    result.probe_wide_multiply_observed.restype = None
    for name, kind, _ in _INTEGRAL_TYPES:
        function = getattr(result, "probe_wide_integral_" + name)
        function.argtypes = [kind, ctypes.c_void_p]
        function.restype = None
    return result


def check_integral_conversions(backend, name, values):
    from fractions import Fraction

    probe = getattr(backend, "probe_wide_integral_" + name)
    for value in values:
        output = np.full(6, 271828.125)
        probe(value, output[1:].ctypes.data)
        np.testing.assert_array_equal(output[[0, -1]], 271828.125)
        np.testing.assert_array_equal(output[1:3].view(np.uint64), output[3:5].view(np.uint64))
        high, low = map(float, output[1:3])
        assert Fraction(high) + Fraction(low) == value, (name, value, high, low)
        assert high == float(value), (name, value, high)
        assert low == float(value - int(high)), (name, value, low)
        with mp.workprec(160):
            assert mp.mpf(high) + mp.mpf(low) == mp.mpf(value), (name, value)


@pytest.mark.parametrize("name,kind,signed", _INTEGRAL_TYPES, ids=[x[0] for x in _INTEGRAL_TYPES])
def test_wide_integral_extrema_and_small_types(backend, name, kind, signed):
    bits = 8 * ctypes.sizeof(kind)
    lower = -(1 << (bits - 1)) if signed else 0
    upper = (1 << (bits - int(signed))) - 1
    if kind is ctypes.c_bool:
        upper = 1
    values = {lower, lower + 1, lower + 2, upper - 2, upper - 1, upper, -1, 0, 1}
    for exponent in range(bits):
        power = 1 << exponent
        for delta in (-1, 0, 1):
            values.update((power + delta, -power + delta))
    if bits == 8:
        values.update(range(lower, upper + 1))
    check_integral_conversions(backend, name, sorted(x for x in values if lower <= x <= upper))


@pytest.mark.parametrize("name,kind,signed", _WIDE_INTEGRAL_TYPES,
                         ids=[x[0] for x in _WIDE_INTEGRAL_TYPES])
def test_wide_integral_nearest_even_ties(backend, name, kind, signed):
    bits = 8 * ctypes.sizeof(kind)
    lower = -(1 << (bits - 1)) if signed else 0
    upper = (1 << (bits - int(signed))) - 1
    values = {lower, lower + 1, upper - 1, upper}
    for exponent in range(53, bits):
        spacing = 1 << (exponent - 52)
        for parity in (0, 1):
            midpoint = (1 << exponent) + parity * spacing + spacing // 2
            for delta in (-1, 0, 1):
                values.update((midpoint + delta, -midpoint + delta))
    for exponent in range(54, bits + 1):
        midpoint = (1 << exponent) - (1 << (exponent - 54))
        for delta in (-1, 0, 1):
            values.update((midpoint + delta, -midpoint + delta))
    check_integral_conversions(backend, name, sorted(x for x in values if lower <= x <= upper))


@pytest.mark.parametrize("name,kind,signed", _WIDE_INTEGRAL_TYPES,
                         ids=[x[0] for x in _WIDE_INTEGRAL_TYPES])
def test_wide_integral_seeded_full_width(backend, name, kind, signed):
    bits = 8 * ctypes.sizeof(kind)
    rng = np.random.default_rng(421031)
    values = [int(word) & ((1 << bits) - 1) for word in rng.bit_generator.random_raw(256)]
    if signed:
        values = [value - (1 << bits) if value >= (1 << (bits - 1)) else value for value in values]
    check_integral_conversions(backend, name, values)


def check_multiply_observation(backend, inputs, expected_calls):
    inputs = np.ascontiguousarray(inputs, dtype=np.float64).reshape(-1, 4)
    before = inputs.view(np.uint64).copy()
    count = len(inputs)
    output = np.full((count+2, 6), 271828.125)
    observations = np.full(count+2, 73, dtype=np.uintp)
    observed = np.full((count+2, 4), 271828.125)
    backend.probe_wide_multiply_observed(
        inputs.ctypes.data, output[1:].ctypes.data, observations[1:].ctypes.data,
        observed[1:].ctypes.data, count,
    )
    np.testing.assert_array_equal(inputs.view(np.uint64), before)
    np.testing.assert_array_equal(output[[0, -1]], 271828.125)
    np.testing.assert_array_equal(observed[[0, -1]], 271828.125)
    np.testing.assert_array_equal(observations[[0, -1]], 73)
    np.testing.assert_array_equal(observations[1:-1], expected_calls)
    bits = output[1:-1].view(np.uint64)
    np.testing.assert_array_equal(bits[:, :2], bits[:, 2:4])
    np.testing.assert_array_equal(bits[:, :2], bits[:, 4:6])
    fallback = np.asarray(expected_calls, dtype=bool)
    np.testing.assert_array_equal(observed[1:-1][fallback].view(np.uint64), before[fallback])
    np.testing.assert_array_equal(observed[1:-1][~fallback], 271828.125)


def test_multiply_observed_thresholds_and_limb_bits(backend):
    cases, calls = [], []
    for boundary in (2.**-900, 2.**900):
        for high in (np.nextafter(boundary, 0.), boundary, np.nextafter(boundary, np.inf)):
            fallback = int(high < 2.**-900 or high > 2.**900)
            for left_sign in (-1., 1.):
                for right_sign in (-1., 1.):
                    for fraction in (-.49, 0., .49):
                        cases.append([left_sign*high, fraction*np.spacing(high),
                                      right_sign, -fraction*2.**-52])
                        calls.append(fallback)
    check_multiply_observation(backend, cases, calls)


def test_multiply_observed_range_edges_and_signed_zeros(backend):
    values = (0., -0., 1., -1., 2.**-1074, -2.**-1074, 2.**-1022,
              2.**-600, 2.**600, np.finfo(float).max, np.inf, -np.inf, np.nan)
    cases, calls = [], []
    for left in values:
        for right in values:
            with np.errstate(over="ignore", invalid="ignore", under="ignore"):
                product = left*right
            fallback = int(not (2.**-900 <= abs(product) <= 2.**900))
            for low in (0., -0.):
                cases.append([left, low, right, low])
                calls.append(fallback)
    check_multiply_observation(backend, cases, calls)


def test_multiply_observed_seeded_hot_and_fallback_products(backend):
    rng = np.random.default_rng(421010)
    exponents = rng.integers(-1000, 1001, (2048, 2))
    high = np.ldexp(rng.uniform(.5, 1., exponents.shape), exponents)
    high *= rng.choice([-1., 1.], high.shape)
    low = abs(np.spacing(high))*rng.uniform(-.49, .49, high.shape)
    cases = np.column_stack((high[:, 0], low[:, 0], high[:, 1], low[:, 1]))
    with np.errstate(over="ignore", under="ignore"):
        product = high[:, 0]*high[:, 1]
    fallback = ~((abs(product) >= 2.**-900) & (abs(product) <= 2.**900))
    check_multiply_observation(backend, cases, fallback.astype(np.uintp))


@pytest.mark.parametrize("count", [0, 1, 3, 17])
def test_multiply_observed_preserves_guards(backend, count):
    check_multiply_observation(backend, np.tile([1., 2.**-54, -1., 2.**-55], (count, 1)),
                              np.zeros(count, dtype=np.uintp))


def elementary_moment_reference(density, shape):
    """Integrate the Hansen density independently; Student/GED have gamma ratios."""
    nu = mp.mpf(shape[0])
    if density == 1:
        return mp.sqrt((nu-2)/mp.pi)*mp.gamma((nu-1)/2)/mp.gamma(nu/2)
    if density == 3:
        return mp.gamma(2/nu)/mp.sqrt(mp.gamma(1/nu)*mp.gamma(3/nu))
    skew = mp.mpf(shape[1])
    constant = mp.gamma((nu+1)/2)/(mp.gamma(nu/2)*mp.sqrt(mp.pi*(nu-2)))
    a = 4*skew*constant*(nu-2)/(nu-1)
    b = mp.sqrt(1+3*skew*skew-a*a)
    def pdf(z):
        shifted = b*z+a
        scaled = shifted/(1+skew*mp.sign(shifted))
        return b*constant*(1+scaled*scaled/(nu-2))**(-(nu+1)/2)
    edges = sorted({mp.mpf(0), abs(a/b), mp.mpf(1), mp.mpf(4), mp.inf})
    return mp.quad(lambda z: z*(pdf(z)+pdf(-z)), edges)


@pytest.mark.parametrize("density,shape", [
    (1, (np.nextafter(2., 3.),)), (1, (18.27325702187228,)),
    (1, (np.nextafter(128., 0.),)), (1, (128.,)), (1, (1e6,)), (1, (1e12,)),
    (2, (2.000001, .999)), (2, (3., -.8)),
    (2, (18.228928175428127, -.02224389417339265)),
    (2, (128., .1)), (2, (1e6, .8)), (2, (6., 0.)), (2, (6., 1e-12)),
    (3, (.2,)), (3, (1.7310258135871597,)), (3, (2.,)), (3, (100.,)),
])
def test_elementary_wide_moment_matches_independent_reference(backend, density, shape):
    shape = np.array(shape)
    with mp.workdps(65):
        expected = elementary_moment_reference(density, shape)
        parts, unused = np.empty(3), np.full(5, 19.)
        assert backend.probe_elementary_moment(density, 0, shape.ctypes.data,
                                              parts.ctypes.data, unused.ctypes.data) == 0
        assert abs(mp.fsum(map(mp.mpf, parts))-expected) < mp.mpf("1e-24")
        np.testing.assert_array_equal(unused, 19.)
        if density == 2:
            shape[1] *= -1
            reflected = np.empty(3)
            assert backend.probe_elementary_moment(density, 0, shape.ctypes.data,
                                                  reflected.ctypes.data, unused.ctypes.data) == 0
            np.testing.assert_array_equal(reflected, parts)


@pytest.mark.parametrize("density,shape", [(1, (2.,)), (1, (np.nan,)),
    (2, (6., 1.)), (2, (6., np.nan)), (3, (0.,)), (3, (np.nan,))])
def test_elementary_invalid_moment_preserves_outputs(backend, density, shape):
    shape = np.array(shape)
    for order in range(3):
        parts, derivatives = np.full(3, 71.), np.full(5, 19.)
        assert backend.probe_elementary_moment(density, order, shape.ctypes.data,
                                              parts.ctypes.data, derivatives.ctypes.data) != 0
        np.testing.assert_array_equal(parts, 71.)
        np.testing.assert_array_equal(derivatives, 19.)


@pytest.mark.parametrize("nu", (2.01, 6., 18.228928175428127, 128., 1e6))
def test_wide_symmetric_skewt_reduces_to_student(backend, nu):
    shape, values = np.array([nu, 0.]), []
    for density in (1, 2):
        parts, derivatives = np.empty(3), np.empty(5)
        assert backend.probe_elementary_moment(density, 2, shape.ctypes.data,
                                              parts.ctypes.data, derivatives.ctypes.data) == 0
        values.append((parts, derivatives[[0, 1] if density == 1 else [0, 2]]))
    np.testing.assert_array_equal(values[0][0], values[1][0])
    for student, skewt in zip(*values):
        np.testing.assert_allclose(student, skewt, rtol=2e-7, atol=2e-10)


def test_student_gamma_ratio_preserves_shape_boundaries(backend):
    shapes = np.r_[2+np.geomspace(1e-15, 126, 160), np.linspace(2.01, 128, 160),
                   np.nextafter(2., np.inf), np.nextafter(128., 0.), 128., 1e3, 1e12]
    with mp.workdps(100):
        for nu in shapes:
            shape = np.array([nu])
            parts, unused = np.empty(3), np.full(5, 19.)
            assert backend.probe_elementary_moment(1, 0, shape.ctypes.data,
                                                  parts.ctypes.data, unused.ctypes.data) == 0
            expected = elementary_moment_reference(1, shape)
            tolerance = mp.mpf("2e-30") if nu < 128 else mp.mpf("1e-24")
            assert abs(mp.fsum(map(mp.mpf, parts))-expected) <= expected*tolerance
            np.testing.assert_array_equal(unused, 19.)


@pytest.mark.parametrize("density,shape", [(1, (2.01,)), (1, (18.2,)),
    (2, (2.01, .8)), (2, (3., -.8)), (2, (18.2, -.02)), (2, (128., .1)),
    (3, (.2,)), (3, (1.7,))])
def test_elementary_moment_derivatives_match_independent_ad(backend, density, shape):
    from scivol._devtools.ad_oracle import (
        _jax_modules, _jax_aparch_power_moment, _jax_hansen_skewt_abs_moment,
    )
    jax, jnp, jsp = _jax_modules()
    shape = np.array(shape)
    references = []
    for count in (64, 128):
        nodes, weights = map(jnp.asarray, np.polynomial.legendre.leggauss(count))
        def moment(parameters):
            if density == 2:
                return _jax_hansen_skewt_abs_moment(jnp, jsp, *parameters, nodes, weights)
            return _jax_aparch_power_moment(jnp, jsp, 0., 1., parameters,
                                           "studentt" if density == 1 else "ged")
        references.append(np.r_[jax.grad(moment)(shape),
            np.asarray(jax.hessian(moment)(shape))[np.triu_indices(len(shape))]])
    np.testing.assert_allclose(references[0], references[1], rtol=2e-10, atol=2e-10)
    for order in range(3):
        parts, derivatives = np.empty(3), np.full(5, 71.)
        assert backend.probe_elementary_moment(density, order, shape.ctypes.data,
                                              parts.ctypes.data, derivatives.ctypes.data) == 0
        count = (0, len(shape), len(references[-1]))[order]
        np.testing.assert_allclose(derivatives[:count], references[-1][:count],
                                   rtol=2e-7, atol=2e-10)
        np.testing.assert_array_equal(derivatives[count:], 71.)


def test_folded_normal_tail_shortcut_preserves_wide_precision(backend):
    with mp.workdps(110):
        root_two, normal_constant = mp.sqrt(2), mp.sqrt(2/mp.pi)
        for scale in (1e-100, 1e-4, 1., 1e4, 1e100):
            for z in (0., .5, 2., 8., 15., np.nextafter(16., 0.),
                      16., np.nextafter(16., 17.), 20., 40., 100.):
                for sign in (-1., 1.):
                    high = sign*z*scale
                    for low in (0., np.spacing(high)/4):
                        inputs = np.array([high, low, scale, 0.])
                        mean, sd = mp.mpf(high)+mp.mpf(low), mp.mpf(scale)
                        t = abs(mean)/sd
                        expected = sd*normal_constant*mp.exp(-t*t/2)
                        expected += abs(mean)*mp.erf(t/root_two)
                        parts = np.empty(3)
                        assert backend.probe_folded_normal(inputs.ctypes.data, parts.ctypes.data) == 0
                        assert abs(mp.fsum(map(mp.mpf, parts))-expected) < mp.mpf("2e-30")*expected
        # A bound on numerical work omission, not a change to the density's tail.
        assert normal_constant*mp.exp(-128)/16 < mp.mpf("1.3e-57")
    for sign in (-1., 1.):
        inputs, parts = np.array([sign*1e300, 0., 1e-300, 0.]), np.empty(3)
        assert backend.probe_folded_normal(inputs.ctypes.data, parts.ctypes.data) == 0
        np.testing.assert_array_equal(parts, [1e300, 0., 0.])


@pytest.mark.parametrize("mean,sd", [(np.nan, 1.), (1., 0.), (1., -1.), (1., np.inf)])
def test_folded_normal_invalid_inputs_preserve_output(backend, mean, sd):
    inputs, parts = np.array([mean, 0., sd, 0.]), np.full(3, 71.)
    assert backend.probe_folded_normal(inputs.ctypes.data, parts.ctypes.data) != 0
    np.testing.assert_array_equal(parts, 71.)


def gh_mixing_moment(shape):
    """Normal-GIG mixture, independent of the marginal/integration-by-parts path."""
    rho, zeta, lam = map(mp.mpf, shape)
    k0 = mp.besselk(lam, zeta)
    first = mp.besselk(lam+1, zeta)/k0
    second = mp.besselk(lam+2, zeta)/k0
    scale = mp.sqrt((1-rho*rho)*first/zeta+rho*rho*(second-first*first))

    def integrand(s):
        weight = mp.exp(s)
        mean = rho*(weight-first)
        sd = mp.sqrt((1-rho*rho)*weight/zeta)
        ratio = mean/sd
        absolute = sd*mp.sqrt(2/mp.pi)*mp.exp(-ratio*ratio/2)
        absolute += mean*mp.erf(ratio/mp.sqrt(2))
        return absolute/scale*mp.exp(lam*s-zeta*mp.cosh(s))/(2*k0)

    transition = mp.log(first)
    edges = sorted(set(map(mp.mpf, [-14, -6, -3, 0, 3, 6, 14])) | {transition})
    return mp.quad(integrand, edges)


@pytest.mark.parametrize("shape", [
    (0., .1, -6.), (.12, 1.8, .4), (-.994, 25., 6.),
    (-.994999999999983, 5.682512243680064, 4.584422050284689),
    (.10369227249375981, 2.0075513862325924, 4.225768640979743),
])
def test_gh_wide_value_matches_independent_log_mixing_integral(backend, shape):
    shape = np.asarray(shape)
    with mp.workdps(65):
        expected = gh_mixing_moment(shape)
        parts = np.empty(3)
        assert backend.probe_gh_moment(shape.ctypes.data, parts.ctypes.data) == 0
        assert abs(mp.fsum(map(mp.mpf, parts))-expected) < mp.mpf("1e-24")


@pytest.mark.parametrize("shape", [
    (.02, 2., 0.), (.02, np.nextafter(2., 3.), 0.),
    (.02, .1, -6.), (.25, 25., 6.), (.1, .5, -.5),
    (.12, 1.8, .4), (.1042686090926312, 2.067957788198044, 4.195805375275462),
    (.10370883795527759, 2.005956743012256, 4.227023869111284),
])
def test_gh_centered_series_matches_independent_mixture(backend, shape):
    shape, parts = np.array(shape), np.empty(5)
    with mp.workdps(65):
        expected = gh_mixing_moment(shape)
        assert backend.probe_gh_series(shape.ctypes.data, parts.ctypes.data) == 0
        assert abs(mp.fsum(map(mp.mpf, parts[:3]))-expected) < mp.mpf("1e-24")
        assert 0 <= parts[3] <= 2e-31 and 0 <= parts[4] < 32
        shape[0] *= -1
        reflected = np.empty(5)
        assert backend.probe_gh_series(shape.ctypes.data, reflected.ctypes.data) == 0
        np.testing.assert_array_equal(reflected, parts)


@pytest.mark.parametrize("shape,attempt", [
    ((0., 1., 1.), False),
    ((-.24767950995798557, .10618558831708105, 5.250536013482545), False),
    ((.1042686090926312, 2.067957788198044, 4.195805375275462), True),
])
def test_gh_series_work_selection_preserves_closed_forms(backend, shape, attempt):
    shape = np.array(shape)
    assert bool(backend.probe_gh_series_route(shape.ctypes.data)) == attempt


@pytest.mark.parametrize("fraction,zeta", [
    (0., .1), (.4, .5), (.5, 2.), (np.nextafter(.5, 1.), np.nextafter(2., 3.)),
    (np.nextafter(1., 0.), 25.),
])
def test_gh_bessel_ladder_matches_independent_orders(backend, fraction, zeta):
    inputs, parts = np.array([fraction, zeta]), np.empty(3)
    with mp.workdps(65):
        for index in (-40, -38, -5, -2, -1, 0, 1, 5, 38, 39):
            assert backend.probe_gh_bessel_ladder(inputs.ctypes.data, index, parts.ctypes.data) == 0
            expected = mp.besselk(mp.mpf(fraction)+index, mp.mpf(zeta))
            assert abs(mp.fsum(map(mp.mpf, parts))/expected-1) < mp.mpf("5e-28")
    for index in (-41, 40):
        parts.fill(71.)
        assert backend.probe_gh_bessel_ladder(inputs.ctypes.data, index, parts.ctypes.data) != 0
        np.testing.assert_array_equal(parts, 71.)


@pytest.mark.parametrize("shape", [(.3, 1.8, .4), (.994, 25., 6.)])
def test_gh_series_exhaustion_uses_canonical_fallback(backend, shape):
    shape, unused, parts = np.array(shape), np.full(5, 71.), np.empty(3)
    assert backend.probe_gh_series(shape.ctypes.data, unused.ctypes.data) != 0
    np.testing.assert_array_equal(unused, 71.)
    assert backend.probe_gh_moment(shape.ctypes.data, parts.ctypes.data) == 0
    with mp.workdps(65):
        expected = gh_mixing_moment(shape)
        assert abs(mp.fsum(map(mp.mpf, parts))-expected) < mp.mpf("1e-24")


@pytest.mark.parametrize("rho,zeta", [(0., .1), (1e-12, 25.), (.12, 1.8),
                                      (.7, .1), (np.nextafter(.995, 0.), 25.)])
def test_hyperbolic_finite_moment_matches_independent_mixture(backend, rho, zeta):
    with mp.workdps(65):
        expected = gh_mixing_moment((rho, zeta, 1.))
        for sign in (-1., 1.):
            shape, parts = np.array([sign*rho, zeta]), np.empty(3)
            assert backend.probe_hyperbolic_moment(shape.ctypes.data, parts.ctypes.data) == 0
            assert abs(mp.fsum(map(mp.mpf, parts))-expected) < mp.mpf("1e-24")


@pytest.mark.parametrize("shape", [(np.nan, 1.), (.995, 1.), (0., .09), (0., 26.)])
def test_hyperbolic_invalid_moment_preserves_outputs(backend, shape):
    shape, parts = np.array(shape), np.full(3, 71.)
    assert backend.probe_hyperbolic_moment(shape.ctypes.data, parts.ctypes.data) != 0
    np.testing.assert_array_equal(parts, 71.)


@pytest.mark.parametrize("shape", [(np.nan, 1., 1.), (.995, 1., 1.), (0., .09, 1.)])
def test_gh_invalid_moment_preserves_outputs(backend, shape):
    shape, parts = np.array(shape), np.full(3, 71.)
    assert backend.probe_gh_moment(shape.ctypes.data, parts.ctypes.data) != 0
    np.testing.assert_array_equal(parts, 71.)


def nig_mixing_moment(rho, zeta):
    """Original Normal-IG mixing integral, independent of the reduced integrand."""
    rho, zeta = mp.mpf(rho), mp.mpf(zeta)

    def integrand(s):
        mean = rho * mp.sqrt(zeta) * mp.expm1(s)
        sd = mp.sqrt((1-rho*rho) * mp.exp(s))
        standardized = mean / sd
        absolute = sd * mp.sqrt(2/mp.pi) * mp.exp(-standardized**2/2)
        absolute += mean * mp.erf(standardized/mp.sqrt(2))
        mass = mp.sqrt(zeta/(2*mp.pi)) * mp.exp(-s/2-zeta*(mp.cosh(s)-1))
        return mass * absolute

    return mp.quad(integrand, [-12, -3, -1, 0, 1, 3, 12])


@pytest.mark.parametrize("shape", [(0., .1), (.5, 1.8), (-.994, 25.)])
def test_nig_wide_value_matches_original_mixing_integral(backend, shape):
    shape = np.asarray(shape)
    with mp.workdps(65):
        expected = nig_mixing_moment(*shape)
        values = []
        for order in range(3):
            parts, derivatives = np.empty(3), np.full(5, 271828.)
            assert backend.probe_nig_moment(order, shape.ctypes.data,
                                           parts.ctypes.data, derivatives.ctypes.data) == 0
            assert abs(mp.fsum(map(mp.mpf, parts))-expected) < mp.mpf("1e-24")
            values.append(parts)
            untouched = (0, 2, 5)[order]
            np.testing.assert_array_equal(derivatives[untouched:], 271828.)
        np.testing.assert_array_equal(values[0], values[1])
        np.testing.assert_array_equal(values[0], values[2])


def test_nig_reduced_moment_derivatives_match_independent_ad(backend):
    from scivol._devtools.ad_oracle import _jax_modules

    jax, jnp, jsp = _jax_modules()
    shapes = np.array([(rho, zeta) for rho in (0., .2, -.8, .994)
                       for zeta in (.1, 1.8, 25.)])
    references = []
    for order in (64, 128):
        nodes, weights = np.polynomial.legendre.leggauss(order)
        edges = np.linspace(-12., 12., 97)
        s = jnp.asarray(((edges[:-1, None]+edges[1:, None])/2 + nodes[None, :]/8).ravel())
        weights = jnp.asarray(np.broadcast_to(weights[None, :]/8, (96, order)).ravel())

        def moment(shape):
            rho, zeta = shape
            mean = rho*jnp.sqrt(zeta)*jnp.expm1(s)
            sd = jnp.sqrt((1-rho*rho)*jnp.exp(s))
            ratio = mean/sd
            absolute = sd*jnp.sqrt(2/jnp.pi)*jnp.exp(-ratio*ratio/2)
            absolute += mean*jsp.special.erf(ratio/jnp.sqrt(2.))
            mass = jnp.sqrt(zeta/(2*jnp.pi))*jnp.exp(-s/2-zeta*(jnp.cosh(s)-1))
            return jnp.sum(weights*mass*absolute)

        references.append(jax.jit(jax.vmap(lambda shape: (
            jax.grad(moment)(shape), jax.hessian(moment)(shape))))(jnp.asarray(shapes)))
    for coarse, fine in zip(*references):
        np.testing.assert_allclose(coarse, fine, rtol=2e-10, atol=2e-12)
    gradients, hessians = map(np.asarray, references[-1])
    for shape, gradient, hessian in zip(shapes, gradients, hessians):
        parts, derivatives = np.empty(3), np.empty(5)
        assert backend.probe_nig_moment(2, shape.ctypes.data,
                                       parts.ctypes.data, derivatives.ctypes.data) == 0
        expected = np.r_[gradient, hessian[0, 0], hessian[0, 1], hessian[1, 1]]
        np.testing.assert_allclose(derivatives, expected, rtol=2e-7, atol=2e-10)


@pytest.mark.parametrize("shape", [(np.nan, 1.), (.995, 1.), (0., .09), (0., 26.)])
def test_nig_invalid_moment_preserves_outputs(backend, shape):
    shape = np.array(shape)
    for order in range(3):
        parts, derivatives = np.full(3, 71.), np.full(5, 19.)
        assert backend.probe_nig_moment(order, shape.ctypes.data,
                                       parts.ctypes.data, derivatives.ctypes.data) != 0
        np.testing.assert_array_equal(parts, 71.)
        np.testing.assert_array_equal(derivatives, 19.)


def test_wide_residual_preserves_product_and_sum_errors(backend):
    rng = np.random.default_rng(739)
    cases = [([2.**-500], [2.**-500], 0.), ([2.**500], [2.**450], 0.),
             ([1., 1.], [1., 2.**-54], 1.), ([0., 1.], [3., 2.**-54], 0.)]
    for columns in (1, 4, 7, 19):
        for _ in range(20):
            theta, features = rng.normal(size=(2, columns))
            cases.append((theta, features, float(theta @ features)))
    with mp.workdps(150):
        for theta, features, observed in cases:
            theta, features = np.asarray(theta), np.asarray(features)
            expected = mp.mpf(observed) - mp.fsum(
                mp.mpf(a)*mp.mpf(b) for a, b in zip(theta, features))
            parts = np.empty(3)
            assert backend.probe_residual(theta.ctypes.data, features.ctypes.data,
                                          len(theta), observed, parts.ctypes.data) == 0
            assert abs(mp.fsum(map(mp.mpf, parts))-expected) <= max(
                abs(expected)*mp.mpf("2e-30"), mp.mpf(2)**-1075)


def test_wide_residual_preserves_separated_cancellation_terms(backend):
    from fractions import Fraction

    # The fourth separated term exceeds the short expansion and needs fallback.
    for powers in ((0, -100, -200), (0, -100, -200, -300)):
        positive = np.exp2(np.array(powers, dtype=float))
        theta = np.r_[positive, -positive[:-1]]
        features = np.ones(len(theta))
        parts = np.empty(3)
        assert backend.probe_residual(theta.ctypes.data, features.ctypes.data,
                                      len(theta), 0., parts.ctypes.data) == 0
        assert sum(map(Fraction, parts)) == -Fraction(float(positive[-1]))


def test_wide_residual_matches_exact_rational_dot_products(backend):
    from fractions import Fraction

    rng = np.random.default_rng(251001)
    for columns in (1, 4, 7, 19, 64):
        for spread in (0, 100, 500):
            theta = np.ldexp(rng.normal(size=columns), rng.integers(-spread, spread+1, columns))
            features = np.ldexp(rng.normal(size=(20, columns)),
                                rng.integers(-spread, spread+1, (20, columns)))
            observed = rng.normal(size=20)
            observed[:10] = features[:10] @ theta
            for x, y in zip(features, observed):
                expected = Fraction(float(y))-sum(Fraction(float(a))*Fraction(float(b))
                                                  for a, b in zip(theta, x))
                parts = np.empty(3)
                assert backend.probe_residual(theta.ctypes.data, x.ctypes.data,
                                              columns, y, parts.ctypes.data) == 0
                assert abs(sum(map(Fraction, parts))-expected) <= max(
                    abs(expected)*Fraction("2e-30"), Fraction(2)**-1075)


def conditional_absolute_moment(rho, lam):
    """Integrate the Normal absolute moment against its Gamma mixing law."""
    rho, lam = mp.mpf(rho), mp.mpf(lam)
    theta = mp.sqrt(2) * rho / mp.sqrt(1 - rho * rho)
    scale = mp.sqrt(lam * (1 + theta * theta))

    def integrand(w):
        mean = theta * (w - lam)
        absolute = mp.sqrt(2 * w / mp.pi) * mp.exp(-mean * mean / (2 * w))
        absolute += mean * mp.erf(mean / mp.sqrt(2 * w))
        return absolute / scale * mp.exp((lam - 1) * mp.log(w) - w - mp.loggamma(lam))

    return mp.quad(integrand, [0, lam, 2 * lam, mp.inf])


@pytest.mark.parametrize(
    "operation,values",
    [
        (0, [-700, -10, -0.1, 0, 0.1, 1, 10, 700]),
        (1, [1e-100, 0.1, 0.999999999999, 1, 2, 1e100]),
        (2, [1e-100, 0.1, 2, 1e100]),
        (3, [-12, -2, -0.01, 0, 0.01, 2, 12]),
        (4, [0.01, 0.5, 1, 2, 3.6453348340489273, 100]),
    ],
)
def test_scalar_functions_match_multiprecision(backend, operation, values):
    function = (mp.exp, mp.log, mp.sqrt, mp.erf, mp.loggamma)[operation]
    with mp.workdps(100):
        for value in values:
            parts = np.empty(3)
            assert backend.probe_portable_math(operation, value, parts.ctypes.data) == 0
            expected = function(mp.mpf(value))
            assert abs(mp.fsum(map(mp.mpf, parts)) - expected) <= max(
                abs(expected), 1
            ) * mp.mpf("2e-30")


def test_wide_erf_branch_boundaries_preserve_both_limbs(backend):
    boundaries = np.array([0., .5, 1., 1.5, 2.25, 3., 3.5, 5.5, 8., 110.])
    values = np.r_[boundaries, np.nextafter(boundaries, -np.inf),
                   np.nextafter(boundaries, np.inf),
                   np.random.default_rng(829).uniform(-9, 9, 160)]
    values = np.r_[values, -values]
    with mp.workdps(100):
        for high in values:
            for fraction in (-.25, .25):
                point = np.array([high, fraction*np.spacing(high)])
                parts = np.empty(3)
                assert backend.probe_wide_erf(point.ctypes.data, parts.ctypes.data) == 0
                expected = mp.erf(mp.fsum(map(mp.mpf, point)))
                error = abs(mp.fsum(map(mp.mpf, parts))-expected)
                assert error <= max(abs(expected)*mp.mpf("2e-30"), mp.mpf(2)**-1075)


def test_compensated_exp_polynomial_roundoff_bound():
    """Bound the five steps, not the table, reduction or final multiplication."""
    from fractions import Fraction as Q
    from math import factorial

    u, eta, radius = Q(1, 2**53), Q(1, 2**1074), Q.from_float(.00136)
    argument_low = u*radius+eta
    tail = (1+u)/factorial(9)
    for power in (8, 7, 6, 5):
        tail = (1+u)*(radius*tail+(1+u)/factorial(power))+eta
    assert tail < Q(1, 119)
    high, low, error = Q(1, 119), Q(0), Q(0)
    for power in (4, 3, 2, 1, 0):
        coefficient = Q(1, factorial(power))
        c_high = coefficient if power <= 2 else (1+u)*coefficient
        c_low = Q(0) if power <= 2 else u*c_high
        product = (1+u)*radius*high+eta
        assert product < (1-u)*coefficient  # FastTwoSum dominance.
        product_error = u*radius*high+eta
        sum_error = u*(c_high+product)+eta
        residual = (1+u)*(product_error+sum_error)+eta
        low_argument = (1+u)*(argument_low*high+c_low)+eta
        combined = (1+u)*(residual+low_argument)+eta
        next_low = (1+u)*(radius*low+combined)+eta
        # Correction roundings, omitted low*low, and gradual underflow.
        local = u*(product_error+sum_error+argument_low*high+c_low
                   +residual+low_argument+radius*low+combined)+argument_low*low+5*eta
        error = (radius+argument_low)*error+local
        high, low = (1+u)*(c_high+product)+eta, next_low
    assert error < Q("3.73e-32")


def test_compensated_exp_polynomial_matches_exact_rational_arithmetic(backend):
    from fractions import Fraction as Q

    rng = np.random.default_rng(300934)
    high = np.r_[rng.uniform(-.00136, .00136, 256), -.00136, 0., .00136]
    coefficients, actual = np.empty((5, 2)), np.empty(2)
    for value in high:
        for fraction in (-.5, .5):
            r = np.array([value, fraction*np.spacing(value)])
            for tail in (-1/119, 0., 1/119):
                backend.probe_exp_polynomial(r.ctypes.data, tail, actual.ctypes.data,
                                             coefficients.ctypes.data)
                argument, exact = sum(map(Q, r)), Q(tail)
                for coefficient in coefficients[::-1]:
                    exact = sum(map(Q, coefficient))+argument*exact
                assert abs(sum(map(Q, actual))-exact) < Q("3.73e-32")


def test_wide_log_range_reduction_preserves_both_limbs(backend):
    rng = np.random.default_rng(735)
    high = np.r_[np.exp(rng.uniform(-50,50,512)), np.exp(rng.uniform(-670,670,128)),
                 np.geomspace(.1,1e5,128)]
    points = list(np.column_stack((high,np.spacing(high)*rng.uniform(-.49,.49,len(high)))))
    for boundary in (.5,1.,2.,np.ldexp(1.,-968),np.ldexp(1.,968)):
        for high in (np.nextafter(boundary,-np.inf),boundary,np.nextafter(boundary,np.inf)):
            points.extend([high,np.spacing(high)*offset] for offset in (-.49,0.,.49))
    with mp.workdps(100):
        for values in points:
            point,parts = np.asarray(values),np.empty(3)
            assert backend.probe_wide_log(point.ctypes.data,parts.ctypes.data)==0
            expected = mp.log(mp.fsum(map(mp.mpf,point)))
            assert abs(mp.fsum(map(mp.mpf,parts))-expected) <= mp.mpf("2e-30")*max(1,abs(expected))


def test_wide_log_near_one_keeps_relative_accuracy(backend):
    with mp.workdps(360):
        for exponent in (-54,-80,-100,-500,-1000):
            for sign in (-1.,1.):
                point,parts = np.array([1.,np.ldexp(sign,exponent)]),np.empty(3)
                assert backend.probe_wide_log(point.ctypes.data,parts.ctypes.data)==0
                expected = mp.log(mp.fsum(map(mp.mpf,point)))
                assert abs(mp.fsum(map(mp.mpf,parts))-expected) <= mp.mpf("2e-30")*abs(expected)


@pytest.mark.parametrize("value,expected", [(0.,-np.inf),(-0.,-np.inf),(-1.,np.nan),
                                           (-np.inf,np.nan),(np.inf,np.inf),(np.nan,np.nan)])
def test_wide_log_preserves_domain_failures(backend,value,expected):
    point,parts = np.array([value,0.]),np.empty(3)
    assert backend.probe_wide_log(point.ctypes.data,parts.ctypes.data)==-3
    np.testing.assert_equal(parts[0],expected)


def test_wide_exp_table_boundaries_preserve_both_limbs(backend):
    indices = np.r_[np.arange(-512, 513), np.arange(-23630, 23631, 37)]
    boundaries = (indices+.5)*np.log(2)/256
    values = np.r_[boundaries, np.nextafter(boundaries, -np.inf),
                   np.nextafter(boundaries, np.inf), [-64., 64., -700., 700.]]
    with mp.workdps(100):
        for high in values:
            for fraction in (-.25, .25):
                point = np.array([high, fraction*np.spacing(high)])
                parts = np.empty(3)
                assert backend.probe_wide_exp(point.ctypes.data, parts.ctypes.data) == 0
                expected = mp.exp(mp.fsum(map(mp.mpf, point)))
                assert abs(mp.fsum(map(mp.mpf, parts))-expected) <= max(
                    abs(expected)*mp.mpf("2e-30"), mp.mpf(2)**-1075)


@pytest.mark.parametrize("operation", range(4))
def test_two_limb_arithmetic_matches_multiprecision(backend, operation):
    rng = np.random.default_rng(2184)
    cases = [
        ([1.0, 2.0**-53], [-1.0, 2.0**-54]),
        ([1.0, 2.0**-53], [1.0, -(2.0**-54)]),
        ([2.0**-450, 2.0**-504], [2.0**450, -(2.0**396)]),
        ([2.0**450, 2.0**396], [2.0**450, -(2.0**396)]),
    ]
    for _ in range(128):
        high = np.ldexp(rng.uniform(0.5, 1.0, 2), rng.integers(-450, 450, 2))
        high *= rng.choice([-1.0, 1.0], 2)
        low = np.spacing(high) * rng.uniform(-0.5, 0.5, 2)
        cases.append(([high[0], low[0]], [high[1], low[1]]))
    with mp.workdps(120):
        for left, right in cases:
            a, b = np.array(left), np.array(right)
            x, y = mp.fsum(map(mp.mpf, a)), mp.fsum(map(mp.mpf, b))
            expected = (lambda: x + y, lambda: x - y, lambda: x * y, lambda: x / y)[
                operation
            ]()
            parts = np.full(3, 31.0)
            assert (
                backend.probe_scalar_binary(
                    operation, a.ctypes.data, b.ctypes.data, parts.ctypes.data
                )
                == 0
            )
            actual = mp.fsum(map(mp.mpf, parts))
            assert abs(actual - expected) <= max(
                abs(expected) * mp.mpf("2e-30"), mp.mpf(2) ** -1075
            ), (operation, left, right, actual, expected)


def test_accurate_double_word_addition_cancellation_and_error_bound(backend):
    rng = np.random.default_rng(410214)
    high = np.ldexp(rng.uniform(-1, 1, (12000, 2)), rng.integers(-900, 900, (12000, 2)))
    high[3000:6000, 1] = -high[3000:6000, 0]
    high[6000:9000, 1] = np.nextafter(-high[6000:9000, 0], 0.)
    low = abs(np.spacing(high))*rng.uniform(-.49, .49, high.shape)
    cases = np.column_stack((high[:, 0], low[:, 0], high[:, 1], low[:, 1]))
    edges = [(a, al, b, bl)
             for a in (0., -0., 1., -1., 2.**-1022, 2.**-1074, 1e308)
             for b in (0., -0., 1., -1., -2.**-1022, -2.**-1074, 1e308, -1e308)
             for al, bl in ((0., 0.), (-0., -0.))]
    # Published counterexample to the older, too-small 2u^2 relative bound.
    counterexample = (2.**53-1, -(2.**53-1)/2.**54,
                     -(2.**53-5)/2, -(2.**53-1)/2.**56)
    inputs = np.ascontiguousarray(np.vstack((cases, edges, counterexample)))
    output = np.full((len(inputs)+2, 2), 271828.125)
    backend.probe_wide_add(inputs.ctypes.data, output[1:].ctypes.data, len(inputs))
    np.testing.assert_array_equal(output[[0, -1]], 271828.125)
    with mp.workprec(4096):
        unit = mp.mpf(2)**-53
        bound = 3*unit**2/(1-4*unit)
        for row, result in zip(inputs, output[1:-1]):
            exact = mp.fsum(map(mp.mpf, row))
            if not np.isfinite(result).all():
                assert abs(exact) > np.finfo(float).max
                # The established exceptional-range adapter may signal overflow
                # with NaN; either non-finite output invalidates the jet request.
                if np.isinf(result[0]):
                    assert np.sign(result[0]) == mp.sign(exact)
                continue
            actual = mp.fsum(map(mp.mpf, result))
            assert abs(actual-exact) <= max(abs(exact)*bound, mp.mpf(2)**-1075)
        exact = mp.fsum(map(mp.mpf, inputs[-1]))
        actual = mp.fsum(map(mp.mpf, output[-2]))
        assert abs((actual-exact)/exact) > 2*unit**2


@pytest.mark.parametrize("count", [0, 1, 3, 17])
def test_accurate_double_word_addition_preserves_output_guards(backend, count):
    inputs = np.tile([1., 2.**-54, -1., 2.**-55], (count, 1))
    output = np.full((count+2, 2), 271828.125)
    backend.probe_wide_add(inputs.ctypes.data, output[1:].ctypes.data, count)
    np.testing.assert_array_equal(output[[0, -1]], 271828.125)
    np.testing.assert_array_equal(output[1:-1, 0], 3*2.**-55)
    np.testing.assert_array_equal(output[1:-1, 1], 0.)


def test_exponential_range_reduction_boundaries(backend):
    boundaries = np.arange(-92, 93) * np.log(2.0)
    values = np.r_[
        boundaries,
        np.nextafter(boundaries, -np.inf),
        np.nextafter(boundaries, np.inf),
        np.random.default_rng(817).uniform(-64, 64, 256),
        [-64.0, np.nextafter(-64.0, 0.0), np.nextafter(64.0, 0.0), 64.0],
    ]
    with mp.workdps(100):
        for value in values:
            parts = np.full(3, 31.0)
            assert backend.probe_portable_math(0, value, parts.ctypes.data) == 0
            exact = mp.exp(mp.mpf(value))
            assert abs(mp.fsum(map(mp.mpf, parts)) / exact - 1) < mp.mpf("2e-30")


@pytest.mark.parametrize(
    "shape",
    [
        (0.0, 3.5),
        (-0.08590362405556805, 3.6453348340489273),
        (1e-100, 50.0),
        (np.nextafter(0.995, 0.0), 50.0),
    ],
)
def test_wide_moment_matches_independent_conditional_integral(backend, shape):
    with mp.workdps(70):
        expected = conditional_absolute_moment(*shape)
        parts = np.empty(3)
        assert backend.probe_moment(*shape, parts.ctypes.data) == 0
        assert abs(mp.fsum(map(mp.mpf, parts)) - expected) < mp.mpf("1e-25")
        reflected = np.empty(3)
        assert backend.probe_moment(-shape[0], shape[1], reflected.ctypes.data) == 0
        np.testing.assert_array_equal(parts, reflected)


@pytest.mark.parametrize(
    "operation,values",
    [
        (0, [-746.0, -745.0, -744.0, -710.0, -709.0, 709.0, 709.5, 709.78]),
        (1, [np.nextafter(0.0, 1.0), 1e-310, 1e-300, 1e300, np.finfo(float).max]),
        (2, [np.nextafter(0.0, 1.0), 1e-310, 1e-300, 1e300, np.finfo(float).max]),
    ],
)
def test_range_edges_preserve_representable_values(backend, operation, values):
    function = (mp.exp, mp.log, mp.sqrt)[operation]
    with mp.workdps(100):
        for value in values:
            parts = np.full(3, 31.0)
            assert backend.probe_portable_math(operation, value, parts.ctypes.data) == 0
            expected = function(mp.mpf(float(value)))
            actual = mp.fsum(map(mp.mpf, parts))
            # DD has a binary64 exponent range; subnormal output has fewer bits.
            tolerance = max(abs(expected) * mp.mpf("2e-30"), mp.mpf(2) ** -1075)
            assert abs(actual - expected) <= tolerance, (
                operation,
                value,
                actual,
                expected,
            )


@pytest.mark.parametrize(
    "operation,value",
    [
        (0, 710.0),
        (0, np.inf),
        (0, np.nan),
        (1, 0.0),
        (1, -1.0),
        (2, -1.0),
        (4, 0.0),
        (99, 1.0),
    ],
)
def test_invalid_scalar_requests_are_quiet_and_preserve_outputs(
    backend, capfd, operation, value
):
    parts = np.full(3, 31.0)
    assert backend.probe_portable_math(operation, value, parts.ctypes.data) < 0
    np.testing.assert_array_equal(parts, 31.0)
    captured = capfd.readouterr()
    assert not captured.out and not captured.err


@pytest.mark.parametrize(
    "a,b,c",
    [
        (0.0, 2.5, 1e-300),
        (5.0, 0.0, -1e-250),
        (0.0, np.finfo(float).max, -0.0),
        (1.0 + 2.0**-27, 1.0 - 2.0**-27, -1.0),
        (1e-160, 1e-160, 1e-310),
        (1e-200, 1e-200, 1e-310),
        (1e200, 1e200, -1e300),
        (1e308, 2.0, -1e308),
    ],
)
def test_fused_product_handles_cancellation_and_range_edges(backend, a, b, c):
    parts = np.full(3, 31.0)
    with mp.workdps(120):
        expected = mp.mpf(a) * mp.mpf(b) + mp.mpf(c)
        status = backend.probe_scalar_fma(a, b, c, parts.ctypes.data)
        if abs(expected) > np.finfo(float).max:
            assert status == -3
            np.testing.assert_array_equal(parts, 31.0)
        else:
            assert status == 0
            actual = mp.fsum(map(mp.mpf, parts))
            tolerance = max(abs(expected) * mp.mpf("2e-30"), mp.mpf(2) ** -1075)
            assert abs(actual - expected) <= tolerance
