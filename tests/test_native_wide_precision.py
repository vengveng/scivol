"""Independent precision and portability checks for the production wide scalar."""

import ctypes
from pathlib import Path
import shutil
import subprocess
import sys

import mpmath as mp
import numpy as np
import pytest


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
        "probe_moment": [ctypes.c_double] * 2 + [ctypes.c_void_p],
    }.items():
        function = getattr(result, name)
        function.argtypes, function.restype = arguments, ctypes.c_int
    return result


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
