"""Shared positive-real polygamma and log-gamma AD regressions against MP80."""

from pathlib import Path
import math
import shutil
import subprocess
import sys

import pytest


@pytest.fixture(scope="module", params=("native", "legacy", "legacy-c"))
def polygamma_executable(request, tmp_path_factory):
    is_c = request.param == "legacy-c"
    compiler = shutil.which("clang" if is_c else "clang++")
    compiler = compiler or shutil.which("gcc" if is_c else "g++")
    if compiler is None:
        pytest.skip("A C/C++ compiler is required.")
    root = Path(__file__).resolve().parents[1]
    executable = tmp_path_factory.mktemp("polygamma-" + request.param) / "probe"
    flags = ["-x", "c", "-std=c11"] if is_c else ["-std=c++20", "-fno-exceptions", "-fno-rtti"]
    if request.param != "native":
        flags += ["-ffast-math"]
    frame_flags = (["-Wframe-larger-than=16384", "-Werror=frame-larger-than"]
        if "clang" in Path(compiler).name else ["-Werror=frame-larger-than=16384"])
    subprocess.run([
        compiler, *flags, "-O3", "-ffp-contract=fast", "-funroll-loops", "-march=native",
        *frame_flags,
        "-I", str(root / "scivol/_csrc"), "-I", str(root / "scivol/_native"),
        str(root / "tests/native_polygamma.cpp"), "-o", str(executable), "-lm",
    ], check=True, capture_output=True, timeout=60)
    return executable, is_c


def evaluate(executable, mode, *points):
    output = subprocess.run([str(executable), mode, *map(str, points)],
        check=True, capture_output=True, text=True, timeout=10).stdout
    return [list(map(float, line.split())) for line in output.splitlines()]


def assert_close(actual, expected, *, rtol=3e-15, atol=0.0):
    expected = float(expected)
    if math.isinf(expected):
        assert actual == expected
    elif math.isnan(expected):
        assert math.isnan(actual)
    else:
        assert math.isfinite(actual)
        assert abs(actual - expected) <= atol + rtol * abs(expected), (actual, expected)


def test_positive_real_values_mp80(polygamma_executable):
    executable, _ = polygamma_executable
    mp = pytest.importorskip("mpmath")
    with mp.workdps(80):
        root = float(mp.findroot(mp.digamma, 1.46))
        points = [math.ulp(0.0), sys.float_info.min, 1e-309, 1e-308,
            1e-200, 1e-155, 1e-154, 1e-100, 1e-20, 1e-8, .01, .1,
            1 / 1.4, 3 / 1.4, .5, 1., 1.5, 2., 6., 8., 10., 128.,
            1e5, 1e50, 1e150, 1e300, sys.float_info.max,
            math.nextafter(root, -math.inf), root, math.nextafter(root, math.inf)]
        points += [k / 16 for k in range(1, 321)]
        points += [math.nextafter(k, direction)
            for k in range(1, 12) for direction in (-math.inf, math.inf)]
        for x, row in zip(points, evaluate(executable, "--scalar", *points)):
            assert_close(row[0], mp.digamma(mp.mpf(x)), atol=2e-15)
            assert_close(row[1], mp.polygamma(1, mp.mpf(x)))


def test_positive_domain_and_nonfinite_semantics(polygamma_executable):
    executable, _ = polygamma_executable
    points = [0., -0., -1., -math.inf, math.nan, math.inf]
    rows = evaluate(executable, "--scalar", *points)
    for row in rows[:-1]:
        assert all(math.isnan(value) for value in row)
    assert rows[-1] == [math.inf, 0.]


def test_lgamma_request_specific_overflow(polygamma_executable):
    executable, is_c = polygamma_executable
    if is_c:
        pytest.skip("The legacy C mode has no C++ differential consumer.")
    mp = pytest.importorskip("mpmath")
    with mp.workdps(80):
        for point in [1e-200, 1e-155, 1e-154]:
            first = evaluate(executable, "--first", point)[0]
            second = evaluate(executable, "--second", point)[0]
            assert first == second[:3]
            x = mp.mpf(point)
            assert_close(first[1], mp.loggamma(x))
            assert_close(first[2], mp.digamma(x))
            assert_close(second[3], mp.polygamma(1, x))


def test_lgamma_derivatives_and_coordinate_curvature_mp80(polygamma_executable):
    executable, is_c = polygamma_executable
    if is_c:
        pytest.skip("The legacy C mode has no C++ differential consumer.")
    mp = pytest.importorskip("mpmath")
    with mp.workdps(80):
        root = float(mp.findroot(mp.digamma, 1.46))
        for point in [1e-8, .1, .5, 1 / 1.4, root, 3 / 1.4, 6., 8., 10., 128., 1e8]:
            first = evaluate(executable, "--first", point)[0]
            second = evaluate(executable, "--second", point)[0]
            assert first == second[:3]
            x = mp.mpf(second[0])
            for actual, expected in zip(second[1:],
                    [mp.loggamma(x), mp.diff(mp.loggamma, x), mp.diff(mp.loggamma, x, 2)]):
                assert_close(actual, expected, atol=2e-15)
            chain = evaluate(executable, "--chain", point)[0]
            x = mp.mpf(chain[0])
            expected = [mp.loggamma(x), x * mp.digamma(x),
                x * mp.digamma(x) + x**2 * mp.polygamma(1, x)]
            for actual, reference in zip(chain[1:], expected):
                assert_close(actual, reference, atol=4e-15)


@pytest.mark.parametrize("shape", [.3, 1.4, 2., 10., 50.])
def test_ged_preparation_and_local_shape_derivatives_mp80(polygamma_executable, shape):
    executable, is_c = polygamma_executable
    if is_c:
        pytest.skip("The legacy C mode has no C++ differential consumer.")
    mp = pytest.importorskip("mpmath")
    residual, variance = 1.1, .7
    actual = evaluate(executable, "--ged", shape, residual, variance)[0]
    with mp.workdps(80):
        def scale(nu):
            return (mp.loggamma(1 / nu) - mp.loggamma(3 / nu)) / 2

        def constant(nu):
            return mp.log(2) + scale(nu) + mp.loggamma(1 / nu) - mp.log(nu)

        def local(nu):
            return constant(nu) + mp.log(mp.mpf(variance)) / 2 + mp.exp(nu * (
                mp.log(mp.mpf(residual)) - scale(nu) - mp.log(mp.mpf(variance)) / 2))

        expected = [mp.diff(function, mp.mpf(shape), order)
            for function in (scale, constant, local) for order in range(3)]
        for value, reference in zip(actual, expected):
            assert_close(value, reference, rtol=2e-14, atol=3e-14)
