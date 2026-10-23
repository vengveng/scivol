"""Source-only range recovery with independent multiprecision forward AD."""

import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess
import time

import mpmath as mp
import numpy as np
import pytest

from test_native_anchored_mean_chart import Jet


ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / "localdev/jet_math_recovery_20261013/density_range_followup"
SENTINEL = 12345.0


@pytest.fixture(scope="module")
def source_rows():
    compiler = shutil.which("clang++") or shutil.which("g++")
    if compiler is None:
        pytest.skip("A C++20 compiler is required")
    deadline = time.monotonic() + 110
    directory = EVIDENCE / f"source-{time.time_ns()}"
    directory.mkdir(parents=True)
    pending = [ROOT / "tests/native_jet_math_density_range.cpp"]
    hashes = {}
    while pending:
        path = pending.pop().resolve()
        relative = str(path.relative_to(ROOT))
        if relative in hashes:
            continue
        content = path.read_bytes()
        hashes[relative] = hashlib.sha256(content).hexdigest()
        target = directory / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(content)
        for name in re.findall(r'^\s*#include\s+"([^"]+)"', content.decode(), re.M):
            candidate = next((base / name for base in
                              (path.parent, ROOT / "scivol/_native", ROOT / "scivol/_csrc")
                              if (base / name).exists()), None)
            assert candidate is not None, name
            pending.append(candidate)
    frame = (["-Wframe-larger-than=16384", "-Werror=frame-larger-than"]
             if "clang" in Path(compiler).name else ["-Werror=frame-larger-than=16384"])
    executable = directory / "density_range"
    command = [compiler, "-std=c++20", "-O3", "-ffp-contract=fast", "-funroll-loops",
               "-march=native", "-fno-exceptions", "-fno-rtti", *frame,
               "-I", str(directory / "scivol/_native"),
               "-I", str(directory / "scivol/_csrc"),
               "-I", str(ROOT / "scivol/_native/third_party/boost"),
               "-I", str(ROOT / "scivol/_native/third_party/tlfloat"),
               str(directory / "tests/native_jet_math_density_range.cpp"), "-o", str(executable)]
    result = subprocess.run(command, capture_output=True, text=True,
                            timeout=max(1, deadline - time.monotonic()))
    identity = dict(command=command, returncode=result.returncode, stderr=result.stderr,
                    source_hashes=hashes)
    (directory / "build.json").write_text(json.dumps(identity, indent=2) + "\n")
    assert result.returncode == 0, result.stderr
    completed = subprocess.run([str(executable)], capture_output=True, text=True,
                               timeout=max(1, deadline - time.monotonic()))
    assert completed.returncode == 0, completed.stdout + completed.stderr
    rows = [json.loads(line) for line in completed.stdout.splitlines()]
    identity["binary_sha256"] = hashlib.sha256(executable.read_bytes()).hexdigest()
    identity["python_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    identity["elapsed_seconds"] = 110 - (deadline - time.monotonic())
    (directory / "build.json").write_text(json.dumps(identity, indent=2) + "\n")
    (directory / "rows.json").write_text(json.dumps(rows, indent=2) + "\n")
    changed = [name for name, digest in hashes.items()
               if hashlib.sha256((ROOT / name).read_bytes()).hexdigest() != digest]
    assert not changed, f"Sources changed during bounded compilation: {changed}"
    return rows


def unary(point, value, first, second):
    width = len(point.gradient)
    return Jet(value, tuple(first * v for v in point.gradient), tuple(
        first * point.hessian[i * width + j] + second * point.gradient[i] * point.gradient[j]
        for i in range(width) for j in range(width)))


def log(point):
    return unary(point, mp.log(point.value), 1 / point.value, -1 / point.value**2)


def lgamma(point):
    return unary(point, mp.loggamma(point.value), mp.digamma(point.value),
                 mp.polygamma(1, point.value))


def ged_objective(mean, variance, shape):
    inverse = shape.reciprocal()
    log_scale = (lgamma(inverse) - lgamma(3 * inverse)) / 2
    residual = float(1e-252) - mean * float(1e100)
    exponent = shape * (log(residual) - log(variance) / 2 - log_scale)
    power = mp.exp(exponent.value)
    return (-log(shape) + mp.log(2) + log_scale + lgamma(inverse)
            + log(variance) / 2 + unary(exponent, power, power, power))


def oracle(row, digits):
    with mp.workdps(digits):
        kind = row["kind"]
        if kind == "ged_constant":
            theta = [0., 1e-152, 4.]
        elif kind == "ged_joint":
            theta = [0., 1., .1, .1, 4.]
        elif kind == "normal_arma":
            theta = [0., 0., 0., 5e307, .1, .5]
        else:
            theta = [0., 5e307, .1] + [0.] * (row["p"] - 1) + [.5] + [0.] * (row["q"] - 1)
        width = len(theta)
        point = [Jet.variable(value, width, i) for i, value in enumerate(theta)]
        if kind.startswith("ged"):
            variance = point[1] if kind == "ged_constant" else Jet.constant(float(1e-152), width)
            objective = ged_objective(point[0], variance, point[-1])
        else:
            arma = kind == "normal_arma"
            omega = 3 if arma else 1
            beta = 5 if arma else 2 + row["p"]
            variance = point[omega] + point[beta] * float(1e308)
            residual = float(1e155) - point[0]
            objective = (mp.log(2 * mp.pi) + log(variance) + residual * residual / variance) / 2
            if not arma:
                objective += (mp.log(2 * mp.pi) + mp.log(float(1e308))) / 2
        return (float(objective.value), np.array(objective.gradient, dtype=float),
                np.array(objective.hessian, dtype=float).reshape(width, width))


def compare(row, expected):
    request = row["request"]
    value, gradient, hessian = expected
    if request & 1:
        np.testing.assert_allclose(row["value"], value, rtol=0., atol=2e-11)
    else:
        assert row["value"] == SENTINEL
    for bit, actual, reference in ((2, row["gradient"], gradient),
                                    (4, row["hessian"], hessian.ravel())):
        actual = np.array(actual, dtype=float)
        if request & bit:
            assert np.isfinite(actual).all()
            np.testing.assert_allclose(actual, reference, rtol=8e-12, atol=8e-323)
        else:
            np.testing.assert_array_equal(actual, SENTINEL)


@pytest.mark.parametrize("kind", ["ged_joint", "ged_constant", "normal_joint", "normal_arma"])
def test_complete_density_range_matches_multiprecision_ad(source_rows, kind):
    rows = [row for row in source_rows if row["kind"] == kind]
    assert rows
    for row in rows:
        assert row["valid"], row
        expected = oracle(row, 100)
        for first, second in zip(expected, oracle(row, 160)):
            np.testing.assert_array_equal(first, second)
        compare(row, expected)
        if kind.startswith("ged") and row["request"] & 4:
            np.testing.assert_allclose(row["hessian"][0], 1.3708397431333905, rtol=8e-13, atol=0.)


@pytest.mark.parametrize("kind", ["student_mean", "student_zero"])
def test_student_extreme_tail_value_only_is_finite(source_rows, kind):
    with mp.workdps(160):
        h, e, nu = map(mp.mpf, (float(1e-320), .1, 8.))
        constant = mp.loggamma((nu + 1) / 2) - mp.loggamma(nu / 2) - mp.log(mp.pi * (nu - 2)) / 2
        expected = float(mp.log(h) / 2 - constant + (nu + 1) / 2 * mp.log1p(e**2 / ((nu - 2) * h)))
        variance_gradient = (1 - (nu + 1) * e**2 / ((nu - 2) * h + e**2)) / (2 * h)
        assert abs(variance_gradient) > mp.mpf(np.finfo(float).max)
    for row in source_rows:
        if row["kind"] != kind:
            continue
        if row["request"] == 1:
            assert row["valid"]
            np.testing.assert_allclose(row["value"], expected, rtol=0., atol=2e-11)
            np.testing.assert_array_equal(row["gradient"], SENTINEL)
            np.testing.assert_array_equal(row["hessian"], SENTINEL)
        else:
            assert not row["valid"], row
            if not row["request"] & 1:
                assert row["value"] == SENTINEL
            if not row["request"] & 2:
                np.testing.assert_array_equal(row["gradient"], SENTINEL)
            if not row["request"] & 4:
                np.testing.assert_array_equal(row["hessian"], SENTINEL)


def test_normal_supplied_square_contract_is_unchanged(source_rows):
    row = next(row for row in source_rows if row["kind"] == "normal_supplied")
    assert not row["overflow_finite"]
    expected = .5 * (np.log(2 * np.pi) + np.log(3.) + 7. / 3.)
    np.testing.assert_allclose(row["value"], expected, rtol=0., atol=2e-15)


@pytest.mark.parametrize("kind", ["student_large_mean", "student_large_zero"])
def test_student_retains_finite_standardized_square_when_raw_square_overflows(source_rows, kind):
    references = []
    for digits in (100, 160):
        with mp.workdps(digits):
            theta = [0., 1e308, 8.] if kind.endswith("mean") else [1e308, 8.]
            width = len(theta)
            point = [Jet.variable(value, width, i) for i, value in enumerate(theta)]
            variance, nu = point[-2:]
            residual = float(1e155) - point[0] if width == 3 else Jet.constant(float(1e155), width)
            constant = lgamma((nu + 1) / 2) - lgamma(nu / 2) - log((nu - 2) * mp.pi) / 2
            objective = log(variance) / 2 - constant + (nu + 1) / 2 * log(
                1 + residual * residual / ((nu - 2) * variance))
            references.append((float(objective.value), np.array(objective.gradient, dtype=float),
                               np.array(objective.hessian, dtype=float).reshape(width, width)))
    for first, second in zip(*references):
        np.testing.assert_array_equal(first, second)
    rows = [row for row in source_rows if row["kind"] == kind]
    assert len(rows) == 14
    for row in rows:
        assert row["valid"], row
        compare(row, references[0])


def exp(point):
    value = mp.exp(point.value)
    return unary(point, value, value, value)


def sqrt(point):
    value = mp.sqrt(point.value)
    return unary(point, value, 1 / (2 * value), -1 / (4 * value * point.value))


def conditioned_oracle(row, digits):
    kind = row["kind"]
    normal = kind.startswith("normal_subnormal")
    skew = kind.startswith("skewt")
    garch = "garch" in kind
    large = kind == "student_raw_large"
    with mp.workdps(digits):
        volatility = [1., .1] + [0.] * (row["p"] - 1) + [.1] + [0.] * (row["q"] - 1)
        theta = ([] if garch else [0.]) + volatility + ([] if normal else [8.])
        if skew:
            theta += [.2 if kind.endswith(("negative", "positive")) else 0.]
        width = len(theta)
        point = [Jet.variable(value, width, i) for i, value in enumerate(theta)]
        variance = Jet.constant(float(1e308 if large else 1e-320), width)
        residual = Jet.constant(float(1e-160 if normal else 1e155 if large
                                      else -.1 if kind.endswith("negative") else .1), width)
        if not garch:
            residual -= point[0] * float(0. if kind.endswith("zero") else 1e-160 if normal else 1.)
        if normal:
            objective = (mp.log(2 * mp.pi) + log(variance) + residual * residual / variance) / 2
        else:
            nu = point[-2] if skew else point[-1]
            constant = lgamma((nu + 1) / 2) - lgamma(nu / 2) - log((nu - 2) * mp.pi) / 2
            innovation = residual / sqrt(variance)
            log_scale = Jet.constant(0., width)
            if skew:
                lam = point[-1]
                shift = 4 * lam * exp(constant) * (nu - 2) / (nu - 1)
                scale = sqrt(1 + 3 * lam * lam - shift * shift)
                centered = scale * innovation + shift
                innovation = centered / (1 + lam * mp.sign(centered.value))
                log_scale = log(scale)
            square = innovation * innovation
            if garch and not skew:
                # Standalone Student consumes the exact supplied double square.
                square = Jet.constant(float(.1 * .1), width) / variance
            objective = log(variance) / 2 - constant - log_scale + (nu + 1) / 2 * log(
                1 + square / (nu - 2))
        return (float(objective.value), np.array(objective.gradient, dtype=float),
                np.array(objective.hessian, dtype=float).reshape(width, width))


@pytest.mark.parametrize("kind", ["student_seed", "skewt_seed", "student_garch_seed",
                                 "skewt_garch_seed", "normal_subnormal", "normal_subnormal_zero",
                                 "student_raw_large", "skewt_seed_positive", "skewt_seed_negative"])
def test_fixed_conditioning_and_explicit_raw_intentions(source_rows, kind):
    rows = [row for row in source_rows if row["kind"] == kind]
    assert len(rows) == (14 if "garch" in kind or kind not in ("student_seed", "skewt_seed") else 28)
    for row in rows:
        assert row["valid"], row
        expected = conditioned_oracle(row, 400)
        for first, second in zip(expected, conditioned_oracle(row, 500)):
            np.testing.assert_array_equal(first, second)
        compare(row, expected)


@pytest.mark.parametrize("garch", [False, True])
def test_conditioned_student_equals_zero_skew_common_parameters(source_rows, garch):
    student_kind = "student_garch_seed" if garch else "student_seed"
    skew_kind = "skewt_garch_seed" if garch else "skewt_seed"
    for student in (row for row in source_rows if row["kind"] == student_kind):
        skew = next(row for row in source_rows if row["kind"] == skew_kind
                    and row["fixed"] == student["fixed"] and row["request"] == student["request"]
                    and row["p"] == student["p"] and row["q"] == student["q"])
        width = len(student["gradient"])
        if student["request"] & 1:
            np.testing.assert_allclose(student["value"], skew["value"], rtol=0., atol=2e-11)
        if student["request"] & 2:
            np.testing.assert_allclose(student["gradient"], skew["gradient"][:width], rtol=8e-12, atol=8e-323)
        if student["request"] & 4:
            expected = np.array(skew["hessian"]).reshape(width + 1, width + 1)[:width, :width]
            np.testing.assert_allclose(np.array(student["hessian"]).reshape(width, width), expected,
                                       rtol=8e-12, atol=8e-323)
