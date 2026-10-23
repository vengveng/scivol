"""Independent high-precision controls for frozen-unit derivative products."""

import math
from pathlib import Path
import random
import shutil
import subprocess

import mpmath as mp
import pytest


ROOT = Path(__file__).resolve().parents[3]


def contraction_cases():
    cases = [
        (1, 1e-100, -1e250, 1e-100),
        (1, 1e200, 1e-250, 1e100),
        (1, 1e-200, 1e-200, 1e-200),
        (1, 0., 1e250, 1e-250),
        (1, 1e-300, 1e-300, 1.),
        (1, 1e300, 1e300, 1.),
        (2, 1e-200, 1e250, 1e-100, 1e-200, 1e100),
        (2, 1e200, 1e-250, 1e100, 1e250, 1e100),
        (2, 0., 1e250, 1e-250, 1e250, 1e-250),
        (2, 1e300, 1e300, 1., 1e-300, 1.),
    ]
    generator = random.Random(8392)
    for factors in (1, 2):
        for _ in range(256):
            values = [math.ldexp(generator.uniform(.5, 1.), generator.randint(-1060, 1023))
                      for _ in range(1 + 2 * factors)]
            values[0] *= generator.choice((-1, 1))
            values[1] *= generator.choice((-1, 1))
            cases.append((factors, *values))
    return cases


@pytest.fixture(scope="module")
def contraction_outputs(tmp_path_factory):
    compiler = shutil.which("clang++") or shutil.which("g++")
    if compiler is None:
        pytest.fail("A C++20 compiler is required")
    executable = tmp_path_factory.mktemp("scaled_derivative") / "contraction"
    command = [
        compiler, "-std=c++20", "-O3", "-ffp-contract=fast", "-funroll-loops",
        "-march=native", "-fno-exceptions", "-fno-rtti",
        "-I", str(ROOT / "core/src"), "-I", str(ROOT / "core/src/math"),
        str(ROOT / "tests/native/transfer_006_scaled_derivative_product.cpp"), "-o", str(executable),
    ]
    compiled = subprocess.run(command, capture_output=True, text=True, timeout=30)
    assert compiled.returncode == 0, compiled.stderr
    cases = contraction_cases()
    inputs = "\n".join(" ".join(map(str, row)) for row in cases) + "\n"
    result = subprocess.run([str(executable)], input=inputs, capture_output=True,
                            text=True, timeout=10)
    assert result.returncode == 0, result.stderr
    outputs = [float(line) for line in result.stdout.splitlines()]
    assert len(outputs) == len(cases)
    return zip(cases, outputs)


def test_frozen_unit_products_preserve_representable_contractions(contraction_outputs):
    with mp.workdps(150):
        for case, actual in contraction_outputs:
            factors, coefficient, row, row_unit, *second = case
            expected = mp.mpf(coefficient) * mp.mpf(row) / mp.mpf(row_unit)
            if factors == 2:
                column, column_unit = second
                expected *= mp.mpf(column) / mp.mpf(column_unit)
            expected = float(expected)
            if math.isinf(expected):
                assert actual == expected, case
            else:
                assert math.isfinite(actual), case
                assert abs(actual - expected) <= 4 * math.ulp(expected), (case, actual, expected)
