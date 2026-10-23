"""Scaled sigmoid range, independent multiprecision AD, and binary ownership.

The oracle differentiates a scalar expression with second-order forward AD.
No candidate derivative formula, finite difference, or installed core is used.
"""

from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import re
import signal
import struct
import subprocess
import sys
import time
from types import SimpleNamespace

import mpmath as mp
import pytest


ROOT = Path(__file__).resolve().parents[3]
SOURCE = ROOT / "tests/native/transfer_006_scaled_logistic.cpp"
HEADER = ROOT / "core/src/math/power/scaled_logistic.hpp"
MINIMUM = math.ulp(0.)
MAXIMUM = sys.float_info.max
RELATIVE_TOLERANCE = 3e-14
WEIGHTS = (
    (1., 1.), (8., 1.), (-8., .5), (8., -.5), (-8., -.5),
    (1e308, 1e308), (-1e308, 1e308), (MAXIMUM, MAXIMUM),
    (1e-308, 1e308), (1e-308, 1e-308), (MINIMUM, 1e308),
    (MINIMUM, MAXIMUM), (1e-200, 1e-200), (0., MAXIMUM), (MAXIMUM, -0.),
)
COORDINATES = [0., -0., MINIMUM, -MINIMUM, 1e-320, -1e-320, 1e-18, -1e-18]
for magnitude in (.01, .249, .25, 1., 2., 4., 8., 36., 700., 708., 709.,
                  744., 746., 1000., 2000., 3000.):
    COORDINATES.extend((magnitude, -magnitude))
for magnitude in (.25, 708., 744., 746., 1000., 2000.):
    for neighbor in (math.nextafter(magnitude, 0.), math.nextafter(magnitude, math.inf)):
        COORDINATES.extend((neighbor, -neighbor))


@dataclass(frozen=True)
class Dual2:
    value: object
    first: object = 0
    second: object = 0

    @staticmethod
    def lift(other):
        return other if isinstance(other, Dual2) else Dual2(mp.mpf(other))

    def __add__(self, other):
        other = self.lift(other)
        return Dual2(self.value + other.value, self.first + other.first,
                     self.second + other.second)

    __radd__ = __add__

    def __neg__(self):
        return Dual2(-self.value, -self.first, -self.second)

    def __mul__(self, other):
        other = self.lift(other)
        return Dual2(self.value * other.value,
                     self.first * other.value + self.value * other.first,
                     self.second * other.value + 2 * self.first * other.first
                     + self.value * other.second)

    __rmul__ = __mul__

    def reciprocal(self):
        inverse = 1 / self.value
        return Dual2(inverse, -self.first * inverse**2,
                     2 * self.first**2 * inverse**3 - self.second * inverse**2)

    def exp(self):
        value = mp.exp(self.value)
        return Dual2(value, value * self.first, value * (self.second + self.first**2))


def oracle(z, a, b, digits):
    with mp.workdps(digits):
        variable = Dual2(mp.mpf(z), mp.mpf(1), mp.mpf(0))
        scale = mp.mpf(a) * mp.mpf(b)
        sigmoid = scale * (1 + (-variable).exp()).reciprocal()
        complement = scale * (1 + variable.exp()).reciprocal()
        return tuple(float(value) for value in
                     (sigmoid.value, sigmoid.first, sigmoid.second, complement.value))


def precision_pair(z):
    # Resolving curvature at minimum subnormal z requires over 324 digits.
    return (400, 460) if 0 < abs(z) < 1e-10 else (100, 160)


@pytest.fixture(scope="module", autouse=True)
def bounded_module():
    started = time.monotonic()
    previous = signal.getsignal(signal.SIGALRM)

    def expired(*_):
        raise TimeoutError("scaled logistic standalone shard exceeded 110 seconds")

    signal.signal(signal.SIGALRM, expired)
    signal.setitimer(signal.ITIMER_REAL, 110)
    try:
        yield SimpleNamespace(started=started, deadline=started + 110)
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous)


def run(command, bounded, **kwargs):
    result = subprocess.run(command, capture_output=True, text=True,
                            timeout=max(.1, bounded.deadline - time.monotonic()), **kwargs)
    assert result.returncode == 0, result.stdout + result.stderr
    return result.stdout


def snapshot_sources(directory):
    pending = [SOURCE, Path(__file__).resolve(), HEADER]
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
        if path.suffix not in (".hpp", ".h", ".cpp"):
            continue
        for include in re.findall(r'^\s*#include\s+"([^"]+)"', content.decode(), re.M):
            candidates = (path.parent / include, HEADER.parent / include, ROOT / "core/src" / include)
            owner = next((candidate for candidate in candidates if candidate.exists()), None)
            assert owner is not None, f"Unresolved production include: {include}"
            pending.append(owner)
    return hashes


@pytest.fixture(scope="module")
def probe(bounded_module, tmp_path_factory):
    assert HEADER.is_file(), "The production owner must supply scaled_logistic.hpp before this shard runs"
    compiler = "/usr/bin/clang++"
    if not Path(compiler).is_file():
        pytest.fail("The configured /usr/bin/clang++ driver is required")
    retained = os.environ.get("SCIVOL_SCALED_LOGISTIC_EVIDENCE_DIR")
    if retained:
        directory = Path(retained).expanduser().resolve() / f"tracked-{os.getpid()}-{time.time_ns()}"
        directory.mkdir(parents=True)
    else:
        directory = tmp_path_factory.mktemp("scaled_logistic")
    hashes = snapshot_sources(directory)
    source = directory / SOURCE.relative_to(ROOT)
    executable, object_file = directory / "probe", directory / "probe.o"
    flags = ["-std=c++20", "-O3", "-ffp-contract=fast", "-funroll-loops", "-march=native",
             "-fno-exceptions", "-fno-rtti", "-Wframe-larger-than=16384",
             "-Werror=frame-larger-than", "-fstack-usage"]
    commands = [
        [compiler, *flags, "-I", str(directory / "core/src"),
         "-c", str(source), "-o", str(object_file)],
        [compiler, *flags, str(object_file), "-o", str(executable)],
    ]
    for command in commands:
        run(command, bounded_module, cwd=directory)
    frames = []
    for path in directory.glob("*.su"):
        for line in path.read_text().splitlines():
            name, size, storage = line.split("\t")
            assert storage == "static" and int(size) <= 16384, line
            frames.append(dict(symbol=name, bytes=int(size), storage=storage))
    assert frames, "The configured compiler did not emit stack-usage evidence"
    undefined = run(["nm", "-u", str(executable)], bounded_module)
    if sys.platform == "darwin":
        imports = run(["otool", "-L", str(executable)], bounded_module)
        assembly = run(["otool", "-tvV", str(executable)], bounded_module)
    else:
        imports = run(["ldd", str(executable)], bounded_module)
        assembly = run(["objdump", "-d", str(executable)], bounded_module)
    (directory / "imports.txt").write_text(imports + "\n" + undefined)
    (directory / "assembly.txt").write_text(assembly)
    assert not re.search(r"scivol|slsqp|blas|lapack|tlfloat|boost|gfortran", undefined + imports,
                         re.I), "Unexpected numerical or solver backend in standalone probe"
    record = dict(source_hashes=hashes, commands=commands, stack_frames=frames,
                  compiler=run([compiler, "--version"], bounded_module),
                  executable_sha256=hashlib.sha256(executable.read_bytes()).hexdigest(),
                  imports=imports, undefined_symbols=undefined,
                  scope="Standalone scaled logistic only; no APARCH consumer or core integration")
    (directory / "build.json").write_text(json.dumps(record, indent=2) + "\n")
    result = SimpleNamespace(directory=directory, executable=executable,
                             bounded=bounded_module, hashes=hashes, evidence=[])
    try:
        yield result
    finally:
        changed = [path for path, digest in hashes.items()
                   if hashlib.sha256((ROOT / path).read_bytes()).hexdigest() != digest]
        (directory / "numerical_evidence.json").write_text(json.dumps(dict(
            cases=result.evidence, sources_changed=changed,
            seconds=time.monotonic() - bounded_module.started), indent=2) + "\n")
        assert not changed, f"Production/test owner changed during this shard: {changed}"


def invoke(probe, points):
    text = "".join(" ".join(float(value).hex() for value in point) + "\n" for point in points)
    output = run([str(probe.executable)], probe.bounded, input=text)
    rows = [tuple(float.fromhex(token) for token in line.split()) for line in output.splitlines()]
    assert len(rows) == len(points) and all(len(row) == 13 for row in rows)
    return rows


def assert_matches(actual, expected, point, output):
    assert not math.isnan(actual), (point, output, actual, expected)
    if not math.isfinite(expected):
        assert actual == expected, (point, output, actual, expected)
    else:
        assert math.isfinite(actual), (point, output, actual, expected)
        # This intentionally becomes exact equality for tiny subnormal targets.
        assert abs(actual - expected) <= RELATIVE_TOLERANCE * abs(expected), (
            point, output, actual.hex(), expected.hex())


@pytest.mark.parametrize("z", COORDINATES, ids=lambda z: z.hex())
def test_scaled_value_gradient_hessian_and_complement_against_ad(probe, z):
    points = [(z, a, b) for a, b in WEIGHTS]
    low, high = precision_pair(z)
    for point, row in zip(points, invoke(probe, points)):
        expected = oracle(*point, low)
        refined = oracle(*point, high)
        assert expected == refined, (point, low, high, expected, refined)
        for output, actual, target in zip(("V", "G", "H", "complement"), row[:4], refined):
            assert_matches(actual, target, point, output)
        probe.evidence.append(dict(point=[value.hex() for value in point],
                                   actual=[value.hex() for value in row[:4]],
                                   expected=[value.hex() for value in refined],
                                   oracle_digits=[low, high]))






def test_held_row_has_twelve_not_sixteen_subnormal_units(probe):
    row = invoke(probe, [(-744., 8., 1.)])[0]
    expected = 12 * MINIMUM
    assert oracle(-744., 8., 1., 160)[:3] == (expected,) * 3
    assert row[:3] == (expected,) * 3
    assert row[4:7] == (16 * MINIMUM,) * 3


def test_center_curvature_survives_subnormal_coordinate(probe):
    points = [(z, 1e308, 1e308) for z in (MINIMUM, -MINIMUM, 1e-320, -1e-320)]
    points.extend((z, 1e150, 1e150) for z in (1e-18, -1e-18))
    for point, row in zip(points, invoke(probe, points)):
        expected = oracle(*point, 460)[2]
        assert expected != 0. and math.isfinite(expected)
        assert_matches(row[2], expected, point, "H")
        assert math.copysign(1., row[2]) == -math.copysign(1., point[0])


def test_exact_zero_curvature_with_overflowing_weight_product(probe):
    points = [(z, a, b) for z in (0., -0.)
              for a, b in ((MAXIMUM, MAXIMUM), (-MAXIMUM, MAXIMUM), (1e308, 1e308))]
    for row in invoke(probe, points):
        assert row[2] == 0. and math.isfinite(row[2])
        assert not math.isfinite(row[0]) and not math.isfinite(row[1])


def test_overflowing_product_recovers_finite_outputs(probe):
    points = [(-1000., MAXIMUM, MAXIMUM), (-2000., MAXIMUM, MAXIMUM),
              (-744., 1e308, 1e308), (1000., MAXIMUM, MAXIMUM)]
    for point, row in zip(points, invoke(probe, points)):
        assert math.isinf(point[1] * point[2])
        targets = oracle(*point, 160)
        for index in ((1, 2) if point[0] > 0 else (0, 1, 2)):
            assert math.isfinite(targets[index]) and targets[index] != 0.
            assert_matches(row[index], targets[index], point, index)


def test_zero_weights_and_signs(probe):
    points = [(z, a, b) for z in (-2000., -744., 0., 744., 2000.)
              for a, b in ((0., MAXIMUM), (MAXIMUM, 0.), (-0., -MAXIMUM))]
    for row in invoke(probe, points):
        assert row[:4] == (0.,) * 4
    points = [(z, a, b) for z in (-4., .01, 4., 1000.)
              for a, b in ((8., .5), (-8., .5), (8., -.5), (-8., -.5))]
    for point, row in zip(points, invoke(probe, points)):
        for index, expected in enumerate(oracle(*point, 160)):
            assert_matches(row[index], expected, point, index)
            if expected != 0.:
                assert math.copysign(1., row[index]) == math.copysign(1., expected)


def test_ordinary_value_and_gradient_preserve_legacy_bits(probe):
    points = [(z, a, 1.) for z in (-8., -4., -2., -1., -.01, 0., .01, 1., 2., 4., 8.)
              for a in (1., 8.)]
    for point, row in zip(points, invoke(probe, points)):
        for new, old in zip(row[:2], row[4:6]):
            assert struct.pack("=d", new) == struct.pack("=d", old), point
        # The center cancellation fix intentionally changes legacy H arithmetic.
        if abs(point[0]) >= .25:
            assert struct.pack("=d", row[2]) == struct.pack("=d", row[6]), point


def test_scalar_ad_oracle_has_known_derivatives():
    with mp.workdps(80):
        x = Dual2(mp.mpf(2), mp.mpf(1), mp.mpf(0))
        polynomial = x * x * x
        assert (polynomial.value, polynomial.first, polynomial.second) == (8, 12, 12)
        inverse = x.reciprocal()
        assert (inverse.value, inverse.first, inverse.second) == (.5, -.25, .25)
        exponential = x.exp()
        assert exponential.value == exponential.first == exponential.second == mp.exp(2)
    assert oracle(0., 8., 1., 160) == (4., 2., 0., 4.)


def test_standalone_owner_and_stack_evidence(probe):
    record = json.loads((probe.directory / "build.json").read_text())
    assert "core/src/math/power/scaled_logistic.hpp" in record["source_hashes"]
    assert max(frame["bytes"] for frame in record["stack_frames"]) <= 16384
    assert all(command[0] == "/usr/bin/clang++" for command in record["commands"])
    for relative, digest in record["source_hashes"].items():
        assert hashlib.sha256((ROOT / relative).read_bytes()).hexdigest() == digest, relative
