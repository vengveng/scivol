"""Standalone bounded proposals, independent scores, arena and binary contracts."""

from decimal import Decimal, localcontext
from pathlib import Path
import json
import os
import re
import shutil
import subprocess

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
NATIVE = ROOT / "scivol" / "_native"
SOURCE = Path(__file__).with_name("native_bounded_direction.cpp")
EPSILON = np.finfo(float).eps
SENTINEL = 817.25

# Original decimal strings and source hashes from the 2026-10-10 direction report.
# These are frozen supplied G/H controls, not regenerated likelihood derivatives.
SAVED_BLOCKS = {
    "native": {
        "sha256": "d411015c4f984e103095218c89e691779b08415ab227485249984340ed92778b",
        "gradient": [
            "7868.7235011102543705887144881070880274782966753177769123745277337713704999916296",
            "-835.89619357720938477322071885614044047162434322658622522437395433207064214474764",
        ],
        "hessian": [
            ["-2470595791.1023193745923386833517202137404399090664230716610852017938304659221363",
             "-593293992.05601957590702417908458606419338572017671924217884333318746974794196479"],
            ["-593293992.05601957590702417908458606419338572017671924217884333318746974794196479",
             "-266349475.00181523704048057679716614936652185891784764525106947242857597961755299"],
        ],
        "stationary": [
            "0.0000084685909693245462938854702096842167184846368340304341900646728216314126797374018",
            "-0.000022002147129508653100417655853296217380719867761651498244631460465691707356367974",
        ],
    },
    "reference": {
        "sha256": "2b45880691877bdaf6c2cbe2e78a9cf8c32d9f04db3d7acced79b86a6b678562",
        "gradient": [
            "-25244.604770106271709303544421749082931791554527438409590525540982038110050782138",
            "-38812.863793562990966281833454586763710710171752228654807246707618229957797580646",
        ],
        "hessian": [
            ["16643811005232.151383718831905726742553954622723590926207183409641136790781712453",
             "3865302469386.3053543431188070397651040590974411533830910124974724521030828550562"],
            ["3865302469386.3053543431188070397651040590974411533830910124974724521030828550562",
             "1565539706742.1610155510519762796462401229839885478022909562756333205569360252512"],
        ],
        "stationary": [
            "-0.0000000099408229538992289348736890978213576918598341404983919057052286607649127150091103",
            "0.000000049335798365491370463623737108090126644299884587160617774132511755730703028973853",
        ],
    },
    "completed": {
        "sha256": "59cb2544df1e05bb8bc5fa5f2adde51433c825595a97c75c73a9d090badaeb24",
        "gradient": [
            "18295.532200221265988882680645486688382378623778871356030497104133008048551002766",
            "20011.568157280486934467987645740737692961546022049969198278912902420355922134101",
        ],
        "hessian": [
            ["-4528496533.1169552522129423372296141290801928519706540126610678394698772207430085",
             "-5855069022.9444860058146031461392919930989502294994784469504803169440239088310142"],
            ["-5855069022.9444860058146031461392919930989502294994784469504803169440239088310142",
             "-8928380235.0645428447569410542375371626808428813799579944704871186459163923384909"],
        ],
        "stationary": [
            "0.000007508627650481667316321124556418063458889033978280350591365203162810205072906893",
            "-0.000002682677526413167242072100444933023444695254788842177398134966173065149547777345",
        ],
    },
}


def precedent_cases():
    cases = []

    def add(name, g, h, radius, status=0, sweeps=64, method=2, fallback=0):
        cases.append(pytest.param(name, np.asarray(g, dtype=float),
            np.asarray(h, dtype=float), radius, status, sweeps, method, fallback,
            id=f"{name}-{radius}"))

    for name, block in SAVED_BLOCKS.items():
        for radius in (1e-8, 1e-6, 1e-5, 1.):
            add(name, block["gradient"], block["hessian"], radius)
    for name, g, h, radius, method, fallback in (
        ("spd", [1., -2.], [[4., 1.], [1., 3.]], 1., 2, 0),
        ("negative", [1., -2.], [[-4., -1.], [-1., -3.]], .1, 2, 0),
        ("indefinite", [2., 3.], [[-2., 0.], [0., 4.]], .25, 2, 0),
        ("singular", [1., 2.], [[0., 0.], [0., 4.]], .5, 3, 1),
        ("zero_hessian", [1., -2.], np.zeros((2, 2)), .5, 3, 1),
        ("unresolved", [1., 2.], [[1e-15, 0.], [0., 4.]], .5, 3, 1),
        ("hard_case", [0., 1.], [[-2., 0.], [0., 1.]], 1., 2, 0),
        ("zero_gradient_negative", [0., 0.], [[-2., 0.], [0., -1.]], 1., 1, 0),
        ("huge_curvature", [1., 2.], [[1e300, 0.], [0., -2e300]], .5, 2, 0),
        ("tiny_curvature", [1., 2.], [[1e-300, 0.], [0., -2e-300]], .5, 2, 0),
        ("huge_gradient", [1e300, -1e300], [[2., 0.], [0., 1.]], 1e-4, 2, 0),
    ):
        add(name, g, h, radius, method=method, fallback=fallback)

    rng = np.random.default_rng(421010)
    for n in (1, 2, 3, 5, 9, 13, 37):
        q, _ = np.linalg.qr(rng.normal(size=(n, n)))
        for kind in ("spd", "negative", "indefinite", "singular"):
            values = np.linspace(.25, 4., n)
            if kind == "negative":
                values *= -1
            if kind == "indefinite":
                values[::2] *= -1
            if kind == "singular":
                values[0] = 0
            h = (q*values) @ q.T
            h = (h+h.T)/2
            g = rng.normal(size=n)
            for radius in (.001, 10.):
                add(f"{kind}_{n}", g, h, radius,
                    method=3 if kind == "singular" else 2,
                    fallback=1 if kind == "singular" else 0)
    g = np.array([1., -2., 3., -4., 5.])
    h = rng.normal(size=(5, 5))
    add("eigen_sweep_cap", g, (h+h.T)/2, .5, sweeps=1, method=3, fallback=2)
    for name, g, h, radius, status, fallback in (
        ("nan_gradient", [np.nan, 1.], np.eye(2), 1., 1, 0),
        ("inf_gradient", [np.inf, 1.], np.eye(2), 1., 1, 0),
        ("nan_hessian", [1., 1.], [[np.nan, 0.], [0., 1.]], 1., 1, 0),
        ("inf_hessian", [1., 1.], [[np.inf, 0.], [0., 1.]], 1., 1, 0),
        ("asymmetric", [1., 1.], [[1., .1], [.2, 1.]], 1., 1, 0),
        ("unrepresentable_step", [1e-300], [[1e300]], 1., 4, 3),
        ("unrepresentable_score", [1e308], [[-1.]], 10., 4, 3),
    ):
        add(name, g, h, radius, status=status, method=0, fallback=fallback)
    assert len(cases) == 87
    return cases


CASES = precedent_cases()


def compiler_and_flags():
    compiler = shutil.which("clang++") or shutil.which("g++")
    if compiler is None:
        pytest.skip("A C++20 compiler is required for bounded direction contracts")
    frame_flags = (
        ["-Wframe-larger-than=16384", "-Werror=frame-larger-than"]
        if "clang" in Path(compiler).name
        else ["-Werror=frame-larger-than=16384"]
    )
    return compiler, ["-std=c++20", "-O3", "-ffp-contract=fast", "-funroll-loops",
        *frame_flags, "-fstack-usage", "-I", str(NATIVE),
        "-I", str(ROOT / "scivol" / "_csrc")]


def run(command, **kwargs):
    result = subprocess.run(command, capture_output=True, text=True, timeout=60, **kwargs)
    assert result.returncode == 0, result.stdout + result.stderr
    return result.stdout


def check_frames(directory):
    frames = []
    for path in directory.glob("*.su"):
        for line in path.read_text().splitlines():
            fields = line.split("\t")
            assert len(fields) == 3 and fields[2] == "static", line
            assert int(fields[1]) <= 16384, line
            frames.append(int(fields[1]))
    assert frames, "Compiler did not emit the required stack audit"


@pytest.fixture(scope="module", params=["production", "sanitized"])
def native(request, tmp_path_factory):
    directory = tmp_path_factory.mktemp(f"bounded_{request.param}")
    executable = directory / "contracts"
    compiler, flags = compiler_and_flags()
    if request.param == "sanitized":
        flags += ["-g", "-fsanitize=address,undefined", "-fno-omit-frame-pointer"]
    run([compiler, *flags, str(SOURCE), "-o", str(executable)], cwd=directory)
    # ASan's fake-stack instrumentation is not the native frame contract.
    if request.param == "production":
        check_frames(directory)
    return executable


def invoke(native, mode, tokens=()):
    environment = {**os.environ, "ASAN_OPTIONS": "detect_leaks=0"}
    output = run([str(native), mode], input=" ".join(map(str, tokens)), env=environment)
    return [json.loads(line) for line in output.splitlines()]


def proposal(native, g, h, radius, fixed=0, sweeps=64):
    return invoke(native, "case", [len(g), fixed, sweeps, radius, *g, *h.ravel()])[0]


def check_scores(result, g, h, radius):
    d = np.asarray(result["direction"])
    assert np.isfinite(d).all() and 0 < result["norm"] <= radius
    np.testing.assert_allclose(result["norm"], np.hypot.reduce(d), rtol=4e-15, atol=0)
    with localcontext() as context:
        context.prec = 100
        gd = sum((Decimal.from_float(float(a))*Decimal.from_float(float(b))
            for a, b in zip(g, d)), Decimal(0))
        hd = sum((Decimal.from_float(float(d[i]))*Decimal.from_float(float(h[i, j]))
            *Decimal.from_float(float(d[j]))
            for i in range(len(g)) for j in range(len(g))), Decimal(0))
        quadratic = gd + hd/2
    for key, value in (("slope", gd), ("curvature", hd), ("quadratic_change", quadratic)):
        expected = float(value)
        assert abs(result[key]-expected) <= 4e-12*abs(expected) + 1e-300, (key, result, expected)
    assert gd < 0 and quadratic < 0


def check_output(result, g, h, radius, status, method, fallback):
    assert result["status"] == status and result["method"] == method
    assert result["fallback"] == fallback and result["allocations"] == 0
    numeric = ("norm", "slope", "curvature", "quadratic_change", "resolution",
        "minimum_eigenvalue", "maximum_eigenvalue")
    assert all(np.isfinite(result[key]) for key in numeric)
    if status:
        assert result["direction"] == [SENTINEL]*len(g)
        assert all(result[key] == 0 for key in numeric[:4])
        assert not result["radius_limited"]
        if status != 4:
            assert all(result[key] == 0 for key in (*numeric, "positive", "negative", "unresolved"))
        return
    if method == 1:
        assert result["direction"] == [0.]*len(g)
        assert all(result[key] == 0 for key in (*numeric, "positive", "negative", "unresolved"))
        assert not result["radius_limited"]
        return

    check_scores(result, g, h, radius)
    hscale, gscale = np.max(np.abs(h)), np.max(np.abs(g))
    if fallback == 2:
        assert result["positive"] == result["negative"] == result["unresolved"] == 0
        assert all(result[key] == 0 for key in numeric[4:])
    elif hscale:
        values, vectors = np.linalg.eigh(h/hscale)
        resolution = 64*len(g)*EPSILON*max(abs(values[0]), abs(values[-1]))
        np.testing.assert_allclose(
            [result["minimum_eigenvalue"], result["maximum_eigenvalue"], result["resolution"]],
            [values[0], values[-1], resolution], rtol=4e-12, atol=8*len(g)*EPSILON)
        assert result["positive"] == np.count_nonzero(values > resolution)
        assert result["negative"] == np.count_nonzero(values < -resolution)
        assert result["unresolved"] == np.count_nonzero(abs(values) <= resolution)
    else:
        assert result["positive"] == result["negative"] == 0 and result["unresolved"] == len(g)
        assert all(result[key] == 0 for key in numeric[4:])
    if method == 2:
        values, vectors = np.linalg.eigh(h/hscale)
        raw = -vectors @ ((vectors.T @ (g/gscale))/np.abs(values))
        norm = np.linalg.norm(raw)
        length = min(radius, (gscale/hscale)*norm)
        expected_d = raw/norm*length
    else:
        unit = -(g/gscale)
        unit /= np.linalg.norm(unit)
        curvature = unit @ h @ unit
        length = min(radius, -(g @ unit)/curvature) if curvature > 0 else radius
        expected_d = unit*length
    assert result["radius_limited"] == (length == radius)
    relative_error = np.linalg.norm(np.asarray(result["direction"])-expected_d)/max(
        np.linalg.norm(expected_d), 1e-300)
    assert relative_error < 4e-10, (relative_error, result, expected_d)
    return relative_error


@pytest.mark.parametrize("name,g,h,radius,status,sweeps,method,fallback", CASES)
def test_precedent_direction_controls(native, name, g, h, radius, status, sweeps, method, fallback):
    result = proposal(native, g, h, radius, sweeps=sweeps)
    check_output(result, g, h, radius, status, method, fallback)
    if len(g) in (1, 2, 3, 5, 9):
        fixed = proposal(native, g, h, radius, fixed=len(g), sweeps=sweeps)
        check_output(fixed, g, h, radius, status, method, fallback)
        for key in ("status", "method", "fallback", "positive", "negative", "unresolved",
                    "radius_limited", "allocations"):
            assert fixed[key] == result[key], key
        for key in ("direction", "norm", "slope", "curvature", "quadratic_change",
                    "resolution", "minimum_eigenvalue", "maximum_eigenvalue"):
            np.testing.assert_allclose(fixed[key], result[key], rtol=4e-13, atol=1e-300,
                err_msg=f"{name}: {key} fixed/runtime")
    if name in SAVED_BLOCKS and radius == 1.:
        sign = 1 if name == "reference" else -1
        np.testing.assert_allclose(result["direction"],
            sign*np.asarray(SAVED_BLOCKS[name]["stationary"], dtype=float), rtol=2e-13, atol=0)
        assert result["negative"] == (0 if name == "reference" else 2)
    if name == "hard_case":
        assert result["quadratic_change"] == -.5  # Not the exact trust-region minimum -7/6.


def test_buffer_guards(native):
    assert invoke(native, "guards") == [{"guard_checks": 42, "allocations": 0}]


def test_physically_exact_and_one_byte_short_arenas(native):
    rows = invoke(native, "arenas")
    assert rows.pop() == {"exact_arena_checks": 24, "allocations": 0}
    assert len(rows) == 24
    for index, row in enumerate(rows):
        n = len(row["direction"])
        check_output(row, np.ones(n), 2*np.eye(n), 1.,
            status=2 if index % 2 else 0, method=0 if index % 2 else 2, fallback=0)


def test_read_only_aliases_and_adjacent_active_prefixes(native):
    rows = invoke(native, "aliases")
    assert rows.pop() == {"alias_checks": 6, "allocations": 0}
    assert len(rows) == 6
    h = np.array([[4., 1.], [1., 3.]])
    for row, g in zip(rows, ([4., 1.], [1., 1.], [4., 1.], [1., 1.], [1., -2.], [1., -2.])):
        check_output(row, np.asarray(g), h, 1., status=0, method=2, fallback=0)


def test_binary_has_no_allocator_and_reuses_shared_owners(tmp_path):
    compiler, flags = compiler_and_flags()
    obj = tmp_path / "bounded_direction.o"
    run([compiler, *flags, "-DVLK_BOUNDED_DIRECTION_AUDIT_ONLY", "-c", str(SOURCE),
        "-o", str(obj)], cwd=tmp_path)
    symbols = run(["nm", "-u", str(obj)])
    assert not re.search(r"\b_?(malloc|calloc|realloc|free|alloca)\b|_Zn[aw]", symbols), symbols
    check_frames(tmp_path)
    header = (NATIVE / "bounded_direction.hpp").read_text()
    assert '#include "active_face_refinement.hpp"' in header
    assert "curvature::symmetric_eigensystem(" in header
    assert "workspace::WorkspaceSizer" in header and "workspace::WorkspaceArena" in header
    assert "localdev" not in header and "aparch42" not in header
    code = re.sub(r"/\*.*?\*/|//[^\n]*", "", header, flags=re.DOTALL)
    assert not re.search(r"\b(?:alloca|_alloca|__builtin_alloca|malloc|calloc|realloc)\s*\(|\bnew\s+", code)
