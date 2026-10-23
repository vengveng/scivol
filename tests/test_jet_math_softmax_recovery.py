"""Original-z simplex pullbacks, compiled without the installed extension."""

from functools import lru_cache
import ctypes
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
SOURCE = Path(__file__).with_name("native_softmax_recovery.cpp")
FAMILIES = {"garch": 0, "gjr": 1, "aparch": 2, "joint": 3}
FLAGS = ["-std=c++20", "-O3", "-ffp-contract=fast", "-funroll-loops", "-march=native",
         "-fno-exceptions", "-fno-rtti", "-Wframe-larger-than=16384",
         "-Werror=frame-larger-than", "-fstack-usage", "-fno-builtin-exp"]


def run(command, **kwargs):
    result = subprocess.run(command, capture_output=True, text=True, timeout=60, **kwargs)
    assert result.returncode == 0, result.stdout + result.stderr
    return result.stdout


def retain_source_identity(directory):
    pending = [SOURCE, Path(__file__), ROOT / "scivol/_native/arma_garch_runtime_reformulation.cpp"]
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
        shutil.copyfile(path, target)
        if path.suffix not in {".cpp", ".h", ".hpp"}:
            continue
        for include in re.findall(r'^\s*#include\s+"([^"]+)"', content.decode(), re.M):
            owner = next((candidate for candidate in
                          (path.parent / include, ROOT / "scivol/_native" / include,
                           ROOT / "scivol/_csrc" / include) if candidate.is_file()), None)
            assert owner is not None, include
            pending.append(owner)
    return hashes


@pytest.fixture(scope="module")
def probe(tmp_path_factory):
    compiler = "/usr/bin/clang++" if sys.platform == "darwin" else shutil.which("clang++")
    if not compiler:
        pytest.skip("clang++ is required for the production-arithmetic probe")
    retained = os.environ.get("SCIVOL_SOFTMAX_EVIDENCE_DIR")
    directory = Path(retained).resolve() if retained else tmp_path_factory.mktemp("softmax_recovery")
    directory.mkdir(parents=True, exist_ok=True)
    hashes = retain_source_identity(directory)
    executable = directory / "probe"
    includes = ["-I", str(directory / "scivol/_native"),
                "-I", str(directory / "scivol/_csrc"),
                "-I", str(ROOT / "scivol/_native/third_party/boost"),
                "-I", str(ROOT / "scivol/_native/third_party/tlfloat")]
    sources = [directory / SOURCE.relative_to(ROOT),
               directory / "scivol/_native/arma_garch_runtime_reformulation.cpp"]
    objects = [directory / (source.stem + ".o") for source in sources]
    commands = [[compiler,*FLAGS,*includes,"-c",str(source),"-o",str(object_file)]
                for source,object_file in zip(sources,objects)]
    commands.append([compiler,*FLAGS,*map(str,objects),
                     "-Wl,-dead_strip" if sys.platform == "darwin" else "-Wl,--gc-sections",
                     *([] if sys.platform == "darwin" else ["-ldl"]),
                     "-o",str(executable)])
    for command in commands:
        run(command)
    frames = []
    for path in directory.glob("*.su"):
        for line in path.read_text().splitlines():
            symbol, size, storage = line.split("\t")
            assert storage == "static" and int(size) <= 16384, line
            frames.append({"symbol": symbol, "bytes": int(size), "storage": storage})
    imports = run(["otool", "-L", str(executable)]) if sys.platform == "darwin" else run(["ldd", str(executable)])
    record = {"commands": commands, "source_hashes": hashes,
              "binary_sha256": hashlib.sha256(executable.read_bytes()).hexdigest(),
              "stack_frames": frames, "imports": imports,
              "scope": "Standalone production policies; no installed extension, optimizer or fit"}
    (directory / "build.json").write_text(json.dumps(record, indent=2) + "\n")
    yield executable


def coordinate_point(family, p, q, dominant, magnitude):
    width = {"garch": 1+p+q, "gjr": 1+2*p+q,
             "aparch": 2+2*p+q, "joint": 4+p+q}[family]
    z = np.full(width, -2., dtype=np.float64)
    offset = 3 if family == "joint" else 0
    z[offset] = -.4
    if family == "joint":
        z[:3] = [.02, .12, -.08]
    if family == "aparch":
        z[1+p:1+2*p] = .15
        z[-1] = np.log(np.expm1(1.7))
        indices = np.r_[np.arange(1, 1+p), np.arange(1+2*p, 1+2*p+q)]
    else:
        count = 2*p+q if family == "gjr" else p+q
        indices = np.arange(offset+1, offset+1+count)
    z[indices[dominant]] = magnitude
    return z, indices


@lru_cache(maxsize=None)
def map_oracle(family, p, q, dominant, shift):
    jax = pytest.importorskip("jax")
    import jax.numpy as jnp
    jax.config.update("jax_enable_x64", True)
    offset = 3 if family == "joint" else 0
    count = 2*p+q if family == "gjr" else p+q
    limit = 1. - 8. * (count+1) * np.finfo(float).eps
    if family == "aparch":
        indices = tuple(range(1, 1+p)) + tuple(range(1+2*p, 1+2*p+q))
    else:
        indices = tuple(range(offset+1, offset+1+count))

    def mapped(z):
        exponentials = jnp.exp(z[jnp.array(indices)] - shift)
        slack = jnp.exp(-shift)
        denominator = slack + jnp.sum(exponentials)
        contributions = limit * exponentials / denominator
        if shift > 0.:
            remainder = slack + sum(exponentials[i] for i in range(count) if i != dominant)
            contributions = contributions.at[dominant].set(limit - limit * remainder / denominator)
        theta = z.at[offset].set(jax.nn.softplus(z[offset]) + np.finfo(float).tiny)
        if family == "joint":
            theta = theta.at[1].set(jnp.tanh(z[1])).at[2].set(-jnp.tanh(z[2]))
        if family == "gjr":
            alpha = 2. * contributions[:p]
            theta = theta.at[1:1+p].set(alpha)
            theta = theta.at[1+p:1+2*p].set(2. * contributions[p:2*p] - alpha)
            return theta.at[1+2*p:].set(contributions[2*p:])
        if family == "aparch":
            gamma = jnp.tanh(z[1+p:1+2*p])
            delta = jax.nn.softplus(z[-1]) + np.finfo(float).tiny
            absolute_moment = jnp.exp(.5*delta*jnp.log(2.)
                + jax.scipy.special.gammaln(.5*(delta+1.)) - .5*jnp.log(jnp.pi))
            moment = .5*((1.-gamma)**delta+(1.+gamma)**delta)*absolute_moment
            theta = theta.at[1:1+p].set(contributions[:p]/moment)
            theta = theta.at[1+p:1+2*p].set(gamma)
            theta = theta.at[-1].set(delta)
            return theta.at[1+2*p:1+2*p+q].set(contributions[p:])
        return theta.at[offset+1:offset+1+count].set(contributions)

    return jax.jit(jax.jacfwd(mapped)), jax.jit(jax.jacfwd(jax.jacfwd(mapped)))


def evaluate_requests(probe, family, p, q, fixed, z):
    payload = "".join(f"{FAMILIES[family]} {p} {q} {int(fixed)} {request} "
                      + " ".join(value.hex() for value in z) + "\n" for request in range(1, 8))
    output = run([str(probe)], input=payload)
    measured = [np.array([float.fromhex(token) for token in line.split()]) for line in output.splitlines()]
    rows = [row[:-1] for row in measured]
    assert len(rows) == 7
    retained = os.environ.get("SCIVOL_SOFTMAX_EVIDENCE_DIR")
    if retained:
        record = {"family": family, "orders": [p,q], "fixed": fixed,
                  "coordinates": [value.hex() for value in z],
                  "outputs": [[value.hex() for value in row] for row in rows],
                  "exp_calls": [int(row[-1]) for row in measured]}
        with (Path(retained) / "points.jsonl").open("a") as stream:
            stream.write(json.dumps(record) + "\n")
    width = z.size
    sections = np.cumsum([width, 1, width, width*width, 1, width, width*width])[:-1]
    return [np.split(row, sections) for row in rows]


@pytest.mark.parametrize("count", [2,4,32])
def test_prepared_simplex_contractions_do_not_call_exp(probe, count):
    prepared,contracted,checksum = run([str(probe),str(count),"40."]).split()
    assert int(prepared)==count+1
    assert int(contracted)==int(prepared)
    assert np.isfinite(float.fromhex(checksum))


@pytest.mark.parametrize("family", ["garch","gjr"])
@pytest.mark.parametrize("p", [1,3])
def test_fixed_all_masks_prepare_shifted_weights_once(probe, family, p):
    z,_ = coordinate_point(family,p,1,0,40.)
    payload = "".join(f"{FAMILIES[family]} {p} 1 1 {request} "
                      + " ".join(value.hex() for value in z) + "\n" for request in range(1,8))
    rows = run([str(probe)],input=payload).splitlines()
    count=2*p+1 if family=="gjr" else p+1
    # H also needs one omega softplus-curvature exponential, outside the simplex.
    expected=[count+2+bool(request & 4) for request in range(1,8)]
    assert [int(float.fromhex(row.split()[-1])) for row in rows]==expected


def assert_original_z_pullback(rows, family, p, q, dominant, z):
    jacobian, curvature = map_oracle(family, p, q, dominant, max(0., z.max()))
    j = np.asarray(jacobian(z))
    t = np.asarray(curvature(z))
    full = rows[-1]
    theta_f, theta_g, theta_h = full[4][0], full[5], full[6].reshape(z.size,z.size)
    expected_g = j.T @ theta_g
    expected_h = j.T @ theta_h @ j + np.einsum("i,ijk->jk",theta_g,t)
    for request, row in enumerate(rows, 1):
        _, f, g, h, _, _, _ = row
        if request & 1:
            assert f[0] == theta_f
            assert f[0] == full[1][0]
        else:
            assert f[0] == 123.
        if request & 2:
            np.testing.assert_allclose(g, expected_g, rtol=3e-12, atol=0.)
            np.testing.assert_allclose(g, full[2], rtol=8*np.finfo(float).eps, atol=0.)
        else:
            assert np.all(g == 456.)
        if request & 4:
            np.testing.assert_allclose(h.reshape(z.size,z.size), expected_h, rtol=3e-11, atol=0.)
            np.testing.assert_allclose(h, full[3], rtol=8*np.finfo(float).eps, atol=0.)
        else:
            assert np.all(h == 789.)


@pytest.mark.parametrize("family", tuple(FAMILIES))
@pytest.mark.parametrize("p,q", [(1,1),(3,1)])
@pytest.mark.parametrize("magnitude", [-.2,20.,36.,40.,700.])
def test_all_masks_retain_original_z_simplex_tails(probe, family, p, q, magnitude):
    z, indices = coordinate_point(family,p,q,0,magnitude)
    rows = evaluate_requests(probe,family,p,q,True,z)
    assert_original_z_pullback(rows,family,p,q,0,z)
    if magnitude >= 40. and family == "garch":
        limit = 1.-8.*(p+q+1)*np.finfo(float).eps
        assert rows[-1][0][indices[0]] == limit
        assert rows[-1][2][indices[0]] != 0.
        assert rows[-1][3].reshape(z.size,z.size)[indices[0],indices[0]] != 0.
    if family in {"aparch","joint"}:
        runtime_rows = evaluate_requests(probe,family,p,q,False,z)
        assert_original_z_pullback(runtime_rows,family,p,q,0,z)
        for fixed_row, runtime_row in zip(rows,runtime_rows):
            for actual, expected in zip(fixed_row,runtime_row):
                np.testing.assert_allclose(actual,expected,rtol=3e-12,atol=0.)


@pytest.mark.parametrize("family", tuple(FAMILIES))
def test_nonfirst_dominant_and_multiple_large_weights(probe, family):
    for dominant in (1,):
        z, _ = coordinate_point(family,1,1,dominant,40.)
        rows = evaluate_requests(probe,family,1,1,True,z)
        assert_original_z_pullback(rows,family,1,1,dominant,z)
    z, indices = coordinate_point(family,1,1,0,40.)
    z[indices[1]] = 40.
    rows = evaluate_requests(probe,family,1,1,True,z)
    assert_original_z_pullback(rows,family,1,1,0,z)


@pytest.fixture(scope="module")
def rebuilt_core():
    from scivol import _native_core
    return _native_core


def mapped_extension_point(core, family, p, q, z):
    theta = np.empty_like(z)
    if family in {"garch", "gjr"}:
        core._map_garch(z.ctypes.data,theta.ctypes.data,p,q,0,True,family=="gjr")
    elif family == "aparch":
        workspace = np.empty(core._aparch_map_workspace_bytes(p,q,0),dtype=np.uint8)
        core._map_aparch(z.ctypes.data,theta.ctypes.data,p,q,0,True,
                         workspace.ctypes.data,workspace.nbytes)
    else:
        workspace = np.empty(core._arma_garch_map_workspace_bytes(1,1,p,q),dtype=np.uint8)
        core._map_arma_garch(z.ctypes.data,theta.ctypes.data,1,1,p,q,0,True,
                            workspace.ctypes.data,workspace.nbytes)
    return theta


def evaluate_extension_point(core, family, p, q, point, request, transformed):
    data = np.array([.3,-.7,1.1,-.2,.9,-.4,.8,-.6])
    squared = data*data
    f,g,h = np.full(1,123.),np.full(point.size,456.),np.full((point.size,point.size),789.)
    if family in {"garch", "gjr"}:
        workspace = np.empty(core._garch_workspace_bytes(p,q,0,transformed,request,family=="gjr"),dtype=np.uint8)
        status,valid = core._garch_jet(point.ctypes.data,data.ctypes.data,squared.ctypes.data,
            data.size,1.,p,q,0,transformed,request,workspace.ctypes.data,workspace.nbytes,
            f.ctypes.data,g.ctypes.data,h.ctypes.data,family=="gjr")
    elif family == "aparch":
        workspace = np.empty(core._aparch_workspace_bytes(p,q,0,transformed,request),dtype=np.uint8)
        status,valid = core._aparch_jet(point.ctypes.data,point.size,data.ctypes.data,data.size,
            1.,p,q,0,transformed,request,workspace.ctypes.data,workspace.nbytes,
            f.ctypes.data,g.ctypes.data,h.ctypes.data)
    else:
        workspace = np.empty(core._arma_garch_workspace_bytes(1,1,p,q,0,transformed,request),dtype=np.uint8)
        status,valid = core._arma_garch_jet(point.ctypes.data,point.size,data.ctypes.data,data.size,
            1.,1,1,p,q,0,transformed,request,workspace.ctypes.data,workspace.nbytes,
            f.ctypes.data,g.ctypes.data,h.ctypes.data)
    assert (status,valid) == (0,request)
    return f,g,h.ravel()


@pytest.mark.parametrize("family", tuple(FAMILIES))
@pytest.mark.parametrize("p,q", [(1,1),(3,1)])
@pytest.mark.parametrize("magnitude", [-.2,40.])
def test_rebuilt_extension_all_masks_runtime_and_fixed(rebuilt_core, family, p, q, magnitude):
    z,_ = coordinate_point(family,p,q,0,magnitude)
    theta = mapped_extension_point(rebuilt_core,family,p,q,z)
    theta_result = evaluate_extension_point(rebuilt_core,family,p,q,theta,7,False)
    rows = [[theta,*evaluate_extension_point(rebuilt_core,family,p,q,z,request,True),
             *theta_result] for request in range(1,8)]
    assert_original_z_pullback(rows,family,p,q,0,z)


@pytest.mark.parametrize("family", ["garch","gjr"])
@pytest.mark.parametrize("magnitude", [-.2,40.])
def test_rebuilt_extension_forced_runtime_matches_fixed(rebuilt_core, family, magnitude):
    z,_ = coordinate_point(family,1,1,0,magnitude)
    theta = mapped_extension_point(rebuilt_core,family,1,1,z)
    theta_result = evaluate_extension_point(rebuilt_core,family,1,1,theta,7,False)
    library = ctypes.CDLL(rebuilt_core.__file__)
    prefix = "vlk_runtime_gjr_garch_sample_jet" if family=="gjr" else "vlk_runtime_garch_sample_jet"
    planner,evaluator = getattr(library,prefix+"_workspace_bytes"),getattr(library,prefix+"_evaluate")
    size,unsigned,pointer = ctypes.c_size_t,ctypes.c_uint32,ctypes.c_void_p
    planner.argtypes = [size,size,unsigned,unsigned,unsigned]
    planner.restype = size
    evaluator.argtypes = [pointer,pointer,pointer,size,ctypes.c_double,size,size,
                         unsigned,unsigned,unsigned,pointer,size,pointer,pointer,pointer,
                         ctypes.POINTER(unsigned)]
    evaluator.restype = ctypes.c_int32
    data = np.array([.3,-.7,1.1,-.2,.9,-.4,.8,-.6])
    squared = data*data
    rows = []
    for request in range(1,8):
        workspace = np.empty(planner(1,1,0,1,request),dtype=np.uint8)
        f,g,h = np.full(1,123.),np.full(z.size,456.),np.full((z.size,z.size),789.)
        valid = unsigned(0)
        status = evaluator(z.ctypes.data,data.ctypes.data,squared.ctypes.data,data.size,1.,
            1,1,0,1,request,workspace.ctypes.data,workspace.nbytes,
            f.ctypes.data,g.ctypes.data,h.ctypes.data,ctypes.byref(valid))
        assert (status,valid.value)==(0,request)
        rows.append([theta,f,g,h.ravel(),*theta_result])
    assert_original_z_pullback(rows,family,1,1,0,z)
