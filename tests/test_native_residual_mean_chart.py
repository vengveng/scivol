"""Bounded shared residual chart/mapper lifecycle and caller-arena contracts.

These private images do not load the installed extension or run a fitting policy.
The parent owns multiprecision AD and phase replay evidence.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import shutil
import subprocess
import sys
import sysconfig
import time
from types import SimpleNamespace

import pytest


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "tests/native_residual_mean_chart.cpp"
REVIEW = ROOT / "localdev/aparch42_chart_integration_20261010/full_controller/canonical_chart_test_review"
MODES = ("intentions", "lifecycle", "guards", "analytic", "poisoned")


def snapshot_sources(directory):
    pending = [SOURCE, Path(__file__).resolve()]
    hashes = {}
    while pending:
        path = pending.pop()
        relative = str(path.relative_to(ROOT))
        if relative in hashes:
            continue
        content = path.read_bytes()
        hashes[relative] = hashlib.sha256(content).hexdigest()
        target = directory / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(content)
        for include in re.findall(rb'^\s*#include\s+"([^"]+)"', content, re.M):
            name = include.decode()
            candidates = (path.parent / name, ROOT / "scivol/_native" / name,
                          ROOT / "scivol/_csrc" / name)
            candidate = next((p for p in candidates if p.exists()), None)
            assert candidate is not None, name
            pending.append(candidate.resolve())
    return hashes


def compiler_and_flags(directory):
    command = os.environ.get("CXX") or sysconfig.get_config_var("CXX") or "clang++"
    compiler = shlex.split(command)
    if not shutil.which(compiler[0]):
        pytest.skip("A C++20 compiler is required for residual chart contracts")
    executable = Path(compiler[0]).name.lower()
    is_gnu = "g++" in executable or "gcc" in executable
    frame = (["-Wframe-larger-than=16384", "-Werror=frame-larger-than"]
             if "clang" in executable or (sys.platform == "darwin" and not is_gnu)
             else ["-Werror=frame-larger-than=16384"])
    flags = ["-std=c++20", "-O3", "-fno-exceptions", "-fno-rtti", *frame, "-fstack-usage"]
    if os.environ.get("SCIVOL_PORTABLE_BUILD") != "1" and not os.environ.get("CIBUILDWHEEL"):
        flags += ["-ffp-contract=fast", "-funroll-loops", "-march=native"]
    flags += ["-I", str(directory / "scivol/_native"),
              "-I", str(directory / "scivol/_csrc"),
              "-I", str(ROOT / "scivol/_native/third_party/boost"),
              "-I", str(ROOT / "scivol/_native/third_party/tlfloat")]
    return compiler, flags


def run(command, *, deadline, **kwargs):
    remaining = deadline - time.monotonic()
    assert remaining > 0, "Residual chart shard exceeded its 115-second hard cap"
    return subprocess.run(command, text=True, capture_output=True,
                          timeout=min(120, remaining), **kwargs)


def check_frames(directory):
    frames = []
    for path in directory.glob("*.su"):
        for line in path.read_text().splitlines():
            fields = line.split("\t")
            assert len(fields) == 3 and fields[2] == "static", line
            assert int(fields[1]) <= 16384, line
            frames.append(int(fields[1]))
    assert frames, "Compiler did not emit the required stack audit"
    return max(frames)


@pytest.fixture(scope="module", params=["production", "sanitized"])
def native(request):
    started = time.monotonic()
    directory = REVIEW / f"{request.param}-{os.getpid()}-{time.time_ns()}"
    directory.mkdir(parents=True)
    hashes = snapshot_sources(directory)
    compiler, flags = compiler_and_flags(directory)
    source = directory / SOURCE.relative_to(ROOT)
    executable = directory / "residual_contract"
    if request.param == "sanitized":
        flags += ["-g", "-fsanitize=address,undefined", "-fno-sanitize-recover=all",
                  "-fno-omit-frame-pointer"]
    command = [*compiler, *flags, str(source), "-o", str(executable)]
    result = run(command, deadline=started+115, cwd=directory)
    build_seconds = time.monotonic()-started
    record = dict(source_hashes=hashes, command=command, returncode=result.returncode,
                  stderr=result.stderr, build_seconds=build_seconds,
                  build_target_seconds=60, hard_cap_seconds=115,
                  scope="Shared chart/mapper storage and lifecycle only; no AD, solver or promotion claim")
    if result.returncode == 0:
        record["image_sha256"] = hashlib.sha256(executable.read_bytes()).hexdigest()
        if request.param == "production":
            record["maximum_static_frame_bytes"] = check_frames(directory)
    (directory / "build.json").write_text(json.dumps(record, indent=2)+"\n")
    assert result.returncode == 0, result.stdout+result.stderr
    state = SimpleNamespace(executable=executable, directory=directory, evidence=[], deadline=started+115)
    yield state
    changed = [path for path, digest in hashes.items()
               if hashlib.sha256((ROOT / path).read_bytes()).hexdigest() != digest]
    record["source_changes_during_run"] = changed
    record["shard_seconds"] = time.monotonic()-started
    (directory / "build.json").write_text(json.dumps(record, indent=2)+"\n")
    (directory / "contracts.json").write_text(json.dumps(state.evidence, indent=2)+"\n")
    assert not changed, f"Owners changed during this snapshot run; rerun: {changed}"


@pytest.mark.parametrize("mode", MODES)
def test_shared_chart_and_mapper_contracts(native, mode):
    environment = {**os.environ, "ASAN_OPTIONS": "detect_leaks=0:halt_on_error=1",
                   "UBSAN_OPTIONS": "halt_on_error=1:print_stacktrace=1"}
    result = run([str(native.executable), mode], deadline=native.deadline, env=environment)
    native.evidence.append(dict(mode=mode, returncode=result.returncode,
                                stdout=result.stdout, stderr=result.stderr))
    assert result.returncode == 0, result.stdout+result.stderr
    record = json.loads(result.stdout)
    assert record["mode"] == mode and record["checks"] > 20 and record["allocations"] == 0


def test_request_binary_has_no_allocator_symbols():
    started = time.monotonic()
    directory = REVIEW / f"allocator-audit-{os.getpid()}-{time.time_ns()}"
    directory.mkdir(parents=True)
    hashes = snapshot_sources(directory)
    compiler, flags = compiler_and_flags(directory)
    obj = directory / "requests.o"
    command = [*compiler, *flags, "-DVLK_RESIDUAL_CHART_AUDIT_ONLY", "-c",
               str(directory / SOURCE.relative_to(ROOT)), "-o", str(obj)]
    result = run(command, deadline=started+115, cwd=directory)
    assert result.returncode == 0, result.stdout+result.stderr
    symbols = run(["nm", "-u", str(obj)], deadline=started+115)
    assert symbols.returncode == 0, symbols.stderr
    (directory / "undefined_symbols.txt").write_text(symbols.stdout)
    record = dict(source_hashes=hashes, command=command, build_seconds=time.monotonic()-started,
                  maximum_static_frame_bytes=check_frames(directory),
                  image_sha256=hashlib.sha256(obj.read_bytes()).hexdigest(), stderr=result.stderr)
    (directory / "build.json").write_text(json.dumps(record, indent=2)+"\n")
    assert not re.search(r"\b_?(malloc|calloc|realloc|free|alloca|aligned_alloc|posix_memalign)\b|_Zn[aw]",
                         symbols.stdout), symbols.stdout


def test_value_basis_cannot_expose_unavailable_map_curvature():
    started = time.monotonic()
    directory = REVIEW / f"intention-api-{os.getpid()}-{time.time_ns()}"
    directory.mkdir(parents=True)
    hashes = snapshot_sources(directory)
    compiler, flags = compiler_and_flags(directory)
    source = directory / "unavailable_curvature.cpp"
    source.write_text('''#include "residual_mean_chart.hpp"
#include "aparch_runtime.hpp"
using Layout = vlk::experiment::runtime::aparch::Layout;
using Basis = vlk::experiment::mean_chart::ResidualPreservingChart<1, Layout, nullptr,
    vlk::experiment::mean::dynamic,
    vlk::experiment::mean_chart::PreparedIntention::value_basis>;
auto unavailable(Basis* basis) { return basis->map_second(0,0,0); }
''')
    command = [*compiler, *flags, "-fsyntax-only", str(source)]
    result = run(command, deadline=started+115, cwd=directory)
    (directory / "build.json").write_text(json.dumps(dict(source_hashes=hashes,
        command=command, returncode=result.returncode, stderr=result.stderr,
        build_seconds=time.monotonic()-started, expected="derivative-intention static assertion"),
        indent=2)+"\n")
    assert result.returncode != 0
    assert "static assertion" in result.stderr and "PreparedIntention::derivatives" in result.stderr
