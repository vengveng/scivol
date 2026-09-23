"""Compile shared header contracts independently of an installed extension."""

from pathlib import Path
import re
import shutil
import subprocess

import pytest


@pytest.mark.parametrize(
    "source",
    [
        "native_location_scale_assembly.cpp",
        "native_reflection_composition.cpp",
        "native_mean_workspace.cpp",
        "native_joint_garch_workspace.cpp",
        "native_linear_power_workspace.cpp",
        "native_runtime_equalities.cpp",
        "native_nnls_workspace.cpp",
        "native_gh_quadrature.cpp",
    ],
)
def test_mean_primitives(source, tmp_path):
    compiler = shutil.which("clang++") or shutil.which("g++")
    if compiler is None:
        pytest.skip("A C++20 compiler is required for the header contract test")
    root = Path(__file__).resolve().parents[1]
    executable = tmp_path / "header_contract"
    frame_flags = (
        ["-Wframe-larger-than=16384", "-Werror=frame-larger-than"]
        if "clang" in Path(compiler).name
        else ["-Werror=frame-larger-than=16384"]
    )
    extra_sources = (
        [str(root / "scivol/_native/gh_math_translation.cpp")]
        if source
        in {
            "native_mean_workspace.cpp",
            "native_joint_garch_workspace.cpp",
            "native_gh_quadrature.cpp",
        }
        else []
    )
    if source in {"native_runtime_equalities.cpp", "native_nnls_workspace.cpp"}:
        extra_sources = [
            str(root / "scivol/_native" / name)
            for name in (
                "slsqp_translation.cpp",
                "portable_linalg.cpp",
                "portable_nnls.cpp",
            )
        ]
    optimization = (
        ["-O3", "-ffp-contract=fast", "-funroll-loops"]
        if source
        in {
            "native_reflection_composition.cpp",
            "native_mean_workspace.cpp",
            "native_linear_power_workspace.cpp",
            "native_runtime_equalities.cpp",
            "native_nnls_workspace.cpp",
            "native_gh_quadrature.cpp",
        }
        else ["-O2"]
    )
    subprocess.run(
        [
            compiler,
            "-std=c++20",
            *optimization,
            *frame_flags,
            "-I",
            str(root / "scivol/_native"),
            "-I",
            str(root / "scivol/_csrc"),
            str(root / "tests" / source),
            *extra_sources,
            "-o",
            str(executable),
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=120 if source == "native_joint_garch_workspace.cpp" else 60,
    )
    subprocess.run([str(executable)], check=True, timeout=10)


def test_nnls_scratch_has_no_stack_or_heap_allocator():
    root = Path(__file__).resolve().parents[1]
    source = (root / "scivol/_native/portable_nnls.cpp").read_text()
    assert "WorkspaceArena" in source
    assert not re.search(
        r"\b(?:alloca|_alloca|__builtin_alloca|malloc|calloc|realloc)\s*\(|\bnew\s+",
        source,
    )


def test_gh_observation_loops_do_not_rebuild_point_normalizers():
    root = Path(__file__).resolve().parents[1]
    first = (root / "scivol/_native/gh_first_order.hpp").read_text()
    observations = first.split("variance_gamma_logpdf(", 1)[1]
    assert "ad::lgamma(" not in observations
    assert "ad::log(cache.alpha)" not in observations
    assert "ad::log(cache.vg_argument_scale)" not in observations
    scalar = (root / "scivol/_csrc/gh_math.c").read_text()
    observations = scalar.split("static gh_jet_t vg_logpdf_jet(", 1)[1].split(
        "int gh_abs_moment_derivatives(", 1
    )[0]
    assert "lgamma(" not in observations
    assert "jet_log(cache->alpha)" not in observations
    assert "log(cache->alpha" not in observations
    assert "log(cache->argument_scale)" not in observations
    assert "vg_standardized_logpdf_scalar(" not in observations
