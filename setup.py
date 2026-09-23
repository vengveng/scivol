from __future__ import annotations

import os
from pathlib import Path
import shlex
import sys
import sysconfig

from setuptools import Extension, setup


def _portable_build() -> bool:
    return (
        os.environ.get("SCIVOL_PORTABLE_BUILD") == "1"
        or os.environ.get("CIBUILDWHEEL") == "1"
    )


def _configured_cxx() -> str:
    return os.environ.get("CXX") or sysconfig.get_config_var("CXX") or ""


def _compiler_executable(command: str) -> str:
    try:
        executable = shlex.split(command)[0]
    except (IndexError, ValueError):
        return ""
    return os.path.basename(executable).lower()


def _frame_limit_args(compiler_command: str | None = None) -> list[str]:
    compiler = _compiler_executable(compiler_command or _configured_cxx())
    is_gnu = "g++" in compiler or "gcc" in compiler
    if "clang" in compiler or (sys.platform == "darwin" and not is_gnu):
        return [
            "-Wframe-larger-than=16384",
            "-Werror=frame-larger-than",
        ]
    return ["-Werror=frame-larger-than=16384"]


def _compile_args() -> list[str]:
    if sys.platform.startswith("win"):
        return ["/O2", "/fp:fast"]

    if _portable_build():
        return ["-O3"]

    return [
        "-O3",
        "-ffast-math",
        "-ffp-contract=fast",
        "-funroll-loops",
        "-march=native",
    ]


def _native_compile_args() -> list[str]:
    if sys.platform.startswith("win"):
        return ["/std:c++20", "/O2", "/fp:precise", "/EHs-c-", "/GR-"]

    base = [
        "-std=c++20",
        "-O3",
        "-fno-exceptions",
        "-fno-rtti",
        *_frame_limit_args(),
    ]
    if _portable_build():
        return base
    return [*base, "-ffp-contract=fast", "-funroll-loops", "-march=native"]


def _libraries() -> list[str]:
    if sys.platform.startswith("win"):
        return []
    return ["m"]


def _included_files(directory: str, suffixes: tuple[str, ...]) -> list[str]:
    root = Path(__file__).resolve().parent
    return sorted(
        path.relative_to(root).as_posix()
        for path in (root / directory).rglob("*")
        if path.is_file() and path.suffix in suffixes
    )


ext_modules = [
    Extension(
        "scivol._core",
        sources=[
            "scivol/_core.c",
            "scivol/_csrc/variance_garch.c",
            "scivol/_csrc/likelihood_garch.c",
            "scivol/_csrc/likelihood_normal.c",
            "scivol/_csrc/likelihood_studentt.c",
            "scivol/_csrc/likelihood_ged.c",
            "scivol/_csrc/likelihood_skewt.c",
            "scivol/_csrc/gh_math.c",
            "scivol/_csrc/likelihood_gh.c",
            "scivol/_csrc/likelihood_aparch.c",
            "scivol/_csrc/arma_vol_gh.c",
            "scivol/_csrc/errors_garch.c",
            "scivol/_csrc/transforms_logspace.c",
            "scivol/_csrc/arma_garch.c",
            "scivol/_csrc/arma_egarch.c",
            "scivol/_csrc/arma_gjr_garch.c",
            "scivol/_csrc/arma.c",
            "scivol/_csrc/mean_x.c",
            "scivol/_csrc/mean_garch.c",
            "scivol/_csrc/mean_gjr_garch.c",
            "scivol/_csrc/mean_egarch.c",
            "scivol/_csrc/variance_gjr_garch.c",
            "scivol/_csrc/likelihood_gjr_garch.c",
            "scivol/_csrc/likelihood_egarch.c",
            "scivol/_csrc/errors_gjr_garch.c",
            "scivol/_csrc/log_wrappers.c",
            "scivol/_csrc/dcc_gaussian.c",
        ],
        include_dirs=["scivol/_csrc"],
        depends=_included_files("scivol/_csrc", (".h",)),
        extra_compile_args=_compile_args(),
        libraries=_libraries(),
    ),
    Extension(
        "scivol._native_core",
        sources=[
            "scivol/_native/native_core.cpp",
            "scivol/_native/mean_runtime.cpp",
            "scivol/_native/linear_joint_runtime.cpp",
            "scivol/_native/linear_garch_runtime.cpp",
            "scivol/_native/linear_power_runtime.cpp",
            "scivol/_native/garch_generated_api.cpp",
            "scivol/_native/garch_runtime.cpp",
            "scivol/_native/egarch_runtime.cpp",
            "scivol/_native/arma_egarch_runtime.cpp",
            "scivol/_native/aparch_runtime.cpp",
            "scivol/_native/arma_garch_runtime.cpp",
            "scivol/_native/arma_garch_runtime_reformulation.cpp",
            "scivol/_native/portable_linalg.cpp",
            "scivol/_native/portable_nnls.cpp",
            "scivol/_native/slsqp_translation.cpp",
            "scivol/_native/gh_math_translation.cpp",
        ],
        include_dirs=[
            "scivol/_native",
            "scivol/_csrc",
            "scivol/_native/third_party/boost",
            "scivol/_native/third_party/tlfloat",
        ],
        depends=(
            _included_files("scivol/_native", (".h", ".hpp", ".c"))
            + _included_files("scivol/_csrc", (".h",))
            + ["scivol/_csrc/gh_math.c"]
        ),
        extra_compile_args=_native_compile_args(),
        libraries=_libraries(),
        language="c++",
    ),
]


setup(ext_modules=ext_modules)
