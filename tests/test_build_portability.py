from __future__ import annotations

import importlib.util
import hashlib
import json
from pathlib import Path
from unittest.mock import patch

import pytest


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def build_configuration():
    module_spec = importlib.util.spec_from_file_location(
        "scivol_build_configuration",
        ROOT / "setup.py",
    )
    assert module_spec is not None
    assert module_spec.loader is not None
    module = importlib.util.module_from_spec(module_spec)
    with patch("setuptools.setup"):
        module_spec.loader.exec_module(module)
    return module


def test_cibuildwheel_selects_portable_flags(
    build_configuration,
    monkeypatch,
):
    monkeypatch.delenv("SCIVOL_PORTABLE_BUILD", raising=False)
    monkeypatch.setenv("CIBUILDWHEEL", "1")

    assert build_configuration._portable_build()
    assert "-march=native" not in build_configuration._compile_args()
    assert "-ffast-math" not in build_configuration._compile_args()


def test_explicit_portable_build_remains_supported(
    build_configuration,
    monkeypatch,
):
    monkeypatch.delenv("CIBUILDWHEEL", raising=False)
    monkeypatch.setenv("SCIVOL_PORTABLE_BUILD", "1")

    assert build_configuration._portable_build()
    assert "-march=native" not in build_configuration._native_compile_args()


def test_frame_limit_flag_matches_compiler(build_configuration):
    assert build_configuration._frame_limit_args("clang++") == [
        "-Wframe-larger-than=16384",
        "-Werror=frame-larger-than",
    ]
    assert build_configuration._frame_limit_args("g++") == [
        "-Werror=frame-larger-than=16384",
    ]


def test_extensions_track_included_mathematics(build_configuration):
    extensions = {
        extension.name: extension for extension in build_configuration.ext_modules
    }
    common = {
        path.relative_to(ROOT).as_posix()
        for path in (ROOT / "scivol/_csrc").rglob("*.h")
    }
    native = {
        path.relative_to(ROOT).as_posix()
        for path in (ROOT / "scivol/_native").rglob("*")
        if path.suffix in {".h", ".hpp", ".c"}
    }
    assert common <= set(extensions["scivol._core"].depends)
    assert common | native | {"scivol/_csrc/gh_math.c"} <= set(
        extensions["scivol._native_core"].depends
    )
    assert all(
        (ROOT / path).is_file()
        for extension in extensions.values()
        for path in extension.depends
    )


def test_native_templates_avoid_floating_non_type_parameters():
    policy_headers = [
        ROOT / "scivol/_native/garch_transform_policy.hpp",
        ROOT / "scivol/_native/garch_extended_density_policy.hpp",
    ]

    for header in policy_headers:
        assert "template<double" not in header.read_text()


def test_portable_blas_option_arguments_are_const():
    declaration = (
        ROOT / "scivol/_native/third_party/scipy_slsqp/blaslapack_declarations.h"
    ).read_text()
    implementation = (ROOT / "scivol/_native/portable_linalg.cpp").read_text()

    assert "const char* trans" in declaration
    assert "const char* trans" in implementation
    assert "const_cast<char*>" not in implementation


def test_native_optimizer_does_not_require_stop_token():
    driver = (ROOT / "scivol/_native/kraft_slsqp_driver.hpp").read_text()

    assert "<stop_token>" not in driver
    assert "std::stop_token" not in driver


@pytest.mark.parametrize("dependency", ["boost", "tlfloat"])
def test_wide_numeric_dependencies_are_pinned(dependency, build_configuration):
    directory = ROOT / "scivol/_native/third_party" / dependency
    hashes = json.loads((directory / "FILES.json").read_text())
    actual = {
        path.relative_to(directory).as_posix()
        for path in directory.rglob("*")
        if path.is_file() and path.name not in {"FILES.json", "UPSTREAM.md"}
    }
    assert actual == set(hashes)
    for filename, expected in hashes.items():
        assert (
            hashlib.sha256((directory / filename).read_bytes()).hexdigest() == expected
        )
    native = next(
        extension
        for extension in build_configuration.ext_modules
        if extension.name == "scivol._native_core"
    )
    assert directory.relative_to(ROOT).as_posix() in native.include_dirs
    assert "-ffast-math" not in native.extra_compile_args
