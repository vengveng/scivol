"""Executed build evidence for the standalone core, not source-name heuristics."""

import importlib
import json
import os
from pathlib import Path
import shlex

import pytest


def _build_directory():
    location = os.environ.get("SCIVOLAPI_BUILD_DIR")
    if not location:
        raise RuntimeError("SCIVOLAPI_BUILD_DIR is required for the standalone build boundary gate.")
    return Path(location).resolve(strict=True)


def test_old_package_is_unavailable():
    with pytest.raises(ImportError, match="Legacy scivol is unavailable"):
        importlib.import_module("scivol")


def test_compilation_has_no_legacy_or_python_include_roots():
    commands = json.loads((_build_directory() / "compile_commands.json").read_text())
    assert commands, "An empty compilation database does not establish isolation."
    for entry in commands:
        arguments = entry.get("arguments") or shlex.split(entry["command"])
        includes = []
        for index, arg in enumerate(arguments):
            if arg in ("-I", "-isystem", "-iquote"):
                includes.append(arguments[index + 1])
            elif arg.startswith("-I"):
                includes.append(arg[2:])
        for include in includes:
            path = Path(entry["directory"], include).resolve()
            assert "/scivol/_native" not in str(path)
            assert "/scivol/_csrc" not in str(path)
            assert not (path / "Python.h").exists()
        assert "-ffast-math" not in arguments


def test_compiled_dependencies_use_new_mathematical_owners():
    dependency_files = list(_build_directory().glob("CMakeFiles/scivolapi.dir/**/*.o.d"))
    assert dependency_files, "Require compiler dependency receipts, not just declared include flags."
    core = Path(__file__).resolve().parents[2] / "core"
    mathematical_dependencies = set()
    for receipt in dependency_files:
        source = receipt.read_text().replace("\\\n", " ")
        _, dependency_text = source.split(":", 1)
        for item in shlex.split(dependency_text):
            path = Path(item).resolve()
            assert "/scivol/_native" not in str(path)
            assert "/scivol/_csrc" not in str(path)
            assert path.name not in ("Python.h", "numpy.h", "arrayobject.h")
            if path.is_relative_to(core / "src"):
                mathematical_dependencies.add(path)
    assert any(path.name == "garch_recursion_policy.hpp" for path in mathematical_dependencies)
    assert any(path.name == "garch_density_policy.hpp" for path in mathematical_dependencies)
