"""The mandatory new-core suite must not import the installed scivol package."""

import importlib.abc
import sys

import pytest


class _BlockLegacyPackage(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "scivol" or fullname.startswith("scivol."):
            raise ImportError("Legacy scivol is unavailable in standalone core conformance tests.")
        return None


if any(name == "scivol" or name.startswith("scivol.") for name in sys.modules):
    raise RuntimeError("Run core conformance in a fresh process without imported scivol.")
sys.meta_path.insert(0, _BlockLegacyPackage())


@pytest.fixture
def native():
    from native_api import evaluate
    return evaluate
