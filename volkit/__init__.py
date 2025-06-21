"""
volkit  –  High-performance volatility models
"""

from importlib.metadata import version as _v

__version__: str = _v("volkit")

# ── optional native extension ─────────────────────────────────────────
try:
    from . import _core as _core  # noqa: F401  (used for side effects)
except ModuleNotFoundError:       # pure-Python fallback
    _core = None                  # noqa: F401

# ── public Python API ────────────────────────────────────────────────
from .roles import Role
from .components import (
    ARMA,
    GARCH,
    Normal,
    StudentT,
    CompositeSpec,
    Component,
)
from ._kernels import get_special_kernel, get_general_kernel
from .estimators import MLE

__all__: list[str] = [
    "Role",
    "ARMA",
    "GARCH",
    "Normal",
    "StudentT",
    "CompositeSpec",
    "Component",
    "get_special_kernel",
    "get_general_kernel",
    "MLE",
    "__version__",
]