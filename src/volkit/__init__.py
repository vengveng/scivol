"""
volkit
------
High-performance volatility models.
"""

from importlib.metadata import version as _v
__version__ = _v(__name__)

# low-level C extension (leave as is)
from . import _core as _core

# ------- public Python API -------
from .roles import Role
from .components import (
    ARMA,
    GARCH,
    Normal,
    StudentT,
    CompositeSpec,
    Component,
)
from ._kernels.__init__ import (
    get_special_kernel,
    get_general_kernel,
)
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
    " __version__",
]