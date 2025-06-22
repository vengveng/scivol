# volkit/__init__.py
from importlib.metadata import version as _v
__version__ = _v("volkit")

# optional C extension
try:
    from . import _core as _core
except ModuleNotFoundError:
    _core = None  # pure-python mode

# public API ---------------------------------------------------------
from .roles import Role
from .components import ARMA, GARCH, Normal, StudentT, Component
from .spec import CompositeSpec
from .estimators import MLE
from ._kernels import get_special_kernel, get_general_kernel

__all__ = [
    "Role",
    "Component",
    "ARMA",
    "GARCH",
    "Normal",
    "StudentT",
    "CompositeSpec",
    "MLE",
    "get_special_kernel",
    "get_general_kernel",
    "__version__",
]