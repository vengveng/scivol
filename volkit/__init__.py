# volkit/__init__.py
from importlib.metadata import version as _v
__version__ = _v("volkit")

from .components import ARMA, GARCH, Normal, StudentT, Component
from .spec import CompositeSpec
from .estimators import MLE
from ._kernels import get_special_kernel, get_general_kernel
from .roles import Role

__all__: list[str] = [
    "ARMA",
    "Component",
    "CompositeSpec",
    "GARCH",
    "get_general_kernel",
    "get_special_kernel",
    "MLE",
    "Normal",
    "Role",
    "StudentT",
    "__version__",
]