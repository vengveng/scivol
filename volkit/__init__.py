# volkit/__init__.py
from importlib.metadata import version as _v
__version__ = _v("volkit")

try:
    from . import _core as _core
except ModuleNotFoundError:
    raise ImportError(
        "volkit was installed without its compiled core."
        "Build wheels or run `pip install .` with a C compiler."
    )

from .components import ARMA, GARCH, Normal, StudentT, Component
from .spec import CompositeSpec
from .estimators import MLE
from .roles import Role

__all__: list[str] = [
    "ARMA",
    "Component",
    "CompositeSpec",
    "GARCH",
    "MLE",
    "Normal",
    "Role",
    "StudentT",
    "__version__",
]