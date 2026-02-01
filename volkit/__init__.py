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

from .components import ARMA, GARCH, Normal, StudentT, SkewT, AutoDensity, Component
from .spec import CompositeSpec
from .estimators import MLE, QMLE
from .roles import Role
# from ._kernels import get_routine

__all__: list[str] = [
    "ARMA",
    "AutoDensity",
    "Component",
    "CompositeSpec",
    "GARCH",
    # "get_routine",
    "MLE",
    "Normal",
    "QMLE",
    "Role",
    "SkewT",
    "StudentT",
    "__version__",
]