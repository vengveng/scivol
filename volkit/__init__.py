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

from .components import ARMA, GARCH, GJRGARCH, Normal, StudentT, SkewT, AutoDensity, AutoVol, Component
from .spec import CompositeSpec
from .roles import Role
from ._settings import settings
from .dcc import DCC, DCCResult, DCCParams
# from ._kernels import get_routine

__all__: list[str] = [
    "ARMA",
    "AutoDensity",
    "AutoVol",
    "Component",
    "CompositeSpec",
    "DCC",
    "DCCParams",
    "DCCResult",
    "GARCH",
    "GJRGARCH",
    # "get_routine",
    "Normal",
    "Role",
    "settings",
    "SkewT",
    "StudentT",
    "__version__",
]
