# scivol/__init__.py
from ._version import __version__

try:
    from . import _core as _core
except ModuleNotFoundError:
    raise ImportError(
        "scivol was installed without its compiled core."
        "Build wheels or run `pip install .` with a C compiler."
    )

from .components import ARMA, ARX, AutoDensity, AutoVol, Component, EGARCH, GED, GARCH, GJRGARCH, HARX, Normal, StudentT, SkewT
from .spec import CompositeSpec
from .roles import Role
from ._settings import settings
from .dcc import CCC, CCCParams, CCCResult, DCC, DCCResult, DCCParams
# from ._kernels import get_routine

__all__: list[str] = [
    "ARMA",
    "ARX",
    "AutoDensity",
    "AutoVol",
    "Component",
    "CompositeSpec",
    "CCC",
    "CCCParams",
    "CCCResult",
    "DCC",
    "DCCParams",
    "DCCResult",
    "EGARCH",
    "GED",
    "GARCH",
    "GJRGARCH",
    "HARX",
    # "get_routine",
    "Normal",
    "Role",
    "settings",
    "SkewT",
    "StudentT",
    "__version__",
]
