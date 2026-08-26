# scivol/__init__.py
from ._version import __version__

try:
    from . import _core as _core
except ModuleNotFoundError:
    raise ImportError(
        "scivol was installed without its compiled core."
        "Build wheels or run `pip install .` with a C compiler."
    )

from .components import (
    ARMA,
    APARCH,
    ARX,
    AutoDensity,
    AutoMean,
    AutoVol,
    Component,
    EGARCH,
    GED,
    GARCH,
    GJRGARCH,
    GeneralizedHyperbolic,
    HARX,
    Hyperbolic,
    Normal,
    NormalInverseGaussian,
    SkewT,
    StudentT,
    VarianceGamma,
)
from .spec import CompositeSpec
from .roles import Role
from ._settings import settings
from .dcc import (
    CCC,
    CCCParams,
    CCCResult,
    DCC,
    DCCParams,
    DCCResult,
    MultivariateScenarioResult,
    RollingDCCResult,
    RollingDCCWindow,
)
from .batch import BatchFitFailure, BatchFitResult, FitTimeoutError, RollingFitResult, RollingFitWindow
# from ._kernels import get_routine

__all__: list[str] = [
    "ARMA",
    "APARCH",
    "ARX",
    "AutoDensity",
    "AutoMean",
    "AutoVol",
    "BatchFitFailure",
    "BatchFitResult",
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
    "GeneralizedHyperbolic",
    "HARX",
    "Hyperbolic",
    "FitTimeoutError",
    "MultivariateScenarioResult",
    # "get_routine",
    "Normal",
    "NormalInverseGaussian",
    "Role",
    "RollingFitResult",
    "RollingFitWindow",
    "RollingDCCResult",
    "RollingDCCWindow",
    "settings",
    "SkewT",
    "StudentT",
    "VarianceGamma",
    "__version__",
]
