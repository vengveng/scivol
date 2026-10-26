# scivol/__init__.py
from ._version import __version__

try:
    from . import _scivolapi as _scivolapi
except ModuleNotFoundError:
    raise ImportError(
        "scivol was installed without its compiled library. "
        "Build wheels or run `pip install .` with a C++20 compiler."
    )

from .components import (
    ARMA,
    APARCH,
    ARX,
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

__all__: list[str] = [
    "ARMA",
    "APARCH",
    "ARX",
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

from .selection import Auto, Search, GeneralToSpecific, InformationCriterion, ConfidenceSet, SelectionResult

__all__ += ["Auto", "Search", "GeneralToSpecific", "InformationCriterion", "ConfidenceSet", "SelectionResult"]
