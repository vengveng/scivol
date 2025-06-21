# from __future__ import annotations

# import importlib
# import warnings
# from typing import Callable, Dict, Optional

# import numpy as np

# from volkit import CompositeSpec, GARCH, Role

# # ------------------------------------------------------------------ #
# # Internal state
# # ------------------------------------------------------------------ #
# _SPECIAL_REGISTRY: Dict[str, Callable[[np.ndarray, np.ndarray], float]] = {}
# _GENERAL_KERNEL: Optional[Callable[[np.ndarray, np.ndarray, CompositeSpec], float]] = None


# # ------------------------------------------------------------------ #
# # Registration helpers (internal)
# # ------------------------------------------------------------------ #
# def _register_special(signature: str, func: Callable[[np.ndarray, np.ndarray], float]):
#     _SPECIAL_REGISTRY[signature] = func

# def _set_general_kernel(func: Callable[[np.ndarray, np.ndarray, CompositeSpec], float]):
#     global _GENERAL_KERNEL
#     _GENERAL_KERNEL = func


# # ------------------------------------------------------------------ #
# # 1. Try to load the compiled extension
# # ------------------------------------------------------------------ #
# try:
#     _core = importlib.import_module("volkit._core")  # the binary module
#     HAVE_CORE = True
# except ModuleNotFoundError:
#     _core = None
#     HAVE_CORE = False
#     warnings.warn(
#         "volkit C extension not found - falling back to pure-Python kernels. "
#         "Performance will be degraded.",
#         RuntimeWarning,
#     )

# # ------------------------------------------------------------------ #
# # 2a. Register compiled kernels (if present)
# # ------------------------------------------------------------------ #
# if HAVE_CORE:
#     # Exact-match kernels -----------------------------------------------------
#     if hasattr(_core, "garch11_normal_loglik"):

#         def _garch11_normal(data: np.ndarray, theta: np.ndarray) -> float:
#             return _core.garch11_normal_loglik(data, theta)

#         _register_special("GARCH(1,1)+Normal", _garch11_normal)

#     if hasattr(_core, "arma11_garch11_normal_loglik"):

#         def _arma11_garch11_normal(data: np.ndarray, theta: np.ndarray) -> float:
#             return _core.arma11_garch11_normal_loglik(data, theta)

#         _register_special("ARMA(1,1)+GARCH(1,1)+Normal", _arma11_garch11_normal)

#     # General kernel ---------------------------------------------------------
#     if hasattr(_core, "garch_pq_normal_loglik"):

#         def _general(
#             data: np.ndarray, theta: np.ndarray, spec: CompositeSpec
#         ) -> float:
#             vol = spec.get_component(Role.VOLATILITY)
#             assert isinstance(vol, GARCH)
#             return _core.garch_pq_normal_loglik(data, theta, vol.p, vol.q)

#         _set_general_kernel(_general)

# # ------------------------------------------------------------------ #
# # 2b. Otherwise register slow Python fall-backs
# # ------------------------------------------------------------------ #
# else:
#     from .fallback import (  # pytype: disable=import-error
#         garch11_normal_loglik_py,
#         arma11_garch11_normal_loglik_py,
#         garch_pq_normal_loglik_py,
#     )

#     _register_special("GARCH(1,1)+Normal", garch11_normal_loglik_py)
#     _register_special("ARMA(1,1)+GARCH(1,1)+Normal", arma11_garch11_normal_loglik_py)
#     _set_general_kernel(garch_pq_normal_loglik_py)


# # ------------------------------------------------------------------ #
# # Public API
# # ------------------------------------------------------------------ #
# def get_special_kernel(
#     signature: str,
# ) -> Optional[Callable[[np.ndarray, np.ndarray], float]]:
#     """Return a fast, signature-specific kernel or None."""
#     return _SPECIAL_REGISTRY.get(signature)


# def get_general_kernel() -> Optional[
#     Callable[[np.ndarray, np.ndarray, CompositeSpec], float]
# ]:
#     """Return the general (p,q) kernel or None."""
#     return _GENERAL_KERNEL