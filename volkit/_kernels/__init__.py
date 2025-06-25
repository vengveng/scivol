# volkit/_kernels/__init__.py
from __future__ import annotations
from importlib import import_module
import re
from typing import Dict

from .routine import Routine

_ROUTINES: Dict[str, Routine] = {}          # uid -> Routine


def _family_module(uid: str) -> str:
    """
    Convert canonical UID to family module path.

        "ARMA(1,1)+GARCH(1,1)+Normal" -> volkit._kernels.arma_garch_normal
        "GARCH(2,1)+Normal"           -> volkit._kernels.garch_normal
        "EGARCH(1,1)+StudentT"        -> volkit._kernels.egarch_studentt
    """
    parts = re.split(r"\+|\(", uid)              # split on '+' and '('
    family = [p.lower() for p in parts if p and "," not in p]
    return f"{__name__}." + "_".join(family)     # __name__ == volkit._kernels


def get_routine(uid: str) -> Routine:
    # 1. cache hit ----------------------------------------------------
    try:
        return _ROUTINES[uid]
    except KeyError:
        pass

    # 2. import the family module ------------------------------------
    mod_name = _family_module(uid)               # e.g. volkit._kernels.garch_normal
    mod = import_module(mod_name)

    # 3. delegate UID parsing to the module itself -------------------
    try:
        routine = mod.get_routine(uid)           # every module must expose this
    except AttributeError:                       # defensive
        raise RuntimeError(f"Module '{mod_name}' does not expose get_routine()")

    _ROUTINES[uid] = routine
    return routine