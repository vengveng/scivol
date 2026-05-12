# scivol/_kernels/__init__.py
from __future__ import annotations
from importlib import import_module
import re
from typing import Dict

from .routine import Routine

_ROUTINES: Dict[str, Routine] = {}          # uid -> Routine


def _family_module(uid: str) -> str:
    """
    Convert canonical UID to family module path.

        "ARMA(1,1)+GARCH(1,1)+Normal" -> scivol._kernels.arma_garch_normal
        "GARCH(2,1)+Normal"           -> scivol._kernels.garch_normal
        "EGARCH(1,1)+StudentT"        -> scivol._kernels.egarch_studentt
        ...
    """
    if uid.startswith("ARMA(") and uid.endswith("+GED") and "+GARCH(" not in uid and "+GJR-GARCH(" not in uid and "+EGARCH(" not in uid:
        return f"{__name__}.arma_ged"
    if uid.startswith("ARMA(") and "+GARCH(" in uid and uid.endswith("+GED"):
        return f"{__name__}.arma_garch_ged"
    if uid.startswith("GARCH(") and uid.endswith("+GED"):
        return f"{__name__}.garch_ged"
    if uid.startswith("ARMA(") and "+EGARCH(" in uid and uid.endswith("+GED"):
        return f"{__name__}.arma_egarch_ged"
    if uid.startswith("EGARCH(") and uid.endswith("+GED"):
        return f"{__name__}.egarch_ged"
    if uid.startswith("GJR-GARCH(") and uid.endswith("+GED"):
        return f"{__name__}.gjr_garch_ged"
    if uid.startswith("ARMA(") and "+GJR-GARCH(" in uid and uid.endswith("+GED"):
        return f"{__name__}.arma_gjr_garch_ged"
    if (
        (uid.startswith("ARX(") or uid.startswith("HARX("))
        and "+GARCH(" not in uid
        and "+GJR-GARCH(" not in uid
        and "+EGARCH(" not in uid
        and uid.endswith("+Normal")
    ):
        return f"{__name__}.meanx_normal"
    if (
        (uid.startswith("ARX(") or uid.startswith("HARX("))
        and "+GARCH(" not in uid
        and "+GJR-GARCH(" not in uid
        and "+EGARCH(" not in uid
        and (uid.endswith("+StudentT") or uid.endswith("+SkewT") or uid.endswith("+GED"))
    ):
        return f"{__name__}.meanx_standalone"
    if (
        (uid.startswith("ARX(") or uid.startswith("HARX("))
        and "+GARCH(" in uid
        and (uid.endswith("+Normal") or uid.endswith("+StudentT") or uid.endswith("+SkewT") or uid.endswith("+GED"))
    ):
        return f"{__name__}.meanx_garch_normal"
    if (
        (uid.startswith("ARX(") or uid.startswith("HARX("))
        and "+GJR-GARCH(" in uid
        and (uid.endswith("+Normal") or uid.endswith("+StudentT") or uid.endswith("+SkewT") or uid.endswith("+GED"))
    ):
        return f"{__name__}.meanx_gjr_garch"
    if (
        (uid.startswith("ARX(") or uid.startswith("HARX("))
        and "+EGARCH(" in uid
        and (uid.endswith("+Normal") or uid.endswith("+StudentT") or uid.endswith("+SkewT") or uid.endswith("+GED"))
    ):
        return f"{__name__}.meanx_egarch"
    if "GED" in uid or uid.startswith("ARX(") or uid.startswith("HARX("):
        return f"{__name__}.generic"
    parts = re.split(r"\+|\(", uid)
    family = [p.lower().replace("-", "_") for p in parts if p and "," not in p]
    return f"{__name__}." + "_".join(family) # __name__ == scivol._kernels


def get_routine(uid: str) -> Routine:
    try:
        return _ROUTINES[uid]
    except KeyError:
        pass

    mod_name = _family_module(uid)
    # mod = import_module(mod_name)
    try:
        mod = import_module(mod_name)
    except ModuleNotFoundError as e:
        msg = f"No implementation for family '{mod_name}'. "
        msg += "Request the module to be implemented."
        raise RuntimeError(msg) from e

    try:
        routine = mod.get_routine(uid)           # every module must expose this
    except AttributeError:                       # defensive
        raise RuntimeError(f"Module '{mod_name}' does not expose get_routine()")

    _ROUTINES[uid] = routine
    return routine