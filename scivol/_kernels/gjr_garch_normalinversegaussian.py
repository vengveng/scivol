from __future__ import annotations

import re
from typing import Dict, Tuple

from ..components.density import NormalInverseGaussian
from .._gh import NIG_LAMBDA
from .garch_gh_common import build_gjr_garch_gh_routine
from .routine import Routine

_CACHE: Dict[Tuple[int, int], Routine] = {}
_UID = re.compile(r"GJR-GARCH\((\d+),(\d+)\)\+NormalInverseGaussian$")


def get_routine(uid: str) -> Routine:
    match = _UID.fullmatch(uid)
    if match is None:
        raise RuntimeError(f"gjr_garch_normalinversegaussian cannot handle '{uid}'")
    key = (int(match.group(1)), int(match.group(2)))
    if key not in _CACHE:
        _CACHE[key] = build_gjr_garch_gh_routine(
            *key, NormalInverseGaussian, fixed_lambda=NIG_LAMBDA
        )
    return _CACHE[key]
