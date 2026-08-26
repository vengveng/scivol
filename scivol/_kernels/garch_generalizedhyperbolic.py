from __future__ import annotations

import re
from typing import Dict, Tuple

from ..components.density import GeneralizedHyperbolic
from .garch_gh_common import build_garch_gh_routine
from .routine import Routine

_CACHE: Dict[Tuple[int, int], Routine] = {}
_UID = re.compile(r"GARCH\((\d+),(\d+)\)\+GeneralizedHyperbolic$")


def get_routine(uid: str) -> Routine:
    match = _UID.fullmatch(uid)
    if match is None:
        raise RuntimeError(f"garch_generalizedhyperbolic cannot handle '{uid}'")
    key = (int(match.group(1)), int(match.group(2)))
    if key not in _CACHE:
        _CACHE[key] = build_garch_gh_routine(
            *key, GeneralizedHyperbolic, fixed_lambda=None
        )
    return _CACHE[key]
