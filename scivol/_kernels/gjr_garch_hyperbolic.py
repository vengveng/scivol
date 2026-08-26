from __future__ import annotations

import re

from .._gh import HYPERBOLIC_LAMBDA
from ..components.density import Hyperbolic
from .garch_gh_common import build_gjr_garch_gh_routine

_UID = re.compile(r"GJR-GARCH\((\d+),(\d+)\)\+Hyperbolic$")


def get_routine(uid: str):
    match = _UID.fullmatch(uid)
    if match is None:
        raise RuntimeError(f"gjr_garch_hyperbolic cannot handle '{uid}'")
    return build_gjr_garch_gh_routine(
        int(match.group(1)),
        int(match.group(2)),
        Hyperbolic,
        fixed_lambda=HYPERBOLIC_LAMBDA,
    )
