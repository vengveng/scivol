from __future__ import annotations

import re

from ..components.density import VarianceGamma
from .garch_gh_common import build_gjr_garch_gh_routine

_UID = re.compile(r"GJR-GARCH\((\d+),(\d+)\)\+VarianceGamma$")


def get_routine(uid: str):
    match = _UID.fullmatch(uid)
    if match is None:
        raise RuntimeError(f"gjr_garch_variancegamma cannot handle '{uid}'")
    return build_gjr_garch_gh_routine(
        int(match.group(1)), int(match.group(2)), VarianceGamma,
        fixed_lambda=None, variance_gamma=True,
    )
