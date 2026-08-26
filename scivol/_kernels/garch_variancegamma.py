from __future__ import annotations

import re

from ..components.density import VarianceGamma
from .garch_gh_common import build_garch_gh_routine

_UID = re.compile(r"GARCH\((\d+),(\d+)\)\+VarianceGamma$")


def get_routine(uid: str):
    match = _UID.fullmatch(uid)
    if match is None:
        raise RuntimeError(f"garch_variancegamma cannot handle '{uid}'")
    return build_garch_gh_routine(
        int(match.group(1)), int(match.group(2)), VarianceGamma,
        fixed_lambda=None, variance_gamma=True,
    )
