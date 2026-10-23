"""Mapped linked005 failure and signed centre stresses, with no domain changes."""

from __future__ import annotations

import numpy as np

from .power_cases import case, with_parameters


LOW_NU = 2.854355244468529
MAPPED_SKEW = 0.2913126124515909
ABSOLUTE_POINTS = (
    ("mapped-positive", (LOW_NU, MAPPED_SKEW)),
    ("mapped-negative", (LOW_NU, -MAPPED_SKEW)),
    ("zero", (LOW_NU, 0.0)),
    ("small-positive", (LOW_NU, 1e-6)),
    ("small-negative", (LOW_NU, -1e-6)),
    ("tiny-positive", (LOW_NU, 1e-10)),
    ("tiny-negative", (LOW_NU, -1e-10)),
    ("student-limit-control", (8.0, 0.0)),
)
POWER_POINTS = (
    ("low-nu-sublinear-positive", (-0.16, 0.7, LOW_NU, MAPPED_SKEW)),
    ("low-nu-sublinear-negative", (0.16, 0.7, LOW_NU, -MAPPED_SKEW)),
    ("low-nu-centre", (0.16, 0.7, LOW_NU, 0.0)),
    ("low-nu-small-positive", (0.16, 0.7, LOW_NU, 1e-6)),
    ("low-nu-small-negative", (0.16, 0.7, LOW_NU, -1e-6)),
    ("fractional-power", (-0.16, 0.37, 8.0, -MAPPED_SKEW)),
    ("noninteger-superlinear", (-0.16, 1.7, 8.0, MAPPED_SKEW)),
)
OBSERVATIONS = np.array([.3, -.7, 1.1, -.2, .5, -1.3, .9, .4, -.6, .8, -.4, 1.2])


def linked_case(p, q):
    z = np.array([-.8, *([.09 / p] * p), *([-.04 / p] * p),
                  *([.25] * q), .3, .3])
    declared = case("egarch", "skewt", p, q, parameters=z,
                    y=OBSERVATIONS, h0=.8, first=2, stop=11,
                    case_id=f"egarch-skewt-{p}{q}-linked005-low-nu")
    return with_parameters(declared, z, transformed=True, original_z=z.copy(),
                           chart="power_unshifted_exact_slack_v1")
