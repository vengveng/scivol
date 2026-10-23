"""Shared profile fixtures; causal ARX cases are declared in profile_arx_cases."""

from __future__ import annotations

from copy import deepcopy

import numpy as np

from .mean_cases import LAYOUTS, arma_case, layout_case, linear_case, scored_suffix


REQUESTS = (1, 2, 3, 4, 5, 6, 7)


def profile_case(mean_case):
    result = deepcopy(mean_case)
    statistical_count = len(result["parameters"])
    result.update(parameters=result["parameters"][:-1].copy(),
                  theta_names=result["theta_names"][:-1], density="normal",
                  scope="exact_normal_variance_profile", profile="exact_profile",
                  statistical_parameter_count=statistical_count,
                  nuisance_names=("sigma2",), fixed=True,
                  case_id="profile-" + result["case_id"])
    result["derivative_width"] = len(result["parameters"])
    return result


def ordinary_cases():
    return tuple(profile_case(layout_case(layout, "normal")) for layout in LAYOUTS)


def original_z_cases():
    result = []
    for case in ordinary_cases():
        if case["case_id"] not in {
            "profile-arma11-normal", "profile-arma32-normal", "profile-arma00-no-intercept-normal",
            "profile-arma04-no-intercept-normal", "profile-arma40-no-intercept-normal",
            "profile-linear-three-plus-intercept-normal", "profile-linear-four-explicit-constant-normal",
            "profile-linear-wide-seventeen-normal",
        }:
            continue
        declared = deepcopy(case)
        z = case["parameters"].copy()
        if case["model"] == "arma":
            start = int(case["intercept"]) + case["p"]
            z[start:] *= -1
        declared.update(parameters=z.copy(), original_z=z.copy(), transformed=True,
                        case_id=case["case_id"] + "-original-z")
        result.append(declared)
    return tuple(result)


def expected_dispatch(case):
    if not case.get("fixed", True) or not case["intercept"] or case.get("linear_ar_order", 0):
        return 0
    if case["model"] == "arma" and case["p"] == case["q"] == 1:
        return 2
    if case["model"] == "linear" and case["design"].shape[1] == 3:
        return 1
    return 0


def suffix_cases():
    return tuple(scored_suffix(case, 7, 11, hold_back=2) for case in ordinary_cases())


def long_horizon_case():
    return profile_case(arma_case("normal", 0, 1, intercept=False,
                                  parameters=[-0.995, 1.3], y=np.ones(96),
                                  case_id="ma-near-boundary-96"))


def zero_sse_case(model, columns=0):
    if model == "arma":
        declared = arma_case("normal", 1, 1, intercept=False, y=np.zeros(6),
                             parameters=[0.2, -0.3, 1.3])
    else:
        declared = linear_case("normal", columns, intercept=False, y=np.zeros(6),
                               design=np.zeros((6, columns)),
                               parameters=[0.2] * columns + [1.3])
    return profile_case(declared)


def range_case(scale, *, quotient_rounding=False):
    y = [scale, -scale, 0.0 if quotient_rounding else scale]
    return profile_case(linear_case("normal", 0, intercept=False, y=y, parameters=[1.3]))


def tail_case(z):
    result = profile_case(arma_case("normal", 1, 0, parameters=[0.1, 0.2, 1.3]))
    result.update(parameters=np.array([0.1, z]), original_z=np.array([0.1, z]), transformed=True,
                  case_id=f"profile-ar-tail-{z}")
    return result


def exact_binary_range_cases():
    cases = []
    inputs = (
        ("finite-variance-sse-sum-overflow", [2.0 ** 511] * 4, np.empty((4, 0)), []),
        ("finite-variance-square-overflow", [2.0 ** 512, 0.0, 0.0, 0.0], np.empty((4, 0)), []),
        ("recoverable-subnormal-double-round", [2.0 ** -537, 2.0 ** -538], np.empty((2, 0)), []),
        ("subnormal-gradient-erased-by-division", [2.0 ** 500] * 4, np.full((4, 1), 2.0 ** -575), [0.0]),
        ("schur-raw-B-overflow-exact-zero", [1.0, 1.0], [[2.0 ** 512], [0.0]], [0.0]),
        ("schur-both-terms-overflow-exact-zero", [1.0, 1.0], [[2.0 ** 513], [0.0]], [0.0]),
        ("schur-near-cancel-positive", [1.0, np.nextafter(1.0, np.inf)], [[2.0 ** 513], [0.0]], [0.0]),
        ("schur-near-cancel-negative", [1.0, np.nextafter(1.0, 0.0)], [[2.0 ** 513], [0.0]], [0.0]),
    )
    for label, y, design, parameters in inputs:
        case = profile_case(linear_case("normal", intercept=False, design=design, y=y,
                                        parameters=parameters + [1.3], case_id=label))
        case["range_requirement"] = label
        cases.append(case)
    return tuple(cases)
