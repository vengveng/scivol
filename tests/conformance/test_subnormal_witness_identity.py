"""Retain the old raw-input witness and distinguish its two objective meanings."""

import json
import math

import mpmath as mp
import numpy as np
import pytest

from fixtures.garch_cases import case
from oracles.values import mp_value


def original_witness(p, q):
    data = np.full(8, 1e-160)
    theta = np.r_[1e-307, 0.1, np.zeros(p - 1), 0.8, np.zeros(q - 1)]
    seed = float(np.mean(data * data))
    return case("garch", p=p, q=q, parameters=theta, y=data, h0=seed,
                case_id=f"legacy-subnormal-1e-160-garch{p}{q}")


def rounded_binary64_diagnostic(declared):
    data, theta, p, q = (declared[key] for key in ("y", "parameters", "p", "q"))
    variances = [declared["h0"]]
    for time in range(1, data.size):
        variance = theta[0]
        for lag in range(min(p, time)):
            variance += theta[1 + lag] * data[time - lag - 1] ** 2
        for lag in range(min(q, time)):
            variance += theta[1 + p + lag] * variances[time - lag - 1]
        variances.append(variance)
    return math.fsum(0.5 * (math.log(2.0 * math.pi) + math.log(h) + float(e * e) / h)
                     for e, h in zip(data, variances))


def compare_reference_objectives(declared):
    diagnostic = rounded_binary64_diagnostic(declared)
    low = mp_value(declared, digits=320)
    raw = mp_value(declared, digits=400)
    with mp.workdps(400):
        assert abs(low - raw) <= mp.mpf("1e-280")
        theta = tuple(mp.mpf(float(value)) for value in declared["parameters"])
        squared = tuple(mp.mpf(float(value)) for value in declared["y"] * declared["y"])
        exact_square = mp.mpf(float(declared["y"][0])) ** 2
        path = [mp.mpf(declared["h0"])]
        p, q = declared["p"], declared["q"]
        for time in range(1, len(squared)):
            path.append(theta[0]
                        + mp.fsum(theta[1 + lag] * squared[time - lag - 1] for lag in range(min(p, time)))
                        + mp.fsum(theta[1 + p + lag] * path[time - lag - 1] for lag in range(min(q, time))))
        rounded_mp = mp.fsum((mp.log(2 * mp.pi) + mp.log(h) + e2 / h) / 2
                            for e2, h in zip(squared, path))
        discrepancy = raw - rounded_mp
        evidence = {
            "case": declared["case_id"], "p": p, "q": q, "request": 1,
            "y_hex": [float(value).hex() for value in declared["y"]],
            "theta_hex": [float(value).hex() for value in declared["parameters"]],
            "h0_hex": float(declared["h0"]).hex(), "scored_rows": list(declared["scored_rows"]),
            "seed_policy": "Original binary64 mean(data*data), held fixed in both objectives",
            "rounded_square_hex": float(squared[0]).hex(),
            "raw_square_decimal": mp.nstr(exact_square, 90),
            "square_difference_decimal": mp.nstr(exact_square - squared[0], 90),
            "old_binary64_diagnostic_hex": diagnostic.hex(),
            "rounded_square_mp_value": mp.nstr(rounded_mp, 90),
            "raw_residual_mp_value": mp.nstr(raw, 90),
            "raw_minus_rounded_square_mp": mp.nstr(discrepancy, 90),
            "raw_reference_refinement_difference": mp.nstr(abs(low - raw), 90),
            "old_absolute_tolerance": 1e-12, "raw_absolute_tolerance": 1e-12,
            "classification": ("rounded-square reference is not the raw-residual likelihood at the old tolerance"
                               if abs(discrepancy) > mp.mpf("1e-12") else
                               "rounding changes the objective but is within the old tolerance at this witness"),
        }
        assert exact_square != squared[0]
    return float(raw), evidence


@pytest.mark.parametrize("p,q", ((1, 1), (3, 2)))
def test_original_subnormal_point_rounded_intermediate_reference_diagnostic(p, q):
    declared = original_witness(p, q)
    assert 0.0 < declared["h0"] < np.finfo(float).tiny
    _, evidence = compare_reference_objectives(declared)
    evidence["role"] = "reference_diagnostic_only_no_native_qualification"
    print("SUBNORMAL_WITNESS_EVIDENCE " + json.dumps(evidence, sort_keys=True, allow_nan=False))


@pytest.mark.parametrize("p,q", ((1, 1), (3, 2)))
def test_original_subnormal_point_matches_raw_residual_mp(native, p, q):
    declared = original_witness(p, q)
    expected, evidence = compare_reference_objectives(declared)
    evidence["role"] = "native_raw_residual_mathematical_comparison"
    try:
        actual = native(declared, 1)
    except ValueError as error:
        evidence.update(native_status="preparation_rejected", error=str(error))
        print("SUBNORMAL_WITNESS_EVIDENCE " + json.dumps(evidence, sort_keys=True, allow_nan=False))
        raise
    difference = float(actual["f"]) - expected
    evidence.update(native_status=int(actual["status"]), valid_mask=int(actual["valid_mask"]),
                    actual_f_hex=float(actual["f"]).hex(), raw_reference_f_hex=expected.hex(),
                    raw_difference_hex=difference.hex(),
                    absolute_raw_difference=abs(difference) if math.isfinite(difference) else None)
    print("SUBNORMAL_WITNESS_EVIDENCE " + json.dumps(evidence, sort_keys=True, allow_nan=False))
    assert actual["status"] == 0 and actual["valid_mask"] == 1, evidence
    assert math.isfinite(actual["f"]) and abs(difference) <= 1e-12, evidence
    for name in ("g", "h"):
        np.testing.assert_array_equal(actual[name], np.full_like(actual[name], actual["sentinel"]))
