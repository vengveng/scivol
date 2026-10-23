"""Independent positive-ARX profile chart delta; no public builder or old imports."""

from __future__ import annotations

from copy import deepcopy

import numpy as np
import pytest

from fixtures.profile_arx_cases import (arx_profile_case, arx_profile_cases, as_original_z,
                                        causal_design, plateau_case)
from fixtures.profile_cases import REQUESTS
from oracles.garch import jax_modules
from oracles.profile import (coordinate_map, full_normal_jet, jet, mp_jet,
                             mp_linear_ar_pullback, nuisance, pullback)
from test_profile_mathematics import (ProfileProblem, SENTINEL, assert_jet,
                                      assert_nuisance_same, assert_unrequested, exercise_requests)


CASES = arx_profile_cases()
CHART_CASES = tuple(as_original_z(case) for case in CASES)


def native(case, request):
    with ProfileProblem(case) as problem:
        assert problem.begin == 0
        assert problem.first == case["first_scored_original_position"]
        assert problem.count == len(case["scored_rows"])
        assert problem.width == int(case["intercept"]) + case["linear_ar_order"] + case["input_exogenous_width"]
        assert problem.statistical_count == problem.width + 1
        assert problem.dispatch == 0
        return problem.evaluate(case["parameters"], request)


@pytest.mark.parametrize("case", CASES, ids=lambda case: case["case_id"])
def test_arx_profile_independent_causal_fixture(case):
    order, hold = case["linear_ar_order"], case["hold_back"]
    assert case["model"] == "linear" and case["p"] == case["q"] == case["recursion_begin"] == 0
    np.testing.assert_array_equal(case["design"][hold, :order], np.zeros(order))
    for row in range(hold, case["stop"]):
        for lag in range(1, order + 1):
            expected = case["y"][row - lag] if row - lag >= hold else 0
            assert case["design"][row, lag - 1] == expected
        np.testing.assert_array_equal(case["design"][row, order:], case["input_exogenous"][row])
    assert case["design"].shape[1] == order + case["input_exogenous_width"]
    if order == 2 and case["intercept"]:
        # Stable AR coefficients need not lie in a componentwise unit box.
        assert case["parameters"][1] > 1


@pytest.mark.parametrize("case", CASES + CHART_CASES, ids=lambda case: case["case_id"])
def test_arx_profile_independent_jax_all_masks(case):
    actual = native(case, 7)
    expected = jet(case, theta=actual["theta"])
    if case["transformed"]:
        expected = pullback(case, expected)
        jax, jnp, _ = jax_modules()
        mapping = coordinate_map(case)
        with ProfileProblem(case) as problem:
            mapped = problem.map(case["original_z"])
        assert mapped["status"] == 0
        np.testing.assert_allclose(mapped["theta"], np.asarray(mapping(jnp.asarray(case["original_z"]))),
                                   rtol=4e-14, atol=4e-15)
        expected_j = np.asarray(jax.jacfwd(mapping)(jnp.asarray(case["original_z"])))
        np.testing.assert_allclose(mapped["jacobian"], expected_j, rtol=4e-12, atol=3e-14)
    exercise_requests(native, case, expected)
    if case["intercept"] and case["input_exogenous_width"]:
        ar = 1
        exog = ar + case["linear_ar_order"]
        assert abs(expected[2][ar, 0]) > 1e-7
        assert abs(expected[2][ar, exog]) > 1e-7
    runtime = deepcopy(case)
    runtime["fixed"] = False
    for mask in REQUESTS:
        left, right = native(case, mask), native(runtime, mask)
        assert_jet(left, expected, mask)
        assert_jet(right, expected, mask)
        assert_nuisance_same(left["nuisance"], right["nuisance"])


@pytest.mark.parametrize("case", CASES, ids=lambda case: case["case_id"])
def test_arx_profile_full_normal_schur(case):
    actual = native(case, 7)
    f, g, h = full_normal_jet(case, actual["theta"], actual["nuisance"]["variance"])
    assert abs(g[-1]) < 2e-12
    schur = h[:-1, :-1] - np.outer(h[:-1, -1], h[-1, :-1]) / h[-1, -1]
    assert_jet(actual, (f, g[:-1], schur), 7)
    assert np.max(abs(h[:-1, -1])) > 1e-5
    assert not np.allclose(schur, h[:-1, :-1], rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("transformed", (False, True))
def test_arx_profile_independent_mp_full_cross_all_masks(transformed):
    case = arx_profile_case()
    if transformed:
        case = as_original_z(case)
    actual = native(case, 7)
    theta_mp = mp_jet(case, actual["theta"])
    expected = theta_mp
    if transformed:
        target = mp_linear_ar_pullback(case, actual["theta"])
        expected = target["jet"]
        j = target["jacobian"]
        assert np.max(abs(expected[2] - j.T @ theta_mp[2] @ j)) > 1e-6
    exercise_requests(native, case, expected)
    jax_expected = jet(case, theta=actual["theta"])
    if transformed:
        jax_expected = pullback(case, jax_expected)
    for left, right in zip(jax_expected, expected):
        np.testing.assert_allclose(left, right, rtol=4e-11, atol=3e-11)


@pytest.mark.parametrize("transformed", (False, True))
def test_arx_profile_scoring_preserves_retained_history(transformed):
    case = arx_profile_case()
    if transformed:
        case = as_original_z(case)
    baseline = native(case, 7)
    excluded = deepcopy(case)
    excluded["y"][:case["hold_back"]] *= -2
    excluded["y"][case["stop"]:] *= -3
    excluded["input_exogenous"][:case["hold_back"]] *= -2
    excluded["input_exogenous"][case["stop"]:] *= -3
    excluded["design"] = causal_design(excluded["y"], excluded["input_exogenous"],
                                        case["linear_ar_order"], case["hold_back"])
    retained = deepcopy(case)
    first = case["first_scored_original_position"]
    assert first > case["hold_back"]
    retained["y"][first - 1] += 0.5
    retained["design"] = causal_design(retained["y"], retained["input_exogenous"],
                                        case["linear_ar_order"], case["hold_back"])
    assert retained["design"][first, 0] == retained["y"][first - 1]
    for variant in (excluded, retained):
        actual = native(variant, 7)
        expected = jet(variant, theta=actual["theta"])
        if transformed:
            expected = pullback(variant, expected)
        exercise_requests(native, variant, expected)
        if variant is excluded:
            for name in ("f", "g", "h"):
                np.testing.assert_array_equal(actual[name], baseline[name])
            assert_nuisance_same(actual["nuisance"], baseline["nuisance"])
        else:
            assert abs(actual["f"] - baseline["f"]) > 1e-7


@pytest.mark.parametrize("sign", (-1, 1))
def test_arx_profile_original_z_mp_plateau(sign):
    cases = (plateau_case(sign, 18.0), plateau_case(sign, 18.05))
    mapped, full, expected = [], [], []
    for case in cases:
        with ProfileProblem(case) as problem:
            mapped.append(problem.map(case["original_z"]))
        assert mapped[-1]["status"] == 0 and abs(mapped[-1]["theta"][1]) < 1
        target = mp_linear_ar_pullback(case, mapped[-1]["theta"], digits=100)
        expected.append(target)
        full.append(native(case, 7))
        assert_jet(full[-1], target["jet"], 7)
        assert target["jet"][1][1] != 0 and target["jet"][2][1, 1] != 0
        assert np.max(abs(target["jet"][2][1, [0, 2, 3]])) > 0
        np.testing.assert_allclose(mapped[-1]["jacobian"], target["jacobian"], rtol=4e-12, atol=1e-29)
        exact = nuisance(case, mapped[-1]["theta"])
        np.testing.assert_allclose(full[-1]["nuisance"]["variance"], float(exact["variance"]), rtol=4e-14, atol=0)
        for mask in REQUESTS:
            output = native(case, mask)
            assert_jet(output, target["jet"], mask)
            assert_nuisance_same(output["nuisance"], full[-1]["nuisance"])
            if mask & 2:
                np.testing.assert_allclose(output["g"][1], target["jet"][1][1], rtol=4e-10, atol=1e-29)
            if mask & 4:
                np.testing.assert_allclose(output["h"][1, :], target["jet"][2][1, :], rtol=4e-10, atol=1e-29)
                np.testing.assert_allclose(output["h"][:, 1], target["jet"][2][:, 1], rtol=4e-10, atol=1e-29)
            for bit, name in ((1, "f"), (2, "g"), (4, "h")):
                if mask & bit:
                    np.testing.assert_allclose(output[name], full[-1][name], rtol=3e-13, atol=1e-29)
    np.testing.assert_array_equal(mapped[0]["theta"], mapped[1]["theta"])
    assert full[0]["f"] == full[1]["f"]
    assert full[0]["g"][1] != full[1]["g"][1]
    assert full[0]["h"][1, 1] != full[1]["h"][1, 1]
    assert expected[0]["jacobian"][1, 1] != expected[1]["jacobian"][1, 1]


@pytest.mark.parametrize("sign", (-1, 1))
def test_arx_profile_original_z_rounded_boundary(sign):
    case = plateau_case(sign, 20.0)
    with ProfileProblem(case) as problem:
        mapped = problem.map(case["original_z"])
    # The reflection map rejects rounded endpoints outside abs(kappa) < 1.
    # A failed map does not promise usable theta or Jacobian outputs.
    assert mapped["status"] == 5
    for mask in REQUESTS:
        output = native(case, mask)
        assert output["status"] == 5 and output["valid_mask"] == 0
        assert output["nuisance"]["variance"] is None
        for bit, reason in zip((1, 2, 4), output["reasons"]):
            if mask & bit:
                assert reason == 5
        for name in ("f", "g", "h"):
            np.testing.assert_array_equal(output[name], np.full_like(output[name], SENTINEL))


@pytest.mark.parametrize("invalid", ("ma_order", "ar_beyond_design"))
def test_arx_profile_private_descriptor_rejection(invalid):
    case = arx_profile_case()
    if invalid == "ma_order":
        case["q"] = 1
    else:
        case["linear_ar_order"] = case["design"].shape[1] + 1
    with pytest.raises(ValueError, match=r"failed \(1\)"):
        ProfileProblem(case)


@pytest.mark.parametrize("sign", (-1, 1))
def test_arx_profile_strict_theta_domain(sign):
    case = arx_profile_case()
    case["parameters"][1:3] = [sign, 0]
    for mask in REQUESTS:
        output = native(case, mask)
        assert output["status"] == 4 and output["valid_mask"] == 0
        assert output["nuisance"]["variance"] is None
        assert_unrequested(output, mask)
        for name in ("f", "g", "h"):
            np.testing.assert_array_equal(output[name], np.full_like(output[name], SENTINEL))
