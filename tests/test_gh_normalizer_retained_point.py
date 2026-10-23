"""Retained ARX-GH normalization, exact-input MP values, and MP forward AD."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import mpmath as mp
import numpy as np
import pytest

from scivol import APARCH, ARX, GeneralizedHyperbolic, _core
from scivol._joint_mean_fit import JointMeanModel
from scivol._validation import prepare_series_input
from test_aparch42_gh_complete_stress import (
    BesselIntegral, G_ATOL, G_RTOL, H_ATOL, H_RTOL, Jet, NLL_ATOL, NormalizedGH,
)
from test_native_aparch_subset_curvature import exp_jet, jet_outputs, log_jet
from test_vg_retained_value_accuracy import exact_joint_paths


FIXTURE = Path(__file__).with_name("gh_normalizer_retained_point.json")


class ScalarBessel:
    def logk(self, order, argument):
        return Jet.constant(mp.log(mp.besselk(order.value, argument.value)), 0)


@pytest.fixture(scope="module")
def retained_case():
    saved = json.loads(FIXTURE.read_text())
    recipe = saved["recipe"]
    rng = np.random.default_rng(recipe["seed"])
    x = rng.normal(size=(recipe["observations"], recipe["exogenous_columns"]))
    data = (rng.standard_t(recipe["student_df"], size=recipe["observations"])
            * recipe["student_scale"] + x @ np.array(recipe["exogenous_coefficients"]))
    for t in range(1, len(data)):
        data[t] += recipe["autoregression"] * data[t - 1]
    spec = ARX(1) + APARCH(1, 1) + GeneralizedHyperbolic()
    sample = prepare_series_input(spec, data, x=x)
    model = JointMeanModel(spec, sample.data, sample.x)
    for name, array in (("data", model.data), ("exogenous", sample.x),
                        ("design", model.features)):
        assert hashlib.sha256(array.tobytes()).hexdigest() == saved["sample"][name + "_sha256"]
    assert model.h0 == saved["sample"]["h0"]
    assert len(model.data) == saved["sample"]["n_eff"] == 95
    assert model.first == saved["sample"]["first_scored"] == 0
    assert model.columns == saved["sample"]["columns"] == 4
    return saved, model, sample.x


def independent_value(model, theta, digits):
    with mp.workdps(digits):
        density = NormalizedGH.prepare(
            *(Jet.constant(mp.mpf(float(v)), 0) for v in theta[-3:]), ScalarBessel())
        residuals, variances = exact_joint_paths(model, theta)
        return mp.fsum(mp.log(h) / 2
                       - density.logpdf(Jet.constant(e / mp.sqrt(h), 0)).value
                       for e, h in zip(residuals, variances))


def independent_joint_jet(model, theta, digits, count):
    with mp.workdps(digits):
        point = [Jet.variable(mp.mpf(float(v)), model.size, i)
                 for i, v in enumerate(theta)]
        m, p, q = model.columns, model.p_vol, model.q_vol
        delta = point[model.shape_start - 1]
        residuals = [Jet.constant(mp.mpf(float(y)), model.size)
                     - sum((mp.mpf(float(x)) * coefficient
                            for x, coefficient in zip(row, point[:m])),
                           Jet.constant(0, model.size))
                     for y, row in zip(model.data, model.features)]
        density = NormalizedGH.prepare(*point[-3:], BesselIntegral(count, digits))
        h0 = mp.mpf(float(model.h0))
        powers = [exp_jet(delta * mp.log(h0) / 2)]
        total = Jet.constant(0, model.size)
        for t, residual in enumerate(residuals):
            if t:
                power = point[m]
                for lag in range(min(t, p)):
                    previous = residuals[t - 1 - lag]
                    assert previous.value != 0
                    shock = ((previous if previous.value > 0 else -previous)
                             - point[m + 1 + p + lag] * previous)
                    power += point[m + 1 + lag] * exp_jet(delta * log_jet(shock))
                for lag in range(min(t, q)):
                    power += point[m + 1 + 2 * p + lag] * powers[t - 1 - lag]
                powers.append(power)
                log_scale = log_jet(power) / delta
            else:
                log_scale = Jet.constant(mp.log(h0) / 2, model.size)
            total += log_scale - density.logpdf(residual * exp_jet(-log_scale))
        return total


@pytest.fixture(scope="module")
def retained_reference(retained_case):
    saved, model, _ = retained_case
    theta = np.array(saved["theta"])
    low, high = [independent_value(model, theta, digits) for digits in (80, 120)]
    with mp.workdps(120):
        assert abs(low - high) < mp.mpf("1e-60")
        assert abs(high - mp.mpf(saved["reference"])) < mp.mpf("1e-100")
    return float(high)


@pytest.fixture(scope="module")
def retained_joint_ad(retained_case, retained_reference):
    saved, model, _ = retained_case
    low, high = [independent_joint_jet(model, saved["theta"], digits, count)
                 for digits, count in ((80, 32), (120, 48))]
    with mp.workdps(120):
        refinement = max(abs(a - b) / (1 + abs(b)) for a, b in zip(
            (low.value, *low.gradient, *low.hessian),
            (high.value, *high.gradient, *high.hessian)))
        assert refinement < mp.mpf("1e-24"), str(refinement)
        assert abs(high.value - mp.mpf(retained_case[0]["reference"])) < mp.mpf("1e-24")
    return jet_outputs(high)


def assert_outputs(actual, expected, mask):
    for bit, output, reference in zip((1, 2, 4), actual[:3], expected[:3]):
        if not mask & bit:
            assert output is None
            continue
        assert np.isfinite(output).all() and np.isfinite(reference).all()
        np.testing.assert_allclose(output, reference,
            rtol=0 if bit == 1 else (G_RTOL if bit == 2 else H_RTOL),
            atol=NLL_ATOL if bit == 1 else (G_ATOL if bit == 2 else H_ATOL))


@pytest.mark.parametrize("fixed", [False, True], ids=["runtime", "fixed"])
@pytest.mark.parametrize("transformed,paths", [(False, False), (False, True), (True, False)],
                         ids=["theta", "theta-paths", "original-z"])
def test_retained_arx_gh_all_value_masks(retained_case, retained_reference, fixed, transformed, paths):
    saved, template, x = retained_case
    model = JointMeanModel(template.spec, template.data, x, fixed=fixed)
    theta = np.array(saved["theta"])
    # Keep the captured original z; inverse(theta) loses its rounded-tail information.
    point = np.array(saved["coordinates"]) if transformed else theta
    full = model.evaluate(point, 7, transformed=transformed, paths=paths)
    for mask in range(1, 8):
        actual = model.evaluate(point, mask, transformed=transformed, paths=paths)
        assert_outputs(actual, full, mask)
        if mask & 1:
            assert abs(actual[0] - retained_reference) <= NLL_ATOL, (fixed, transformed, paths, mask)
        if paths:
            with mp.workdps(120):
                residuals, variances = exact_joint_paths(model, theta)
            np.testing.assert_allclose(actual[3], np.array(list(map(float, residuals))), rtol=0, atol=2e-15)
            np.testing.assert_allclose(actual[4], np.array(list(map(float, variances))), rtol=3e-10, atol=2e-11)


@pytest.mark.parametrize("fixed", [False, True], ids=["runtime", "fixed"])
def test_retained_arx_gh_theta_all_masks_match_mp_ad(retained_case, retained_joint_ad, fixed):
    saved, template, x = retained_case
    model = JointMeanModel(template.spec, template.data, x, fixed=fixed)
    for mask in range(1, 8):
        assert_outputs(model.evaluate(saved["theta"], mask), retained_joint_ad, mask)


@pytest.mark.parametrize("lam", [None, -.5, 1.], ids=["gh", "nig", "hyperbolic"])
def test_legacy_standardized_normalizer_matches_independent_mp(retained_case, lam):
    saved, _, _ = retained_case
    shape = np.array(saved["theta"][-3:])
    if lam is not None:
        shape[-1] = lam
    values = np.array([-.9, .27, 1.2])
    references = []
    for digits in (80, 120):
        with mp.workdps(digits):
            density = NormalizedGH.prepare(
                *(Jet.constant(mp.mpf(float(v)), 0) for v in shape), ScalarBessel())
            references.append(-mp.fsum(density.logpdf(Jet.constant(mp.mpf(float(x)), 0)).value
                                       for x in values))
    with mp.workdps(120):
        assert abs(references[0] - references[1]) < mp.mpf("1e-60")
    actual = _core._gh_standardized_nll(shape.ctypes.data, values.ctypes.data, len(values))
    assert np.isfinite(actual)
    assert abs(actual - float(references[1])) <= NLL_ATOL
