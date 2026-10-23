"""Shared density reductions and original-coordinate consumer regressions."""

import numpy as np
import pytest

from scivol import ARMA, ARX, GeneralizedHyperbolic, NormalInverseGaussian, _native_core
from scivol._devtools.mean_oracle import mean_value_grad_hess
from scivol._mean_fit import DENSITIES, MeanModel


@pytest.fixture(
    scope="module", params=range(1, 8),
    ids=["studentt", "skewt", "ged", "gh", "nig", "hyperbolic", "vg"],
)
def mean_shape_case(request):
    pytest.importorskip("jax")
    density = request.param
    shapes = {
        1: [7.0], 2: [7.0, .2], 3: [1.7], 4: [.12, 1.8, .4],
        5: [.12, 1.8], 6: [.12, 1.8], 7: [.12, 2.0],
    }
    data = np.array([-.8, .4, 1.2, -.3, .7, -1.1])
    exogenous = np.random.default_rng(421013).normal(size=(len(data), 3))
    model = MeanModel(ARX(0) + DENSITIES[density](), data, exogenous)
    z = model.map(np.r_[np.full(4, .03), 1.3, shapes[density]], forward=False)
    # Anchor the theta oracle to the actual mapped doubles, keeping original z.
    theta = model.map(z, forward=True)
    theta_ad = mean_value_grad_hess(
        theta, data, model.features, model.q, model.first, density,
    )
    coordinate_ad = mean_value_grad_hess(
        z, data, model.features, model.q, model.first, density, coordinates=True,
    )
    np.testing.assert_allclose(theta_ad[0], coordinate_ad[0], rtol=0, atol=2e-7)
    return model.spec, data, exogenous, theta, z, theta_ad, coordinate_ad


@pytest.mark.parametrize("fixed", [False, True], ids=["runtime", "fixed"])
@pytest.mark.parametrize("output", [0, 1, 2], ids=["F", "G", "H"])
def test_mean_shape_pullback_uses_original_z(mean_shape_case, fixed, output):
    spec, data, exogenous, theta, z, theta_ad, coordinate_ad = mean_shape_case
    model = MeanModel(spec, data, exogenous, fixed=fixed)
    dispatch = _native_core._mean_dispatch_kind(model.descriptor)
    assert dispatch > 0 if fixed else dispatch == 0
    for actual, expected in zip(model.evaluate(theta), theta_ad):
        np.testing.assert_allclose(actual, expected, rtol=2e-6, atol=2e-7)
    for mask in range(1, 8):
        if mask & (1 << output):
            actual = model.evaluate(z, mask, coordinates=True)
            np.testing.assert_allclose(
                actual[output], coordinate_ad[output], rtol=2e-6, atol=2e-7,
                err_msg=f"original-z output={output}, request={mask}, fixed={fixed}",
            )


@pytest.mark.parametrize("mean", [ARMA(1, 1), ARX(1)], ids=["arma11", "arx1"])
def test_mean_gh_minus_half_reduces_to_nig_against_independent_ad(mean):
    pytest.importorskip("jax")
    rng = np.random.default_rng(421012)
    data = rng.normal(size=9)
    exogenous = None if isinstance(mean, ARMA) else rng.normal(size=(9, 2))
    nig = MeanModel(mean + NormalInverseGaussian(), data, exogenous)
    gh = MeanModel(mean + GeneralizedHyperbolic(), data, exogenous)
    theta = np.r_[np.full(nig.width - 1, .03), 1.3, .12, 1.8]
    features = nig.features
    if features is None:
        features = np.ones((len(data), nig.columns))
        for lag in range(1, nig.p + 1):
            features[lag:, lag] = data[:-lag]
    oracle = mean_value_grad_hess(
        theta, data, features, nig.q, nig.first, 5, ar_order=nig.p
    )
    general_oracle = mean_value_grad_hess(
        np.r_[theta, -.5], data, features, gh.q, gh.first, 4, ar_order=gh.p
    )
    for actual, expected in zip(
        (general_oracle[0], general_oracle[1][:-1], general_oracle[2][:-1, :-1]),
        oracle,
    ):
        np.testing.assert_allclose(actual, expected, rtol=2e-12, atol=2e-12)

    full = nig.evaluate(theta, 7)
    for actual, expected in zip(full, oracle):
        np.testing.assert_allclose(actual, expected, rtol=2e-6, atol=2e-7)
    for fixed in (False, True):
        reduced = MeanModel(nig.spec, data, exogenous, fixed=fixed)
        general = MeanModel(gh.spec, data, exogenous, fixed=fixed)
        for model in (reduced, general):
            dispatch = _native_core._mean_dispatch_kind(model.descriptor)
            assert dispatch > 0 if fixed else dispatch == 0
        for mask in range(1, 8):
            special = reduced.evaluate(theta, mask)
            shared = general.evaluate(np.r_[theta, -.5], mask)
            for index, bit in enumerate((1, 2, 4)):
                if not mask & bit:
                    assert special[index] is None and shared[index] is None
                    continue
                common = shared[index]
                if bit == 2:
                    common = common[:-1]
                elif bit == 4:
                    common = common[:-1, :-1]
                np.testing.assert_allclose(
                    special[index], full[index], rtol=1e-11, atol=1e-10
                )
                np.testing.assert_allclose(
                    common, special[index], rtol=2e-11, atol=2e-11
                )
