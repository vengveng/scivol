"""Standard errors at estimates where closed bounds or constraints bind."""

from __future__ import annotations

import numpy as np
import pytest

from scivol import GARCH, GJRGARCH, StudentT


@pytest.fixture(scope="module")
def garch_data() -> np.ndarray:
    return GARCH(1, 1).simulate(3000, [0.05, 0.08, 0.9], seed=4).data


@pytest.fixture(scope="module")
def shifted() -> np.ndarray:
    """A variance shift halfway through: GARCH persistence runs to one."""
    rng = np.random.default_rng(3)
    return np.concatenate([rng.standard_normal(1000), 2.5 * rng.standard_normal(1000)])


def test_withheld_by_default(garch_data):
    result = GARCH(2, 1).fit(garch_data)
    assert result.fit_info.inference_status == "at_boundary"
    assert result.fit_info.binding == ("alpha[2] >= 0",)
    assert result.std_errors is None


def test_a_coordinate_face_gives_the_child_models_errors(garch_data):
    parent = GARCH(2, 1).fit(garch_data, boundary="restricted")
    child = GARCH(1, 1).fit(garch_data)
    assert parent.fit_info.inference_status == "restricted"
    assert parent.params.iloc[2] == 0.0 and np.isnan(parent.std_errors.iloc[2])
    keep = [0, 1, 3]
    np.testing.assert_allclose(parent.params.iloc[keep], child.params, rtol=1e-5)
    np.testing.assert_allclose(parent.std_errors.iloc[keep], child.std_errors, rtol=1e-4)
    np.testing.assert_allclose(parent.std_errors_robust.iloc[keep], child.std_errors_robust, rtol=1e-4)
    assert any("as equalities" in w for w in parent.warnings)
    assert not any("standard errors" in w and "not" in w for w in parent.warnings)


def test_the_integrated_face_leaves_no_variance_in_the_persistence(shifted):
    result = GARCH(1, 1, stationarity="integrated").fit(shifted, boundary="restricted")
    assert result.fit_info.binding == ("persistence <= 1",)
    assert result.persistence == pytest.approx(1.0, abs=1e-12)
    c = result.cov_matrix.to_numpy()
    assert c[1, 1] + 2 * c[1, 2] + c[2, 2] == pytest.approx(0.0, abs=1e-12 * c[2, 2])
    assert np.all(np.isfinite(result.std_errors)) and np.all(result.std_errors > 0)


def test_an_interior_estimate_is_unchanged(garch_data):
    plain = GARCH(1, 1).fit(garch_data)
    restricted = GARCH(1, 1).fit(garch_data, boundary="restricted")
    assert restricted.fit_info.inference_status == "available"
    np.testing.assert_array_equal(plain.std_errors, restricted.std_errors)


def test_qmle_reports_the_restricted_sandwich(garch_data):
    result = GARCH(2, 1).fit(garch_data, method="qmle", boundary="restricted")
    assert result.fit_info.inference_status == "restricted"
    np.testing.assert_array_equal(result.std_errors, result.std_errors_robust)
    child = GARCH(1, 1).fit(garch_data, method="qmle")
    np.testing.assert_allclose(result.std_errors.iloc[[0, 1, 3]], child.std_errors, rtol=1e-4)


def test_a_staged_qmle_fit_keeps_its_boundary_errors_withheld(garch_data):
    result = (GARCH(2, 1) + StudentT()).fit(garch_data, method="qmle", boundary="restricted")
    assert result.fit_info.inference_status != "restricted"


def test_the_option_is_validated(garch_data):
    with pytest.raises(ValueError, match="boundary must be"):
        GARCH(1, 1).fit(garch_data, boundary="bootstrap")


# The m-out-of-n parametric bootstrap.

def test_bootstrap_is_reproducible_and_chooses_m_on_its_grid(garch_data):
    from scivol._bootstrap import size_grid

    result = GARCH(1, 1).fit(garch_data)
    a, b = result.bootstrap(60, seed=3), result.bootstrap(60, seed=3)
    np.testing.assert_array_equal(a.draws, b.draws)
    assert a.m in size_grid(3000) and a.m < 3000
    assert a.draws.shape == (60 - a.failures, 3) and a.param_names == result.param_names


def test_bootstrap_errors_are_close_to_the_information_errors_inside(garch_data):
    result = GARCH(1, 1).fit(garch_data)
    boot = result.bootstrap(150, seed=1)
    ratio = boot.std_errors / result.std_errors
    assert np.all((ratio > 0.6) & (ratio < 2.5)), ratio
    intervals = boot.intervals(0.9)
    assert np.all((intervals[:, 0] <= result.params) & (result.params <= intervals[:, 1]))


def test_bootstrap_keeps_the_boundary_law(garch_data):
    result = GARCH(2, 1).fit(garch_data)
    boot = result.bootstrap(150, seed=1)
    assert 0.1 < np.mean(boot.draws[:, 2] == 0.0) < 0.9  # a point mass at the bound
    assert np.all(boot.intervals()[:, 0] >= np.nan_to_num(boot.lower, nan=-np.inf))


def test_bootstrap_runs_on_an_integrated_fit(shifted):
    result = GARCH(1, 1, stationarity="integrated").fit(shifted)
    boot = result.bootstrap(40, m=500, seed=2)
    assert boot.m == 500 and len(boot.draws) >= 20


def test_bootstrap_validates_its_arguments(garch_data):
    result = GARCH(1, 1).fit(garch_data)
    with pytest.raises(ValueError, match="m must lie"):
        result.bootstrap(10, m=10_000)
    with pytest.raises(ValueError, match="replications"):
        result.bootstrap(1)
