# tests/test_spec_core.py
import numpy as np
import pytest

# import objects from your package
from volkit import (
    ARMA,
    GARCH,
    Normal,
    StudentT,
    CompositeSpec,
    Role,
)


# ------------------------------------------------------------------
# 1. Canonicalisation & defaults
# ------------------------------------------------------------------
@pytest.mark.parametrize(
    "spec_a, spec_b, expected_uid",
    [
        # order-independence
        (ARMA(1, 1) + GARCH(1, 1), GARCH(1, 1) + ARMA(1, 1), "ARMA(1,1)+GARCH(1,1)+Normal"),
        # explicit vs implicit Normal
        (GARCH(2, 1), GARCH(2, 1) + Normal(), "GARCH(2,1)+Normal"),
        # StudentT replaces Normal
        (GARCH(1, 1) + StudentT(), (GARCH(1, 1) + Normal()) + StudentT(), "GARCH(1,1)+StudentT"),
    ],
)
def test_uid_and_default_density(spec_a, spec_b, expected_uid):
    assert str(spec_a) == str(spec_b) == expected_uid
    assert spec_a == spec_b
    # density component present and unique
    dens = spec_a.get_component(Role.DENSITY)
    assert isinstance(dens, (Normal, StudentT))


# ------------------------------------------------------------------
# 2. Duplicate-role guard
# ------------------------------------------------------------------
def test_duplicate_role_raises():
    with pytest.raises(ValueError, match="Multiple components with role"):
        CompositeSpec(GARCH(1, 1), GARCH(2, 1))  # two volatilities


# ------------------------------------------------------------------
# 3. Slice-map integrity
# ------------------------------------------------------------------
def test_slice_map_matches_parameter_lengths():
    spec = CompositeSpec(ARMA(1, 0), GARCH(1, 1))
    slices = spec._slice_map()
    theta_len = sum(comp.n_params for comp in spec.components)
    # end of last slice == total length
    assert max(sl.stop for sl in slices.values()) == theta_len
    # slices are non-overlapping
    stops = sorted((sl.start, sl.stop) for sl in slices.values())
    for (_, a_stop), (b_start, _) in zip(stops, stops[1:]):
        assert a_stop == b_start


# ------------------------------------------------------------------
# 4. total_params accuracy
# ------------------------------------------------------------------
def test_total_params():
    spec = CompositeSpec(ARMA(2, 1), GARCH(3, 0))  # n = 1+2+1 + 1+3 = 8
    assert spec.total_params == 8


# ------------------------------------------------------------------
# 5. GARCH helpers after fitting-mock
# ------------------------------------------------------------------
def test_garch_persistence_and_stationarity():
    g = GARCH(1, 1)
    # fake a fitted_param dict (bypass optimiser)
    g.unpack(np.array([0.01, 0.08, 0.90]))
    assert pytest.approx(g.persistence()) == 0.98
    assert g.is_stationary()
    assert g.unconditional_variance() > 0


# ------------------------------------------------------------------
# 6. Hash / equality contract
# ------------------------------------------------------------------
def test_hash_equality_contract():
    s1 = CompositeSpec(ARMA(1, 1), GARCH(1, 1))
    s2 = CompositeSpec(GARCH(1, 1), ARMA(1, 1))
    s3 = CompositeSpec(GARCH(2, 1))
    assert s1 == s2 and hash(s1) == hash(s2)
    assert s1 != s3 and hash(s1) != hash(s3)


def test_arrow_syntax_on_composite():
    spec = GARCH(1, 2) << ARMA(1, 1) <- StudentT()
    assert str(spec) == "ARMA(1,1)+GARCH(1,2)+StudentT"