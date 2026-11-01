# tests/test_spec_core.py
import numpy as np
import pytest

# import objects from your package
from scivol import (
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
# 3. The native model owns the parameter layout
# ------------------------------------------------------------------
def test_native_model_names_every_parameter():
    from scivol import _scivolapi as api
    from scivol._fit import native_spec

    model = api.Model(native_spec(CompositeSpec(ARMA(2, 1), GARCH(3, 0))))
    assert model.parameter_names == ["const", "ar[1]", "ar[2]", "ma[1]", "omega", "alpha[1]", "alpha[2]", "alpha[3]"]


# ------------------------------------------------------------------
# 6. Hash / equality contract
# ------------------------------------------------------------------
def test_hash_equality_contract():
    s1 = CompositeSpec(ARMA(1, 1), GARCH(1, 1))
    s2 = CompositeSpec(GARCH(1, 1), ARMA(1, 1))
    s3 = CompositeSpec(GARCH(2, 1))
    assert s1 == s2 and hash(s1) == hash(s2)
    assert s1 != s3 and hash(s1) != hash(s3)


def test_components_compose_only_with_plus():
    spec = GARCH(1, 2) + ARMA(1, 1) + StudentT()
    assert str(spec) == "ARMA(1,1)+GARCH(1,2)+StudentT"
    for compose in (lambda a, b: a < b, lambda a, b: a << b, lambda a, b: a >> b, lambda a, b: a - b):
        with pytest.raises(TypeError):
            compose(GARCH(), StudentT())