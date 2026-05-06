"""
Tests for scivol._settings  (ParamNames / Settings) and
display-name integration in EstimationResult.
"""

from __future__ import annotations

import re
from types import SimpleNamespace

import numpy as np
import pytest

import scivol
from scivol._settings import ParamNames, Settings, settings


# ------------------------------------------------------------------ #
# helpers
# ------------------------------------------------------------------ #

class DummyOpt(SimpleNamespace):
    """Minimal scipy.optimize.OptimizeResult stand-in."""

    def __init__(
        self,
        x: np.ndarray,
        fun: float = 10.0,
        success: bool = True,
        nit: int = 5,
        message: str = "converged",
    ) -> None:
        super().__init__(x=x, fun=fun, success=success, nit=nit, message=message)


# ================================================================== #
# 1. ParamNames unit tests
# ================================================================== #

class TestParamNames:
    """Test ParamNames resolution, override, and reset."""

    def setup_method(self):
        self.pn = ParamNames()

    # -- defaults --

    def test_defaults_returned(self):
        assert self.pn.resolve("omega") == "omega"
        assert self.pn.resolve("alpha") == "alpha"
        assert self.pn.resolve("gamma") == "gamma"
        assert self.pn.resolve("beta") == "beta"
        assert self.pn.resolve("nu") == "nu"
        assert self.pn.resolve("lambda") == "lambda"

    def test_unknown_name_passes_through(self):
        assert self.pn.resolve("foobar") == "foobar"

    # -- overrides --

    def test_override_bare_name(self):
        self.pn.gamma = "zeta"
        assert self.pn.resolve("gamma") == "zeta"
        assert self.pn.gamma == "zeta"

    def test_override_does_not_affect_others(self):
        self.pn.gamma = "zeta"
        assert self.pn.resolve("alpha") == "alpha"

    # -- indexed names --

    def test_indexed_name_default(self):
        assert self.pn.resolve("alpha[1]") == "alpha[1]"
        assert self.pn.resolve("beta[3]") == "beta[3]"

    def test_indexed_name_with_override(self):
        self.pn.alpha = "a"
        assert self.pn.resolve("alpha[1]") == "a[1]"
        assert self.pn.resolve("alpha[2]") == "a[2]"

    def test_indexed_gamma_override(self):
        self.pn.gamma = "zeta"
        assert self.pn.resolve("gamma[1]") == "zeta[1]"

    # -- reset --

    def test_reset_clears_overrides(self):
        self.pn.gamma = "zeta"
        self.pn.nu = "df"
        self.pn.reset()
        assert self.pn.resolve("gamma") == "gamma"
        assert self.pn.resolve("nu") == "nu"

    # -- attribute access --

    def test_getattr_returns_display_name(self):
        self.pn.omega = "w"
        assert self.pn.omega == "w"

    def test_getattr_default(self):
        assert self.pn.beta == "beta"

    # -- type safety --

    def test_setting_non_string_raises(self):
        with pytest.raises(TypeError):
            self.pn.omega = 42

    # -- repr --

    def test_repr_defaults(self):
        r = repr(self.pn)
        assert "defaults" in r

    def test_repr_overrides(self):
        self.pn.gamma = "zeta"
        r = repr(self.pn)
        assert "zeta" in r


# ================================================================== #
# 2. Module-level singleton
# ================================================================== #

class TestSettingsSingleton:
    """The global settings object works as expected."""

    def setup_method(self):
        settings.names.reset()

    def teardown_method(self):
        settings.names.reset()

    def test_singleton_accessible(self):
        assert scivol.settings is settings

    def test_set_and_read_back(self):
        scivol.settings.names.gamma = "zeta"
        assert scivol.settings.names.resolve("gamma[1]") == "zeta[1]"

    def test_repr(self):
        assert "Settings" in repr(settings)


# ================================================================== #
# 3. Integration with EstimationResult (pure-Python, no C kernels)
# ================================================================== #

def _make_garch_result(gamma: bool = False):
    """Build a minimal EstimationResult with fake fitted params."""
    from scivol import GARCH, GJRGARCH, Normal
    from scivol.result import EstimationResult

    if gamma:
        vol = GJRGARCH(1, 1)
        vol.unpack(np.array([1e-5, 0.05, 0.10, 0.85]))  # omega, alpha, gamma, beta
        n_vol = 4
    else:
        vol = GARCH(1, 1)
        vol.unpack(np.array([1e-5, 0.05, 0.90]))  # omega, alpha, beta
        n_vol = 3

    dens = Normal()
    spec = vol + dens
    params = np.zeros(n_vol)
    opt = DummyOpt(x=params)

    return EstimationResult(spec, opt, np.random.default_rng(0).standard_normal(100))


class TestParamNamesInResult:
    """_get_param_names() and to_dict() honour display overrides."""

    def setup_method(self):
        settings.names.reset()

    def teardown_method(self):
        settings.names.reset()

    # -- _get_param_names --

    def test_default_names_garch(self):
        res = _make_garch_result(gamma=False)
        names = res._get_param_names()
        assert names == ["omega", "alpha[1]", "beta[1]"]

    def test_default_names_gjr(self):
        res = _make_garch_result(gamma=True)
        names = res._get_param_names()
        assert names == ["omega", "alpha[1]", "gamma[1]", "beta[1]"]

    def test_override_alpha_in_names(self):
        settings.names.alpha = "a"
        res = _make_garch_result(gamma=False)
        names = res._get_param_names()
        assert names[1] == "a[1]"
        # Others unchanged
        assert names[0] == "omega"
        assert names[2] == "beta[1]"

    def test_override_gamma_in_gjr(self):
        settings.names.gamma = "zeta"
        res = _make_garch_result(gamma=True)
        names = res._get_param_names()
        assert names[2] == "zeta[1]"

    def test_override_omega(self):
        settings.names.omega = "w"
        res = _make_garch_result()
        names = res._get_param_names()
        assert names[0] == "w"

    # -- to_dict --

    def test_to_dict_keys_default(self):
        res = _make_garch_result()
        d = res.to_dict()
        gp = d["garch_params"]
        assert "omega" in gp
        assert "alpha" in gp
        assert "beta" in gp

    def test_to_dict_keys_overridden(self):
        settings.names.alpha = "a"
        settings.names.beta = "b"
        res = _make_garch_result()
        d = res.to_dict()
        gp = d["garch_params"]
        assert "a" in gp
        assert "b" in gp
        # Original canonical keys should not appear
        assert "alpha" not in gp
        assert "beta" not in gp

    def test_to_dict_gjr_gamma_key(self):
        settings.names.gamma = "zeta"
        res = _make_garch_result(gamma=True)
        d = res.to_dict()
        gp = d["garch_params"]
        assert "zeta" in gp
        assert "gamma" not in gp

    # -- summary output --

    def test_summary_uses_display_names(self, capsys):
        settings.names.alpha = "a"
        settings.names.beta = "b"
        res = _make_garch_result()
        res.summary()
        out = capsys.readouterr().out
        assert "a[1]" in out
        assert "b[1]" in out

    def test_summary_gjr_persistence_label(self, capsys):
        settings.names.gamma = "zeta"
        res = _make_garch_result(gamma=True)
        res.summary()
        out = capsys.readouterr().out
        assert "zeta" in out
        assert "Persistence" in out

    # -- __str__ uses display names --

    def test_str_uses_display_names(self):
        settings.names.omega = "w"
        res = _make_garch_result()
        text = str(res)
        assert "w" in text


# ================================================================== #
# 4. GARCHParams gamma support
# ================================================================== #

class TestGARCHParamsGamma:
    """GARCHParams with gamma field behaves correctly."""

    def test_no_gamma(self):
        from scivol.result import GARCHParams
        gp = GARCHParams(omega=1e-5, alpha=np.array([0.05]), beta=np.array([0.90]))
        assert gp.gamma is None
        assert gp.persistence == pytest.approx(0.95)
        arr = gp.to_array()
        np.testing.assert_array_equal(arr, [1e-5, 0.05, 0.90])

    def test_with_gamma(self):
        from scivol.result import GARCHParams
        gp = GARCHParams(
            omega=1e-5,
            alpha=np.array([0.05]),
            beta=np.array([0.85]),
            gamma=np.array([0.10]),
        )
        assert gp.gamma is not None
        # persistence = alpha + 0.5*gamma + beta = 0.05 + 0.05 + 0.85
        assert gp.persistence == pytest.approx(0.95)
        arr = gp.to_array()
        np.testing.assert_array_equal(arr, [1e-5, 0.05, 0.10, 0.85])

    def test_gjr_result_has_gamma(self):
        res = _make_garch_result(gamma=True)
        gp = res.garch_params
        assert gp is not None
        assert gp.gamma is not None
        assert len(gp.gamma) == 1

    def test_garch_result_no_gamma(self):
        res = _make_garch_result(gamma=False)
        gp = res.garch_params
        assert gp is not None
        assert gp.gamma is None
