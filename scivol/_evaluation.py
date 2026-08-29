from __future__ import annotations

from collections.abc import Iterator, Mapping
from copy import deepcopy
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Optional

import numpy as np
from numpy.typing import NDArray
from scipy.integrate import quad
from scipy.special import gammaln
from scipy.stats import gennorm, norm, t as t_dist

from . import _core
from ._distribution_moments import negative_half_second_moment
from ._validation import validate_data, validate_spec
from .components.density import (
    GED,
    GeneralizedHyperbolic,
    Hyperbolic,
    Normal,
    NormalInverseGaussian,
    SkewT,
    StudentT,
    VarianceGamma,
)
from .components.mean import ARMA, ARX, HARX
from .components.vol import APARCH, EGARCH, GARCH, GJRGARCH
from ._gh import (
    fixed_gh_lambda,
    standardized_gh_abs_moment,
    standardized_gh_cdf,
    standardized_gh_ppf,
    standardized_gh_rvs,
)
from ._vg import (
    standardized_vg_abs_moment,
    standardized_vg_cdf,
    standardized_vg_ppf,
    standardized_vg_rvs,
)
from ._kernels.arma_egarch_common import arma_egarch_total_derivative, arma_egarch_total_nll
from ._kernels.linear_mean_garch_common import build_linear_mean_features
from .result import EstimationResult
from .roles import Role


_HALF_LOG_2PI = 0.9189385332046727


def _as_cptr(arr: NDArray[np.float64]) -> int:
    return np.ascontiguousarray(arr, dtype=np.float64).ctypes.data


def _arma_volatility_total_nll(
    average_nll: float,
    n: int,
    p_ar: int,
    q_ma: int,
    p_vol: int,
    q_vol: int,
    *,
    normal_constant_omitted: bool,
) -> float:
    """Restore the complete additive objective returned by ARMA-volatility cores."""
    n_eff = n - max(p_ar, q_ma, p_vol, q_vol, 1)
    per_observation = average_nll
    if normal_constant_omitted:
        per_observation += _HALF_LOG_2PI
    return float(n_eff) * per_observation


def _arma_volatility_total_derivative(
    average_derivative: NDArray[np.float64],
    n: int,
    p_ar: int,
    q_ma: int,
    p_vol: int,
    q_vol: int,
) -> NDArray[np.float64]:
    """Restore derivatives of the complete additive ARMA-volatility objective."""
    n_eff = n - max(p_ar, q_ma, p_vol, q_vol, 1)
    return float(n_eff) * average_derivative


def _std_t_cdf(x: NDArray[np.float64], nu: float) -> NDArray[np.float64]:
    scale = np.sqrt(nu / (nu - 2.0))
    return t_dist.cdf(x * scale, df=nu)


def _std_t_ppf(levels: NDArray[np.float64], nu: float) -> NDArray[np.float64]:
    scale = np.sqrt((nu - 2.0) / nu)
    return t_dist.ppf(levels, df=nu) * scale


def _hansen_skewt_cdf(z: NDArray[np.float64], nu: float, lam: float) -> NDArray[np.float64]:
    c = gammaln(0.5 * (nu + 1.0)) - gammaln(0.5 * nu) - 0.5 * np.log(np.pi * (nu - 2.0))
    a = 4.0 * lam * np.exp(c) * (nu - 2.0) / (nu - 1.0)
    b = np.sqrt(1.0 + 3.0 * lam * lam - a * a)
    bz_a = b * z + a
    left = bz_a < 0.0
    scale = np.where(left, 1.0 - lam, 1.0 + lam)
    base_cdf = _std_t_cdf(bz_a / scale, nu)
    return np.where(
        left,
        (1.0 - lam) * base_cdf,
        (1.0 + lam) * base_cdf - lam,
    )


def _hansen_skewt_ppf(levels: NDArray[np.float64], nu: float, lam: float) -> NDArray[np.float64]:
    levels = np.asarray(levels, dtype=np.float64)
    left = levels < 0.5 * (1.0 - lam)
    base_levels = np.where(
        left,
        levels / (1.0 - lam),
        (levels + lam) / (1.0 + lam),
    )
    scale = np.where(left, 1.0 - lam, 1.0 + lam)

    c_log = gammaln(0.5 * (nu + 1.0)) - gammaln(0.5 * nu) - 0.5 * np.log(np.pi * (nu - 2.0))
    a = 4.0 * lam * np.exp(c_log) * (nu - 2.0) / (nu - 1.0)
    b = np.sqrt(1.0 + 3.0 * lam * lam - a * a)
    return (scale * _std_t_ppf(base_levels, nu) - a) / b


def _hansen_skewt_pdf_scalar(z: float, nu: float, lam: float) -> float:
    c_log = gammaln(0.5 * (nu + 1.0)) - gammaln(0.5 * nu) - 0.5 * np.log(np.pi * (nu - 2.0))
    a = 4.0 * lam * np.exp(c_log) * (nu - 2.0) / (nu - 1.0)
    b = np.sqrt(1.0 + 3.0 * lam * lam - a * a)
    bz_a = b * z + a
    denom = (1.0 - lam) if bz_a < 0.0 else (1.0 + lam)
    z_adj = bz_a / denom
    return float(np.exp(c_log) * b * (1.0 + z_adj * z_adj / (nu - 2.0)) ** (-0.5 * (nu + 1.0)))


@lru_cache(maxsize=64)
def _egarch_skewt_abs_moment(nu: float, lam: float) -> float:
    c_log = gammaln(0.5 * (nu + 1.0)) - gammaln(0.5 * nu) - 0.5 * np.log(np.pi * (nu - 2.0))
    a = 4.0 * lam * np.exp(c_log) * (nu - 2.0) / (nu - 1.0)
    b = np.sqrt(1.0 + 3.0 * lam * lam - a * a)
    split = float(-a / b)

    def integrand(z: float) -> float:
        return abs(z) * _hansen_skewt_pdf_scalar(z, nu, lam)

    bound = 50.0
    points = [split] if -bound < split < bound else None
    value, _ = quad(integrand, -bound, bound, epsabs=1e-9, epsrel=1e-9, limit=400, points=points)
    return float(value)


def _ged_scale(nu: float) -> float:
    return float(np.sqrt(np.exp(gammaln(1.0 / nu) - gammaln(3.0 / nu))))


def _std_ged_cdf(x: NDArray[np.float64], nu: float) -> NDArray[np.float64]:
    return gennorm.cdf(x, beta=nu, scale=_ged_scale(nu))


def _std_ged_ppf(levels: NDArray[np.float64], nu: float) -> NDArray[np.float64]:
    return gennorm.ppf(levels, beta=nu, scale=_ged_scale(nu))


def _distribution_quantile(
    distribution: str,
    levels: NDArray[np.float64],
    nu: Optional[float],
    lam: Optional[float],
    rho: Optional[float] = None,
    zeta: Optional[float] = None,
    shape_lambda: Optional[float] = None,
) -> NDArray[np.float64]:
    if distribution == "Normal":
        return norm.ppf(levels)
    if distribution == "StudentT":
        if nu is None:
            raise ValueError("StudentT forecast quantiles require `nu`.")
        return _std_t_ppf(levels, nu)
    if distribution == "SkewT":
        if nu is None or lam is None:
            raise ValueError("SkewT forecast quantiles require `nu` and `lam`.")
        return _hansen_skewt_ppf(levels, nu, lam)
    if distribution == "GED":
        if nu is None:
            raise ValueError("GED forecast quantiles require `nu`.")
        return _std_ged_ppf(levels, nu)
    if distribution in {"GeneralizedHyperbolic", "Hyperbolic", "NormalInverseGaussian"}:
        if rho is None or zeta is None or shape_lambda is None:
            raise ValueError(f"{distribution} forecast quantiles require GH shape parameters.")
        return standardized_gh_ppf(levels, rho, zeta, shape_lambda)
    if distribution == "VarianceGamma":
        if rho is None or shape_lambda is None:
            raise ValueError("VarianceGamma forecast quantiles require `rho` and `lambda`.")
        return standardized_vg_ppf(levels, rho, shape_lambda)
    raise ValueError(f"Unsupported distribution '{distribution}'.")


@dataclass(slots=True)
class FilteredState:
    data: NDArray[np.float64]
    mean: NDArray[np.float64]
    residuals: NDArray[np.float64]
    sigma2: NDArray[np.float64]
    log_likelihood: float
    score: Optional[NDArray[np.float64]]
    hessian: Optional[NDArray[np.float64]]
    distribution: str
    nu: Optional[float] = None
    lam: Optional[float] = None
    rho: Optional[float] = None
    zeta: Optional[float] = None
    shape_lambda: Optional[float] = None

    @property
    def residual_variance(self) -> NDArray[np.float64]:
        return self.sigma2

    @property
    def std_resid(self) -> NDArray[np.float64]:
        return self.residuals / np.sqrt(self.sigma2)

    def to_dict(self) -> dict[str, Any]:
        return {
            "log_likelihood": self.log_likelihood,
            "mean": self.mean.copy(),
            "residuals": self.residuals.copy(),
            "sigma2": self.sigma2.copy(),
            "std_resid": self.std_resid.copy(),
            "distribution": self.distribution,
            "nu": self.nu,
            "lam": self.lam,
            "rho": self.rho,
            "zeta": self.zeta,
            "lambda": self.shape_lambda,
            "score": None if self.score is None else self.score.copy(),
            "hessian": None if self.hessian is None else self.hessian.copy(),
        }


@dataclass(slots=True)
class ForecastResult(Mapping[str, NDArray[np.float64]]):
    mean: NDArray[np.float64]
    variance: NDArray[np.float64]
    residual_variance: NDArray[np.float64]
    distribution: str
    nu: Optional[float] = None
    lam: Optional[float] = None
    rho: Optional[float] = None
    zeta: Optional[float] = None
    shape_lambda: Optional[float] = None

    def __getitem__(self, key: str) -> NDArray[np.float64]:
        if key == "mean":
            return self.mean
        if key == "variance":
            return self.variance
        if key == "residual_variance":
            return self.residual_variance
        if key == "sigma2":
            return self.sigma2
        if key == "volatility":
            return self.volatility
        raise KeyError(key)

    def __iter__(self) -> Iterator[str]:
        return iter(("mean", "variance", "residual_variance", "sigma2", "volatility"))

    def __len__(self) -> int:
        return 5

    @property
    def sigma2(self) -> NDArray[np.float64]:
        return self.residual_variance

    @property
    def volatility(self) -> NDArray[np.float64]:
        return np.sqrt(self.variance)

    @property
    def residual_volatility(self) -> NDArray[np.float64]:
        return np.sqrt(self.residual_variance)

    def quantile(self, level: float | NDArray[np.float64]) -> NDArray[np.float64]:
        levels = np.asarray(level, dtype=np.float64)
        if np.any((levels <= 0.0) | (levels >= 1.0)):
            raise ValueError("Quantile levels must lie strictly between 0 and 1.")

        if self.distribution == "Normal" or not np.allclose(self.variance, self.residual_variance):
            innovation_q = norm.ppf(levels)
            scale = np.sqrt(self.variance)
        else:
            innovation_q = _distribution_quantile(
                self.distribution,
                np.atleast_1d(levels),
                self.nu,
                self.lam,
                self.rho,
                self.zeta,
                self.shape_lambda,
            )
            scale = np.sqrt(self.residual_variance)

        if levels.ndim == 0:
            return self.mean + scale * float(np.atleast_1d(innovation_q)[0])
        return self.mean[:, None] + scale[:, None] * np.atleast_1d(innovation_q)[None, :]

    def var(self, level: float | NDArray[np.float64]) -> NDArray[np.float64]:
        return -self.quantile(level)

    def to_dict(self) -> dict[str, Any]:
        return {
            "mean": self.mean.copy(),
            "variance": self.variance.copy(),
            "residual_variance": self.residual_variance.copy(),
            "sigma2": self.sigma2.copy(),
            "volatility": self.volatility.copy(),
            "distribution": self.distribution,
            "nu": self.nu,
            "lam": self.lam,
            "rho": self.rho,
            "zeta": self.zeta,
            "lambda": self.shape_lambda,
        }


@dataclass(slots=True)
class SimulationResult:
    data: NDArray[np.float64]
    mean: NDArray[np.float64]
    residuals: NDArray[np.float64]
    sigma2: NDArray[np.float64]
    innovations: NDArray[np.float64]

    def to_dict(self) -> dict[str, NDArray[np.float64]]:
        return {
            "data": self.data.copy(),
            "mean": self.mean.copy(),
            "residuals": self.residuals.copy(),
            "sigma2": self.sigma2.copy(),
            "innovations": self.innovations.copy(),
        }


class _FixedOptResult:
    def __init__(self, x: NDArray[np.float64], fun: float) -> None:
        self.x = x
        self.fun = fun
        self.success = True
        self.nit = 0
        self.message = "Parameters supplied by user."


class FixedResult(EstimationResult):
    def __repr__(self) -> str:
        return f"FixedResult({self.spec}, LL={self.log_likelihood:.4f})"


def _component_param_dict(component: Any, params: Mapping[str, Any]) -> dict[str, Any]:
    if isinstance(component, ARMA):
        if {"const", "ar", "ma"} <= params.keys():
            return {"const": params["const"], "ar": params["ar"], "ma": params["ma"]}
        if {"c", "phi", "theta"} <= params.keys():
            return {"const": params["c"], "ar": params["phi"], "ma": params["theta"]}
    if isinstance(component, APARCH):
        return {
            "omega": params["omega"],
            "alpha": params["alpha"],
            "gamma": params["gamma"],
            "beta": params["beta"],
            "delta": params["delta"],
        }
    if isinstance(component, EGARCH):
        return {
            "omega": params["omega"],
            "alpha": params["alpha"],
            "gamma": params["gamma"],
            "beta": params["beta"],
        }
    if isinstance(component, GJRGARCH):
        return {
            "omega": params["omega"],
            "alpha": params["alpha"],
            "gamma": params["gamma"],
            "beta": params["beta"],
        }
    if isinstance(component, GARCH):
        return {"omega": params["omega"], "alpha": params["alpha"], "beta": params["beta"]}
    if isinstance(component, StudentT):
        return {"nu": params["nu"]}
    if isinstance(component, GED):
        return {"nu": params["nu"]}
    if isinstance(component, SkewT):
        return {"nu": params["nu"], "lam": params["lam"]}
    if isinstance(component, GeneralizedHyperbolic):
        return {
            "rho": params["rho"],
            "zeta": params["zeta"],
            "lambda": params["lambda"],
        }
    if isinstance(component, NormalInverseGaussian):
        return {"rho": params["rho"], "zeta": params["zeta"]}
    if isinstance(component, Hyperbolic):
        return {"rho": params["rho"], "zeta": params["zeta"]}
    if isinstance(component, VarianceGamma):
        return {"rho": params["rho"], "lambda": params["lambda"]}
    return {}


def _has_explicit_arma_density_variance(spec: Any) -> bool:
    mean_comp = spec.get_component(Role.MEAN)
    vol = spec.get_component(Role.VOLATILITY)
    dens = spec.get_component(Role.DENSITY)
    return (
        isinstance(mean_comp, ARMA)
        and vol is None
        and isinstance(dens, (GED, GeneralizedHyperbolic, Hyperbolic, NormalInverseGaussian, VarianceGamma))
    )


def _has_explicit_meanx_variance(spec: Any) -> bool:
    mean_comp = spec.get_component(Role.MEAN)
    vol = spec.get_component(Role.VOLATILITY)
    dens = spec.get_component(Role.DENSITY)
    return (
        isinstance(mean_comp, (ARX, HARX))
        and vol is None
        and isinstance(
            dens,
            (StudentT, SkewT, GED, GeneralizedHyperbolic, Hyperbolic, NormalInverseGaussian, VarianceGamma),
        )
    )


def _has_explicit_constant_variance(spec: Any) -> bool:
    return _has_explicit_arma_density_variance(spec) or _has_explicit_meanx_variance(spec)


def _expected_param_count(spec: Any) -> int:
    extra = 1 if _has_explicit_constant_variance(spec) else 0
    return spec.total_params + extra


def _apply_theta_to_spec(spec: Any, theta: NDArray[np.float64]) -> None:
    if _has_explicit_arma_density_variance(spec):
        mean_comp = spec.get_component(Role.MEAN)
        dens = spec.get_component(Role.DENSITY)
        assert isinstance(mean_comp, ARMA)
        assert isinstance(dens, (GED, GeneralizedHyperbolic, Hyperbolic, NormalInverseGaussian, VarianceGamma))
        n_mean = 1 + mean_comp.p + mean_comp.q
        mean_comp.unpack(theta[:n_mean])
        dens.unpack(theta[n_mean + 1:])
        return
    if _has_explicit_meanx_variance(spec):
        mean_comp = spec.get_component(Role.MEAN)
        dens = spec.get_component(Role.DENSITY)
        assert isinstance(mean_comp, (ARX, HARX))
        assert isinstance(
            dens,
            (StudentT, SkewT, GED, GeneralizedHyperbolic, Hyperbolic, NormalInverseGaussian, VarianceGamma),
        )
        n_mean = mean_comp.n_params
        mean_comp.unpack(theta[:n_mean])
        dens.unpack(theta[n_mean + 1:])
        return

    for component, slc in spec.slice_map.items():
        component.unpack(theta[slc])


def _coerce_params(spec: Any, params: Any) -> NDArray[np.float64]:
    if params is None:
        if _has_explicit_arma_density_variance(spec):
            raise ValueError("Standalone ARMA non-Normal surfaces require an explicit constant `sigma2` parameter.")
        if _has_explicit_meanx_variance(spec):
            raise ValueError("Standalone ARX/HARX non-Normal surfaces require an explicit constant `sigma2` parameter.")
        pieces: list[NDArray[np.float64]] = []
        for component in spec.components:
            fitted = getattr(component, "fitted_params", None)
            if fitted is None:
                raise ValueError("Parameters are required when the specification has not been fitted.")
            pieces.append(np.asarray(component.pack(fitted), dtype=np.float64))
        return np.concatenate(pieces) if pieces else np.zeros(0, dtype=np.float64)

    if isinstance(params, Mapping):
        if _has_explicit_arma_density_variance(spec):
            mean_comp = spec.get_component(Role.MEAN)
            dens = spec.get_component(Role.DENSITY)
            assert isinstance(mean_comp, ARMA)
            assert isinstance(dens, (GED, GeneralizedHyperbolic, Hyperbolic, NormalInverseGaussian, VarianceGamma))
            if "sigma2" not in params:
                raise ValueError("Standalone ARMA non-Normal parameter dictionaries must include `sigma2`.")
            mean_params = _component_param_dict(mean_comp, params)
            dens_params = _component_param_dict(dens, params)
            return np.ascontiguousarray(
                np.concatenate([
                    np.asarray(mean_comp.pack(mean_params), dtype=np.float64),
                    np.array([float(params["sigma2"])], dtype=np.float64),
                    np.asarray(dens.pack(dens_params), dtype=np.float64),
                ]),
                dtype=np.float64,
            )
        if _has_explicit_meanx_variance(spec):
            mean_comp = spec.get_component(Role.MEAN)
            dens = spec.get_component(Role.DENSITY)
            assert isinstance(mean_comp, (ARX, HARX))
            assert isinstance(
                dens,
                (StudentT, SkewT, GED, GeneralizedHyperbolic, Hyperbolic, NormalInverseGaussian, VarianceGamma),
            )
            if "sigma2" not in params:
                raise ValueError("Standalone ARX/HARX non-Normal parameter dictionaries must include `sigma2`.")
            mean_params = _component_param_dict(mean_comp, params)
            dens_params = _component_param_dict(dens, params)
            return np.ascontiguousarray(
                np.concatenate([
                    np.asarray(mean_comp.pack(mean_params), dtype=np.float64),
                    np.array([float(params["sigma2"])], dtype=np.float64),
                    np.asarray(dens.pack(dens_params), dtype=np.float64),
                ]),
                dtype=np.float64,
            )
        pieces = []
        for component in spec.components:
            nested = None
            for key in (component.signature, component.role.name.lower()):
                value = params.get(key)
                if isinstance(value, Mapping):
                    nested = dict(value)
                    break
            if nested is None:
                nested = _component_param_dict(component, params)
            pieces.append(np.asarray(component.pack(nested), dtype=np.float64))
        return np.concatenate(pieces) if pieces else np.zeros(0, dtype=np.float64)

    theta = np.asarray(params, dtype=np.float64).ravel()
    expected = _expected_param_count(spec)
    if theta.size != expected:
        raise ValueError(f"Expected {expected} parameters, received {theta.size}.")
    return np.ascontiguousarray(theta, dtype=np.float64)


def _clone_spec_with_params(spec: Any, theta: NDArray[np.float64]) -> Any:
    cloned = deepcopy(spec)
    _apply_theta_to_spec(cloned, theta)
    return cloned


def _numerical_score(func: Any, x0: NDArray[np.float64], eps: float = 1e-5) -> NDArray[np.float64]:
    grad = np.empty_like(x0, dtype=np.float64)
    for i in range(x0.size):
        step = eps * max(1.0, abs(float(x0[i])))
        x_hi = x0.copy()
        x_lo = x0.copy()
        x_hi[i] += step
        x_lo[i] -= step
        grad[i] = (func(x_hi) - func(x_lo)) / (2.0 * step)
    return grad


def _numerical_hessian_ll(func: Any, x0: NDArray[np.float64], eps: float = 1e-4) -> NDArray[np.float64]:
    hess = np.empty((x0.size, x0.size), dtype=np.float64)
    for i in range(x0.size):
        step_i = eps * max(1.0, abs(float(x0[i])))
        for j in range(i, x0.size):
            step_j = eps * max(1.0, abs(float(x0[j])))
            x_pp = x0.copy(); x_pp[i] += step_i; x_pp[j] += step_j
            x_pm = x0.copy(); x_pm[i] += step_i; x_pm[j] -= step_j
            x_mp = x0.copy(); x_mp[i] -= step_i; x_mp[j] += step_j
            x_mm = x0.copy(); x_mm[i] -= step_i; x_mm[j] -= step_j
            value = (func(x_pp) - func(x_pm) - func(x_mp) + func(x_mm)) / (4.0 * step_i * step_j)
            hess[i, j] = value
            hess[j, i] = value
    return hess


def _distribution_loglikelihood(
    distribution: str,
    resid: NDArray[np.float64],
    sigma2: NDArray[np.float64],
    *,
    nu: Optional[float],
    lam: Optional[float],
) -> float:
    resid2 = resid ** 2
    sigma2 = np.maximum(np.ascontiguousarray(sigma2, dtype=np.float64), 1e-12)
    if distribution == "Normal":
        nll = float(_core._normal_ll(_as_cptr(sigma2), _as_cptr(resid2), resid.size))
        nll += 0.5 * resid.size * np.log(2.0 * np.pi)
        return -nll
    if distribution == "StudentT":
        if nu is None:
            raise ValueError("StudentT requires `nu`.")
        z2 = resid2 / sigma2
        return float(_core._studentt_ll(_as_cptr(sigma2), _as_cptr(z2), resid.size, nu))
    if distribution == "SkewT":
        if nu is None or lam is None:
            raise ValueError("SkewT requires `nu` and `lam`.")
        return float(_core._skewt_ll(_as_cptr(resid), _as_cptr(sigma2), resid.size, nu, lam))
    if distribution == "GED":
        if nu is None:
            raise ValueError("GED requires `nu`.")
        return float(_core._ged_ll(_as_cptr(resid), _as_cptr(sigma2), resid.size, nu))
    raise ValueError(f"Unsupported distribution '{distribution}'.")


def _compute_arx_mean(
    component: ARX,
    data: NDArray[np.float64],
    x: Optional[NDArray[np.float64]],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    params = component.fitted_params or {}
    const = float(params.get("const", 0.0))
    ar = np.asarray(params.get("ar", []), dtype=np.float64)
    exog = np.asarray(params.get("exog", []), dtype=np.float64)
    mean = np.zeros_like(data)

    for t in range(data.size):
        mu = const
        for lag, coef in enumerate(ar, start=1):
            if t - lag >= 0:
                mu += float(coef) * float(data[t - lag])
        if x is not None and exog.size:
            mu += float(x[t] @ exog)
        mean[t] = mu

    return mean, data - mean


def _compute_harx_mean(
    component: HARX,
    data: NDArray[np.float64],
    x: Optional[NDArray[np.float64]],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    params = component.fitted_params or {}
    const = float(params.get("const", 0.0))
    har = np.asarray(params.get("har", []), dtype=np.float64)
    exog = np.asarray(params.get("exog", []), dtype=np.float64)
    mean = np.zeros_like(data)

    for t in range(data.size):
        mu = const
        for horizon, coef in zip(component.horizons, har):
            start = max(0, t - horizon)
            history = data[start:t]
            avg = float(np.mean(history)) if history.size else 0.0
            mu += float(coef) * avg
        if x is not None and exog.size:
            mu += float(x[t] @ exog)
        mean[t] = mu

    return mean, data - mean


def _compute_arma_resid_sigma2(
    mean_comp: ARMA,
    data: NDArray[np.float64],
    mean_theta: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    y = np.ascontiguousarray(data, dtype=np.float64)
    resid = np.zeros_like(y)
    p = mean_comp.p
    q = mean_comp.q
    if p == 1 and q == 1:
        _core._arma_nll_11_normal(_as_cptr(mean_theta), _as_cptr(y), _as_cptr(resid), y.size)
    else:
        e0 = np.zeros(max(q, 1), dtype=np.float64)
        _core._arma_nll_pq_normal(_as_cptr(mean_theta), _as_cptr(y), _as_cptr(resid), _as_cptr(e0), y.size, p, q)
    n_eff = max(y.size - 1, 1)
    sigma2 = np.full(y.size, max(float(np.sum(resid[1:] ** 2) / n_eff), 1e-12), dtype=np.float64)
    return y - resid, resid, sigma2


def _compute_arma_garch_resid_sigma2(
    mean_comp: ARMA,
    vol: GARCH,
    data: NDArray[np.float64],
    core_theta: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    y = np.ascontiguousarray(data, dtype=np.float64)
    resid = np.zeros_like(y)
    sigma2 = np.zeros_like(y)
    h0 = float(np.mean(y ** 2))
    max_lag = max(mean_comp.p, mean_comp.q, vol.p, vol.q, 1)
    e0 = np.zeros(max_lag, dtype=np.float64)
    h0_arr = np.full(max_lag, h0, dtype=np.float64)
    special = mean_comp.p == mean_comp.q == vol.p == vol.q == 1
    if special:
        _core._arma_garch_nll_11_normal(_as_cptr(core_theta), _as_cptr(y), _as_cptr(resid), _as_cptr(sigma2), h0, y.size)
    else:
        _core._arma_garch_nll_pq_normal(
            _as_cptr(core_theta),
            _as_cptr(y),
            _as_cptr(resid),
            _as_cptr(sigma2),
            _as_cptr(e0),
            _as_cptr(h0_arr),
            y.size,
            mean_comp.p,
            mean_comp.q,
            vol.p,
            vol.q,
        )
    return y - resid, resid, sigma2


def _filter_generic(
    spec: Any,
    data: NDArray[np.float64],
    theta: NDArray[np.float64],
    *,
    x: Optional[NDArray[np.float64]],
    compute_score: bool,
    compute_hessian: bool,
) -> FilteredState:
    resolved = _clone_spec_with_params(spec, theta)
    mean_comp = resolved.get_component(Role.MEAN)
    vol = resolved.get_component(Role.VOLATILITY)
    dens = resolved.get_component(Role.DENSITY)

    if isinstance(mean_comp, ARX):
        mean, resid = _compute_arx_mean(mean_comp, data, x)
        sigma2 = np.empty_like(resid)
    elif isinstance(mean_comp, HARX):
        mean, resid = _compute_harx_mean(mean_comp, data, x)
        sigma2 = np.empty_like(resid)
    elif isinstance(mean_comp, ARMA) and isinstance(dens, GED):
        mean_theta = np.asarray(mean_comp.pack(mean_comp.fitted_params or {}), dtype=np.float64)
        if isinstance(vol, GARCH):
            vol_theta = np.asarray(vol.pack(vol.fitted_params or {}), dtype=np.float64)
            core_theta = np.concatenate([mean_theta, vol_theta])
            mean, resid, sigma2 = _compute_arma_garch_resid_sigma2(mean_comp, vol, data, core_theta)
        elif vol is None:
            mean, resid, sigma2 = _compute_arma_resid_sigma2(mean_comp, data, mean_theta)
        else:
            raise NotImplementedError("ARMA + GED currently supports either no volatility or GARCH volatility.")
    else:
        mean = np.zeros_like(data)
        resid = np.ascontiguousarray(data, dtype=np.float64)
        sigma2 = np.empty_like(resid)

    if isinstance(vol, GARCH) and not (isinstance(mean_comp, ARMA) and isinstance(dens, GED)):
        vol_theta = np.asarray(vol.pack(vol.fitted_params or {}), dtype=np.float64)
        sigma2[0] = float(np.mean(resid ** 2))
        _compute_garch_variance(vol_theta, resid ** 2, sigma2, vol.p, vol.q)
    elif isinstance(vol, GJRGARCH):
        vol_theta = np.asarray(vol.pack(vol.fitted_params or {}), dtype=np.float64)
        sigma2[0] = float(np.mean(resid ** 2))
        _compute_gjr_variance(vol_theta, resid, sigma2, vol.p, vol.q)
    elif vol is None and not (isinstance(mean_comp, ARMA) and isinstance(dens, GED)):
        sigma2.fill(max(float(np.var(resid)), 1e-12))

    sigma2 = np.maximum(np.ascontiguousarray(sigma2, dtype=np.float64), 1e-12)
    distribution = dens.signature if dens is not None else "Normal"
    nu = None if dens is None else dens.fitted_params.get("nu")  # type: ignore[union-attr]
    lam = None if dens is None else dens.fitted_params.get("lam")  # type: ignore[union-attr]

    log_likelihood = _distribution_loglikelihood(distribution, resid, sigma2, nu=nu, lam=lam)
    score = None
    hessian = None
    if compute_score or compute_hessian:
        raise NotImplementedError(
            "Analytical score/hessian are not available for generic fallback families. "
            "This path intentionally does not fall back to Python numerical derivatives."
        )

    return FilteredState(
        data=np.ascontiguousarray(data, dtype=np.float64),
        mean=np.ascontiguousarray(mean, dtype=np.float64),
        residuals=np.ascontiguousarray(resid, dtype=np.float64),
        sigma2=sigma2,
        log_likelihood=log_likelihood,
        score=score,
        hessian=hessian,
        distribution=distribution,
        nu=nu,
        lam=lam,
    )


def _eval_garch_like_core(
    family: str,
    params: NDArray[np.float64],
    data: NDArray[np.float64],
    sigma2: NDArray[np.float64],
    p: int,
    q: int,
    suffix: str,
    *,
    compute_score: bool,
    compute_hessian: bool,
) -> tuple[float, Optional[NDArray[np.float64]], Optional[NDArray[np.float64]]]:
    special = p == 1 and q == 1
    infix = "11" if special else "pq"
    obj_fn = getattr(_core, f"_{family}_ll_{infix}_{suffix}")
    grad_fn = getattr(_core, f"_{family}_ll_grad_{infix}_{suffix}")
    hess_fn = getattr(_core, f"_{family}_ll_hess_{infix}_{suffix}")

    params_c = _as_cptr(params)
    data_c = _as_cptr(data)
    sigma2_c = _as_cptr(sigma2)
    n = data.size
    k = params.size

    if special:
        nll = float(obj_fn(params_c, data_c, sigma2_c, n))
    else:
        nll = float(obj_fn(params_c, data_c, sigma2_c, n, p, q))

    score = None
    if compute_score:
        grad = np.empty(k, dtype=np.float64)
        if special:
            grad_fn(params_c, data_c, sigma2_c, _as_cptr(grad), n)
        else:
            grad_fn(params_c, data_c, sigma2_c, _as_cptr(grad), n, p, q)
        score = -grad

    hessian = None
    if compute_hessian:
        hess = np.empty((k, k), dtype=np.float64)
        if special:
            hess_fn(params_c, data_c, sigma2_c, _as_cptr(hess), n)
        else:
            hess_fn(params_c, data_c, sigma2_c, _as_cptr(hess), n, p, q)
        hessian = -hess

    return nll, score, hessian


def _compute_garch_variance(theta: NDArray[np.float64], resid2: NDArray[np.float64], sigma2: NDArray[np.float64], p: int, q: int) -> None:
    if p == 1 and q == 1:
        _core._garch_variance_11(_as_cptr(theta), _as_cptr(resid2), _as_cptr(sigma2), resid2.size)
    else:
        _core._garch_variance_pq(_as_cptr(theta), _as_cptr(resid2), _as_cptr(sigma2), resid2.size, p, q)


def _compute_gjr_variance(theta: NDArray[np.float64], resid: NDArray[np.float64], sigma2: NDArray[np.float64], p: int, q: int) -> None:
    if p == 1 and q == 1:
        _core._gjr_garch_variance_11(_as_cptr(theta), _as_cptr(resid), _as_cptr(sigma2), resid.size)
    else:
        _core._gjr_garch_variance_pq(_as_cptr(theta), _as_cptr(resid), _as_cptr(sigma2), resid.size, p, q)


def _compute_egarch_variance(
    theta: NDArray[np.float64],
    resid: NDArray[np.float64],
    sigma2: NDArray[np.float64],
    p: int,
    q: int,
    distribution: str = "Normal",
) -> None:
    if distribution == "Normal":
        if p == 1 and q == 1:
            _core._egarch_variance_11(_as_cptr(theta[: 1 + 2 * p + q]), _as_cptr(resid), _as_cptr(sigma2), resid.size)
        else:
            _core._egarch_variance_pq(_as_cptr(theta[: 1 + 2 * p + q]), _as_cptr(resid), _as_cptr(sigma2), resid.size, p, q)
        return
    if distribution == "StudentT":
        if p == 1 and q == 1:
            _core._egarch_ll_11_studentt(_as_cptr(theta), _as_cptr(resid), _as_cptr(sigma2), resid.size)
        else:
            _core._egarch_ll_pq_studentt(_as_cptr(theta), _as_cptr(resid), _as_cptr(sigma2), resid.size, p, q)
        return
    if distribution == "SkewT":
        if p == 1 and q == 1:
            _core._egarch_ll_11_skewt(_as_cptr(theta), _as_cptr(resid), _as_cptr(sigma2), resid.size)
        else:
            _core._egarch_ll_pq_skewt(_as_cptr(theta), _as_cptr(resid), _as_cptr(sigma2), resid.size, p, q)
        return
    if distribution == "GED":
        if p == 1 and q == 1:
            _core._egarch_ll_11_ged(_as_cptr(theta), _as_cptr(resid), _as_cptr(sigma2), resid.size)
        else:
            _core._egarch_ll_pq_ged(_as_cptr(theta), _as_cptr(resid), _as_cptr(sigma2), resid.size, p, q)
        return
    raise RuntimeError(f"EGARCH variance recursion is not shipped for distribution '{distribution}'.")


def _egarch_abs_moment(
    distribution: str,
    nu: Optional[float],
    lam: Optional[float],
    rho: Optional[float] = None,
    zeta: Optional[float] = None,
    shape_lambda: Optional[float] = None,
) -> float:
    if distribution == "Normal":
        return float(np.sqrt(2.0 / np.pi))
    if distribution == "StudentT":
        if nu is None or nu <= 2.0:
            raise ValueError("StudentT EGARCH centering requires `nu > 2`.")
        log_value = (
            np.log(2.0)
            + 0.5 * np.log(nu - 2.0)
            + gammaln(0.5 * (nu + 1.0))
            - np.log(nu - 1.0)
            - 0.5 * np.log(np.pi)
            - gammaln(0.5 * nu)
        )
        return float(np.exp(log_value))
    if distribution == "SkewT":
        if nu is None or lam is None or nu <= 2.0:
            raise ValueError("SkewT EGARCH centering requires `nu > 2` and finite `lam`.")
        return _egarch_skewt_abs_moment(float(nu), float(lam))
    if distribution == "GED":
        if nu is None or nu <= 0.0:
            raise ValueError("GED EGARCH centering requires finite `nu > 0`.")
        log_value = gammaln(2.0 / nu) - 0.5 * (gammaln(1.0 / nu) + gammaln(3.0 / nu))
        return float(np.exp(log_value))
    if distribution in {"GeneralizedHyperbolic", "Hyperbolic", "NormalInverseGaussian"}:
        if rho is None or zeta is None or shape_lambda is None:
            raise ValueError(f"{distribution} EGARCH centering requires GH shape parameters.")
        return standardized_gh_abs_moment(rho, zeta, shape_lambda)
    if distribution == "VarianceGamma":
        if rho is None or shape_lambda is None:
            raise ValueError("VarianceGamma EGARCH centering requires `rho` and `lambda`.")
        return standardized_vg_abs_moment(rho, shape_lambda)
    raise NotImplementedError(f"Unsupported EGARCH distribution '{distribution}'.")


def _filter_aparch(
    spec: Any,
    data: NDArray[np.float64],
    theta: NDArray[np.float64],
    *,
    compute_score: bool,
    compute_hessian: bool,
) -> FilteredState:
    from ._native_fit import (
        _aparch_volatility_domain,
        _describe_model,
        _evaluate_theta_request,
        _materialize_native_aparch_state,
    )

    vol = spec.get_component(Role.VOLATILITY)
    dens = spec.get_component(Role.DENSITY)
    assert isinstance(vol, APARCH)
    model = _describe_model(spec)
    request = 1 | (2 if compute_score else 0) | (4 if compute_hessian else 0)
    nll, gradient, hessian = _evaluate_theta_request(
        model, data, theta, request=request
    )
    residuals, sigma2 = _materialize_native_aparch_state(
        model, data, theta
    )
    if vol.fitted_params is not None:
        _, _, persistence = _aparch_volatility_domain(model, theta)
        vol.fitted_params["persistence"] = persistence
    score = -gradient if compute_score else None
    observed_hessian = -hessian if compute_hessian else None

    shape_start = model.n_mean + model.n_vol
    kwargs: dict[str, float] = {}
    if isinstance(dens, (StudentT, GED)):
        kwargs["nu"] = float(theta[shape_start])
    elif isinstance(dens, SkewT):
        kwargs["nu"] = float(theta[shape_start])
        kwargs["lam"] = float(theta[shape_start + 1])
    elif isinstance(dens, GeneralizedHyperbolic):
        kwargs["rho"] = float(theta[shape_start])
        kwargs["zeta"] = float(theta[shape_start + 1])
        kwargs["shape_lambda"] = float(theta[shape_start + 2])
    elif isinstance(dens, NormalInverseGaussian):
        kwargs["rho"] = float(theta[shape_start])
        kwargs["zeta"] = float(theta[shape_start + 1])
        kwargs["shape_lambda"] = -0.5
    elif isinstance(dens, Hyperbolic):
        kwargs["rho"] = float(theta[shape_start])
        kwargs["zeta"] = float(theta[shape_start + 1])
        kwargs["shape_lambda"] = 1.0
    elif isinstance(dens, VarianceGamma):
        kwargs["rho"] = float(theta[shape_start])
        kwargs["shape_lambda"] = float(theta[shape_start + 1])

    return FilteredState(
        data,
        data - residuals,
        residuals,
        sigma2,
        -nll,
        score,
        observed_hessian,
        dens.signature,
        **kwargs,
    )


def _filter_garch(spec: Any, data: NDArray[np.float64], theta: NDArray[np.float64], *, compute_score: bool, compute_hessian: bool) -> FilteredState:
    vol = spec.get_component(Role.VOLATILITY)
    dens = spec.get_component(Role.DENSITY)
    assert isinstance(vol, GARCH)
    p, q = vol.p, vol.q

    resid = np.ascontiguousarray(data, dtype=np.float64)
    resid2 = resid ** 2
    sigma2 = np.empty_like(resid)
    sigma2[0] = float(np.mean(resid2))
    mean = np.zeros_like(resid)

    if isinstance(dens, Normal):
        nll_var, score, hess = _eval_garch_like_core(
            "garch",
            theta,
            resid2,
            sigma2,
            p,
            q,
            "normal",
            compute_score=compute_score,
            compute_hessian=compute_hessian,
        )
        nll = nll_var + 0.5 * resid.size * np.log(2.0 * np.pi)
        return FilteredState(data, mean, resid, sigma2, -nll, score, hess, "Normal")

    if isinstance(dens, StudentT):
        nll, score, hess = _eval_garch_like_core(
            "garch",
            theta,
            resid2,
            sigma2,
            p,
            q,
            "studentt",
            compute_score=compute_score,
            compute_hessian=compute_hessian,
        )
        return FilteredState(data, mean, resid, sigma2, -nll, score, hess, "StudentT", nu=float(theta[-1]))

    if isinstance(dens, GED):
        nll, score, hess = _eval_garch_like_core(
            "garch",
            theta,
            resid2,
            sigma2,
            p,
            q,
            "ged",
            compute_score=compute_score,
            compute_hessian=compute_hessian,
        )
        return FilteredState(data, mean, resid, sigma2, -nll, score, hess, "GED", nu=float(theta[-1]))

    if isinstance(dens, SkewT):
        _compute_garch_variance(theta[: 1 + p + q], resid2, sigma2, p, q)
        nll = float(_core._skewt_nll(_as_cptr(resid), _as_cptr(sigma2), resid.size, float(theta[-2]), float(theta[-1])))

        score = None
        if compute_score:
            grad = np.empty(theta.size, dtype=np.float64)
            if p == 1 and q == 1:
                _core._garch_ll_grad_11_skewt(_as_cptr(theta), _as_cptr(resid), _as_cptr(grad), resid.size)
            else:
                _core._garch_ll_grad_pq_skewt(_as_cptr(theta), _as_cptr(resid), _as_cptr(sigma2), _as_cptr(grad), resid.size, p, q)
            score = -grad

        hess = None
        if compute_hessian:
            hess_mat = np.empty((theta.size, theta.size), dtype=np.float64)
            if p == 1 and q == 1:
                _core._garch_ll_hess_11_skewt(_as_cptr(theta), _as_cptr(resid), _as_cptr(sigma2), _as_cptr(hess_mat), resid.size)
            else:
                _core._garch_ll_hess_pq_skewt(_as_cptr(theta), _as_cptr(resid), _as_cptr(sigma2), _as_cptr(hess_mat), resid.size, p, q)
            hess = -hess_mat

        return FilteredState(data, mean, resid, sigma2, -nll, score, hess, "SkewT", nu=float(theta[-2]), lam=float(theta[-1]))

    raise NotImplementedError(f"Unsupported density for GARCH filtering: {dens}")


def _filter_gjr(spec: Any, data: NDArray[np.float64], theta: NDArray[np.float64], *, compute_score: bool, compute_hessian: bool) -> FilteredState:
    vol = spec.get_component(Role.VOLATILITY)
    dens = spec.get_component(Role.DENSITY)
    assert isinstance(vol, GJRGARCH)
    p, q = vol.p, vol.q

    resid = np.ascontiguousarray(data, dtype=np.float64)
    sigma2 = np.empty_like(resid)
    sigma2[0] = float(np.mean(resid ** 2))
    mean = np.zeros_like(resid)

    if isinstance(dens, Normal):
        nll_var, score, hess = _eval_garch_like_core(
            "gjr_garch",
            theta,
            resid,
            sigma2,
            p,
            q,
            "normal",
            compute_score=compute_score,
            compute_hessian=compute_hessian,
        )
        nll = nll_var + 0.5 * resid.size * np.log(2.0 * np.pi)
        return FilteredState(data, mean, resid, sigma2, -nll, score, hess, "Normal")

    if isinstance(dens, StudentT):
        nll, score, hess = _eval_garch_like_core(
            "gjr_garch",
            theta,
            resid,
            sigma2,
            p,
            q,
            "studentt",
            compute_score=compute_score,
            compute_hessian=compute_hessian,
        )
        return FilteredState(data, mean, resid, sigma2, -nll, score, hess, "StudentT", nu=float(theta[-1]))

    if isinstance(dens, SkewT):
        _compute_gjr_variance(theta[: 1 + 2 * p + q], resid, sigma2, p, q)
        nll = float(_core._skewt_nll(_as_cptr(resid), _as_cptr(sigma2), resid.size, float(theta[-2]), float(theta[-1])))

        score = None
        if compute_score:
            grad = np.empty(theta.size, dtype=np.float64)
            if p == 1 and q == 1:
                _core._gjr_garch_ll_grad_11_skewt(_as_cptr(theta), _as_cptr(resid), _as_cptr(sigma2), _as_cptr(grad), resid.size)
            else:
                _core._gjr_garch_ll_grad_pq_skewt(_as_cptr(theta), _as_cptr(resid), _as_cptr(sigma2), _as_cptr(grad), resid.size, p, q)
            score = -grad

        hess = None
        if compute_hessian:
            hess_mat = np.empty((theta.size, theta.size), dtype=np.float64)
            if p == 1 and q == 1:
                _core._gjr_garch_ll_hess_11_skewt(_as_cptr(theta), _as_cptr(resid), _as_cptr(sigma2), _as_cptr(hess_mat), resid.size)
            else:
                _core._gjr_garch_ll_hess_pq_skewt(_as_cptr(theta), _as_cptr(resid), _as_cptr(sigma2), _as_cptr(hess_mat), resid.size, p, q)
            hess = -hess_mat

        return FilteredState(data, mean, resid, sigma2, -nll, score, hess, "SkewT", nu=float(theta[-2]), lam=float(theta[-1]))

    if isinstance(dens, GED):
        y = np.ascontiguousarray(data, dtype=np.float64)
        features = np.empty((y.size, 0), dtype=np.float64)
        resid = np.zeros_like(y)
        sigma2 = np.zeros_like(y)
        sigma2[0] = float(np.mean(y * y))

        params_c = _as_cptr(theta)
        y_c = _as_cptr(y)
        feat_c = _as_cptr(features)
        resid_c = _as_cptr(resid)
        sigma2_c = _as_cptr(sigma2)

        if p == 1 and q == 1:
            nll = float(_core._linear_mean_gjr_garch_nll_11_ged(params_c, y_c, feat_c, resid_c, sigma2_c, y.size, 0))
        else:
            nll = float(_core._linear_mean_gjr_garch_nll_pq_ged(params_c, y_c, feat_c, resid_c, sigma2_c, y.size, 0, p, q))

        score = None
        if compute_score:
            grad = np.empty(theta.size, dtype=np.float64)
            if p == 1 and q == 1:
                _core._linear_mean_gjr_garch_nll_grad_11_ged(params_c, y_c, feat_c, resid_c, sigma2_c, _as_cptr(grad), y.size, 0)
            else:
                _core._linear_mean_gjr_garch_nll_grad_pq_ged(
                    params_c, y_c, feat_c, resid_c, sigma2_c, _as_cptr(grad), y.size, 0, p, q
                )
            score = -grad

        hess = None
        if compute_hessian:
            hess_mat = np.empty((theta.size, theta.size), dtype=np.float64)
            if p == 1 and q == 1:
                _core._linear_mean_gjr_garch_hess_11_ged(params_c, y_c, feat_c, resid_c, sigma2_c, _as_cptr(hess_mat), y.size, 0)
            else:
                _core._linear_mean_gjr_garch_hess_pq_ged(
                    params_c, y_c, feat_c, resid_c, sigma2_c, _as_cptr(hess_mat), y.size, 0, p, q
                )
            hess = -hess_mat

        return FilteredState(data, mean, resid, sigma2, -nll, score, hess, "GED", nu=float(theta[-1]))

    raise NotImplementedError(f"Unsupported density for GJR-GARCH filtering: {dens}")


def _egarch_density_metadata(
    dens: Any,
    shape: NDArray[np.float64],
) -> dict[str, Optional[float]]:
    if isinstance(dens, Normal):
        return {}
    if isinstance(dens, (StudentT, GED)):
        return {"nu": float(shape[0])}
    if isinstance(dens, SkewT):
        return {"nu": float(shape[0]), "lam": float(shape[1])}
    if isinstance(dens, GeneralizedHyperbolic):
        return {
            "rho": float(shape[0]),
            "zeta": float(shape[1]),
            "shape_lambda": float(shape[2]),
        }
    if isinstance(dens, (NormalInverseGaussian, Hyperbolic)):
        fixed_lambda = fixed_gh_lambda(dens.signature)
        assert fixed_lambda is not None
        return {
            "rho": float(shape[0]),
            "zeta": float(shape[1]),
            "shape_lambda": float(fixed_lambda),
        }
    if isinstance(dens, VarianceGamma):
        return {
            "rho": float(shape[0]),
            "shape_lambda": float(shape[1]),
        }
    raise NotImplementedError(f"Unsupported density for EGARCH filtering: {dens}")


def _filter_egarch(
    spec: Any,
    data: NDArray[np.float64],
    theta: NDArray[np.float64],
    *,
    compute_score: bool,
    compute_hessian: bool,
) -> FilteredState:
    from ._native_fit import (
        _describe_model,
        _evaluate_theta_request,
        _materialize_native_egarch_variance,
    )

    vol = spec.get_component(Role.VOLATILITY)
    dens = spec.get_component(Role.DENSITY)
    assert isinstance(vol, EGARCH)
    resid = np.ascontiguousarray(data, dtype=np.float64)
    model = _describe_model(spec)
    request = 1 | (2 if compute_score else 0) | (4 if compute_hessian else 0)
    nll, gradient, hessian = _evaluate_theta_request(
        model, resid, np.ascontiguousarray(theta), request=request
    )
    sigma2 = _materialize_native_egarch_variance(model, resid, theta)
    metadata = _egarch_density_metadata(dens, theta[model.n_vol :])
    return FilteredState(
        data=data,
        mean=np.zeros_like(resid),
        residuals=resid,
        sigma2=sigma2,
        log_likelihood=-nll,
        score=-gradient if compute_score else None,
        hessian=-hessian if compute_hessian else None,
        distribution=dens.signature,
        **metadata,
    )


def _filter_arma(spec: Any, data: NDArray[np.float64], theta: NDArray[np.float64], *, compute_score: bool, compute_hessian: bool) -> FilteredState:
    mean_comp = spec.get_component(Role.MEAN)
    dens = spec.get_component(Role.DENSITY)
    assert isinstance(mean_comp, ARMA)
    if not isinstance(dens, Normal):
        raise NotImplementedError("Standalone ARMA filtering is only available for Normal density.")

    p, q = mean_comp.p, mean_comp.q
    y = np.ascontiguousarray(data, dtype=np.float64)
    resid = np.zeros_like(y)
    e0 = np.zeros(max(q, 1), dtype=np.float64)
    n = y.size

    special = p == 1 and q == 1
    if special:
        nll = float(_core._arma_nll_11_normal(_as_cptr(theta), _as_cptr(y), _as_cptr(resid), n))
    else:
        nll = float(_core._arma_nll_pq_normal(_as_cptr(theta), _as_cptr(y), _as_cptr(resid), _as_cptr(e0), n, p, q))

    score = None
    if compute_score:
        grad = np.empty(theta.size, dtype=np.float64)
        if special:
            _core._arma_nll_grad_11_normal(_as_cptr(theta), _as_cptr(y), _as_cptr(resid), _as_cptr(grad), n)
        else:
            _core._arma_nll_grad_pq_normal(_as_cptr(theta), _as_cptr(y), _as_cptr(resid), _as_cptr(e0), _as_cptr(grad), n, p, q)
        score = -grad

    hess = None
    if compute_hessian:
        hess_mat = np.empty((theta.size, theta.size), dtype=np.float64)
        if special:
            _core._arma_hess_11_normal(_as_cptr(theta), _as_cptr(y), _as_cptr(resid), _as_cptr(hess_mat), n)
        else:
            _core._arma_hess_pq_normal(_as_cptr(theta), _as_cptr(y), _as_cptr(resid), _as_cptr(e0), _as_cptr(hess_mat), n, p, q)
        hess = -hess_mat

    n_eff = max(n - 1, 1)
    sigma2_hat = float(np.sum(resid[1:] ** 2) / n_eff)
    sigma2 = np.full(n, sigma2_hat, dtype=np.float64)
    mean = y - resid
    return FilteredState(y, mean, resid, sigma2, -nll, score, hess, "Normal")


def _filter_arma_ged(spec: Any, data: NDArray[np.float64], theta: NDArray[np.float64], *, compute_score: bool, compute_hessian: bool) -> FilteredState:
    mean_comp = spec.get_component(Role.MEAN)
    dens = spec.get_component(Role.DENSITY)
    assert isinstance(mean_comp, ARMA)
    assert isinstance(dens, GED)

    p, q = mean_comp.p, mean_comp.q
    y = np.ascontiguousarray(data, dtype=np.float64)
    resid = np.zeros_like(y)
    e0 = np.zeros(max(q, 1), dtype=np.float64)
    n = y.size
    special = p == 1 and q == 1

    if special:
        nll = float(_core._arma_nll_11_ged(_as_cptr(theta), _as_cptr(y), _as_cptr(resid), n))
    else:
        nll = float(_core._arma_nll_pq_ged(_as_cptr(theta), _as_cptr(y), _as_cptr(resid), _as_cptr(e0), n, p, q))

    score = None
    if compute_score:
        grad = np.empty(theta.size, dtype=np.float64)
        if special:
            _core._arma_nll_grad_11_ged(_as_cptr(theta), _as_cptr(y), _as_cptr(resid), _as_cptr(grad), n)
        else:
            _core._arma_nll_grad_pq_ged(_as_cptr(theta), _as_cptr(y), _as_cptr(resid), _as_cptr(e0), _as_cptr(grad), n, p, q)
        score = -grad

    hess = None
    if compute_hessian:
        hess_mat = np.empty((theta.size, theta.size), dtype=np.float64)
        if special:
            _core._arma_hess_11_ged(_as_cptr(theta), _as_cptr(y), _as_cptr(resid), _as_cptr(hess_mat), n)
        else:
            _core._arma_hess_pq_ged(_as_cptr(theta), _as_cptr(y), _as_cptr(resid), _as_cptr(e0), _as_cptr(hess_mat), n, p, q)
        hess = -hess_mat

    sigma2 = np.full(n, float(theta[1 + p + q]), dtype=np.float64)
    mean = y - resid
    return FilteredState(y, mean, resid, sigma2, -nll, score, hess, "GED", nu=float(theta[-1]))


def _filter_arma_garch(spec: Any, data: NDArray[np.float64], theta: NDArray[np.float64], *, compute_score: bool, compute_hessian: bool) -> FilteredState:
    mean_comp = spec.get_component(Role.MEAN)
    vol = spec.get_component(Role.VOLATILITY)
    dens = spec.get_component(Role.DENSITY)
    assert isinstance(mean_comp, ARMA)
    assert isinstance(vol, GARCH)

    p_ar, q_ma = mean_comp.p, mean_comp.q
    p_arch, q_garch = vol.p, vol.q
    y = np.ascontiguousarray(data, dtype=np.float64)
    resid = np.zeros_like(y)
    sigma2 = np.zeros_like(y)
    h0 = float(np.mean(y ** 2))
    max_lag = max(p_ar, q_ma, p_arch, q_garch, 1)
    e0 = np.zeros(max_lag, dtype=np.float64)
    h0_arr = np.full(max_lag, h0, dtype=np.float64)
    n = y.size
    special = p_ar == 1 and q_ma == 1 and p_arch == 1 and q_garch == 1

    if isinstance(dens, Normal):
        suffix = "normal"
    elif isinstance(dens, StudentT):
        suffix = "studentt"
    elif isinstance(dens, SkewT):
        suffix = "skewt"
    elif isinstance(dens, GED):
        suffix = "ged"
    else:
        raise NotImplementedError(f"Unsupported density for ARMA-GARCH filtering: {dens}")

    infix = "11" if special else "pq"
    nll_fn = getattr(_core, f"_arma_garch_nll_{infix}_{suffix}")
    grad_fn = getattr(_core, f"_arma_garch_nll_grad_{infix}_{suffix}")
    hess_fn = getattr(_core, f"_arma_garch_hess_{infix}_{suffix}")

    params_c = _as_cptr(theta)
    y_c = _as_cptr(y)
    resid_c = _as_cptr(resid)
    sigma2_c = _as_cptr(sigma2)
    if special:
        nll = float(nll_fn(params_c, y_c, resid_c, sigma2_c, h0, n))
    else:
        nll = float(
            nll_fn(
                params_c,
                y_c,
                resid_c,
                sigma2_c,
                _as_cptr(e0),
                _as_cptr(h0_arr),
                n,
                p_ar,
                q_ma,
                p_arch,
                q_garch,
            )
        )

    score = None
    if compute_score:
        grad = np.empty(theta.size, dtype=np.float64)
        if special:
            grad_fn(params_c, y_c, resid_c, sigma2_c, _as_cptr(grad), h0, n)
        else:
            grad_fn(
                params_c,
                y_c,
                resid_c,
                sigma2_c,
                _as_cptr(e0),
                _as_cptr(h0_arr),
                _as_cptr(grad),
                n,
                p_ar,
                q_ma,
                p_arch,
                q_garch,
            )
        score = -_arma_volatility_total_derivative(
            grad, n, p_ar, q_ma, p_arch, q_garch
        )

    hess = None
    if compute_hessian:
        hess_mat = np.empty((theta.size, theta.size), dtype=np.float64)
        if special:
            hess_fn(params_c, y_c, resid_c, sigma2_c, _as_cptr(hess_mat), h0, n)
        else:
            hess_fn(
                params_c,
                y_c,
                resid_c,
                sigma2_c,
                _as_cptr(e0),
                _as_cptr(h0_arr),
                _as_cptr(hess_mat),
                n,
                p_ar,
                q_ma,
                p_arch,
                q_garch,
            )
        hess = -_arma_volatility_total_derivative(
            hess_mat, n, p_ar, q_ma, p_arch, q_garch
        )

    mean = y - resid
    nu = float(theta[-1]) if isinstance(dens, (StudentT, GED)) else (float(theta[-2]) if isinstance(dens, SkewT) else None)
    lam = float(theta[-1]) if isinstance(dens, SkewT) else None
    nll_total = _arma_volatility_total_nll(
        nll,
        n,
        p_ar,
        q_ma,
        p_arch,
        q_garch,
        normal_constant_omitted=isinstance(dens, Normal),
    )
    return FilteredState(y, mean, resid, sigma2, -nll_total, score, hess, dens.signature, nu=nu, lam=lam)


def _filter_arma_gjr(spec: Any, data: NDArray[np.float64], theta: NDArray[np.float64], *, compute_score: bool, compute_hessian: bool) -> FilteredState:
    mean_comp = spec.get_component(Role.MEAN)
    vol = spec.get_component(Role.VOLATILITY)
    dens = spec.get_component(Role.DENSITY)
    assert isinstance(mean_comp, ARMA)
    assert isinstance(vol, GJRGARCH)

    p_ar, q_ma = mean_comp.p, mean_comp.q
    p_arch, q_garch = vol.p, vol.q
    y = np.ascontiguousarray(data, dtype=np.float64)
    resid = np.zeros_like(y)
    sigma2 = np.zeros_like(y)
    h0 = float(np.mean(y ** 2))
    max_lag = max(p_ar, q_ma, p_arch, q_garch, 1)
    e0 = np.zeros(max_lag, dtype=np.float64)
    h0_arr = np.full(max_lag, h0, dtype=np.float64)
    n = y.size
    special = p_ar == 1 and q_ma == 1 and p_arch == 1 and q_garch == 1

    if isinstance(dens, Normal):
        suffix = "normal"
    elif isinstance(dens, StudentT):
        suffix = "studentt"
    elif isinstance(dens, SkewT):
        suffix = "skewt"
    elif isinstance(dens, GED):
        suffix = "ged"
    else:
        raise NotImplementedError(f"Unsupported density for ARMA-GJR-GARCH filtering: {dens}")

    infix = "11" if special else "pq"
    nll_fn = getattr(_core, f"_arma_gjr_garch_nll_{infix}_{suffix}")
    grad_fn = getattr(_core, f"_arma_gjr_garch_nll_grad_{infix}_{suffix}")
    hess_fn = getattr(_core, f"_arma_gjr_garch_hess_{infix}_{suffix}")

    params_c = _as_cptr(theta)
    y_c = _as_cptr(y)
    resid_c = _as_cptr(resid)
    sigma2_c = _as_cptr(sigma2)
    if special:
        nll = float(nll_fn(params_c, y_c, resid_c, sigma2_c, h0, n))
    else:
        nll = float(
            nll_fn(
                params_c,
                y_c,
                resid_c,
                sigma2_c,
                _as_cptr(e0),
                _as_cptr(h0_arr),
                n,
                p_ar,
                q_ma,
                p_arch,
                q_garch,
            )
        )

    score = None
    if compute_score:
        grad = np.empty(theta.size, dtype=np.float64)
        if special:
            grad_fn(params_c, y_c, resid_c, sigma2_c, _as_cptr(grad), h0, n)
        else:
            grad_fn(
                params_c,
                y_c,
                resid_c,
                sigma2_c,
                _as_cptr(e0),
                _as_cptr(h0_arr),
                _as_cptr(grad),
                n,
                p_ar,
                q_ma,
                p_arch,
                q_garch,
            )
        score = -_arma_volatility_total_derivative(
            grad, n, p_ar, q_ma, p_arch, q_garch
        )

    hess = None
    if compute_hessian:
        hess_mat = np.empty((theta.size, theta.size), dtype=np.float64)
        if special:
            hess_fn(params_c, y_c, resid_c, sigma2_c, _as_cptr(hess_mat), h0, n)
        else:
            hess_fn(
                params_c,
                y_c,
                resid_c,
                sigma2_c,
                _as_cptr(e0),
                _as_cptr(h0_arr),
                _as_cptr(hess_mat),
                n,
                p_ar,
                q_ma,
                p_arch,
                q_garch,
            )
        hess = -_arma_volatility_total_derivative(
            hess_mat, n, p_ar, q_ma, p_arch, q_garch
        )

    mean = y - resid
    nu = float(theta[-1]) if isinstance(dens, (StudentT, GED)) else (float(theta[-2]) if isinstance(dens, SkewT) else None)
    lam = float(theta[-1]) if isinstance(dens, SkewT) else None
    nll_total = _arma_volatility_total_nll(
        nll,
        n,
        p_ar,
        q_ma,
        p_arch,
        q_garch,
        normal_constant_omitted=isinstance(dens, Normal),
    )
    return FilteredState(y, mean, resid, sigma2, -nll_total, score, hess, dens.signature, nu=nu, lam=lam)


def _filter_meanx_garch(
    spec: Any,
    data: NDArray[np.float64],
    theta: NDArray[np.float64],
    *,
    x: NDArray[np.float64] | None,
    compute_score: bool,
    compute_hessian: bool,
) -> FilteredState:
    mean_comp = spec.get_component(Role.MEAN)
    vol = spec.get_component(Role.VOLATILITY)
    dens = spec.get_component(Role.DENSITY)
    assert isinstance(mean_comp, (ARX, HARX))
    assert isinstance(vol, GARCH)
    assert isinstance(dens, (Normal, StudentT, SkewT, GED))

    if x is None:
        raise ValueError(f"{mean_comp.signature}+GARCH({vol.p},{vol.q}) requires exogenous regressors via `x=`.")

    mean_comp.set_n_exog(int(x.shape[1]))
    y = np.ascontiguousarray(data, dtype=np.float64)
    features = build_linear_mean_features(mean_comp, y, x)
    resid = np.zeros_like(y)
    sigma2 = np.zeros_like(y)
    sigma2[0] = np.mean(y * y)
    n = y.size
    n_mean = mean_comp.n_params
    special = vol.p == 1 and vol.q == 1
    if isinstance(dens, Normal):
        suffix = "normal"
    elif isinstance(dens, StudentT):
        suffix = "studentt"
    elif isinstance(dens, SkewT):
        suffix = "skewt"
    else:
        suffix = "ged"

    params_c = _as_cptr(theta)
    y_c = _as_cptr(y)
    feat_c = _as_cptr(features)
    resid_c = _as_cptr(resid)
    sigma2_c = _as_cptr(sigma2)

    if special:
        nll = float(getattr(_core, f"_linear_mean_garch_nll_11_{suffix}")(params_c, y_c, feat_c, resid_c, sigma2_c, n, n_mean))
    else:
        nll = float(getattr(_core, f"_linear_mean_garch_nll_pq_{suffix}")(params_c, y_c, feat_c, resid_c, sigma2_c, n, n_mean, vol.p, vol.q))

    score = None
    if compute_score:
        grad = np.empty(theta.size, dtype=np.float64)
        if special:
            getattr(_core, f"_linear_mean_garch_nll_grad_11_{suffix}")(params_c, y_c, feat_c, resid_c, sigma2_c, _as_cptr(grad), n, n_mean)
        else:
            getattr(_core, f"_linear_mean_garch_nll_grad_pq_{suffix}")(
                params_c, y_c, feat_c, resid_c, sigma2_c, _as_cptr(grad), n, n_mean, vol.p, vol.q
            )
        score = -grad

    hess = None
    if compute_hessian:
        hess_mat = np.empty((theta.size, theta.size), dtype=np.float64)
        if special:
            getattr(_core, f"_linear_mean_garch_hess_11_{suffix}")(params_c, y_c, feat_c, resid_c, sigma2_c, _as_cptr(hess_mat), n, n_mean)
        else:
            getattr(_core, f"_linear_mean_garch_hess_pq_{suffix}")(
                params_c, y_c, feat_c, resid_c, sigma2_c, _as_cptr(hess_mat), n, n_mean, vol.p, vol.q
            )
        hess = -hess_mat

    mean = y - resid
    nu = float(theta[-1]) if isinstance(dens, (StudentT, GED)) else (float(theta[-2]) if isinstance(dens, SkewT) else None)
    lam = float(theta[-1]) if isinstance(dens, SkewT) else None
    return FilteredState(y, mean, resid, sigma2, -nll, score, hess, dens.signature, nu=nu, lam=lam)


def _filter_meanx_normal(
    spec: Any,
    data: NDArray[np.float64],
    theta: NDArray[np.float64],
    *,
    x: NDArray[np.float64] | None,
    compute_score: bool,
    compute_hessian: bool,
) -> FilteredState:
    mean_comp = spec.get_component(Role.MEAN)
    dens = spec.get_component(Role.DENSITY)
    assert isinstance(mean_comp, (ARX, HARX))
    assert isinstance(dens, Normal)

    if x is None:
        raise ValueError(f"{mean_comp.signature} requires exogenous regressors via `x=`.")

    mean_comp.set_n_exog(int(x.shape[1]))
    y = np.ascontiguousarray(data, dtype=np.float64)
    features = build_linear_mean_features(mean_comp, y, x)
    resid = np.zeros_like(y)
    n = y.size
    n_mean = mean_comp.n_params

    nll = float(_core._linear_mean_nll_normal(_as_cptr(theta), _as_cptr(y), _as_cptr(features), _as_cptr(resid), n, n_mean))

    score = None
    if compute_score:
        grad = np.empty(theta.size, dtype=np.float64)
        _core._linear_mean_nll_grad_normal(_as_cptr(theta), _as_cptr(y), _as_cptr(features), _as_cptr(resid), _as_cptr(grad), n, n_mean)
        score = -grad

    hess = None
    if compute_hessian:
        hess_mat = np.empty((theta.size, theta.size), dtype=np.float64)
        _core._linear_mean_hess_normal(_as_cptr(theta), _as_cptr(y), _as_cptr(features), _as_cptr(resid), _as_cptr(hess_mat), n, n_mean)
        hess = -hess_mat

    mean = y - resid
    sigma2 = np.full(y.size, max(float(np.mean(resid * resid)), 1e-12), dtype=np.float64)
    return FilteredState(y, mean, resid, sigma2, -nll, score, hess, dens.signature, nu=None, lam=None)


def _filter_meanx_standalone(
    spec: Any,
    data: NDArray[np.float64],
    theta: NDArray[np.float64],
    *,
    x: NDArray[np.float64] | None,
    compute_score: bool,
    compute_hessian: bool,
) -> FilteredState:
    mean_comp = spec.get_component(Role.MEAN)
    dens = spec.get_component(Role.DENSITY)
    assert isinstance(mean_comp, (ARX, HARX))
    assert isinstance(dens, (StudentT, SkewT, GED))

    if x is None:
        raise ValueError(f"{mean_comp.signature}+{dens.signature} requires exogenous regressors via `x=`.")

    mean_comp.set_n_exog(int(x.shape[1]))
    y = np.ascontiguousarray(data, dtype=np.float64)
    features = build_linear_mean_features(mean_comp, y, x)
    resid = np.zeros_like(y)
    sigma2 = np.zeros_like(y)
    n = y.size
    n_mean = mean_comp.n_params
    suffix = "studentt" if isinstance(dens, StudentT) else "skewt" if isinstance(dens, SkewT) else "ged"

    params_c = _as_cptr(theta)
    y_c = _as_cptr(y)
    feat_c = _as_cptr(features)
    resid_c = _as_cptr(resid)
    sigma2_c = _as_cptr(sigma2)
    sigma2[0] = float(theta[n_mean])
    nll = float(getattr(_core, f"_linear_mean_garch_nll_pq_{suffix}")(params_c, y_c, feat_c, resid_c, sigma2_c, n, n_mean, 0, 0))

    score = None
    if compute_score:
        grad = np.empty(theta.size, dtype=np.float64)
        sigma2[0] = float(theta[n_mean])
        getattr(_core, f"_linear_mean_garch_nll_grad_pq_{suffix}")(
            params_c, y_c, feat_c, resid_c, sigma2_c, _as_cptr(grad), n, n_mean, 0, 0
        )
        score = -grad

    hess = None
    if compute_hessian:
        hess_mat = np.empty((theta.size, theta.size), dtype=np.float64)
        sigma2[0] = float(theta[n_mean])
        getattr(_core, f"_linear_mean_garch_hess_pq_{suffix}")(
            params_c, y_c, feat_c, resid_c, sigma2_c, _as_cptr(hess_mat), n, n_mean, 0, 0
        )
        hess = -hess_mat

    mean = y - resid
    nu = float(theta[-1]) if isinstance(dens, (StudentT, GED)) else float(theta[-2])
    lam = float(theta[-1]) if isinstance(dens, SkewT) else None
    return FilteredState(y, mean, resid, sigma2, -nll, score, hess, dens.signature, nu=nu, lam=lam)


def _filter_meanx_gjr(
    spec: Any,
    data: NDArray[np.float64],
    theta: NDArray[np.float64],
    *,
    x: NDArray[np.float64] | None,
    compute_score: bool,
    compute_hessian: bool,
) -> FilteredState:
    mean_comp = spec.get_component(Role.MEAN)
    vol = spec.get_component(Role.VOLATILITY)
    dens = spec.get_component(Role.DENSITY)
    assert isinstance(mean_comp, (ARX, HARX))
    assert isinstance(vol, GJRGARCH)
    assert isinstance(dens, (Normal, StudentT, SkewT, GED))

    if x is None:
        raise ValueError(f"{mean_comp.signature}+GJR-GARCH({vol.p},{vol.q}) requires exogenous regressors via `x=`.")

    mean_comp.set_n_exog(int(x.shape[1]))
    y = np.ascontiguousarray(data, dtype=np.float64)
    features = build_linear_mean_features(mean_comp, y, x)
    resid = np.zeros_like(y)
    sigma2 = np.zeros_like(y)
    sigma2[0] = np.mean(y * y)
    n = y.size
    n_mean = mean_comp.n_params
    special = vol.p == 1 and vol.q == 1
    if isinstance(dens, Normal):
        suffix = "normal"
    elif isinstance(dens, StudentT):
        suffix = "studentt"
    elif isinstance(dens, SkewT):
        suffix = "skewt"
    else:
        suffix = "ged"

    params_c = _as_cptr(theta)
    y_c = _as_cptr(y)
    feat_c = _as_cptr(features)
    resid_c = _as_cptr(resid)
    sigma2_c = _as_cptr(sigma2)

    if special:
        nll = float(getattr(_core, f"_linear_mean_gjr_garch_nll_11_{suffix}")(params_c, y_c, feat_c, resid_c, sigma2_c, n, n_mean))
    else:
        nll = float(getattr(_core, f"_linear_mean_gjr_garch_nll_pq_{suffix}")(params_c, y_c, feat_c, resid_c, sigma2_c, n, n_mean, vol.p, vol.q))

    score = None
    if compute_score:
        grad = np.empty(theta.size, dtype=np.float64)
        if special:
            getattr(_core, f"_linear_mean_gjr_garch_nll_grad_11_{suffix}")(params_c, y_c, feat_c, resid_c, sigma2_c, _as_cptr(grad), n, n_mean)
        else:
            getattr(_core, f"_linear_mean_gjr_garch_nll_grad_pq_{suffix}")(
                params_c, y_c, feat_c, resid_c, sigma2_c, _as_cptr(grad), n, n_mean, vol.p, vol.q
            )
        score = -grad

    hess = None
    if compute_hessian:
        hess_mat = np.empty((theta.size, theta.size), dtype=np.float64)
        if special:
            getattr(_core, f"_linear_mean_gjr_garch_hess_11_{suffix}")(params_c, y_c, feat_c, resid_c, sigma2_c, _as_cptr(hess_mat), n, n_mean)
        else:
            getattr(_core, f"_linear_mean_gjr_garch_hess_pq_{suffix}")(
                params_c, y_c, feat_c, resid_c, sigma2_c, _as_cptr(hess_mat), n, n_mean, vol.p, vol.q
            )
        hess = -hess_mat

    mean = y - resid
    nu = float(theta[-1]) if isinstance(dens, (StudentT, GED)) else (float(theta[-2]) if isinstance(dens, SkewT) else None)
    lam = float(theta[-1]) if isinstance(dens, SkewT) else None
    return FilteredState(y, mean, resid, sigma2, -nll, score, hess, dens.signature, nu=nu, lam=lam)


def _filter_arma_egarch(spec: Any, data: NDArray[np.float64], theta: NDArray[np.float64], *, compute_score: bool, compute_hessian: bool) -> FilteredState:
    from ._native_fit import (
        _describe_model,
        _evaluate_theta_request,
        _materialize_native_arma_egarch_state,
    )

    dens = spec.get_component(Role.DENSITY)
    y = np.ascontiguousarray(data, dtype=np.float64)
    theta_c = np.ascontiguousarray(theta, dtype=np.float64)
    model = _describe_model(spec)
    request = 1 | (2 if compute_score else 0) | (4 if compute_hessian else 0)
    nll, gradient, hessian = _evaluate_theta_request(
        model, y, theta_c, request=request
    )
    resid, sigma2 = _materialize_native_arma_egarch_state(
        model, y, theta_c
    )
    mean = y - resid
    metadata = _egarch_density_metadata(
        dens, theta_c[model.n_mean + model.n_vol :]
    )
    return FilteredState(
        data=y,
        mean=mean,
        residuals=resid,
        sigma2=sigma2,
        log_likelihood=-nll,
        score=-gradient if compute_score else None,
        hessian=-hessian if compute_hessian else None,
        distribution=dens.signature,
        **metadata,
    )


def _filter_meanx_egarch(
    spec: Any,
    data: NDArray[np.float64],
    theta: NDArray[np.float64],
    *,
    x: NDArray[np.float64] | None,
    compute_score: bool,
    compute_hessian: bool,
) -> FilteredState:
    mean_comp = spec.get_component(Role.MEAN)
    vol = spec.get_component(Role.VOLATILITY)
    dens = spec.get_component(Role.DENSITY)
    assert isinstance(mean_comp, (ARX, HARX))
    assert isinstance(vol, EGARCH)
    assert isinstance(dens, (Normal, StudentT, SkewT, GED))

    if x is None:
        raise ValueError(f"{mean_comp.signature}+EGARCH({vol.p},{vol.q}) requires exogenous regressors via `x=`.")

    mean_comp.set_n_exog(int(x.shape[1]))
    y = np.ascontiguousarray(data, dtype=np.float64)
    features = build_linear_mean_features(mean_comp, y, x)
    resid = np.zeros_like(y)
    sigma2 = np.zeros_like(y)
    sigma2[0] = np.mean(y * y)
    n = y.size
    n_mean = mean_comp.n_params
    special = vol.p == 1 and vol.q == 1
    if isinstance(dens, Normal):
        suffix = "normal"
    elif isinstance(dens, StudentT):
        suffix = "studentt"
    elif isinstance(dens, SkewT):
        suffix = "skewt"
    else:
        suffix = "ged"

    params_c = _as_cptr(theta)
    y_c = _as_cptr(y)
    feat_c = _as_cptr(features)
    resid_c = _as_cptr(resid)
    sigma2_c = _as_cptr(sigma2)

    if special:
        nll = float(getattr(_core, f"_linear_mean_egarch_nll_11_{suffix}")(params_c, y_c, feat_c, resid_c, sigma2_c, n, n_mean))
    else:
        nll = float(getattr(_core, f"_linear_mean_egarch_nll_pq_{suffix}")(params_c, y_c, feat_c, resid_c, sigma2_c, n, n_mean, vol.p, vol.q))

    score = None
    if compute_score:
        grad = np.empty(theta.size, dtype=np.float64)
        sigma2[0] = np.mean(y * y)
        if special:
            getattr(_core, f"_linear_mean_egarch_nll_grad_11_{suffix}")(params_c, y_c, feat_c, resid_c, sigma2_c, _as_cptr(grad), n, n_mean)
        else:
            getattr(_core, f"_linear_mean_egarch_nll_grad_pq_{suffix}")(
                params_c, y_c, feat_c, resid_c, sigma2_c, _as_cptr(grad), n, n_mean, vol.p, vol.q
            )
        score = -grad

    hess = None
    if compute_hessian:
        hess_mat = np.empty((theta.size, theta.size), dtype=np.float64)
        sigma2[0] = np.mean(y * y)
        if special:
            getattr(_core, f"_linear_mean_egarch_hess_11_{suffix}")(params_c, y_c, feat_c, resid_c, sigma2_c, _as_cptr(hess_mat), n, n_mean)
        else:
            getattr(_core, f"_linear_mean_egarch_hess_pq_{suffix}")(
                params_c, y_c, feat_c, resid_c, sigma2_c, _as_cptr(hess_mat), n, n_mean, vol.p, vol.q
            )
        hess = -hess_mat

    mean = y - resid
    nu = float(theta[-1]) if isinstance(dens, (StudentT, GED)) else (float(theta[-2]) if isinstance(dens, SkewT) else None)
    lam = float(theta[-1]) if isinstance(dens, SkewT) else None
    return FilteredState(y, mean, resid, sigma2, -nll, score, hess, dens.signature, nu=nu, lam=lam)


def _filter_gh(
    spec: Any,
    data: NDArray[np.float64],
    theta: NDArray[np.float64],
    *,
    x: NDArray[np.float64] | None,
    compute_score: bool,
    compute_hessian: bool,
) -> FilteredState:
    """Filter any shipped interior GH-family surface through its native core."""
    mean_comp = spec.get_component(Role.MEAN)
    vol = spec.get_component(Role.VOLATILITY)
    dens = spec.get_component(Role.DENSITY)
    assert isinstance(dens, (GeneralizedHyperbolic, Hyperbolic, NormalInverseGaussian, VarianceGamma))

    y = np.ascontiguousarray(data, dtype=np.float64)
    resid = np.empty_like(y)
    sigma2 = np.empty_like(y)
    initial_variance = max(
        float(np.mean(y * y)), np.finfo(np.float64).tiny
    )
    variance_gamma = isinstance(dens, VarianceGamma)
    fixed_lambda = fixed_gh_lambda(dens.signature)
    estimate_lambda = fixed_lambda is None and not variance_gamma
    native_fixed_lambda = (
        np.inf if variance_gamma
        else np.nan if fixed_lambda is None
        else fixed_lambda
    )
    params_ptr = _as_cptr(theta)
    y_ptr = _as_cptr(y)
    resid_ptr = _as_cptr(resid)
    sigma2_ptr = _as_cptr(sigma2)

    if isinstance(mean_comp, ARMA):
        p_ar, q_ma = mean_comp.p, mean_comp.q
        p_vol = 0 if vol is None else vol.p
        q_vol = 0 if vol is None else vol.q
        volatility_kind = 2 if isinstance(vol, EGARCH) else 1 if isinstance(vol, GJRGARCH) else 0
        max_lag = max(p_ar, q_ma, p_vol, q_vol, 1)
        e0 = np.zeros(max_lag, dtype=np.float64)
        h0 = np.full(max_lag, initial_variance, dtype=np.float64)
        e0_ptr, h0_ptr = _as_cptr(e0), _as_cptr(h0)

        def objective() -> float:
            return float(_core._arma_vol_gh_nll_pq(
                params_ptr, y_ptr, resid_ptr, sigma2_ptr, e0_ptr, h0_ptr,
                y.size, p_ar, q_ma, p_vol, q_vol, volatility_kind,
                native_fixed_lambda,
            ))

        def gradient(buffer: NDArray[np.float64]) -> None:
            _core._arma_vol_gh_nll_grad_pq(
                params_ptr, y_ptr, resid_ptr, sigma2_ptr, e0_ptr, h0_ptr,
                _as_cptr(buffer), y.size, p_ar, q_ma, p_vol, q_vol,
                volatility_kind, native_fixed_lambda,
            )

        def hessian(buffer: NDArray[np.float64]) -> None:
            _core._arma_vol_gh_nll_hess_pq(
                params_ptr, y_ptr, resid_ptr, sigma2_ptr, e0_ptr, h0_ptr,
                _as_cptr(buffer), y.size, p_ar, q_ma, p_vol, q_vol,
                volatility_kind, native_fixed_lambda,
            )

    elif isinstance(mean_comp, (ARX, HARX)):
        if x is not None:
            mean_comp.set_n_exog(int(x.shape[1]))
        features = build_linear_mean_features(mean_comp, y, x)
        features_ptr = _as_cptr(features)
        n_mean = mean_comp.n_params
        p = 0 if vol is None else vol.p
        q = 0 if vol is None else vol.q
        if isinstance(vol, EGARCH):
            core_nll = _core._linear_mean_egarch_nll_pq_gh
            core_grad = _core._linear_mean_egarch_nll_grad_pq_gh
            core_hess = _core._linear_mean_egarch_hess_pq_gh
        elif isinstance(vol, GJRGARCH):
            core_nll = _core._linear_mean_gjr_garch_nll_pq_gh
            core_grad = _core._linear_mean_gjr_garch_nll_grad_pq_gh
            core_hess = _core._linear_mean_gjr_garch_hess_pq_gh
        else:
            core_nll = _core._linear_mean_garch_nll_pq_gh
            core_grad = _core._linear_mean_garch_nll_grad_pq_gh
            core_hess = _core._linear_mean_garch_hess_pq_gh

        def reset_variance() -> None:
            sigma2[0] = float(theta[n_mean]) if vol is None else initial_variance

        def objective() -> float:
            reset_variance()
            return float(core_nll(
                params_ptr, y_ptr, features_ptr, resid_ptr, sigma2_ptr,
                y.size, n_mean, p, q, native_fixed_lambda,
            ))

        def gradient(buffer: NDArray[np.float64]) -> None:
            reset_variance()
            core_grad(
                params_ptr, y_ptr, features_ptr, resid_ptr, sigma2_ptr,
                _as_cptr(buffer), y.size, n_mean, p, q, native_fixed_lambda,
            )

        def hessian(buffer: NDArray[np.float64]) -> None:
            reset_variance()
            core_hess(
                params_ptr, y_ptr, features_ptr, resid_ptr, sigma2_ptr,
                _as_cptr(buffer), y.size, n_mean, p, q, native_fixed_lambda,
            )

    else:
        assert isinstance(vol, (GARCH, GJRGARCH, EGARCH))
        resid[:] = y
        p, q = vol.p, vol.q
        if isinstance(vol, EGARCH):
            core_nll = _core._egarch_gh_nll_pq
            core_grad = _core._egarch_gh_nll_grad_pq
            core_hess = _core._egarch_gh_nll_hess_pq
        elif isinstance(vol, GJRGARCH):
            core_nll = _core._gjr_garch_gh_nll_pq
            core_grad = _core._gjr_garch_gh_nll_grad_pq
            core_hess = _core._gjr_garch_gh_nll_hess_pq
        else:
            core_nll = _core._garch_gh_nll_pq
            core_grad = _core._garch_gh_nll_grad_pq
            core_hess = _core._garch_gh_nll_hess_pq

        def reset_variance() -> None:
            sigma2[0] = initial_variance

        def objective() -> float:
            reset_variance()
            return float(core_nll(
                params_ptr, resid_ptr, sigma2_ptr, y.size, p, q,
                native_fixed_lambda,
            ))

        def gradient(buffer: NDArray[np.float64]) -> None:
            reset_variance()
            core_grad(
                params_ptr, resid_ptr, sigma2_ptr, _as_cptr(buffer),
                y.size, p, q, native_fixed_lambda,
            )

        def hessian(buffer: NDArray[np.float64]) -> None:
            reset_variance()
            core_hess(
                params_ptr, resid_ptr, sigma2_ptr, _as_cptr(buffer),
                y.size, p, q, native_fixed_lambda,
            )

    nll = objective()
    arma_scale = 1.0
    if isinstance(mean_comp, ARMA):
        arma_scale = float(
            y.size - max(mean_comp.p, mean_comp.q, p_vol, q_vol, 1)
        )
        nll *= arma_scale
    score = None
    if compute_score:
        grad = np.empty(theta.size, dtype=np.float64)
        gradient(grad)
        score = -(arma_scale * grad)
    observed_hessian = None
    if compute_hessian:
        hess = np.empty((theta.size, theta.size), dtype=np.float64)
        hessian(hess)
        observed_hessian = -(arma_scale * hess)

    mean = y - resid
    rho = float(theta[-3] if estimate_lambda else theta[-2])
    zeta = None if variance_gamma else float(theta[-2] if estimate_lambda else theta[-1])
    shape_lambda = (
        float(theta[-1])
        if estimate_lambda or variance_gamma
        else float(fixed_lambda)
    )
    return FilteredState(
        y,
        mean,
        resid,
        sigma2,
        -nll,
        score,
        observed_hessian,
        dens.signature,
        rho=rho,
        zeta=zeta,
        shape_lambda=shape_lambda,
    )


def filter_spec(
    spec_like: Any,
    data: Any,
    params: Any,
    *,
    x: Any = None,
    compute_score: bool = False,
    compute_hessian: bool = False,
) -> FilteredState:
    spec = validate_spec(spec_like)
    data_np = validate_data(np.asarray(data, dtype=np.float64))
    theta = _coerce_params(spec, params)
    _apply_theta_to_spec(spec, theta)
    if x is None:
        x_np = None
    else:
        x_arr = np.asarray(x, dtype=np.float64)
        if x_arr.ndim == 1:
            x_arr = x_arr[:, None]
        x_np = np.ascontiguousarray(x_arr, dtype=np.float64)

    mean_comp = spec.get_component(Role.MEAN)
    vol = spec.get_component(Role.VOLATILITY)
    dens = spec.get_component(Role.DENSITY)

    if isinstance(vol, APARCH):
        if mean_comp is not None and not isinstance(mean_comp, ARMA):
            raise NotImplementedError(
                "APARCH filtering currently supports zero mean or a genuine "
                "joint ARMA mean."
            )
        return _filter_aparch(
            spec,
            data_np,
            theta,
            compute_score=compute_score,
            compute_hessian=compute_hessian,
        )

    # Reuse the routine registry as the support matrix for historical kernels.
    from ._kernels import get_routine

    get_routine(str(spec))

    if mean_comp is None and isinstance(vol, EGARCH):
        return _filter_egarch(
            spec,
            data_np,
            theta,
            compute_score=compute_score,
            compute_hessian=compute_hessian,
        )

    if isinstance(mean_comp, ARMA) and isinstance(vol, EGARCH):
        return _filter_arma_egarch(
            spec,
            data_np,
            theta,
            compute_score=compute_score,
            compute_hessian=compute_hessian,
        )

    if isinstance(dens, (GeneralizedHyperbolic, Hyperbolic, NormalInverseGaussian, VarianceGamma)):
        return _filter_gh(
            spec,
            data_np,
            theta,
            x=x_np,
            compute_score=compute_score,
            compute_hessian=compute_hessian,
        )

    if isinstance(mean_comp, (ARX, HARX)) and vol is None and isinstance(dens, Normal):
        return _filter_meanx_normal(
            spec,
            data_np,
            theta,
            x=x_np,
            compute_score=compute_score,
            compute_hessian=compute_hessian,
        )

    if isinstance(mean_comp, (ARX, HARX)) and vol is None and isinstance(dens, (StudentT, SkewT, GED)):
        return _filter_meanx_standalone(
            spec,
            data_np,
            theta,
            x=x_np,
            compute_score=compute_score,
            compute_hessian=compute_hessian,
        )

    if isinstance(mean_comp, (ARX, HARX)) and isinstance(vol, GARCH) and isinstance(dens, (Normal, StudentT, SkewT, GED)):
        return _filter_meanx_garch(
            spec,
            data_np,
            theta,
            x=x_np,
            compute_score=compute_score,
            compute_hessian=compute_hessian,
        )

    if isinstance(mean_comp, (ARX, HARX)) and isinstance(vol, GJRGARCH) and isinstance(dens, (Normal, StudentT, SkewT, GED)):
        return _filter_meanx_gjr(
            spec,
            data_np,
            theta,
            x=x_np,
            compute_score=compute_score,
            compute_hessian=compute_hessian,
        )

    if isinstance(mean_comp, (ARX, HARX)) and isinstance(vol, EGARCH) and isinstance(dens, (Normal, StudentT, SkewT, GED)):
        return _filter_meanx_egarch(
            spec,
            data_np,
            theta,
            x=x_np,
            compute_score=compute_score,
            compute_hessian=compute_hessian,
        )

    if isinstance(mean_comp, (ARX, HARX)):
        return _filter_generic(spec, data_np, theta, x=x_np, compute_score=compute_score, compute_hessian=compute_hessian)

    if isinstance(mean_comp, ARMA) and vol is None and isinstance(dens, GED):
        return _filter_arma_ged(spec, data_np, theta, compute_score=compute_score, compute_hessian=compute_hessian)
    if isinstance(mean_comp, ARMA) and isinstance(vol, GARCH):
        return _filter_arma_garch(spec, data_np, theta, compute_score=compute_score, compute_hessian=compute_hessian)
    if isinstance(mean_comp, ARMA) and isinstance(vol, GJRGARCH):
        return _filter_arma_gjr(spec, data_np, theta, compute_score=compute_score, compute_hessian=compute_hessian)
    if isinstance(dens, GED) and vol is None:
        return _filter_generic(spec, data_np, theta, x=x_np, compute_score=compute_score, compute_hessian=compute_hessian)
    if isinstance(mean_comp, ARMA) and vol is None:
        return _filter_arma(spec, data_np, theta, compute_score=compute_score, compute_hessian=compute_hessian)
    if isinstance(vol, GJRGARCH):
        return _filter_gjr(spec, data_np, theta, compute_score=compute_score, compute_hessian=compute_hessian)
    if isinstance(vol, GARCH):
        return _filter_garch(spec, data_np, theta, compute_score=compute_score, compute_hessian=compute_hessian)

    raise NotImplementedError(f"Filtering is not implemented for spec '{spec}'.")


def loglikelihood_spec(spec_like: Any, data: Any, params: Any, *, x: Any = None) -> float:
    return filter_spec(spec_like, data, params, x=x).log_likelihood


def score_spec(spec_like: Any, data: Any, params: Any, *, x: Any = None) -> NDArray[np.float64]:
    state = filter_spec(spec_like, data, params, x=x, compute_score=True)
    if state.score is None:
        raise RuntimeError("Score computation did not produce a gradient.")
    return state.score


def hessian_spec(spec_like: Any, data: Any, params: Any, *, x: Any = None) -> NDArray[np.float64]:
    state = filter_spec(spec_like, data, params, x=x, compute_hessian=True)
    if state.hessian is None:
        raise RuntimeError("Hessian computation did not produce a matrix.")
    return state.hessian


def _arma_mean_forecast(
    data: NDArray[np.float64],
    residuals: NDArray[np.float64],
    c: float,
    phi: NDArray[np.float64],
    theta: NDArray[np.float64],
    horizon: int,
) -> NDArray[np.float64]:
    forecasts = np.empty(horizon, dtype=np.float64)
    p = phi.size
    q = theta.size

    y_future: list[float] = []
    for h in range(horizon):
        mu = c
        for i in range(p):
            idx = h - 1 - i
            y_lag = y_future[idx] if idx >= 0 else float(data[idx])
            mu += float(phi[i]) * y_lag
        for j in range(q):
            idx = h - 1 - j
            e_lag = 0.0 if idx >= 0 else float(residuals[idx])
            mu += float(theta[j]) * e_lag
        forecasts[h] = mu
        y_future.append(float(mu))
    return forecasts


def _arma_psi_weights(phi: NDArray[np.float64], theta: NDArray[np.float64], horizon: int) -> NDArray[np.float64]:
    psi = np.zeros(horizon, dtype=np.float64)
    psi[0] = 1.0
    p = phi.size
    q = theta.size
    for k in range(1, horizon):
        theta_k = float(theta[k - 1]) if k <= q else 0.0
        total = theta_k
        for i in range(1, min(p, k) + 1):
            total += float(phi[i - 1]) * float(psi[k - i])
        psi[k] = total
    return psi


def _arx_mean_forecast(
    component: ARX,
    data: NDArray[np.float64],
    future_x: NDArray[np.float64],
    horizon: int,
) -> NDArray[np.float64]:
    params = component.fitted_params or {}
    const = float(params.get("const", 0.0))
    ar = np.asarray(params.get("ar", []), dtype=np.float64)
    exog = np.asarray(params.get("exog", []), dtype=np.float64)
    forecasts = np.empty(horizon, dtype=np.float64)
    y_future: list[float] = []

    for h in range(horizon):
        mu = const
        for lag, coef in enumerate(ar, start=1):
            idx = h - lag
            y_lag = y_future[idx] if idx >= 0 else float(data[idx])
            mu += float(coef) * y_lag
        if exog.size:
            mu += float(future_x[h] @ exog)
        forecasts[h] = mu
        y_future.append(float(mu))

    return forecasts


def _harx_mean_forecast(
    component: HARX,
    data: NDArray[np.float64],
    future_x: NDArray[np.float64],
    horizon: int,
) -> NDArray[np.float64]:
    params = component.fitted_params or {}
    const = float(params.get("const", 0.0))
    har = np.asarray(params.get("har", []), dtype=np.float64)
    exog = np.asarray(params.get("exog", []), dtype=np.float64)
    forecasts = np.empty(horizon, dtype=np.float64)
    history = list(np.asarray(data, dtype=np.float64))

    for h in range(horizon):
        mu = const
        for horizon_lag, coef in zip(component.horizons, har):
            window = history[-horizon_lag:]
            avg = float(np.mean(window)) if window else 0.0
            mu += float(coef) * avg
        if exog.size:
            mu += float(future_x[h] @ exog)
        forecasts[h] = mu
        history.append(float(mu))

    return forecasts


def _variance_forecast(
    vol: APARCH | EGARCH | GARCH | GJRGARCH,
    residuals: NDArray[np.float64],
    sigma2: NDArray[np.float64],
    horizon: int,
    *,
    distribution: str,
    nu: Optional[float],
    lam: Optional[float],
    rho: Optional[float],
    zeta: Optional[float],
    shape_lambda: Optional[float],
    negative_second_moment: float,
) -> NDArray[np.float64]:
    if isinstance(vol, APARCH):
        omega = float(vol.fitted_params["omega"])  # type: ignore[index]
        alpha = np.asarray(vol.fitted_params["alpha"], dtype=np.float64)  # type: ignore[index]
        gamma = np.asarray(vol.fitted_params["gamma"], dtype=np.float64)  # type: ignore[index]
        beta = np.asarray(vol.fitted_params["beta"], dtype=np.float64)  # type: ignore[index]
        delta = float(vol.fitted_params["delta"])  # type: ignore[index]
        p, q = alpha.size, beta.size
        n_paths = 4000
        rng = np.random.default_rng(0)
        draws = _draw_standardized_innovations(
            rng,
            n_paths * max(horizon - 1, 0),
            distribution,
            nu,
            lam,
            rho,
            zeta,
            shape_lambda,
        ).reshape(n_paths, max(horizon - 1, 0))
        path_residuals = (
            np.tile(np.asarray(residuals[-p:], dtype=np.float64), (n_paths, 1))
            if p else np.zeros((n_paths, 0), dtype=np.float64)
        )
        path_powers = (
            np.tile(np.asarray(sigma2[-q:], dtype=np.float64) ** (0.5 * delta), (n_paths, 1))
            if q else np.zeros((n_paths, 0), dtype=np.float64)
        )
        forecasts = np.empty(horizon, dtype=np.float64)

        for step in range(horizon):
            power = np.full(n_paths, omega, dtype=np.float64)
            for lag in range(p):
                residual_lag = path_residuals[:, -1 - lag]
                shock = np.abs(residual_lag) - float(gamma[lag]) * residual_lag
                power += float(alpha[lag]) * shock ** delta
            for lag in range(q):
                power += float(beta[lag]) * path_powers[:, -1 - lag]
            path_variance = np.maximum(power, 1e-16) ** (2.0 / delta)
            forecasts[step] = max(float(np.mean(path_variance)), 1e-12)
            if q:
                path_powers = np.roll(path_powers, -1, axis=1)
                path_powers[:, -1] = power
            if p and step < horizon - 1:
                path_residuals = np.roll(path_residuals, -1, axis=1)
                path_residuals[:, -1] = np.sqrt(path_variance) * draws[:, step]
        return forecasts

    if isinstance(vol, EGARCH):
        omega = float(vol.fitted_params["omega"])  # type: ignore[index]
        alpha = np.asarray(vol.fitted_params["alpha"], dtype=np.float64)  # type: ignore[index]
        gamma = np.asarray(vol.fitted_params["gamma"], dtype=np.float64)  # type: ignore[index]
        beta = np.asarray(vol.fitted_params["beta"], dtype=np.float64)  # type: ignore[index]
        p = alpha.size
        q = beta.size
        abs_moment = _egarch_abs_moment(
            distribution, nu, lam, rho, zeta, shape_lambda
        )

        forecasts = np.empty(horizon, dtype=np.float64)
        z_hist = [
            float(residuals[-p + i]) / np.sqrt(max(float(sigma2[-p + i]), 1e-12))
            for i in range(p)
        ] if p else []
        x_hist = [float(np.log(max(v, 1e-12))) for v in sigma2[-q:]] if q else []

        def _egarch_forecast_step(z_lags: list[float], x_lags: list[float]) -> float:
            value = omega
            for i in range(p):
                z_i = z_lags[-1 - i]
                value += float(alpha[i]) * (abs(z_i) - abs_moment) + float(gamma[i]) * z_i
            for j in range(q):
                value += float(beta[j]) * x_lags[-1 - j]
            return float(value)

        x1 = _egarch_forecast_step(z_hist, x_hist)
        forecasts[0] = max(float(np.exp(x1)), 1e-12)
        if horizon == 1:
            return forecasts

        rng = np.random.default_rng(0)
        n_paths = 4000
        path_x_history = (
            np.tile(np.array((x_hist + [x1])[-q:], dtype=np.float64), (n_paths, 1))
            if q
            else np.zeros((n_paths, 0), dtype=np.float64)
        )
        path_z_history = (
            np.tile(np.array(z_hist, dtype=np.float64), (n_paths, 1))
            if p
            else np.zeros((n_paths, 0), dtype=np.float64)
        )
        draws = _draw_standardized_innovations(
            rng,
            n_paths * (horizon - 1),
            distribution,
            nu,
            lam,
            rho,
            zeta,
            shape_lambda,
        ).reshape(n_paths, horizon - 1)
        for h in range(1, horizon):
            path_x = np.full(n_paths, omega, dtype=np.float64)
            for i in range(p):
                z_prev = path_z_history[:, -1 - i]
                path_x += float(alpha[i]) * (np.abs(z_prev) - abs_moment) + float(gamma[i]) * z_prev
            for j in range(q):
                path_x += float(beta[j]) * path_x_history[:, -1 - j]
            forecasts[h] = max(float(np.mean(np.exp(path_x))), 1e-12)
            if q:
                path_x_history = np.roll(path_x_history, -1, axis=1)
                path_x_history[:, -1] = path_x
            if p:
                path_z_history = np.roll(path_z_history, -1, axis=1)
                path_z_history[:, -1] = draws[:, h - 1]
        return forecasts

    omega = float(vol.fitted_params["omega"])  # type: ignore[index]
    alpha = np.asarray(vol.fitted_params["alpha"], dtype=np.float64)  # type: ignore[index]
    beta = np.asarray(vol.fitted_params["beta"], dtype=np.float64)  # type: ignore[index]
    gamma = (
        np.asarray(vol.fitted_params["gamma"], dtype=np.float64)  # type: ignore[index]
        if isinstance(vol, GJRGARCH)
        else None
    )

    forecasts = np.empty(horizon, dtype=np.float64)
    p = alpha.size
    q = beta.size

    for h in range(horizon):
        value = omega
        for i in range(p):
            idx = h - 1 - i
            if idx >= 0:
                eps2 = forecasts[idx]
                leverage = (
                    negative_second_moment * forecasts[idx]
                    if gamma is not None
                    else 0.0
                )
            else:
                resid_lag = float(residuals[idx])
                eps2 = resid_lag * resid_lag
                leverage = (gamma[i] * eps2 if gamma is not None and resid_lag < 0.0 else 0.0)
            value += float(alpha[i]) * eps2
            if gamma is not None and idx >= 0:
                value += float(gamma[i]) * leverage
            elif gamma is not None and idx < 0:
                value += float(leverage)
        for j in range(q):
            idx = h - 1 - j
            value += float(beta[j]) * (forecasts[idx] if idx >= 0 else float(sigma2[idx]))
        forecasts[h] = max(value, 1e-12)
    return forecasts


def forecast_from_state(
    spec_like: Any,
    state: FilteredState,
    horizon: int,
    *,
    x: Any = None,
) -> ForecastResult:
    if horizon <= 0:
        raise ValueError("horizon must be positive")

    spec = validate_spec(spec_like)
    mean_comp = spec.get_component(Role.MEAN)
    vol = spec.get_component(Role.VOLATILITY)

    if vol is None and mean_comp is None:
        raise ValueError("Forecasting requires either a mean or a volatility component.")

    if vol is None:
        residual_variance = np.full(horizon, float(state.sigma2[-1]), dtype=np.float64)
    else:
        negative_second_moment = negative_half_second_moment(
            state.distribution,
            state.nu,
            state.lam,
            state.rho,
            state.zeta,
            state.shape_lambda,
        )
        residual_variance = _variance_forecast(
            vol,
            state.residuals,
            state.sigma2,
            horizon,
            distribution=state.distribution,
            nu=state.nu,
            lam=state.lam,
            rho=state.rho,
            zeta=state.zeta,
            shape_lambda=state.shape_lambda,
            negative_second_moment=negative_second_moment,
        )

    if isinstance(mean_comp, ARMA):
        arma_params = mean_comp.fitted_params or {}
        c = float(arma_params.get("const", arma_params.get("c", 0.0)))
        phi = np.asarray(arma_params.get("ar", arma_params.get("phi", [])), dtype=np.float64)
        theta = np.asarray(arma_params.get("ma", arma_params.get("theta", [])), dtype=np.float64)
        mean = _arma_mean_forecast(state.data, state.residuals, c, phi, theta, horizon)
        psi = _arma_psi_weights(phi, theta, horizon)
        variance = np.empty(horizon, dtype=np.float64)
        for h in range(horizon):
            total = 0.0
            for k in range(h + 1):
                total += float(psi[k] ** 2) * float(residual_variance[h - k])
            variance[h] = max(total, 1e-12)
    elif isinstance(mean_comp, ARX):
        expected = getattr(mean_comp, "n_exog", 0)
        future_x = np.zeros((horizon, 0), dtype=np.float64) if expected == 0 else np.asarray(x, dtype=np.float64)
        if expected > 0:
            if future_x.ndim == 1:
                future_x = future_x[:, None]
            if future_x.shape != (horizon, expected):
                raise ValueError(f"ARX forecast requires x with shape ({horizon}, {expected}).")
        mean = _arx_mean_forecast(mean_comp, state.data, np.ascontiguousarray(future_x, dtype=np.float64), horizon)
        variance = residual_variance.copy()
    elif isinstance(mean_comp, HARX):
        expected = getattr(mean_comp, "n_exog", 0)
        future_x = np.zeros((horizon, 0), dtype=np.float64) if expected == 0 else np.asarray(x, dtype=np.float64)
        if expected > 0:
            if future_x.ndim == 1:
                future_x = future_x[:, None]
            if future_x.shape != (horizon, expected):
                raise ValueError(f"HARX forecast requires x with shape ({horizon}, {expected}).")
        mean = _harx_mean_forecast(mean_comp, state.data, np.ascontiguousarray(future_x, dtype=np.float64), horizon)
        variance = residual_variance.copy()
    else:
        mean = np.zeros(horizon, dtype=np.float64)
        variance = residual_variance.copy()

    return ForecastResult(
        mean=mean,
        variance=variance,
        residual_variance=residual_variance,
        distribution=state.distribution,
        nu=state.nu,
        lam=state.lam,
        rho=state.rho,
        zeta=state.zeta,
        shape_lambda=state.shape_lambda,
    )


def fixed_result_from_spec(
    spec_like: Any,
    data: NDArray[np.float64],
    params: Any,
    *,
    x: Any = None,
) -> FixedResult:
    spec = validate_spec(spec_like)
    theta = _coerce_params(spec, params)
    state = filter_spec(spec, data, theta, x=x)
    fixed_spec = _clone_spec_with_params(spec, theta)
    opt = _FixedOptResult(theta.copy(), -state.log_likelihood)
    result = FixedResult(
        fixed_spec,
        opt,
        state.data,
        sigma2=state.sigma2.copy(),
        hessian=None if state.hessian is None else -state.hessian.copy(),
        fit_info=None,
        method="FIXED",
        parameter_source="supplied",
    )
    result._resid = state.residuals.copy()
    result._filtered_state_cache = state
    result._fit_x = None if x is None else np.ascontiguousarray(np.asarray(x, dtype=np.float64), dtype=np.float64)
    return result


def _draw_standardized_innovations(
    rng: np.random.Generator,
    n_obs: int,
    distribution: str,
    nu: Optional[float],
    lam: Optional[float],
    rho: Optional[float] = None,
    zeta: Optional[float] = None,
    shape_lambda: Optional[float] = None,
) -> NDArray[np.float64]:
    if distribution == "Normal":
        return rng.standard_normal(n_obs)
    if distribution == "StudentT":
        if nu is None:
            raise ValueError("StudentT simulation requires `nu`.")
        return rng.standard_t(nu, size=n_obs) * np.sqrt((nu - 2.0) / nu)
    if distribution == "SkewT":
        if nu is None or lam is None:
            raise ValueError("SkewT simulation requires `nu` and `lam`.")
        return _hansen_skewt_ppf(rng.uniform(size=n_obs), nu, lam)
    if distribution == "GED":
        if nu is None:
            raise ValueError("GED simulation requires `nu`.")
        return np.asarray(gennorm.rvs(beta=nu, scale=_ged_scale(nu), size=n_obs, random_state=rng), dtype=np.float64)
    if distribution in {"GeneralizedHyperbolic", "Hyperbolic", "NormalInverseGaussian"}:
        if rho is None or zeta is None or shape_lambda is None:
            raise ValueError(f"{distribution} simulation requires GH shape parameters.")
        return standardized_gh_rvs(rng, n_obs, rho, zeta, shape_lambda)
    if distribution == "VarianceGamma":
        if rho is None or shape_lambda is None:
            raise ValueError("VarianceGamma simulation requires `rho` and `lambda`.")
        return standardized_vg_rvs(rng, n_obs, rho, shape_lambda)
    raise ValueError(f"Unsupported distribution '{distribution}'.")


def simulate_spec(
    spec_like: Any,
    n_obs: int,
    params: Any,
    *,
    burn: int = 500,
    seed: Optional[int] = None,
    x: Any = None,
) -> SimulationResult:
    spec = validate_spec(spec_like)
    if n_obs <= 0:
        raise ValueError("n_obs must be positive")

    mean_for_dims = spec.get_component(Role.MEAN)
    if isinstance(mean_for_dims, (ARX, HARX)) and x is not None:
        x_arr = np.asarray(x, dtype=np.float64)
        if x_arr.ndim == 1:
            x_arr = x_arr[:, None]
        mean_for_dims.set_n_exog(x_arr.shape[1])

    theta = _coerce_params(spec, params)
    sim_spec = _clone_spec_with_params(spec, theta)

    mean_comp = sim_spec.get_component(Role.MEAN)
    vol = sim_spec.get_component(Role.VOLATILITY)
    dens = sim_spec.get_component(Role.DENSITY)
    distribution = dens.signature if dens is not None else "Normal"
    nu = None if dens is None else dens.fitted_params.get("nu")  # type: ignore[union-attr]
    lam = None if dens is None else dens.fitted_params.get("lam")  # type: ignore[union-attr]
    rho = None if dens is None else dens.fitted_params.get("rho")  # type: ignore[union-attr]
    zeta = None if dens is None else dens.fitted_params.get("zeta")  # type: ignore[union-attr]
    shape_lambda = None if dens is None else fixed_gh_lambda(dens.signature)
    if shape_lambda is None and dens is not None:
        shape_lambda = dens.fitted_params.get("lambda")  # type: ignore[union-attr]

    explicit_constant_sigma2 = None
    if _has_explicit_arma_density_variance(spec):
        assert isinstance(mean_comp, ARMA)
        explicit_constant_sigma2 = float(theta[1 + mean_comp.p + mean_comp.q])
    elif _has_explicit_meanx_variance(spec):
        assert isinstance(mean_comp, (ARX, HARX))
        explicit_constant_sigma2 = float(theta[mean_comp.n_params])

    if isinstance(mean_comp, ARMA) and vol is None and explicit_constant_sigma2 is None:
        raise NotImplementedError("Standalone ARMA simulation is not yet exposed because the concentrated innovation variance is not explicit.")

    total = n_obs + max(burn, 0)
    rng = np.random.default_rng(seed)
    innovations = _draw_standardized_innovations(
        rng, total, distribution, nu, lam, rho, zeta, shape_lambda
    )

    data = np.zeros(total, dtype=np.float64)
    mean = np.zeros(total, dtype=np.float64)
    resid = np.zeros(total, dtype=np.float64)
    sigma2 = np.ones(total, dtype=np.float64)

    phi = np.zeros(0, dtype=np.float64)
    theta_ma = np.zeros(0, dtype=np.float64)
    arx = np.zeros(0, dtype=np.float64)
    har = np.zeros(0, dtype=np.float64)
    xbeta = np.zeros(0, dtype=np.float64)
    c = 0.0
    if isinstance(mean_comp, ARMA):
        arma_params = mean_comp.fitted_params or {}
        c = float(arma_params.get("const", 0.0))
        phi = np.asarray(arma_params.get("ar", []), dtype=np.float64)
        theta_ma = np.asarray(arma_params.get("ma", []), dtype=np.float64)
    elif isinstance(mean_comp, ARX):
        mean_params = mean_comp.fitted_params or {}
        c = float(mean_params.get("const", 0.0))
        arx = np.asarray(mean_params.get("ar", []), dtype=np.float64)
        xbeta = np.asarray(mean_params.get("exog", []), dtype=np.float64)
    elif isinstance(mean_comp, HARX):
        mean_params = mean_comp.fitted_params or {}
        c = float(mean_params.get("const", 0.0))
        har = np.asarray(mean_params.get("har", []), dtype=np.float64)
        xbeta = np.asarray(mean_params.get("exog", []), dtype=np.float64)

    unconditional_mean = c / (1.0 - phi.sum()) if phi.size and abs(phi.sum()) < 1.0 else c
    total_x = np.zeros((total, 0), dtype=np.float64)
    if isinstance(mean_comp, (ARX, HARX)):
        expected = getattr(mean_comp, "n_exog", xbeta.size)
        if expected != xbeta.size:
            expected = xbeta.size
        if expected > 0:
            if x is None:
                raise ValueError(f"{mean_comp.signature} simulation requires x with shape ({total}, {expected}).")
            total_x = np.asarray(x, dtype=np.float64)
            if total_x.ndim == 1:
                total_x = total_x[:, None]
            if total_x.shape != (total, expected):
                raise ValueError(f"{mean_comp.signature} simulation requires x with shape ({total}, {expected}).")
            total_x = np.ascontiguousarray(total_x, dtype=np.float64)

    delta = 2.0
    aparch_power = np.ones(total, dtype=np.float64)
    if isinstance(vol, APARCH):
        from ._native_fit import _aparch_volatility_domain, _describe_model

        omega = float(vol.fitted_params["omega"])  # type: ignore[index]
        alpha = np.asarray(vol.fitted_params["alpha"], dtype=np.float64)  # type: ignore[index]
        beta = np.asarray(vol.fitted_params["beta"], dtype=np.float64)  # type: ignore[index]
        gamma = np.asarray(vol.fitted_params["gamma"], dtype=np.float64)  # type: ignore[index]
        delta = float(vol.fitted_params["delta"])  # type: ignore[index]
        violation, feasible, persistence = _aparch_volatility_domain(
            _describe_model(sim_spec), theta
        )
        if not feasible:
            raise ValueError(
                "APARCH simulation requires parameters in the exact stationary "
                f"domain; maximum violation={violation:.3e}."
            )
    elif isinstance(vol, EGARCH):
        omega = float(vol.fitted_params["omega"])  # type: ignore[index]
        alpha = np.asarray(vol.fitted_params["alpha"], dtype=np.float64)  # type: ignore[index]
        beta = np.asarray(vol.fitted_params["beta"], dtype=np.float64)  # type: ignore[index]
        gamma = np.asarray(vol.fitted_params["gamma"], dtype=np.float64)  # type: ignore[index]
        persistence = float(beta.sum())
    elif isinstance(vol, GARCH):
        omega = float(vol.fitted_params["omega"])  # type: ignore[index]
        alpha = np.asarray(vol.fitted_params["alpha"], dtype=np.float64)  # type: ignore[index]
        beta = np.asarray(vol.fitted_params["beta"], dtype=np.float64)  # type: ignore[index]
        gamma = None
        persistence = float(alpha.sum() + beta.sum())
    elif isinstance(vol, GJRGARCH):
        omega = float(vol.fitted_params["omega"])  # type: ignore[index]
        alpha = np.asarray(vol.fitted_params["alpha"], dtype=np.float64)  # type: ignore[index]
        beta = np.asarray(vol.fitted_params["beta"], dtype=np.float64)  # type: ignore[index]
        gamma = np.asarray(vol.fitted_params["gamma"], dtype=np.float64)  # type: ignore[index]
        negative_second_moment = negative_half_second_moment(
            distribution, nu, lam, rho, zeta, shape_lambda
        )
        persistence = float(
            alpha.sum() + beta.sum() + negative_second_moment * gamma.sum()
        )
    else:
        omega = explicit_constant_sigma2 if explicit_constant_sigma2 is not None else float(np.var(data[:1]) if total == 1 else 1.0)
        alpha = np.zeros(0, dtype=np.float64)
        beta = np.zeros(0, dtype=np.float64)
        gamma = None
        persistence = 0.0

    if isinstance(vol, APARCH):
        unconditional_power = omega / (1.0 - persistence)
        unconditional_variance = unconditional_power ** (2.0 / delta)
        aparch_power[:] = unconditional_power
    elif isinstance(vol, EGARCH):
        unconditional_variance = float(np.exp(omega / (1.0 - persistence))) if persistence < 1.0 else 1.0
    else:
        unconditional_variance = omega / (1.0 - persistence) if vol is not None and persistence < 1.0 else max(omega, 1e-6)
    egarch_abs = (
        _egarch_abs_moment(distribution, nu, lam, rho, zeta, shape_lambda)
        if isinstance(vol, EGARCH)
        else None
    )

    for t in range(total):
        if vol is None:
            sigma2[t] = unconditional_variance
        elif t == 0:
            sigma2[t] = (
                unconditional_variance
                if isinstance(vol, APARCH)
                else max(unconditional_variance, 1e-12)
            )
        elif isinstance(vol, APARCH):
            value = omega
            for i, alpha_i in enumerate(alpha, start=1):
                idx = t - i
                residual_lag = float(resid[idx]) if idx >= 0 else 0.0
                shock = abs(residual_lag) - float(gamma[i - 1]) * residual_lag
                value += float(alpha_i) * shock ** delta
            for j, beta_j in enumerate(beta, start=1):
                idx = t - j
                value += float(beta_j) * float(
                    aparch_power[idx] if idx >= 0 else unconditional_power
                )
            aparch_power[t] = value
            sigma2[t] = aparch_power[t] ** (2.0 / delta)
        elif isinstance(vol, EGARCH):
            value = omega
            for i, alpha_i in enumerate(alpha, start=1):
                idx = t - i
                if idx >= 0:
                    h_lag = max(float(sigma2[idx]), 1e-12)
                    z_lag = float(resid[idx]) / np.sqrt(h_lag)
                else:
                    z_lag = 0.0
                value += float(alpha_i) * (abs(z_lag) - float(egarch_abs))
                value += float(gamma[i - 1]) * z_lag
            for j, beta_j in enumerate(beta, start=1):
                idx = t - j
                x_lag = np.log(max(float(sigma2[idx]), 1e-12)) if idx >= 0 else np.log(max(unconditional_variance, 1e-12))
                value += float(beta_j) * float(x_lag)
            sigma2[t] = max(float(np.exp(value)), 1e-12)
        else:
            value = omega
            for i, a in enumerate(alpha, start=1):
                idx = t - i
                eps = resid[idx] if idx >= 0 else 0.0
                value += float(a) * float(eps * eps)
                if gamma is not None and eps < 0.0:
                    value += float(gamma[i - 1]) * float(eps * eps)
            for j, b in enumerate(beta, start=1):
                idx = t - j
                value += float(b) * float(sigma2[idx] if idx >= 0 else unconditional_variance)
            sigma2[t] = max(value, 1e-12)

        if t == 0:
            mu_t = unconditional_mean
        else:
            if isinstance(mean_comp, ARMA):
                mu_t = c
                for i, phi_i in enumerate(phi, start=1):
                    y_lag = data[t - i] if t - i >= 0 else unconditional_mean
                    mu_t += float(phi_i) * float(y_lag)
                for j, theta_j in enumerate(theta_ma, start=1):
                    e_lag = resid[t - j] if t - j >= 0 else 0.0
                    mu_t += float(theta_j) * float(e_lag)
            elif isinstance(mean_comp, ARX):
                mu_t = c
                for i, phi_i in enumerate(arx, start=1):
                    y_lag = data[t - i] if t - i >= 0 else 0.0
                    mu_t += float(phi_i) * float(y_lag)
                if xbeta.size:
                    mu_t += float(total_x[t] @ xbeta)
            elif isinstance(mean_comp, HARX):
                mu_t = c
                for horizon_lag, coef in zip(mean_comp.horizons, har):
                    window = data[max(0, t - horizon_lag):t]
                    avg = float(np.mean(window)) if window.size else 0.0
                    mu_t += float(coef) * avg
                if xbeta.size:
                    mu_t += float(total_x[t] @ xbeta)
            else:
                mu_t = c

        mean[t] = mu_t
        resid[t] = np.sqrt(sigma2[t]) * innovations[t]
        data[t] = mean[t] + resid[t]

    start = max(burn, 0)
    return SimulationResult(
        data=data[start:].copy(),
        mean=mean[start:].copy(),
        residuals=resid[start:].copy(),
        sigma2=sigma2[start:].copy(),
        innovations=innovations[start:].copy(),
    )
