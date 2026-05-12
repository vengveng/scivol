"""
scivol/dcc.py
=============
Gaussian multivariate correlation models.

Shipped surfaces:
  - CCC(): constant conditional correlation baseline / diagnostic model
  - DCC(p, q): dynamic conditional correlation model

Both models use the existing univariate result layer for marginal forecasting.
Dynamic DCC fitting stays on the shipped C likelihood / gradient / Hessian
kernels; the Python layer only orchestrates inputs, cached post-fit recursions,
and result surfaces.
"""

from __future__ import annotations

import time
from collections.abc import Iterator, Mapping
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Union

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import minimize

from . import _dcc_kernels as _dk
from .components import ARMA, ARX, GARCH, GJRGARCH, HARX, Normal
from .roles import Role

if TYPE_CHECKING:
    from ._evaluation import ForecastResult, FilteredState
    from .result import EstimationResult
    from .spec import CompositeSpec


def _normalize_correlation(matrix: NDArray[np.float64]) -> NDArray[np.float64]:
    diag = np.sqrt(np.maximum(np.diag(matrix), 1e-12))
    return matrix / np.outer(diag, diag)


def _extract_multivariate_input(
    returns: Any,
) -> Tuple[NDArray[np.float64], Optional[Any], Optional[List[Any]]]:
    index = None
    columns = None
    try:
        import pandas as pd

        if isinstance(returns, pd.DataFrame):
            index = returns.index
            columns = list(returns.columns)
    except ImportError:
        pass

    returns_np = np.asarray(returns, dtype=np.float64)
    if returns_np.ndim != 2:
        raise ValueError("returns must be 2-D (T, N)")
    return np.ascontiguousarray(returns_np, dtype=np.float64), index, columns


def _prepare_univariate_results(
    returns_np: NDArray[np.float64],
    univariate_spec: Optional["CompositeSpec"],
    univariate_results: Optional[List["EstimationResult"]],
) -> Tuple[NDArray[np.float64], List["EstimationResult"]]:
    n_series = returns_np.shape[1]

    if univariate_results is not None:
        if len(univariate_results) != n_series:
            raise ValueError(f"Expected {n_series} univariate results, got {len(univariate_results)}")
        lengths = {int(np.asarray(r.std_resid).shape[0]) for r in univariate_results}
        if len(lengths) != 1:
            raise ValueError("All univariate results must share a common sample length.")
        std_resid_cols = [np.asarray(r.std_resid, dtype=np.float64) for r in univariate_results]
        return np.ascontiguousarray(np.column_stack(std_resid_cols), dtype=np.float64), list(univariate_results)

    if univariate_spec is None:
        univariate_spec = GARCH(1, 1) + Normal()

    fitted: List["EstimationResult"] = []
    std_resid_cols: List[NDArray[np.float64]] = []
    for i in range(n_series):
        result = univariate_spec.fit(returns_np[:, i])
        fitted.append(result)
        std_resid_cols.append(np.asarray(result.std_resid, dtype=np.float64))
    return np.ascontiguousarray(np.column_stack(std_resid_cols), dtype=np.float64), fitted


def _resolve_per_series_inputs(
    x: Any,
    n_series: int,
    columns: Optional[List[Any]],
    *,
    label: str,
) -> List[Any]:
    if x is None:
        return [None] * n_series

    if isinstance(x, Mapping):
        resolved: List[Any] = []
        missing: List[str] = []
        for i in range(n_series):
            keys: List[Any] = []
            if columns is not None:
                keys.append(columns[i])
            keys.extend((i, str(i)))
            found = False
            for key in keys:
                if key in x:
                    resolved.append(x[key])
                    found = True
                    break
            if not found:
                missing.append(str(columns[i] if columns is not None else i))
        if missing:
            raise KeyError(f"{label} mapping is missing entries for series {missing}.")
        return resolved

    if isinstance(x, (list, tuple)):
        if len(x) != n_series:
            raise ValueError(f"{label} must contain one entry per series ({n_series}).")
        return list(x)

    raise TypeError(
        f"{label} must be None, a mapping keyed by series name/index, or a "
        f"sequence with one entry per series."
    )


def _stack_marginal_forecasts(
    results: List["EstimationResult"],
    horizon: int,
    x: Any,
    columns: Optional[List[Any]],
) -> Tuple[List["ForecastResult"], NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    per_series_x = _resolve_per_series_inputs(x, len(results), columns, label="x")
    forecasts = [result.forecast(horizon=horizon, x=per_series_x[i]) for i, result in enumerate(results)]
    mean = np.column_stack([np.asarray(fc.mean, dtype=np.float64) for fc in forecasts])
    variance = np.column_stack([np.asarray(fc.variance, dtype=np.float64) for fc in forecasts])
    residual_variance = np.column_stack([np.asarray(fc.residual_variance, dtype=np.float64) for fc in forecasts])
    return forecasts, mean, variance, residual_variance


def _residual_covariance_from_correlation(
    correlation: NDArray[np.float64],
    residual_variance: NDArray[np.float64],
) -> NDArray[np.float64]:
    std = np.sqrt(np.maximum(residual_variance, 1e-12))
    return correlation * std[:, :, None] * std[:, None, :]


def _constant_correlation_nll(
    eps: NDArray[np.float64],
    corr: NDArray[np.float64],
) -> float:
    chol = np.linalg.cholesky(corr)
    logdet = 2.0 * float(np.sum(np.log(np.diag(chol))))
    inv_corr = np.linalg.inv(corr)
    quad = np.einsum("ti,ij,tj->t", eps, inv_corr, eps, optimize=True)
    baseline = np.einsum("ti,ti->t", eps, eps, optimize=True)
    return float(0.5 * np.mean(logdet + quad - baseline))


@dataclass(slots=True)
class MultivariateForecastResult(Mapping[str, NDArray[np.float64]]):
    mean: NDArray[np.float64]
    variance: NDArray[np.float64]
    residual_variance: NDArray[np.float64]
    correlation: NDArray[np.float64]
    residual_covariance: NDArray[np.float64]
    marginals: Tuple["ForecastResult", ...] = ()
    columns: Optional[List[Any]] = None

    def __getitem__(self, key: str) -> NDArray[np.float64]:
        if key == "mean":
            return self.mean
        if key == "variance":
            return self.variance
        if key == "residual_variance":
            return self.residual_variance
        if key == "sigma2":
            return self.sigma2
        if key == "correlation":
            return self.correlation
        if key == "residual_covariance":
            return self.residual_covariance
        raise KeyError(key)

    def __iter__(self) -> Iterator[str]:
        return iter(
            ("mean", "variance", "residual_variance", "sigma2", "correlation", "residual_covariance")
        )

    def __len__(self) -> int:
        return 6

    @property
    def sigma2(self) -> NDArray[np.float64]:
        return self.residual_variance

    @property
    def volatility(self) -> NDArray[np.float64]:
        return np.sqrt(self.variance)

    @property
    def residual_volatility(self) -> NDArray[np.float64]:
        return np.sqrt(self.residual_variance)

    def _resolve_pair(self, series_i: Union[int, str], series_j: Union[int, str]) -> Tuple[int, int]:
        def _resolve(key: Union[int, str]) -> int:
            if isinstance(key, int):
                if key < 0 or key >= self.mean.shape[1]:
                    raise IndexError(f"Series index {key} out of range [0, {self.mean.shape[1]})")
                return key
            if self.columns is None:
                raise ValueError("Name-based access requires stored column names.")
            try:
                return self.columns.index(key)
            except ValueError as exc:
                raise KeyError(f"Series '{key}' not found. Available: {self.columns}") from exc

        return _resolve(series_i), _resolve(series_j)

    def corr(self, series_i: Union[int, str], series_j: Union[int, str]) -> NDArray[np.float64]:
        i, j = self._resolve_pair(series_i, series_j)
        return self.correlation[:, i, j]

    def residual_cov(self, series_i: Union[int, str], series_j: Union[int, str]) -> NDArray[np.float64]:
        i, j = self._resolve_pair(series_i, series_j)
        return self.residual_covariance[:, i, j]

    def to_dict(self) -> dict[str, Any]:
        return {
            "mean": self.mean.copy(),
            "variance": self.variance.copy(),
            "residual_variance": self.residual_variance.copy(),
            "sigma2": self.sigma2.copy(),
            "correlation": self.correlation.copy(),
            "residual_covariance": self.residual_covariance.copy(),
            "marginals": tuple(fc.to_dict() for fc in self.marginals),
        }


@dataclass(slots=True)
class MultivariateSimulationResult:
    data: NDArray[np.float64]
    mean: NDArray[np.float64]
    residuals: NDArray[np.float64]
    sigma2: NDArray[np.float64]
    innovations: NDArray[np.float64]
    correlation: NDArray[np.float64]
    residual_covariance: NDArray[np.float64]
    columns: Optional[List[Any]] = None

    @property
    def std_resid(self) -> NDArray[np.float64]:
        return self.innovations

    def to_dict(self) -> dict[str, NDArray[np.float64]]:
        return {
            "data": self.data.copy(),
            "mean": self.mean.copy(),
            "residuals": self.residuals.copy(),
            "sigma2": self.sigma2.copy(),
            "innovations": self.innovations.copy(),
            "correlation": self.correlation.copy(),
            "residual_covariance": self.residual_covariance.copy(),
        }


@dataclass
class DCCParams:
    """Container for DCC parameters."""

    a: NDArray[np.float64]
    b: NDArray[np.float64]

    @property
    def persistence(self) -> float:
        return float(np.sum(self.a) + np.sum(self.b))

    def __repr__(self) -> str:
        a_str = ", ".join(f"{x:.6f}" for x in self.a)
        b_str = ", ".join(f"{x:.6f}" for x in self.b)
        return f"DCCParams(a=[{a_str}], b=[{b_str}], persistence={self.persistence:.6f})"


@dataclass
class CCCParams:
    """Container for constant conditional correlation parameters."""

    corr: NDArray[np.float64]

    @property
    def avg_pairwise_corr(self) -> float:
        if self.corr.shape[0] <= 1:
            return 1.0
        upper = self.corr[np.triu_indices_from(self.corr, k=1)]
        return float(np.mean(upper))

    def __repr__(self) -> str:
        n_series = self.corr.shape[0]
        return f"CCCParams(corr=<{n_series}x{n_series} matrix>, avg_pairwise_corr={self.avg_pairwise_corr:.6f})"


def _dcc_recursion(
    theta: NDArray[np.float64],
    eps: NDArray[np.float64],
    Qbar: NDArray[np.float64],
    p: int,
    q: int,
    *,
    return_q: bool = False,
) -> Union[NDArray[np.float64], Tuple[NDArray[np.float64], NDArray[np.float64]]]:
    """Run the DCC recursion and return the correlation path (and optional Q path)."""
    t_count, n_series = eps.shape
    a_arr = theta[:p]
    b_arr = theta[p:]
    intercept = (1.0 - float(np.sum(theta))) * Qbar

    rt_out = np.empty((t_count, n_series, n_series), dtype=np.float64)
    qt_out = np.empty((t_count, n_series, n_series), dtype=np.float64) if return_q else None

    q_buf = [Qbar.copy() for _ in range(max(q, 1))]
    eps_outer_buf = [np.zeros((n_series, n_series), dtype=np.float64) for _ in range(max(p, 1))]

    for t in range(t_count):
        q_t = intercept.copy()
        for i in range(p):
            q_t = q_t + float(a_arr[i]) * eps_outer_buf[i]
        for j in range(q):
            q_t = q_t + float(b_arr[j]) * q_buf[j]

        rt_out[t] = _normalize_correlation(q_t)
        if qt_out is not None:
            qt_out[t] = q_t

        if p > 0:
            eps_outer_buf = [np.outer(eps[t], eps[t])] + eps_outer_buf[: p - 1]
        if q > 0:
            q_buf = [q_t.copy()] + q_buf[: q - 1]

    if qt_out is None:
        return rt_out
    return rt_out, qt_out


def _dcc_forecast_correlation_path(
    theta: NDArray[np.float64],
    eps: NDArray[np.float64],
    Qbar: NDArray[np.float64],
    qt_path: NDArray[np.float64],
    p: int,
    q: int,
    horizon: int,
) -> NDArray[np.float64]:
    n_series = Qbar.shape[0]
    a_arr = theta[:p]
    b_arr = theta[p:]
    intercept = (1.0 - float(np.sum(theta))) * Qbar

    eps_outer_buf = [
        np.outer(eps[-1 - i], eps[-1 - i]) if eps.shape[0] > i else np.zeros((n_series, n_series), dtype=np.float64)
        for i in range(max(p, 1))
    ]
    q_buf = [qt_path[-1 - j].copy() if qt_path.shape[0] > j else Qbar.copy() for j in range(max(q, 1))]

    out = np.empty((horizon, n_series, n_series), dtype=np.float64)
    for h in range(horizon):
        q_h = intercept.copy()
        for i in range(p):
            q_h = q_h + float(a_arr[i]) * eps_outer_buf[i]
        for j in range(q):
            q_h = q_h + float(b_arr[j]) * q_buf[j]

        r_h = _normalize_correlation(q_h)
        out[h] = r_h

        if p > 0:
            eps_outer_buf = [r_h.copy()] + eps_outer_buf[: p - 1]
        if q > 0:
            q_buf = [q_h.copy()] + q_buf[: q - 1]

    return out


class _CorrelationProcess:
    def next(self) -> NDArray[np.float64]:
        raise NotImplementedError

    def observe(self, innovation: NDArray[np.float64]) -> None:
        raise NotImplementedError


class _ConstantCorrelationProcess(_CorrelationProcess):
    def __init__(self, corr: NDArray[np.float64]) -> None:
        self._corr = corr

    def next(self) -> NDArray[np.float64]:
        return self._corr

    def observe(self, innovation: NDArray[np.float64]) -> None:
        return None


class _DCCCorrelationProcess(_CorrelationProcess):
    def __init__(
        self,
        theta: NDArray[np.float64],
        Qbar: NDArray[np.float64],
        qt_path: NDArray[np.float64],
        eps: NDArray[np.float64],
        p: int,
        q: int,
    ) -> None:
        self._a = theta[:p].copy()
        self._b = theta[p:].copy()
        self._intercept = (1.0 - float(np.sum(theta))) * Qbar
        self._p = p
        self._q = q
        n_series = Qbar.shape[0]
        self._eps_outer_buf = [
            np.outer(eps[-1 - i], eps[-1 - i]) if eps.shape[0] > i else np.zeros((n_series, n_series), dtype=np.float64)
            for i in range(max(p, 1))
        ]
        self._q_buf = [qt_path[-1 - j].copy() if qt_path.shape[0] > j else Qbar.copy() for j in range(max(q, 1))]
        self._current_q: Optional[NDArray[np.float64]] = None

    def next(self) -> NDArray[np.float64]:
        q_t = self._intercept.copy()
        for i in range(self._p):
            q_t = q_t + float(self._a[i]) * self._eps_outer_buf[i]
        for j in range(self._q):
            q_t = q_t + float(self._b[j]) * self._q_buf[j]
        self._current_q = q_t
        return _normalize_correlation(q_t)

    def observe(self, innovation: NDArray[np.float64]) -> None:
        if self._current_q is None:
            raise RuntimeError("next() must be called before observe().")
        if self._p > 0:
            self._eps_outer_buf = [np.outer(innovation, innovation)] + self._eps_outer_buf[: self._p - 1]
        if self._q > 0:
            self._q_buf = [self._current_q.copy()] + self._q_buf[: self._q - 1]


@dataclass
class _MarginalSimulationRuntime:
    mean_kind: str
    vol_kind: str
    const: float
    unconditional_mean: float
    phi: NDArray[np.float64]
    theta_ma: NDArray[np.float64]
    arx: NDArray[np.float64]
    har: NDArray[np.float64]
    xbeta: NDArray[np.float64]
    horizons: Tuple[int, ...]
    omega: float
    alpha: NDArray[np.float64]
    beta: NDArray[np.float64]
    gamma: Optional[NDArray[np.float64]]
    future_x: NDArray[np.float64]
    data_history: List[float]
    resid_history: List[float]
    sigma2_history: List[float]

    def one_step(self, step_index: int) -> Tuple[float, float]:
        if self.vol_kind == "dynamic":
            sigma2_t = self.omega
            for i, a in enumerate(self.alpha, start=1):
                resid_lag = self.resid_history[-i] if len(self.resid_history) >= i else 0.0
                resid2 = float(resid_lag * resid_lag)
                sigma2_t += float(a) * resid2
                if self.gamma is not None and resid_lag < 0.0:
                    sigma2_t += float(self.gamma[i - 1]) * resid2
            fallback_sigma2 = self.sigma2_history[-1] if self.sigma2_history else max(self.omega, 1e-12)
            for j, b in enumerate(self.beta, start=1):
                sigma_lag = self.sigma2_history[-j] if len(self.sigma2_history) >= j else fallback_sigma2
                sigma2_t += float(b) * float(sigma_lag)
            sigma2_t = max(float(sigma2_t), 1e-12)
        else:
            sigma2_t = max(float(self.sigma2_history[-1]), 1e-12)

        if self.mean_kind == "arma":
            mean_t = self.const
            for i, phi_i in enumerate(self.phi, start=1):
                y_lag = self.data_history[-i] if len(self.data_history) >= i else self.unconditional_mean
                mean_t += float(phi_i) * float(y_lag)
            for j, theta_j in enumerate(self.theta_ma, start=1):
                resid_lag = self.resid_history[-j] if len(self.resid_history) >= j else 0.0
                mean_t += float(theta_j) * float(resid_lag)
        elif self.mean_kind == "arx":
            mean_t = self.const
            for i, phi_i in enumerate(self.arx, start=1):
                y_lag = self.data_history[-i] if len(self.data_history) >= i else 0.0
                mean_t += float(phi_i) * float(y_lag)
            if self.xbeta.size:
                mean_t += float(self.future_x[step_index] @ self.xbeta)
        elif self.mean_kind == "harx":
            mean_t = self.const
            for horizon_lag, coef in zip(self.horizons, self.har):
                window = self.data_history[-horizon_lag:] if horizon_lag > 0 else self.data_history
                avg = float(np.mean(window)) if window else 0.0
                mean_t += float(coef) * avg
            if self.xbeta.size:
                mean_t += float(self.future_x[step_index] @ self.xbeta)
        else:
            mean_t = self.const

        return float(mean_t), float(sigma2_t)

    def observe(self, y_t: float, resid_t: float, sigma2_t: float) -> None:
        self.data_history.append(float(y_t))
        self.resid_history.append(float(resid_t))
        self.sigma2_history.append(float(sigma2_t))


def _coerce_future_x(
    mean_comp: Any,
    x_item: Any,
    n_obs: int,
) -> NDArray[np.float64]:
    if not isinstance(mean_comp, (ARX, HARX)):
        return np.zeros((n_obs, 0), dtype=np.float64)

    expected = getattr(mean_comp, "n_exog", 0)
    if expected <= 0:
        return np.zeros((n_obs, 0), dtype=np.float64)
    if x_item is None:
        raise ValueError(f"{mean_comp.signature} multivariate simulation requires x with shape ({n_obs}, {expected}).")

    arr = np.asarray(x_item, dtype=np.float64)
    if arr.ndim == 1:
        arr = arr[:, None]
    if arr.shape != (n_obs, expected):
        raise ValueError(f"{mean_comp.signature} multivariate simulation requires x with shape ({n_obs}, {expected}).")
    return np.ascontiguousarray(arr, dtype=np.float64)


def _build_marginal_runtimes(
    results: List["EstimationResult"],
    n_obs: int,
    x: Any,
    columns: Optional[List[Any]],
) -> List[_MarginalSimulationRuntime]:
    per_series_x = _resolve_per_series_inputs(x, len(results), columns, label="x")
    runtimes: List[_MarginalSimulationRuntime] = []

    for i, result in enumerate(results):
        spec = result._get_resolved_spec()
        state = result.filter()
        mean_comp = spec.get_component(Role.MEAN)
        vol = spec.get_component(Role.VOLATILITY)

        if state.distribution != "Normal":
            raise NotImplementedError(
                "Multivariate simulation currently requires Normal marginals. "
                "Non-Normal joint innovation simulation is intentionally unshipped "
                "until it can match the package contract without approximation."
            )

        const = 0.0
        unconditional_mean = 0.0
        phi = np.zeros(0, dtype=np.float64)
        theta_ma = np.zeros(0, dtype=np.float64)
        arx = np.zeros(0, dtype=np.float64)
        har = np.zeros(0, dtype=np.float64)
        xbeta = np.zeros(0, dtype=np.float64)
        horizons: Tuple[int, ...] = ()
        mean_kind = "constant"

        if isinstance(mean_comp, ARMA):
            params = mean_comp.fitted_params or {}
            const = float(params.get("const", params.get("c", 0.0)))
            phi = np.asarray(params.get("ar", params.get("phi", [])), dtype=np.float64)
            theta_ma = np.asarray(params.get("ma", params.get("theta", [])), dtype=np.float64)
            unconditional_mean = const / (1.0 - float(np.sum(phi))) if phi.size and abs(float(np.sum(phi))) < 1.0 else const
            mean_kind = "arma"
        elif isinstance(mean_comp, ARX):
            params = mean_comp.fitted_params or {}
            const = float(params.get("const", 0.0))
            arx = np.asarray(params.get("ar", []), dtype=np.float64)
            xbeta = np.asarray(params.get("exog", []), dtype=np.float64)
            mean_kind = "arx"
        elif isinstance(mean_comp, HARX):
            params = mean_comp.fitted_params or {}
            const = float(params.get("const", 0.0))
            har = np.asarray(params.get("har", []), dtype=np.float64)
            xbeta = np.asarray(params.get("exog", []), dtype=np.float64)
            horizons = tuple(int(h) for h in mean_comp.horizons)
            mean_kind = "harx"

        omega = 0.0
        alpha = np.zeros(0, dtype=np.float64)
        beta = np.zeros(0, dtype=np.float64)
        gamma = None
        vol_kind = "static"

        if isinstance(vol, GARCH):
            omega = float(vol.fitted_params["omega"])  # type: ignore[index]
            alpha = np.asarray(vol.fitted_params["alpha"], dtype=np.float64)  # type: ignore[index]
            beta = np.asarray(vol.fitted_params["beta"], dtype=np.float64)  # type: ignore[index]
            vol_kind = "dynamic"
        elif isinstance(vol, GJRGARCH):
            omega = float(vol.fitted_params["omega"])  # type: ignore[index]
            alpha = np.asarray(vol.fitted_params["alpha"], dtype=np.float64)  # type: ignore[index]
            beta = np.asarray(vol.fitted_params["beta"], dtype=np.float64)  # type: ignore[index]
            gamma = np.asarray(vol.fitted_params["gamma"], dtype=np.float64)  # type: ignore[index]
            vol_kind = "dynamic"

        runtimes.append(
            _MarginalSimulationRuntime(
                mean_kind=mean_kind,
                vol_kind=vol_kind,
                const=const,
                unconditional_mean=unconditional_mean,
                phi=phi,
                theta_ma=theta_ma,
                arx=arx,
                har=har,
                xbeta=xbeta,
                horizons=horizons,
                omega=omega,
                alpha=alpha,
                beta=beta,
                gamma=gamma,
                future_x=_coerce_future_x(mean_comp, per_series_x[i], n_obs),
                data_history=[float(x) for x in np.asarray(state.data, dtype=np.float64)],
                resid_history=[float(x) for x in np.asarray(state.residuals, dtype=np.float64)],
                sigma2_history=[float(x) for x in np.asarray(state.sigma2, dtype=np.float64)],
            )
        )

    return runtimes


def _simulate_multivariate_path(
    process: _CorrelationProcess,
    runtimes: List[_MarginalSimulationRuntime],
    n_obs: int,
    seed: Optional[int],
    columns: Optional[List[Any]],
) -> MultivariateSimulationResult:
    n_series = len(runtimes)
    rng = np.random.default_rng(seed)

    data = np.empty((n_obs, n_series), dtype=np.float64)
    mean = np.empty((n_obs, n_series), dtype=np.float64)
    residuals = np.empty((n_obs, n_series), dtype=np.float64)
    sigma2 = np.empty((n_obs, n_series), dtype=np.float64)
    innovations = np.empty((n_obs, n_series), dtype=np.float64)
    correlation = np.empty((n_obs, n_series, n_series), dtype=np.float64)
    residual_covariance = np.empty((n_obs, n_series, n_series), dtype=np.float64)

    for t in range(n_obs):
        for i, runtime in enumerate(runtimes):
            mean[t, i], sigma2[t, i] = runtime.one_step(t)

        corr_t = process.next()
        correlation[t] = corr_t
        residual_covariance[t] = corr_t * np.sqrt(sigma2[t])[:, None] * np.sqrt(sigma2[t])[None, :]
        try:
            chol = np.linalg.cholesky(corr_t)
        except np.linalg.LinAlgError:
            chol = np.linalg.cholesky(corr_t + 1e-10 * np.eye(n_series, dtype=np.float64))
        innovations[t] = chol @ rng.standard_normal(n_series)

        residuals[t] = np.sqrt(sigma2[t]) * innovations[t]
        data[t] = mean[t] + residuals[t]

        for i, runtime in enumerate(runtimes):
            runtime.observe(float(data[t, i]), float(residuals[t, i]), float(sigma2[t, i]))
        process.observe(innovations[t])

    return MultivariateSimulationResult(
        data=data,
        mean=mean,
        residuals=residuals,
        sigma2=sigma2,
        innovations=innovations,
        correlation=correlation,
        residual_covariance=residual_covariance,
        columns=columns,
    )


class _CorrelationResultMixin:
    N: int
    Qbar: NDArray[np.float64]
    univariate_results: Optional[List["EstimationResult"]]
    _index: Optional[Any]
    _columns: Optional[List[Any]]

    def _model_name(self) -> str:
        return type(self).__name__.replace("Result", "")

    def _resolve_pair(self, series_i: Union[int, str], series_j: Union[int, str]) -> Tuple[int, int]:
        def _resolve(key: Union[int, str]) -> int:
            if isinstance(key, int):
                if key < 0 or key >= self.N:
                    raise IndexError(f"Series index {key} out of range [0, {self.N})")
                return key
            if self._columns is None:
                raise ValueError(
                    f"Cannot look up series by name '{key}' — no column names stored. "
                    f"Pass a pandas DataFrame to {self._model_name()}.fit() to enable name-based access."
                )
            try:
                return self._columns.index(key)
            except ValueError as exc:
                raise KeyError(f"Series '{key}' not found. Available: {self._columns}") from exc

        return _resolve(series_i), _resolve(series_j)

    def corr(self, series_i: Union[int, str], series_j: Union[int, str]) -> Any:
        i, j = self._resolve_pair(series_i, series_j)
        rho = self.Rt[:, i, j]
        if self._index is not None:
            import pandas as pd

            i_name = self._columns[i] if self._columns is not None else str(i)
            j_name = self._columns[j] if self._columns is not None else str(j)
            return pd.Series(rho, index=self._index, name=f"rho_{i_name}_{j_name}")
        return rho

    @property
    def unconditional_corr(self) -> Any:
        corr = _normalize_correlation(self.Qbar)
        if self._columns is not None:
            import pandas as pd

            return pd.DataFrame(corr, index=self._columns, columns=self._columns)
        return corr

    def _require_univariate_results(self, method: str) -> List["EstimationResult"]:
        if self.univariate_results is None:
            raise RuntimeError(
                f"{self._model_name()}Result.{method}() requires stored univariate marginal fits. "
                f"Create the result via {self._model_name()}.fit(...) or pass "
                f"`univariate_results=` into {self._model_name()}.fit(...)."
            )
        return self.univariate_results

    def _forecast_correlation_matrices(self, horizon: int) -> NDArray[np.float64]:
        raise NotImplementedError

    def _make_simulation_process(self) -> _CorrelationProcess:
        raise NotImplementedError

    def forecast(
        self,
        horizon: int = 10,
        *,
        x: Any = None,
    ) -> MultivariateForecastResult:
        if horizon <= 0:
            raise ValueError("horizon must be positive")

        correlation = self._forecast_correlation_matrices(horizon)
        marginals = self._require_univariate_results("forecast")
        marginal_forecasts, mean, variance, residual_variance = _stack_marginal_forecasts(
            marginals,
            horizon,
            x,
            self._columns,
        )
        residual_covariance = _residual_covariance_from_correlation(correlation, residual_variance)
        return MultivariateForecastResult(
            mean=mean,
            variance=variance,
            residual_variance=residual_variance,
            correlation=correlation,
            residual_covariance=residual_covariance,
            marginals=tuple(marginal_forecasts),
            columns=self._columns,
        )

    def simulate(
        self,
        n_obs: int,
        *,
        seed: Optional[int] = None,
        x: Any = None,
    ) -> MultivariateSimulationResult:
        if n_obs <= 0:
            raise ValueError("n_obs must be positive")

        marginals = self._require_univariate_results("simulate")
        runtimes = _build_marginal_runtimes(marginals, n_obs, x, self._columns)
        return _simulate_multivariate_path(self._make_simulation_process(), runtimes, n_obs, seed, self._columns)


@dataclass
class CCCResult(_CorrelationResultMixin):
    """Result of a CCC estimation."""

    params: CCCParams
    nll: float
    T: int
    N: int
    Qbar: NDArray[np.float64]
    converged: bool
    nit: int
    time_elapsed: float
    univariate_results: Optional[List["EstimationResult"]] = None
    _eps: Optional[NDArray[np.float64]] = field(default=None, repr=False)
    _index: Optional[Any] = field(default=None, repr=False)
    _columns: Optional[List[Any]] = field(default=None, repr=False)
    _Rt_cache: Optional[NDArray[np.float64]] = field(default=None, repr=False)

    @property
    def K(self) -> int:
        return 0

    @property
    def theta(self) -> NDArray[np.float64]:
        return np.zeros(0, dtype=np.float64)

    @property
    def log_likelihood(self) -> float:
        return -self.nll * self.T

    @property
    def aic(self) -> float:
        return -2.0 * self.log_likelihood

    @property
    def bic(self) -> float:
        return -2.0 * self.log_likelihood

    @property
    def std_errors(self) -> NDArray[np.float64]:
        return np.empty(0, dtype=np.float64)

    @property
    def std_errors_robust(self) -> NDArray[np.float64]:
        return np.empty(0, dtype=np.float64)

    @property
    def Rt(self) -> NDArray[np.float64]:
        if self._Rt_cache is None:
            repeated = np.broadcast_to(self.params.corr, (self.T, self.N, self.N)).copy()
            object.__setattr__(self, "_Rt_cache", repeated)
        return self._Rt_cache  # type: ignore[return-value]

    def _forecast_correlation_matrices(self, horizon: int) -> NDArray[np.float64]:
        return np.broadcast_to(self.params.corr, (horizon, self.N, self.N)).copy()

    def _make_simulation_process(self) -> _CorrelationProcess:
        return _ConstantCorrelationProcess(self.params.corr)

    def summary(self) -> str:
        lines = []
        lines.append("=" * 72)
        lines.append(f"{'CCC Estimation Results':^72}")
        lines.append("=" * 72)
        lines.append("  Model: CCC")
        lines.append("  Method: Closed-form constant correlation")
        lines.append(
            f"  Obs: {self.T}  Series: {self.N}  "
            f"Converged: {self.converged}"
        )
        lines.append(f"  Time: {self.time_elapsed:.4f}s  Iterations: {self.nit}")
        lines.append(
            f"  Log-lik: {self.log_likelihood:.4f}  "
            f"AIC: {self.aic:.4f}  BIC: {self.bic:.4f}"
        )
        lines.append(f"  Marginals stored: {'Yes' if self.univariate_results is not None else 'No'}")
        lines.append("-" * 72)
        lines.append(f"  Average pairwise correlation: {self.params.avg_pairwise_corr:.6f}")
        lines.append("=" * 72)
        return "\n".join(lines)


@dataclass
class DCCResult(_CorrelationResultMixin):
    """Result of a DCC estimation."""

    params: DCCParams
    theta: NDArray[np.float64]
    nll: float
    p: int
    q: int
    T: int
    N: int
    Qbar: NDArray[np.float64]
    converged: bool
    nit: int
    time_elapsed: float
    hessian: Optional[NDArray[np.float64]] = None
    opg: Optional[NDArray[np.float64]] = None
    cov_mle: Optional[NDArray[np.float64]] = None
    cov_robust: Optional[NDArray[np.float64]] = None
    univariate_results: Optional[List["EstimationResult"]] = None
    _eps: Optional[NDArray[np.float64]] = field(default=None, repr=False)
    _index: Optional[Any] = field(default=None, repr=False)
    _columns: Optional[List[Any]] = field(default=None, repr=False)
    _Rt_cache: Optional[NDArray[np.float64]] = field(default=None, repr=False)
    _Qt_cache: Optional[NDArray[np.float64]] = field(default=None, repr=False)

    @property
    def K(self) -> int:
        return self.p + self.q

    @property
    def log_likelihood(self) -> float:
        return -self.nll * self.T

    @property
    def aic(self) -> float:
        return 2.0 * self.K - 2.0 * self.log_likelihood

    @property
    def bic(self) -> float:
        return self.K * np.log(self.T) - 2.0 * self.log_likelihood

    @property
    def std_errors(self) -> NDArray[np.float64]:
        if self.cov_mle is not None:
            return np.sqrt(np.maximum(np.diag(self.cov_mle), 0.0))
        return np.full(self.K, np.nan)

    @property
    def std_errors_robust(self) -> NDArray[np.float64]:
        if self.cov_robust is not None:
            return np.sqrt(np.maximum(np.diag(self.cov_robust), 0.0))
        return np.full(self.K, np.nan)

    def _ensure_recursion(self) -> None:
        if self._Rt_cache is not None and self._Qt_cache is not None:
            return
        if self._eps is None:
            raise RuntimeError(
                "Standardised residuals not stored on this result. "
                "Use DCC.fit() or DCC.fit_from_residuals() to create results."
            )
        rt, qt = _dcc_recursion(self.theta, self._eps, self.Qbar, self.p, self.q, return_q=True)
        object.__setattr__(self, "_Rt_cache", rt)
        object.__setattr__(self, "_Qt_cache", qt)

    @property
    def Rt(self) -> NDArray[np.float64]:
        self._ensure_recursion()
        return self._Rt_cache  # type: ignore[return-value]

    def _forecast_correlation_matrices(self, horizon: int) -> NDArray[np.float64]:
        self._ensure_recursion()
        return _dcc_forecast_correlation_path(
            self.theta,
            self._eps,  # type: ignore[arg-type]
            self.Qbar,
            self._Qt_cache,  # type: ignore[arg-type]
            self.p,
            self.q,
            horizon,
        )

    def _make_simulation_process(self) -> _CorrelationProcess:
        self._ensure_recursion()
        return _DCCCorrelationProcess(
            self.theta,
            self.Qbar,
            self._Qt_cache,  # type: ignore[arg-type]
            self._eps,  # type: ignore[arg-type]
            self.p,
            self.q,
        )

    def summary(self) -> str:
        names = [f"a[{i + 1}]" for i in range(self.p)] + [f"b[{j + 1}]" for j in range(self.q)]
        se = self.std_errors
        se_r = self.std_errors_robust

        lines = []
        lines.append("=" * 72)
        lines.append(f"{'DCC(' + str(self.p) + ',' + str(self.q) + ') Estimation Results':^72}")
        lines.append("=" * 72)
        lines.append(f"  Model: DCC({self.p},{self.q})")
        lines.append("  Method: Gaussian correlation MLE")
        lines.append(
            f"  Obs: {self.T}  Series: {self.N}  "
            f"Converged: {self.converged}"
        )
        lines.append(f"  Time: {self.time_elapsed:.4f}s  Iterations: {self.nit}")
        lines.append(
            f"  Log-lik: {self.log_likelihood:.4f}  "
            f"AIC: {self.aic:.4f}  BIC: {self.bic:.4f}"
        )
        lines.append(f"  Marginals stored: {'Yes' if self.univariate_results is not None else 'No'}")
        lines.append("-" * 72)
        lines.append(f"{'Param':<8} {'Estimate':>10} {'MLE SE':>10} {'Rob SE':>10} {'t-stat':>10}")
        lines.append("-" * 72)
        for i, name in enumerate(names):
            t_stat = self.theta[i] / se[i] if se[i] > 0 else np.nan
            lines.append(
                f"{name:<8} {self.theta[i]:>10.6f} {se[i]:>10.6f} "
                f"{se_r[i]:>10.6f} {t_stat:>10.2f}"
            )
        lines.append(f"\n  Persistence: {self.params.persistence:.6f}")
        lines.append("=" * 72)
        return "\n".join(lines)


class CCC:
    """CCC() - Constant Conditional Correlation baseline / diagnostic model."""

    @property
    def signature(self) -> str:
        return "CCC"

    @property
    def n_params(self) -> int:
        return 0

    def fit(
        self,
        returns: Any,
        univariate_spec: Optional["CompositeSpec"] = None,
        univariate_results: Optional[List["EstimationResult"]] = None,
    ) -> CCCResult:
        returns_np, index, columns = _extract_multivariate_input(returns)
        std_resid, uni_res = _prepare_univariate_results(returns_np, univariate_spec, univariate_results)
        result = self.fit_from_residuals(std_resid)
        object.__setattr__(result, "univariate_results", uni_res)
        object.__setattr__(result, "_index", index)
        object.__setattr__(result, "_columns", columns)
        return result

    def fit_from_residuals(
        self,
        eps: NDArray[np.float64],
        Qbar: Optional[NDArray[np.float64]] = None,
    ) -> CCCResult:
        eps = np.ascontiguousarray(eps, dtype=np.float64)
        if eps.ndim != 2:
            raise ValueError("eps must be 2-D (T, N)")
        t_count, n_series = eps.shape

        if Qbar is None:
            Qbar = _dk.compute_qbar(eps)
        Qbar = np.ascontiguousarray(Qbar, dtype=np.float64)

        start = time.perf_counter()
        corr = _normalize_correlation(Qbar)
        nll = _constant_correlation_nll(eps, corr)
        elapsed = time.perf_counter() - start

        return CCCResult(
            params=CCCParams(corr=corr),
            nll=nll,
            T=t_count,
            N=n_series,
            Qbar=Qbar,
            converged=True,
            nit=0,
            time_elapsed=elapsed,
            _eps=eps,
        )

    def __repr__(self) -> str:
        return self.signature


class DCC:
    """
    DCC(p, q) - Dynamic Conditional Correlation model.

    Parameters
    ----------
    p : int
        Number of ARCH-type lags for correlation dynamics.
    q : int
        Number of GARCH-type lags for correlation dynamics.
    """

    def __init__(self, p: int = 1, q: int = 1) -> None:
        if p < 1 or q < 1:
            raise ValueError("DCC requires p >= 1 and q >= 1")
        self.p = p
        self.q = q

    @property
    def signature(self) -> str:
        return f"DCC({self.p},{self.q})"

    @property
    def n_params(self) -> int:
        return self.p + self.q

    def fit(
        self,
        returns: Any,
        univariate_spec: Optional["CompositeSpec"] = None,
        univariate_results: Optional[List["EstimationResult"]] = None,
        **kwargs: Any,
    ) -> DCCResult:
        returns_np, index, columns = _extract_multivariate_input(returns)
        std_resid, uni_res = _prepare_univariate_results(returns_np, univariate_spec, univariate_results)

        result = self.fit_from_residuals(std_resid, **kwargs)
        object.__setattr__(result, "univariate_results", uni_res)
        object.__setattr__(result, "_index", index)
        object.__setattr__(result, "_columns", columns)
        return result

    def fit_from_residuals(
        self,
        eps: NDArray[np.float64],
        Qbar: Optional[NDArray[np.float64]] = None,
        theta0: Optional[NDArray[np.float64]] = None,
        compute_se: bool = True,
    ) -> DCCResult:
        eps = np.ascontiguousarray(eps, dtype=np.float64)
        if eps.ndim != 2:
            raise ValueError("eps must be 2-D (T, N)")
        t_count, n_series = eps.shape
        p, q = self.p, self.q
        k = p + q

        if Qbar is None:
            Qbar = _dk.compute_qbar(eps)
        Qbar = np.ascontiguousarray(Qbar, dtype=np.float64)

        if theta0 is None:
            theta0 = np.full(k, 0.05, dtype=np.float64)
        theta0 = np.ascontiguousarray(theta0, dtype=np.float64)

        bounds = [(1e-6, 0.999)] * k
        constraints = [{"type": "ineq", "fun": lambda x: 0.9999 - np.sum(x)}]
        cache: Dict[Tuple[float, ...], Tuple[float, NDArray[np.float64]]] = {}

        def _eval(x: NDArray[np.float64]) -> Tuple[float, NDArray[np.float64]]:
            x_arr = np.ascontiguousarray(x, dtype=np.float64)
            key = tuple(float(v) for v in x_arr)
            cached = cache.get(key)
            if cached is None:
                cached = _dk.dcc_nll_grad(x_arr, eps, Qbar, p, q)
                cache[key] = cached
                if len(cache) > 20:
                    cache.pop(next(iter(cache)))
            return cached

        start = time.perf_counter()
        # Always use the shipped analytical C gradient on the optimization path.
        opt = minimize(
            fun=lambda x: _eval(np.asarray(x, dtype=np.float64))[0],
            x0=theta0,
            method="SLSQP",
            jac=lambda x: _eval(np.asarray(x, dtype=np.float64))[1],
            bounds=bounds,
            constraints=constraints,
            options={"maxiter": 500, "ftol": 1e-12},
        )
        elapsed = time.perf_counter() - start
        theta_hat = np.asarray(opt.x, dtype=np.float64).copy()

        hessian = opg = cov_mle = cov_robust = None
        if compute_se:
            _, _, hess_h, scores_h = _dk.dcc_nll_grad_hess(theta_hat, eps, Qbar, p, q, return_scores=True)
            hessian = hess_h
            opg = (scores_h.T @ scores_h) / t_count
            try:
                h_inv = np.linalg.inv(hess_h)
                cov_mle = h_inv / t_count
                cov_robust = (h_inv @ opg @ h_inv) / t_count
            except np.linalg.LinAlgError:
                pass

        params = DCCParams(a=theta_hat[:p].copy(), b=theta_hat[p:].copy())
        return DCCResult(
            params=params,
            theta=theta_hat,
            nll=float(opt.fun),
            p=p,
            q=q,
            T=t_count,
            N=n_series,
            Qbar=Qbar,
            converged=bool(opt.success),
            nit=int(opt.nit),
            time_elapsed=elapsed,
            hessian=hessian,
            opg=opg,
            cov_mle=cov_mle,
            cov_robust=cov_robust,
            _eps=eps,
        )

    def __repr__(self) -> str:
        return self.signature
