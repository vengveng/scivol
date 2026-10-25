"""
Multivariate correlation models: CCC (constant) and DCC(p, q).

Both are the second stage of a staged estimator: univariate marginal models are
fitted first; the correlation model is fitted on their standardized residuals
with the target matrix Qbar = eps'eps / T. The correlation likelihood, its
derivatives, the qualification of the estimate, its covariance (which treats
the residuals and Qbar as given), the correlation paths, and simulated
continuations (copula draws mapped through each marginal's quantile function,
then every marginal continued natively) come from the native library.
"""

from __future__ import annotations

import time
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Union

import numpy as np
from numpy.typing import NDArray

from . import _scivolapi as api
from .components import GARCH, Normal

if TYPE_CHECKING:
    from ._evaluation import ForecastResult
    from .result import EstimationResult
    from .spec import CompositeSpec


def _normalize_correlation(matrix: NDArray[np.float64]) -> NDArray[np.float64]:
    diag = np.sqrt(np.diag(matrix))
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
    *,
    returns_source: Any = None,
    initial_results: Any = None,
    n_jobs: Optional[int] = None,
    timeout: Optional[float] = None,
    fit_kwargs: Optional[Dict[str, Any]] = None,
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

    batch = univariate_spec.fit_many(
        returns_np if returns_source is None else returns_source,
        initial_results=initial_results,
        n_jobs=n_jobs,
        timeout=timeout,
        on_error="record",
        **(fit_kwargs or {}),
    )
    if batch.failures:
        details = "; ".join(
            f"{name}: {failure.message}" for name, failure in batch.failures.items()
        )
        raise RuntimeError(f"Univariate marginal fitting failed: {details}")

    fitted = list(batch.results.values())
    std_resid_cols = [np.asarray(result.std_resid, dtype=np.float64) for result in fitted]
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
    copula_distribution: str = "gaussian"
    copula_eta: Optional[float] = None
    columns: Optional[List[Any]] = None

    @property
    def std_resid(self) -> NDArray[np.float64]:
        return self.innovations

    def to_dict(self) -> dict[str, Any]:
        return {
            "data": self.data.copy(),
            "mean": self.mean.copy(),
            "residuals": self.residuals.copy(),
            "sigma2": self.sigma2.copy(),
            "innovations": self.innovations.copy(),
            "correlation": self.correlation.copy(),
            "residual_covariance": self.residual_covariance.copy(),
            "copula_distribution": self.copula_distribution,
            "copula_eta": self.copula_eta,
        }


@dataclass(slots=True)
class MultivariateScenarioResult:
    data: NDArray[np.float64]
    mean: NDArray[np.float64]
    residuals: NDArray[np.float64]
    residual_variance: NDArray[np.float64]
    innovations: NDArray[np.float64]
    correlation: NDArray[np.float64]
    residual_covariance: NDArray[np.float64]
    innovation_method: str
    copula_distribution: str = "gaussian"
    copula_eta: Optional[float] = None
    columns: Optional[List[Any]] = None

    @property
    def sigma2(self) -> NDArray[np.float64]:
        return self.residual_variance

    def to_dict(self) -> dict[str, Any]:
        return {
            "data": self.data.copy(),
            "mean": self.mean.copy(),
            "residuals": self.residuals.copy(),
            "residual_variance": self.residual_variance.copy(),
            "sigma2": self.sigma2.copy(),
            "innovations": self.innovations.copy(),
            "correlation": self.correlation.copy(),
            "residual_covariance": self.residual_covariance.copy(),
            "innovation_method": self.innovation_method,
            "copula_distribution": self.copula_distribution,
            "copula_eta": self.copula_eta,
        }


@dataclass(frozen=True)
class RollingDCCWindow:
    start: int
    stop: int
    label: Any
    result: "DCCResult"


@dataclass(frozen=True)
class RollingDCCResult(Sequence[RollingDCCWindow]):
    windows: tuple[RollingDCCWindow, ...]

    def __getitem__(self, index):
        return self.windows[index]

    def __len__(self) -> int:
        return len(self.windows)


@dataclass
class DCCParams:
    """Container for DCC parameters."""

    a: NDArray[np.float64]
    b: NDArray[np.float64]
    eta: Optional[float] = None

    @property
    def persistence(self) -> float:
        return float(np.sum(self.a) + np.sum(self.b))

    def __repr__(self) -> str:
        a_str = ", ".join(f"{x:.6f}" for x in self.a)
        b_str = ", ".join(f"{x:.6f}" for x in self.b)
        eta_str = "" if self.eta is None else f", eta={self.eta:.6f}"
        return (
            f"DCCParams(a=[{a_str}], b=[{b_str}]"
            f"{eta_str}, persistence={self.persistence:.6f})"
        )


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


def correlation_model(p: int, q: int, distribution: str) -> "api.CorrelationModel":
    spec = api.CorrelationSpec()
    spec.p, spec.q = int(p), int(q)
    spec.density = api.CorrelationDensity.student_t if distribution == "studentt" else api.CorrelationDensity.normal
    return api.CorrelationModel(spec)


def _marginal_density(result: "EstimationResult") -> "api.InnovationDensity":
    return result._native()[0].innovation_density(np.ascontiguousarray(result.params, dtype=np.float64))


class _CorrelationResultMixin:
    """Shared by CCC and DCC results: a native correlation model at `_dynamics`
    (the DCC coefficients; zeros for CCC) on the residuals `_eps` with target `Qbar`."""

    N: int
    Qbar: NDArray[np.float64]
    univariate_results: Optional[List["EstimationResult"]]
    distribution: str
    _eps: Optional[NDArray[np.float64]]
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

    # Native model and paths ------------------------------------------------- #
    def _correlation(self) -> tuple["api.CorrelationModel", NDArray[np.float64]]:
        raise NotImplementedError

    def _copula_eta(self) -> Optional[float]:
        return None

    @property
    def Rt(self) -> NDArray[np.float64]:
        if self._Rt_cache is None:
            model, theta = self._correlation()
            out = api.CorrelationJet(model, self._eps, self.Qbar).theta(theta, api.PATHS)
            if out["status"] != "ok":
                raise ValueError(f"The correlation path is not available ({out['status']}).")
            object.__setattr__(self, "_Rt_cache", np.asarray(out["correlations"]))
        return self._Rt_cache

    def _joint_paths(self, paths: int, horizon: int, seed: Any, x: Any, method: str) -> dict[str, NDArray[np.float64]]:
        """Correlation-driven innovations and every marginal continued along them: paths x horizon (x series)."""
        marginals = self._require_univariate_results(method)
        rng = np.random.default_rng(seed)
        normals = rng.standard_normal((paths, horizon, self.N))
        eta = self._copula_eta()
        chi2 = None if eta is None or eta == 0.0 else rng.gamma(0.5 / eta, 2.0 * eta, (paths, horizon))
        model, theta = self._correlation()
        correlation = api.simulate_correlation(model, theta, self._eps, [_marginal_density(m) for m in marginals],
                                               normals, chi2, self.Qbar)
        future = _resolve_per_series_inputs(x, self.N, self._columns, label="x")
        columns = {key: np.empty((paths, horizon, self.N)) for key in ("y", "mean", "residual", "variance")}
        for i, marginal in enumerate(marginals):
            z = np.ascontiguousarray(correlation["innovation"][:, :, i])
            future_x = None if future[i] is None else np.ascontiguousarray(
                np.asarray(future[i], dtype=np.float64).reshape(horizon, -1))
            out = api.simulate(marginal._native_model, np.ascontiguousarray(marginal.params), marginal._native_y, z,
                               marginal._native_x, future_x)
            for key in columns:
                columns[key][:, :, i] = out[key]
        columns["innovation"] = correlation["innovation"]
        columns["correlation"] = correlation["correlation"]
        return columns

    def forecast_correlation(self, horizon: int = 10, *, simulations: int = 10_000,
                             seed: Any = 0) -> NDArray[np.float64]:
        """E[R_{T+h}]: exact one step ahead, an average over `simulations` paths beyond."""
        if horizon <= 0:
            raise ValueError("horizon must be positive")
        paths = 1 if horizon == 1 else int(simulations)
        return self._joint_paths(paths, horizon, seed, None, "forecast_correlation")["correlation"].mean(axis=0)

    def forecast(self, horizon: int = 10, *, x: Any = None, simulations: int = 10_000,
                 seed: Any = 0) -> MultivariateForecastResult:
        """Marginal forecasts, E[R_{T+h}] and the residual covariance E[e_{T+h} e_{T+h}'],
        the last two averaged over `simulations` joint paths (exact one step ahead for R)."""
        if horizon <= 0:
            raise ValueError("horizon must be positive")
        marginals = self._require_univariate_results("forecast")
        forecasts, mean, variance, residual_variance = _stack_marginal_forecasts(marginals, horizon, x, self._columns)
        paths = self._joint_paths(int(simulations), horizon, seed, x, "forecast")
        e = paths["residual"]
        return MultivariateForecastResult(
            mean=mean, variance=variance, residual_variance=residual_variance,
            correlation=paths["correlation"].mean(axis=0),
            residual_covariance=np.einsum("phi,phj->hij", e, e) / e.shape[0],
            marginals=tuple(forecasts), columns=self._columns,
        )

    def simulate(self, n_obs: int, *, seed: Optional[int] = None, x: Any = None) -> MultivariateSimulationResult:
        """One path of n_obs rows continuing the sample: correlations, copula innovations, marginals."""
        if n_obs <= 0:
            raise ValueError("n_obs must be positive")
        paths = self._joint_paths(1, n_obs, seed, x, "simulate")
        sigma = np.sqrt(paths["variance"][0])
        return MultivariateSimulationResult(
            data=paths["y"][0], mean=paths["mean"][0], residuals=paths["residual"][0], sigma2=paths["variance"][0],
            innovations=paths["innovation"][0], correlation=paths["correlation"][0],
            residual_covariance=paths["correlation"][0] * sigma[:, :, None] * sigma[:, None, :],
            copula_distribution=self.distribution, copula_eta=self._copula_eta(), columns=self._columns,
        )

    def scenarios(self, n_draws: int, *, seed: Optional[int] = None, x: Any = None,
                  innovations: str = "parametric") -> MultivariateScenarioResult:
        """n_draws one-step outcomes from the forecast origin.

        parametric: the copula draws mapped through each marginal's density.
        empirical: the same copula probabilities mapped through the empirical
        quantiles of each marginal's standardized residuals.
        """
        if not isinstance(n_draws, (int, np.integer)) or int(n_draws) <= 0:
            raise ValueError("n_draws must be a positive integer")
        if innovations not in {"parametric", "empirical"}:
            raise ValueError("innovations must be 'parametric' or 'empirical'.")
        paths = self._joint_paths(int(n_draws), 1, seed, x, "scenarios")
        mean, variance, correlation = paths["mean"][0, 0], paths["variance"][0, 0], paths["correlation"][0, 0]
        z = paths["innovation"][:, 0]
        if innovations == "empirical":
            z = self._empirical_innovations(z)
        sigma = np.sqrt(variance)
        residuals = z * sigma
        return MultivariateScenarioResult(
            data=mean + residuals, mean=mean, residuals=residuals, residual_variance=variance, innovations=z,
            correlation=correlation, residual_covariance=correlation * np.outer(sigma, sigma),
            innovation_method=innovations, copula_distribution=self.distribution, copula_eta=self._copula_eta(),
            columns=self._columns,
        )

    def _empirical_innovations(self, parametric: NDArray[np.float64]) -> NDArray[np.float64]:
        """Each marginal's copula probability u = F_i(z_i), mapped through the empirical
        quantiles of its standardized residuals (centred and scaled to unit variance)."""
        out = np.empty_like(parametric)
        for i, marginal in enumerate(self._require_univariate_results("scenarios")):
            density = _marginal_density(marginal)
            u, _ = api.distribution(density.density, density.shape, np.ascontiguousarray(parametric[:, i]))
            empirical = np.asarray(marginal.std_resid, dtype=np.float64)
            empirical = empirical[np.isfinite(empirical)]
            if empirical.size < 10:
                raise ValueError(f"Series {i} has fewer than 10 finite standardized residuals.")
            empirical = (empirical - empirical.mean()) / empirical.std(ddof=0)
            out[:, i] = np.quantile(empirical, u, method="linear")
        return out


@dataclass
class CCCResult(_CorrelationResultMixin):
    """Result of a CCC estimation: R = Qbar normalized to a correlation matrix."""

    params: CCCParams
    nll: float
    T: int
    N: int
    Qbar: NDArray[np.float64]
    converged: bool
    nit: int
    time_elapsed: float
    distribution: str = "gaussian"
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

    def _correlation(self) -> tuple["api.CorrelationModel", NDArray[np.float64]]:
        # DCC(1,1) at a = b = 0: Q_t = Qbar for every row.
        return correlation_model(1, 1, "gaussian"), np.zeros(2)

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
    warm_started: bool = False
    maxiter: int = 500
    distribution: str = "gaussian"
    hessian: Optional[NDArray[np.float64]] = None
    opg: Optional[NDArray[np.float64]] = None
    cov_mle: Optional[NDArray[np.float64]] = None
    cov_robust: Optional[NDArray[np.float64]] = None
    success: bool = False
    inference: str = ""
    kkt_residual: float = float("nan")
    warnings: tuple[str, ...] = ()
    phases: tuple[str, ...] = ()
    univariate_results: Optional[List["EstimationResult"]] = None
    _eps: Optional[NDArray[np.float64]] = field(default=None, repr=False)
    _index: Optional[Any] = field(default=None, repr=False)
    _columns: Optional[List[Any]] = field(default=None, repr=False)
    _Rt_cache: Optional[NDArray[np.float64]] = field(default=None, repr=False)

    @property
    def K(self) -> int:
        return int(self.theta.size)

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
            return np.sqrt(np.diag(self.cov_mle))
        return np.full(self.K, np.nan)

    @property
    def std_errors_robust(self) -> NDArray[np.float64]:
        if self.cov_robust is not None:
            return np.sqrt(np.diag(self.cov_robust))
        return np.full(self.K, np.nan)

    def _correlation(self) -> tuple["api.CorrelationModel", NDArray[np.float64]]:
        return correlation_model(self.p, self.q, self.distribution), np.ascontiguousarray(self.theta)

    def _copula_eta(self) -> Optional[float]:
        return self.params.eta

    def summary(self) -> str:
        model, theta = self._correlation()
        names = model.parameter_names
        se = self.std_errors
        se_r = self.std_errors_robust

        lines = []
        lines.append("=" * 72)
        lines.append(f"{'DCC(' + str(self.p) + ',' + str(self.q) + ') Estimation Results':^72}")
        lines.append("=" * 72)
        lines.append(f"  Model: DCC({self.p},{self.q})")
        method = "Student-t copula correlation MLE" if self.distribution == "studentt" else "Gaussian correlation MLE"
        lines.append(f"  Method: {method}")
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
        for name, value in model.innovation_density(theta).derived.items():
            lines.append(f"  {name}: {value:.6g}")
        lines.append(f"\n  Persistence: {self.params.persistence:.6f}  Inference: {self.inference}")
        lines.append("=" * 72)
        return "\n".join(lines)




def _residual_input(eps: Any) -> NDArray[np.float64]:
    eps = np.ascontiguousarray(eps, dtype=np.float64)
    if eps.ndim != 2:
        raise ValueError("eps must be 2-D (T, N)")
    return eps


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
        univariate_initial_results: Any = None,
        univariate_fit_kwargs: Optional[Dict[str, Any]] = None,
        n_jobs: Optional[int] = None,
        marginal_timeout: Optional[float] = None,
    ) -> CCCResult:
        returns_np, index, columns = _extract_multivariate_input(returns)
        std_resid, marginals = _prepare_univariate_results(
            returns_np, univariate_spec, univariate_results, returns_source=returns,
            initial_results=univariate_initial_results, n_jobs=n_jobs, timeout=marginal_timeout,
            fit_kwargs=univariate_fit_kwargs,
        )
        result = self.fit_from_residuals(std_resid)
        result.univariate_results = marginals
        result._index = None if index is None else index[-result.T:]
        result._columns = columns
        return result

    def fit_from_residuals(self, eps: NDArray[np.float64], Qbar: Optional[NDArray[np.float64]] = None) -> CCCResult:
        """R = Qbar / sqrt(diag Qbar diag Qbar'), with Qbar = eps'eps / T unless given."""
        eps = _residual_input(eps)
        t_count, n_series = eps.shape
        started = time.perf_counter()
        target = None if Qbar is None else np.ascontiguousarray(Qbar, dtype=np.float64)
        jet = api.CorrelationJet(correlation_model(1, 1, "gaussian"), eps, target)
        out = jet.theta(np.zeros(2), api.VALUE)
        if out["status"] != "ok":
            raise ValueError(f"The constant correlation likelihood is not available ({out['status']}).")
        qbar = np.asarray(jet.target)
        return CCCResult(
            params=CCCParams(corr=_normalize_correlation(qbar)), nll=float(out["value"]) / t_count, T=t_count,
            N=n_series, Qbar=qbar, converged=True, nit=0, time_elapsed=time.perf_counter() - started, _eps=eps,
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
    distribution : str
        "gaussian" or "studentt" (the Student-t copula's correlation likelihood).
    """

    def __init__(self, p: int = 1, q: int = 1, distribution: str = "gaussian") -> None:
        if p < 1 or q < 1:
            raise ValueError("DCC requires p >= 1 and q >= 1")
        distribution_key = distribution.lower().replace("-", "").replace("_", "")
        aliases = {"gaussian": "gaussian", "normal": "gaussian", "studentt": "studentt", "t": "studentt"}
        if distribution_key not in aliases:
            raise ValueError("distribution must be 'gaussian' or 'studentt'.")
        self.p = p
        self.q = q
        self.distribution = aliases[distribution_key]

    @property
    def signature(self) -> str:
        suffix = "" if self.distribution == "gaussian" else "+StudentT"
        return f"DCC({self.p},{self.q}){suffix}"

    @property
    def n_params(self) -> int:
        return self.p + self.q + (1 if self.distribution == "studentt" else 0)

    def fit(
        self,
        returns: Any,
        univariate_spec: Optional["CompositeSpec"] = None,
        univariate_results: Optional[List["EstimationResult"]] = None,
        univariate_initial_results: Any = None,
        univariate_fit_kwargs: Optional[Dict[str, Any]] = None,
        n_jobs: Optional[int] = None,
        marginal_timeout: Optional[float] = None,
        **kwargs: Any,
    ) -> DCCResult:
        returns_np, index, columns = _extract_multivariate_input(returns)
        std_resid, marginals = _prepare_univariate_results(
            returns_np, univariate_spec, univariate_results, returns_source=returns,
            initial_results=univariate_initial_results, n_jobs=n_jobs, timeout=marginal_timeout,
            fit_kwargs=univariate_fit_kwargs,
        )
        result = self.fit_from_residuals(std_resid, **kwargs)
        result.univariate_results = marginals
        result._index = None if index is None else index[-result.T:]
        result._columns = columns
        return result

    def fit_from_residuals(
        self,
        eps: NDArray[np.float64],
        Qbar: Optional[NDArray[np.float64]] = None,
        theta0: Optional[NDArray[np.float64]] = None,
        initial_result: Optional[DCCResult] = None,
        maxiter: Optional[int] = None,
        accuracy: float = 1e-10,
    ) -> DCCResult:
        """Fit DCC(p, q) on standardized residuals eps (T x N), Qbar = eps'eps / T unless given."""
        eps = _residual_input(eps)
        if theta0 is not None and initial_result is not None:
            raise ValueError("Pass either theta0 or initial_result, not both.")
        if initial_result is not None:
            if (initial_result.p, initial_result.q, initial_result.distribution) != (self.p, self.q, self.distribution):
                raise ValueError(f"Warm-start result is for {initial_result.distribution} "
                                 f"DCC({initial_result.p},{initial_result.q}); this fit is {self.signature}.")
            theta0 = initial_result.theta
        p, q, t_count = self.p, self.q, eps.shape[0]
        maxiter = 500 if maxiter is None else int(maxiter)
        model = correlation_model(p, q, self.distribution)
        options = api.FitOptions()
        options.maximum_iterations = maxiter
        options.accuracy = float(accuracy)
        if theta0 is not None:
            options.start = [float(v) for v in np.asarray(theta0, dtype=np.float64).ravel()]
        target = None if Qbar is None else np.ascontiguousarray(Qbar, dtype=np.float64)
        started = time.perf_counter()
        r = api.fit_correlation(model, eps, options, target)
        elapsed = time.perf_counter() - started
        if not r.estimate_available:
            raise RuntimeError("DCC fit failed: " + "; ".join(r.warnings))
        theta = np.array(r.theta)
        return DCCResult(
            params=DCCParams(a=theta[:p].copy(), b=theta[p:p + q].copy(),
                             eta=model.innovation_density(theta).named.get("eta")),
            theta=theta,
            nll=-r.log_likelihood / t_count,
            p=p,
            q=q,
            T=t_count,
            N=eps.shape[1],
            Qbar=np.asarray(api.CorrelationJet(model, eps, target).target),
            converged=r.termination == api.Termination.converged,
            nit=sum(phase.iterations for phase in r.phases),
            time_elapsed=elapsed,
            warm_started=theta0 is not None,
            maxiter=maxiter,
            distribution=self.distribution,
            hessian=np.array(r.hessian) / t_count if r.hessian.size else None,  # of the per-row average
            opg=np.array(r.opg) / t_count if r.opg.size else None,
            cov_mle=np.array(r.covariance) if r.covariance.size else None,
            cov_robust=np.array(r.robust_covariance) if r.robust_covariance.size else None,
            success=bool(r.success),
            inference=r.inference.name,
            kkt_residual=float(r.kkt_residual),
            warnings=tuple(r.warnings),
            phases=tuple(phase.name for phase in r.phases),
            _eps=eps,
        )

    def fit_rolling(
        self,
        returns: Any,
        *,
        window: int,
        step: int = 1,
        expanding: bool = False,
        univariate_spec: Optional["CompositeSpec"] = None,
        initial_result: Optional[DCCResult] = None,
        univariate_fit_kwargs: Optional[Dict[str, Any]] = None,
        n_jobs: Optional[int] = None,
        marginal_timeout: Optional[float] = None,
        maxiter: Optional[int] = None,
    ) -> RollingDCCResult:
        """Fit rolling DCC windows using prior marginal and DCC estimates."""
        returns_np, index, _ = _extract_multivariate_input(returns)
        if not isinstance(window, int) or window <= 0:
            raise ValueError("window must be a positive integer.")
        if not isinstance(step, int) or step <= 0:
            raise ValueError("step must be a positive integer.")
        if window > returns_np.shape[0]:
            raise ValueError("window cannot exceed the number of observations.")

        previous = initial_result
        windows = []
        for stop in range(window, returns_np.shape[0] + 1, step):
            start = 0 if expanding else stop - window
            try:
                window_returns = returns.iloc[start:stop]
            except AttributeError:
                window_returns = returns_np[start:stop]
            label = index[stop - 1] if index is not None else stop - 1
            fitted = self.fit(
                window_returns,
                univariate_spec=univariate_spec,
                univariate_initial_results=(
                    None if previous is None else previous.univariate_results
                ),
                univariate_fit_kwargs=univariate_fit_kwargs,
                n_jobs=n_jobs,
                marginal_timeout=marginal_timeout,
                initial_result=previous,
                maxiter=maxiter,
            )
            windows.append(
                RollingDCCWindow(start=start, stop=stop, label=label, result=fitted)
            )
            previous = fitted
        return RollingDCCResult(tuple(windows))

    def __repr__(self) -> str:
        return self.signature
