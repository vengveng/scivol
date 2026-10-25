"""DCC correlation fits through scivolapi.

The correlation stage of a staged estimator: the marginal fits produce the
standardized residuals; the native library fits DCC(p, q) on them with exact
derivatives, qualifies the estimate and forms its covariance treating the
residuals and the target matrix as given.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np
from numpy.typing import NDArray

from . import _scivolapi as api
from .dcc import (
    DCCParams,
    DCCResult,
    MultivariateForecastResult,
    MultivariateScenarioResult,
    MultivariateSimulationResult,
    _resolve_per_series_inputs,
    _stack_marginal_forecasts,
)



@dataclass
class NativeDCCResult(DCCResult):
    """A DCC result whose estimate, inference and correlation path come from scivolapi."""

    success: bool = False
    inference: str = ""
    kkt_residual: float = float("nan")
    warnings: tuple[str, ...] = ()
    phases: tuple[str, ...] = ()
    _native_model: Any = field(default=None, repr=False)

    def _native_marginals(self, method: str) -> list[Any]:
        from ._scivolapi_eval import NativeEstimationResult

        marginals = self._require_univariate_results(method)
        if not all(isinstance(m, NativeEstimationResult) for m in marginals):
            raise RuntimeError(f"DCC {method}() with native correlation paths needs marginals fitted by scivolapi.")
        return marginals

    def _joint_paths(self, paths: int, horizon: int, seed: Any, x: Any, method: str) -> dict[str, NDArray[np.float64]]:
        """Correlation-driven innovations and every marginal continued along them: paths x horizon (x series)."""
        marginals = self._native_marginals(method)
        rng = np.random.default_rng(seed)
        normals = rng.standard_normal((paths, horizon, self.N))
        chi2 = rng.chisquare(self.params.nu, (paths, horizon)) if self.distribution == "studentt" else None
        correlation = api.simulate_correlation(self._native_model, np.ascontiguousarray(self.theta), self._eps,
                                               [_marginal_density(m) for m in marginals], normals, chi2, self.Qbar)
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
            copula_distribution=self.distribution, copula_nu=self.params.nu, columns=self._columns,
        )

    def scenarios(self, n_draws: int, *, seed: Optional[int] = None, x: Any = None,
                  innovations: str = "parametric") -> MultivariateScenarioResult:
        """n_draws one-step outcomes from the forecast origin (parametric innovations natively)."""
        if innovations != "parametric":
            return super().scenarios(n_draws, seed=seed, x=x, innovations=innovations)
        if not isinstance(n_draws, (int, np.integer)) or int(n_draws) <= 0:
            raise ValueError("n_draws must be a positive integer")
        paths = self._joint_paths(int(n_draws), 1, seed, x, "scenarios")
        variance, correlation = paths["variance"][0, 0], paths["correlation"][0, 0]
        sigma = np.sqrt(variance)
        return MultivariateScenarioResult(
            data=paths["y"][:, 0], mean=paths["mean"][0, 0], residuals=paths["residual"][:, 0],
            residual_variance=variance, innovations=paths["innovation"][:, 0], correlation=correlation,
            residual_covariance=correlation * np.outer(sigma, sigma), innovation_method="parametric",
            copula_distribution=self.distribution, copula_nu=self.params.nu, columns=self._columns,
        )

    def forecast_correlation(self, horizon: int = 10, *, simulations: int = 10_000,
                             seed: Any = 0) -> NDArray[np.float64]:
        """E[R_{T+h}]: exact one step ahead, an average over `simulations` paths beyond."""
        if horizon <= 0:
            raise ValueError("horizon must be positive")
        paths = self._joint_paths(1 if horizon == 1 else int(simulations), horizon, seed, None, "forecast_correlation")
        return paths["correlation"].mean(axis=0)

    def forecast(self, horizon: int = 10, *, x: Any = None, simulations: int = 10_000,
                 seed: Any = 0) -> MultivariateForecastResult:
        """Marginal forecasts (exact where the marginals are), E[R_{T+h}] and the residual covariance
        E[e_{T+h} e_{T+h}'] averaged over `simulations` joint paths."""
        if horizon <= 0:
            raise ValueError("horizon must be positive")
        marginals = self._native_marginals("forecast")
        forecasts, mean, variance, residual_variance = _stack_marginal_forecasts(marginals, horizon, x, self._columns)
        paths = self._joint_paths(int(simulations), horizon, seed, x, "forecast")
        e = paths["residual"]
        return MultivariateForecastResult(
            mean=mean, variance=variance, residual_variance=residual_variance,
            correlation=paths["correlation"].mean(axis=0),
            residual_covariance=np.einsum("phi,phj->hij", e, e) / e.shape[0],
            marginals=tuple(forecasts), columns=self._columns,
        )

    @property
    def Rt(self) -> NDArray[np.float64]:
        if self._Rt_cache is None:
            jet = api.CorrelationJet(self._native_model, self._eps, self.Qbar)
            out = jet.theta(np.ascontiguousarray(self.theta), api.PATHS)
            if out["status"] != "ok":
                raise ValueError(f"The correlation path is not available ({out['status']}).")
            object.__setattr__(self, "_Rt_cache", np.asarray(out["correlations"]))
        return self._Rt_cache


def _marginal_density(result: Any) -> "api.MarginalDensity":
    from ._scivolapi_eval import innovation_density

    return innovation_density(result.spec, np.asarray(result.params, dtype=np.float64))


def correlation_model(p: int, q: int, distribution: str) -> "api.CorrelationModel":
    spec = api.CorrelationSpec()
    spec.p, spec.q = int(p), int(q)
    spec.density = api.CorrelationDensity.student_t if distribution == "studentt" else api.CorrelationDensity.normal
    return api.CorrelationModel(spec)


def fit_from_residuals(
    p: int,
    q: int,
    distribution: str,
    eps: NDArray[np.float64],
    *,
    Qbar: Optional[NDArray[np.float64]] = None,
    theta0: Optional[NDArray[np.float64]] = None,
    maxiter: int = 500,
    accuracy: float = 1e-10,
    warm_started: bool = False,
) -> NativeDCCResult:
    import time

    eps = np.ascontiguousarray(eps, dtype=np.float64)
    target = None if Qbar is None else np.ascontiguousarray(Qbar, dtype=np.float64)
    model = correlation_model(p, q, distribution)
    options = api.FitOptions()
    options.maximum_iterations = int(maxiter)
    options.accuracy = float(accuracy)
    if theta0 is not None:
        options.start = [float(v) for v in np.asarray(theta0, dtype=np.float64)]
    started = time.perf_counter()
    r = api.fit_correlation(model, eps, options, target)
    elapsed = time.perf_counter() - started
    if not r.estimate_available:
        raise RuntimeError("DCC fit failed: " + "; ".join(r.warnings))

    t_count, n_series = eps.shape
    theta = np.array(r.theta)
    covariance = np.array(r.covariance) if r.covariance.size else None
    robust = np.array(r.robust_covariance) if r.robust_covariance.size else None
    qbar = np.asarray(api.CorrelationJet(model, eps, target).target) if target is None else target
    return NativeDCCResult(
        params=DCCParams(a=theta[:p].copy(), b=theta[p:p + q].copy(),
                         nu=float(theta[-1]) if distribution == "studentt" else None),
        theta=theta,
        nll=-r.log_likelihood / t_count,
        p=p,
        q=q,
        T=t_count,
        N=n_series,
        Qbar=qbar,
        converged=r.termination == api.Termination.converged,
        nit=sum(phase.iterations for phase in r.phases),
        time_elapsed=elapsed,
        warm_started=warm_started,
        maxiter=int(maxiter),
        distribution=distribution,
        hessian=np.array(r.hessian) / t_count if r.hessian.size else None,  # of the per-row average, as DCCResult
        opg=np.array(r.opg) / t_count if r.opg.size else None,
        cov_mle=covariance,
        cov_robust=robust,
        _eps=eps,
        success=bool(r.success),
        inference=r.inference.name,
        kkt_residual=float(r.kkt_residual),
        warnings=tuple(r.warnings),
        phases=tuple(phase.name for phase in r.phases),
        _native_model=model,
    )
