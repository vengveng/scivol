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
from .dcc import DCCParams, DCCResult


@dataclass
class NativeDCCResult(DCCResult):
    """A DCC result whose estimate, inference and correlation path come from scivolapi."""

    success: bool = False
    inference: str = ""
    kkt_residual: float = float("nan")
    warnings: tuple[str, ...] = ()
    phases: tuple[str, ...] = ()
    _native_model: Any = field(default=None, repr=False)

    @property
    def Rt(self) -> NDArray[np.float64]:
        if self._Rt_cache is None:
            jet = api.CorrelationJet(self._native_model, self._eps, self.Qbar)
            out = jet.theta(np.ascontiguousarray(self.theta), api.PATHS)
            if out["status"] != "ok":
                raise ValueError(f"The correlation path is not available ({out['status']}).")
            object.__setattr__(self, "_Rt_cache", np.asarray(out["correlations"]))
        return self._Rt_cache


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
