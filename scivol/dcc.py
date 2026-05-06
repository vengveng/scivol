"""
scivol/dcc.py
=============
DCC-GARCH (Dynamic Conditional Correlation) model.

Two-step estimation:
  Step 1: Estimate univariate GARCH-type models for each series.
  Step 2: Estimate DCC parameters from the standardised residuals
          using Gaussian correlation likelihood.

Usage::

    from scivol import DCC, GARCH, Normal

    dcc = DCC(p=1, q=1)
    result = dcc.fit(returns, univariate_spec=GARCH(1,1) + Normal())

    # Or pass pre-computed standardised residuals:
    result = dcc.fit_from_residuals(std_resid)

    # Access results:
    print(result.params)            # DCCParams(a=..., b=...)
    print(result.Rt)                # time-varying correlations (T, N, N)
    print(result.corr("stock", "bond"))   # pairwise correlation series
    print(result.unconditional_corr)      # long-run correlation matrix
    print(result.std_errors)        # MLE standard errors
    print(result.std_errors_robust) # Sandwich (robust) standard errors
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any, Union, TYPE_CHECKING
import time
import numpy as np
from numpy.typing import NDArray
from scipy.optimize import minimize

from . import _dcc_kernels as _dk

if TYPE_CHECKING:
    from .spec import CompositeSpec
    from .result import EstimationResult


# =============================================================================
# Parameter container
# =============================================================================

@dataclass
class DCCParams:
    """Container for DCC parameters."""
    a: NDArray[np.float64]   # length p
    b: NDArray[np.float64]   # length q

    @property
    def persistence(self) -> float:
        return float(np.sum(self.a) + np.sum(self.b))

    def __repr__(self) -> str:
        a_str = ", ".join(f"{x:.6f}" for x in self.a)
        b_str = ", ".join(f"{x:.6f}" for x in self.b)
        return f"DCCParams(a=[{a_str}], b=[{b_str}], persistence={self.persistence:.6f})"


# =============================================================================
# DCC recursion (pure Python, for post-estimation extraction)
# =============================================================================

def _dcc_recursion(
    theta: NDArray[np.float64],
    eps: NDArray[np.float64],
    Qbar: NDArray[np.float64],
    p: int, q: int,
) -> NDArray[np.float64]:
    """
    Run the DCC(p,q) recursion and return time-varying correlation matrices.

    Called once after estimation for result extraction — not performance-critical.

    Returns
    -------
    Rt : (T, N, N)  normalised correlation matrices
    """
    T, N = eps.shape
    a_arr = theta[:p]
    b_arr = theta[p:]
    c = 1.0 - float(np.sum(theta))

    Rt_out = np.empty((T, N, N), dtype=np.float64)

    maxp = max(p, 1)
    maxq = max(q, 1)
    Q_buf = [Qbar.copy() for _ in range(maxq)]
    e_buf = [np.zeros(N, dtype=np.float64) for _ in range(maxp)]

    for t in range(T):
        e = eps[t]
        Q = c * Qbar
        for i in range(p):
            ep = e_buf[i]
            Q = Q + a_arr[i] * np.outer(ep, ep)
        for j in range(q):
            Q = Q + b_arr[j] * Q_buf[j]

        diag = np.sqrt(np.maximum(np.diag(Q), 1e-12))
        Rt_out[t] = Q / np.outer(diag, diag)

        if p > 0:
            e_buf = [e.copy()] + e_buf[:p - 1]
        if q > 0:
            Q_buf = [Q.copy()] + Q_buf[:q - 1]

    return Rt_out


# =============================================================================
# DCC Result
# =============================================================================

@dataclass
class DCCResult:
    """Result of a DCC estimation."""

    params: DCCParams
    theta: NDArray[np.float64]          # flat [a_1..a_p, b_1..b_q]
    nll: float
    p: int
    q: int
    T: int
    N: int
    Qbar: NDArray[np.float64]           # (N, N)

    converged: bool
    nit: int
    time_elapsed: float

    # Standard errors
    hessian: Optional[NDArray[np.float64]] = None       # (K, K)
    opg: Optional[NDArray[np.float64]] = None           # (K, K)
    cov_mle: Optional[NDArray[np.float64]] = None       # (K, K)
    cov_robust: Optional[NDArray[np.float64]] = None    # (K, K)

    # Univariate results (if fitted in step 1)
    univariate_results: Optional[List[EstimationResult]] = None

    # Private: stored by fit methods, used for lazy R_t computation
    _eps: Optional[NDArray[np.float64]] = field(default=None, repr=False)
    _index: Optional[Any] = field(default=None, repr=False)
    _columns: Optional[List[str]] = field(default=None, repr=False)
    _Rt_cache: Optional[NDArray[np.float64]] = field(default=None, repr=False)

    # ── Scalars ─────────────────────────────────────────────────────────────

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

    # ── Standard errors ─────────────────────────────────────────────────────

    @property
    def std_errors(self) -> NDArray[np.float64]:
        """MLE standard errors from inverse Hessian."""
        if self.cov_mle is not None:
            return np.sqrt(np.maximum(np.diag(self.cov_mle), 0.0))
        return np.full(self.K, np.nan)

    @property
    def std_errors_robust(self) -> NDArray[np.float64]:
        """Sandwich (robust) standard errors."""
        if self.cov_robust is not None:
            return np.sqrt(np.maximum(np.diag(self.cov_robust), 0.0))
        return np.full(self.K, np.nan)

    # ── Time-varying matrices (lazy, computed once) ─────────────────────────

    def _ensure_recursion(self) -> None:
        """Run the DCC recursion and cache Rt array."""
        if self._Rt_cache is not None:
            return
        if self._eps is None:
            raise RuntimeError(
                "Standardised residuals not stored on this result. "
                "Use DCC.fit() or DCC.fit_from_residuals() to create results."
            )
        Rt = _dcc_recursion(self.theta, self._eps, self.Qbar, self.p, self.q)
        object.__setattr__(self, '_Rt_cache', Rt)

    @property
    def Rt(self) -> NDArray[np.float64]:
        """
        Time-varying correlation matrices (T, N, N).

        Computed lazily on first access and cached.
        Returns a numpy array regardless of input type.
        """
        self._ensure_recursion()
        return self._Rt_cache  # type: ignore[return-value]

    # ── Pairwise correlation access ─────────────────────────────────────────

    def _resolve_pair(
        self, series_i: Union[int, str], series_j: Union[int, str],
    ) -> Tuple[int, int]:
        """Resolve series identifiers to integer indices."""
        def _resolve(s: Union[int, str]) -> int:
            if isinstance(s, int):
                if s < 0 or s >= self.N:
                    raise IndexError(f"Series index {s} out of range [0, {self.N})")
                return s
            if self._columns is None:
                raise ValueError(
                    f"Cannot look up series by name '{s}' — no column names stored. "
                    "Pass a pandas DataFrame to DCC.fit() to enable name-based access."
                )
            try:
                return self._columns.index(s)
            except ValueError:
                raise KeyError(
                    f"Series '{s}' not found. Available: {self._columns}"
                ) from None
        return _resolve(series_i), _resolve(series_j)

    def corr(
        self, series_i: Union[int, str], series_j: Union[int, str],
    ) -> Any:
        """
        Extract the pairwise time-varying correlation ρ_{ij,t}.

        Parameters
        ----------
        series_i, series_j : int or str
            Series identifiers. Integers are column indices; strings are
            looked up from column names (requires pandas input to fit()).

        Returns
        -------
        pandas.Series if a pandas index is available, otherwise numpy array.
        """
        i, j = self._resolve_pair(series_i, series_j)
        rho = self.Rt[:, i, j]
        if self._index is not None:
            import pandas as pd
            i_name = self._columns[i] if self._columns else str(i)
            j_name = self._columns[j] if self._columns else str(j)
            return pd.Series(rho, index=self._index, name=f"rho_{i_name}_{j_name}")
        return rho

    @property
    def unconditional_corr(self) -> Any:
        """
        Unconditional correlation matrix (normalised Q̄).

        Returns
        -------
        pandas.DataFrame if column names are available, otherwise (N, N) numpy array.
        """
        d = np.sqrt(np.diag(self.Qbar))
        R_bar = self.Qbar / np.outer(d, d)
        if self._columns is not None:
            import pandas as pd
            return pd.DataFrame(R_bar, index=self._columns, columns=self._columns)
        return R_bar

    # ── Summary ─────────────────────────────────────────────────────────────

    def summary(self) -> str:
        """Print estimation summary."""
        K = self.K
        names = []
        for i in range(self.p):
            names.append(f"a[{i+1}]")
        for j in range(self.q):
            names.append(f"b[{j+1}]")

        se = self.std_errors
        se_r = self.std_errors_robust
        lines = []
        lines.append("=" * 72)
        lines.append(f"{'DCC(' + str(self.p) + ',' + str(self.q) + ') Estimation Results':^72}")
        lines.append("=" * 72)
        lines.append(f"  Obs: {self.T}  Series: {self.N}  "
                     f"Log-lik: {self.log_likelihood:.4f}  "
                     f"Converged: {self.converged}")
        lines.append(f"  Time: {self.time_elapsed:.4f}s  Iterations: {self.nit}")
        lines.append("-" * 72)
        lines.append(f"{'Param':<8} {'Estimate':>10} {'MLE SE':>10} "
                     f"{'Rob SE':>10} {'t-stat':>10}")
        lines.append("-" * 72)
        for i, nm in enumerate(names):
            ts = self.theta[i] / se[i] if se[i] > 0 else np.nan
            lines.append(f"{nm:<8} {self.theta[i]:>10.6f} {se[i]:>10.6f} "
                        f"{se_r[i]:>10.6f} {ts:>10.2f}")
        lines.append(f"\n  Persistence: {self.params.persistence:.6f}")
        lines.append("=" * 72)
        return "\n".join(lines)


# =============================================================================
# DCC class
# =============================================================================

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
        univariate_spec: Optional[CompositeSpec] = None,
        univariate_results: Optional[List[EstimationResult]] = None,
        **kwargs: Any,
    ) -> DCCResult:
        """
        Two-step DCC estimation.

        Parameters
        ----------
        returns : array-like (T, N)
            Multivariate return series.  Can be a numpy array or pandas
            DataFrame.  If pandas, column names and index are preserved
            on the result for convenient access.
        univariate_spec : CompositeSpec, optional
            Specification for univariate models (applied to each series).
            Default: GARCH(1,1) + Normal().
        univariate_results : list of EstimationResult, optional
            Pre-fitted univariate results. If provided, skip step 1.

        Returns
        -------
        DCCResult
        """
        # Extract pandas metadata before converting to numpy
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
        T, N = returns_np.shape

        # Step 1: Univariate models
        if univariate_results is not None:
            if len(univariate_results) != N:
                raise ValueError(f"Expected {N} univariate results, got {len(univariate_results)}")
            std_resid = np.column_stack([
                np.asarray(r.std_resid) for r in univariate_results
            ])
            uni_res = univariate_results
        else:
            if univariate_spec is None:
                from .components import GARCH, Normal
                univariate_spec = GARCH(1, 1) + Normal()
            uni_res = []
            std_resid_cols = []
            for i in range(N):
                r = univariate_spec.fit(returns_np[:, i])
                uni_res.append(r)
                std_resid_cols.append(np.asarray(r.std_resid))
            std_resid = np.column_stack(std_resid_cols)

        # Step 2: DCC estimation
        result = self.fit_from_residuals(std_resid, **kwargs)
        object.__setattr__(result, 'univariate_results', uni_res)
        object.__setattr__(result, '_index', index)
        object.__setattr__(result, '_columns', columns)
        return result

    def fit_from_residuals(
        self,
        eps: NDArray[np.float64],
        Qbar: Optional[NDArray[np.float64]] = None,
        theta0: Optional[NDArray[np.float64]] = None,
        compute_se: bool = True,
    ) -> DCCResult:
        """
        Fit DCC(p,q) from standardised residuals.

        Parameters
        ----------
        eps : array (T, N)
            Standardised residuals.
        Qbar : array (N, N), optional
            Unconditional second-moment matrix.  Computed from eps if None.
        theta0 : array (K,), optional
            Starting values.  Default: 0.05 for each parameter.
        compute_se : bool
            Whether to compute standard errors (Hessian + sandwich).
        """
        eps = np.ascontiguousarray(eps, dtype=np.float64)
        if eps.ndim != 2:
            raise ValueError("eps must be 2-D (T, N)")
        T, N = eps.shape
        p, q = self.p, self.q
        K = p + q

        if Qbar is None:
            Qbar = _dk.compute_qbar(eps)
        Qbar = np.ascontiguousarray(Qbar, dtype=np.float64)

        if theta0 is None:
            theta0 = np.full(K, 0.05, dtype=np.float64)

        # Bounds: each param in (1e-6, 0.999), sum < 1
        bounds = [(1e-6, 0.999)] * K
        constraints = [{"type": "ineq", "fun": lambda x: 0.9999 - np.sum(x)}]

        t0 = time.perf_counter()

        if K <= 2:
            # For K<=2, SLSQP's numerical FD is faster than analytical
            # gradient (FD needs K+1 cheap NLL calls vs 1 costly grad call).
            opt = minimize(
                fun=lambda x: _dk.dcc_nll(x, eps, Qbar, p, q),
                x0=theta0,
                method="SLSQP",
                bounds=bounds,
                constraints=constraints,
                options={"maxiter": 500, "ftol": 1e-12},
            )
        else:
            # For K>2, analytical gradient pays off.
            _cache: Dict[tuple, Tuple[float, NDArray[np.float64]]] = {}

            def _eval(x: NDArray[np.float64]) -> Tuple[float, NDArray[np.float64]]:
                key = tuple(x)
                if key not in _cache:
                    nll, grad = _dk.dcc_nll_grad(x, eps, Qbar, p, q)
                    _cache[key] = (nll, grad)
                    if len(_cache) > 20:
                        _cache.pop(next(iter(_cache)))
                return _cache[key]

            opt = minimize(
                fun=lambda x: _eval(x)[0],
                x0=theta0,
                method="SLSQP",
                jac=lambda x: _eval(x)[1],
                bounds=bounds,
                constraints=constraints,
                options={"maxiter": 500, "ftol": 1e-12},
            )
        elapsed = time.perf_counter() - t0
        theta_hat = opt.x.copy()

        # Standard errors
        hessian = opg = cov_mle = cov_robust = None
        if compute_se:
            nll_h, grad_h, hess_h, scores_h = _dk.dcc_nll_grad_hess(
                theta_hat, eps, Qbar, p, q, return_scores=True)
            hessian = hess_h
            opg = (scores_h.T @ scores_h) / T
            try:
                H_inv = np.linalg.inv(hess_h)
                cov_mle = H_inv / T
                cov_robust = (H_inv @ opg @ H_inv) / T
            except np.linalg.LinAlgError:
                pass

        params = DCCParams(
            a=theta_hat[:p].copy(),
            b=theta_hat[p:].copy(),
        )

        return DCCResult(
            params=params,
            theta=theta_hat,
            nll=float(opt.fun),
            p=p, q=q, T=T, N=N,
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
