"""The m-out-of-n parametric bootstrap of a fitted univariate model.

Each replication continues the fitted model from the end of the sample for m
observations (native simulation from the filtered state, so integrated and
nonstationary declarations work too) and refits the specification from the
estimate. The law of sqrt(m) (theta* - theta_hat) then stands in for that of
sqrt(n) (theta_hat - theta_0). The refits respect the domain, so an estimate
on a boundary keeps its boundary law, which the ordinary (m = n) bootstrap
does not reproduce (Andrews, 2000); m must grow more slowly than n.
"""

from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Optional

import numpy as np
from numpy.typing import NDArray

from . import _scivolapi as api
from ._evaluation import innovations

if TYPE_CHECKING:
    from .result import EstimationResult


@dataclass(frozen=True)
class BootstrapResult:
    """Bootstrap draws of the estimates and the inference they give.

    `draws` holds the successful refits' estimates (rows) in the order of
    `param_names`; `failures` counts replications whose refit raised or did not
    pass its first-order check.
    """

    param_names: tuple[str, ...]
    estimate: NDArray[np.float64]
    draws: NDArray[np.float64]
    m: int
    n: int
    replications: int
    failures: int
    lower: NDArray[np.float64]  # the parameters' bounds (NaN: none)
    upper: NDArray[np.float64]

    @property
    def _roots(self) -> NDArray[np.float64]:
        """sqrt(m) (theta* - theta_hat), the stand-in for sqrt(n) (theta_hat - theta_0)."""
        return np.sqrt(self.m) * (self.draws - self.estimate)

    @property
    def std_errors(self) -> NDArray[np.float64]:
        """The standard deviation of the bootstrap law, scaled to the sample size n."""
        return np.std(self._roots, axis=0, ddof=1) / np.sqrt(self.n)

    def intervals(self, level: float = 0.95) -> NDArray[np.float64]:
        """Basic bootstrap intervals (rows: parameters; columns: lower, upper):
        theta_hat - q(1 - a/2) / sqrt(n) to theta_hat - q(a/2) / sqrt(n), with q
        the quantiles of sqrt(m) (theta* - theta_hat), cut at the parameter's
        bounds. Their coverage holds at each parameter point, including one on
        a bound; it is not uniform over points near a bound."""
        if not 0.0 < level < 1.0:
            raise ValueError("level must lie strictly between 0 and 1")
        tail = 0.5 * (1.0 - level)
        q = np.quantile(self._roots, [tail, 1.0 - tail], axis=0)
        low = self.estimate - q[1] / np.sqrt(self.n)
        high = self.estimate - q[0] / np.sqrt(self.n)
        low = np.where(np.isnan(self.lower), low, np.maximum(low, self.lower))
        high = np.where(np.isnan(self.upper), high, np.minimum(high, self.upper))
        return np.column_stack([low, high])

    def to_dict(self) -> dict[str, Any]:
        return {"param_names": list(self.param_names), "estimate": self.estimate.copy(), "draws": self.draws.copy(),
                "m": self.m, "n": self.n, "replications": self.replications, "failures": self.failures,
                "std_errors": self.std_errors}


# Bickel and Sakov's (2008) rule: the bootstrap laws on m_j = n q^j, from the
# first below n down to `smallest`, and the m_j whose law is closest to the next.
GRID_RATIO = 0.75
SMALLEST = 250


def size_grid(n: int) -> list[int]:
    """n q^j for j = 1, 2, ... while at least 250 observations (just n for smaller samples)."""
    grid, j = [], 1
    while round(n * GRID_RATIO ** j) >= SMALLEST:
        grid.append(int(round(n * GRID_RATIO ** j)))
        j += 1
    return grid or [n]


def law_distance(a: NDArray[np.float64], b: NDArray[np.float64]) -> float:
    """The largest Kolmogorov distance between two samples' columns."""
    distance = 0.0
    for k in range(a.shape[1]):
        x, y = np.sort(a[:, k]), np.sort(b[:, k])
        grid = np.concatenate([x, y])
        gap = np.abs(np.searchsorted(x, grid, side="right") / len(x) - np.searchsorted(y, grid, side="right") / len(y))
        distance = max(distance, float(gap.max()))
    return distance


def bootstrap(result: "EstimationResult", replications: int = 499, *, m: Optional[int] = None, seed: Any = 0,
              n_jobs: Optional[int] = None) -> BootstrapResult:
    model, y, x = result._native()
    theta = result._theta
    n = int(result.fit_info.likelihood_n_obs or result.n_obs)
    if replications < 2:
        raise ValueError("replications must be at least 2")
    sizes = size_grid(n) if m is None else [int(m)]
    if not all(1 <= size <= n for size in sizes):
        raise ValueError(f"m must lie between 1 and the sample size {n}")
    longest = max(sizes)
    if x is not None and longest > x.shape[0]:
        raise ValueError("m exceeds the regressor rows available for the replications")
    density = model.innovation_density(theta)
    seeds = np.random.default_rng(seed).integers(0, 2**63, size=replications)
    method = str(result.method).lower()

    # Each replication draws its innovations once; every m on the grid uses
    # their first m (common random numbers across the grid). Regressors are
    # held at the sample's first m rows.
    def replicate(child_seed: int) -> list[Optional[NDArray[np.float64]]]:
        z = innovations(density, 1, longest, int(child_seed))
        estimates = []
        for size in sizes:
            future = None if x is None else np.ascontiguousarray(x[:size])
            series = api.simulate(model, theta, y, np.ascontiguousarray(z[:, :size]), x, future)["y"][0]
            try:
                refit = result.spec.fit(series, x=future, method=method, initial_params=theta)
                estimates.append(refit._theta if refit.success else None)
            except Exception:  # noqa: BLE001 - a failed refit is counted, not raised
                estimates.append(None)
        return estimates

    workers = min(4, os.cpu_count() or 1) if n_jobs in (None, -1) else int(n_jobs)
    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        outcomes = list(pool.map(replicate, seeds))
    laws = [np.array([o[j] for o in outcomes if o[j] is not None]).reshape(-1, theta.size) for j in range(len(sizes))]
    chosen = 0
    if len(sizes) > 1:
        roots = [np.sqrt(size) * (law - theta) for size, law in zip(sizes, laws)]
        distances = [law_distance(roots[j], roots[j + 1]) if min(len(roots[j]), len(roots[j + 1])) >= 2 else np.inf
                     for j in range(len(sizes) - 1)]
        chosen = int(np.argmin(distances))
    draws = laws[chosen]
    if len(draws) < 2:
        raise RuntimeError("fewer than two bootstrap refits succeeded")
    lower, upper = (np.asarray(b, dtype=np.float64) for b in model.bounds())
    return BootstrapResult(param_names=result.param_names, estimate=theta.copy(), draws=draws, m=sizes[chosen], n=n,
                           replications=replications, failures=replications - len(draws), lower=lower, upper=upper)
