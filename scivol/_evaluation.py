"""Fixed-parameter evaluation, filtering, forecasting and simulation.

The likelihood, its derivatives, the filtered paths, forecasts, distribution
functions and simulated paths all come from the native library; this module
translates arguments, packs parameters and draws the uniforms that simulations
invert through the native quantile functions.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable, Optional

import numpy as np
from numpy.typing import NDArray

from . import _scivolapi as api
from .roles import Role

if TYPE_CHECKING:
    from .result import EstimationResult


# --------------------------------------------------------------------------- #
# Results
# --------------------------------------------------------------------------- #

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
    eta: Optional[float] = None
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
        return api.standardized_residuals(np.ascontiguousarray(self.residuals), np.ascontiguousarray(self.sigma2))

    def to_dict(self) -> dict[str, Any]:
        return {
            "log_likelihood": self.log_likelihood,
            "mean": self.mean.copy(),
            "residuals": self.residuals.copy(),
            "sigma2": self.sigma2.copy(),
            "std_resid": self.std_resid.copy(),
            "distribution": self.distribution,
            "eta": self.eta,
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
    """Forecasts of the mean, its variance and the residual variance h steps ahead.

    One step ahead y_{T+1} = mean + sqrt(h_{T+1}) z exactly, so its quantiles are
    the innovation density's. Beyond one step y_{T+k} mixes over the random
    path of h and the mean's lags; its quantiles are those of simulated paths.
    """

    mean: NDArray[np.float64]
    variance: NDArray[np.float64]
    residual_variance: NDArray[np.float64]
    distribution: str
    eta: Optional[float] = None
    nu: Optional[float] = None
    lam: Optional[float] = None
    rho: Optional[float] = None
    zeta: Optional[float] = None
    shape_lambda: Optional[float] = None
    density: Any = None  # api.InnovationDensity
    simulated_paths: Optional[Callable[[], NDArray[np.float64]]] = None

    def __getitem__(self, key: str) -> NDArray[np.float64]:
        if key in ("mean", "variance", "residual_variance", "sigma2", "volatility"):
            return getattr(self, key)
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
        return api.square_roots(np.ascontiguousarray(self.variance).ravel()).reshape(self.variance.shape)

    @property
    def residual_volatility(self) -> NDArray[np.float64]:
        return api.square_roots(np.ascontiguousarray(self.residual_variance).ravel()).reshape(self.residual_variance.shape)

    def quantile(self, level: float | NDArray[np.float64]) -> NDArray[np.float64]:
        levels = np.asarray(level, dtype=np.float64)
        flat = np.atleast_1d(levels)
        if np.any((flat <= 0.0) | (flat >= 1.0)):
            raise ValueError("Quantile levels must lie strictly between 0 and 1.")
        out = np.empty((len(self.mean), flat.size))
        out[0] = api.location_scale_quantiles(float(self.mean[0]), float(self.residual_variance[0]),
                                              self.density, np.ascontiguousarray(flat))
        if len(self.mean) > 1:
            paths = self.simulated_paths()
            for horizon in range(1, len(self.mean)):
                out[horizon] = api.empirical_quantiles(np.ascontiguousarray(paths[:, horizon]),
                                                      np.ascontiguousarray(flat))
        return out[:, 0] if levels.ndim == 0 else out

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
            "eta": self.eta,
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


# --------------------------------------------------------------------------- #
# Parameters
# --------------------------------------------------------------------------- #

def component_slices(spec: Any) -> list[tuple[Any, slice]]:
    """Each component's entries of theta in the native layout [mean | volatility | shapes]:
    the mean's block first, the density's last, the volatility's in between (a
    mean-only model's sigma2 lies in between and belongs to no component)."""
    from ._fit import native_spec

    mean, volatility, density = (spec.get_component(role) for role in (Role.MEAN, Role.VOLATILITY, Role.DENSITY))
    native = native_spec(spec, int(getattr(mean, "n_exog", 0) or 0))
    if native is None:
        raise NotImplementedError(f"{spec} has no native model")
    dimension = api.Model(native).dimension
    first = 0 if mean is None else mean.n_params
    last = dimension - (0 if density is None else density.n_params)
    slices = []
    if mean is not None:
        slices.append((mean, slice(0, first)))
    if volatility is not None:
        slices.append((volatility, slice(first, last)))
    if density is not None:
        slices.append((density, slice(last, dimension)))
    return slices


def unpack_components(spec: Any, theta: NDArray[np.float64]) -> None:
    """Store theta in the components' fitted parameters."""
    for component, entries in component_slices(spec):
        component.unpack(theta[entries])


_DISPLAY = {"phi": "ar", "theta": "ma", "x": "exog", "c": "const", "lambda": "lam"}


def parameters(spec: Any, model: "api.Model", params: Any) -> NDArray[np.float64]:
    """theta in the model's order from an array, a mapping by parameter name
    ("alpha[1]", or "alpha" with one value per lag), or None (the components'
    fitted values)."""
    names = list(model.parameter_names)
    if params is None:
        theta = np.zeros(len(names))
        for component, entries in component_slices(spec):
            fitted = getattr(component, "fitted_params", None)
            if not fitted and component.n_params:
                raise ValueError("Parameters are required when the specification has not been fitted.")
            if component.n_params:
                theta[entries] = np.asarray(component.pack(fitted), dtype=np.float64)
        if spec.get_component(Role.VOLATILITY) is None:
            raise ValueError("A mean-only model needs explicit parameters: sigma2 has no component.")
    elif isinstance(params, Mapping):
        flat: dict[str, float] = {}
        for key, value in params.items():
            base, bracket, rest = str(key).partition("[")
            base = _DISPLAY.get(base, base)
            if base == "lam" and "lam" not in names:
                base = "lambda"
            values = np.atleast_1d(np.asarray(value, dtype=np.float64))
            if bracket or values.size == 1 and f"{base}[1]" not in names:
                flat[base + bracket + rest] = float(values[0])
            else:
                for i, v in enumerate(values, start=1):
                    flat[f"{base}[{i}]"] = float(v)
        missing = [name for name in names if name not in flat]
        if missing:
            raise ValueError(f"Parameters missing: {', '.join(missing)}.")
        unknown = [name for name in flat if name not in names]
        if unknown:
            raise ValueError(f"Unknown parameters: {', '.join(unknown)}.")
        theta = np.array([flat[name] for name in names])
    else:
        theta = np.asarray(params, dtype=np.float64).ravel()
    if theta.size != len(names):
        raise ValueError(f"Expected {len(names)} parameters ({', '.join(names)}), received {theta.size}.")
    return np.ascontiguousarray(theta)


# --------------------------------------------------------------------------- #
# Densities and innovations
# --------------------------------------------------------------------------- #

def density_name(spec: Any) -> str:
    density = spec.get_component(Role.DENSITY)
    return "Normal" if density is None else density.signature


def _shapes(model: "api.Model", theta: NDArray[np.float64]) -> dict[str, Optional[float]]:
    """Density shapes by name, as the result containers report them (fixed ones included)."""
    named = model.innovation_density(theta).named
    return {
        "eta": named.get("eta"),
        "nu": named.get("nu"),
        "lam": named.get("lam"),
        "rho": named.get("rho"),
        "zeta": named.get("zeta"),
        "shape_lambda": named.get("lambda"),
    }


def uniforms(rng: np.random.Generator, size: Any) -> NDArray[np.float64]:
    """Uniform draws on the midpoints of a 2^-53 grid: strictly inside (0, 1)."""
    return (rng.integers(0, 2 ** 53, size=size) + 0.5) * 2.0 ** -53


def innovations(density: "api.InnovationDensity", paths: int, horizon: int, seed: Any) -> NDArray[np.float64]:
    """paths x horizon standardized innovations by inversion of the native distribution function."""
    u = uniforms(np.random.default_rng(seed), paths * horizon)
    return np.ascontiguousarray(api.quantiles(density.density, density.shape, u).reshape(paths, horizon))


def regressors(x: Any, rows: int, columns: Optional[int] = None) -> Optional[NDArray[np.float64]]:
    if x is None:
        return None
    array = np.asarray(x, dtype=np.float64)
    if array.ndim == 1:
        array = array[:, None]
    if array.shape[0] != rows or (columns is not None and array.shape[1] != columns):
        raise ValueError(f"x must have shape ({rows}, {columns if columns is not None else 'k'}).")
    return np.ascontiguousarray(array)


# --------------------------------------------------------------------------- #
# Fixed-parameter evaluation
# --------------------------------------------------------------------------- #

class _Bound:
    """A spec at fixed parameters on the rows fit() would use."""

    def __init__(self, spec: Any, data: Any, params: Any, x: Any = None, hold_back: int = 0,
                 scale: Optional[float] = None) -> None:
        from ._fit import prepare_sample

        self.sample = prepare_sample(spec, data, x, hold_back, scale)
        self.y, self.x = self.sample.y, self.sample.x
        self.model = api.Model(self.sample.native)
        self.theta = parameters(spec, self.model, params)
        self.jet = api.Jet(self.model, self.y, self.x)

    def evaluate(self, request: int) -> dict[str, Any]:
        out = self.jet.theta(self.theta, request)
        if out["status"] != "ok":
            raise ValueError(f"The likelihood is not available at these parameters ({out['status']}).")
        return out


def loglikelihood(spec: Any, data: Any, params: Any, *, x: Any = None, hold_back: int = 0,
                  scale: Optional[float] = None) -> float:
    return -float(_Bound(spec, data, params, x, hold_back, scale).evaluate(api.VALUE)["value"])


def score(spec: Any, data: Any, params: Any, *, x: Any = None, hold_back: int = 0,
          scale: Optional[float] = None) -> NDArray[np.float64]:
    return -np.asarray(_Bound(spec, data, params, x, hold_back, scale).evaluate(api.GRADIENT)["gradient"])


def hessian(spec: Any, data: Any, params: Any, *, x: Any = None, hold_back: int = 0,
            scale: Optional[float] = None) -> NDArray[np.float64]:
    return -np.asarray(_Bound(spec, data, params, x, hold_back, scale).evaluate(api.HESSIAN)["hessian"])


def filter_state(spec: Any, data: Any, params: Any, *, x: Any = None, hold_back: int = 0,
                 scale: Optional[float] = None) -> FilteredState:
    bound = _Bound(spec, data, params, x, hold_back, scale)
    return state(spec, bound.model, bound.jet, bound.y, bound.theta)


def fixed_result(spec: Any, data: Any, params: Any, *, x: Any = None, hold_back: int = 0,
                 scale: Optional[float] = None, index: Any = None, name: Optional[str] = None) -> "EstimationResult":
    """A result at supplied parameters: the fitted paths, likelihood and Hessian without optimization."""
    from .result import FitMetadata, FixedResult

    bound = _Bound(spec, data, params, x, hold_back, scale)
    filtered = state(spec, bound.model, bound.jet, bound.y, bound.theta)
    unpack_components(spec, bound.theta)
    first = bound.sample.skip + bound.jet.first_scored
    result = FixedResult(
        spec, _SuppliedParameters(bound.theta.copy(), -filtered.log_likelihood), filtered.data,
        sigma2=filtered.sigma2.copy(), hessian=None if filtered.hessian is None else -filtered.hessian,
        fit_info=FitMetadata(likelihood_n_obs=len(filtered.data),
                             requested_hold_back=bound.sample.hold_back, effective_hold_back=first,
                             original_n_obs=bound.sample.original_n_obs, effective_n_obs=len(bound.y),
                             scale=bound.sample.scale),
        method="FIXED", parameter_source="supplied", index=None if index is None else index[first:], name=name,
    )
    result._resid = filtered.residuals.copy()
    result._fit_x = bound.x
    result.attach_sample(bound.model, bound.y, bound.x, bound.sample.skip)
    result._filtered_state_cache = filtered
    return result


class _SuppliedParameters:
    def __init__(self, x: NDArray[np.float64], fun: float) -> None:
        self.x = x
        self.fun = fun
        self.success = True
        self.nit = 0
        self.message = "Parameters supplied by user."


def state(spec: Any, model: "api.Model", jet: "api.Jet", y: NDArray[np.float64],
          theta: NDArray[np.float64]) -> FilteredState:
    """Paths, log-likelihood, score and Hessian of the log-likelihood on the scored rows."""
    out = jet.theta(theta, api.VALUE | api.GRADIENT | api.HESSIAN | api.PATHS)
    if not out["valid"] & api.PATHS:
        raise ValueError(f"The model cannot be filtered at these parameters ({out['status']}).")
    scored = y[jet.first_scored:]
    distribution = density_name(spec)
    return FilteredState(
        data=scored.copy(),
        mean=scored - out["residual"],
        residuals=np.asarray(out["residual"]),
        sigma2=np.asarray(out["variance"]),
        log_likelihood=-float(out["value"]),
        score=-np.asarray(out["gradient"]) if "gradient" in out else None,
        hessian=-np.asarray(out["hessian"]) if "hessian" in out else None,
        distribution=distribution,
        **_shapes(model, theta),
    )


# --------------------------------------------------------------------------- #
# Forecasts and simulations
# --------------------------------------------------------------------------- #

def forecast(spec: Any, model: "api.Model", theta: NDArray[np.float64], y: NDArray[np.float64],
             x: Optional[NDArray[np.float64]], horizon: int, future_x: Any = None, *,
             simulations: int = 10_000, seed: Any = 0) -> ForecastResult:
    """Forecasts after the sample (y, x).

    Exact for the mean, its variance given the residual variances, and the
    residual variances up to the model's exact variance horizon; variances beyond
    it and quantiles beyond one step average `simulations` paths drawn with `seed`.
    """
    if horizon <= 0:
        raise ValueError("horizon must be positive")
    columns = 0 if x is None else x.shape[1]
    future = regressors(future_x, horizon, columns) if columns else None
    if columns and future is None:
        raise ValueError(f"Forecasts need the exogenous regressors of the horizon: x with shape ({horizon}, {columns}).")
    distribution = density_name(spec)
    density = model.innovation_density(theta)
    def draws() -> NDArray[np.float64]:
        return innovations(density, int(simulations), horizon, seed)

    simulated = horizon > model.capabilities.exact_variance_horizon
    out = api.forecast(model, theta, y, horizon, x, future, draws() if simulated else None)
    cache: dict[str, NDArray[np.float64]] = {}

    def paths() -> NDArray[np.float64]:
        if "y" not in cache:
            cache["y"] = api.simulate(model, theta, y, draws(), x, future)["y"]
        return cache["y"]

    return ForecastResult(
        mean=np.asarray(out["mean"]),
        variance=np.asarray(out["variance"]),
        residual_variance=np.asarray(out["residual_variance"]),
        distribution=distribution,
        **_shapes(model, theta),
        density=density,
        simulated_paths=paths,
    )


def simulate(spec: Any, n_obs: int, params: Any, *, burn: int = 500, seed: Any = None,
             x: Any = None) -> SimulationResult:
    """A series of n_obs rows after `burn` discarded rows, started at the stationary values."""
    from ._fit import native_spec

    if n_obs <= 0:
        raise ValueError("n_obs must be positive")
    burn = max(int(burn), 0)
    total = n_obs + burn
    rows = regressors(x, total)
    mean = spec.get_component(Role.MEAN)
    if rows is not None and hasattr(mean, "set_n_exog"):
        mean.set_n_exog(rows.shape[1])
    native = native_spec(spec, 0 if rows is None else rows.shape[1])
    if native is None:
        raise NotImplementedError(f"{spec} cannot be simulated: unsupported component combination")
    model = api.Model(native)
    theta = parameters(spec, model, params)
    z = innovations(model.innovation_density(theta), 1, total, seed)
    out = api.simulate_series(model, theta, z, rows)
    keep = slice(burn, total)
    return SimulationResult(
        data=np.asarray(out["y"][0, keep]),
        mean=np.asarray(out["mean"][0, keep]),
        residuals=np.asarray(out["residual"][0, keep]),
        sigma2=np.asarray(out["variance"][0, keep]),
        innovations=np.asarray(z[0, keep]),
    )


def oos_losses(result: "EstimationResult", full_data: Any, *, full_x: Any, start: int, stop: int,
               loss: str) -> NDArray[np.float64]:
    """Losses of rows [start, stop) of `full_data` (indexed like the data passed to fit) at the fitted parameters.

    The sample is filtered from the fit's first row through `stop`. logscore is
    each row's negative log density (the likelihood's own terms); qlike and mse
    compare the squared residual with the conditional variance.
    """
    skip = result._native_skip
    scale = float(result.fit_info.scale or 1.0)
    y = np.ascontiguousarray(np.asarray(full_data, dtype=np.float64)[skip:stop] * scale)
    x = None if full_x is None else regressors(np.asarray(full_x, dtype=np.float64)[skip:stop], len(y))
    jet = api.Jet(result._native_model, y, x)
    first = skip + jet.first_scored
    if start < first or stop <= start:
        raise ValueError("The OOS slice must lie after the rows that condition the fit.")
    out = jet.theta(np.ascontiguousarray(result.params), api.TERMS | api.PATHS)
    if out["status"] != "ok":
        raise ValueError(f"The model cannot be filtered over the OOS slice ({out['status']}).")
    rows = slice(start - first, stop - first)
    key = loss.lower()
    if key == "logscore":
        return np.ascontiguousarray(out["terms"][rows])
    if key in {"qlike", "mse"}:
        return api.variance_losses(np.ascontiguousarray(out["residual"][rows]),
                                  np.ascontiguousarray(out["variance"][rows]), key == "qlike")
    raise ValueError("oos_loss must be 'logscore', 'qlike', or 'mse'.")
