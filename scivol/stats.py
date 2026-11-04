"""Statistical tests.

Tests of any series (`jarque_bera`, `ljung_box`, `arch_lm`), of probability
integral transforms (`dgt`, `berkowitz`), of forecast losses
(`diebold_mariano`, `model_confidence_set`) and of fitted models (`li_mak`,
`absolute_li_mak`, `sign_bias`, `absolute_sign_bias`, `wald`,
`likelihood_ratio`). Each test returns a `TestResult`; the statistics and
their reference distributions are computed by the native library.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Optional, Union

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from . import _scivolapi as api

if TYPE_CHECKING:
    from .result import EstimationResult

__all__ = [
    "TestResult", "Diagnostics", "ConfidenceSetResult",
    "jarque_bera", "ljung_box", "arch_lm", "dgt", "berkowitz", "sign_bias",
    "diebold_mariano", "model_confidence_set",
    "li_mak", "absolute_li_mak", "absolute_sign_bias", "wald", "likelihood_ratio",
]

_COMMON = ("statistic", "p_value", "df", "status", "log_p_value", "reject")


# =============================================================================
# RESULTS
# =============================================================================

@dataclass(frozen=True, repr=False)
class TestResult:
    """One test: its statistic, p-value and degrees of freedom, the `status` of
    the computation ("ok" when the statistic and its reference distribution
    were evaluated) and the natural log of the p-value, which stays finite when
    the p-value itself underflows. Fields particular to a test (Jarque-Bera's
    `skewness`, Ljung-Box's `autocorrelations`) are in `details` and also read
    as attributes."""

    __test__ = False  # not a pytest test class

    name: str
    statistic: float
    p_value: float
    df: Optional[float]
    status: str
    log_p_value: float
    details: dict = field(default_factory=dict)

    def rejects(self, level: float = 0.05) -> bool:
        """Whether the test rejects at `level`; False when it could not be evaluated."""
        if self.status == "probability_underflow":
            return True
        return bool(np.isfinite(self.p_value) and self.p_value <= level)

    def __getattr__(self, key: str) -> Any:
        details = object.__getattribute__(self, "details")
        if key in details:
            return details[key]
        raise AttributeError(f"{type(self).__name__} has no attribute or detail {key!r}")

    def __repr__(self) -> str:
        parts = [f"statistic={self.statistic:.6g}"]
        if self.df is not None:
            parts.append(f"df={self.df:g}")
        error = self.details.get("numerical_error", 0.0)
        if self.status == "probability_underflow":
            parts.append(f"p_value=0 (log p = {self.log_p_value:.1f})")
        elif error > 1e-12:
            parts.append(f"p_value={self.p_value:.4g} (simulated, error {error:.2g})")
        else:
            parts.append(f"p_value={self.p_value:.4g}")
        if self.status not in ("ok", "probability_underflow"):
            parts.append(f"status={self.status}")
        return f"{self.name}: " + ", ".join(parts)

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "statistic": self.statistic, "p_value": self.p_value, "df": self.df,
                "status": self.status, "log_p_value": self.log_p_value,
                **{k: (v.to_dict() if isinstance(v, TestResult) else v) for k, v in self.details.items()}}


def _from_native(name: str, values: Mapping[str, Any], drop: Sequence[str] = ()) -> TestResult:
    details = {key: value for key, value in values.items() if key not in _COMMON and key not in drop}
    df = values.get("df")
    p_value = 0.0 if values["status"] == "probability_underflow" else float(values["p_value"])
    return TestResult(name=name, statistic=float(values["statistic"]), p_value=p_value,
                      df=None if df is None else float(df), status=str(values["status"]),
                      log_p_value=float(values["log_p_value"]), details=details)


def _portmanteau(name: str, values: Mapping[str, Any]) -> TestResult:
    result = _from_native(name, values, drop=("q_stat", "box_pierce"))
    result.details["box_pierce"] = _from_native(name.replace("Ljung-Box", "Box-Pierce"), values["box_pierce"])
    return result


def _sign_bias(values: Mapping[str, Any], name: str = "Engle-Ng sign bias") -> TestResult:
    result = _from_native(name, values["joint"])
    for part, label in (("sign", "sign"), ("negative_size", "negative size"), ("positive_size", "positive size")):
        result.details[part] = _from_native(f"{name} ({label})", values[part])
    return result


def _series(values: Any, what: str = "the series") -> NDArray[np.float64]:
    arr = np.asarray(values, dtype=np.float64)
    if arr.ndim != 1:
        arr = arr.ravel() if arr.size == max(arr.shape, default=0) else arr
    if arr.ndim != 1:
        raise ValueError(f"{what} must be one-dimensional")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{what} has missing or infinite values; remove them first (for example with .dropna())")
    return np.ascontiguousarray(arr)


# =============================================================================
# TESTS OF A SERIES
# =============================================================================

def jarque_bera(x: Any) -> TestResult:
    """Jarque and Bera's (1987) test of zero skewness and kurtosis 3:
    n/6 (S^2 + (K - 3)^2/4) against chi-square(2). `skewness` and `kurtosis`
    (not excess) use moments about the sample mean."""
    return _from_native("Jarque-Bera", api.jarque_bera(_series(x)))


def ljung_box(x: Any, lags: int = 10, *, df_adjustment: int = 0) -> TestResult:
    """Ljung and Box's (1978) test of no autocorrelation up to `lags`, against
    chi-square(lags - df_adjustment). The autocorrelations use the full-sample
    mean and variance; `box_pierce` holds Box and Pierce's (1970) statistic."""
    return _portmanteau(f"Ljung-Box({lags})", api.portmanteau(_series(x), int(lags), int(df_adjustment)))


def arch_lm(x: Any, lags: int = 10) -> TestResult:
    """Engle's (1982) ARCH-LM test: (n - lags) R^2 from regressing x^2 on an
    intercept and `lags` of its own lags, against chi-square(lags)."""
    return _from_native(f"ARCH-LM({lags})", api.arch_lm(_series(x), int(lags)))


def sign_bias(standardized: Any, residual: Any = None) -> TestResult:
    """Engle and Ng's (1993) joint sign and size bias test, regressing squared
    standardized residuals on the lagged indicators I(e < 0), I(e < 0) e and
    I(e >= 0) e of `residual` (default: the standardized residuals). The
    individual tests are `sign`, `negative_size` and `positive_size`. For a
    fitted model, `sign_bias(result)` uses its residuals."""
    if hasattr(standardized, "std_resid") and hasattr(standardized, "resid"):
        standardized, residual = standardized.std_resid, standardized.resid
    z = _series(standardized, "the standardized residuals")
    e = z if residual is None else _series(residual, "the residuals")
    return _sign_bias(api.sign_bias(z, e))


# =============================================================================
# TESTS OF PROBABILITY INTEGRAL TRANSFORMS
# =============================================================================

def dgt(pit: Any, cells: int = 40) -> TestResult:
    """Diebold, Gunther and Tay's (1998) test that PITs are uniform: Pearson's
    statistic over `cells` equal cells, against chi-square(cells - 1)."""
    return _from_native(f"DGT({cells} cells)", api.dgt(_series(pit, "the PITs"), int(cells)),
                        drop=("chi2_stat", "n_cells"))


def berkowitz(pit: Any) -> TestResult:
    """Berkowitz's (2001) test that the Normal scores of the PITs are iid
    standard Normal, against a Gaussian AR(1) (likelihood conditional on the
    first observation), chi-square(3). Reports the fitted `intercept`,
    `variance` and `ar1`."""
    return _from_native("Berkowitz", api.berkowitz(_series(pit, "the PITs")), drop=("likelihood",))


# =============================================================================
# TESTS OF FORECAST LOSSES
# =============================================================================

def diebold_mariano(loss_a: Any, loss_b: Any, horizon: int = 1, bandwidth: Optional[int] = None) -> TestResult:
    """Diebold and Mariano's (1995) test of equal expected loss, with Harvey,
    Leybourne and Newbold's (1997) small-sample correction and a Student-t
    reference. A positive statistic means the losses of `a` are larger. The
    long-run variance is Bartlett's with `bandwidth` (default floor(n^(1/3)),
    at least horizon - 1)."""
    a, b = _series(loss_a, "loss_a"), _series(loss_b, "loss_b")
    if a.size != b.size:
        raise ValueError("the two loss series must have the same length")
    return _from_native("Diebold-Mariano",
                        api.diebold_mariano(a, b, int(horizon), -1 if bandwidth is None else int(bandwidth)))


@dataclass(frozen=True)
class ConfidenceSetResult:
    """Hansen, Lunde and Nason's (2011) model confidence set: the models kept
    (`included`, in the order of the loss columns' labels), those eliminated in
    order, and each model's step-down p-value."""

    included: tuple
    eliminated: tuple
    p_values: pd.Series
    confidence: float
    block_length: int
    statistic: str


def model_confidence_set(losses: Any, *, confidence: float = 0.90, block_length: Optional[int] = None,
                         draws: int = 4999, statistic: str = "max", seed: Any = None) -> ConfidenceSetResult:
    """The model confidence set of the columns of `losses` (rows: periods;
    columns: models, labelled when a DataFrame) by circular block bootstrap
    with `draws` resamples. `statistic` is "max" (T_max) or "R" (the range
    statistic)."""
    labels = list(losses.columns) if isinstance(losses, pd.DataFrame) else None
    matrix = np.ascontiguousarray(np.asarray(losses, dtype=np.float64))
    if matrix.ndim != 2 or matrix.shape[1] < 2:
        raise ValueError("losses must be a matrix with one column per model (at least two)")
    if not np.all(np.isfinite(matrix)):
        raise ValueError("losses has missing or infinite values")
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must lie strictly between 0 and 1")
    rows = matrix.shape[0]
    length = api.mcs_block_length(rows, -1 if block_length is None else int(block_length))
    uniforms = np.random.default_rng(seed).random((int(draws), (rows + length - 1) // length))
    out = api.model_confidence_set(matrix, uniforms, length, 1.0 - confidence, statistic)
    labels = labels or list(range(matrix.shape[1]))
    return ConfidenceSetResult(included=tuple(labels[i] for i in out["included"]),
                               eliminated=tuple(labels[i] for i in out["eliminated"]),
                               p_values=pd.Series(np.asarray(out["p_values"]), index=labels, name="p_value"),
                               confidence=confidence, block_length=length, statistic=statistic)


# =============================================================================
# TESTS OF FITTED MODELS
# =============================================================================

def _model_inputs(result: "EstimationResult") -> tuple[Any, NDArray[np.float64], Any, bool]:
    model, y, x = result._native()
    estimator = api.Estimator.qmle if str(result.method).lower() == "qmle" else api.Estimator.mle
    return api.Jet(model, y, x), result._theta, estimator, result.parameter_source != "supplied"


def li_mak(result: "EstimationResult", lags: int = 10) -> TestResult:
    """Li and Mak's (1994) test of no autocorrelation in squared standardized
    residuals up to `lags`, corrected for the estimation of every parameter
    (chi-square(lags)). Needs a finite fourth moment of the innovations."""
    jet, theta, estimator, estimated = _model_inputs(result)
    return _from_native(f"Li-Mak({lags})", api.li_mak(jet, theta, int(lags), estimator, estimated),
                        drop=("covariance", "derivative"))


def absolute_li_mak(result: "EstimationResult", lags: int = 10) -> TestResult:
    """The Li-Mak test on absolute standardized residuals; needs only finite
    second moments of the innovations under maximum likelihood."""
    jet, theta, estimator, estimated = _model_inputs(result)
    return _from_native(f"absolute Li-Mak({lags})", api.absolute_li_mak(jet, theta, int(lags), estimator, estimated),
                        drop=("covariance", "derivative"))


def absolute_sign_bias(result: "EstimationResult") -> TestResult:
    """The joint sign and size bias test on absolute standardized residuals,
    corrected for parameter estimation (chi-square(3))."""
    jet, theta, estimator, estimated = _model_inputs(result)
    return _from_native("absolute sign bias", api.absolute_sign_bias(jet, theta, estimator, estimated))


def wald(result: "EstimationResult", names: Union[str, Sequence[str]], values: Any = 0.0, *,
         robust: bool = False) -> TestResult:
    """Wald test that the parameters `names` equal `values` (default zero),
    from the estimate's covariance (`robust=True`: the sandwich covariance),
    against chi-square(len(names)).

    The chi-square reference needs the hypothesized values inside the domain.
    For a value on a bound, such as the Student-t's eta = 0 (the Normal), the
    status is "at_boundary"; `likelihood_ratio` uses the boundary law there."""
    names = [names] if isinstance(names, str) else list(names)
    if not names:
        raise ValueError("name at least one parameter")
    unknown = [n for n in names if n not in result.param_names]
    if unknown:
        raise ValueError(f"unknown parameters {', '.join(unknown)}; the model's are {', '.join(result.param_names)}")
    target = np.broadcast_to(np.asarray(values, dtype=np.float64), (len(names),)).copy()
    label = f"Wald({', '.join(names)})"
    lower, upper = (np.asarray(b, dtype=np.float64) for b in result._native()[0].bounds())
    index = [result.param_names.index(n) for n in names]
    if np.any((target == lower[index]) | (target == upper[index])):
        return TestResult(label, np.nan, np.nan, float(len(names)), "at_boundary", np.nan,
                          {"message": "a hypothesized value is on a bound of the domain; "
                                      "use stats.likelihood_ratio, which uses the boundary law"})
    covariance = result.cov_robust if robust else result.cov_matrix
    if covariance is None:
        kind = "robust covariance" if robust else "covariance"
        raise ValueError(f"the result has no {kind} (inference status: {result.fit_info.inference_status})")
    block = covariance.loc[names, names].to_numpy()
    if not np.all(np.isfinite(block)):
        raise ValueError("a binding face fixes some of these parameters, so they have no standard error")
    difference = np.ascontiguousarray(result.params[names].to_numpy() - target)
    return _from_native(label, api.wald(difference, np.ascontiguousarray(block)))


def likelihood_ratio(restricted: "EstimationResult", unrestricted: "EstimationResult", *,
                     level: float = 0.05, max_steps: int = 8) -> TestResult:
    """Likelihood-ratio test of a model against a larger model that nests it.

    The restriction is found among the larger model's nested restrictions (any
    chain of them up to `max_steps`), and its reference law follows from the
    restriction: chi-square for interior equalities; for parameters fixed at a
    bound of the domain (the Student-t's eta = 0, a lag's coefficient at zero),
    the chi-bar-square mixture or the cone projection law. `law`, `weights`
    and `restriction` are in the details. Both fits must be maximum likelihood
    estimates on the same data. Both are refitted from their estimates on the
    larger model's rows, as in model selection, so a restricted model with
    shorter mean lags is compared on the same sample (`log_likelihood` and
    `restricted_log_likelihood` report the two fits). Mixture weights are
    computed by cubature, and when other parameters also sit on bounds the law
    is simulated (`law` "cone_projection"); `numerical_error` bounds the
    p-value's error in both cases, and a p-value within it of `level` has
    the status "unresolved"."""
    for r in (restricted, unrestricted):
        if str(r.method).lower() != "mle" or r.parameter_source == "supplied":
            raise ValueError("likelihood_ratio needs two maximum likelihood fits")
        if getattr(r, "_native_fit", None) is None:
            raise ValueError("likelihood_ratio needs results returned by fit()")
    parent, y, x = unrestricted._native()
    child, y_child, x_child = restricted._native()
    if (restricted.fit_info.original_n_obs != unrestricted.fit_info.original_n_obs
            or restricted.fit_info.scale != unrestricted.fit_info.scale
            or not np.array_equal(_tail(y, len(y_child)), _tail(y_child, len(y)))):
        raise ValueError("the two models were fitted to different data")
    if restricted.fit_info.effective_hold_back > unrestricted.fit_info.effective_hold_back:
        raise ValueError("the restricted model scores fewer rows than the larger model; fit both with the same "
                         "hold_back")
    restriction = _restriction_path(parent, child, max_steps)
    if restriction is None:
        raise ValueError(f"{restricted.spec} is not a restriction of {unrestricted.spec}")
    # Both models are fitted on one comparison plan (the larger model's rows),
    # from their estimates; on the same rows this reproduces the ordinary fits.
    from ._fit import fit_spec_on_plan

    plan = api.ComparisonPlan(y, parent.spec.mean.conditioning, x, None)
    fits = []
    for result, model in ((unrestricted, parent), (restricted, child)):
        options = api.FitOptions()
        options.start = [float(v) for v in result._theta]
        fits.append(fit_spec_on_plan(result.spec, plan, options=options, exog=model.spec.mean.exog))
    test = api.restriction_test(parent, child, restriction, fits[0], fits[1], plan, float(level))
    status = "ok" if test.status == api.LRStatus.ok else test.status.name
    law = test.law.name
    p_value = float(test.p_value) if np.isfinite(test.p_value) else np.nan
    details = {"law": law, "weights": tuple(test.weights), "interior_df": int(test.interior_df),
               "numerical_error": float(test.numerical_error), "restriction": restriction.name, "log_likelihood": float(fits[0].log_likelihood),
               "restricted_log_likelihood": float(fits[1].log_likelihood), "message": test.message}
    return TestResult(name=f"LR({restriction.name})", statistic=float(test.statistic), p_value=p_value,
                      df=float(test.interior_df) if law == "chi_square" else None, status=status,
                      log_p_value=float(np.log(p_value)) if p_value > 0 else np.nan, details=details)


def _tail(values: NDArray[np.float64], length: int) -> NDArray[np.float64]:
    return values[len(values) - min(length, len(values)):]


def _restriction_path(parent: Any, child: Any, max_steps: int) -> Any:
    """The chain of nested restrictions from `parent` to `child`, composed, or None."""
    target = child.key
    frontier = [(parent, None)]
    seen = {parent.key}
    for _ in range(max_steps):
        following = []
        for model, joint in frontier:
            for restriction in model.restrictions():
                nested = api.Model(restriction.child)
                combined = restriction if joint is None else joint.then(restriction)
                if nested.key == target:
                    return combined
                if nested.key not in seen:
                    seen.add(nested.key)
                    following.append((nested, combined))
        frontier = following
        if not frontier:
            return None
    return None


# =============================================================================
# THE DIAGNOSTIC BATTERY OF A FITTED MODEL
# =============================================================================

class Diagnostics:
    """Specification tests of a fitted model on its scored sample.

    Residual tests: `residual_ljung_box`, `li_mak`, `sign_bias`. PIT tests
    under the fitted density: `dgt`, `pit_ljung_box` (keys 1 to 4, the moments
    (u - 1/2)^r), `pit_normal_ljung_box` and `berkowitz`. The PIT tests use a
    fixed-parameter reference, which in-sample estimation does not calibrate.
    `print(diagnostics)` shows the report; `to_frame()` gives one row per test.
    """

    def __init__(self, values: Mapping[str, Any], distribution: str, index: Any = None) -> None:
        lags = values["residual_ljung_box"].get("lags", "")
        self.residual_ljung_box = _portmanteau(f"Ljung-Box({lags})", values["residual_ljung_box"])
        self.li_mak = _from_native(f"Li-Mak({values['li_mak'].get('lags', '')})", values["li_mak"],
                                   drop=("covariance", "derivative"))
        self.sign_bias = _sign_bias(values["sign_bias"])
        self.dgt = _from_native(f"DGT({values['dgt'].get('n_cells', '')} cells)", values["dgt"],
                                drop=("chi2_stat", "n_cells"))
        self.pit_ljung_box = {power: _portmanteau(f"PIT (u-1/2)^{power} Ljung-Box({lags})", test)
                              for power, test in values["ljung_box"].items()}
        self.pit_normal_ljung_box = _portmanteau(f"PIT Normal-score Ljung-Box({lags})", values["pit_normal_ljung_box"])
        self.berkowitz = _from_native("Berkowitz", values["berkowitz"], drop=("likelihood",))
        pit = np.asarray(values["pit"])
        self.pit = pit if index is None else pd.Series(pit, index=index, name="pit")
        self.n_obs = int(values["n_obs"])
        self.distribution = distribution
        self.shapes = dict(values.get("shapes", {}))
        self.pit_calibration = str(values["pit_calibration"])

    def tests(self) -> list[TestResult]:
        """Every test, in report order."""
        return [self.residual_ljung_box, self.li_mak, self.sign_bias, self.dgt,
                *self.pit_ljung_box.values(), self.pit_normal_ljung_box, self.berkowitz]

    def to_frame(self) -> pd.DataFrame:
        """One row per test: statistic, df, p-value and status."""
        return pd.DataFrame([{"test": t.name, "statistic": t.statistic, "df": t.df, "p_value": t.p_value,
                              "status": t.status} for t in self.tests()]).set_index("test")

    def __str__(self) -> str:
        width = 74
        shapes = ", ".join(f"{name}={value:.6g}" for name, value in self.shapes.items())
        lines = ["=" * width, f"{'Model Diagnostics':^{width}}", "=" * width,
                 f"Distribution:  {self.distribution}" + (f" ({shapes})" if shapes else ""),
                 f"Observations:  {self.n_obs}", "-" * width,
                 f"{'Test':<38} {'Statistic':>11} {'df':>5} {'p-value':>10}  Status", "-" * width]
        for t in self.tests():
            df = "" if t.df is None else f"{t.df:g}"
            p = "<1e-300" if t.status == "probability_underflow" else f"{t.p_value:.4f}"
            status = "" if t.status in ("ok", "probability_underflow") else t.status
            lines.append(f"{t.name:<38} {t.statistic:>11.3f} {df:>5} {p:>10}  {status}")
        lines += ["-" * width, "PIT p-values use a fixed-parameter reference distribution.", "=" * width]
        return "\n".join(lines)

    def _repr_pretty_(self, printer: Any, cycle: bool) -> None:
        printer.text(str(self))

    def __repr__(self) -> str:
        return str(self)
