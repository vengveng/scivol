# scivol/_autoselect.py
"""
Automatic model selection engine for scivol.

This module provides functionality to automatically select optimal model
specifications based on a user-supplied criterion callable or the default
blended AIC + diagnostic test criterion.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List, Tuple, Optional, Dict, Any, TYPE_CHECKING
import numpy as np
import warnings
from scipy.special import gammaln, ndtr

from ._gh import standardized_gh_logpdf
from ._vg import standardized_vg_logpdf

if TYPE_CHECKING:
    from .spec import CompositeSpec
    from .result import EstimationResult

# Type alias for the criterion callable
CriterionFunc = Callable[["EstimationResult", Optional[Dict[str, Any]]], float]
MeanCandidate = Tuple[str, Tuple[Any, ...]]


@dataclass
class ModelCandidate:
    """
    Container for a candidate model and its evaluation metrics.
    
    Attributes
    ----------
    spec : CompositeSpec
        The model specification.
    result : EstimationResult or None
        The estimation result (None if fitting failed).
    aic : float
        Akaike Information Criterion (lower is better).
    diagnostic_penalty : float
        Penalty from failed diagnostic tests (default criterion only).
    score : float
        Selection score from the criterion callable (lower is better).
    diagnostics : dict or None
        Raw output from ``result.diagnostic_tests()``, or None if
        diagnostics failed or were not run.
    fit_time : float
        Time taken to fit the model in seconds.
    error_message : str or None
        Error message if fitting failed.
    """
    spec: CompositeSpec
    result: Optional[EstimationResult] = None
    aic: float = np.inf
    bic: float = np.inf
    hqic: float = np.inf
    diagnostic_penalty: float = 0.0
    score: float = np.inf
    diagnostics: Optional[Dict[str, Any]] = None
    fit_time: float = 0.0
    error_message: Optional[str] = None
    dgt_passed: bool = False
    lb_failures: int = 0
    oos_loss: float = np.inf
    oos_losses: Optional[np.ndarray] = None
    oos_loss_name: Optional[str] = None
    oos_start: Optional[int] = None
    oos_stop: Optional[int] = None
    vol_type: Optional[str] = None
    p: Optional[int] = None
    q: Optional[int] = None
    density: Optional[str] = None
    mean_type: Optional[str] = None
    mean_signature: Optional[str] = None
    mean_p: Optional[int] = None
    mean_q: Optional[int] = None
    delta_score: float = np.inf
    weight: float = 0.0
    warm_started_from: Optional[str] = None
    boundary_hugging: bool = False
    boundary_parameters: Tuple[str, ...] = ()
    near_integrated: bool = False
    persistence: Optional[float] = None
    degeneracy_penalty: float = 0.0
    dm_stat_vs_best: float = np.nan
    dm_pvalue_vs_best: float = np.nan


class SelectionResult:
    """Public wrapper around the evaluated auto-selection candidates."""

    def __init__(
        self,
        candidates: List[ModelCandidate],
        *,
        select_by: str,
        criterion_name: str,
        search_space: Dict[str, Any],
    ) -> None:
        self.candidates = candidates
        self.select_by = select_by
        self.criterion_name = criterion_name
        self.search_space = search_space

    @property
    def best(self) -> Optional[ModelCandidate]:
        for candidate in self.candidates:
            if candidate.result is not None and np.isfinite(candidate.score):
                return candidate
        return None

    @property
    def decisive(self) -> Optional[bool]:
        """Whether the OOS winner beats the runner-up at the 5% level."""
        if self.select_by != "oos" or len(self.candidates) < 2:
            return None
        pvalue = self.candidates[1].dm_pvalue_vs_best
        return bool(pvalue < 0.05) if np.isfinite(pvalue) else None

    def to_dataframe(self) -> Any:
        try:
            import pandas as pd
        except ImportError as exc:
            raise ImportError("selection.to_dataframe() requires pandas.") from exc

        rows = []
        for rank, candidate in enumerate(self.candidates, start=1):
            rows.append(
                {
                    "rank": rank,
                    "model": str(candidate.spec),
                    "vol_type": candidate.vol_type,
                    "p": candidate.p,
                    "q": candidate.q,
                    "density": candidate.density,
                    "mean_type": candidate.mean_type,
                    "mean": candidate.mean_signature,
                    "mean_p": candidate.mean_p,
                    "mean_q": candidate.mean_q,
                    "loglik": (
                        candidate.result.log_likelihood
                        if candidate.result is not None else np.nan
                    ),
                    "aic": candidate.aic,
                    "bic": candidate.bic,
                    "hqic": candidate.hqic,
                    "score": candidate.score,
                    "delta_score": candidate.delta_score,
                    "weight": candidate.weight,
                    "oos_loss": candidate.oos_loss,
                    "oos_loss_name": candidate.oos_loss_name,
                    "dgt_passed": candidate.dgt_passed,
                    "lb_failures": candidate.lb_failures,
                    "converged": (
                        bool(candidate.result.converged)
                        if candidate.result is not None else False
                    ),
                    "fit_time": candidate.fit_time,
                    "error_message": candidate.error_message,
                    "warm_started_from": candidate.warm_started_from,
                    "boundary_hugging": candidate.boundary_hugging,
                    "boundary_parameters": ", ".join(candidate.boundary_parameters),
                    "near_integrated": candidate.near_integrated,
                    "persistence": candidate.persistence,
                    "degeneracy_penalty": candidate.degeneracy_penalty,
                    "dm_stat_vs_best": candidate.dm_stat_vs_best,
                    "dm_pvalue_vs_best": candidate.dm_pvalue_vs_best,
                }
            )
        return pd.DataFrame(rows)

    def summary(self, top_n: int = 10) -> None:
        _print_selection_summary(self.candidates, top_n=top_n, select_by=self.select_by)

    def confidence_set(
        self,
        alpha: float = 0.10,
        *,
        n_bootstrap: int = 1000,
        block_length: Optional[int] = None,
        random_state: Optional[int] = 0,
    ) -> ModelConfidenceSet:
        """Estimate a block-bootstrap model confidence set from OOS losses."""
        if self.select_by != "oos":
            raise ValueError("confidence_set() requires select_by='oos'.")
        return _model_confidence_set(
            self.candidates,
            alpha=alpha,
            n_bootstrap=n_bootstrap,
            block_length=block_length,
            random_state=random_state,
        )


@dataclass(frozen=True)
class ModelConfidenceSet:
    """Models not rejected as inferior by the bootstrap elimination procedure."""

    candidates: Tuple[ModelCandidate, ...]
    eliminated: Tuple[ModelCandidate, ...]
    alpha: float
    final_pvalue: float
    elimination_pvalues: Dict[str, float]

    def __len__(self) -> int:
        return len(self.candidates)

    def to_dataframe(self) -> Any:
        try:
            import pandas as pd
        except ImportError as exc:
            raise ImportError("confidence_set().to_dataframe() requires pandas.") from exc
        rows = []
        members = {id(candidate) for candidate in self.candidates}
        for candidate in (*self.candidates, *self.eliminated):
            rows.append({
                "model": str(candidate.spec),
                "mean_loss": candidate.oos_loss,
                "in_confidence_set": id(candidate) in members,
                "elimination_pvalue": self.elimination_pvalues.get(str(candidate.spec), np.nan),
            })
        return pd.DataFrame(rows).sort_values(
            ["in_confidence_set", "mean_loss"], ascending=[False, True]
        ).reset_index(drop=True)

    def summary(self) -> None:
        total = len(self.candidates) + len(self.eliminated)
        print(
            f"Model confidence set: {len(self.candidates)} of {total} models survive "
            f"at alpha={self.alpha:.2f}."
        )
        for candidate in self.candidates:
            print(f"  {candidate.spec}: mean loss={candidate.oos_loss:.6g}")


# =====================================================================
# Criterion helpers
# =====================================================================

def make_default_criterion(diagnostic_weight: float = 50.0) -> CriterionFunc:
    """
    Build the default selection criterion callable.
    
    The default scores candidates as::
    
        Score = AIC + diagnostic_weight * n_failed_tests
    
    Where ``n_failed_tests`` counts DGT and Ljung-Box moment failures.
    
    Parameters
    ----------
    diagnostic_weight : float
        AIC penalty per failed diagnostic test.
    
    Returns
    -------
    callable
        ``(result, diagnostics) -> float``
    """
    def criterion(
        result: EstimationResult,
        diagnostics: Optional[Dict[str, Any]],
    ) -> float:
        score = result.aic
        if diagnostics is None:
            return score + diagnostic_weight
        n_failed = int(diagnostics['dgt']['reject'])
        n_failed += sum(
            1 for lb in diagnostics['ljung_box'].values() if lb['reject']
        )
        return score + diagnostic_weight * n_failed
    return criterion


def make_string_criterion(select_by: str, diagnostic_weight: float = 50.0) -> CriterionFunc:
    """Build a built-in in-sample selection criterion from a public string."""
    key = _normalize_select_by(select_by)
    if key == "aic":
        return lambda result, _diagnostics: float(result.aic)
    if key == "bic":
        return lambda result, _diagnostics: float(result.bic)
    if key == "hqic":
        return lambda result, _diagnostics: float(result.hqic)
    if key == "aic+diag":
        return make_default_criterion(diagnostic_weight)
    raise ValueError(
        "select_by must be one of 'aic+diag', 'aic', 'bic', 'hqic', or 'oos'."
    )


def _normalize_select_by(select_by: str) -> str:
    if not isinstance(select_by, str):
        raise TypeError("select_by must be a string.")
    key = select_by.lower().replace("_", "-").strip()
    if key in {"aicdiag", "aic-diag", "aic+diagnostics"}:
        return "aic+diag"
    return key


def _resolve_selection_mode(
    *,
    select_by: str,
    criterion: Optional[CriterionFunc],
    diagnostic_weight: float,
) -> tuple[str, CriterionFunc, str]:
    key = _normalize_select_by(select_by)
    if criterion is not None:
        if not callable(criterion):
            raise TypeError("criterion must be callable; use select_by for built-in string criteria.")
        if key != "aic+diag":
            raise ValueError("Pass either a custom criterion callable or a non-default select_by, not both.")
        return "custom", criterion, "custom"
    if key == "oos":
        return key, make_string_criterion("aic", diagnostic_weight), "oos"
    return key, make_string_criterion(key, diagnostic_weight), key


def _diagnostic_penalty(diagnostics: Optional[Dict[str, Any]], diagnostic_weight: float) -> float:
    if diagnostics is None:
        return float(diagnostic_weight)
    n_failed = int(diagnostics["dgt"]["reject"])
    n_failed += sum(1 for lb in diagnostics["ljung_box"].values() if lb["reject"])
    return float(diagnostic_weight * n_failed)


def _score_candidate(
    result: EstimationResult,
    criterion: CriterionFunc,
    diagnostic_kwargs: Optional[Dict[str, Any]] = None,
) -> Tuple[float, Optional[Dict[str, Any]]]:
    """
    Run diagnostics and compute the selection score for a fitted candidate.
    
    Parameters
    ----------
    result : EstimationResult
        Successfully fitted model result.
    criterion : callable
        ``(result, diagnostics) -> float``. Lower is better.
    diagnostic_kwargs : dict, optional
        Keyword arguments forwarded to ``result.diagnostic_tests()``
        (e.g. ``lags``, ``n_cells``, ``alpha``).
    
    Returns
    -------
    score : float
        The criterion score.  ``float('inf')`` if the criterion raises.
    diagnostics : dict or None
        Raw diagnostics dict, or ``None`` if diagnostics failed.
    """
    diag_kw = diagnostic_kwargs or {}
    diagnostics: Optional[Dict[str, Any]] = None

    try:
        diagnostics = result.diagnostic_tests(print_results=False, **diag_kw)
    except Exception:
        pass  # diagnostics stays None; criterion must handle it

    try:
        score = float(criterion(result, diagnostics))
    except Exception:
        score = float('inf')

    return score, diagnostics


def _component_params(component: Any, values: np.ndarray) -> Dict[str, Any]:
    """Unpack a component without leaving its fitted state changed."""
    previous = getattr(component, "fitted_params", None)
    unpacked = component.unpack(np.asarray(values, dtype=np.float64).copy())
    component.fitted_params = previous
    return dict(unpacked)


def _mapped_component_start(
    source_component: Any,
    source_values: np.ndarray,
    target_component: Any,
    target_values: np.ndarray,
) -> np.ndarray:
    """Map shared scalar and lag parameters into a target component start."""
    if type(source_component) is not type(target_component):
        return target_values

    source = _component_params(source_component, source_values)
    target = _component_params(target_component, target_values)
    zero = (
        0.0
        if target_component.role.name == "MEAN"
        or target_component.signature.startswith("EGARCH")
        else 1e-6
    )
    for key, target_value in target.items():
        if key not in source:
            continue
        source_value = source[key]
        if isinstance(target_value, (list, tuple, np.ndarray)):
            target_list = [zero] * len(target_value)
            source_list = list(source_value)
            target_list[:min(len(target_list), len(source_list))] = source_list[:len(target_list)]
            target[key] = target_list
        else:
            target[key] = source_value
    return np.asarray(target_component.pack(target), dtype=np.float64)


def _warm_start_params(
    spec: CompositeSpec,
    data: np.ndarray,
    prior_candidates: List[ModelCandidate],
    *,
    bridge_density: str = "Normal",
    x: Any = None,
) -> tuple[Optional[np.ndarray], Optional[str]]:
    """Build a compatible start from the closest successful lattice neighbor."""
    from .roles import Role

    target_vol = spec.get_component(Role.VOLATILITY)
    target_density = spec.get_component(Role.DENSITY)
    if target_vol is None or target_density is None:
        return None, None

    target_mean = spec.get_component(Role.MEAN)
    # Linked mean kernels do not yet share a uniform public initial-parameter
    # contract, so provenance must remain honest until that contract is added.
    if target_mean is not None:
        return None, None
    target_mean_type = type(target_mean).__name__ if target_mean is not None else "Zero"
    target_mean_p = int(getattr(target_mean, "p", 0)) if target_mean is not None else 0
    target_mean_q = int(getattr(target_mean, "q", 0)) if target_mean is not None else 0
    ranked_sources = []
    for candidate in prior_candidates:
        if candidate.result is None or not candidate.result.converged:
            continue
        if candidate.vol_type != type(target_vol).__name__:
            continue
        if candidate.mean_type != target_mean_type:
            continue
        if candidate.p is None or candidate.q is None:
            continue
        target_p = int(getattr(target_vol, "p", 0))
        target_q = int(getattr(target_vol, "q", 0))
        if candidate.p > target_p or candidate.q > target_q:
            continue
        if candidate.mean_p is not None and candidate.mean_p > target_mean_p:
            continue
        if candidate.mean_q is not None and candidate.mean_q > target_mean_q:
            continue
        mean_distance = (
            target_mean_p - int(candidate.mean_p or 0)
            + target_mean_q - int(candidate.mean_q or 0)
        )
        same_mean_order = mean_distance == 0
        same_order = candidate.p == target_p and candidate.q == target_q
        same_density = candidate.density == target_density.signature
        density_bridge = candidate.density == bridge_density
        if same_order and same_mean_order and density_bridge and not same_density:
            priority = (0, 0)
        elif same_density:
            priority = (
                1,
                target_p - candidate.p + target_q - candidate.q + mean_distance,
            )
        elif density_bridge:
            priority = (
                2,
                target_p - candidate.p + target_q - candidate.q + mean_distance,
            )
        else:
            continue
        ranked_sources.append((priority, candidate))

    if not ranked_sources:
        return None, None
    source = min(ranked_sources, key=lambda item: item[0])[1]
    source_spec = source.result.spec
    source_params = np.asarray(source.result.params, dtype=np.float64)
    pieces = []
    for target_component in spec.components:
        if hasattr(target_component, "set_n_exog") and x is not None:
            x_array = np.asarray(x)
            target_component.set_n_exog(1 if x_array.ndim == 1 else x_array.shape[1])
        target_start = np.asarray(target_component.default_start(data), dtype=np.float64)
        source_component = source_spec.get_component(target_component.role)
        if source_component is not None:
            source_slice = source_spec.slice_map[source_component]
            target_start = _mapped_component_start(
                source_component,
                source_params[source_slice],
                target_component,
                target_start,
            )
        pieces.append(target_start)
    return np.ascontiguousarray(np.concatenate(pieces), dtype=np.float64), str(source.spec)


def _assess_candidate_quality(
    candidate: ModelCandidate,
    *,
    degeneracy_weight: float,
    boundary_tolerance: float = 1e-4,
    persistence_threshold: float = 0.995,
) -> None:
    """Record boundary and near-unit-root evidence for one fitted candidate."""
    result = candidate.result
    if result is None:
        return

    bounds = [bound for component in result.spec.components for bound in component.bounds()]
    params = np.asarray(result.params, dtype=np.float64)
    try:
        names = result._get_param_names()
    except Exception:
        names = [f"parameter[{i}]" for i in range(params.size)]
    pinned = []
    for index, (value, bound) in enumerate(zip(params, bounds)):
        lower, upper = bound
        finite_scale = max(
            1.0,
            abs(float(value)),
            abs(float(lower)) if np.isfinite(lower) else 0.0,
            abs(float(upper)) if np.isfinite(upper) else 0.0,
        )
        tolerance = boundary_tolerance * finite_scale
        if (np.isfinite(lower) and value - lower <= tolerance) or (
            np.isfinite(upper) and upper - value <= tolerance
        ):
            pinned.append(names[index] if index < len(names) else f"parameter[{index}]")

    candidate.boundary_parameters = tuple(pinned)
    candidate.boundary_hugging = bool(pinned)
    gp = result.garch_params
    if gp is not None:
        candidate.persistence = float(gp.persistence)
        candidate.near_integrated = abs(candidate.persistence) >= persistence_threshold
    flags = int(candidate.boundary_hugging) + int(candidate.near_integrated)
    candidate.degeneracy_penalty = float(degeneracy_weight * flags)


def _diebold_mariano(loss_a: np.ndarray, loss_b: np.ndarray) -> tuple[float, float]:
    """Return a two-sided HAC Diebold-Mariano statistic for equal mean loss."""
    difference = np.asarray(loss_a, dtype=np.float64) - np.asarray(loss_b, dtype=np.float64)
    difference = difference[np.isfinite(difference)]
    n_obs = difference.size
    if n_obs < 3:
        return np.nan, np.nan
    centered = difference - float(np.mean(difference))
    max_lag = min(int(np.floor(n_obs ** (1.0 / 3.0))), n_obs - 1)
    long_run_variance = float(np.dot(centered, centered) / n_obs)
    for lag in range(1, max_lag + 1):
        covariance = float(np.dot(centered[lag:], centered[:-lag]) / n_obs)
        long_run_variance += 2.0 * (1.0 - lag / (max_lag + 1.0)) * covariance
    if not np.isfinite(long_run_variance) or long_run_variance <= 0.0:
        return np.nan, np.nan
    statistic = float(np.mean(difference) / np.sqrt(long_run_variance / n_obs))
    return statistic, float(2.0 * ndtr(-abs(statistic)))


def _finalize_oos_comparisons(candidates: List[ModelCandidate]) -> None:
    """Compare each OOS candidate's loss path with the selected winner."""
    if not candidates or candidates[0].oos_losses is None:
        return
    best_losses = candidates[0].oos_losses
    for candidate in candidates[1:]:
        if candidate.oos_losses is None or candidate.oos_losses.shape != best_losses.shape:
            continue
        candidate.dm_stat_vs_best, candidate.dm_pvalue_vs_best = _diebold_mariano(
            candidate.oos_losses, best_losses
        )


def _split_x(x: Any, stop: int) -> Any:
    if x is None:
        return None
    return x.iloc[:stop] if hasattr(x, "iloc") else x[:stop]


def _resolve_oos_split(
    n_obs: int,
    *,
    effective_hold_back: int,
    oos_test_size: float | int,
    min_train_size: int,
    min_test_size: int,
) -> tuple[int, int, int]:
    effective_n = n_obs - effective_hold_back
    if effective_n <= 0:
        raise ValueError("OOS selection requires positive effective sample size after hold_back.")
    if isinstance(oos_test_size, float):
        if not 0.0 < oos_test_size < 1.0:
            raise ValueError("Float oos_test_size must be between 0 and 1.")
        test_n = int(np.ceil(effective_n * oos_test_size))
    else:
        test_n = int(oos_test_size)
        if test_n <= 0:
            raise ValueError("Integer oos_test_size must be positive.")
        if test_n < min_test_size:
            warnings.warn(
                f"oos_test_size={test_n} gives fewer than {min_test_size} effective "
                "holdout observations; OOS decisiveness tests will be weak.",
                UserWarning,
                stacklevel=3,
            )
    train_effective_n = effective_n - test_n
    if train_effective_n < min_train_size:
        raise ValueError(
            f"OOS selection requires at least {min_train_size} effective training "
            f"observations after hold_back; got {train_effective_n}."
        )
    if test_n < min_test_size and isinstance(oos_test_size, float):
        raise ValueError(
            f"OOS selection requires at least {min_test_size} effective holdout "
            f"observations; got {test_n}."
        )
    train_stop = n_obs - test_n
    return train_stop, train_stop, n_obs


def _per_observation_loglik(
    distribution: str,
    resid: np.ndarray,
    sigma2: np.ndarray,
    *,
    nu: Optional[float],
    lam: Optional[float],
    rho: Optional[float] = None,
    zeta: Optional[float] = None,
    shape_lambda: Optional[float] = None,
) -> np.ndarray:
    sigma2 = np.maximum(np.asarray(sigma2, dtype=np.float64), 1e-12)
    resid = np.asarray(resid, dtype=np.float64)
    z = resid / np.sqrt(sigma2)
    if distribution == "Normal":
        return -0.5 * (np.log(2.0 * np.pi) + np.log(sigma2) + z * z)
    if distribution == "StudentT":
        if nu is None:
            raise ValueError("StudentT OOS log score requires `nu`.")
        c_log = (
            gammaln(0.5 * (nu + 1.0))
            - gammaln(0.5 * nu)
            - 0.5 * np.log(np.pi * (nu - 2.0))
        )
        return c_log - 0.5 * np.log(sigma2) - 0.5 * (nu + 1.0) * np.log1p((z * z) / (nu - 2.0))
    if distribution == "GED":
        if nu is None:
            raise ValueError("GED OOS log score requires `nu`.")
        log_scale = 0.5 * (gammaln(1.0 / nu) - gammaln(3.0 / nu))
        log_const = np.log(nu) - np.log(2.0) - log_scale - gammaln(1.0 / nu)
        z_std = np.abs(z) / np.exp(log_scale)
        return log_const - 0.5 * np.log(sigma2) - np.power(z_std, nu)
    if distribution == "SkewT":
        if nu is None or lam is None:
            raise ValueError("SkewT OOS log score requires `nu` and `lam`.")
        c_log = (
            gammaln(0.5 * (nu + 1.0))
            - gammaln(0.5 * nu)
            - 0.5 * np.log(np.pi * (nu - 2.0))
        )
        a = 4.0 * lam * np.exp(c_log) * (nu - 2.0) / (nu - 1.0)
        b = np.sqrt(max(1.0 + 3.0 * lam * lam - a * a, 1e-12))
        bz_plus_a = b * z + a
        sign_term = np.where(bz_plus_a >= 0.0, 1.0, -1.0)
        z_adj = bz_plus_a / (1.0 + lam * sign_term)
        return (
            c_log
            + np.log(b)
            - 0.5 * np.log(sigma2)
            - 0.5 * (nu + 1.0) * np.log1p((z_adj * z_adj) / (nu - 2.0))
        )
    if distribution in {"GeneralizedHyperbolic", "Hyperbolic", "NormalInverseGaussian"}:
        if rho is None or zeta is None or shape_lambda is None:
            raise ValueError(f"{distribution} OOS log score requires GH shape parameters.")
        return (
            standardized_gh_logpdf(z, rho, zeta, shape_lambda)
            - 0.5 * np.log(sigma2)
        )
    if distribution == "VarianceGamma":
        if rho is None or shape_lambda is None:
            raise ValueError("VarianceGamma OOS log score requires `rho` and `lambda`.")
        return (
            standardized_vg_logpdf(z, rho, shape_lambda)
            - 0.5 * np.log(sigma2)
        )
    raise ValueError(f"Unsupported distribution '{distribution}'.")


def _compute_oos_losses(
    result: EstimationResult,
    full_data: np.ndarray,
    *,
    full_x: Any,
    start: int,
    stop: int,
    loss: str,
) -> np.ndarray:
    state = result.spec.filter(full_data, result.params, x=full_x)
    hold_back = int(getattr(result.fit_info, "effective_hold_back", 0) or 0)
    effective_start = max(start - hold_back, 0)
    effective_stop = max(stop - hold_back, effective_start)
    resid = np.asarray(state.residuals[effective_start:effective_stop], dtype=np.float64)
    sigma2 = np.asarray(state.sigma2[effective_start:effective_stop], dtype=np.float64)
    if resid.size == 0:
        raise ValueError("OOS holdout slice is empty.")
    key = loss.lower()
    if key == "logscore":
        ll = _per_observation_loglik(
            state.distribution,
            resid,
            sigma2,
            nu=state.nu,
            lam=state.lam,
            rho=state.rho,
            zeta=state.zeta,
            shape_lambda=state.shape_lambda,
        )
        return -ll
    realized_var = resid * resid
    if key == "qlike":
        return np.log(sigma2) + realized_var / sigma2
    if key == "mse":
        return (realized_var - sigma2) ** 2
    raise ValueError("oos_loss must be 'logscore', 'qlike', or 'mse'.")


def _compute_rolling_oos_losses(
    spec: CompositeSpec,
    initial_result: EstimationResult,
    full_data: np.ndarray,
    *,
    full_x: Any,
    start: int,
    stop: int,
    loss: str,
    refit_every: int,
    fit_kwargs: Dict[str, Any],
) -> np.ndarray:
    """Score an expanding-window holdout, refitting after each completed chunk."""
    if not isinstance(refit_every, (int, np.integer)) or int(refit_every) <= 0:
        raise ValueError("oos_refit must be a positive integer or None.")
    refit_every = int(refit_every)
    current_result = initial_result
    chunks = []
    for chunk_start in range(start, stop, refit_every):
        chunk_stop = min(chunk_start + refit_every, stop)
        data_through_chunk = full_data[:chunk_stop]
        x_through_chunk = _split_x(full_x, chunk_stop)
        chunks.append(_compute_oos_losses(
            current_result,
            data_through_chunk,
            full_x=x_through_chunk,
            start=chunk_start,
            stop=chunk_stop,
            loss=loss,
        ))
        if chunk_stop >= stop:
            continue
        current_result = _refit_rolling_window(
            spec,
            data_through_chunk,
            x_through_chunk=x_through_chunk,
            fit_kwargs=fit_kwargs,
            previous_result=current_result,
            observation=chunk_stop,
        )
    return np.ascontiguousarray(np.concatenate(chunks), dtype=np.float64)


def _refit_rolling_window(
    spec: CompositeSpec,
    data: np.ndarray,
    *,
    x_through_chunk: Any,
    fit_kwargs: Dict[str, Any],
    previous_result: EstimationResult,
    observation: int,
) -> EstimationResult:
    """Refit one expanding window, retrying a failed warm start from defaults."""
    refit_kwargs = fit_kwargs.copy()
    refit_kwargs["n_jobs"] = 1
    if "x" in refit_kwargs:
        refit_kwargs["x"] = x_through_chunk

    method = str(refit_kwargs.get("method", "mle")).lower()
    if method == "qmle":
        result = spec.fit(data, **refit_kwargs)
        if result.converged:
            return result
        raise RuntimeError(
            f"Rolling OOS refit failed to converge at observation {observation}."
        )

    refit_kwargs["initial_params"] = previous_result.params
    warm_result = spec.fit(data, **refit_kwargs)
    if warm_result.converged:
        return warm_result

    refit_kwargs.pop("initial_params")
    cold_result = spec.fit(data, **refit_kwargs)
    if cold_result.converged:
        cold_result.fit_info.add_warning(
            f"Rolling OOS warm start failed at observation {observation}; "
            "the refit converged from the model's default initialization."
        )
        return cold_result

    raise RuntimeError(
        f"Rolling OOS refit failed to converge at observation {observation} "
        "from both warm and default initializations."
    )


def _circular_block_indices(
    rng: np.random.Generator,
    n_obs: int,
    block_length: int,
) -> np.ndarray:
    n_blocks = int(np.ceil(n_obs / block_length))
    starts = rng.integers(0, n_obs, size=n_blocks)
    offsets = np.arange(block_length)
    return ((starts[:, None] + offsets[None, :]) % n_obs).ravel()[:n_obs]


def _model_confidence_set(
    candidates: List[ModelCandidate],
    *,
    alpha: float,
    n_bootstrap: int,
    block_length: Optional[int],
    random_state: Optional[int],
) -> ModelConfidenceSet:
    """Run a Hansen-style Tmax elimination on aligned OOS loss paths."""
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must be between 0 and 1.")
    if not isinstance(n_bootstrap, (int, np.integer)) or int(n_bootstrap) < 100:
        raise ValueError("n_bootstrap must be an integer of at least 100.")
    usable = [
        candidate for candidate in candidates
        if candidate.oos_losses is not None
        and np.all(np.isfinite(candidate.oos_losses))
    ]
    if not usable:
        raise ValueError("No finite aligned OOS loss vectors are available.")
    lengths = {candidate.oos_losses.size for candidate in usable}
    if len(lengths) != 1:
        raise ValueError("MCS requires equal-length OOS loss vectors.")
    n_obs = lengths.pop()
    if n_obs < 20:
        raise ValueError("MCS requires at least 20 OOS observations.")
    if block_length is None:
        block_length = max(2, int(round(n_obs ** (1.0 / 3.0))))
    if not isinstance(block_length, (int, np.integer)) or not 1 <= int(block_length) <= n_obs:
        raise ValueError("block_length must be between 1 and the OOS sample size.")

    losses = np.column_stack([candidate.oos_losses for candidate in usable])
    rng = np.random.default_rng(random_state)
    active = list(range(len(usable)))
    eliminated = []
    elimination_pvalues: Dict[str, float] = {}
    final_pvalue = 1.0
    while len(active) > 1:
        active_losses = losses[:, active]
        relative = active_losses - np.mean(active_losses, axis=1, keepdims=True)
        means = np.mean(relative, axis=0)
        centered = relative - means
        bootstrap_means = np.empty(
            (int(n_bootstrap), len(active)), dtype=np.float64
        )
        for draw in range(int(n_bootstrap)):
            indices = _circular_block_indices(rng, n_obs, int(block_length))
            bootstrap_means[draw] = np.mean(centered[indices], axis=0)
        standard_errors = np.std(bootstrap_means, axis=0, ddof=1)
        standard_errors = np.maximum(standard_errors, np.finfo(np.float64).eps)
        observed = float(np.max(means / standard_errors))
        bootstrap_max = np.max(bootstrap_means / standard_errors, axis=1)
        final_pvalue = float((1 + np.sum(bootstrap_max >= observed)) / (n_bootstrap + 1))
        if final_pvalue >= alpha:
            break
        worst_position = int(np.argmax(means / standard_errors))
        worst_index = active.pop(worst_position)
        worst_candidate = usable[worst_index]
        eliminated.append(worst_candidate)
        elimination_pvalues[str(worst_candidate.spec)] = final_pvalue

    survivors = tuple(usable[index] for index in active)
    return ModelConfidenceSet(
        candidates=survivors,
        eliminated=tuple(eliminated),
        alpha=float(alpha),
        final_pvalue=final_pvalue,
        elimination_pvalues=elimination_pvalues,
    )


def _has_auto_density(density_candidates: List[str]) -> bool:
    return len(set(density_candidates)) > 1


def _check_density_blind_loss(
    *,
    oos_loss: str,
    density_candidates: List[str],
    allow_density_blind_loss: bool,
) -> None:
    if oos_loss.lower() not in {"qlike", "mse"}:
        return
    if not _has_auto_density(density_candidates) or allow_density_blind_loss:
        return
    raise ValueError(
        f"oos_loss='{oos_loss}' is density-blind: different densities can alter "
        "the fitted variance path, but QLIKE/MSE do not score the predictive "
        "density itself and users often over-read those rankings. Use "
        "oos_loss='logscore', fix a single density, or pass "
        "allow_density_blind_loss=True."
    )


def _finalize_candidate_ranking(candidates: List[ModelCandidate]) -> None:
    finite = [candidate.score for candidate in candidates if np.isfinite(candidate.score)]
    if not finite:
        return
    best = min(finite)
    weights_raw = []
    for candidate in candidates:
        candidate.delta_score = float(candidate.score - best) if np.isfinite(candidate.score) else np.inf
        weights_raw.append(np.exp(-0.5 * candidate.delta_score) if np.isfinite(candidate.delta_score) else 0.0)
    total = float(np.sum(weights_raw))
    if total > 0.0:
        for candidate, weight in zip(candidates, weights_raw):
            candidate.weight = float(weight / total)


def _print_selection_summary(
    candidates: List[ModelCandidate],
    *,
    top_n: int = 10,
    select_by: str = "aic+diag",
) -> None:
    WIDTH = 100
    print()
    print("=" * WIDTH)
    print(f"{'Model Selection Summary':^{WIDTH}}")
    print("=" * WIDTH)
    oos_mode = select_by == "oos"
    if oos_mode:
        print(f"{'Rank':<6} {'Model':<28} {'AIC':>10} {'neg. log score':>16} {'Score':>12} {'Delta':>10} {'Time':>8}")
    else:
        print(
            f"{'Rank':<6} {'Model':<28} {'AIC':>10} {'BIC':>10} "
            f"{'Diag Pen':>10} {'Score':>12} {'Delta':>10} {'Time':>8}"
        )
    print("-" * WIDTH)
    for i, candidate in enumerate(candidates[:top_n], 1):
        model_str = str(candidate.spec)[:27]
        aic_str = f"{candidate.aic:.2f}" if np.isfinite(candidate.aic) else "Failed"
        score_str = f"{candidate.score:.2f}" if np.isfinite(candidate.score) else "Failed"
        delta_str = f"{candidate.delta_score:.2f}" if np.isfinite(candidate.delta_score) else "-"
        time_str = f"{candidate.fit_time:.2f}s"
        marker = " *" if i == 1 else ""
        if oos_mode:
            loss_str = f"{candidate.oos_loss:.4f}" if np.isfinite(candidate.oos_loss) else "Failed"
            print(
                f"{i:<6} {model_str:<28} {aic_str:>10} {loss_str:>16} "
                f"{score_str:>12} {delta_str:>10} {time_str:>8}{marker}"
            )
        else:
            bic_str = f"{candidate.bic:.2f}" if np.isfinite(candidate.bic) else "Failed"
            pen_str = f"{candidate.diagnostic_penalty:.1f}"
            print(
                f"{i:<6} {model_str:<28} {aic_str:>10} {bic_str:>10} "
                f"{pen_str:>10} {score_str:>12} {delta_str:>10} {time_str:>8}{marker}"
            )
    print("-" * WIDTH)
    total_time = sum(candidate.fit_time for candidate in candidates)
    n_failed = sum(1 for candidate in candidates if candidate.result is None or not np.isfinite(candidate.score))
    print(f"Total: {len(candidates)} models evaluated ({n_failed} failed), {total_time:.2f}s total time")
    if candidates:
        print(f"Best:  {candidates[0].spec}")
    if len(candidates) > 1 and np.isfinite(candidates[1].delta_score) and candidates[1].delta_score < 2.0:
        print(f"Selection not decisive: top 2 models are within {candidates[1].delta_score:.2f} score points.")
    if oos_mode and len(candidates) > 1:
        pvalue = candidates[1].dm_pvalue_vs_best
        if np.isfinite(pvalue):
            verdict = "decisive" if pvalue < 0.05 else "not decisive"
            print(f"Winner vs runner-up: DM p={pvalue:.3f} ({verdict} at 5%).")
    n_degenerate = sum(
        candidate.boundary_hugging or candidate.near_integrated
        for candidate in candidates
    )
    if n_degenerate:
        print(f"Degeneracy flags: {n_degenerate} candidate(s) near a bound or unit root.")


# =====================================================================
# Volatility / density class maps
# =====================================================================

def _get_vol_map() -> Dict[str, Any]:
    """Return mapping of volatility type names to classes."""
    from .components.vol import APARCH, EGARCH, GARCH, GJRGARCH
    return {
        'GARCH': GARCH,
        'GJRGARCH': GJRGARCH,
        'EGARCH': EGARCH,
        'APARCH': APARCH,
    }


def _get_density_map() -> Dict[str, Any]:
    """Return mapping of density names to classes."""
    from .components.density import (
        GED,
        GeneralizedHyperbolic,
        Hyperbolic,
        Normal,
        NormalInverseGaussian,
        StudentT,
        SkewT,
        VarianceGamma,
    )
    return {
        'Normal': Normal,
        'StudentT': StudentT,
        'SkewT': SkewT,
        'GED': GED,
        'GeneralizedHyperbolic': GeneralizedHyperbolic,
        'Hyperbolic': Hyperbolic,
        'NormalInverseGaussian': NormalInverseGaussian,
        'VarianceGamma': VarianceGamma,
    }


def _mean_signature(candidate: MeanCandidate) -> str:
    kind, args = candidate
    if kind == "Zero":
        return "Zero"
    if kind == "ARMA":
        return f"ARMA({args[0]},{args[1]})"
    if kind == "ARX":
        return f"ARX({args[0]})"
    if kind == "HARX":
        return f"HARX({','.join(str(value) for value in args[0])})"
    raise ValueError(f"Unknown mean candidate '{kind}'.")


def _build_candidate_spec(
    mean_candidate: MeanCandidate,
    vol_type: str,
    p: int,
    q: int,
    density_name: str,
) -> CompositeSpec:
    """Build one concrete mean-volatility-density candidate."""
    from .components.mean import ARMA, ARX, HARX

    vol_map = _get_vol_map()
    density_map = _get_density_map()
    vol = vol_map[vol_type](p, q)
    density = density_map[density_name]()
    kind, args = mean_candidate
    if kind == "Zero":
        return vol + density
    if kind == "ARMA":
        return ARMA(int(args[0]), int(args[1])) + vol + density
    if kind == "ARX":
        return ARX(int(args[0]), constant=bool(args[1])) + vol + density
    if kind == "HARX":
        return HARX(tuple(args[0]), constant=bool(args[1])) + vol + density
    raise ValueError(f"Unknown mean candidate '{kind}'.")


def _candidate_record(
    spec: CompositeSpec,
    mean_candidate: MeanCandidate,
    vol_type: str,
    p: int,
    q: int,
    density_name: str,
) -> ModelCandidate:
    kind, args = mean_candidate
    return ModelCandidate(
        spec=spec,
        mean_type=kind,
        mean_signature=_mean_signature(mean_candidate),
        mean_p=int(args[0]) if kind == "ARMA" else 0,
        mean_q=int(args[1]) if kind == "ARMA" else 0,
        vol_type=vol_type,
        p=p,
        q=q,
        density=density_name,
    )


# =====================================================================
# Main selection entry point
# =====================================================================

def select_best_model(
    data: np.ndarray,
    vol_candidates: List[Tuple[str, int, int]],
    density_candidates: List[str],
    *,
    mean_candidates: Optional[List[MeanCandidate]] = None,
    mean_auto: bool = False,
    candidate_budget: int = 120,
    mean_finalists: int = 2,
    select_by: str = "aic+diag",
    criterion: Optional[CriterionFunc] = None,
    diagnostic_kwargs: Optional[Dict[str, Any]] = None,
    diagnostic_weight: float = 50.0,
    selection_verbose: bool = False,
    show_progress: bool = False,
    n_jobs: Optional[int] = None,
    oos_loss: str = "logscore",
    oos_test_size: float | int = 0.2,
    oos_refit: Optional[int] = None,
    min_train_size: int = 250,
    min_test_size: int = 50,
    allow_density_blind_loss: bool = False,
    selection_warm_start: bool = True,
    degeneracy_weight: float = 50.0,
    **fit_kwargs: Any,
) -> Tuple[EstimationResult, List[ModelCandidate]]:
    """
    Fit all candidate models and select the best by a criterion callable.
    
    Parameters
    ----------
    data : np.ndarray
        1-D array of returns/residuals to fit.
    vol_candidates : List[Tuple[str, int, int]]
        List of ``(vol_type, p, q)`` tuples for volatility models/orders.
    density_candidates : List[str]
        Distribution names accepted by :class:`AutoDensity`. GH-family densities are
        available as explicit candidates but are not in its default grid.
    criterion : callable, optional
        ``(result, diagnostics) -> float``.  Lower score is better.
        *result* is an :class:`EstimationResult`; *diagnostics* is the full
        dict returned by ``result.diagnostic_tests()`` (or ``None`` if
        diagnostics failed).  When ``None`` (default), the built-in
        ``AIC + diagnostic_weight * n_failed_tests`` criterion is used.
    diagnostic_kwargs : dict, optional
        Keyword arguments forwarded to ``result.diagnostic_tests()``
        (e.g. ``{'lags': 5, 'n_cells': 50, 'alpha': 0.01}``).
    diagnostic_weight : float, default 50.0
        AIC penalty per failed diagnostic test (only used when
        *criterion* is ``None``).
    selection_verbose : bool, default False
        Print detailed per-candidate results.
    show_progress : bool, default False
        Show a tqdm progress bar.
    n_jobs : int, optional
        Number of parallel workers.  Default uses all CPU cores.
    **fit_kwargs
        Additional keyword arguments passed to ``spec.fit()``.
        
    Returns
    -------
    best_result : EstimationResult
        The estimation result for the best model.
    candidates : List[ModelCandidate]
        All evaluated candidates, sorted by score (best first).
        
    Raises
    ------
    RuntimeError
        If all candidate models fail to fit.
    """
    import time
    from ._parallel import get_default_workers
    
    vol_map = _get_vol_map()
    density_map = _get_density_map()
    
    selection_mode, criterion_func, criterion_name = _resolve_selection_mode(
        select_by=select_by,
        criterion=criterion,
        diagnostic_weight=diagnostic_weight,
    )
    _check_density_blind_loss(
        oos_loss=oos_loss,
        density_candidates=density_candidates,
        allow_density_blind_loss=allow_density_blind_loss,
    )
    if oos_refit is not None:
        if selection_mode != "oos":
            raise ValueError("oos_refit is only valid with select_by='oos'.")
        if not isinstance(oos_refit, (int, np.integer)) or int(oos_refit) <= 0:
            raise ValueError("oos_refit must be a positive integer or None.")
    
    # Handle QMLE + AutoDensity case
    method = fit_kwargs.get('method', 'MLE')
    if isinstance(method, str) and method.upper() == 'QMLE' and len(density_candidates) > 1:
        warnings.warn(
            "AutoDensity with QMLE is redundant: QMLE always fits with Normal "
            "likelihood and uses robust standard errors. Proceeding with Normal only. "
            "To test other distributions, fit them manually with MLE.",
            UserWarning,
        )
        density_candidates = ['Normal']

    mean_candidates = mean_candidates or [("Zero", ())]
    requested_hold_back = int(fit_kwargs.get("hold_back", 0) or 0)
    common_sample = fit_kwargs.get("common_sample", None)
    if common_sample is not False and vol_candidates:
        common_hold_back = max(max(p, q) for _, p, q in vol_candidates)
        for kind, args in mean_candidates:
            if kind == "ARMA":
                common_hold_back = max(common_hold_back, int(args[0]), int(args[1]))
            elif kind == "ARX":
                common_hold_back = max(common_hold_back, int(args[0]))
            elif kind == "HARX":
                common_hold_back = max(common_hold_back, max(args[0]))
        fit_kwargs["hold_back"] = max(requested_hold_back, common_hold_back)
        fit_kwargs["common_sample"] = True
    else:
        fit_kwargs["hold_back"] = requested_hold_back
        fit_kwargs["common_sample"] = False
    
    full_total = len(mean_candidates) * len(vol_candidates) * len(density_candidates)
    cross_size = len(vol_candidates) * len(density_candidates)
    staged_mean_search = bool(mean_auto and full_total > candidate_budget and cross_size > 1)
    if staged_mean_search:
        if candidate_budget < len(mean_candidates):
            raise ValueError(
                "AutoMean candidate_budget must be large enough to screen every "
                f"mean candidate ({len(mean_candidates)} required)."
            )
        allowed_finalists = min(
            int(mean_finalists),
            len(mean_candidates),
            (candidate_budget - len(mean_candidates)) // (cross_size - 1),
        )
        if allowed_finalists < 1:
            raise ValueError(
                "AutoMean candidate_budget leaves no room to expand a screened "
                "mean finalist across the volatility-density grid."
            )
        total = len(mean_candidates) + allowed_finalists * (cross_size - 1)
    else:
        allowed_finalists = len(mean_candidates)
        total = full_total
    n_jobs_actual = n_jobs if n_jobs is not None else get_default_workers()

    oos_split: Optional[tuple[int, int, int]] = None
    if selection_mode == "oos":
        oos_split = _resolve_oos_split(
            len(data),
            effective_hold_back=int(fit_kwargs.get("hold_back", 0) or 0),
            oos_test_size=oos_test_size,
            min_train_size=min_train_size,
            min_test_size=min_test_size,
        )
    
    # Parallel execution for auto-selection.
    #
    # Process-based parallelism (joblib/loky) incurs ~1-2 seconds of
    # spawn + serialization overhead on macOS/Linux.  Individual model
    # fits typically take 3-60 ms, so the overhead dwarfs the computation
    # for the candidate counts produced by AutoVol+AutoDensity (≤54).
    #
    # Threshold: only go parallel when the user explicitly requested
    # n_jobs > 1 *and* there are enough candidates that the total
    # sequential work is likely to exceed the spawn overhead (~2 s).
    # With ~20 ms average per candidate, break-even is ~100 candidates.
    _MIN_PARALLEL_CANDIDATES = 100
    use_parallel = (
        n_jobs_actual > 1
        and total >= _MIN_PARALLEL_CANDIDATES
        and mean_candidates == [("Zero", ())]
    )
    
    if use_parallel:
        from ._parallel import select_best_parallel
        return select_best_parallel(
            data,
            vol_candidates,
            density_candidates,
            select_by=selection_mode,
            criterion=criterion_func,
            diagnostic_kwargs=diagnostic_kwargs,
            n_jobs=n_jobs_actual,
            selection_verbose=selection_verbose,
            show_progress=show_progress,
            oos_loss=oos_loss,
            oos_test_size=oos_test_size,
            oos_refit=oos_refit,
            min_train_size=min_train_size,
            min_test_size=min_test_size,
            allow_density_blind_loss=allow_density_blind_loss,
            selection_warm_start=selection_warm_start,
            degeneracy_weight=degeneracy_weight,
            diagnostic_weight=diagnostic_weight,
            **fit_kwargs,
        )
    
    # Sequential execution
    from ._progress import get_progress_bar

    candidates: List[ModelCandidate] = []
    
    if selection_verbose:
        print(f"Auto-selecting from {total} candidate models...")
    
    baseline_vol = next(
        (candidate for candidate in vol_candidates if candidate == ("GARCH", 1, 1)),
        vol_candidates[0],
    )
    baseline_density = "Normal" if "Normal" in density_candidates else density_candidates[0]
    if staged_mean_search:
        phases = [[
            (mean_candidate, *baseline_vol, baseline_density)
            for mean_candidate in mean_candidates
        ]]
    else:
        phases = [[
            (mean_candidate, vol_type, p, q, density_name)
            for mean_candidate in mean_candidates
            for vol_type, p, q in vol_candidates
            for density_name in density_candidates
        ]]

    evaluated = 0
    phase_index = 0
    while phase_index < len(phases):
        tasks = phases[phase_index]
        if n_jobs_actual > 1 and len(tasks) >= _MIN_PARALLEL_CANDIDATES:
            from joblib import Parallel, delayed
            from ._parallel import _fit_candidate_worker
            from ._progress import tqdm_joblib

            actual_workers = min(n_jobs_actual, len(tasks))
            with tqdm_joblib(
                total=len(tasks),
                desc="Auto-selecting",
                disable=not show_progress,
            ):
                phase_candidates = Parallel(
                    n_jobs=actual_workers, prefer="processes"
                )(
                    delayed(_fit_candidate_worker)(
                        vol_type,
                        p,
                        q,
                        density_name,
                        data,
                        fit_kwargs,
                        criterion_func,
                        diagnostic_kwargs,
                        selection_mode,
                        oos_split,
                        oos_loss,
                        diagnostic_weight,
                        degeneracy_weight,
                        oos_refit,
                        mean_candidate,
                    )
                    for mean_candidate, vol_type, p, q, density_name in tasks
                )
            candidates.extend(phase_candidates)
            evaluated += len(tasks)
            if selection_verbose:
                for candidate in phase_candidates:
                    status = (
                        f"AIC={candidate.aic:.2f}, Score={candidate.score:.2f}"
                        if np.isfinite(candidate.score)
                        else f"FAILED: {candidate.error_message}"
                    )
                    print(f"  [{candidate.spec}] {status}")
            tasks = []
        for mean_candidate, vol_type, p, q, density_name in get_progress_bar(
            tasks,
            total=len(tasks),
            desc="Auto-selecting",
            disable=not show_progress,
        ):
            evaluated += 1
            if vol_type not in vol_map:
                warnings.warn(f"Unknown volatility model '{vol_type}', skipping.")
                continue
            if density_name not in density_map:
                warnings.warn(f"Unknown density '{density_name}', skipping.")
                continue

            spec = _build_candidate_spec(
                mean_candidate, vol_type, p, q, density_name
            )
            candidate = _candidate_record(
                spec, mean_candidate, vol_type, p, q, density_name
            )
            if selection_verbose:
                print(f"  [{evaluated}/{total}] Fitting {spec}...", end=" ")
            start_time = time.perf_counter()

            try:
                fit_kwargs_copy = fit_kwargs.copy()
                fit_kwargs_copy["n_jobs"] = 1
                fit_data = data
                if oos_split is not None:
                    train_stop, _, _ = oos_split
                    fit_data = data[:train_stop]
                    if "x" in fit_kwargs_copy:
                        fit_kwargs_copy["x"] = _split_x(
                            fit_kwargs_copy["x"], train_stop
                        )
                if selection_warm_start:
                    initial_params, source_name = _warm_start_params(
                        spec,
                        fit_data,
                        candidates,
                        bridge_density=baseline_density,
                        x=fit_kwargs_copy.get("x"),
                    )
                    if initial_params is not None:
                        fit_kwargs_copy["initial_params"] = initial_params
                        candidate.warm_started_from = source_name
                result = spec.fit(fit_data, **fit_kwargs_copy)
                candidate.result = result
                candidate.aic = result.aic
                candidate.bic = result.bic
                candidate.hqic = result.hqic

                score, diagnostics = _score_candidate(
                    result, criterion_func, diagnostic_kwargs
                )
                candidate.diagnostics = diagnostics
                candidate.diagnostic_penalty = _diagnostic_penalty(
                    diagnostics, diagnostic_weight
                )
                _assess_candidate_quality(
                    candidate,
                    degeneracy_weight=degeneracy_weight,
                )
                if oos_split is not None:
                    _, holdout_start, holdout_stop = oos_split
                    if oos_refit is None:
                        losses = _compute_oos_losses(
                            result,
                            data,
                            full_x=fit_kwargs.get("x"),
                            start=holdout_start,
                            stop=holdout_stop,
                            loss=oos_loss,
                        )
                    else:
                        losses = _compute_rolling_oos_losses(
                            spec,
                            result,
                            data,
                            full_x=fit_kwargs.get("x"),
                            start=holdout_start,
                            stop=holdout_stop,
                            loss=oos_loss,
                            refit_every=int(oos_refit),
                            fit_kwargs=fit_kwargs,
                        )
                    if not np.all(np.isfinite(losses)):
                        raise FloatingPointError(
                            "OOS filter produced non-finite losses."
                        )
                    candidate.oos_losses = losses
                    candidate.oos_loss = float(np.mean(losses))
                    candidate.oos_loss_name = oos_loss.lower()
                    candidate.oos_start = holdout_start
                    candidate.oos_stop = holdout_stop
                    candidate.score = candidate.oos_loss
                else:
                    candidate.score = score + (
                        candidate.degeneracy_penalty
                        if selection_mode == "aic+diag" else 0.0
                    )

                if diagnostics is not None:
                    candidate.dgt_passed = not diagnostics["dgt"]["reject"]
                    candidate.lb_failures = sum(
                        1 for lb in diagnostics["ljung_box"].values()
                        if lb["reject"]
                    )
                if selection_verbose:
                    print(
                        f"AIC={candidate.aic:.2f}, Score={candidate.score:.2f}"
                    )
            except Exception as exc:
                candidate.score = np.inf
                if oos_split is not None:
                    candidate.oos_loss = np.inf
                candidate.error_message = str(exc)
                if selection_verbose:
                    print(f"FAILED: {exc}")
            candidate.fit_time = time.perf_counter() - start_time
            candidates.append(candidate)

        if staged_mean_search and phase_index == 0:
            ranked_screen = sorted(candidates, key=lambda candidate: candidate.score)
            finalist_signatures = []
            for candidate in ranked_screen:
                if not np.isfinite(candidate.score):
                    continue
                if candidate.mean_signature not in finalist_signatures:
                    finalist_signatures.append(candidate.mean_signature)
                if len(finalist_signatures) == allowed_finalists:
                    break
            finalist_means = [
                mean_candidate for mean_candidate in mean_candidates
                if _mean_signature(mean_candidate) in finalist_signatures
            ]
            baseline_task_keys = {
                (candidate.mean_signature, candidate.vol_type, candidate.p,
                 candidate.q, candidate.density)
                for candidate in candidates
            }
            phases.append([
                (mean_candidate, vol_type, p, q, density_name)
                for mean_candidate in finalist_means
                for vol_type, p, q in vol_candidates
                for density_name in density_candidates
                if (
                    _mean_signature(mean_candidate), vol_type, p, q, density_name
                ) not in baseline_task_keys
            ])
        phase_index += 1
    
    # Sort by score (lower is better)
    candidates.sort(key=lambda c: c.score)
    _finalize_candidate_ranking(candidates)
    if selection_mode == "oos":
        _finalize_oos_comparisons(candidates)
    
    # Find best successful candidate
    best = None
    for c in candidates:
        if c.result is not None and np.isfinite(c.score):
            best = c
            break
    
    if best is None or best.result is None:
        raise RuntimeError(
            "All candidate models failed to fit. "
            f"Errors: {[c.error_message for c in candidates if c.error_message]}"
        )
    
    if selection_mode == "oos":
        fit_kwargs_copy = fit_kwargs.copy()
        fit_kwargs_copy["n_jobs"] = 1
        final_result = best.spec.fit(data, **fit_kwargs_copy)
    else:
        final_result = best.result

    search_space = {
        "mean_candidates": [
            _mean_signature(candidate) for candidate in mean_candidates
        ],
        "mean_auto": bool(mean_auto),
        "mean_search_staged": staged_mean_search,
        "mean_finalists": sorted({
            candidate.mean_signature
            for candidate in candidates
            if candidate.mean_signature is not None
            and (
                not staged_mean_search
                or (candidate.vol_type, candidate.p, candidate.q, candidate.density)
                != (*baseline_vol, baseline_density)
            )
        }),
        "vol_candidates": list(vol_candidates),
        "density_candidates": list(density_candidates),
        "n_candidates": len(candidates),
        "full_grid_candidates": full_total,
        "candidate_budget": candidate_budget if mean_auto else None,
        "warm_start": bool(selection_warm_start),
        "degeneracy_weight": float(degeneracy_weight),
        "oos_refit": oos_refit,
    }
    selection = SelectionResult(
        candidates,
        select_by=selection_mode,
        criterion_name=criterion_name if selection_mode != "oos" else f"oos:{oos_loss.lower()}",
        search_space=search_space,
    )
    final_result.selection = selection
    final_result._selection_candidates = candidates
    final_result.fit_info.selected_by = selection.criterion_name
    final_result.fit_info.selection_search_space = search_space
    final_result.fit_info.selection_refit_on_full = selection_mode == "oos"
    if oos_split is not None:
        train_stop, holdout_start, holdout_stop = oos_split
        final_result.fit_info.oos_train_size = train_stop
        final_result.fit_info.oos_test_size = holdout_stop - holdout_start
        final_result.fit_info.oos_loss = oos_loss.lower()
        final_result.fit_info.oos_refit = oos_refit

    if selection_verbose:
        print(f"\nBest model: {best.spec} (Score={best.score:.2f})")

    return final_result, candidates
