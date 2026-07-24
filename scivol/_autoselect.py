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
from scipy.special import gammaln

if TYPE_CHECKING:
    from .spec import CompositeSpec
    from .result import EstimationResult

# Type alias for the criterion callable
CriterionFunc = Callable[["EstimationResult", Optional[Dict[str, Any]]], float]


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
    delta_score: float = np.inf
    weight: float = 0.0


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
                }
            )
        return pd.DataFrame(rows)

    def summary(self, top_n: int = 10) -> None:
        _print_selection_summary(self.candidates, top_n=top_n, select_by=self.select_by)


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
        )
        return -ll
    realized_var = resid * resid
    if key == "qlike":
        return np.log(sigma2) + realized_var / sigma2
    if key == "mse":
        return (realized_var - sigma2) ** 2
    raise ValueError("oos_loss must be 'logscore', 'qlike', or 'mse'.")


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


# =====================================================================
# Volatility / density class maps
# =====================================================================

def _get_vol_map() -> Dict[str, Any]:
    """Return mapping of volatility type names to classes."""
    from .components.vol import EGARCH, GARCH, GJRGARCH
    return {'GARCH': GARCH, 'GJRGARCH': GJRGARCH, 'EGARCH': EGARCH}


def _get_density_map() -> Dict[str, Any]:
    """Return mapping of density names to classes."""
    from .components.density import GED, Normal, StudentT, SkewT
    return {'Normal': Normal, 'StudentT': StudentT, 'SkewT': SkewT, 'GED': GED}


# =====================================================================
# Main selection entry point
# =====================================================================

def select_best_model(
    data: np.ndarray,
    vol_candidates: List[Tuple[str, int, int]],
    density_candidates: List[str],
    *,
    select_by: str = "aic+diag",
    criterion: Optional[CriterionFunc] = None,
    diagnostic_kwargs: Optional[Dict[str, Any]] = None,
    diagnostic_weight: float = 50.0,
    selection_verbose: bool = False,
    show_progress: bool = False,
    n_jobs: Optional[int] = None,
    oos_loss: str = "logscore",
    oos_test_size: float | int = 0.2,
    min_train_size: int = 250,
    min_test_size: int = 50,
    allow_density_blind_loss: bool = False,
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
        List of distribution names: ``'Normal'``, ``'StudentT'``, ``'SkewT'``.
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

    requested_hold_back = int(fit_kwargs.get("hold_back", 0) or 0)
    common_sample = fit_kwargs.get("common_sample", None)
    if common_sample is not False and vol_candidates:
        common_hold_back = max(max(p, q) for _, p, q in vol_candidates)
        fit_kwargs["hold_back"] = max(requested_hold_back, common_hold_back)
        fit_kwargs["common_sample"] = True
    else:
        fit_kwargs["hold_back"] = requested_hold_back
        fit_kwargs["common_sample"] = False
    
    total = len(vol_candidates) * len(density_candidates)
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
            min_train_size=min_train_size,
            min_test_size=min_test_size,
            allow_density_blind_loss=allow_density_blind_loss,
            diagnostic_weight=diagnostic_weight,
            **fit_kwargs,
        )
    
    # Sequential execution
    from ._progress import get_progress_bar

    candidates: List[ModelCandidate] = []
    
    if selection_verbose:
        print(f"Auto-selecting from {total} candidate models...")
    
    # Build flat task list for progress bar wrapping
    _tasks = [
        (vol_type, p, q, density_name)
        for vol_type, p, q in vol_candidates
        for density_name in density_candidates
    ]

    for idx, (vol_type, p, q, density_name) in enumerate(
        get_progress_bar(_tasks, total=total, desc="Auto-selecting", disable=not show_progress)
    ):
        # Build spec
        vol_cls = vol_map.get(vol_type)
        density_cls = density_map.get(density_name)
        if vol_cls is None:
            warnings.warn(f"Unknown volatility model '{vol_type}', skipping.")
            continue
        if density_cls is None:
            warnings.warn(f"Unknown density '{density_name}', skipping.")
            continue
        
        spec = vol_cls(p, q) + density_cls()
        candidate = ModelCandidate(
            spec=spec,
            vol_type=vol_type,
            p=p,
            q=q,
            density=density_name,
        )
        
        if selection_verbose:
            print(f"  [{idx + 1}/{total}] Fitting {spec}...", end=" ")
        
        start_time = time.perf_counter()
        
        try:
            # Force sequential for nested fits
            fit_kwargs_copy = fit_kwargs.copy()
            fit_kwargs_copy['n_jobs'] = 1
            fit_data = data
            if oos_split is not None:
                train_stop, _, _ = oos_split
                fit_data = data[:train_stop]
                if "x" in fit_kwargs_copy:
                    fit_kwargs_copy["x"] = _split_x(fit_kwargs_copy["x"], train_stop)
            result = spec.fit(fit_data, **fit_kwargs_copy)
            candidate.result = result
            candidate.aic = result.aic
            candidate.bic = result.bic
            candidate.hqic = result.hqic
            candidate.fit_time = time.perf_counter() - start_time
            
            score, diagnostics = _score_candidate(
                result, criterion_func, diagnostic_kwargs
            )
            candidate.diagnostics = diagnostics
            candidate.diagnostic_penalty = _diagnostic_penalty(diagnostics, diagnostic_weight)
            if oos_split is not None:
                _, holdout_start, holdout_stop = oos_split
                losses = _compute_oos_losses(
                    result,
                    data,
                    full_x=fit_kwargs.get("x"),
                    start=holdout_start,
                    stop=holdout_stop,
                    loss=oos_loss,
                )
                if not np.all(np.isfinite(losses)):
                    raise FloatingPointError("OOS filter produced non-finite losses.")
                candidate.oos_losses = losses
                candidate.oos_loss = float(np.mean(losses))
                candidate.oos_loss_name = oos_loss.lower()
                candidate.oos_start = holdout_start
                candidate.oos_stop = holdout_stop
                candidate.score = candidate.oos_loss
            else:
                candidate.score = score
            
            # Populate convenience fields from diagnostics
            if diagnostics is not None:
                candidate.dgt_passed = not diagnostics['dgt']['reject']
                candidate.lb_failures = sum(
                    1 for lb in diagnostics['ljung_box'].values()
                    if lb['reject']
                )
            
            if selection_verbose:
                print(f"AIC={candidate.aic:.2f}, Score={candidate.score:.2f}")
                
        except Exception as e:
            candidate.fit_time = time.perf_counter() - start_time
            candidate.score = np.inf
            if oos_split is not None:
                candidate.oos_loss = np.inf
            candidate.error_message = str(e)
            
            if selection_verbose:
                print(f"FAILED: {e}")
        
        candidates.append(candidate)
    
    # Sort by score (lower is better)
    candidates.sort(key=lambda c: c.score)
    _finalize_candidate_ranking(candidates)
    
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
        "vol_candidates": list(vol_candidates),
        "density_candidates": list(density_candidates),
        "n_candidates": total,
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

    if selection_verbose:
        print(f"\nBest model: {best.spec} (Score={best.score:.2f})")

    return final_result, candidates
