# volkit/_autoselect.py
"""
Automatic model selection engine for volkit.

This module provides functionality to automatically select optimal model
specifications based on a user-supplied criterion callable or the default
blended AIC + diagnostic test criterion.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, List, Tuple, Optional, Dict, Any, TYPE_CHECKING
import numpy as np
import warnings

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
    diagnostic_penalty: float = 0.0
    score: float = np.inf
    diagnostics: Optional[Dict[str, Any]] = None
    fit_time: float = 0.0
    error_message: Optional[str] = None
    dgt_passed: bool = False
    lb_failures: int = 0


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


# =====================================================================
# Volatility / density class maps
# =====================================================================

def _get_vol_map() -> Dict[str, Any]:
    """Return mapping of volatility type names to classes."""
    from .components.vol import GARCH, GJRGARCH
    return {'GARCH': GARCH, 'GJRGARCH': GJRGARCH}


def _get_density_map() -> Dict[str, Any]:
    """Return mapping of density names to classes."""
    from .components.density import Normal, StudentT, SkewT
    return {'Normal': Normal, 'StudentT': StudentT, 'SkewT': SkewT}


# =====================================================================
# Main selection entry point
# =====================================================================

def select_best_model(
    data: np.ndarray,
    vol_candidates: List[Tuple[str, int, int]],
    density_candidates: List[str],
    *,
    criterion: Optional[CriterionFunc] = None,
    diagnostic_kwargs: Optional[Dict[str, Any]] = None,
    diagnostic_weight: float = 50.0,
    verbose: bool = False,
    show_progress: bool = False,
    n_jobs: Optional[int] = None,
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
    verbose : bool, default False
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
    
    # Build the criterion callable
    if criterion is None:
        criterion = make_default_criterion(diagnostic_weight)
    
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
    
    total = len(vol_candidates) * len(density_candidates)
    n_jobs_actual = n_jobs if n_jobs is not None else get_default_workers()
    
    # Use parallel execution if n_jobs > 1 and multiple candidates
    if n_jobs_actual > 1 and total > 2:
        from ._parallel import select_best_parallel
        return select_best_parallel(
            data,
            vol_candidates,
            density_candidates,
            criterion=criterion,
            diagnostic_kwargs=diagnostic_kwargs,
            n_jobs=n_jobs_actual,
            verbose=verbose,
            show_progress=show_progress,
            **fit_kwargs,
        )
    
    # Sequential execution
    from ._progress import get_progress_bar

    candidates: List[ModelCandidate] = []
    
    if verbose:
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
        candidate = ModelCandidate(spec=spec)
        
        if verbose:
            print(f"  [{idx + 1}/{total}] Fitting {spec}...", end=" ")
        
        start_time = time.perf_counter()
        
        try:
            # Force sequential for nested fits
            fit_kwargs_copy = fit_kwargs.copy()
            fit_kwargs_copy['n_jobs'] = 1
            result = spec.fit(data, **fit_kwargs_copy)
            candidate.result = result
            candidate.aic = result.aic
            candidate.fit_time = time.perf_counter() - start_time
            
            # Score via criterion
            score, diagnostics = _score_candidate(
                result, criterion, diagnostic_kwargs
            )
            candidate.score = score
            candidate.diagnostics = diagnostics
            
            # Populate convenience fields from diagnostics
            if diagnostics is not None:
                candidate.dgt_passed = not diagnostics['dgt']['reject']
                candidate.lb_failures = sum(
                    1 for lb in diagnostics['ljung_box'].values()
                    if lb['reject']
                )
            
            if verbose:
                print(f"AIC={candidate.aic:.2f}, Score={candidate.score:.2f}")
                
        except Exception as e:
            candidate.fit_time = time.perf_counter() - start_time
            candidate.score = np.inf
            candidate.error_message = str(e)
            
            if verbose:
                print(f"FAILED: {e}")
        
        candidates.append(candidate)
    
    # Sort by score (lower is better)
    candidates.sort(key=lambda c: c.score)
    
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
    
    if verbose:
        print(f"\nBest model: {best.spec} (Score={best.score:.2f})")
    
    return best.result, candidates
