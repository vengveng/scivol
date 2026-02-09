# volkit/_autoselect.py
"""
Automatic model selection engine for volkit.

This module provides functionality to automatically select optimal model
specifications based on a blended criterion of AIC and diagnostic tests.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict, Any, TYPE_CHECKING
import numpy as np
import warnings

if TYPE_CHECKING:
    from .spec import CompositeSpec
    from .result import EstimationResult


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
        Penalty from failed diagnostic tests.
    score : float
        Combined score = AIC + diagnostic_penalty (lower is better).
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
    fit_time: float = 0.0
    error_message: Optional[str] = None
    dgt_passed: bool = False
    lb_failures: int = 0


def select_best_model(
    data: np.ndarray,
    vol_candidates: List[Tuple[int, int]],
    density_candidates: List[str],
    diagnostic_weight: float = 50.0,
    verbose: bool = False,
    n_jobs: Optional[int] = None,
    **fit_kwargs,
) -> Tuple[EstimationResult, List[ModelCandidate]]:
    """
    Fit all candidate models and select best by blended criterion.
    
    The selection criterion is:
        Score = AIC + diagnostic_weight * n_failed_tests
    
    Where n_failed_tests includes:
    - 1 if DGT (Diebold-Gunther-Tay) test fails (PIT not uniform)
    - 1 for each Ljung-Box moment test that fails
    
    Parameters
    ----------
    data : np.ndarray
        1-D array of returns/residuals to fit.
    vol_candidates : List[Tuple[int, int]]
        List of (p, q) tuples for GARCH lag orders to try.
    density_candidates : List[str]
        List of distribution names to try: 'Normal', 'StudentT', 'SkewT'.
    diagnostic_weight : float, default 50.0
        AIC penalty per failed diagnostic test.
    verbose : bool, default False
        If True, print progress during model selection.
    n_jobs : int, optional
        Number of parallel workers. Default (None) uses all CPU cores.
        Set to 1 for sequential execution.
    **fit_kwargs
        Additional keyword arguments passed to spec.fit().
        
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
    from .components.vol import GARCH
    from .components.density import Normal, StudentT, SkewT
    from ._parallel import get_default_workers
    
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
            diagnostic_weight=diagnostic_weight,
            n_jobs=n_jobs_actual,
            verbose=verbose,
            **fit_kwargs,
        )
    
    # Sequential execution (original code path)
    # Map density names to classes
    density_map = {
        'Normal': Normal,
        'StudentT': StudentT,
        'SkewT': SkewT,
    }
    
    candidates: List[ModelCandidate] = []
    
    if verbose:
        print(f"Auto-selecting from {total} candidate models...")
    
    for i, (p, q) in enumerate(vol_candidates):
        for j, density_name in enumerate(density_candidates):
            # Build spec
            density_cls = density_map.get(density_name)
            if density_cls is None:
                warnings.warn(f"Unknown density '{density_name}', skipping.")
                continue
            
            spec = GARCH(p, q) + density_cls()
            candidate = ModelCandidate(spec=spec)
            
            if verbose:
                idx = i * len(density_candidates) + j + 1
                print(f"  [{idx}/{total}] Fitting {spec}...", end=" ")
            
            start_time = time.perf_counter()
            
            try:
                # Force sequential for nested fits
                fit_kwargs_copy = fit_kwargs.copy()
                fit_kwargs_copy['n_jobs'] = 1
                result = spec.fit(data, **fit_kwargs_copy)
                candidate.result = result
                candidate.aic = result.aic
                candidate.fit_time = time.perf_counter() - start_time
                
                # Run diagnostics
                try:
                    diag = result.diagnostic_tests(print_results=False)
                    
                    # Count failures
                    n_failed = 0
                    candidate.dgt_passed = not diag['dgt']['reject']
                    if diag['dgt']['reject']:
                        n_failed += 1
                    
                    lb_failures = sum(
                        1 for lb in diag['ljung_box'].values() if lb['reject']
                    )
                    candidate.lb_failures = lb_failures
                    n_failed += lb_failures
                    
                    candidate.diagnostic_penalty = n_failed * diagnostic_weight
                    candidate.score = candidate.aic + candidate.diagnostic_penalty
                    
                except Exception as diag_err:
                    # Diagnostics failed, use AIC only with small penalty
                    candidate.score = candidate.aic + diagnostic_weight
                    candidate.error_message = f"Diagnostics failed: {diag_err}"
                
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
