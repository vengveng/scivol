# volkit/_parallel.py
"""
Parallel fitting utilities for volkit.

This module provides functions for fitting multiple time series in parallel
using ProcessPoolExecutor.
"""
from __future__ import annotations

import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Dict, List, Tuple, Any, Optional
import numpy as np


def get_default_workers() -> int:
    """
    Get default number of workers (CPU count).
    
    Returns
    -------
    int
        Number of CPU cores, minimum 1.
    """
    return os.cpu_count() or 1


def _reconstruct_spec(params: Dict[str, Any]) -> Any:
    """
    Reconstruct a CompositeSpec from serialized parameters.
    
    This is called in worker processes to avoid pickling issues
    with component objects.
    
    Parameters
    ----------
    params : dict
        Serialized spec parameters from _spec_to_params()
        
    Returns
    -------
    CompositeSpec
    """
    from .components.vol import GARCH
    from .components.density import Normal, StudentT, SkewT, AutoDensity
    from .components.mean import ARMA
    
    # Build volatility component
    if params['vol'] is not None:
        if params['vol_auto']:
            vol = GARCH(auto=params['vol_auto'])
        else:
            vol = GARCH(*params['vol'])
    else:
        vol = GARCH(1, 1)  # Default
    
    # Build density component
    density_map = {'Normal': Normal, 'StudentT': StudentT, 'SkewT': SkewT}
    if params['density_auto']:
        candidates = params.get('density_candidates')
        density = AutoDensity(candidates=candidates)
    else:
        density_name = params.get('density', 'Normal')
        density = density_map.get(density_name, Normal)()
    
    # Build mean component if present
    if params.get('mean') is not None:
        mean = ARMA(*params['mean'])
        return mean + vol + density
    
    return vol + density


def _fit_single_series(args: Tuple) -> Tuple[str, Any]:
    """
    Worker function for fitting a single series.
    
    This function is called by ProcessPoolExecutor.
    
    Parameters
    ----------
    args : tuple
        (name, series_data, spec_params, fit_kwargs, index)
        
    Returns
    -------
    tuple
        (name, result)
    """
    name, series_data, spec_params, fit_kwargs, index = args
    
    # Reconstruct spec in worker process
    spec = _reconstruct_spec(spec_params)
    
    # Fit the model (don't pass n_jobs to avoid nested parallelism)
    fit_kwargs_copy = fit_kwargs.copy()
    fit_kwargs_copy.pop('n_jobs', None)
    
    result = spec.fit(series_data, n_jobs=1, **fit_kwargs_copy)
    
    # Attach metadata
    result._index = index
    result._name = name
    
    return name, result


def fit_multi_parallel(
    spec_params: Dict[str, Any],
    data_dict: Dict[str, np.ndarray],
    index: Optional[Any],
    n_jobs: Optional[int] = None,
    **fit_kwargs,
) -> Dict[str, Any]:
    """
    Fit multiple series in parallel using ProcessPoolExecutor.
    
    Parameters
    ----------
    spec_params : dict
        Serialized spec parameters from _spec_to_params()
    data_dict : dict
        Mapping of series names to 1-D numpy arrays
    index : pandas Index or None
        Original index to attach to results
    n_jobs : int, optional
        Number of workers. Default uses cpu_count().
    **fit_kwargs
        Additional arguments passed to spec.fit()
        
    Returns
    -------
    Dict[str, EstimationResult]
        Mapping of series names to fitted results
    """
    n_jobs = n_jobs if n_jobs is not None else get_default_workers()
    n_series = len(data_dict)
    
    # Build task list
    tasks = [
        (name, series, spec_params, fit_kwargs, index)
        for name, series in data_dict.items()
    ]
    
    results: Dict[str, Any] = {}
    
    if n_jobs == 1 or n_series == 1:
        # Sequential execution
        for task in tasks:
            name, result = _fit_single_series(task)
            results[name] = result
    else:
        # Parallel execution
        # Limit workers to number of series
        actual_workers = min(n_jobs, n_series)
        
        with ProcessPoolExecutor(max_workers=actual_workers) as executor:
            futures = {
                executor.submit(_fit_single_series, task): task[0] 
                for task in tasks
            }
            
            for future in as_completed(futures):
                name = futures[future]
                try:
                    _, result = future.result()
                    results[name] = result
                except Exception as e:
                    # Store error message instead of result
                    results[name] = {'error': str(e)}
    
    return results


def _fit_candidate_worker(args: Tuple) -> Any:
    """
    Worker function for fitting a single model candidate (for auto-selection).
    
    Parameters
    ----------
    args : tuple
        (p, q, density_name, data, fit_kwargs, diagnostic_weight)
        
    Returns
    -------
    ModelCandidate
    """
    import time
    from .components.vol import GARCH
    from .components.density import Normal, StudentT, SkewT
    from ._autoselect import ModelCandidate
    
    p, q, density_name, data, fit_kwargs, diagnostic_weight = args
    
    # Map density names to classes
    density_map = {
        'Normal': Normal,
        'StudentT': StudentT,
        'SkewT': SkewT,
    }
    
    # Build spec
    density_cls = density_map.get(density_name, Normal)
    spec = GARCH(p, q) + density_cls()
    
    candidate = ModelCandidate(spec=spec)
    start_time = time.perf_counter()
    
    try:
        # Don't pass n_jobs to avoid nested parallelism
        fit_kwargs_copy = fit_kwargs.copy()
        fit_kwargs_copy.pop('n_jobs', None)
        
        result = spec.fit(data, n_jobs=1, **fit_kwargs_copy)
        candidate.result = result
        candidate.aic = result.aic
        candidate.fit_time = time.perf_counter() - start_time
        
        # Run diagnostics
        try:
            diag = result.diagnostic_tests(print_results=False)
            
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
            candidate.score = candidate.aic + diagnostic_weight
            candidate.error_message = f"Diagnostics failed: {diag_err}"
            
    except Exception as e:
        candidate.fit_time = time.perf_counter() - start_time
        candidate.score = float('inf')
        candidate.error_message = str(e)
    
    return candidate


def select_best_parallel(
    data: np.ndarray,
    vol_candidates: List[Tuple[int, int]],
    density_candidates: List[str],
    diagnostic_weight: float = 50.0,
    n_jobs: Optional[int] = None,
    verbose: bool = False,
    **fit_kwargs,
) -> Tuple[Any, List[Any]]:
    """
    Parallel version of model selection.
    
    Parameters
    ----------
    data : np.ndarray
        1-D array of returns/residuals
    vol_candidates : list of tuples
        List of (p, q) GARCH orders to try
    density_candidates : list of str
        List of density names to try
    diagnostic_weight : float
        AIC penalty per failed diagnostic
    n_jobs : int, optional
        Number of workers
    verbose : bool
        Print progress
    **fit_kwargs
        Additional arguments for fitting
        
    Returns
    -------
    tuple
        (best_result, all_candidates)
    """
    import numpy as np
    
    n_jobs = n_jobs if n_jobs is not None else get_default_workers()
    
    # Build task list
    tasks = [
        (p, q, d, data, fit_kwargs, diagnostic_weight)
        for p, q in vol_candidates
        for d in density_candidates
    ]
    
    total = len(tasks)
    
    if verbose:
        print(f"Auto-selecting from {total} candidate models (parallel, {n_jobs} workers)...")
    
    candidates = []
    
    if n_jobs == 1 or total <= 2:
        # Sequential for small number of candidates
        for i, task in enumerate(tasks):
            if verbose:
                print(f"  [{i+1}/{total}] Fitting GARCH({task[0]},{task[1]})+{task[2]}...", end=" ")
            
            candidate = _fit_candidate_worker(task)
            candidates.append(candidate)
            
            if verbose:
                if candidate.score < float('inf'):
                    print(f"AIC={candidate.aic:.2f}, Score={candidate.score:.2f}")
                else:
                    print(f"FAILED: {candidate.error_message}")
    else:
        # Parallel execution with as_completed for better load balancing
        # (StudentT/SkewT take longer than Normal, higher p,q take longer)
        actual_workers = min(n_jobs, total)
        completed = 0
        
        with ProcessPoolExecutor(max_workers=actual_workers) as executor:
            # Submit all tasks and track which future maps to which task
            future_to_task = {
                executor.submit(_fit_candidate_worker, task): task
                for task in tasks
            }
            
            # Collect results as they complete (dynamic work stealing)
            for future in as_completed(future_to_task):
                task = future_to_task[future]
                completed += 1
                
                try:
                    candidate = future.result()
                    candidates.append(candidate)
                    
                    if verbose:
                        if candidate.score < float('inf'):
                            print(f"  [{completed}/{total}] GARCH({task[0]},{task[1]})+{task[2]}: "
                                  f"AIC={candidate.aic:.2f}, Score={candidate.score:.2f}")
                        else:
                            print(f"  [{completed}/{total}] GARCH({task[0]},{task[1]})+{task[2]}: "
                                  f"FAILED: {candidate.error_message}")
                except Exception as e:
                    # Create failed candidate
                    from ._autoselect import ModelCandidate
                    from .components.vol import GARCH
                    from .components.density import Normal, StudentT, SkewT
                    
                    density_map = {'Normal': Normal, 'StudentT': StudentT, 'SkewT': SkewT}
                    spec = GARCH(task[0], task[1]) + density_map.get(task[2], Normal)()
                    candidate = ModelCandidate(spec=spec, error_message=str(e))
                    candidates.append(candidate)
                    
                    if verbose:
                        print(f"  [{completed}/{total}] GARCH({task[0]},{task[1]})+{task[2]}: "
                              f"FAILED: {e}")
    
    # Sort by score
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
