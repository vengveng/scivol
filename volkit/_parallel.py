# volkit/_parallel.py
"""
Parallel fitting utilities for volkit.

This module provides functions for fitting multiple time series in parallel
using joblib for robust cross-platform parallelism.
"""
from __future__ import annotations

import os
from typing import Dict, List, Tuple, Any, Optional
import numpy as np

# Use joblib for parallelism (handles spawn/fork issues gracefully)
from joblib import Parallel, delayed


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


def _fit_single_series_worker(
    name: str,
    series_data: np.ndarray,
    spec_params: Dict[str, Any],
    fit_kwargs: Dict[str, Any],
    index: Optional[Any],
) -> Tuple[str, Any]:
    """
    Worker function for fitting a single series.
    
    Parameters
    ----------
    name : str
        Series name
    series_data : np.ndarray
        1-D array of data
    spec_params : dict
        Serialized spec parameters
    fit_kwargs : dict
        Additional fit arguments
    index : any
        Pandas index to attach
        
    Returns
    -------
    tuple
        (name, result)
    """
    # Reconstruct spec in worker process
    spec = _reconstruct_spec(spec_params)
    
    # Fit the model (n_jobs=1 to avoid nested parallelism)
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
    Fit multiple series in parallel using joblib.
    
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
    
    if n_jobs == 1 or n_series == 1:
        # Sequential execution
        results = {}
        for name, series in data_dict.items():
            _, result = _fit_single_series_worker(
                name, series, spec_params, fit_kwargs, index
            )
            results[name] = result
        return results
    
    # Parallel execution with joblib
    actual_workers = min(n_jobs, n_series)
    
    parallel_results = Parallel(n_jobs=actual_workers, prefer="processes")(
        delayed(_fit_single_series_worker)(
            name, series, spec_params, fit_kwargs, index
        )
        for name, series in data_dict.items()
    )
    
    # Convert list of tuples to dict
    return {name: result for name, result in parallel_results}


def _fit_candidate_worker(
    p: int,
    q: int,
    density_name: str,
    data: np.ndarray,
    fit_kwargs: Dict[str, Any],
    diagnostic_weight: float,
) -> Any:
    """
    Worker function for fitting a single model candidate (for auto-selection).
    
    Parameters
    ----------
    p, q : int
        GARCH order
    density_name : str
        Name of density ('Normal', 'StudentT', 'SkewT')
    data : np.ndarray
        Data to fit
    fit_kwargs : dict
        Additional fit arguments
    diagnostic_weight : float
        Penalty weight for failed diagnostics
        
    Returns
    -------
    ModelCandidate
    """
    import time
    from .components.vol import GARCH
    from .components.density import Normal, StudentT, SkewT
    from ._autoselect import ModelCandidate
    
    density_map = {
        'Normal': Normal,
        'StudentT': StudentT,
        'SkewT': SkewT,
    }
    
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
    Parallel version of model selection using joblib.
    
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
    n_jobs = n_jobs if n_jobs is not None else get_default_workers()
    
    # Build task list
    tasks = [
        (p, q, d)
        for p, q in vol_candidates
        for d in density_candidates
    ]
    
    total = len(tasks)
    
    if verbose:
        print(f"Auto-selecting from {total} candidate models (parallel, {n_jobs} workers)...")
    
    if n_jobs == 1 or total <= 2:
        # Sequential for small number of candidates
        candidates = []
        for i, (p, q, d) in enumerate(tasks):
            if verbose:
                print(f"  [{i+1}/{total}] Fitting GARCH({p},{q})+{d}...", end=" ")
            
            candidate = _fit_candidate_worker(p, q, d, data, fit_kwargs, diagnostic_weight)
            candidates.append(candidate)
            
            if verbose:
                if candidate.score < float('inf'):
                    print(f"AIC={candidate.aic:.2f}, Score={candidate.score:.2f}")
                else:
                    print(f"FAILED: {candidate.error_message}")
    else:
        # Parallel execution with joblib
        actual_workers = min(n_jobs, total)
        
        candidates = Parallel(n_jobs=actual_workers, prefer="processes")(
            delayed(_fit_candidate_worker)(p, q, d, data, fit_kwargs, diagnostic_weight)
            for p, q, d in tasks
        )
        
        if verbose:
            for i, (candidate, (p, q, d)) in enumerate(zip(candidates, tasks)):
                if candidate.score < float('inf'):
                    print(f"  [{i+1}/{total}] GARCH({p},{q})+{d}: "
                          f"AIC={candidate.aic:.2f}, Score={candidate.score:.2f}")
                else:
                    print(f"  [{i+1}/{total}] GARCH({p},{q})+{d}: "
                          f"FAILED: {candidate.error_message}")
    
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


def _fit_series_candidate_worker(
    series_name: str,
    p: int,
    q: int,
    density_name: str,
    data: np.ndarray,
    fit_kwargs: Dict[str, Any],
    diagnostic_weight: float,
    index: Optional[Any],
) -> Tuple[str, Any]:
    """
    Worker for fitting a single (series, candidate) pair.
    
    Used by fit_multi_auto_parallel for flattened parallel execution.
    
    Parameters
    ----------
    series_name : str
        Name of the series
    p, q : int
        GARCH order
    density_name : str
        Density name
    data : np.ndarray
        Data for this series
    fit_kwargs : dict
        Additional fit arguments
    diagnostic_weight : float
        Penalty for failed diagnostics
    index : any
        Pandas index to attach
        
    Returns
    -------
    tuple
        (series_name, ModelCandidate)
    """
    import time
    from .components.vol import GARCH
    from .components.density import Normal, StudentT, SkewT
    from ._autoselect import ModelCandidate
    
    density_map = {'Normal': Normal, 'StudentT': StudentT, 'SkewT': SkewT}
    density_cls = density_map.get(density_name, Normal)
    spec = GARCH(p, q) + density_cls()
    
    candidate = ModelCandidate(spec=spec)
    start_time = time.perf_counter()
    
    try:
        result = spec.fit(data, n_jobs=1, **fit_kwargs)
        # Attach pandas metadata
        result._index = index
        result._name = series_name
        
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
            
            lb_failures = sum(1 for lb in diag['ljung_box'].values() if lb['reject'])
            candidate.lb_failures = lb_failures
            n_failed += lb_failures
            
            candidate.diagnostic_penalty = n_failed * diagnostic_weight
            candidate.score = candidate.aic + candidate.diagnostic_penalty
        except Exception:
            candidate.score = candidate.aic + diagnostic_weight
            
    except Exception as e:
        candidate.fit_time = time.perf_counter() - start_time
        candidate.score = float('inf')
        candidate.error_message = str(e)
    
    return series_name, candidate


def fit_multi_auto_parallel(
    data_dict: Dict[str, np.ndarray],
    vol_candidates: List[Tuple[int, int]],
    density_candidates: List[str],
    index: Optional[Any],
    diagnostic_weight: float = 50.0,
    n_jobs: Optional[int] = None,
    verbose: bool = False,
    **fit_kwargs,
) -> Dict[str, Any]:
    """
    Fit multiple series with auto-selection using flattened parallelism.
    
    Instead of:  n_series tasks × sequential auto-selection
    We do:       n_series × n_candidates tasks in one flat pool
    
    This maximizes CPU utilization when fitting few series with many candidates.
    
    Parameters
    ----------
    data_dict : dict
        Mapping of series names to 1-D numpy arrays
    vol_candidates : list of tuples
        List of (p, q) GARCH orders to try
    density_candidates : list of str
        List of density names to try
    index : pandas Index or None
        Original index for results
    diagnostic_weight : float
        AIC penalty per failed diagnostic
    n_jobs : int, optional
        Number of workers
    verbose : bool
        Print progress
    **fit_kwargs
        Additional fitting arguments
        
    Returns
    -------
    Dict[str, EstimationResult]
        Best result for each series
    """
    n_jobs = n_jobs if n_jobs is not None else get_default_workers()
    
    # Build flattened task list: (series × candidates)
    tasks = [
        (series_name, p, q, d, data)
        for series_name, data in data_dict.items()
        for p, q in vol_candidates
        for d in density_candidates
    ]
    
    total = len(tasks)
    n_series = len(data_dict)
    n_candidates_per = len(vol_candidates) * len(density_candidates)
    
    if verbose:
        print(f"Auto-selecting {n_series} series × {n_candidates_per} candidates = {total} fits "
              f"(parallel, {min(n_jobs, total)} workers)...")
    
    # Parallel execution with joblib
    actual_workers = min(n_jobs, total)
    
    parallel_results = Parallel(n_jobs=actual_workers, prefer="processes")(
        delayed(_fit_series_candidate_worker)(
            series_name, p, q, d, data, fit_kwargs, diagnostic_weight, index
        )
        for series_name, p, q, d, data in tasks
    )
    
    # Group results by series
    series_candidates: Dict[str, List[Any]] = {name: [] for name in data_dict.keys()}
    
    for i, (series_name, candidate) in enumerate(parallel_results):
        series_candidates[series_name].append(candidate)
        
        if verbose:
            task = tasks[i]
            p, q, d = task[1], task[2], task[3]
            if candidate.score < float('inf'):
                print(f"  [{i+1}/{total}] {series_name}:GARCH({p},{q})+{d}: "
                      f"AIC={candidate.aic:.2f}, Score={candidate.score:.2f}")
            else:
                print(f"  [{i+1}/{total}] {series_name}:GARCH({p},{q})+{d}: "
                      f"FAILED: {candidate.error_message}")
    
    # Select best for each series
    results: Dict[str, Any] = {}
    
    for series_name, candidates in series_candidates.items():
        if not candidates:
            results[series_name] = {'error': 'No candidates fitted'}
            continue
        
        # Sort by score and pick best
        candidates.sort(key=lambda c: c.score)
        
        best = None
        for c in candidates:
            if c.result is not None and np.isfinite(c.score):
                best = c
                break
        
        if best is not None:
            # Attach selection metadata
            best.result._selection_candidates = candidates
            results[series_name] = best.result
            
            if verbose:
                print(f"\n  Best for {series_name}: {best.spec} (Score={best.score:.2f})")
        else:
            results[series_name] = {'error': 'All candidates failed'}
    
    return results
