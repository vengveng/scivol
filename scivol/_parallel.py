# scivol/_parallel.py
"""
Parallel fitting utilities for scivol.

This module provides functions for fitting multiple time series in parallel
using joblib for robust cross-platform parallelism.
"""
from __future__ import annotations

import os
from typing import Callable, Dict, List, Tuple, Any, Optional
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


def _get_vol_map() -> Dict[str, Any]:
    """Return mapping of volatility type names to classes."""
    from .components.vol import GARCH, GJRGARCH
    return {'GARCH': GARCH, 'GJRGARCH': GJRGARCH}


def _get_density_map() -> Dict[str, Any]:
    """Return mapping of density names to classes."""
    from .components.density import Normal, StudentT, SkewT
    return {'Normal': Normal, 'StudentT': StudentT, 'SkewT': SkewT}


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
    from .components.density import AutoDensity
    from .components.mean import ARMA
    
    vol_map = _get_vol_map()
    density_map = _get_density_map()
    
    # Determine volatility class from vol_type key
    vol_type = params.get('vol_type', 'GARCH')
    vol_cls = vol_map.get(vol_type, vol_map['GARCH'])
    
    # Build volatility component
    if params['vol'] is not None:
        if params['vol_auto']:
            vol = vol_cls(auto=params['vol_auto'])
        else:
            vol = vol_cls(*params['vol'])
    else:
        vol = vol_cls(1, 1)  # Default
    
    # Build density component
    if params['density_auto']:
        candidates = params.get('density_candidates')
        density = AutoDensity(candidates=candidates)
    else:
        density_name = params.get('density', 'Normal')
        density = density_map.get(density_name, density_map['Normal'])()
    
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
    show_progress: bool = False,
    **fit_kwargs: Any,
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
    show_progress : bool, default False
        Show tqdm progress bar.
    **fit_kwargs
        Additional arguments passed to spec.fit()
        
    Returns
    -------
    Dict[str, EstimationResult]
        Mapping of series names to fitted results
    """
    from ._progress import get_progress_bar, tqdm_joblib

    n_jobs = n_jobs if n_jobs is not None else get_default_workers()
    n_series = len(data_dict)
    
    if n_jobs == 1 or n_series == 1:
        # Sequential execution
        results = {}
        for name, series in get_progress_bar(
            data_dict.items(), total=n_series, desc="Fitting series", disable=not show_progress
        ):
            _, result = _fit_single_series_worker(
                name, series, spec_params, fit_kwargs, index
            )
            results[name] = result
        return results
    
    # Parallel execution with joblib
    actual_workers = min(n_jobs, n_series)
    
    with tqdm_joblib(total=n_series, desc="Fitting series", disable=not show_progress):
        parallel_results = Parallel(n_jobs=actual_workers, prefer="processes")(
            delayed(_fit_single_series_worker)(
                name, series, spec_params, fit_kwargs, index
            )
            for name, series in data_dict.items()
        )
    
    # Convert list of tuples to dict
    return {name: result for name, result in parallel_results}


def _fit_candidate_worker(
    vol_type: str,
    p: int,
    q: int,
    density_name: str,
    data: np.ndarray,
    fit_kwargs: Dict[str, Any],
    criterion: Callable[..., float],
    diagnostic_kwargs: Optional[Dict[str, Any]],
) -> Any:
    """
    Worker function for fitting a single model candidate (for auto-selection).
    
    Parameters
    ----------
    vol_type : str
        Volatility model type ('GARCH', 'GJRGARCH')
    p, q : int
        Volatility model order
    density_name : str
        Name of density ('Normal', 'StudentT', 'SkewT')
    data : np.ndarray
        Data to fit
    fit_kwargs : dict
        Additional fit arguments
    criterion : callable
        ``(result, diagnostics) -> float``
    diagnostic_kwargs : dict or None
        Keyword arguments for ``diagnostic_tests()``
        
    Returns
    -------
    ModelCandidate
    """
    import time
    from ._autoselect import ModelCandidate, _score_candidate
    
    vol_map = _get_vol_map()
    density_map = _get_density_map()
    
    vol_cls = vol_map.get(vol_type, vol_map['GARCH'])
    density_cls = density_map.get(density_name, density_map['Normal'])
    spec = vol_cls(p, q) + density_cls()
    
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
            
    except Exception as e:
        candidate.fit_time = time.perf_counter() - start_time
        candidate.score = float('inf')
        candidate.error_message = str(e)
    
    return candidate


def select_best_parallel(
    data: np.ndarray,
    vol_candidates: List[Tuple[str, int, int]],
    density_candidates: List[str],
    *,
    criterion: Callable[..., float],
    diagnostic_kwargs: Optional[Dict[str, Any]] = None,
    n_jobs: Optional[int] = None,
    verbose: bool = False,
    show_progress: bool = False,
    **fit_kwargs: Any,
) -> Tuple[Any, List[Any]]:
    """
    Parallel version of model selection using joblib.
    
    Parameters
    ----------
    data : np.ndarray
        1-D array of returns/residuals
    vol_candidates : list of tuples
        List of (vol_type, p, q) volatility model/order candidates to try
    density_candidates : list of str
        List of density names to try
    criterion : callable
        ``(result, diagnostics) -> float``
    diagnostic_kwargs : dict or None
        Keyword arguments for ``diagnostic_tests()``
    n_jobs : int, optional
        Number of workers
    verbose : bool
        Print detailed per-candidate results
    show_progress : bool
        Show tqdm progress bar
    **fit_kwargs
        Additional arguments for fitting
        
    Returns
    -------
    tuple
        (best_result, all_candidates)
    """
    from ._progress import get_progress_bar, tqdm_joblib

    n_jobs = n_jobs if n_jobs is not None else get_default_workers()
    
    # Build task list
    tasks = [
        (vol_type, p, q, d)
        for vol_type, p, q in vol_candidates
        for d in density_candidates
    ]
    
    total = len(tasks)
    
    if verbose:
        print(f"Auto-selecting from {total} candidate models (parallel, {n_jobs} workers)...")
    
    # Sequential fallback: process spawn overhead (~1-2 s) exceeds the
    # total computation time when candidate counts are small (≤100).
    _MIN_PARALLEL_CANDIDATES = 100
    if n_jobs == 1 or total < _MIN_PARALLEL_CANDIDATES:
        candidates = []
        for i, (vol_type, p, q, d) in enumerate(
            get_progress_bar(tasks, total=total, desc="Auto-selecting", disable=not show_progress)
        ):
            if verbose:
                print(f"  [{i+1}/{total}] Fitting {vol_type}({p},{q})+{d}...", end=" ")
            
            candidate = _fit_candidate_worker(
                vol_type, p, q, d, data, fit_kwargs,
                criterion, diagnostic_kwargs,
            )
            candidates.append(candidate)
            
            if verbose:
                if candidate.score < float('inf'):
                    print(f"AIC={candidate.aic:.2f}, Score={candidate.score:.2f}")
                else:
                    print(f"FAILED: {candidate.error_message}")
    else:
        # Parallel execution with joblib
        actual_workers = min(n_jobs, total)
        
        with tqdm_joblib(total=total, desc="Auto-selecting", disable=not show_progress):
            candidates = Parallel(n_jobs=actual_workers, prefer="processes")(
                delayed(_fit_candidate_worker)(
                    vol_type, p, q, d, data, fit_kwargs,
                    criterion, diagnostic_kwargs,
                )
                for vol_type, p, q, d in tasks
            )
        
        if verbose:
            for i, (candidate, (vol_type, p, q, d)) in enumerate(zip(candidates, tasks)):
                if candidate.score < float('inf'):
                    print(f"  [{i+1}/{total}] {vol_type}({p},{q})+{d}: "
                          f"AIC={candidate.aic:.2f}, Score={candidate.score:.2f}")
                else:
                    print(f"  [{i+1}/{total}] {vol_type}({p},{q})+{d}: "
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
    vol_type: str,
    p: int,
    q: int,
    density_name: str,
    data: np.ndarray,
    fit_kwargs: Dict[str, Any],
    criterion: Callable[..., float],
    diagnostic_kwargs: Optional[Dict[str, Any]],
    index: Optional[Any],
) -> Tuple[str, Any]:
    """
    Worker for fitting a single (series, candidate) pair.
    
    Used by fit_multi_auto_parallel for flattened parallel execution.
    
    Parameters
    ----------
    series_name : str
        Name of the series
    vol_type : str
        Volatility model type ('GARCH', 'GJRGARCH')
    p, q : int
        Volatility model order
    density_name : str
        Density name
    data : np.ndarray
        Data for this series
    fit_kwargs : dict
        Additional fit arguments
    criterion : callable
        ``(result, diagnostics) -> float``
    diagnostic_kwargs : dict or None
        Keyword arguments for ``diagnostic_tests()``
    index : any
        Pandas index to attach
        
    Returns
    -------
    tuple
        (series_name, ModelCandidate)
    """
    import time
    from ._autoselect import ModelCandidate, _score_candidate
    
    vol_map = _get_vol_map()
    density_map = _get_density_map()
    
    vol_cls = vol_map.get(vol_type, vol_map['GARCH'])
    density_cls = density_map.get(density_name, density_map['Normal'])
    spec = vol_cls(p, q) + density_cls()
    
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
        
        # Score via criterion
        score, diagnostics = _score_candidate(
            result, criterion, diagnostic_kwargs
        )
        candidate.score = score
        candidate.diagnostics = diagnostics
        
        # Populate convenience fields
        if diagnostics is not None:
            candidate.dgt_passed = not diagnostics['dgt']['reject']
            candidate.lb_failures = sum(
                1 for lb in diagnostics['ljung_box'].values()
                if lb['reject']
            )
            
    except Exception as e:
        candidate.fit_time = time.perf_counter() - start_time
        candidate.score = float('inf')
        candidate.error_message = str(e)
    
    return series_name, candidate


def fit_multi_auto_parallel(
    data_dict: Dict[str, np.ndarray],
    vol_candidates: List[Tuple[str, int, int]],
    density_candidates: List[str],
    index: Optional[Any],
    *,
    criterion: Callable[..., float],
    diagnostic_kwargs: Optional[Dict[str, Any]] = None,
    n_jobs: Optional[int] = None,
    verbose: bool = False,
    show_progress: bool = False,
    **fit_kwargs: Any,
) -> Dict[str, Any]:
    """
    Fit multiple series with auto-selection using flattened parallelism.
    
    Instead of:  n_series tasks x sequential auto-selection
    We do:       n_series x n_candidates tasks in one flat pool
    
    This maximizes CPU utilization when fitting few series with many candidates.
    
    Parameters
    ----------
    data_dict : dict
        Mapping of series names to 1-D numpy arrays
    vol_candidates : list of tuples
        List of (vol_type, p, q) volatility model/order candidates to try
    density_candidates : list of str
        List of density names to try
    index : pandas Index or None
        Original index for results
    criterion : callable
        ``(result, diagnostics) -> float``
    diagnostic_kwargs : dict or None
        Keyword arguments for ``diagnostic_tests()``
    n_jobs : int, optional
        Number of workers
    verbose : bool
        Print detailed per-candidate results
    show_progress : bool
        Show tqdm progress bar
    **fit_kwargs
        Additional fitting arguments
        
    Returns
    -------
    Dict[str, EstimationResult]
        Best result for each series
    """
    from ._progress import tqdm_joblib

    n_jobs = n_jobs if n_jobs is not None else get_default_workers()
    
    # Build flattened task list: (series x candidates)
    tasks = [
        (series_name, vol_type, p, q, d, data)
        for series_name, data in data_dict.items()
        for vol_type, p, q in vol_candidates
        for d in density_candidates
    ]
    
    total = len(tasks)
    n_series = len(data_dict)
    n_candidates_per = len(vol_candidates) * len(density_candidates)
    
    if verbose:
        print(f"Auto-selecting {n_series} series x {n_candidates_per} candidates = {total} fits "
              f"(parallel, {min(n_jobs, total)} workers)...")
    
    # Parallel execution with joblib
    actual_workers = min(n_jobs, total)
    
    with tqdm_joblib(total=total, desc="Auto-selecting (multi)", disable=not show_progress):
        parallel_results = Parallel(n_jobs=actual_workers, prefer="processes")(
            delayed(_fit_series_candidate_worker)(
                series_name, vol_type, p, q, d, data, fit_kwargs,
                criterion, diagnostic_kwargs, index,
            )
            for series_name, vol_type, p, q, d, data in tasks
        )
    
    # Group results by series
    series_candidates: Dict[str, List[Any]] = {name: [] for name in data_dict.keys()}
    
    for i, (series_name, candidate) in enumerate(parallel_results):
        series_candidates[series_name].append(candidate)
        
        if verbose:
            task = tasks[i]
            vol_type, p, q, d = task[1], task[2], task[3], task[4]
            if candidate.score < float('inf'):
                print(f"  [{i+1}/{total}] {series_name}:{vol_type}({p},{q})+{d}: "
                      f"AIC={candidate.aic:.2f}, Score={candidate.score:.2f}")
            else:
                print(f"  [{i+1}/{total}] {series_name}:{vol_type}({p},{q})+{d}: "
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
