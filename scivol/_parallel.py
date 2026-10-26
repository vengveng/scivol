# scivol/_parallel.py
"""
Parallel fitting utilities for scivol.

This module provides functions for fitting multiple time series in parallel
using joblib with spawn-based workers for cross-platform parallelism.
"""
from __future__ import annotations

import os
import sys
from typing import Callable, Dict, List, Tuple, Any, Optional
import numpy as np

from . import _scivolapi as api

# Spawn avoids forking a process after BLAS or a development AD oracle has
# initialized native worker threads.
from joblib import Parallel, delayed
from joblib.externals.loky.backend.context import set_start_method

set_start_method("spawn", force=True)


def _parallel_preference() -> str:
    """Avoid process creation after a development AD oracle starts JAX."""
    return "threads" if "jax" in sys.modules else "processes"


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
    effective_hold_back = result.fit_info.effective_hold_back
    if effective_hold_back is None:
        effective_hold_back = int(fit_kwargs_copy.get("hold_back", 0) or 0)
    result._index = None if index is None else index[effective_hold_back:]
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
        parallel_results = Parallel(n_jobs=actual_workers, prefer=_parallel_preference())(
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
    select_by: str = "aic+diag",
    oos_split: Optional[Tuple[int, int, int]] = None,
    oos_loss: str = "logscore",
    diagnostic_weight: float = 50.0,
    degeneracy_weight: float = 50.0,
    oos_refit: Optional[int] = None,
    mean_candidate: Tuple[str, Tuple[Any, ...]] = ("Zero", ()),
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
    from ._autoselect import (
        ModelCandidate,
        _assess_candidate_quality,
        _build_candidate_spec,
        _candidate_record,
        _compute_oos_losses,
        _compute_rolling_oos_losses,
        _diagnostic_penalty,
        _score_candidate,
        _split_x,
    )
    
    spec = _build_candidate_spec(mean_candidate, vol_type, p, q, density_name)
    candidate = _candidate_record(
        spec, mean_candidate, vol_type, p, q, density_name
    )
    start_time = time.perf_counter()
    
    try:
        # Don't pass n_jobs to avoid nested parallelism
        fit_kwargs_copy = fit_kwargs.copy()
        fit_kwargs_copy.pop('n_jobs', None)
        
        fit_data = data
        if oos_split is not None:
            train_stop, _, _ = oos_split
            fit_data = data[:train_stop]
            if "x" in fit_kwargs_copy:
                fit_kwargs_copy["x"] = _split_x(fit_kwargs_copy["x"], train_stop)
        result = spec.fit(fit_data, n_jobs=1, **fit_kwargs_copy)
        candidate.result = result
        candidate.aic = result.aic
        candidate.bic = result.bic
        candidate.hqic = result.hqic
        candidate.fit_time = time.perf_counter() - start_time
        
        # Score via criterion
        score, diagnostics = _score_candidate(
            result, criterion, diagnostic_kwargs
        )
        candidate.diagnostics = diagnostics
        candidate.diagnostic_penalty = _diagnostic_penalty(diagnostics, diagnostic_weight)
        _assess_candidate_quality(candidate, degeneracy_weight=degeneracy_weight)
        if select_by == "oos" and oos_split is not None:
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
                raise FloatingPointError("OOS filter produced non-finite losses.")
            candidate.oos_losses = losses
            candidate.oos_loss = api.sample_mean(np.ascontiguousarray(losses, dtype=np.float64))
            candidate.oos_loss_name = oos_loss.lower()
            candidate.oos_start = holdout_start
            candidate.oos_stop = holdout_stop
            candidate.score = candidate.oos_loss
        else:
            candidate.score = score + (
                candidate.degeneracy_penalty if select_by == "aic+diag" else 0.0
            )
        
        # Populate convenience fields from diagnostics
        if diagnostics is not None:
            candidate.dgt_passed = not diagnostics['dgt']['reject']
            candidate.lb_failures = sum(
                1 for lb in diagnostics['ljung_box'].values()
                if lb['reject']
            )
        candidate.fit_time = time.perf_counter() - start_time
            
    except Exception as e:
        candidate.fit_time = time.perf_counter() - start_time
        candidate.score = float('inf')
        if select_by == "oos":
            candidate.oos_loss = float("inf")
        candidate.error_message = str(e)
    
    return candidate


def select_best_parallel(
    data: np.ndarray,
    vol_candidates: List[Tuple[str, int, int]],
    density_candidates: List[str],
    *,
    criterion: Callable[..., float],
    select_by: str = "aic+diag",
    diagnostic_kwargs: Optional[Dict[str, Any]] = None,
    n_jobs: Optional[int] = None,
    selection_verbose: bool = False,
    show_progress: bool = False,
    oos_loss: str = "logscore",
    oos_test_size: float | int = 0.2,
    oos_refit: Optional[int] = None,
    min_train_size: int = 250,
    min_test_size: int = 50,
    allow_density_blind_loss: bool = False,
    diagnostic_weight: float = 50.0,
    selection_warm_start: bool = True,
    degeneracy_weight: float = 50.0,
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
    selection_verbose : bool
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
    from ._autoselect import (
        SelectionResult,
        _check_density_blind_loss,
        _finalize_candidate_ranking,
        _finalize_oos_comparisons,
        _get_density_map as get_density_map,
        _get_vol_map as get_vol_map,
        _resolve_oos_split,
        _warm_start_params,
    )

    n_jobs = n_jobs if n_jobs is not None else get_default_workers()
    _check_density_blind_loss(
        oos_loss=oos_loss,
        density_candidates=density_candidates,
        allow_density_blind_loss=allow_density_blind_loss,
    )
    if oos_refit is not None:
        if select_by != "oos":
            raise ValueError("oos_refit is only valid with select_by='oos'.")
        if not isinstance(oos_refit, (int, np.integer)) or int(oos_refit) <= 0:
            raise ValueError("oos_refit must be a positive integer or None.")
    oos_split = None
    if select_by == "oos":
        oos_split = _resolve_oos_split(
            len(data),
            effective_hold_back=int(fit_kwargs.get("hold_back", 0) or 0),
            oos_test_size=oos_test_size,
            min_train_size=min_train_size,
            min_test_size=min_test_size,
        )
    
    # Build task list
    tasks = [
        (vol_type, p, q, d)
        for vol_type, p, q in vol_candidates
        for d in density_candidates
    ]
    
    total = len(tasks)
    
    if selection_verbose:
        print(f"Auto-selecting from {total} candidate models (parallel, {n_jobs} workers)...")
    
    # Sequential fallback: process spawn overhead (~1-2 s) exceeds the
    # total computation time when candidate counts are small (≤100).
    _MIN_PARALLEL_CANDIDATES = 100
    if n_jobs == 1 or total < _MIN_PARALLEL_CANDIDATES:
        candidates = []
        for i, (vol_type, p, q, d) in enumerate(
            get_progress_bar(tasks, total=total, desc="Auto-selecting", disable=not show_progress)
        ):
            if selection_verbose:
                print(f"  [{i+1}/{total}] Fitting {vol_type}({p},{q})+{d}...", end=" ")
            
            candidate = _fit_candidate_worker(
                vol_type, p, q, d, data, fit_kwargs,
                criterion, diagnostic_kwargs, select_by, oos_split, oos_loss,
                diagnostic_weight, degeneracy_weight,
                oos_refit,
            )
            candidates.append(candidate)
            
            if selection_verbose:
                if candidate.score < float('inf'):
                    print(f"AIC={candidate.aic:.2f}, Score={candidate.score:.2f}")
                else:
                    print(f"FAILED: {candidate.error_message}")
    else:
        # Dependency waves preserve warm-start relationships while fitting
        # independent candidates within each wave in parallel.
        actual_workers = min(n_jobs, total)
        bridge_density = "Normal" if "Normal" in density_candidates else density_candidates[0]
        if selection_warm_start:
            waves: List[List[Tuple[str, int, int, str]]] = []
            wave_keys = sorted({
                (2 * (p + q) + (0 if density == bridge_density else 1))
                for _, p, q, density in tasks
            })
            for wave_key in wave_keys:
                waves.append([
                    task for task in tasks
                    if 2 * (task[1] + task[2])
                    + (0 if task[3] == bridge_density else 1) == wave_key
                ])
        else:
            waves = [tasks]

        candidates = []
        vol_map = get_vol_map()
        density_map = get_density_map()
        fit_data = data if oos_split is None else data[:oos_split[0]]
        with tqdm_joblib(total=total, desc="Auto-selecting", disable=not show_progress):
            with Parallel(n_jobs=actual_workers, prefer=_parallel_preference()) as parallel:
                for wave in waves:
                    wave_calls = []
                    sources = []
                    for vol_type, p, q, density_name in wave:
                        task_kwargs = fit_kwargs.copy()
                        source_name = None
                        if selection_warm_start:
                            spec = vol_map[vol_type](p, q) + density_map[density_name]()
                            initial_params, source_name = _warm_start_params(
                                spec,
                                fit_data,
                                candidates,
                                bridge_density=bridge_density,
                            )
                            if initial_params is not None:
                                task_kwargs["initial_params"] = initial_params
                        sources.append(source_name)
                        wave_calls.append(delayed(_fit_candidate_worker)(
                            vol_type, p, q, density_name, data, task_kwargs,
                            criterion, diagnostic_kwargs, select_by, oos_split,
                            oos_loss, diagnostic_weight, degeneracy_weight,
                            oos_refit,
                        ))
                    wave_candidates = parallel(wave_calls)
                    for candidate, source_name in zip(wave_candidates, sources):
                        candidate.warm_started_from = source_name
                    candidates.extend(wave_candidates)
        
        if selection_verbose:
            for i, (candidate, (vol_type, p, q, d)) in enumerate(zip(candidates, tasks)):
                if candidate.score < float('inf'):
                    print(f"  [{i+1}/{total}] {vol_type}({p},{q})+{d}: "
                          f"AIC={candidate.aic:.2f}, Score={candidate.score:.2f}")
                else:
                    print(f"  [{i+1}/{total}] {vol_type}({p},{q})+{d}: "
                          f"FAILED: {candidate.error_message}")
    
    # Sort by score
    candidates.sort(key=lambda c: c.score)
    _finalize_candidate_ranking(candidates)
    if select_by == "oos":
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
    
    if select_by == "oos":
        fit_kwargs_copy = fit_kwargs.copy()
        fit_kwargs_copy["n_jobs"] = 1
        final_result = best.spec.fit(data, **fit_kwargs_copy)
    else:
        final_result = best.result

    search_space = {
        "vol_candidates": list(vol_candidates),
        "density_candidates": list(density_candidates),
        "n_candidates": total,
        "warm_start": bool(selection_warm_start),
        "degeneracy_weight": float(degeneracy_weight),
        "oos_refit": oos_refit,
    }
    selection = SelectionResult(
        candidates,
        select_by=select_by,
        criterion_name=f"oos:{oos_loss.lower()}" if select_by == "oos" else select_by,
        search_space=search_space,
    )
    final_result.selection = selection
    final_result._selection_candidates = candidates
    final_result.fit_info.selected_by = selection.criterion_name
    final_result.fit_info.selection_search_space = search_space
    final_result.fit_info.selection_refit_on_full = select_by == "oos"
    if oos_split is not None:
        train_stop, holdout_start, holdout_stop = oos_split
        final_result.fit_info.oos_train_size = train_stop
        final_result.fit_info.oos_test_size = holdout_stop - holdout_start
        final_result.fit_info.oos_loss = oos_loss.lower()
        final_result.fit_info.oos_refit = oos_refit

    if selection_verbose:
        print(f"\nBest model: {best.spec} (Score={best.score:.2f})")

    return final_result, candidates


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
    diagnostic_weight: float = 50.0,
    degeneracy_weight: float = 50.0,
    select_by: str = "aic+diag",
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
    from ._autoselect import (
        ModelCandidate,
        _assess_candidate_quality,
        _diagnostic_penalty,
        _score_candidate,
    )
    
    vol_map = _get_vol_map()
    density_map = _get_density_map()
    
    vol_cls = vol_map.get(vol_type, vol_map['GARCH'])
    density_cls = density_map.get(density_name, density_map['Normal'])
    spec = vol_cls(p, q) + density_cls()
    
    candidate = ModelCandidate(spec=spec, vol_type=vol_type, p=p, q=q, density=density_name)
    start_time = time.perf_counter()
    
    try:
        result = spec.fit(data, n_jobs=1, **fit_kwargs)
        # Attach pandas metadata
        result._index = index
        result._name = series_name
        
        candidate.result = result
        candidate.aic = result.aic
        candidate.bic = result.bic
        candidate.hqic = result.hqic
        candidate.fit_time = time.perf_counter() - start_time
        
        # Score via criterion
        score, diagnostics = _score_candidate(
            result, criterion, diagnostic_kwargs
        )
        candidate.diagnostics = diagnostics
        candidate.diagnostic_penalty = _diagnostic_penalty(diagnostics, diagnostic_weight)
        _assess_candidate_quality(candidate, degeneracy_weight=degeneracy_weight)
        candidate.score = score + (
            candidate.degeneracy_penalty if select_by == "aic+diag" else 0.0
        )
        
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
    select_by: str = "aic+diag",
    diagnostic_kwargs: Optional[Dict[str, Any]] = None,
    diagnostic_weight: float = 50.0,
    degeneracy_weight: float = 50.0,
    selection_warm_start: bool = True,
    n_jobs: Optional[int] = None,
    selection_verbose: bool = False,
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
    selection_verbose : bool
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
    from ._autoselect import (
        SelectionResult,
        _finalize_candidate_ranking,
        _warm_start_params,
    )
    from ._progress import tqdm_joblib

    n_jobs = n_jobs if n_jobs is not None else get_default_workers()

    requested_hold_back = int(fit_kwargs.get("hold_back", 0) or 0)
    common_sample = fit_kwargs.get("common_sample", None)
    if common_sample is not False and vol_candidates:
        common_hold_back = max(max(p, q) for _, p, q in vol_candidates)
        fit_kwargs["hold_back"] = max(requested_hold_back, common_hold_back)
        fit_kwargs["common_sample"] = True
    else:
        fit_kwargs["hold_back"] = requested_hold_back
        fit_kwargs["common_sample"] = False
    
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
    
    if selection_verbose:
        print(f"Auto-selecting {n_series} series x {n_candidates_per} candidates = {total} fits "
              f"(parallel, {min(n_jobs, total)} workers)...")
    
    # Fit dependency waves so each series can reuse lower-order and Normal starts.
    actual_workers = min(n_jobs, total)
    bridge_density = "Normal" if "Normal" in density_candidates else density_candidates[0]
    if selection_warm_start:
        wave_keys = sorted({
            2 * (p + q) + (0 if density == bridge_density else 1)
            for _, _, p, q, density, _ in tasks
        })
        waves = [
            [
                task for task in tasks
                if 2 * (task[2] + task[3])
                + (0 if task[4] == bridge_density else 1) == wave_key
            ]
            for wave_key in wave_keys
        ]
    else:
        waves = [tasks]

    series_candidates: Dict[str, List[Any]] = {name: [] for name in data_dict.keys()}
    vol_map = _get_vol_map()
    density_map = _get_density_map()
    with tqdm_joblib(total=total, desc="Auto-selecting (multi)", disable=not show_progress):
        with Parallel(n_jobs=actual_workers, prefer=_parallel_preference()) as parallel:
            completed = 0
            for wave in waves:
                calls = []
                sources = []
                for series_name, vol_type, p, q, density_name, series_data in wave:
                    task_kwargs = fit_kwargs.copy()
                    source_name = None
                    if selection_warm_start:
                        spec = vol_map[vol_type](p, q) + density_map[density_name]()
                        initial_params, source_name = _warm_start_params(
                            spec,
                            series_data,
                            series_candidates[series_name],
                            bridge_density=bridge_density,
                        )
                        if initial_params is not None:
                            task_kwargs["initial_params"] = initial_params
                    sources.append(source_name)
                    calls.append(delayed(_fit_series_candidate_worker)(
                        series_name, vol_type, p, q, density_name, series_data,
                        task_kwargs, criterion, diagnostic_kwargs, index,
                        diagnostic_weight, degeneracy_weight, select_by,
                    ))
                wave_results = parallel(calls)
                for (series_name, candidate), source_name in zip(wave_results, sources):
                    candidate.warm_started_from = source_name
                    series_candidates[series_name].append(candidate)
                    completed += 1
                    if selection_verbose:
                        status = (
                            f"AIC={candidate.aic:.2f}, Score={candidate.score:.2f}"
                            if np.isfinite(candidate.score)
                            else f"FAILED: {candidate.error_message}"
                        )
                        print(f"  [{completed}/{total}] {series_name}:{candidate.spec}: {status}")
    
    # Select best for each series
    results: Dict[str, Any] = {}
    
    for series_name, candidates in series_candidates.items():
        if not candidates:
            results[series_name] = {'error': 'No candidates fitted'}
            continue
        
        # Sort by score and pick best
        candidates.sort(key=lambda c: c.score)
        _finalize_candidate_ranking(candidates)
        
        best = None
        for c in candidates:
            if c.result is not None and np.isfinite(c.score):
                best = c
                break
        
        if best is not None:
            # Attach selection metadata
            search_space = {
                "vol_candidates": list(vol_candidates),
                "density_candidates": list(density_candidates),
                "n_candidates": n_candidates_per,
                "warm_start": bool(selection_warm_start),
                "degeneracy_weight": float(degeneracy_weight),
            }
            best.result.selection = SelectionResult(
                candidates,
                select_by=select_by,
                criterion_name=select_by,
                search_space=search_space,
            )
            best.result._selection_candidates = candidates
            best.result.fit_info.selected_by = select_by
            best.result.fit_info.selection_search_space = search_space
            best.result.fit_info.selection_refit_on_full = False
            results[series_name] = best.result
            
            if selection_verbose:
                print(f"\n  Best for {series_name}: {best.spec} (Score={best.score:.2f})")
        else:
            results[series_name] = {'error': 'All candidates failed'}
    
    return results


def _fit_series_auto_spec_worker(
    series_name: str,
    data: np.ndarray,
    index: Optional[Any],
    mean_candidates: List[Tuple[str, Tuple[Any, ...]]],
    vol_candidates: List[Tuple[str, int, int]],
    density_candidates: List[str],
    select_by: str,
    criterion: Optional[Callable[..., float]],
    diagnostic_kwargs: Optional[Dict[str, Any]],
    diagnostic_weight: float,
    degeneracy_weight: float,
    selection_warm_start: bool,
    mean_auto: bool,
    candidate_budget: int,
    mean_finalists: int,
    fit_kwargs: Dict[str, Any],
) -> Tuple[str, Any]:
    """Fit a complete auto-selection search for one series worker."""
    from ._autoselect import select_best_model

    try:
        result, _ = select_best_model(
            data,
            vol_candidates,
            density_candidates,
            mean_candidates=mean_candidates,
            mean_auto=mean_auto,
            candidate_budget=candidate_budget,
            mean_finalists=mean_finalists,
            select_by=select_by,
            criterion=criterion,
            diagnostic_kwargs=diagnostic_kwargs,
            diagnostic_weight=diagnostic_weight,
            degeneracy_weight=degeneracy_weight,
            selection_warm_start=selection_warm_start,
            n_jobs=1,
            show_progress=False,
            **fit_kwargs,
        )
        result._index = index
        result._name = series_name
        return series_name, result
    except Exception as exc:
        return series_name, {"error": str(exc)}


def fit_multi_auto_specs_parallel(
    data_dict: Dict[str, np.ndarray],
    *,
    index: Optional[Any],
    mean_candidates: List[Tuple[str, Tuple[Any, ...]]],
    vol_candidates: List[Tuple[str, int, int]],
    density_candidates: List[str],
    select_by: str,
    criterion: Optional[Callable[..., float]],
    diagnostic_kwargs: Optional[Dict[str, Any]],
    diagnostic_weight: float,
    degeneracy_weight: float,
    selection_warm_start: bool,
    mean_auto: bool,
    candidate_budget: int,
    mean_finalists: int,
    n_jobs: Optional[int],
    show_progress: bool,
    **fit_kwargs: Any,
) -> Dict[str, Any]:
    """Parallelize mean-aware auto selection across independent series."""
    from ._progress import tqdm_joblib

    workers = min(n_jobs or get_default_workers(), len(data_dict))
    with tqdm_joblib(
        total=len(data_dict),
        desc="Auto-selecting series",
        disable=not show_progress,
    ):
        fitted = Parallel(n_jobs=workers, prefer=_parallel_preference())(
            delayed(_fit_series_auto_spec_worker)(
                name,
                data,
                index,
                mean_candidates,
                vol_candidates,
                density_candidates,
                select_by,
                criterion,
                diagnostic_kwargs,
                diagnostic_weight,
                degeneracy_weight,
                selection_warm_start,
                mean_auto,
                candidate_budget,
                mean_finalists,
                fit_kwargs,
            )
            for name, data in data_dict.items()
        )
    return dict(fitted)
