# volkit/_mixins.py  (tiny helper module)
from __future__ import annotations
from typing import TYPE_CHECKING, Any, Optional, List, Tuple, Dict, Union
from abc import abstractmethod
import numpy as np

if TYPE_CHECKING:
    from .result import EstimationResult
    from .spec.composite import CompositeSpec
    from .estimators.base import Estimator


class FitsMixin:
    """
    Adds a .fit(...) convenience wrapper that delegates to the default estimator (MLE) and returns an EstimationResult.
    
    Supports:
    - Automatic model selection when components have auto=True
    - Pandas Series/DataFrame input with index preservation
    - Parallel fitting of multiple time series
    """

    # the concrete class will supply .spec  (Component already has it, CompositeSpec can return self)
    @property
    @abstractmethod
    def spec(self) -> CompositeSpec: ...

    def fit(
        self,
        data: Union[np.ndarray, Any],  # Any allows pandas without hard dependency
        estimator: Optional[Estimator] = None,
        *,
        n_jobs: Optional[int] = None,
        diagnostic_weight: float = 50.0,
        verbose_selection: bool = False,
        **kwargs: Any,
    ) -> Union[EstimationResult, Dict[str, EstimationResult]]:
        """
        Fit the model specification to data.
        
        If any component has auto=True, performs automatic model selection
        over candidate specifications.

        Parameters
        ----------
        data : 1-D ndarray, pandas Series, or pandas DataFrame
            Returns or residuals to fit.
            - 1-D array or Series: Fit single model, return EstimationResult
            - DataFrame with 1 column: Fit single model, return EstimationResult
            - DataFrame with multiple columns: Fit each column in parallel,
              return Dict[str, EstimationResult]
        estimator :  
            - None: default `MLE()`
            - instance: used as-is
            - class/callable: instantiated with **kwargs
        n_jobs : int, optional
            Number of parallel workers for multi-series fitting or auto-selection.
            Default is None which uses all available CPU cores.
            Set to 1 for sequential processing.
        diagnostic_weight : float, default 50.0
            For auto selection: AIC penalty per failed diagnostic test.
        verbose_selection : bool, default False
            For auto selection: print progress during model selection.
        **kwargs :  
            Additional arguments passed to estimator.fit() or model selection.
            Common: solver, log_mode, method.
        
        Returns
        -------
        EstimationResult or Dict[str, EstimationResult]
            - Single series: EstimationResult with fitted model
            - Multiple series: Dict mapping column names to EstimationResult
            
            If auto-selection was used, results have `_selection_candidates` 
            attribute and `selection_summary()` method.
        """
        # Extract pandas metadata and convert to numpy
        index, name, data_np, columns = self._extract_pandas_info(data)
        
        # Check if multi-series (DataFrame with multiple columns)
        if columns is not None and len(columns) > 1:
            return self._fit_multi(
                data_np,
                columns=columns,
                index=index,
                estimator=estimator,
                n_jobs=n_jobs,
                diagnostic_weight=diagnostic_weight,
                verbose_selection=verbose_selection,
                **kwargs,
            )
        
        # Single series fit
        return self._fit_single(
            data_np,
            index=index,
            name=name,
            estimator=estimator,
            n_jobs=n_jobs,
            diagnostic_weight=diagnostic_weight,
            verbose_selection=verbose_selection,
            **kwargs,
        )
    
    def _extract_pandas_info(
        self, data: Any
    ) -> Tuple[Optional[Any], Optional[str], np.ndarray, Optional[List[str]]]:
        """
        Extract pandas index/name and convert to numpy.
        
        Returns
        -------
        index : pandas Index or None
        name : str or None (series name)
        data_np : numpy array
        columns : list of column names or None (for multi-series)
        """
        try:
            import pandas as pd
            
            if isinstance(data, pd.Series):
                return data.index, data.name, data.to_numpy(), None
            elif isinstance(data, pd.DataFrame):
                if data.shape[1] == 1:
                    # Single column DataFrame - treat like Series
                    col_name = data.columns[0]
                    return data.index, col_name, data.iloc[:, 0].to_numpy(), None
                else:
                    # Multi-column DataFrame - return columns for parallel fitting
                    return data.index, None, data.to_numpy(), list(data.columns)
        except ImportError:
            pass
        
        # Plain numpy array
        arr = np.asarray(data)
        if arr.ndim == 2 and arr.shape[1] > 1:
            # 2D numpy array - use integer column names
            columns = [str(i) for i in range(arr.shape[1])]
            return None, None, arr, columns
        
        return None, None, arr.ravel() if arr.ndim > 1 else arr, None
    
    def _fit_single(
        self,
        data: np.ndarray,
        *,
        index: Optional[Any] = None,
        name: Optional[str] = None,
        estimator: Optional[Estimator] = None,
        n_jobs: Optional[int] = None,
        diagnostic_weight: float = 50.0,
        verbose_selection: bool = False,
        **kwargs: Any,
    ) -> EstimationResult:
        """Fit a single time series."""
        # Check for auto components
        if self._has_auto_components():
            result = self._auto_fit(
                data,
                estimator=estimator,
                n_jobs=n_jobs,
                diagnostic_weight=diagnostic_weight,
                verbose=verbose_selection,
                **kwargs,
            )
        else:
            # Normal fit path
            from .estimators import MLE

            if estimator is None:
                est = MLE()
            elif callable(estimator) and not isinstance(estimator, MLE):
                est = estimator()
            else:
                est = estimator

            result = est.fit(self.spec, data, **kwargs)
        
        # Attach pandas metadata
        result._index = index
        result._name = name
        
        return result
    
    def _has_auto_components(self) -> bool:
        """Check if any component has auto selection enabled."""
        for comp in self.spec.components:
            if getattr(comp, '_is_auto', False):
                return True
        return False
    
    def _get_vol_candidates(self) -> List[Tuple[int, int]]:
        """Get list of (p, q) candidates from volatility component."""
        from .roles import Role
        
        vol = self.spec.get_component(Role.VOLATILITY)
        if vol is None:
            return [(1, 1)]  # Default
        
        if hasattr(vol, 'get_candidates'):
            return vol.get_candidates()
        
        return [(vol.p, vol.q)]
    
    def _get_density_candidates(self) -> List[str]:
        """Get list of density candidate names from density component."""
        from .roles import Role
        
        density = self.spec.get_component(Role.DENSITY)
        if density is None:
            return ['Normal']
        
        if hasattr(density, 'get_candidates'):
            return density.get_candidates()
        
        # Fixed density - just return its name
        return [density.signature]
    
    def _spec_to_params(self) -> Dict[str, Any]:
        """
        Serialize spec to dict for cross-process reconstruction.
        
        This avoids pickling issues with component objects.
        """
        from .roles import Role
        from .components.density import AutoDensity
        
        vol = self.spec.get_component(Role.VOLATILITY)
        density = self.spec.get_component(Role.DENSITY)
        mean = self.spec.get_component(Role.MEAN)
        
        return {
            'vol': (vol.p, vol.q) if vol else None,
            'vol_auto': getattr(vol, '_auto_config', None) if vol else None,
            'density': density.signature if density and not isinstance(density, AutoDensity) else 'Normal',
            'density_auto': isinstance(density, AutoDensity) if density else False,
            'density_candidates': density.candidates if isinstance(density, AutoDensity) else None,
            'mean': (mean.p, mean.q) if mean and hasattr(mean, 'p') else None,
        }
    
    def _fit_multi(
        self,
        data: np.ndarray,
        columns: List[str],
        index: Optional[Any],
        estimator: Optional[Estimator] = None,
        n_jobs: Optional[int] = None,
        diagnostic_weight: float = 50.0,
        verbose_selection: bool = False,
        **kwargs: Any,
    ) -> Dict[str, EstimationResult]:
        """
        Fit multiple time series in parallel.
        
        Parameters
        ----------
        data : 2-D numpy array
            Shape (n_obs, n_series)
        columns : list of str
            Column names for each series
        index : pandas Index or None
        n_jobs : int or None
            Number of workers (None = cpu_count)
        
        Returns
        -------
        Dict[str, EstimationResult]
        """
        from ._parallel import fit_multi_parallel, fit_multi_auto_parallel, get_default_workers
        
        n_jobs = n_jobs if n_jobs is not None else get_default_workers()
        
        # Build data dict
        data_dict = {col: data[:, i] for i, col in enumerate(columns)}
        
        # Check if auto-selection is enabled
        if self._has_auto_components():
            # Use flattened parallelism: (n_series × n_candidates) tasks
            # This maximizes CPU utilization
            vol_candidates = self._get_vol_candidates()
            density_candidates = self._get_density_candidates()
            
            return fit_multi_auto_parallel(
                data_dict=data_dict,
                vol_candidates=vol_candidates,
                density_candidates=density_candidates,
                index=index,
                diagnostic_weight=diagnostic_weight,
                n_jobs=n_jobs,
                verbose=verbose_selection,
                **kwargs,
            )
        
        # Non-auto: parallelize only across series
        spec_params = self._spec_to_params()
        
        fit_kwargs = {
            'diagnostic_weight': diagnostic_weight,
            'verbose_selection': verbose_selection,
            **kwargs,
        }
        
        results = fit_multi_parallel(
            spec_params=spec_params,
            data_dict=data_dict,
            index=index,
            n_jobs=n_jobs,
            **fit_kwargs,
        )
        
        return results
    
    def _auto_fit(
        self,
        data: np.ndarray,
        estimator: Optional[Estimator] = None,
        n_jobs: Optional[int] = None,
        diagnostic_weight: float = 50.0,
        verbose: bool = False,
        **kwargs: Any,
    ) -> EstimationResult:
        """
        Perform automatic model selection.
        
        Searches over candidate (p, q) and density combinations,
        fitting each and selecting the best by blended AIC + diagnostic criterion.
        """
        from ._autoselect import select_best_model
        
        # Build candidate lists from auto configs
        vol_candidates = self._get_vol_candidates()
        density_candidates = self._get_density_candidates()
        
        # Run model selection
        result, all_candidates = select_best_model(
            data,
            vol_candidates,
            density_candidates,
            diagnostic_weight=diagnostic_weight,
            verbose=verbose,
            n_jobs=n_jobs,
            **kwargs,
        )
        
        # Attach selection info to result for later inspection
        result._selection_candidates = all_candidates
        
        return result