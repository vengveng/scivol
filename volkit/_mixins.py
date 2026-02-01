# volkit/_mixins.py  (tiny helper module)
from __future__ import annotations
from typing import TYPE_CHECKING, Any, Optional, List, Tuple
from abc import abstractmethod
import numpy as np

if TYPE_CHECKING:
    from .result import EstimationResult
    from .spec.composite import CompositeSpec
    from .estimators.base import Estimator


class FitsMixin:
    """
    Adds a .fit(...) convenience wrapper that delegates to the default estimator (MLE) and returns an EstimationResult.
    
    Supports automatic model selection when components have auto=True.
    """

    # the concrete class will supply .spec  (Component already has it, CompositeSpec can return self)
    @property
    @abstractmethod
    def spec(self) -> CompositeSpec: ...

    def fit(
        self,
        data: np.ndarray,
        estimator: Optional[Estimator] = None,
        *,
        diagnostic_weight: float = 50.0,
        verbose_selection: bool = False,
        **kwargs: Any,
    ) -> EstimationResult:
        """
        Fit the model specification to data.
        
        If any component has auto=True, performs automatic model selection
        over candidate specifications.

        Parameters
        ----------
        data : 1-D ndarray
            Returns or residuals to fit.
        estimator :  
            - None: default `MLE()`
            - instance: used as-is
            - class/callable: instantiated with **kwargs
        diagnostic_weight : float, default 50.0
            For auto selection: AIC penalty per failed diagnostic test.
        verbose_selection : bool, default False
            For auto selection: print progress during model selection.
        **kwargs :  
            Additional arguments passed to estimator.fit() or model selection.
            Common: solver, log_mode, method.
        
        Returns
        -------
        EstimationResult
            Fitted model results. If auto-selection was used, the result
            will have a `_selection_candidates` attribute with all evaluated
            models and a `selection_summary()` method.
        """
        # Check for auto components
        if self._has_auto_components():
            return self._auto_fit(
                data,
                estimator=estimator,
                diagnostic_weight=diagnostic_weight,
                verbose=verbose_selection,
                **kwargs,
            )
        
        # Normal fit path
        from .estimators import MLE

        if estimator is None:
            est = MLE()
        elif callable(estimator) and not isinstance(estimator, MLE):
            est = estimator()
        else:
            est = estimator

        return est.fit(self.spec, data, **kwargs)
    
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
    
    def _auto_fit(
        self,
        data: np.ndarray,
        estimator: Optional[Estimator] = None,
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
            **kwargs,
        )
        
        # Attach selection info to result for later inspection
        result._selection_candidates = all_candidates
        
        return result