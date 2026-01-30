from __future__ import annotations

from typing import TYPE_CHECKING, Any, Union
import numpy as np

from .._kernels import get_routine
from .base import Estimator

if TYPE_CHECKING:
    from ..result import EstimationResult
    from ..spec import CompositeSpec
    from ..components import Component


class MLE(Estimator):
    """
    Maximum Likelihood Estimator.
    
    Dispatches to the appropriate kernel routine based on the model specification.
    
    Usage:
        spec = GARCH(1, 1) + Normal()
        estimator = MLE()
        result = estimator.fit(spec, data)
    
    Or with solver configuration:
        result = estimator.fit(spec, data, solver="trust-constr", verbose=True)
    """

    def fit(
        self,
        spec: Union[CompositeSpec, Component],
        data: np.ndarray,
        solver: str = "trust",
        log_mode: bool = True,
        verbose: bool = False,
        **kwargs: Any,
    ) -> EstimationResult:
        """
        Fit model via Maximum Likelihood Estimation.
        
        Parameters
        ----------
        spec : CompositeSpec or Component
            Model specification (e.g., GARCH(1,1) + Normal())
        data : array
            Residual series (demeaned returns or AR residuals)
        solver : str
            Optimization method:
            - "nelder-mead": derivative-free, robust but slow
            - "slsqp": uses gradient, good for constrained problems  
            - "trust" or "trust-constr": uses gradient + Hessian (recommended)
            - "trust-exact": uses gradient + Hessian in log-space (Normal only)
        log_mode : bool
            If True (default), optimize in unconstrained log-space (Normal only).
            Stationarity is enforced by construction via softmax transformation.
        verbose : bool
            Print optimization progress
        **kwargs
            Additional arguments passed to the kernel fit function
            
        Returns
        -------
        EstimationResult
            Contains estimated parameters, log-likelihood, conditional variances,
            standardized residuals, and standard errors.
        """
        spec = self._validate_spec(spec)
        data = self._validate_data(data)
        self._warn_small_sample(spec, data)

        routine = get_routine(str(spec))
        result = routine.fit(data, solver=solver, log_mode=log_mode, verbose=verbose, **kwargs)
        self._last_result = result
        return result