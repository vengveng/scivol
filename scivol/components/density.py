# scivol/components/density.py
from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np

from ..components.base import Component
from ..roles import Role

class Normal(Component):
    role = Role.DENSITY
    
    def __init__(self):
        self.fitted_params = {}
    
    @property
    def signature(self): return 'Normal'
    
    @property  
    def n_params(self): return 0
    
    def default_start(self, data): return np.array([])
    def bounds(self): return []
    def pack(self, params_dict): return np.array([])
    def unpack(self, flat_params): 
        self.fitted_params = {}
        return {}

class StudentT(Component):
    role = Role.DENSITY
    
    def __init__(self):
        self.fitted_params = None
    
    @property
    def signature(self): return "StudentT"
    
    @property
    def n_params(self): return 1  # degrees of freedom
    
    def default_start(self, data): return np.array([10.0])
    def bounds(self): return [(2.01, 100.0)]
    def pack(self, params_dict): return np.array([params_dict['nu']])
    def unpack(self, flat_params):
        self.fitted_params = {'nu': flat_params[0]} if len(flat_params) > 0 else {}
        return self.fitted_params


class SkewT(Component):
    """
    Skewed Student-t distribution (Hansen, 1994 parameterization).
    
    Parameters
    ----------
    nu : float
        Degrees of freedom (must be > 2 for finite variance)
    lam : float
        Skewness parameter (-1, 1), where lam=0 gives symmetric Student-t
    """
    role = Role.DENSITY
    
    def __init__(self):
        self.fitted_params = None
    
    @property
    def signature(self): return "SkewT"
    
    @property
    def n_params(self): return 2  # degrees of freedom + skewness
    
    def default_start(self, data): 
        return np.array([10.0, 0.0])  # nu=10, lambda=0 (symmetric)
    
    def bounds(self): 
        return [(2.01, 100.0), (-0.99, 0.99)]  # nu bounds, lambda bounds
    
    def pack(self, params_dict): 
        return np.array([params_dict['nu'], params_dict['lam']])
    
    def unpack(self, flat_params):
        if len(flat_params) >= 2:
            self.fitted_params = {'nu': flat_params[0], 'lam': flat_params[1]}
        else:
            self.fitted_params = {}
        return self.fitted_params


class GED(Component):
    """
    Generalized Error Distribution (exponential power), standardized to unit variance.

    The shape parameter ``nu`` controls tail thickness and peakedness. ``nu=2``
    recovers the Normal distribution, while smaller values imply heavier tails.
    """

    role = Role.DENSITY

    def __init__(self):
        self.fitted_params = None

    @property
    def signature(self):
        return "GED"

    @property
    def n_params(self):
        return 1

    def default_start(self, data):
        return np.array([1.5], dtype=np.float64)

    def bounds(self):
        return [(1.01, 100.0)]

    def pack(self, params_dict):
        return np.array([params_dict["nu"]], dtype=np.float64)

    def unpack(self, flat_params):
        self.fitted_params = {"nu": flat_params[0]} if len(flat_params) > 0 else {}
        return self.fitted_params


class AutoDensity(Component):
    """
    Placeholder component for automatic density/distribution selection.
    
    When used in a spec, the fit method will search over candidate distributions
    and select the best one based on a blended criterion of AIC and diagnostic tests.
    
    Parameters
    ----------
    candidates : list of str, optional
        List of distribution names to search over.
        Default is ['Normal', 'StudentT', 'SkewT'].
        
    Examples
    --------
    >>> from scivol import GARCH, AutoDensity
    >>> spec = GARCH(1, 1) + AutoDensity()  # Search all distributions
    >>> spec = GARCH(auto=True) + AutoDensity(candidates=['Normal', 'StudentT'])
    
    Notes
    -----
    When used with QMLE estimation, AutoDensity is redundant since QMLE always
    fits with Normal likelihood. A warning will be issued and Normal will be used.
    """
    role = Role.DENSITY
    
    # Keep the default auto-selection set aligned with families that expose
    # a dedicated public fitting path. GED remains available for read-only
    # evaluation/fixed-parameter workflows, but not for generic fitting.
    CANDIDATES = ['Normal', 'StudentT', 'SkewT']
    
    def __init__(self, candidates: Optional[List[str]] = None):
        self.candidates = candidates or self.CANDIDATES.copy()
        self.fitted_params = {}
        self._is_auto = True
    
    @property
    def signature(self) -> str:
        return "AutoDensity"
    
    @property
    def n_params(self) -> int:
        # Placeholder - actual n_params depends on selected distribution
        return 0
    
    def default_start(self, data: np.ndarray) -> np.ndarray:
        return np.array([])
    
    def bounds(self) -> List[Tuple[float, float]]:
        return []
    
    def pack(self, params_dict: dict) -> np.ndarray:
        return np.array([])
    
    def unpack(self, flat_params: np.ndarray) -> dict:
        return {}
    
    def get_candidates(self) -> List[str]:
        """Return list of candidate distribution names."""
        return self.candidates