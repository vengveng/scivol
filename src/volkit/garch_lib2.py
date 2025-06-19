from __future__ import annotations
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from scipy.optimize import minimize, LinearConstraint
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Tuple
import warnings
from ctypes import CDLL, c_double, POINTER, c_size_t
from utils.stats import Test
from numpy.typing import NDArray

# from . import _core
# from utils.tools import ModelResult, CombinedModelResult


# ── dual-import idiom ───────────────────────────────────────────────
if TYPE_CHECKING:
    import _core
else:
    import importlib as _implib
    _core = _implib.import_module(__package__ + "._core")

from utils.tools import ModelResult, CombinedModelResult, DataHandler

class VolatilityModel():
    # Scipy objects here
    def fit(self, resid, distr='normal', framework='QMLE', multivar=None):
        assert resid is not None
        data_handler = DataHandler(resid)

        if multivar and multivar.lower() == 'dcc':
            # return self._fit_dcc(data_handler, framework_instance)
            pass
        elif data_handler.is_univariate:
            return self._fit_univariate(data_handler.data, distr, framework) 
        else:
            return self._fit_multiple_univariate(data_handler.data, distr, framework)

    def _fit_univariate(self, resid, distr, framework):
        result = ModelResult(self.p, self.q, distr)

        if framework == 'MLE':
            if distr == 'normal':
                raw_result = self._fit_normal(resid)
            else:
                raw_result = self._fit_aparametric(resid, distr)
            result.update(raw_result, resid, self.var, self._std)

        elif framework == 'QMLE':
            raw_normal_result = self._fit_normal(resid)
            result.update(raw_normal_result, resid, self.var, self._std)
            if not distr == 'normal':
                raw_parametric_result = self._fit_parametric(resid, distr, raw_normal_result)
                result.add_secondary(raw_parametric_result, 'NORMAL')

        return result
    
    def _fit_multiple_univariate(self, data_handler, distr='normal', framework='QMLE'):
        n = data_handler.shape[1]
        results = {}
        
        for i in range(n):
            slice_handler = data_handler.slice(i)
            series_name = slice_handler.columns[0] if slice_handler.columns else f"Series {i+1}"
            results[series_name] = self._fit_univariate(slice_handler.data, distr, framework)

        return CombinedModelResult(results)    
    
    def _fit_parametric(self, resid, distr, normal_result):
        if distr == 'studentt':
            result = self._fit_studentt(resid, normal_result)
            return self._fit_studentt(resid, normal_result)
        else:
            ...
        return normal_result.add_secondary(result, distr)

    def _fit_aparametric(self, resid, distr):
        raise UserWarning('MLE with non-normal densities not yet implemented')

    def _fit_studentt(self, resid, normal_result):
        n = len(resid)
        sigma2 = self.var(resid, normal_result.x)

        resid2 = resid**2
        r2os2  = resid2 / sigma2

        sigma2_c = np.ascontiguousarray(sigma2, dtype=np.float64).ctypes.data
        r2os2_c  = np.ascontiguousarray(r2os2,  dtype=np.float64).ctypes.data

        def max_likelihood(nu_vector):
            return _core._any_studentt_likelihood(sigma2_c, r2os2_c, n, nu_vector[0])
        
        result = minimize(max_likelihood, [10], method='Nelder-Mead', bounds=[(2.0001, 10000)])
        return result

    @staticmethod
    def _np_ptr(vector: np.ndarray, dtypes = c_double):
        "Returns a c pointer to numpy array"
        contig_vec = np.ascontiguousarray(vector, dtype=np.float64)
        return contig_vec.ctypes.data_as(POINTER(dtypes))
    
    def auto(self):
        ...
        
    @abstractmethod
    def _fit_normal(self, resid):
        pass

    @abstractmethod
    def var(self, resid, params):
        pass

    @abstractmethod
    def _std(*args, **kwargs) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
        pass

class GARCH(VolatilityModel):

    SPECIAL = [(1, 1, 'normal'), (1, 1, None)]

    def __init__(self, p: int, q: int, data=None):
        self.p = p
        self.q = q
        self.params = None
        self.data = ...
        self.res = ModelResult(p, q)

    def var(self, resid=None, params=None):
        if self.res.var is not None and np.array_equal(params, self.res.params):
            return self.res.var
        elif params is None:
            params = self.res.params
        assert resid is not None
        assert params is not None
        if not isinstance(params, np.ndarray):
            params = np.array(params)

        assert len(params) == 1 + self.p + self.q

        n = len(resid)
        sigma2 = np.zeros(n, dtype=np.float64)
        sigma2[0] = np.sum(resid**2) / n
        resid2 = resid**2

        params_c = np.ascontiguousarray(params, dtype=np.float64).ctypes.data
        sigma2_c = np.ascontiguousarray(sigma2, dtype=np.float64).ctypes.data
        resid2_c = np.ascontiguousarray(resid2, dtype=np.float64).ctypes.data

        if (self.p, self.q, None) in self.SPECIAL:
            _core._special_garch_oo_normal_variance(params_c, resid2_c, sigma2_c, n)
        else:
            _core._garch_variance_pq(params_c, resid2_c, sigma2_c, n, self.p, self.q)
        return sigma2
    
    def fit(self, resid, distr='normal', framework='QMLE'):
        self.res = super().fit(resid, distr, framework)        
        return self.res

    # def _fit_parametric(self, resid, distr, normal_params=None):
    #     ...
    
    def _fit_normal(self, resid):

        optimization_bounds = [(0, +np.inf)] + [(0.000001, 1)] * (self.p + self.q)
        initial_params = [0.00001] + [0.05 / self.p] * self.p + [0.9 / self.q] * self.q

        n = len(resid)
        sigma2 = np.zeros(len(resid), dtype=np.float64)
        sigma2[0] = np.sum(resid**2) / len(resid)
        resid2 = resid**2
        constant_ll = 0.5 * n * np.log(2 * np.pi)

        sigma2_c = np.ascontiguousarray(sigma2, dtype=np.float64).ctypes.data
        resid2_c = np.ascontiguousarray(resid2, dtype=np.float64).ctypes.data

        if [self.p, self.q, 'normal'] in self.SPECIAL:
            def max_likelihood(params: NDArray[np.float64]) -> float:
                params_c = np.ascontiguousarray(params, dtype=np.float64).ctypes.data
                return _core._special_garch_oo_normal(params_c, resid2_c, sigma2_c, n)
            
        else:
            def max_likelihood(params: NDArray[np.float64]) -> float:
                params_c = np.ascontiguousarray(params, dtype=np.float64).ctypes.data
                _core._garch_variance_pq(params_c, resid2_c, sigma2_c, n, self.p, self.q)
                return _core._normal_likelihood(sigma2_c, resid2_c, n)
            
        result = minimize(max_likelihood, initial_params, method='Nelder-Mead', bounds=optimization_bounds, options={'maxfev': int(1e5)}) 
     
        result.fun += constant_ll
        return result
    
    def _std(self, p=None, q=None, resid=None, sigma2=None, params=None) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
        #TODO: delegate data to DataHandler
        p = self.p if p is None else p
        q = self.q if q is None else q

        resid = self.resid if resid is None else resid
        params = self.params if params is None else params

        assert resid is not None
        assert p is not None and q is not None
    
        if sigma2 is None:
            sigma2 = self.res.var
        if sigma2 is None:
            assert params is not None
            sigma2 = self.var(resid, params)

        resid2 = resid**2
        n = len(resid2)

        size = 1 + p + q
        OPG  = np.zeros((size, size))
        HESS = np.zeros((size, size))

        resid2_c = np.ascontiguousarray(resid2, dtype=np.float64).ctypes.data
        sigma2_c = np.ascontiguousarray(sigma2, dtype=np.float64).ctypes.data
        OPG_c    = np.ascontiguousarray(OPG,    dtype=np.float64).ctypes.data
        HESS_c   = np.ascontiguousarray(HESS,   dtype=np.float64).ctypes.data

        if p == 1 and q == 1:
            _core._special_garch_11_std_err_robust(resid2_c, sigma2_c, OPG_c, HESS_c, n)
        else:
            _core._general_garch_pq_std_err_robust(resid2_c, sigma2_c, OPG_c, HESS_c, n, p, q)

        return OPG, HESS
    
    def auto(self):

        return super().auto()
