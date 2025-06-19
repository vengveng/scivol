import numpy as np
import pandas as pd
from scipy.optimize import minimize, LinearConstraint
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Tuple
import warnings
from ctypes import CDLL, c_double, POINTER, c_size_t
from utils.stats import Test
# from utils.tools import ModelResult, CombinedModelResult

import utils.bindingManager as bindingManager
MANAGER = bindingManager.FunctionBindingManager()

from utils.tools import ModelResult, CombinedModelResult, DataHandler

class VolatilityModel():
    # Scipy objects here
    def fit(self, residuals, distr='normal', framework='QMLE', multivar=None):
        assert residuals is not None
        data_handler = DataHandler(residuals)

        if multivar and multivar.lower() == 'dcc':
            # return self._fit_dcc(data_handler, framework_instance)
            pass
        elif data_handler.is_univariate:
            return self._fit_univariate(data_handler.data, distr, framework) 
        else:
            return self._fit_multiple_univariate(data_handler.data, distr, framework)

    def _fit_univariate(self, residuals, distr, framework):
        result = ModelResult(self.p, self.q, distr)

        if framework == 'MLE':
            if distr == 'normal':
                raw_result = self._fit_normal(residuals)
            else:
                raw_result = self._fit_aparametric(residuals, distr)
            result.update(raw_result, residuals, self.var, self._std)

        elif framework == 'QMLE':
            raw_normal_result = self._fit_normal(residuals)
            result.update(raw_normal_result, residuals, self.var, self._std)
            if not distr == 'normal':
                raw_parametric_result = self._fit_parametric(residuals, distr, raw_normal_result)
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
    
    def _fit_parametric(self, residuals, distr, normal_result):
        if distr == 'studentt':
            result = self._fit_studentt(residuals, normal_result)
            return self._fit_studentt(residuals, normal_result)
        else:
            ...
        return normal_result.add_secondary(result, distr)

    def _fit_aparametric(self, residuals, distr):
        raise UserWarning('MLE with non-normal densities not yet implemented')

    def _fit_studentt(self, residuals, normal_result):
        n = len(residuals)
        sigma2 = self.var(residuals, normal_result.x)

        residuals2 = residuals**2
        r2os2 = residuals2 / sigma2

        sigma2c = self._np_ptr(sigma2)
        r2os2c = self._np_ptr(r2os2)

        def max_likelihood(nu_vector):
            return MANAGER.studentt.general.likelihood(sigma2c, r2os2c, c_size_t(n), c_double(nu_vector[0]))
        
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
    def _fit_normal(self, residuals):
        pass

    @abstractmethod
    def var(self, residuals, params):
        pass

    @abstractmethod
    def _std(*args, **kwargs):
        pass

class GARCH(VolatilityModel):

    SPECIAL = [(1, 1, 'normal'), (1, 1, None)]

    def __init__(self, p, q, data=None):
        self.p = p
        self.q = q
        self.params = None
        self.data = ...
        self.res = ModelResult(p, q)

    @staticmethod
    def _np_ptr(vector: np.ndarray, dtypes = c_double):
        "Returns a c pointer to numpy array"
        contig_vec = np.ascontiguousarray(vector, dtype=np.float64)
        return contig_vec.ctypes.data_as(POINTER(dtypes))

    def var(self, residuals=None, params=None):
        if self.res.var is not None and np.array_equal(params, self.res.params):
            return self.res.var
        elif params is None:
            params = self.res.params
        assert residuals is not None
        assert params is not None
        if not isinstance(params, np.ndarray):
            params = np.array(params)

        assert len(params) == 1 + self.p + self.q

        sigma2 = np.zeros(len(residuals), dtype=np.float64)
        sigma2[0] = np.sum(residuals**2) / len(residuals)
        residuals2 = residuals**2

        sigma2_c = self._np_ptr(sigma2)
        residuals2_c = self._np_ptr(residuals2)
        parameters_c = self._np_ptr(params)

        if (self.p, self.q, None) in self.SPECIAL:
            MANAGER.garch.special.variance(parameters_c, residuals2_c, sigma2_c, c_size_t(len(residuals)))
        else:
            MANAGER.garch.general.variance(parameters_c, residuals2_c, sigma2_c, c_size_t(len(residuals)), c_size_t(self.p), c_size_t(self.q))
        return sigma2
    
    def fit(self, residuals, distr='normal', framework='QMLE'):
        self.res = super().fit(residuals, distr, framework)        
        return self.res

    # def _fit_parametric(self, residuals, distr, normal_parameters=None):
    #     ...
    
    def _fit_normal(self, residuals):

        optimization_bounds = [(0, +np.inf)] + [(0.000001, 1)] * (self.p + self.q)
        initial_parameters = [0.00001] + [0.05 / self.p] * self.p + [0.9 / self.q] * self.q

        n = len(residuals)
        sigma2 = np.zeros(len(residuals), dtype=np.float64)
        sigma2[0] = np.sum(residuals**2) / len(residuals)
        residuals2 = residuals**2
        constant_ll = 0.5 * n * np.log(2 * np.pi)

        sigma2_c = self._np_ptr(sigma2)
        residuals2_c = self._np_ptr(residuals2)

        if [self.p, self.q, 'normal'] in self.SPECIAL:
            def max_likelihood(parameters):
                #TODO: generalize SPECIAL logic to flexibly deal with any special case
                # _np_ptr uncalled for minimal overhead
                parameters_c = np.ascontiguousarray(parameters, dtype=np.float64).ctypes.data_as(POINTER(c_double))
                return MANAGER.garch.special.objective(parameters_c, residuals2_c, sigma2_c, c_size_t(n))
            
        else:
            def max_likelihood(parameters):
                # _np_ptr uncalled for minimal overhead
                parameters_c = np.ascontiguousarray(parameters, dtype=np.float64).ctypes.data_as(POINTER(c_double))
                MANAGER.garch.general.variance(parameters_c, residuals2_c, sigma2_c, c_size_t(len(residuals)), c_size_t(self.p), c_size_t(self.q))
                likelihood = MANAGER.garch.general.likelihood(sigma2_c, residuals2_c, c_size_t(n))
                return likelihood
            
        result = minimize(max_likelihood, initial_parameters, method='Nelder-Mead', bounds=optimization_bounds, options={'maxfev': int(1e8)}) 
        # result = minimize(max_likelihood, initial_parameters, method='SLSQP', bounds=optimization_bounds, options={'maxfev': int(1e5), 'disp': True}) 
        # result = minimize(max_likelihood, initial_parameters, method='trust-constr', bounds=optimization_bounds, options={'disp': True}) 
        result.fun += constant_ll
        return result
    
    def _std(self, p=None, q=None, resid=None, sigma2=None, params=None):
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

        residuals2 = resid**2
        n = len(residuals2)

        size = 1 + p + q
        OPG = np.zeros((size, size))
        HESS = np.zeros((size, size))

        residuals2_c = np.ascontiguousarray(residuals2, dtype=np.float64).ctypes.data_as(POINTER(c_double))
        sigma2_c = np.ascontiguousarray(sigma2, dtype=np.float64).ctypes.data_as(POINTER(c_double))
        OPG_c = np.ascontiguousarray(OPG, dtype=np.float64).ctypes.data_as(POINTER(c_double))
        HESS_c = np.ascontiguousarray(HESS, dtype=np.float64).ctypes.data_as(POINTER(c_double))

        if p == 1 and q == 1:
            MANAGER.garch.special.std_err_robust(residuals2_c, sigma2_c, OPG_c, HESS_c, c_size_t(n))
        else:
            MANAGER.garch.general.std_err_robust(residuals2_c, sigma2_c, OPG_c, HESS_c, c_size_t(n), c_size_t(p), c_size_t(q))

        return OPG, HESS
    
    def auto(self):

        return super().auto()
