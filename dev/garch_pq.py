import time
from typing import Callable, Any
import numpy as np
import pandas as pd
from scipy.optimize import minimize
import os
from statsmodels.tsa.ar_model import AutoReg
from numba import jit, njit, float64
from scipy.optimize import minimize
from scipy.linalg import inv
from scipy.optimize import approx_fprime
import statsmodels.api as sm


class Timer:
    def __init__(self):
        self.results = []

    def time_function(self, name: str, func: Callable, *args, **kwargs) -> Any:
        """
        Times a function and stores the result.

        :param name: Name of the implementation.
        :param func: Function to time.
        :param args: Positional arguments for the function.
        :param kwargs: Keyword arguments for the function.
        :return: The result of the function execution.
        """
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        elapsed = end_time - start_time
        self.results.append((name, elapsed))
        time_str = f"[{name}] Execution Time: {elapsed:.4f} seconds | "
        res_str = ' '.join([f'{x:.4f}' for x in result.x]) + f' | {result.fun:.2f}' 
        print(time_str + res_str)
        return result

    def print_results(self):
        """
        Prints a summary of timing results.
        """
        print("\nTiming Summary:")
        for name, elapsed in self.results:
            print(f"{name}: {elapsed:.4f} seconds")

# Load and preprocess data
root = os.path.dirname(__file__)
info_path = os.path.join(root, 'data', 'info.csv')
data = pd.read_csv(info_path, index_col=0, usecols=[0, 1, 2, 3], skiprows=1, parse_dates=False, names=['date', 'DAX', 'S&P', 'rate'])
data.index = pd.DatetimeIndex(data.index, freq='W-MON')
data = data[data.index.year < 2024]
data['rate'] = data['rate'] / 100 / 52
returns = data.pct_change().dropna()

# Compute weekly returns and residuals
weekly_returns = returns[['DAX', 'S&P']].reset_index()
# weekly_returns = pd.concat([weekly_returns, weekly_returns], axis=0)
# weekly_returns = pd.concat([weekly_returns, weekly_returns], axis=0)
# weekly_returns = pd.concat([weekly_returns, weekly_returns], axis=0)
# weekly_returns = pd.concat([weekly_returns, weekly_returns], axis=0)
# weekly_returns = pd.concat([weekly_returns, weekly_returns], axis=0)
# weekly_returns = pd.concat([weekly_returns, weekly_returns], axis=0)
# weekly_returns = pd.concat([weekly_returns, weekly_returns], axis=0)
# weekly_returns = pd.concat([weekly_returns, weekly_returns], axis=0)
# weekly_returns = pd.concat([weekly_returns, weekly_returns], axis=0)
# weekly_returns = pd.concat([weekly_returns, weekly_returns], axis=0)
# weekly_returns = pd.concat([weekly_returns, weekly_returns], axis=0)
# weekly_returns = pd.concat([weekly_returns, weekly_returns], axis=0)
# weekly_returns = pd.concat([weekly_returns, weekly_returns], axis=0)
print(weekly_returns.shape)

dax_residuals = AutoReg(weekly_returns['DAX'], lags=1, ).fit().resid
sp_residuals = AutoReg(weekly_returns['S&P'], lags=1).fit().resid

dax_residuals = sm.tsa.arima.ARIMA(weekly_returns['DAX'], order=(1, 0, 1)).fit().resid

residuals = dax_residuals.to_numpy()

stationarity_vector = [0, 1, 1]
def generate_optimization_bounds(p, q):
    return [(0, +np.inf)] + [(0.000001, 1)] * (p + q)
# p, q = 2, 2
p, q = 1, 1
def generate_initial_parameters(p, q):
    return [0.00001] + [0.05 / p] * p + [0.9 / q] * q

optimization_bounds = generate_optimization_bounds(p, q)
initial_parameters = generate_initial_parameters(p, q)

from ctypes import CDLL, c_double, POINTER, c_size_t
lib_general = CDLL('lib/objective_general.so')
lib_special = CDLL('lib/objective_special.so')
lib = CDLL('lib/cvm_lib.so')

robust_argtypes = {'residuals2': POINTER(c_double), 
                'sigma2': POINTER(c_double),
                'OPG': POINTER(c_double),
                'HESS': POINTER(c_double), 
                'n': c_size_t, 
                'p': c_size_t, 
                'q': c_size_t}

robust_special_argtypes = {'residuals2': POINTER(c_double), 
                'sigma2': POINTER(c_double),
                'OPG': POINTER(c_double),
                'HESS': POINTER(c_double), 
                'n': c_size_t, }

lib.general_garch_pq_std_err_robust.argtypes = [c_type for c_type in robust_argtypes.values()]
lib.general_garch_pq_std_err_robust.restype = None
lib.special_garch_11_std_err_robust.argtypes = [c_type for c_type in robust_special_argtypes.values()]
lib.special_garch_11_std_err_robust.restype = None



garch_argtypes = {'parameters': POINTER(c_double), 
                'residuals2': POINTER(c_double), 
                'sigma2': POINTER(c_double), 
                'n': c_size_t, 
                'p': c_size_t, 
                'q': c_size_t}

normal_ll_argtypes = {'sigma2': POINTER(c_double),
                      'residuals2': POINTER(c_double),
                      'n': c_size_t}

lib_general.garch_variance_pq.argtypes = [c_type for c_type in garch_argtypes.values()]
lib_general.garch_variance_pq.restype = None

lib_general.normal_likelihood.argtypes = [c_type for c_type in normal_ll_argtypes.values()]
lib_general.normal_likelihood.restype = c_double

special_argtypes = {'parameters': POINTER(c_double),
                    'residuals2': POINTER(c_double),
                    'sigma2': POINTER(c_double),
                    'n': c_size_t}

lib_special.special_garch_oo_normal.argtypes = [c_type for c_type in special_argtypes.values()]
lib_special.special_garch_oo_normal.restype = c_double


def implementation_1():  
    n = len(residuals)
    residuals2 = residuals**2
    sigma2 = np.zeros(n, dtype=np.float64)
    sigma2[0] = np.sum(residuals2) / n
    constant_ll = 0.5 * n * np.log(2 * np.pi)

    residuals2_c = np.ascontiguousarray(residuals2, dtype=np.float64).ctypes.data_as(POINTER(c_double))
    sigma2_c = np.ascontiguousarray(sigma2, dtype=np.float64).ctypes.data_as(POINTER(c_double))

    if p == 1 and q == 1:
        def max_likelihood(parameters):
            parameters_c = np.ascontiguousarray(parameters, dtype=np.float64).ctypes.data_as(POINTER(c_double))
            return lib_special.special_garch_oo_normal(parameters_c, residuals2_c, sigma2_c, c_size_t(n))
                
    else:
        def compute_variance(parameters):
            parameters_ptr = np.ascontiguousarray(parameters, dtype=np.float64).ctypes.data_as(POINTER(c_double))
            lib_general.garch_variance_pq(parameters_ptr, residuals2_c, sigma2_c, c_size_t(n), c_size_t(p), c_size_t(q))

        def max_likelihood(parameters):
            compute_variance(parameters)
            likelihood = lib_general.normal_likelihood(sigma2_c, residuals2_c, c_size_t(n))
            return likelihood

    result = minimize(max_likelihood, initial_parameters, method='Nelder-Mead', bounds=optimization_bounds)
    # result = minimize(max_likelihood, initial_parameters, method='trust-constr', bounds=optimization_bounds)
    result.fun += constant_ll
    return result

timer = Timer()
result1 = timer.time_function("Implementation 1", implementation_1)

timer.print_results()

# gcc -Ofast -o compute_variance.so compute_variance.c -shared -fPIC -lm 
# objdump -d compute_variance.so | grep -A20 compute_variance

def compute_variance(parameters):
    parameters_ptr = np.ascontiguousarray(parameters, dtype=np.float64).ctypes.data_as(POINTER(c_double))
    lib_general.garch_variance_pq(parameters_ptr, residuals2_c, sigma2_c, c_size_t(n), c_size_t(p), c_size_t(q))
    return sigma2

n = len(residuals)
residuals2 = residuals**2
sigma2 = np.zeros(n, dtype=np.float64)
sigma2[0] = np.sum(residuals2) / n
constant_ll = 0.5 * n * np.log(2 * np.pi)

residuals2_c = np.ascontiguousarray(residuals2, dtype=np.float64).ctypes.data_as(POINTER(c_double))
sigma2_c = np.ascontiguousarray(sigma2, dtype=np.float64).ctypes.data_as(POINTER(c_double))




def compute_variance(parameters):
    parameters_ptr = np.ascontiguousarray(parameters, dtype=np.float64).ctypes.data_as(POINTER(c_double))
    lib_general.garch_variance_pq(parameters_ptr, residuals2_c, sigma2_c, c_size_t(n), c_size_t(p), c_size_t(q))

def max_likelihood(parameters):
    compute_variance(parameters)
    likelihood = lib_general.normal_likelihood(sigma2_c, residuals2_c, c_size_t(n))
    return likelihood

def max_likelihood(parameters):
    compute_variance(parameters)
    likelihood = lib_general.normal_likelihood(sigma2_c, residuals2_c, c_size_t(n))
    return likelihood

from scipy.special import loggamma

def log_likelihood(self, data: np.ndarray, sigma2: np.ndarray, nu: float) -> float:
    n = len(data)
    sigma2 = sigma2
    r2os2 = residuals2 / sigma2
    conts = n * (loggamma((nu + 1) / 2) - loggamma(nu / 2) - 0.5 * np.log(np.pi * nu))
    var1 = 0
    var2 = 0
    for i in range(n):
        var1 += np.log(sigma2[i])
        var2 += np.log(1 + r2os2[i] / nu)
    ll = conts - 0.5 * (var1 - (nu + 1)* var2)
    constant = loggamma((nu + 1) / 2) - loggamma(nu / 2) - 0.5 * np.log(np.pi * nu)
    log_likelihood = n * constant - np.sum(np.log(np.sqrt(sigma2)) + (nu + 1) / 2 * np.log(1 + data**2 / (nu * sigma2)))
    return -log_likelihood

def robust_covariance2(sigma2, residuals2, p, q):

    n = len(residuals2)

    size = 1 + p + q
    OPG = np.zeros((size, size))
    HESS = np.zeros((size, size))

    residuals2_c = np.ascontiguousarray(residuals2, dtype=np.float64).ctypes.data_as(POINTER(c_double))
    sigma2_c = np.ascontiguousarray(sigma2, dtype=np.float64).ctypes.data_as(POINTER(c_double))
    OPG_c = np.ascontiguousarray(OPG, dtype=np.float64).ctypes.data_as(POINTER(c_double))
    HESS_c = np.ascontiguousarray(HESS, dtype=np.float64).ctypes.data_as(POINTER(c_double))

    if p == 1 and q == 1:
        lib.special_garch_11_std_err_robust(residuals2_c, sigma2_c, OPG_c, HESS_c, c_size_t(n))
    else:
        lib.general_garch_pq_std_err_robust(residuals2_c, sigma2_c, OPG_c, HESS_c, c_size_t(n), c_size_t(p), c_size_t(q))

    HESS_INV = np.linalg.inv(HESS)
    return HESS_INV @ OPG @ HESS_INV



result = minimize(max_likelihood, initial_parameters, method='Nelder-Mead', bounds=optimization_bounds)
# sigma2 = compute_variance(result.x)
print(sigma2)
result.fun += constant_ll
print(result)

# Timing function
def time_function(func, *args):
    start = time.perf_counter()
    result = func(*args)
    end = time.perf_counter()
    elapsed_time = end - start
    return result, elapsed_time

# Second function timing
cov, time2 = time_function(robust_covariance2, sigma2, residuals2, p, q)
std_err = np.sqrt(np.diag(cov)) / np.sqrt(n)
print(f"robust_covariance2 took {time2:.6f} seconds")
print('z', np.round(result.x / std_err, 4))