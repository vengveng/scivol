import time
from typing import Callable, Any
import numpy as np
import pandas as pd
from scipy.optimize import minimize
import os
from statsmodels.tsa.ar_model import AutoReg
from numba import jit, njit
from itertools import product

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
        print(time_str)
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

dax_residuals = AutoReg(weekly_returns['DAX'], lags=1, ).fit().resid
sp_residuals = AutoReg(weekly_returns['S&P'], lags=1).fit().resid

residuals = sp_residuals.to_numpy()
stationarity_vector = [0, 1, 1]
optimization_bounds = [(0, +np.inf), (0.000001, 1), (0.000001, 1)]
initial_parameters = [0.00001, 0.05, 0.9]

def surface_1(mesh_density):
    n = len(residuals)
    sigma2 = np.zeros(n)
    residuals2 = residuals**2
    sigma2[0] = np.sum(residuals2) / n
    constant_ll = 0.5 * n * np.log(2 * np.pi)

    omega = np.linspace(0.00001, 1, mesh_density)
    alpha = np.linspace(0.00001, 1, mesh_density)
    beta = np.linspace(0.00001, 1, mesh_density)

    cartesian_product = np.array(list(product(omega, alpha, beta)))
    ll_column = np.zeros(cartesian_product.shape[0])
    ll_mesh = np.column_stack((cartesian_product, ll_column))

    @njit
    def compute_variance(parameters, residuals2, sigma2, n):
        assert np.isscalar(parameters[0]) and np.isscalar(parameters[1]) and np.isscalar(parameters[2])
        for i in range(1, n):
            sigma2[i] = parameters[0] + parameters[1] * residuals2[i-1] + parameters[2] * sigma2[i-1]

    def log_likelihood(sigma2) -> float:
        return constant_ll + 0.5 * (np.sum(np.log(sigma2) + residuals2 / sigma2))

    for i in range(ll_mesh.shape[0]):
        compute_variance(ll_mesh[i, :3], residuals2, sigma2, n)
        ll_mesh[i, 3] = log_likelihood(sigma2)
    



timer = Timer()
mesh1 = timer.time_function("Surface 1", surface_1, 100)

timer.print_results()

