from garch_lib2 import *
from alpha.garch_lib import CVM

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
weekly_returns = pd.concat([weekly_returns, weekly_returns], axis=0)
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

# dax_residuals = AutoReg(weekly_returns['DAX'], lags=1, ).fit().resid
# sp_residuals = AutoReg(weekly_returns['S&P'], lags=1).fit().resid
dax_residuals = sm.tsa.arima.ARIMA(weekly_returns['DAX'], order=(1, 0, 1)).fit().resid
sp_residuals = sm.tsa.arima.ARIMA(weekly_returns['S&P'], order=(1, 0, 1)).fit().resid
residuals = dax_residuals.to_numpy()

# m = GARCH(1, 1)
# res = m.fit(residuals, 'normal')
# print([f"{x:.4f}" for x in res.params])
# print([f"{x:.4f}" for x in res.std])
# print([f"{x:.4f}" for x in res.t])
# print(res.params)

# m = CVM('garch', 'normal')
# res = m.fit(residuals)
# print(res)

# Timing GARCH(1,1)
m = GARCH(1, 1)
start_time = time.time()
res_garch = m.fit(residuals, 'normal')
end_time = time.time()
garch_time = end_time - start_time
print(f"GARCH(1,1) fit_normal execution time: {garch_time:.6f} seconds")
print(f"Data shape: {residuals.shape}")
print(res_garch.summary())

# # Timing CVM('garch', 'normal')
# m = CVM('garch', 'normal')
# start_time = time.time()
# res_cvm = m.fit(residuals)
# end_time = time.time()
# cvm_time = end_time - start_time
# print(f"CVM('garch', 'normal') fit execution time: {cvm_time:.6f} seconds")

# Timing GARCH(1,1)
m = GARCH(1, 1)
start_time = time.time()
res_garch = m.fit(residuals, 'studentt')
end_time = time.time()
garch_time = end_time - start_time
print(f"GARCH(1,1) fit_normal execution time: {garch_time:.6f} seconds")
print(res_garch.summary())
print(f"Data shape: {residuals.shape}")

