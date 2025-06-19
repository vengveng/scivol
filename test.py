# from src.volkit.garch_lib2 import *
# from volkit import garch_lib2
from volkit.garch_lib2 import *
from alpha.garch_lib import CVM

import time
import pandas as pd
import os
import statsmodels.api as sm

# Load and preprocess data
root = os.path.dirname(__file__)
info_path = os.path.join(root, 'data', 'info.csv')
data = pd.read_csv(info_path, index_col=0, usecols=[0, 1, 2, 3], skiprows=1, parse_dates=False, names=['date', 'DAX', 'S&P', 'rate'])
data.index = pd.DatetimeIndex(data.index, freq='W-MON')
data = data[data.index.year < 2024]
data['rate'] = data['rate'] / 100 / 52
returns = data.pct_change().dropna()

# # Compute weekly returns and residuals
# weekly_returns = returns[['DAX', 'S&P']].reset_index()
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
# # weekly_returns = pd.concat([weekly_returns, weekly_returns], axis=0)
# # weekly_returns = pd.concat([weekly_returns, weekly_returns], axis=0)
# # weekly_returns = pd.concat([weekly_returns, weekly_returns], axis=0)
# print(weekly_returns.shape)

# # dax_residuals = AutoReg(weekly_returns['DAX'], lags=1, ).fit().resid
# # sp_residuals = AutoReg(weekly_returns['S&P'], lags=1).fit().resid
# dax_residuals = sm.tsa.arima.ARIMA(weekly_returns['DAX'], order=(1, 0, 1)).fit().resid
# sp_residuals = sm.tsa.arima.ARIMA(weekly_returns['S&P'], order=(1, 0, 1)).fit().resid
# residuals = dax_residuals.to_numpy()

residuals = pd.read_csv(os.path.join(root, 'data', 'residuals.csv'))['DAX'].to_numpy()
# residuals = residuals[:2398]

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
print(f"GARCH(1,1) fit_studentt execution time: {garch_time:.6f} seconds")
print(res_garch.summary())
print(f"Data shape: {residuals.shape}")

# Nelder-Mead
# (2398, 3)
# GARCH(1,1) fit_normal execution time: 1.489414 seconds
# Data shape: (1227776,)
# Model: GARCH(1,1), Distribution: normal
# Log-Likelihood: 2540054.5613
# Parameters: [5.33591065e-05 1.74006894e-01 7.88862612e-01]
# Standard Errors: [2.24468718e-03 1.96708019e+00 3.33372034e+00]
# T-Statistics: [ 26.33978903  98.01757635 262.1994509 ]
# None
# GARCH(1,1) fit_normal execution time: 2.170479 seconds
# Model: GARCH(1,1), Distribution: studentt
# Log-Likelihood: 2571287.3949
# Parameters: [5.33591065e-05 1.74006894e-01 7.88862612e-01]
# Standard Errors: [2.24468718e-03 1.96708019e+00 3.33372034e+00]
# T-Statistics: [ 26.33978903  98.01757635 262.1994509 ]
# NORMAL: 6.1130
# None
# Data shape: (1227776,)

# trust-constr
# GARCH(1,1) fit_normal execution time: 12.722743 seconds
# Data shape: (1227776,)
# Model: GARCH(1,1), Distribution: normal
# Log-Likelihood: 2540054.5602
# Parameters: [5.33446946e-05 1.73969851e-01 7.88904592e-01]
# Standard Errors: [2.24473697e-03 1.96714765e+00 3.33380817e+00]
# T-Statistics: [ 26.33209083  97.99334947 262.20649626]
# GARCH(1,1) fit_studentt execution time: 13.377369 seconds
# Model: GARCH(1,1), Distribution: studentt
# Log-Likelihood: 2571288.2027
# Parameters: [5.33446946e-05 1.73969851e-01 7.88904592e-01]
# Standard Errors: [2.24473697e-03 1.96714765e+00 3.33380817e+00]
# T-Statistics: [ 26.33209083  97.99334947 262.20649626]
# NORMAL: 6.1131
# None
# Data shape: (1227776,)