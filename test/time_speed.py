# Test definitions below


import time
from typing import Callable, Any
import numpy as np
import pandas as pd
from scipy.optimize import minimize, direct, shgo, brute, Bounds, differential_evolution
import os
from statsmodels.tsa.ar_model import AutoReg
from numba import jit, njit, float64

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
        res_str = f'{result.x[0]:.4f} {result.x[1]:.4f} {result.x[2]:.4f} | {result.fun:.2f}'
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
info_path = "data/info.csv"
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

residuals = dax_residuals.to_numpy()
stationarity_vector = [0, 1, 1]
optimization_bounds = [(0, +np.inf), (0.000001, 1), (0.000001, 1)]
initial_parameters = [0.00001, 0.05, 0.9]
lim = np.var(residuals)


def implementation_1():
    def log_likelihood(data: np.ndarray, sigma2: np.ndarray) -> float:
        n = len(data)
        constant = -0.5 * n * np.log(2 * np.pi)
        log_likelihood = constant - 0.5 * (np.sum(np.log(sigma2) + data**2 / sigma2))
        return -log_likelihood

    # @njit
    def compute_variance(parameters, residuals):
        n = len(residuals)
        sigma2 = np.zeros(n)
        sigma2[0] = np.sum(residuals**2) / n

        residuals2 = residuals**2
        for t in range(1, n):
            sigma2[t] = step_forecast(parameters, residuals2[t-1], sigma2[t-1])
        return sigma2

    # @njit
    def step_forecast(parameters, residuals2, sigma2):
        omega, alpha, beta = parameters
        return omega + alpha * residuals2 + beta * sigma2

    def max_likelihood(parameters):
        return log_likelihood(residuals, compute_variance(parameters, residuals))

    result = minimize(max_likelihood, initial_parameters, method='Nelder-Mead', bounds=optimization_bounds)
    return result

def implementation_2():
    n = len(residuals)
    sigma2 = np.zeros(n)
    residuals2 = residuals**2
    sigma2[0] = np.sum(residuals2) / n
    constant_ll = 0.5 * n * np.log(2 * np.pi)

    def compute_variance(parameters):
        for i in range(1, n):
            sigma2[i] = parameters[0] + parameters[1] * residuals2[i-1] + parameters[2] * sigma2[i-1]

    def log_likelihood() -> float:
        return constant_ll + 0.5 * (np.sum(np.log(sigma2) + residuals2 / sigma2))

    def max_likelihood(parameters):
        compute_variance(parameters)
        return log_likelihood()

    result = minimize(max_likelihood, initial_parameters, method='Nelder-Mead', bounds=optimization_bounds)
    return result

def implementation_3():
    n = len(residuals)
    sigma2 = np.zeros(n)
    residuals2 = residuals**2
    sigma2[0] = np.sum(residuals2) / n
    constant_ll = 0.5 * n * np.log(2 * np.pi)

    @njit
    def compute_variance(parameters, residuals2, sigma2, n):
        for i in range(1, n):
            sigma2[i] = parameters[0] + parameters[1] * residuals2[i - 1] + parameters[2] * sigma2[i - 1]
        return sigma2

    def log_likelihood(sigma2) -> float:
        return constant_ll + 0.5 * (np.sum(np.log(sigma2) + residuals2 / sigma2))
    
    def max_likelihood(parameters):
        return log_likelihood(compute_variance(parameters, residuals2, sigma2, n))

    result = minimize(max_likelihood, initial_parameters, method='Nelder-Mead', bounds=optimization_bounds)
    return result


from ctypes import CDLL, c_double, POINTER, c_size_t
lib = CDLL('lib/DEPRECATED_compute_variance.so')
lib.compute_variance.argtypes = [
    POINTER(c_double),  # parameters
    POINTER(c_double),  # residuals2
    POINTER(c_double),  # sigma2
    c_size_t        # n
]

lib.compute_variance.restype = None

def implementation_4():
    n = len(residuals)
    sigma2 = np.zeros(n)
    residuals2 = residuals**2
    sigma2[0] = np.sum(residuals2) / n
    constant_ll = 0.5 * n * np.log(2 * np.pi)
    residuals2 = np.ascontiguousarray(residuals2, dtype=np.float64)
    sigma2 = np.ascontiguousarray(sigma2, dtype=np.float64)

    def compute_variance(parameters, residuals2, sigma2, n):
        # Ensure arrays are contiguous and of type float64

        parameters = np.ascontiguousarray(parameters, dtype=np.float64)
        # Convert numpy arrays to C pointers
        parameters_c = parameters.ctypes.data_as(POINTER(c_double))
        residuals2_c = residuals2.ctypes.data_as(POINTER(c_double))
        sigma2_c = sigma2.ctypes.data_as(POINTER(c_double))

        # Call the C function
        lib.compute_variance(parameters_c, residuals2_c, sigma2_c, c_size_t(n))
        return sigma2

    def log_likelihood(sigma2) -> float:
        return constant_ll + 0.5 * (np.sum(np.log(sigma2) + residuals2 / sigma2))
    
    def max_likelihood(parameters):
        return log_likelihood(compute_variance(parameters, residuals2, sigma2, n))

    result = minimize(max_likelihood, initial_parameters, method='Nelder-Mead', bounds=optimization_bounds)
    return result

lib.log_likelihood.argtypes = [
    POINTER(c_double),  # parameters
    POINTER(c_double),  # residuals2
    c_size_t            # n
]
lib.log_likelihood.restype = c_double

def implementation_5():
    # Prepare data
    n = len(residuals)
    residuals2 = np.ascontiguousarray(residuals**2, dtype=np.float64)

    # Likelihood function in C
    def log_likelihood_c(parameters):
        parameters = np.ascontiguousarray(parameters, dtype=np.float64)
        parameters_c = parameters.ctypes.data_as(POINTER(c_double))
        residuals2_c = residuals2.ctypes.data_as(POINTER(c_double))

        return lib.log_likelihood(parameters_c, residuals2_c, c_size_t(n))

    # Use scipy.optimize to find the maximum likelihood
    result = minimize(log_likelihood_c, initial_parameters, method='Nelder-Mead', bounds=optimization_bounds)
    return result

lib.compute_log_likelihood.argtypes = [
    POINTER(c_double),  # parameters
    POINTER(c_double),  # residuals2
    POINTER(c_double),  # sigma2
    c_size_t        # n
]

lib.compute_log_likelihood.restype = c_double

def implementation_6():
    # Prepare data
    n = len(residuals)
    residuals2 = residuals**2
    sigma2 = np.zeros(n, dtype=np.float64)
    sigma2[0] = np.sum(residuals2) / n
    constant_ll = 0.5 * n * np.log(2 * np.pi)

    # Convert them to contiguous arrays
    residuals2_c = np.ascontiguousarray(residuals2, dtype=np.float64)
    sigma2_c = np.ascontiguousarray(sigma2, dtype=np.float64)

    # Get pointers
    residuals2_ptr = residuals2_c.ctypes.data_as(POINTER(c_double))
    sigma2_ptr     = sigma2_c.ctypes.data_as(POINTER(c_double))

    def log_likelihood_c(parameters):
        # Convert parameters to contiguous array + pointer
        parameters = np.ascontiguousarray(parameters, dtype=np.float64)
        parameters_ptr = parameters.ctypes.data_as(POINTER(c_double))

        # Now call the C function with the correct pointer types
        ll = lib.compute_log_likelihood(
            parameters_ptr,
            residuals2_ptr,
            sigma2_ptr,
            c_size_t(n)
        )
        return ll

    # Then do your optimization
    result = minimize(log_likelihood_c, initial_parameters, method='Nelder-Mead', bounds=optimization_bounds)
    result.fun += constant_ll
    return result


timer = Timer()
result1 = timer.time_function("Implementation 1", implementation_1)
# result2 = timer.time_function("Implementation 2", implementation_2)
# result3 = timer.time_function("Implementation 3", implementation_3)
# result4 = timer.time_function("Implementation 4", implementation_4)
# result5 = timer.time_function("Implementation 5", implementation_5)
result6 = timer.time_function("Implementation 6", implementation_6)

timer.print_results()


import statsmodels.api as sm
from scipy.stats import chisquare, chi2
import numpy as np

def pit_adequacy_test(ut, K=4, N=10):
    """
    Perform PIT adequacy test using LM test for serial correlation and Pearson's chi-squared test for uniformity.
    
    Parameters:
    - ut: np.ndarray of PIT-transformed values.
    - K: int, number of lags for LM test.
    - N: int, number of bins for Pearson's chi-squared test.
    
    Returns:
    - dict with LM test and uniformity test results.
    """
    import statsmodels.api as sm
    from scipy.stats import chisquare, chi2
    import numpy as np

    # Step 1: Serial Correlation Test (Custom LM Test)
    lags = sm.tsa.lagmat(ut - np.mean(ut), maxlag=K, trim="both")
    lags = sm.add_constant(lags)  # Add constant for regression
    dependent_var = ut[K:] - np.mean(ut)

    # Perform regression
    model = sm.OLS(dependent_var, lags)
    result = model.fit()

    # LM statistic: T * R^2
    lm_stat = len(dependent_var) * result.rsquared
    p_value_lm = 1 - chi2.cdf(lm_stat, df=K)

    # Step 2: Uniformity Test (Pearson's Chi-Squared Test)
    counts, _ = np.histogram(ut, bins=np.linspace(0, 1, N + 1))
    expected_counts = len(ut) / N
    chi2_stat, p_value_uniformity = chisquare(counts, f_exp=[expected_counts] * N)

    return {
        "LM Statistic": lm_stat,
        "LM P-value": p_value_lm,
        "Chi-Squared Statistic": chi2_stat,
        "Uniformity P-value": p_value_uniformity,
    }

sigma2 = np.zeros(len(residuals))
sigma2[0] = np.var(residuals)

for i in range(1, len(residuals)):
    sigma2[i] = result6.x[0] + result6.x[1] * residuals[i-1]**2 + result6.x[2] * sigma2[i-1]

from scipy.stats import norm

zt = residuals / np.sqrt(sigma2)
ut = norm.cdf(zt)
results = pit_adequacy_test(ut)
print(results)

import numpy as np
import matplotlib.pyplot as plt

def plot_pit_results(ut, bins=10):
    """
    Plot the PIT-transformed values (u_t) as a histogram with a uniform distribution overlay.

    Parameters:
    - ut: np.ndarray of PIT-transformed values.
    - bins: int, number of bins for the histogram.

    Returns:
    - None (shows the plot).
    """
    # Compute histogram
    hist, edges = np.histogram(ut, bins=np.linspace(0, 1, bins + 1), density=True)

    # Plot the histogram
    plt.figure(figsize=(8, 5))
    plt.bar(edges[:-1], hist, width=np.diff(edges), align="edge", alpha=0.7, label="Histogram of u_t")

    # Add uniform distribution line
    plt.axhline(1, color="black", linestyle="-", linewidth=2, label="Uniform distribution (U[0,1])")

    # Customize plot
    plt.title("PIT Transform Histogram and Uniform Distribution")
    plt.xlabel("u (PIT-transformed values)")
    plt.ylabel("Density")
    plt.legend(loc="upper right")
    plt.grid(axis="y", linestyle="--", alpha=0.7)

    # Show the plot
    plt.show()

# Example usage with your PIT-transformed values (ut)
plot_pit_results(ut, bins=20)

import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats

def plot_qq(ut, theoretical_dist="uniform"):
    """
    Create a QQ plot for the PIT-transformed data (ut) with a black background.
    
    Parameters:
    - ut: np.ndarray of PIT-transformed values.
    - theoretical_dist: str, either "uniform" (U[0,1]) or "normal" (N(0,1)).
    
    Returns:
    - None (shows the plot).
    """
    plt.figure(figsize=(6, 6), facecolor='w')  # Set figure background to w
    
    if theoretical_dist == "uniform":
        # Uniform QQ plot
        stats.probplot(ut, dist="uniform", plot=plt)
        plt.title("QQ Plot - Uniform Distribution (U[0,1])", color="white")
        plt.xlabel("Theoretical Quantiles", color="white")
        plt.ylabel("Sample Quantiles", color="white")
    elif theoretical_dist == "normal":
        # Normal QQ plot
        stats.probplot(ut, dist="norm", plot=plt)
        plt.title("QQ Plot - Normal Distribution (N(0,1])", color="black")
        plt.xlabel("Theoretical Quantiles", color="black")
        plt.ylabel("Sample Quantiles", color="black")
    else:
        raise ValueError("Unsupported theoretical_dist. Choose 'uniform' or 'normal'.")
    
    # Adjust plot colors for w background
    ax = plt.gca()
    ax.set_facecolor("w")  # Set axis background to black
    ax.tick_params(colors="black")  # Set tick colors to b
    for spine in ax.spines.values():
        spine.set_color("black")  # Set axis spine color to white
    
    plt.grid(alpha=0.3, color="gray")  # Grid with gray lines
    plt.show()

# Example usage with your PIT-transformed values (ut)
plot_qq(ut, theoretical_dist="uniform")  # For uniform distribution


import gnuplotlib as gp
from scipy.stats import probplot

def ascii_qq_plot(data, bins=20, pad=2):
    """
    Create a custom ASCII QQ plot with padding, axes, and labels.

    Parameters:
    - data: np.ndarray, the sample quantiles (data values).
    - bins: int, the size of the X by X grid.
    - pad: int, the horizontal padding between columns (default=2).

    Returns:
    - None (prints the ASCII plot).
    """
    # Generate theoretical and sample quantiles
    prob = probplot(data, dist="uniform", plot=None)
    theoretical_quantiles, sample_quantiles = prob[0]

    # Define grid size
    grid_size = bins
    grid = [[" " for _ in range(grid_size * pad)] for _ in range(grid_size)]

    # Normalize the data to fit the grid
    # min_x, max_x = min(theoretical_quantiles), max(theoretical_quantiles)
    min_y, max_y = min(sample_quantiles), max(sample_quantiles)
    min_x, max_x = min_y, max_y

    def scale_to_grid(value, min_val, max_val):
        """Scale a value to fit within the grid."""
        return int((value - min_val) / (max_val - min_val) * (grid_size - 1))

    # Place the theoretical line (inverse diagonal)
    for i in range(grid_size):
        grid[grid_size - 1 - i][i * pad] = "."  # Inverse diagonal: y = x

    # Bin the data and place empirical points
    for x, y in zip(theoretical_quantiles, sample_quantiles):
        grid_x = scale_to_grid(x, min_x, max_x) * pad
        grid_y = grid_size - 1 - scale_to_grid(y, min_y, max_y)  # Flip y-axis
        if grid[grid_y][grid_x] == " ":  # Only place if empty
            grid[grid_y][grid_x] = "*"

    # Add axes
    for i in range(grid_size):
        grid[i][0] = "|"  # Vertical axis
    horizontal_axis = ["-" if j % pad == 0 else " " for j in range(grid_size * pad)]
    horizontal_axis[0] = "."
    grid.append(horizontal_axis)  # Add horizontal axis

    # Add labels
    x_label = f"Theoretical Quantiles: [.] [{min_x:.2f}, {max_x:.2f}]"
    y_label = f"Emprical Quantiles:    [*] [{min_y:.2f}, {max_y:.2f}]"

    print("QQ Plot")
    # Print the grid
    for row in grid:
        print("".join(row))

    # Print labels below the grid
    
    print(f"{' ' * (len(horizontal_axis) // 2 - len(x_label) // 2)}")
    print(x_label)
    print(f"{y_label}")


# Example Usage
ascii_qq_plot(ut, 20)

# # gcc -Ofast -o compute_variance.so compute_variance.c -shared -fPIC -lm 
# # objdump -d compute_variance.so | grep -A20 compute_variance

