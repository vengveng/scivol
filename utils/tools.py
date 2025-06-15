import numpy as np
from tabulate import tabulate
import pandas as pd

class ModelResult:
    """Encapsulates results from a GARCH-like model, supporting nested estimations."""
    def __init__(self, p, q, distr=None, params=None, ll=None):
        self.p = p
        self.q = q
        self.distr = distr  # 'normal', 'studentt', etc.
        self.params = params  # Model parameters (e.g., GARCH omega, alpha, beta)
        self.ll = ll  # Log-likelihood
        self.n = None  # Sample size
        self.var = None  # Conditional variance
        self.opg = None  # Outer-product gradient
        self.hess = None  # Hessian matrix
        self.cov = None  # Covariance matrix
        self.rcov = None  # Robust covariance matrix
        self.std = None  # Standard errors
        self.t = None  # T-statistics
        self.secondary = None  # Store additional fits (e.g., Student-t nu)
        self.resid = None
        self.stdresid = None

    def update(self, result, residuals, variance_func: callable, std_func: callable):
        """Update ModelResult with fitted parameters and statistics."""
        self.params = result.x
        self.ll = -result.fun
        self.n = len(residuals)
        self.var = variance_func(residuals, self.params)
        self.resid = residuals
        self.stdresid = residuals / np.sqrt(self.var)

        OPG, HESS = std_func(self.p, self.q, residuals, self.var)
        self.opg = OPG
        self.hess = HESS
        self.cov = np.linalg.inv(HESS)
        self.rcov = self.cov @ OPG @ self.cov
        self.std = np.sqrt(np.diag(self.rcov))
        self.t = self.params / self.std * np.sqrt(len(residuals))

    def add_secondary(self, result, param_name):
        """Stores secondary estimation results (e.g., nu for Student-t)."""
        if self.secondary is None:
            self.secondary = {}
        self.secondary[param_name] = result.x[0]  # Store single parameter
        self.ll = -result.fun  # Update log-likelihood

    def summary(self):
        """Prints a summary of the results."""
        print(f"Model: GARCH({self.p},{self.q}), Distribution: {self.distr}")
        print(f"Log-Likelihood: {self.ll:.4f}")
        print(f"Parameters: {self.params}")
        print(f"Standard Errors: {self.std}")
        print(f"T-Statistics: {self.t}")
        if self.secondary:
            for param, value in self.secondary.items():
                print(f"{param}: {value:.4f}")

class CombinedModelResult:
    """Encapsulates results from a combined GARCH-like model, supporting nested estimations."""
    def __init__(self, results):
        self.r = results

class DataHandler:
    def __init__(self, data):
        self.original_data = data
        
        # Convert input data to numpy and store metadata
        if isinstance(data, pd.Series):
            self.data = data.to_numpy()  
            self.index = data.index      
            self.columns = None       
            self.is_univariate = True
            self.shape = data.shape 
        elif isinstance(data, pd.DataFrame):
            self.data = data.to_numpy()
            self.index = data.index      
            self.columns = data.columns  
            self.is_univariate = False    
            self.shape = data.shape
        elif isinstance(data, np.ndarray):
            self.data = data
            if data.ndim == 1:
                self.index = None
                self.columns = None
                self.is_univariate = True
                self.shape = data.shape
            elif data.ndim == 2:
                self.index = None
                self.columns = None
                self.is_univariate = False
                self.shape = data.shape
            else:
                raise ValueError("Input numpy array must be 1D or 2D.")
        else:
            raise TypeError("Input data must be a pandas Series, DataFrame, or numpy array.")

    def get_numpy(self):
        """Return the numpy version of the data for internal calculations."""
        return self.data

    def to_original_format(self, data):
        """Convert numpy array back to the original format (Series, DataFrame, or ndarray)."""
        if self.is_univariate:
            if isinstance(self.original_data, pd.Series):
                return pd.Series(data, index=self.index)
            else:
                return data
        else:
            if isinstance(self.original_data, pd.DataFrame):
                return pd.DataFrame(data, index=self.index, columns=self.columns)
            else:
                return data
            
    def slice(self, column_index):
        """Returns a new DataHandler for a specific column slice (for univariate model fitting)."""
        sliced_data = self.data[:, column_index] if self.data.ndim > 1 else self.data
        slice_handler = DataHandler(sliced_data)
        slice_handler.index = self.index
        slice_handler.columns = [self.columns[column_index]] if any(self.columns) else None
        slice_handler.is_univariate = True
        slice_handler.shape = [self.shape[0], 1]
        return slice_handler




class Plot:
    def __init__(self):
        self.results = None

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
    # plot_pit_results(ut, bins=20)

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
            probplot(ut, dist="uniform", plot=plt)
            plt.title("QQ Plot - Uniform Distribution (U[0,1])", color="white")
            plt.xlabel("Theoretical Quantiles", color="white")
            plt.ylabel("Sample Quantiles", color="white")
        elif theoretical_dist == "normal":
            # Normal QQ plot
            probplot(ut, dist="norm", plot=plt)
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
    # plot_qq(ut, theoretical_dist="uniform")  # For uniform distribution


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
    # ascii_qq_plot(ut, 20)

