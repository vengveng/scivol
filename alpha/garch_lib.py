import numpy as np
import pandas as pd
from scipy.optimize import minimize, LinearConstraint
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from scipy.special import loggamma
from scipy.stats import jf_skew_t, multivariate_normal
from tabulate import tabulate
from typing import List, Tuple
import warnings

LIKELIHOOD_LIBRARY = {
    ('garch', 'normal'): 'NormalLikelihoodTypeA',
    ('gjr', 'normal'): 'NormalLikelihoodTypeA',
    ('garch', 'studentt'): 'StudentTLikelihoodTypeA',
    ('gjr', 'studentt'): 'StudentTLikelihoodTypeA',
    ('garch', 'skewt'): 'SkewTLikelihoodTypeA',
    ('gjr', 'skewt'): 'SkewTLikelihoodTypeA',
    ('dcc', 'normal'): 'MultivariateNormalLikelihoodTypeA',
    ('dcc', 'studentt'): 'MultivariateStudentTLikelihoodTypeA',
    ('dcc', 'skewt'): 'MultivariateSkewTLikelihoodTypeA',
}

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

class LikelihoodFunction(ABC):
    @abstractmethod
    def log_likelihood() -> float:
        """Calculate the likelihood for the given model and data."""
        pass
    
    @abstractmethod
    def get_param_format() -> dict:
        """Return the parameter format for the model."""
        pass

class NormalLikelihoodTypeA(LikelihoodFunction):
    def log_likelihood(self, data: np.ndarray, sigma2: np.ndarray) -> float:
        n = len(data)
        constant = -0.5 * n * np.log(2 * np.pi)
        log_likelihood = constant - 0.5 * (np.sum(np.log(sigma2) + data**2 / sigma2))
        return -log_likelihood
    
    def get_bounds_and_shape_parameters(self):
        return [], []

    def extract_shape_parameters(self, params):
        return {}, params
    
    def get_param_format(self):
        return {}
    
class StudentTLikelihoodTypeA(LikelihoodFunction):
    def log_likelihood(self, data: np.ndarray, sigma2: np.ndarray, nu: float) -> float:
        n = len(data)
        constant = loggamma((nu + 1) / 2) - loggamma(nu / 2) - 0.5 * np.log(np.pi * nu)
        log_likelihood = n * constant - np.sum(np.log(np.sqrt(sigma2)) + (nu + 1) / 2 * np.log(1 + data**2 / (nu * sigma2)))
        return -log_likelihood
    
    def get_bounds_and_shape_parameters(self):
        return [(0.0001, +np.inf)], [10.0]
    
    def extract_shape_parameters(self, params):
        nu = params[-1]
        return {'nu': nu}, params[:-1]
    
    def get_param_format(self):
        return {'nu': 0}
    
class SkewTLikelihoodTypeA(LikelihoodFunction):
    # TODO
    # FIXME
    def log_likelihood(self, data: np.ndarray, sigma2: np.ndarray, nu: float, xi: float) -> float:
        log_likelihood = np.sum(jf_skew_t.logpdf(data, nu, xi, scale=np.sqrt(sigma2)))
        return -log_likelihood
    
    def get_bounds_and_shape_parameters(self):
        return [(0.0001, +np.inf), (0.0001, +np.inf)], [10.0, 5]
    
    def extract_shape_parameters(self, params):
        nu, xi = params[-2:]
        return {'a': nu, 'b': xi}, params[:-2]
    
    def get_param_format(self):
        return {'a': 0, 'b': 1}

class MultivariateNormalLikelihoodTypeA(LikelihoodFunction):
    def log_likelihood(self, correlations, normalized_residuals):
        """
        - normalized_residuals: (T x N) matrix, where T is the number of time steps, and N is the number of variables.
        - correlations: (T x N x N) matrix of time-varying correlation matrices.
        """
        T = normalized_residuals.shape[0]
        log_likelihood = 0

        for t in range(T):
            u_t = normalized_residuals[t, :]
            correlation_t = correlations[t, :]
            log_determinant = np.log(np.linalg.det(correlation_t))
            inverse_correlation_t = np.linalg.inv(correlation_t)
            quadratic_form = u_t.T @ inverse_correlation_t @ u_t
            log_likelihood += -0.5 * (log_determinant + quadratic_form)
        return -log_likelihood
    
    def get_bounds_and_shape_parameters(self):
        return [], []
    
    def extract_shape_parameters(self, params):
        return {}, params
    
    def get_param_format(self):
        return {}
    
class MultivariateStudentTLikelihoodTypeA(LikelihoodFunction):
    def log_likelihood(self, correlations, normalized_residuals, nu):
        """
        - normalized_residuals: (T x N) matrix, where T is the number of time steps, and N is the number of variables.
        - correlations: (T x N x N) matrix of time-varying correlation matrices.
        """
        T, n = normalized_residuals.shape
        log_likelihood = 0

        for t in range(T):
            u_t = normalized_residuals[t, :]
            correlation_t = correlations[t, :]
            log_determinant = np.log(np.linalg.det(correlation_t))
            quadratic_form = u_t.T @ np.linalg.inv(correlation_t) @ u_t
            A = loggamma((nu + n) / 2) - loggamma(nu / 2) - 0.5 * np.log(np.pi * (nu - 2))
            B = -0.5 * (log_determinant + (nu + n) * np.log(1 + quadratic_form / (nu - 2)))
            log_likelihood += A + B
        return -log_likelihood
    
    def get_bounds_and_shape_parameters(self):
        return [(0.0001, 1e6)], [10.0]
    
    def extract_shape_parameters(self, params):
        nu = params[-1]
        return {'ddof': nu}, params[:-1]
    
    def get_param_format(self):
        return {'ddof': 0}
    
class MultivariateSkewTLikelihoodTypeA(LikelihoodFunction):
    def log_likelihood(self, correlations, normalized_residuals, nu, delta, D_inv_diag):
        """
        - normalized_residuals: (T x N) matrix, where T is the number of time steps, and N is the number of variables.
        - correlations: (T x N x N) matrix of time-varying correlation matrices (R_t for each t).
        - delta: Scalar skewness parameter, assumed uniform across assets.
        - D_inv_diag: T x N x N diagonal matrix of the inverse square root of the conditional variances (D^-1).
        """
        T, N = normalized_residuals.shape
        log_likelihood = 0
        delta = delta * np.ones(N)  # Assume uniform skewness across assets

        for t in range(T):
            u_t = normalized_residuals[t, :]
            R_t = correlations[t, :]  # Correlation matrix at time t

            # Regularize correlation matrix to avoid issues in sqrt and ensure positive definiteness
            R_t = np.clip(R_t, a_min=1e-6, a_max=None)  # Ensure no zero/negative values
            R_t += np.eye(N) * 1e-6  # Regularization

            log_determinant = np.log(np.linalg.det(R_t))
            quadratic_form = u_t.T @ np.linalg.inv(R_t) @ u_t

            # Step 1: Standard t-distribution part
            A = loggamma((nu + N) / 2) - loggamma(nu / 2) - 0.5 * np.log(np.pi * (nu - 2))
            B = -0.5 * (log_determinant + (nu + N) * np.log(1 + quadratic_form / (nu - 2)))

            # Step 2: Skewness adjustment based on Azzalini’s skewed t-distribution
            D_inv_t = np.diag(D_inv_diag[t, :, :])  # Diagonal elements (D^-1)

            # Compute the inverse square root of R_t (R_t^{-1/2}) using eigenvalue decomposition
            eigvals, eigvecs = np.linalg.eigh(R_t)
            R_t_inv_sqrt = eigvecs @ np.diag(1.0 / np.sqrt(eigvals)) @ eigvecs.T

            # Construct H_t^{-1/2} = D^{-1} @ R_t^{-1/2} @ D^{-1}
            H_inv_sqrt = np.diag(D_inv_t) @ R_t_inv_sqrt @ np.diag(D_inv_t)

            # Compute skew term
            xi = D_inv_t @ delta  # Element-wise multiplication of D_inv and delta
            skew_term = delta * D_inv_t @ (H_inv_sqrt @ u_t - xi)
            skew_term = skew_term * np.sqrt((nu + N) / (quadratic_form + nu))

            # Use the simplified expression for C
            C = np.log(2) - 0.5 * skew_term ** 2

            # Update the log-likelihood
            log_likelihood += A + B + C - 0.5 * np.log(np.linalg.det(R_t))

        # Check for NaN or Inf values
        if np.isnan(log_likelihood) or np.isinf(log_likelihood):
            print("NaN or Inf detected in log_likelihood")
            return np.inf  # Return a large value to penalize invalid points

        return -log_likelihood
    
    def get_bounds_and_shape_parameters(self):
        return [(2, 1e6), (-1, 1)], [10.0, 0.8]
    
    def extract_shape_parameters(self, params):
        nu, xi = params[-2:]
        return {'ddof': nu, 'asymmetry': xi}, params[:-2]
    
    def get_param_format(self):
        return {'ddof': 0, 'asymmetry': 1}

class LikelihoodFactory:
    @staticmethod
    def get_likelihood(volatility_model: str, distribution: str) -> LikelihoodFunction:
        """Return the correct likelihood function class based on the model and distribution."""
        key = (volatility_model.lower(), distribution.lower())
        likelihood_class_name = LIKELIHOOD_LIBRARY.get(key)
        
        if likelihood_class_name is None:
            raise ValueError(f"No likelihood function found for model '{volatility_model}' with distribution '{distribution}'")
        
        return globals()[likelihood_class_name]()

class VolatilityModel(ABC):
    @abstractmethod
    def compute_variance(self):
        """Calculate conditional variances for each series."""
        pass

    @abstractmethod
    def initial_params(self):
        """Initial parameter guess for the volatility model."""
        pass
    
    @abstractmethod
    def optimization_bounds(self):
        """Return optimization bounds for the parameters."""
        pass

    @abstractmethod
    def get_param_format(self):
        """Return the parameter format for the model."""
        pass

    @abstractmethod
    def step_forecast(self):
        """Calculate the next forecasted variance."""
        pass

class GARCHVolatilityModel(VolatilityModel):
    def compute_variance(self, parameters, residuals):
        """Calculate conditional variances for each series using GARCH model."""
        n = len(residuals)
        sigma2 = np.zeros(n)
        sigma2[0] = np.sum(residuals**2) / n

        residuals2 = residuals**2
        omega, alpha, beta = parameters
        
        for t in range(1, n):
            sigma2[t] = omega + alpha * residuals2[t-1] + beta * sigma2[t-1]
        return sigma2
    
    def step_forecast(self, parameters, residuals, sigma2):
        omega, alpha, beta = parameters
        return omega + alpha * residuals**2 + beta * sigma2
    
    def initial_params(self):
        """Initial parameter guess for GARCH model."""
        return GARCHParameters()
    
    def optimization_bounds(self):
        return [(0, +np.inf), (0.000001, 1), (0.000001, 1)]
    
    def get_param_format(self):
        return {'omega': 0, 'alpha': 1, 'beta': 2}
    
class GJRVolatilityModel(VolatilityModel):
    def compute_variance(self, parameters, residuals):
        n = len(residuals)
        omega, alpha, beta, gamma = parameters
        sigma2 = np.zeros(n)
        sigma2[0] = sum(residuals**2) / n

        residuals2 = residuals**2
        negative_residuals2 = (residuals * (residuals < 0))**2

        for t in range(1, n):
            sigma2[t] = omega + alpha * residuals2[t-1] + beta * sigma2[t-1] + gamma * negative_residuals2[t-1]
        return sigma2
    
    def step_forecast(self, parameters, residuals, sigma2):
        omega, alpha, beta, gamma = parameters
        residuals2 = residuals**2
        negative_residuals2 = (residuals * (residuals < 0))**2
        return omega + alpha * residuals2 + beta * sigma2 + gamma * negative_residuals2
    
    def initial_params(self):
        """Initial parameter guess for GJR-GARCH model."""
        return GJRParameters()
    
    def optimization_bounds(self):
        return [(0, +np.inf), (0.000001, 1), (0.000001, 1), (0.000001, 1)]
    
    def get_param_format(self):
        return {'omega': 0, 'alpha': 1, 'beta': 3, 'gamma': 2}
    
class DCCVolatilityModel(VolatilityModel):
    def __init__(self):
        self._dimension = "multivariate"

    def initial_params(self):
        """Initial parameter guess for DCC-GARCH model."""
        return DCCParameters()
    
    def compute_unconditional_q_matrix(self, data):
        T, n = data.shape
        unconditional_correlation = np.zeros((n, n))
        for t in range(T):
            innovation = data[t, :].reshape(n, 1)
            unconditional_correlation += innovation @ innovation.T
        return unconditional_correlation / T
    
    def compute_variance(self, results):
        pass
        
    def compute_correlation(self, parameters, data, sigma2):
        """Calculate conditional variances for each series and time-varying correlations for DCC-GARCH."""
        sigma2 = np.column_stack(sigma2)
        sigma = np.sqrt(sigma2)
        data = np.column_stack(data).T
        delta1, delta2 = parameters
        n, dimension = sigma2.shape # n time steps, dimension assets

        correlation_matrix = np.zeros((n, dimension, dimension))
        q_matrix = np.zeros((n, dimension, dimension))
        unconditional_q_matrix = np.cov(data.T, ddof=0)
        delta_constant = (1 - delta1 - delta2)
    
        scaled_innovations = data / sigma
        innovation_product = np.outer(scaled_innovations[-1], scaled_innovations[-1])
        q_matrix[0] = delta_constant * unconditional_q_matrix + delta1 * innovation_product + delta2 * unconditional_q_matrix

        for t in range(1, n):
            innovation_product = np.outer(scaled_innovations[t-1], scaled_innovations[t-1])
            q_matrix[t] = delta_constant * unconditional_q_matrix + delta1 * innovation_product + delta2 * q_matrix[t-1]
            q_diagonal_inverse_root = np.diag(1 / np.sqrt(np.diag(q_matrix[t])))
            correlation_matrix[t] = q_diagonal_inverse_root @ q_matrix[t] @ q_diagonal_inverse_root
            # covariances_matrix[t] = np.diag(sigma) @ correlation_matrix[t] @ np.diag(sigma)
        
        correlation_matrix[0] = correlation_matrix[-1]
        return correlation_matrix
    
    def step_forecast(self, sigma2_last, correlation_structure_last):
        '''Conditional covariance matrix forecast'''
        sigma2 = np.diag(sigma2_last)
        assert(sigma2.shape[0] == correlation_structure_last.shape[1])
        return sigma2 @ correlation_structure_last @ sigma2

    def optimization_bounds(self):
        return [(1e-6, 1), (1e-6, 1)]
    
    def optimization_constraints(self, distribution_name):
        if distribution_name == 'normal':
            constraints = [LinearConstraint(np.array([1, 1]),lb = [0.0], ub=[1], keep_feasible=True),
                           LinearConstraint(np.eye(2),  lb=[0, 0], ub=[1, 1], keep_feasible=True)]
            return constraints
        elif distribution_name == 'studentt':
            constraints = [
                LinearConstraint(np.array([1, 1, 0]), lb=[0.0], ub=[1], keep_feasible=True),
                LinearConstraint(np.eye(3), lb=[0, 0, 2+1e-6], ub=[1, 1, 100], keep_feasible=True)]
            return constraints
        elif distribution_name == 'skewt':
            print('User Warning: Skew-t implementation is provisionary.')
            constraints = [
                LinearConstraint(np.array([1, 1, 0, 0]), lb=[0.0], ub=[1], keep_feasible=True),
                LinearConstraint(np.eye(4), lb=[0, 0, 2 + 1e-6, -1], ub=[1, 1, 100, 1], keep_feasible=True)
            ]
            return constraints
        else:
            raise ValueError(f"Unsupported distribution: {distribution_name}")
        
    def get_param_format(self):
        return {'delta1': 0, 'delta2': 1}
        
class VolatilityModelFactory:
    @staticmethod
    def get_model(model_type: str) -> VolatilityModel:
        """Return the correct VolatilityModel instance based on model_type."""
        model_type = model_type.lower()
        if model_type == 'garch' or model_type == 'garchvolatilitymodel':
            return GARCHVolatilityModel()
        elif model_type == 'gjr' or model_type == 'gjrvolatilitymodel':
            return GJRVolatilityModel()
        else:
            raise ValueError(f"Unknown model type: {model_type}")

@dataclass
class ModelParameters(ABC):
    omega: float = 0.00001
    alpha: float = 0.05
    beta: float = 0.9

    @abstractmethod
    def stationarity_condition(self):
        """Abstract method for checking the stationarity condition."""
        pass

    @abstractmethod
    def to_list(self):
        """Convert parameters to a list for optimization purposes."""
        pass

@dataclass
class GARCHParameters(ModelParameters):
    def stationarity_condition(self, internal=False):
        if internal:
            return [0, 1, 1]
        else:
            return self.alpha + self.beta
    
    def to_list(self):
        return [self.omega, self.alpha, self.beta]

@dataclass
class GJRParameters(ModelParameters):
    gamma: float = 0.05
    
    def stationarity_condition(self, internal=False):
        if internal:
            return [0, 1, 1, 0.5]
        else:
            return self.alpha + self.beta + 0.5 * self.gamma
    
    def to_list(self):
        return [self.omega, self.alpha, self.beta, self.gamma]

@dataclass
class DCCParameters:
    delta1: float = 0.49
    delta2: float = 0.49

    def stationarity_condition(self):
        return True
    
    def to_list(self):
        return [self.delta1, self.delta2]

class ModelResult:
    def __init__(self, parameters, model_type, distribution, log_likelihood, shape_params=None, parameter_sum=None, correlation_structure=None):
        self.parameters = parameters
        self.model_type = model_type
        self.distribution = distribution
        self.log_likelihood = log_likelihood
        self.shape_params = shape_params if shape_params else {}
        self.parameter_sum = parameter_sum
        self.correlation_structure = correlation_structure

        self.core_param_format = globals()[model_type]().get_param_format()
        self.shape_param_format = globals()[distribution]().get_param_format()

    def _get_core_params(self):
        """Fetch core parameters in the correct order based on format."""
        table = []
        for param_name, position in sorted(self.core_param_format.items(), key=lambda x: x[1]):
            param_value = getattr(self.parameters, param_name, None)
            if param_value is not None:
                table.append([param_name, f"{param_value:.6f}"])
        return table

    def _get_shape_params(self):
        """Fetch shape parameters in the correct order based on format."""
        table = []
        for param_name, position in sorted(self.shape_param_format.items(), key=lambda x: x[1]):
            param_value = self.shape_params.get(param_name, None)
            if param_value is not None:
                table.append([param_name, f"{param_value:.6f}"])
        return table

    def _format_table(self, param_table, title="Parameters"):
        """Helper to format parameter tables using tabulate."""
        return tabulate(param_table, headers=["Parameter", "Estimate"], floatfmt=".6f", tablefmt="simple")

    def get_info(self):
        """Returns the basic model information."""
        return "\n".join([
            f"Model Type: {self.model_type}"[:-15],
            f"Distribution: {self.distribution}"[:-15],
            f"Log-Likelihood: {self.log_likelihood:.2f}"
        ])

    def get_parameters_table(self):
        """Returns the core and shape parameters as a formatted table."""
        core_param_table = self._get_core_params()
        shape_param_table = self._get_shape_params()

        combined_table = core_param_table
        if shape_param_table:
            combined_table.append(["--------------", ""])
            combined_table.extend(shape_param_table)

        combined_table.append(["--------------", ""])
        combined_table.append(["Parameter Sum", f"{self.parameter_sum:.6f}" if self.parameter_sum is not None else ""])

        return tabulate(combined_table, headers=["Parameter", "Estimate"], floatfmt=".6f", tablefmt="simple")

    def __str__(self):
        combined_info = self.get_info()
        parameter_table_str = self.get_parameters_table()

        final_table = [
            ["Information", combined_info],
            ["Parameters", parameter_table_str],
        ]
        return tabulate(final_table, headers=["Description", "Details"], tablefmt="grid")
    
    def get_kwargs(self):
        return {
            'parameters': self.parameters,
            'model_type': self.model_type,
            'distribution': self.distribution,
            'log_likelihood': self.log_likelihood,
            'shape_params': self.shape_params,
            'parameter_sum': self.parameter_sum,
            'correlation_structure': self.correlation_structure
        }

    def get_model(self):
        return VolatilityModelFactory.get_model(self.model_type)
    
    def compute_variance(self, data):
        data = DataHandler(data).get_numpy()
        model_type = self.get_model()
        return model_type.compute_variance(self.parameters.to_list(), data)
    
class CombinedModelResult:
    def __init__(self, results_dict):
        self.results_dict = results_dict

    def get_combined_info(self):
        """Returns the combined info as a string for the top section."""
        combined_info = "\n".join([
            f"Model Type: {list(self.results_dict.values())[0].model_type}"[:-15],
            f"Distribution: {list(self.results_dict.values())[0].distribution}"[:-15],
            f"Total-Likelihood: {sum([result.log_likelihood for result in self.results_dict.values()]):.2f}"
        ])
        return combined_info

    def get_combined_table(self):
        """Returns the combined parameters and likelihood as a table string."""
        sample_result = list(self.results_dict.values())[0]
        core_params = sample_result._get_core_params()
        shape_params = sample_result._get_shape_params()

        if not isinstance(core_params, list):
            core_params = [[core_params, ""]]
        if not isinstance(shape_params, list):
            shape_params = [[shape_params, ""]]

        all_params = core_params
        if shape_params:
            all_params.append(["--------------", ""])
            all_params.extend(shape_params)

        headers = ["Parameter"] + list(self.results_dict.keys())

        table = []
        for param_row in all_params:
            if isinstance(param_row, list) and len(param_row) > 0:
                param = param_row[0]
                if param == '--------------':
                    table.append(['--------------'] + [''] * len(self.results_dict))
                else:
                    row = [param]
                    for result in self.results_dict.values():
                        if param in result.core_param_format:
                            row.append(f"{getattr(result.parameters, param, ''):.6f}")
                        elif param in result.shape_param_format:
                            row.append(f"{result.shape_params.get(param, ''):.6f}")
                        else:
                            row.append('')
                    table.append(row)

        log_likelihood_row = ["Log-Likelihood"]
        parameter_sum_row = ["Parameter Sum"]
        for result in self.results_dict.values():
            log_likelihood_row.append(f"{result.log_likelihood:.6f}")
            parameter_sum_row.append(f"{result.parameter_sum:.6f}")

        table.append(["--------------"])
        table.append(parameter_sum_row)
        table.append(["--------------"])
        table.append(log_likelihood_row)

        return tabulate(table, headers=headers, floatfmt=".6f", tablefmt="simple")

    def __str__(self):
        combined_info = self.get_combined_info()
        parameter_table_str = self.get_combined_table()

        final_table = [
            ["Information", combined_info],
            ["Parameters", parameter_table_str],
        ]
        return tabulate(final_table, headers=["Description", "Estimates"], tablefmt="grid") + "\n\n"
    
    def compute_variance(self, data):
        data = DataHandler(data).get_numpy()
        model = list(self.results_dict.values())[0].get_model()
        parameters = [result.parameters.to_list() for result in self.results_dict.values()]
        return np.array([model.compute_variance(parameters[i], data[:, i]) for i in range(data.shape[1])])
    
class MultivariateModelResult:
    def __init__(self, combined_result: CombinedModelResult, multivariate_result: ModelResult):
        self.combined_result = combined_result
        self.multivariate_result = multivariate_result
        self.correlation_structure = multivariate_result.correlation_structure
        self.single_result = list(combined_result.results_dict.values())[0]
        self.all_likelihoods = np.sum([result.log_likelihood for result in combined_result.results_dict.values()])

    def __str__(self):

        univariate_table_str = self.combined_result.get_combined_table()
        univariate_info = "\n".join([
            f"Model Type: {self.single_result.model_type}"[:-15],
            f"Distribution: {self.single_result.distribution}"[:-15],
            f"Framework: {self.multivariate_result.framework}",
            f"Total Likelihood: {self.all_likelihoods:.2f}"])
        
        multivariate_info = "\n".join([
            f"Model Type: {self.multivariate_result.model_type}"[:-15],
            f"Distribution: {self.multivariate_result.distribution}"[:-15]])
        
        multivar_core_params = self.multivariate_result._get_core_params()
        multivar_shape_params = self.multivariate_result._get_shape_params()

        multivar_table = multivar_core_params
        if multivar_shape_params:
            multivar_table.append(["--------------", ""])
            multivar_table.extend(multivar_shape_params)

        multivar_table.append(["--------------", ""])
        multivar_table.append(["Log-Likelihood", f"{-self.multivariate_result.log_likelihood:.6f}"])
        multivariate_table_str = tabulate(multivar_table, headers=["Parameter", "Estimate"], floatfmt=".6f", tablefmt="simple")

        final_table = [
            ["Univariate Model", univariate_info],
            ["Parameters", univariate_table_str],
            ["Multivariate Model", multivariate_info],
            ["", multivariate_table_str]]
        return tabulate(final_table, headers=["Description", "Details"], tablefmt="grid")
    
class EstimationFramework(ABC):
    @abstractmethod
    def run(self, likelihood_class: LikelihoodFunction, params: List[float], data: np.ndarray):
        """Abstract method to run the estimation framework."""
        pass

class MLE(EstimationFramework):
    def run(self, model_type: VolatilityModel, likelihood_class: LikelihoodFunction, data: np.ndarray):
        if isinstance(likelihood_class, NormalLikelihoodTypeA):
            return FitActions(model_type)._fit_normal(data, likelihood_class)
        elif isinstance(likelihood_class, StudentTLikelihoodTypeA):
            return FitActions(model_type)._fit_non_symmetric(data, likelihood_class)
        elif isinstance(likelihood_class, SkewTLikelihoodTypeA):
            return FitActions(model_type)._fit_non_symmetric(data, likelihood_class)

class QMLE(EstimationFramework):
    # TODO
    def run(self, model_type: VolatilityModel, likelihood_class: LikelihoodFunction, data: np.ndarray):
        raise NotImplementedError("QMLE not implemented yet.")

class CVM:
    def __init__(self, model_type: str, distribution: str):
        self.model_type = VolatilityModelFactory.get_model(model_type)
        self.distribution = distribution
        self.likelihood_class = LikelihoodFactory.get_likelihood(model_type, distribution)

    def fit(self, data, framework='MLE', multivar=None) -> ModelResult:
        """Fit the model using the specified framework and handle multivariate cases."""
        self._data = data
        data_handler = DataHandler(data)
        framework_instance = FrameworkFactory.get_framework(framework)
        
        if multivar and multivar.lower() == 'dcc':
            return self._fit_dcc(data_handler, framework_instance)
        elif data_handler.is_univariate:
            return self._fit_univariate(data_handler, framework_instance) 
        else:
            return self._fit_multiple_univariate(data_handler, framework_instance)
        
    def calc_var(self, result=None, parameters: list=None, data=None):
        if not hasattr(self, '_data') and data is None:
            raise ValueError("No data available for variance calculation. Fit the model, or provide data to method.")
        if data is None:
            data = self._data
        if result is not None and self._check_model_result(result):
            return result.compute_variance(data)
        elif parameters is not None and len(self.model_type.get_param_format) == len(parameters):
            return self.model_type.compute_variance(parameters, data)
        

    def _fit_univariate(self, data_handler, framework):
        fitter = UnivariateFitter(self.model_type, self.likelihood_class, framework)
        return fitter._fit(data_handler)
    
    def _fit_multiple_univariate(self, data_handler, framework):
        n = data_handler.shape[1]
        results = {}
        fitter = UnivariateFitter(self.model_type, self.likelihood_class, framework)
        
        for i in range(n):
            slice_handler = data_handler.slice(i)
            series_name = slice_handler.columns[0] if slice_handler.columns else f"Series {i+1}"
            results[series_name] = fitter._fit(slice_handler)

        return CombinedModelResult(results)

    def _fit_dcc(self, residual_handler, framework):
        if isinstance(framework, QMLE):
            likelihood_store = self.likelihood_class
            self.likelihood_class = NormalLikelihoodTypeA()
            univariate_results = self._fit_multiple_univariate(residual_handler, framework)
            self.likelihood_class = likelihood_store
        else:
            univariate_results = self._fit_multiple_univariate(residual_handler, framework)

        variance_handler = DataHandler(univariate_results.compute_variance(residual_handler.get_numpy()))

        fitter = DCCFitter(self.model_type, self.likelihood_class, framework)
        multivariate_result = fitter.fit(residual_handler, variance_handler)
        multivariate_result.framework = framework.__class__.__name__
        return MultivariateModelResult(univariate_results, multivariate_result)
    
    @staticmethod
    def _check_model_result(result):
        if isinstance(result, (ModelResult, CombinedModelResult, MultivariateModelResult)):
            return True
        else:
            raise ValueError("Invalid result type")
    
class DCCFitter:
    def __init__(self, model_type: VolatilityModel, likelihood_class: LikelihoodFunction, framework: EstimationFramework):
        self.model_type = DCCVolatilityModel()
        self.likelihood_class = likelihood_class
        self.framework = framework

    def fit(self, residual_handler, variance_handler):
        self.dcc_likelihood = self._get_multivariate_likelihood(self.likelihood_class)
        result, correlation_structure = self._fit_multivariate(residual_handler, variance_handler)
        return self._prepare_model_result(result, correlation_structure)
    
    def _fit_multivariate(self, residual_handler, variance_handler):
        distribution_name = self._get_likelihood_name(self.likelihood_class)
        variances = variance_handler.get_numpy() # N x T
        residuals = residual_handler.get_numpy()
        initial_parameters = self.model_type.initial_params().to_list()
        optimization_bounds = self.model_type.optimization_bounds()
        optimization_constraints = self.model_type.optimization_constraints(distribution_name)
        likelihood_function = self.dcc_likelihood.log_likelihood
        normalized_residuals = residuals / np.sqrt(variances).T

        distribution_bounds, shape_parameters = self.dcc_likelihood.get_bounds_and_shape_parameters()
        initial_parameters.extend(shape_parameters)
        optimization_bounds.extend(distribution_bounds)
        number_of_shape_parameters = len(shape_parameters)

        if isinstance(self.dcc_likelihood, MultivariateSkewTLikelihoodTypeA):
            D_inv = self._create_inv_D(variances) # T x N x N
            def log_likelihood(correlation_parameters):
                core_parameters = correlation_parameters[:-number_of_shape_parameters]
                shape_parameters = correlation_parameters[-number_of_shape_parameters:]
                nu = shape_parameters[0]
                delta = np.array(shape_parameters[1:])
                correlation = self.model_type.compute_correlation(core_parameters, residuals, variances)
                return likelihood_function(correlation, normalized_residuals, nu, delta, D_inv)

        elif number_of_shape_parameters > 0 and not isinstance(self.dcc_likelihood, MultivariateSkewTLikelihoodTypeA):
            def log_likelihood(correlation_parameters):
                core_parameters = correlation_parameters[:-number_of_shape_parameters]
                shape_parameters = correlation_parameters[-number_of_shape_parameters:]
                correlation = self.model_type.compute_correlation(core_parameters, residuals, variances)
                return likelihood_function(correlation, normalized_residuals, *shape_parameters)
        else:
            def log_likelihood(correlation_parameters):
                correlation = self.model_type.compute_correlation(correlation_parameters, residuals, variances)
                return likelihood_function(correlation, normalized_residuals)
            
        warnings.filterwarnings("ignore", message="delta_grad == 0.0. Check if the approximated function is linear.")
        result = minimize(log_likelihood, initial_parameters, method='trust-constr', bounds=optimization_bounds, constraints=[*optimization_constraints])
        # result = minimize(log_likelihood, initial_parameters, bounds=optimization_bounds, constraints=[*optimization_constraints])

        if number_of_shape_parameters > 0:
            result.shape_params = result.x[-number_of_shape_parameters:]
            return result, self.model_type.compute_correlation(result.x[:-number_of_shape_parameters], residuals, variances)
        else:
            return result, self.model_type.compute_correlation(result.x, residuals, variances)
    
    def _prepare_model_result(self, result, correlation_structure):
        result_kwargs = UnivariateFitter(self.model_type, self.dcc_likelihood, self.framework)._prepare_model_result(result).get_kwargs()
        result_kwargs['correlation_structure'] = correlation_structure
        return ModelResult(**result_kwargs)
    
    @staticmethod
    def _get_multivariate_likelihood(likelihood_class):
        if isinstance(likelihood_class, NormalLikelihoodTypeA):
            return MultivariateNormalLikelihoodTypeA()
        if isinstance(likelihood_class, StudentTLikelihoodTypeA):
            return MultivariateStudentTLikelihoodTypeA()
        if isinstance(likelihood_class, SkewTLikelihoodTypeA):
            return MultivariateSkewTLikelihoodTypeA()
        
    @staticmethod
    def _get_likelihood_name(likelihood_class):
        if isinstance(likelihood_class, NormalLikelihoodTypeA):
            return "normal"
        if isinstance(likelihood_class, StudentTLikelihoodTypeA):
            return "studentt"
        if isinstance(likelihood_class, SkewTLikelihoodTypeA):
            return "skewt"
        
    @staticmethod
    def _create_inv_D(variances):
        """
        Converts variances from a N X T array into a T x N x N diagonal matrix of volatilities.
        Each row of roots of variances (of length N) will form the diagonal of an N x N matrix.
        """
        D_inv = 1 / np.sqrt(variances.T)
        T, N = D_inv.shape
        D_inv_diag = np.zeros((T, N, N))

        for t in range(T):
            np.fill_diagonal(D_inv_diag[t], D_inv[t])
        return D_inv_diag

class UnivariateFitter:
    def __init__(self, model_type: VolatilityModel, likelihood_class: LikelihoodFunction, framework: EstimationFramework):
        self.model_type = model_type
        self.likelihood_class = likelihood_class
        self.framework = framework

    def _fit(self, data_handler):
        result = self.framework.run(self.model_type, self.likelihood_class, data_handler.get_numpy())
        return self._prepare_model_result(result, data_handler)

    def _prepare_model_result(self, result, data_handler=None):
        """Helper method to prepare ModelResult from fitting."""
        parameter_class = type(self.model_type.initial_params())
        shape_parameters, core_parameters = self.likelihood_class.extract_shape_parameters(result.x)
        fitted_parameters = parameter_class(*core_parameters)

        result_kwargs = {
            'parameters': fitted_parameters,
            'model_type': self.model_type.__class__.__name__,
            'distribution': self.likelihood_class.__class__.__name__,
            'log_likelihood': -result.fun,
            'shape_params': shape_parameters,
            'parameter_sum': fitted_parameters.stationarity_condition(),
        }

        return ModelResult(**result_kwargs)
    
class FitActions:
    def __init__(self, model_type):
        self.model_type = model_type

    def _fit_normal(self, data, likelihood_class):
        """Fits core parameters under the assumption of normal density, returns the parameters only."""
        volatility_model = self.model_type
        initial_parameters = volatility_model.initial_params().to_list()
        stationarity_vector = volatility_model.initial_params().stationarity_condition(internal=True)
        optimization_bounds = volatility_model.optimization_bounds()
        likelihood_function = likelihood_class.log_likelihood
        compute_variance = volatility_model.compute_variance

        def log_likelihood(parameters):
            if parameters @ stationarity_vector >= 1:
                return 1e4
            sigma2 = compute_variance(parameters, data)
            return likelihood_function(data, sigma2)
        
        return minimize(log_likelihood, initial_parameters, method='Nelder-Mead', bounds=optimization_bounds)
    
    def _fit_non_symmetric(self, data, likelihood_class):
        """Fits core parameters under the assumption of non-symmetric density."""
        volatility_model = self.model_type
        initial_parameters = volatility_model.initial_params().to_list()
        stationarity_vector = volatility_model.initial_params().stationarity_condition(internal=True)
        optimization_bounds = volatility_model.optimization_bounds()
        likelihood_function = likelihood_class.log_likelihood
        compute_variance = volatility_model.compute_variance

        distribution_bounds, shape_parameters = likelihood_class.get_bounds_and_shape_parameters()
        initial_parameters.extend(shape_parameters)
        optimization_bounds.extend(distribution_bounds)
        number_of_shape_parameters = len(shape_parameters)

        def log_likelihood(volatility_parameters):
            core_parameters = volatility_parameters[:-number_of_shape_parameters]
            if core_parameters @ stationarity_vector >= 1:
                return 1e4
            
            shape_parameters =  volatility_parameters[-number_of_shape_parameters:]
            sigma2 = compute_variance(core_parameters, data)
            return likelihood_function(data, sigma2, *shape_parameters)

        return minimize(log_likelihood, initial_parameters, method='Nelder-Mead', bounds=optimization_bounds)
    
class FrameworkFactory:
    @staticmethod
    def get_framework(framework_name: str, **kwargs) -> EstimationFramework:
        """Return the appropriate estimation framework instance."""
        framework_name = framework_name.lower()

        if framework_name == 'mle':
            return MLE(**kwargs)
        elif framework_name == 'qmle':
            return QMLE(**kwargs)
        else:
            raise ValueError(f"Unknown framework: {framework_name}")