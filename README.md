# volkit

A high-performance Python library for volatility modeling with GARCH models.

## Features

- **Component-based model specification** - Build models by combining volatility, mean, and density components
- **Multiple distributions** - Normal, Student-t, and Skewed Student-t
- **Fast C extensions** - Optimized likelihood, gradient, and Hessian computation
- **Robust standard errors** - QMLE with sandwich covariance estimation
- **Professional output** - Clean, tabular estimation summaries

## Installation

```bash
# Development install (requires C compiler)
pip install -e .

# Or use the Makefile
make dev
```

## Quick Start

```python
import numpy as np
from volkit import GARCH, Normal, StudentT, MLE

# Generate sample data
np.random.seed(42)
returns = np.random.randn(1000) * 0.01

# Specify and fit a GARCH(1,1) model with Normal distribution
spec = GARCH(1, 1) + Normal()
result = spec.fit(returns)

# View results
result.summary()
```

Output:
```
══════════════════════════════════════════════════════════════════════
                    GARCH Model Estimation Results                    
══════════════════════════════════════════════════════════════════════
Model:       GARCH(1,1)+Normal
Method:      MLE
Date:        2026-01-30 15:42:38
──────────────────────────────────────────────────────────────────────
No. Observations:    1000            Converged:      Yes
No. Parameters:      3               Iterations:     42
Time Elapsed:        0.030s
──────────────────────────────────────────────────────────────────────
Log-Likelihood:            3456.7890
AIC:                      -6907.5780
BIC:                      -6892.8560
──────────────────────────────────────────────────────────────────────

                         Parameter Estimates                          
──────────────────────────────────────────────────────────────────────
Parameter            Coef      Std Err     t-stat      P>|t|
──────────────────────────────────────────────────────────────────────
omega          1.2345e-06    2.34e-07      5.27    <0.001
alpha[1]           0.0523      0.0084      6.23    <0.001
beta[1]            0.9412      0.0092    102.30    <0.001
──────────────────────────────────────────────────────────────────────

                          Model Diagnostics                           
──────────────────────────────────────────────────────────────────────
Persistence (α + β):      0.993500
Stationary:               Yes
Unconditional Variance:   1.900000e-04
Half-life (periods):      106.3
══════════════════════════════════════════════════════════════════════
```

---

## Table of Contents

1. [Model Specification](#model-specification)
2. [Components](#components)
3. [Estimators](#estimators)
4. [Results](#results)
5. [Diagnostic Tools](#diagnostic-tools)
6. [API Reference](#api-reference)

---

## Model Specification

volkit uses a **component composition** pattern. Models are built by combining components with the `+` operator.

### Basic Pattern

```python
from volkit import GARCH, ARMA, Normal, StudentT, SkewT

# Volatility only (density defaults to Normal)
spec = GARCH(1, 1)

# Volatility + explicit density
spec = GARCH(1, 1) + StudentT()

# Mean + Volatility + Density
spec = ARMA(1, 1) + GARCH(1, 1) + SkewT()
```

### Composition Rules

- **Canonical ordering**: Components are automatically ordered as MEAN → VOLATILITY → DENSITY
- **Auto-injection**: If no density is specified, `Normal()` is added automatically
- **Single role**: Only one component per role is allowed

### Alternative Syntax

Multiple syntaxes produce identical results:

```python
spec = garch + normal      # __add__
spec = garch < normal      # __lt__
spec = garch << normal     # __lshift__
spec = normal >> garch     # __rlshift__
```

---

## Components

### Volatility Components

#### GARCH(p, q)

Generalized Autoregressive Conditional Heteroskedasticity model.

```python
from volkit import GARCH

# GARCH(1,1) - most common
spec = GARCH(1, 1)

# GARCH(2,1)
spec = GARCH(2, 1)
```

**Parameters:**
- `omega` (ω): Constant term (must be > 0)
- `alpha[1:p]` (α): ARCH coefficients (reaction to shocks)
- `beta[1:q]` (β): GARCH coefficients (persistence)

**Model equation:**
```
σ²_t = ω + Σᵢ αᵢ·ε²_{t-i} + Σⱼ βⱼ·σ²_{t-j}
```

**Stationarity condition:** Σα + Σβ < 1

### Mean Components

#### ARMA(p, q)

Autoregressive Moving Average model for the conditional mean.

```python
from volkit import ARMA

spec = ARMA(1, 1) + GARCH(1, 1)
```

**Note:** ARMA support is currently limited.

### Density Components

#### Normal()

Standard Gaussian distribution (no additional parameters).

```python
from volkit import Normal

spec = GARCH(1, 1) + Normal()
# Or simply:
spec = GARCH(1, 1)  # Normal is default
```

#### StudentT()

Student-t distribution with heavier tails.

```python
from volkit import StudentT

spec = GARCH(1, 1) + StudentT()
```

**Parameters:**
- `nu` (ν): Degrees of freedom (must be > 2)
  - Lower ν → heavier tails
  - As ν → ∞, approaches Normal

#### SkewT()

Hansen's Skewed Student-t distribution.

```python
from volkit import SkewT

spec = GARCH(1, 1) + SkewT()
```

**Parameters:**
- `nu` (ν): Degrees of freedom (must be > 2)
- `lambda` (λ): Skewness parameter (-1 < λ < 1)
  - λ = 0: Symmetric (same as StudentT)
  - λ < 0: Left skew (negative returns more likely)
  - λ > 0: Right skew (positive returns more likely)

---

## Estimators

### MLE (Maximum Likelihood Estimation)

The default and recommended estimator.

```python
from volkit import GARCH, Normal, MLE

spec = GARCH(1, 1) + Normal()

# Method 1: Direct MLE estimator
estimator = MLE()
result = estimator.fit(spec, data)

# Method 2: Shortcut via spec.fit()
result = spec.fit(data)

# With options
result = spec.fit(
    data,
    solver="trust",       # Optimization method
    log_mode=True,        # Optimize in log-space (recommended)
    verbose=True,         # Print progress
)
```

**Solver options:**
| Solver | Description | Speed | Robustness |
|--------|-------------|-------|------------|
| `"nelder-mead"` | Derivative-free simplex | Slow | High |
| `"slsqp"` | Sequential quadratic programming | Medium | Medium |
| `"trust"` | Trust-region with gradient + Hessian | Fast | High |
| `"trust-exact"` | Trust-region in log-space | Fast | Highest |

**Unconstrained Optimization (`log_mode`):**

The `log_mode=True` option transforms the constrained GARCH optimization problem into an unconstrained one via parameter transformations:

| Parameter | Constraint | Transformation |
|-----------|------------|----------------|
| ω (omega) | ω > 0 | ω = exp(z_ω) |
| α (alpha) | α > 0, Σα < 1 | softmax transform |
| β (beta) | β > 0, Σβ < 1 | softmax transform |
| ν (nu, Student-t) | ν > 2 | ν = 2 + exp(z_ν) |
| λ (lambda, SkewT) | -1 < λ < 1 | λ = tanh(z_λ) |

**Benefits:**
- Eliminates boundary issues during optimization
- Guarantees stationarity (α + β < 1) by construction
- More robust convergence, especially for gradient-based solvers
- All likelihood evaluations use valid parameters

**Usage:**
```python
# Recommended: use log_mode for robust optimization
result = spec.fit(data, solver="trust", log_mode=True)

# For all distributions
spec = GARCH(1, 1) + Normal()
spec.fit(data, log_mode=True)  # ✓

spec = GARCH(1, 1) + StudentT()
spec.fit(data, log_mode=True)  # ✓

spec = GARCH(1, 1) + SkewT()
spec.fit(data, log_mode=True)  # ✓
```

### QMLE (Quasi-Maximum Likelihood Estimation)

Uses Normal likelihood but computes **robust (sandwich) standard errors** that are valid even when the true distribution is non-Normal.

```python
from volkit import GARCH, Normal, QMLE

spec = GARCH(1, 1) + Normal()
estimator = QMLE()
result = estimator.fit(spec, data)

# Access both types of standard errors
print(result.std_errors)        # MLE standard errors
print(result.std_errors_robust) # Robust (sandwich) standard errors
```

**When to use QMLE:**
- When you suspect the true distribution is not Normal
- For inference that is robust to distribution misspecification
- When comparing MLE and robust standard errors for diagnostic purposes

---

## Results

The `EstimationResult` object provides access to all estimation outputs.

### Parameter Access

```python
result = spec.fit(data)

# All parameters as flat array
params = result.params  # [omega, alpha_1, ..., alpha_p, beta_1, ..., beta_q, nu?, lambda?]

# Structured GARCH parameters
gp = result.garch_params
print(gp.omega)       # Constant term
print(gp.alpha)       # ARCH coefficients (array)
print(gp.beta)        # GARCH coefficients (array)
print(gp.persistence) # Sum of alpha + beta

# Distribution parameters
dp = result.dist_params
print(dp.nu)          # Degrees of freedom (if StudentT/SkewT)
print(dp.lam)         # Skewness (if SkewT)
```

### Model Fit Statistics

```python
result.loglikelihood  # Log-likelihood value
result.aic            # Akaike Information Criterion
result.bic            # Bayesian Information Criterion  
result.hqic           # Hannan-Quinn Information Criterion
```

### Conditional Variances and Residuals

```python
result.sigma2         # Conditional variances (σ²_t)
result.volatility     # Conditional volatility (σ_t = sqrt(σ²_t))
result.std_resid      # Standardized residuals (ε_t / σ_t)
```

### Standard Errors

```python
# MLE standard errors
result.std_errors     # From inverse Hessian

# Robust standard errors (QMLE only)
result.std_errors_robust  # From sandwich estimator

# Covariance matrices
result.cov_matrix     # MLE covariance (H⁻¹)
result.cov_robust     # Robust covariance (H⁻¹ @ OPG @ H⁻¹)
```

### Summary Output

```python
# Full summary with parameter table
result.summary()

# With robust standard errors (if available)
result.summary(robust=True)

# Compact string representation
print(result)

# For programmatic access
result.to_dict()
```

### Component Access

```python
# Access individual components
result.vol            # Volatility component (GARCH)
result.mean           # Mean component (ARMA)
result.density        # Density component (Normal/StudentT/SkewT)
```

---

## Diagnostic Tools

### Derivative Validation

Validate that analytical gradient and Hessian match numerical finite differences.

```python
from volkit import GARCH, Normal

spec = GARCH(1, 1) + Normal()

# Method 1: Via spec
report = spec.validate_derivatives(data)
report.summary()

# Method 2: Direct function
from volkit._devtools import validate_derivatives, quick_check

report = validate_derivatives(spec, data, params=None)
report.summary()

# Quick pass/fail check
passed = quick_check(spec, data)
```

**Report contents:**
```python
report.gradient_passed      # bool
report.hessian_passed       # bool
report.passed               # Overall bool

report.gradient_analytical  # Analytical gradient
report.gradient_numerical   # Finite difference gradient
report.gradient_max_rel_error  # Max relative error (%)

report.hessian_analytical   # Analytical Hessian
report.hessian_numerical    # Finite difference Hessian
report.hessian_max_rel_error   # Max relative error (%)
```

---

## API Reference

### Top-level exports

```python
from volkit import (
    # Components
    GARCH,        # GARCH(p, q) volatility model
    ARMA,         # ARMA(p, q) mean model
    Normal,       # Normal distribution
    StudentT,     # Student-t distribution
    SkewT,        # Skewed Student-t distribution
    Component,    # Base class for components
    
    # Specification
    CompositeSpec,  # Model specification container
    Role,           # Enum: MEAN, VOLATILITY, DENSITY
    
    # Estimators
    MLE,          # Maximum likelihood estimator
    QMLE,         # Quasi-MLE with robust SEs
    
    # Version
    __version__,
)
```

### GARCH Component

```python
garch = GARCH(p: int, q: int)

# Properties
garch.p             # ARCH order
garch.q             # GARCH order
garch.n_params      # Total parameters (1 + p + q)
garch.signature     # "GARCH(p,q)"

# After fitting
garch.fitted_params # {'omega': float, 'alpha': list, 'beta': list}
garch.persistence() # Sum of alpha + beta
garch.is_stationary()
garch.unconditional_variance()
```

### EstimationResult

```python
result = spec.fit(data)

# Parameters
result.params               # NDArray - all parameters
result.garch_params         # GARCHParams object
result.dist_params          # DistributionParams object

# Fit statistics
result.loglikelihood        # float
result.aic                  # float
result.bic                  # float
result.hqic                 # float

# Time series
result.sigma2               # NDArray - conditional variances
result.volatility           # NDArray - sqrt(sigma2)
result.std_resid            # NDArray - standardized residuals

# Standard errors
result.std_errors           # NDArray - MLE standard errors
result.std_errors_robust    # NDArray - Robust standard errors (QMLE)
result.cov_matrix           # NDArray - MLE covariance
result.cov_robust           # NDArray - Robust covariance

# Optimization info
result.success              # bool - convergence status
result.niter                # int - iterations
result.time_elapsed         # float - seconds

# Methods
result.summary()            # Print detailed summary
result.summary(robust=True) # Use robust SEs in summary
result.to_dict()            # Export as dictionary
```

### MLE Estimator

```python
from volkit import MLE

estimator = MLE()
result = estimator.fit(
    spec,                   # Model specification
    data,                   # 1D array of returns/residuals
    solver="trust",         # Optimization method
    log_mode=True,          # Optimize in log-space
    verbose=False,          # Print progress
)
```

### QMLE Estimator

```python
from volkit import QMLE

estimator = QMLE()
result = estimator.fit(
    spec,                   # Model specification
    data,                   # 1D array of returns/residuals
    solver="trust",         # Optimization method
    verbose=False,          # Print progress
)

# Robust SEs are automatically computed
result.std_errors_robust
```

---

## Examples

### Basic GARCH(1,1) Estimation

```python
import numpy as np
from volkit import GARCH, Normal

# Load or generate data
returns = np.random.randn(1000) * 0.01

# Fit model
spec = GARCH(1, 1) + Normal()
result = spec.fit(returns)

# Check results
print(f"Persistence: {result.garch_params.persistence:.4f}")
print(f"Log-likelihood: {result.loglikelihood:.2f}")
result.summary()
```

### Student-t Distribution

```python
from volkit import GARCH, StudentT

spec = GARCH(1, 1) + StudentT()
result = spec.fit(returns, log_mode=True)  # log_mode supported for all distributions

# Access degrees of freedom
nu = result.dist_params.nu
print(f"Degrees of freedom: {nu:.2f}")
```

### Robust Standard Errors

```python
from volkit import GARCH, Normal, QMLE

spec = GARCH(1, 1) + Normal()
result = QMLE().fit(spec, returns)

# Compare MLE vs robust SEs
print("Parameter       MLE SE    Robust SE")
print("-" * 40)
for i, name in enumerate(['omega', 'alpha', 'beta']):
    se_mle = result.std_errors[i]
    se_robust = result.std_errors_robust[i]
    print(f"{name:10}  {se_mle:10.6f}  {se_robust:10.6f}")
```

### Model Comparison

```python
from volkit import GARCH, Normal, StudentT, SkewT

# All distributions support log_mode=True
models = [
    GARCH(1, 1) + Normal(),
    GARCH(1, 1) + StudentT(),
    GARCH(1, 1) + SkewT(),
]

results = [spec.fit(returns, log_mode=True) for spec in models]

print("Model                  LL         AIC        BIC")
print("-" * 55)
for r in results:
    print(f"{str(r.spec):20}  {r.loglikelihood:10.2f}  {r.aic:10.2f}  {r.bic:10.2f}")
```

### Volatility Forecasting

```python
from volkit import GARCH, Normal
import numpy as np

spec = GARCH(1, 1) + Normal()
result = spec.fit(returns)

# Get fitted volatility
sigma = result.volatility

# One-step-ahead forecast (at last observation)
omega = result.garch_params.omega
alpha = result.garch_params.alpha[0]
beta = result.garch_params.beta[0]

last_resid2 = returns[-1]**2
last_sigma2 = result.sigma2[-1]

forecast_sigma2 = omega + alpha * last_resid2 + beta * last_sigma2
forecast_sigma = np.sqrt(forecast_sigma2)

print(f"Next period volatility forecast: {forecast_sigma:.4f}")
```

---

## Development

### Building from source

```bash
# Clone repository
git clone <repo-url>
cd volkit_cursor

# Development install
make dev

# Run tests
make t

# Full rebuild
make b
```

### Derivative validation

Always validate derivatives when modifying C code:

```python
from volkit import GARCH, Normal

spec = GARCH(1, 1) + Normal()
data = np.random.randn(500) * 0.01

# Quick check
report = spec.validate_derivatives(data)
print(f"Derivatives valid: {report.passed}")
```

---

## License

[Add license information]

## Citation

[Add citation information]
