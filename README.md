# volkit

A high-performance Python library for volatility modeling with GARCH-family models.

## Features

- **Component-based model specification** - Build models by combining volatility, mean, and density components
- **Symmetric and asymmetric volatility** - GARCH and GJR-GARCH (leverage effects)
- **Multiple distributions** - Normal, Student-t, and Skewed Student-t
- **Automatic model selection** - Search over GARCH orders and distributions with parallel fitting
- **Fast C extensions** - Optimized likelihood, gradient, and Hessian computation
- **Robust standard errors** - QMLE with sandwich covariance estimation
- **Multi-series support** - Fit the same model to multiple time series in parallel
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
4. [Automatic Model Selection](#automatic-model-selection)
5. [Results](#results)
6. [Display Settings](#display-settings)
7. [Diagnostic Tools](#diagnostic-tools)
8. [API Reference](#api-reference)

---

## Model Specification

volkit uses a **component composition** pattern. Models are built by combining components with the `+` operator.

### Basic Pattern

```python
from volkit import GARCH, GJRGARCH, ARMA, Normal, StudentT, SkewT

# Volatility only (density defaults to Normal)
spec = GARCH(1, 1)

# Volatility + explicit density
spec = GARCH(1, 1) + StudentT()

# Asymmetric volatility (leverage effects)
spec = GJRGARCH(1, 1) + StudentT()

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

#### GJRGARCH(p, q)

GJR-GARCH (Glosten-Jagannathan-Runkle) model with asymmetric volatility response. Negative shocks ("bad news") have a larger impact on future volatility than positive shocks of the same magnitude.

```python
from volkit import GJRGARCH

# GJR-GARCH(1,1) - most common
spec = GJRGARCH(1, 1)

# With Student-t innovations
spec = GJRGARCH(1, 1) + StudentT()
```

**Parameters:**
- `omega` (ω): Constant term (must be > 0)
- `alpha[1:p]` (α): ARCH coefficients (symmetric shock response)
- `gamma[1:p]` (γ): Leverage coefficients (additional response to negative shocks)
- `beta[1:q]` (β): GARCH coefficients (persistence)

**Model equation:**
```
σ²_t = ω + Σᵢ (αᵢ·ε²_{t-i} + γᵢ·I(ε_{t-i}<0)·ε²_{t-i}) + Σⱼ βⱼ·σ²_{t-j}
```

where I(·) is the indicator function (1 if the condition is true, 0 otherwise).

**Stationarity condition:**
- Symmetric distributions (Normal, Student-t): α + 0.5·γ + β < 1
- Asymmetric distributions (Skew-t): α + γ·P(z < 0) + β < 1

**Interpretation:**
- γ > 0 means negative shocks increase volatility more than positive shocks (leverage effect)
- The total impact of a negative shock is (α + γ)·ε², while a positive shock contributes α·ε²

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
| γ (gamma, GJR) | γ > 0 | 4-class softmax (α, γ, β, slack) |
| β (beta) | β > 0, Σβ < 1 | softmax transform |
| ν (nu, Student-t) | ν > 2 | ν = 2 + exp(z_ν) |
| λ (lambda, SkewT) | -1 < λ < 1 | λ = tanh(z_λ) |

**Benefits:**
- Eliminates boundary issues during optimization
- Guarantees stationarity (α + β < 1 or α + 0.5γ + β < 1) by construction
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

## Automatic Model Selection

volkit can automatically search over multiple model specifications and select the best one based on a blended criterion of AIC and diagnostic tests.

### Auto-selection for GARCH Orders

Search over different lag orders using `auto=True`. Works for both `GARCH` and `GJRGARCH`:

```python
from volkit import GARCH, GJRGARCH, Normal

# Search p, q in range [1, 3]
spec = GARCH(auto=True) + Normal()
result = spec.fit(returns)

# Same for GJR-GARCH
spec = GJRGARCH(auto=True) + Normal()
result = spec.fit(returns)

# Inspect what was selected
print(f"Selected model: {result.spec}")  # e.g., GARCH(2,1)+Normal
print(result.selection_summary())  # Show all candidates and scores
```

**Customizing the search range:**

```python
# Search p in [1, 2], q in [1, 2]
spec = GARCH(auto={'max_p': 2, 'max_q': 2}) + Normal()

# Fix p=1, auto-select q from [1, 3]
spec = GJRGARCH(p=1, q='auto') + StudentT()
```

### Auto-selection for Distributions

Use `AutoDensity` to automatically select the best distribution:

```python
from volkit import GARCH, AutoDensity

# Search over all distributions (Normal, StudentT, SkewT)
spec = GARCH(1, 1) + AutoDensity()
result = spec.fit(returns)

# Or specify candidates
spec = GARCH(1, 1) + AutoDensity(candidates=['Normal', 'StudentT'])
result = spec.fit(returns)
```

### Combined Auto-selection

Search over both GARCH orders and distributions simultaneously:

```python
from volkit import GARCH, AutoDensity

# Search GARCH(p,q) where p,q in [1,3] × 3 distributions = 27 models
spec = GARCH(auto=True) + AutoDensity()
result = spec.fit(returns, verbose=True)

print(f"Best model: {result.spec}")
result.selection_summary()
```

### Selection Criterion

The selection score is computed as:

```
Score = AIC + diagnostic_weight × n_failed_tests
```

Where `n_failed_tests` includes:
- 1 if DGT (Density Goodness-of-fit Test) fails
- 1 for each Ljung-Box test (on moments) that fails

By default, `diagnostic_weight = 50.0`, meaning a failed diagnostic test is equivalent to an AIC penalty of 50.

**Customize the selection criterion:**

```python
# Increase diagnostic penalty (favors models with better diagnostics)
result = spec.fit(returns, diagnostic_weight=100.0)

# Disable diagnostic penalties (use AIC only)
result = spec.fit(returns, diagnostic_weight=0.0)
```

### Parallel Model Selection

Auto-selection can fit multiple candidates in parallel for faster execution:

```python
# Use all CPU cores (default)
result = spec.fit(returns)

# Specify number of workers
result = spec.fit(returns, n_jobs=4)

# Sequential execution (useful for debugging)
result = spec.fit(returns, n_jobs=1)
```

### Multi-series Fitting

Fit the same model specification to multiple time series at once:

```python
from volkit import GARCH, Normal

# List of return series
returns_list = [returns1, returns2, returns3, returns4]

spec = GARCH(1, 1) + Normal()

# Fit all series in parallel
results = spec.fit_multiple(returns_list, n_jobs=4)

# Each result is independent
for i, result in enumerate(results):
    print(f"Series {i}: persistence = {result.garch_params.persistence:.4f}")
```

**Multi-series with auto-selection:**

```python
# Auto-select best model for each series independently
spec = GARCH(auto=True) + AutoDensity()
results = spec.fit_multiple(returns_list, n_jobs=4, verbose=True)
```

### Inspecting Selection Results

Access the full set of candidates and their scores:

```python
# After fitting with auto-selection
result = spec.fit(returns)

# Summary table
result.selection_summary()

# Programmatic access to candidates
candidates = result._selection_candidates  # List of ModelCandidate objects

for candidate in candidates[:5]:  # Top 5
    print(f"{candidate.spec}: AIC={candidate.aic:.2f}, Score={candidate.score:.2f}")
```

**Note:** QMLE with AutoDensity is redundant since QMLE always uses Normal likelihood. A warning will be issued and only Normal will be fitted.

---

## Results

The `EstimationResult` object provides access to all estimation outputs.

### Parameter Access

```python
result = spec.fit(data)

# All parameters as flat array
# GARCH:     [omega, alpha_1, ..., alpha_p, beta_1, ..., beta_q, nu?, lambda?]
# GJR-GARCH: [omega, alpha_1, ..., alpha_p, gamma_1, ..., gamma_p, beta_1, ..., beta_q, nu?, lambda?]
params = result.params

# Structured GARCH parameters
gp = result.garch_params
print(gp.omega)       # Constant term
print(gp.alpha)       # ARCH coefficients (array)
print(gp.gamma)       # Leverage coefficients (array, GJR-GARCH only)
print(gp.beta)        # GARCH coefficients (array)
print(gp.persistence) # α + β (GARCH) or α + 0.5γ + β (GJR-GARCH)

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

## Display Settings

volkit provides a global `settings` object that lets you customize how parameter names appear in summaries, printed output, and `to_dict()` keys -- without changing any internal variable names or code logic.

### Renaming Parameters

Set a display name once and it applies everywhere -- including indexed variants for higher-order models (e.g., GARCH(2,1) has `alpha[1]` and `alpha[2]`):

```python
import volkit

# Rename parameters globally
volkit.settings.names.gamma = "leverage"
volkit.settings.names.nu = "df"
volkit.settings.names.alpha = "a"

# All subsequent output uses the new names
spec = GJRGARCH(1, 1) + StudentT()
result = spec.fit(returns)
result.summary()
# Parameter table now shows: omega, a[1], leverage[1], beta[1], df

# For higher-order models like GARCH(2,1), the same override covers all indices:
# a[1], a[2] instead of alpha[1], alpha[2]
```

### Effect on `to_dict()`

Display names also apply to the dictionary keys returned by `result.to_dict()`:

```python
volkit.settings.names.alpha = "a"
volkit.settings.names.beta = "b"

d = result.to_dict()
print(d['garch_params'].keys())
# dict_keys(['w', 'a', 'b', 'persistence'])
```

### Resetting to Defaults

```python
# Clear all overrides
volkit.settings.names.reset()
```

### Available Parameter Names

The following canonical names can be overridden:

| Canonical Name | Used By | Default Display |
|---------------|---------|-----------------|
| `omega` | GARCH, GJR-GARCH | `omega` |
| `alpha` | GARCH, GJR-GARCH | `alpha` |
| `gamma` | GJR-GARCH | `gamma` |
| `beta` | GARCH, GJR-GARCH | `beta` |
| `nu` | StudentT, SkewT | `nu` |
| `lambda` | SkewT | `lambda` |
| `const` | ARMA | `const` |
| `ar` | ARMA | `ar` |
| `ma` | ARMA | `ma` |

**Note:** Only display output is affected. Internal attribute names on dataclasses (`gp.omega`, `gp.alpha`, `dp.nu`, etc.), C function signatures, and component `fitted_params` dictionary keys remain unchanged.

---

## Diagnostic Tools

### Model Diagnostic Tests

After fitting a model, run the DGT (Density Goodness-of-Fit) and Ljung-Box tests to check whether the fitted distribution adequately captures the data:

```python
from volkit import GARCH, StudentT

spec = GARCH(1, 1) + StudentT()
result = spec.fit(returns)

# Run diagnostics and print formatted results
result.diagnostic_tests()
```

Output:
```
======================================================================
                     Model Diagnostic Tests
======================================================================
Distribution:  StudentT (nu=7.42)
Observations:  1000
Alpha:         0.05

DGT Test (Density Goodness-of-Fit)
----------------------------------------------------------------------
  Cells:       40         df:          39
  Chi2 stat:   34.20      p-value:     0.6891
  Reject H0:   No (uniform PIT)

Ljung-Box Tests on PIT Moments
----------------------------------------------------------------------
  Moment       Lags       Q-stat      p-value     Reject
  ---------- ------ ------------ ------------ ----------
  (u-0.5)^1     10         8.42       0.5880         No
  (u-0.5)^2     10         7.91       0.6371         No
  (u-0.5)^3     10        11.23       0.3396         No
  (u-0.5)^4     10         9.87       0.4518         No
======================================================================
```

**How to interpret:**
- **DGT test**: Checks if the PIT (Probability Integral Transform) residuals are uniform. A "No" rejection means the fitted distribution is adequate.
- **Ljung-Box tests**: Check for serial correlation in PIT moments. Moment 1 targets mean misspecification, moment 2 targets variance, moments 3-4 target skewness and kurtosis.

**Customizing the tests:**

```python
# Adjust significance level, number of cells, and lags
result.diagnostic_tests(alpha=0.01, n_cells=50, lags=20)

# Suppress printed output and use the returned dict programmatically
diag = result.diagnostic_tests(print_results=False)

print(diag['dgt']['p_value'])          # DGT p-value
print(diag['dgt']['reject'])           # True/False

for power in (1, 2, 3, 4):
    lb = diag['ljung_box'][power]
    print(f"Moment {power}: Q={lb['q_stat']:.2f}, p={lb['p_value']:.4f}")

# PIT values for custom analysis (e.g., histogram)
pit_values = diag['pit']
```

**Note:** These tests are also used internally by the [auto-selection](#automatic-model-selection) machinery to penalize models with poor diagnostics.

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
    GJRGARCH,     # GJR-GARCH(p, q) asymmetric volatility model
    ARMA,         # ARMA(p, q) mean model
    Normal,       # Normal distribution
    StudentT,     # Student-t distribution
    SkewT,        # Skewed Student-t distribution
    AutoDensity,  # Automatic distribution selection
    Component,    # Base class for components
    
    # Specification
    CompositeSpec,  # Model specification container
    Role,           # Enum: MEAN, VOLATILITY, DENSITY
    
    # Estimators
    MLE,          # Maximum likelihood estimator
    QMLE,         # Quasi-MLE with robust SEs
    
    # Settings
    settings,     # Global display settings (parameter names, etc.)
    
    # Version
    __version__,
)
```

### GARCH Component

```python
# Fixed orders
garch = GARCH(p: int, q: int)

# Auto-selection
garch = GARCH(auto=True)  # Search p,q in [1,3]
garch = GARCH(auto={'max_p': 2, 'max_q': 2})  # Custom range
garch = GARCH(p=1, q='auto')  # Fix p, auto-select q

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

### GJRGARCH Component

```python
from volkit import GJRGARCH

# Fixed orders
gjr = GJRGARCH(p: int, q: int)

# Auto-selection (same interface as GARCH)
gjr = GJRGARCH(auto=True)  # Search p,q in [1,3]
gjr = GJRGARCH(auto={'max_p': 2, 'max_q': 2})  # Custom range
gjr = GJRGARCH(p=1, q='auto')  # Fix p, auto-select q

# Properties
gjr.p             # ARCH/leverage order
gjr.q             # GARCH order
gjr.n_params      # Total parameters (1 + 2p + q)
gjr.signature     # "GJR-GARCH(p,q)"

# After fitting
gjr.fitted_params  # {'omega': float, 'alpha': list, 'gamma': list, 'beta': list}
gjr.persistence()  # α + 0.5·γ + β (default, symmetric)
gjr.persistence(p_neg=0.6)  # α + 0.6·γ + β (asymmetric)
gjr.is_stationary()
gjr.unconditional_variance()
```

### AutoDensity Component

```python
from volkit import AutoDensity

# All distributions
auto_dens = AutoDensity()

# Specific candidates
auto_dens = AutoDensity(candidates=['Normal', 'StudentT'])

# Properties
auto_dens.candidates     # List of distribution names to search
auto_dens.signature      # "AutoDensity"
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
result.diagnostic_tests()   # Run DGT + Ljung-Box tests
result.selection_summary()  # Show auto-selection candidates (if used)

# Auto-selection attributes (if auto=True or AutoDensity used)
result._selection_candidates  # List[ModelCandidate] - all evaluated models
```

### MLE Estimator

```python
from volkit import MLE

estimator = MLE()
result = estimator.fit(
    spec,                      # Model specification
    data,                      # 1D array of returns/residuals
    solver="trust",            # Optimization method
    log_mode=True,             # Optimize in log-space
    verbose=False,             # Print progress
    n_jobs=None,               # Parallel workers (for auto-selection)
    diagnostic_weight=50.0,    # AIC penalty per failed test (auto-selection)
)

# Multi-series fitting
results = spec.fit_multiple(
    data_list,                 # List of 1D arrays
    n_jobs=4,                  # Parallel workers
    **fit_kwargs               # Other fit() arguments
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

### Settings

```python
import volkit

# Override display names (applies to all indexed variants automatically)
volkit.settings.names.gamma = "leverage"  # gamma[1] → leverage[1], gamma[2] → leverage[2], ...
volkit.settings.names.nu = "df"

# Read back the current display name
volkit.settings.names.gamma               # "leverage"

# Reset all overrides to defaults
volkit.settings.names.reset()
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
from volkit import GARCH, GJRGARCH, Normal, StudentT, SkewT

# Compare symmetric and asymmetric volatility models
models = [
    GARCH(1, 1) + Normal(),
    GARCH(1, 1) + StudentT(),
    GARCH(1, 1) + SkewT(),
    GJRGARCH(1, 1) + Normal(),
    GJRGARCH(1, 1) + StudentT(),
    GJRGARCH(1, 1) + SkewT(),
]

results = [spec.fit(returns) for spec in models]

print("Model                        LL         AIC        BIC")
print("-" * 60)
for r in results:
    print(f"{str(r.spec):25}  {r.loglikelihood:10.2f}  {r.aic:10.2f}  {r.bic:10.2f}")
```

### Automatic Model Selection

```python
from volkit import GARCH, AutoDensity

# Automatically select best GARCH order and distribution
spec = GARCH(auto=True) + AutoDensity()
result = spec.fit(returns, verbose=True)

# View what was selected
print(f"Best model: {result.spec}")
result.selection_summary()

# Access all candidates
for candidate in result._selection_candidates[:3]:
    print(f"{candidate.spec}: AIC={candidate.aic:.2f}, Score={candidate.score:.2f}")
```

### Multi-series Fitting

```python
from volkit import GARCH, Normal

# Multiple return series
returns_list = [returns1, returns2, returns3]

# Fit all in parallel
spec = GARCH(1, 1) + Normal()
results = spec.fit_multiple(returns_list, n_jobs=4)

# Compare results
for i, result in enumerate(results):
    print(f"Series {i}: ω={result.params[0]:.2e}, α={result.params[1]:.3f}, β={result.params[2]:.3f}")
```

### GJR-GARCH: Asymmetric Volatility

```python
from volkit import GJRGARCH, StudentT

# Fit GJR-GARCH(1,1) with Student-t innovations
spec = GJRGARCH(1, 1) + StudentT()
result = spec.fit(returns)

# Access parameters
gp = result.garch_params
print(f"omega: {gp.omega:.2e}")
print(f"alpha: {gp.alpha[0]:.4f}")
print(f"gamma: {gp.gamma[0]:.4f}")  # Leverage coefficient
print(f"beta:  {gp.beta[0]:.4f}")
print(f"nu:    {result.dist_params.nu:.2f}")

# Check leverage effect
if gp.gamma[0] > 0:
    print("Leverage effect detected: negative shocks increase volatility more")
    neg_impact = gp.alpha[0] + gp.gamma[0]
    pos_impact = gp.alpha[0]
    print(f"Impact ratio (neg/pos): {neg_impact / pos_impact:.2f}x")

result.summary()
```

### Custom Parameter Display Names

```python
import volkit
from volkit import GJRGARCH, StudentT

# Customize how parameters are displayed
volkit.settings.names.gamma = "leverage"
volkit.settings.names.nu = "df"

spec = GJRGARCH(1, 1) + StudentT()
result = spec.fit(returns)

# summary() now shows "leverage[1]" instead of "gamma[1]", "df" instead of "nu"
result.summary()

# to_dict() keys also reflect the custom names
d = result.to_dict()
print(d['garch_params'].keys())  # includes 'leverage' instead of 'gamma'

# Reset when done
volkit.settings.names.reset()
```

### Volatility Forecasting

```python
from volkit import GARCH, GJRGARCH, Normal
import numpy as np

# --- GARCH(1,1) forecast ---
spec = GARCH(1, 1) + Normal()
result = spec.fit(returns)

omega = result.garch_params.omega
alpha = result.garch_params.alpha[0]
beta = result.garch_params.beta[0]

last_resid2 = returns[-1]**2
last_sigma2 = result.sigma2[-1]

forecast_sigma2 = omega + alpha * last_resid2 + beta * last_sigma2
print(f"GARCH forecast: {np.sqrt(forecast_sigma2):.4f}")

# --- GJR-GARCH(1,1) forecast ---
spec_gjr = GJRGARCH(1, 1) + Normal()
result_gjr = spec_gjr.fit(returns)

gp = result_gjr.garch_params
last_resid = returns[-1]
indicator = 1.0 if last_resid < 0 else 0.0

forecast_gjr = (gp.omega
    + gp.alpha[0] * last_resid**2
    + gp.gamma[0] * indicator * last_resid**2
    + gp.beta[0] * result_gjr.sigma2[-1])
print(f"GJR-GARCH forecast: {np.sqrt(forecast_gjr):.4f}")
```

---

## License

[Add license information]

## Citation

[Add citation information]
