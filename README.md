# volkit

Volatility modeling in Python. GARCH-family models with C extensions for speed.

## Install

```bash
# Requires a C compiler
pip install -e .

# Or:
make dev
```

## Quick start

```python
import numpy as np
from volkit import GARCH, Normal, StudentT

np.random.seed(42)
returns = np.random.randn(1000) * 0.01

spec = GARCH(1, 1) + Normal()
result = spec.fit(returns)
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

## Contents

1. [Model specification](#model-specification)
2. [Components](#components)
3. [Estimation](#estimation)
4. [Automatic model selection](#automatic-model-selection)
5. [Results](#results)
6. [Display settings](#display-settings)
7. [Diagnostics](#diagnostics)
8. [API reference](#api-reference)

---

## Model specification

Build models by combining components with `+`. volkit orders them automatically: MEAN, then VOLATILITY, then DENSITY.

```python
from volkit import GARCH, GJRGARCH, ARMA, Normal, StudentT, SkewT

spec = GARCH(1, 1)                          # Normal density by default
spec = GARCH(1, 1) + StudentT()             # Explicit density
spec = GJRGARCH(1, 1) + StudentT()          # Asymmetric volatility
spec = ARMA(1, 1) + GARCH(1, 1) + SkewT()  # Mean + volatility + density
```

One component per role. If you omit the density, `Normal()` is added for you.

Alternative operators produce the same result:

```python
spec = garch + normal      # __add__
spec = garch < normal      # __lt__
spec = garch << normal     # __lshift__
spec = normal >> garch     # __rlshift__
```

---

## Components

### GARCH(p, q)

Conditional variance:

```
σ²_t = ω + Σᵢ αᵢ·ε²_{t-i} + Σⱼ βⱼ·σ²_{t-j}
```

Stationary when Σα + Σβ < 1.

```python
spec = GARCH(1, 1)  # most common
spec = GARCH(2, 1)
```

Parameters: `omega` (ω > 0), `alpha[1:p]` (ARCH terms), `beta[1:q]` (GARCH terms).

### GJRGARCH(p, q)

Adds a leverage term so that negative shocks raise volatility more than positive shocks of the same size:

```
σ²_t = ω + Σᵢ (αᵢ + γᵢ·I(ε_{t-i}<0))·ε²_{t-i} + Σⱼ βⱼ·σ²_{t-j}
```

A negative shock contributes (α + γ)·ε² to the next period's variance; a positive shock contributes α·ε².

Stationarity:
- Symmetric densities (Normal, Student-t): α + 0.5·γ + β < 1
- Asymmetric densities (Skew-t): α + γ·P(z < 0) + β < 1

```python
spec = GJRGARCH(1, 1) + StudentT()
```

Parameters: `omega`, `alpha[1:p]`, `gamma[1:p]` (leverage), `beta[1:q]`.

### ARMA(p, q)

Conditional mean. Currently limited to ARMA(1,1).

```python
spec = ARMA(1, 1) + GARCH(1, 1)
```

### Normal()

Gaussian density. No extra parameters. Default when none is specified.

### StudentT()

Heavier tails than Normal. One extra parameter: `nu` (ν > 2). Lower ν means fatter tails; as ν grows, the distribution approaches Normal.

### SkewT()

Hansen's skewed Student-t. Two extra parameters: `nu` (ν > 2) and `lambda` (−1 < λ < 1). λ = 0 gives a symmetric Student-t; λ < 0 shifts weight to the left tail.

---

## Estimation

### MLE

The default method.

```python
result = spec.fit(data)

result = spec.fit(
    data,
    solver="trust",
    log_mode=True,
    verbose=True,
)
```

Solvers:

| Solver | Method | Notes |
|--------|--------|-------|
| `"slsqp"` | Sequential quadratic programming | Default; fast and reliable |
| `"nelder-mead"` | Derivative-free simplex | Slow but dependable |
| `"trust"` | Trust-region (gradient + Hessian) | Fast when it converges |
| `"trust-exact"` | Trust-region in log-space | Most stable for difficult data |

**Log-mode** (`log_mode=True`) transforms constrained parameters into unconstrained space before optimization:

| Parameter | Constraint | Transform |
|-----------|------------|-----------|
| ω | ω > 0 | exp(z) |
| α, β | > 0, sum < 1 | softmax |
| γ (GJR) | γ > 0 | 4-class softmax |
| ν | ν > 2 | 2 + exp(z) |
| λ | −1 < λ < 1 | tanh(z) |

This guarantees stationarity by construction and avoids boundary problems during optimization.

### QMLE

Quasi-maximum likelihood: fit under Normal likelihood, then compute sandwich standard errors valid under distributional misspecification. Pass `method='qmle'`:

```python
spec = GARCH(1, 1) + Normal()
result = spec.fit(data, method='qmle')

result.std_errors        # MLE standard errors
result.std_errors_robust # sandwich (robust) standard errors
```

For Student-t or Skew-t, QMLE runs a two-step procedure: first it estimates GARCH parameters under Normal likelihood with sandwich errors, then it fixes those parameters and estimates the distribution parameters by MLE.

```python
spec = GARCH(1, 1) + StudentT()
result = spec.fit(data, method='qmle')

spec = GJRGARCH(1, 1) + Normal()
result = spec.fit(data, method='qmle')
```

---

## Automatic model selection

### By GARCH order

Search over lag orders with `auto=True`:

```python
spec = GARCH(auto=True) + Normal()      # p, q in [1, 3]
spec = GJRGARCH(auto=True) + Normal()

spec = GARCH(auto={'max_p': 2, 'max_q': 2}) + Normal()  # narrower grid
spec = GJRGARCH(p=1, q='auto') + StudentT()              # fix p, search q
```

### By volatility model

`AutoVol` searches across both GARCH and GJRGARCH families:

```python
from volkit import AutoVol

spec = AutoVol() + Normal()
result = spec.fit(returns)

spec = AutoVol(candidates=['GJRGARCH'], max_p=2, max_q=2) + StudentT()
```

### By distribution

`AutoDensity` searches across Normal, StudentT, and SkewT:

```python
from volkit import AutoDensity

spec = GARCH(1, 1) + AutoDensity()
spec = GARCH(1, 1) + AutoDensity(candidates=['Normal', 'StudentT'])
```

### Full search

Combine them to search volatility model, order, and distribution at once:

```python
from volkit import AutoVol, AutoDensity

spec = AutoVol() + AutoDensity()
result = spec.fit(returns, verbose_selection=True)

print(result.spec)
result.selection_summary()
```

### Selection criterion

The default score is:

```
Score = AIC + diagnostic_weight × n_failed_tests
```

where `n_failed_tests` counts failures of the DGT and Ljung-Box tests. Default `diagnostic_weight` is 50.

```python
# Heavier diagnostic penalty
result = spec.fit(returns, diagnostic_weight=100.0)

# AIC only
result = spec.fit(returns, diagnostic_weight=0.0)
```

For full control, pass a callable:

```python
def my_criterion(result, diagnostics):
    score = result.bic
    if diagnostics is not None:
        if diagnostics['dgt']['p_value'] < 0.01:
            score += 200
        if diagnostics['ljung_box'][1]['reject']:
            score += 100
    return score

spec = AutoVol() + AutoDensity()
result = spec.fit(returns, criterion=my_criterion)
```

The `diagnostics` dict matches what `result.diagnostic_tests()` returns:

```python
{
    'distribution': 'StudentT',
    'dist_params': {'nu': 7.42, 'lam': None},
    'n_obs': 1000,
    'alpha': 0.05,
    'dgt': {
        'n_cells': 40, 'chi2_stat': 34.2,
        'df': 39, 'p_value': 0.689, 'reject': False,
    },
    'ljung_box': {
        1: {'lags': 10, 'q_stat': 8.42, 'p_value': 0.588, 'reject': False},
        2: {'lags': 10, 'q_stat': 7.91, 'p_value': 0.637, 'reject': False},
        3: {'lags': 10, 'q_stat': 11.2, 'p_value': 0.340, 'reject': False},
        4: {'lags': 10, 'q_stat': 9.87, 'p_value': 0.452, 'reject': False},
    },
    'pit': np.ndarray,
}
```

Pass `diagnostic_kwargs` to tune test settings:

```python
result = spec.fit(returns, diagnostic_kwargs={'lags': 20, 'n_cells': 50})
```

When `criterion` is provided, `diagnostic_weight` is ignored.

### Parallel fitting

Auto-selection fits candidates in parallel by default:

```python
result = spec.fit(returns)           # all cores
result = spec.fit(returns, n_jobs=4) # 4 workers
result = spec.fit(returns, n_jobs=1) # sequential
```

### Multi-series fitting

Fit one specification to many series at once:

```python
spec = GARCH(1, 1) + Normal()
results = spec.fit_multiple([returns1, returns2, returns3], n_jobs=4)

for i, r in enumerate(results):
    print(f"Series {i}: persistence = {r.garch_params.persistence:.4f}")
```

Auto-selection works here too -- each series gets its own best model:

```python
spec = GARCH(auto=True) + AutoDensity()
results = spec.fit_multiple(returns_list, n_jobs=4)
```

### Inspecting candidates

```python
result.selection_summary()

for c in result._selection_candidates[:5]:
    print(f"{c.spec}: AIC={c.aic:.2f}, Score={c.score:.2f}")
```

QMLE with AutoDensity is redundant (QMLE always uses Normal likelihood). volkit warns and fits Normal only.

---

## Results

`spec.fit()` returns an `EstimationResult`.

### Parameters

```python
result.params            # flat array: [omega, alpha_1, ..., beta_q, nu?, lambda?]

gp = result.garch_params
gp.omega                 # constant
gp.alpha                 # ARCH coefficients (array)
gp.gamma                 # leverage coefficients (GJR-GARCH only)
gp.beta                  # GARCH coefficients (array)
gp.persistence           # α+β (GARCH) or α+0.5γ+β (GJR-GARCH)

dp = result.dist_params
dp.nu                    # degrees of freedom (StudentT, SkewT)
dp.lam                   # skewness (SkewT)
```

### Fit statistics

```python
result.loglikelihood
result.aic
result.bic
result.hqic
```

### Conditional variances and residuals

```python
result.sigma2        # σ²_t
result.volatility    # σ_t
result.std_resid     # ε_t / σ_t
```

### Standard errors

```python
result.std_errors        # MLE (from inverse Hessian)
result.std_errors_robust # sandwich (QMLE only)
result.cov_matrix        # H⁻¹
result.cov_robust        # H⁻¹ @ OPG @ H⁻¹
```

### Output

```python
result.summary()              # full table
result.summary(robust=True)   # with sandwich SEs
print(result)                 # compact
result.to_dict()              # for programmatic use
```

---

## Display settings

Override how parameter names appear in summaries, print output, and `to_dict()` keys:

```python
import volkit

volkit.settings.names.gamma = "leverage"
volkit.settings.names.nu = "df"
volkit.settings.names.alpha = "a"

# Now result.summary() shows "leverage[1]" instead of "gamma[1]"
# and result.to_dict() uses "leverage" as a key
```

Overrides apply to indexed variants too: renaming `alpha` to `a` turns `alpha[1]`, `alpha[2]` into `a[1]`, `a[2]`.

Reset with:

```python
volkit.settings.names.reset()
```

Available names: `omega`, `alpha`, `gamma`, `beta`, `nu`, `lambda`, `const`, `ar`, `ma`.

Internal attribute names (`gp.omega`, `dp.nu`, etc.) never change.

---

## Diagnostics

### DGT and Ljung-Box tests

Check whether the fitted distribution captures the data:

```python
spec = GARCH(1, 1) + StudentT()
result = spec.fit(returns)
result.diagnostic_tests()
```

```
======================================================================
                     Model Diagnostic Tests
======================================================================
Distribution:  StudentT (nu=7.42)
Observations:  1000
Alpha:         0.05

DGT Test (Diebold-Gunther-Tay)
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

The DGT test checks whether PIT residuals are uniform -- "No" rejection means the distribution fits. Ljung-Box tests check for serial correlation in PIT moments: moment 1 targets the mean, moment 2 the variance, moments 3--4 skewness and kurtosis.

```python
result.diagnostic_tests(alpha=0.01, n_cells=50, lags=20)

diag = result.diagnostic_tests(print_results=False)
diag['dgt']['p_value']
diag['ljung_box'][2]['reject']
```

Auto-selection uses these tests internally to penalize poorly fitting candidates.

### Derivative validation

Confirm that analytical gradients and Hessians match finite differences:

```python
report = spec.validate_derivatives(data)
report.summary()

report.gradient_passed       # bool
report.hessian_passed        # bool
report.gradient_max_rel_error
report.hessian_max_rel_error
```

Or use the standalone functions:

```python
from volkit._devtools import validate_derivatives, quick_check

report = validate_derivatives(spec, data)
passed = quick_check(spec, data)
```

---

## API reference

### Exports

```python
from volkit import (
    GARCH, GJRGARCH, ARMA,
    Normal, StudentT, SkewT,
    AutoDensity, AutoVol,
    Component, CompositeSpec, Role,
    settings, __version__,
)
```

### spec.fit()

```python
result = spec.fit(
    data,                      # 1D array, Series, or DataFrame
    method="mle",              # "mle" or "qmle"
    solver="trust",
    log_mode=True,
    verbose=False,
    n_jobs=None,               # parallel workers (auto-selection)
    diagnostic_weight=50.0,    # AIC penalty per failed test
    criterion=None,            # custom scoring callable
    diagnostic_kwargs=None,    # forwarded to diagnostic_tests()
)
```

### GARCH

```python
g = GARCH(1, 1)
g = GARCH(auto=True)
g = GARCH(auto={'max_p': 2, 'max_q': 2})

g.p, g.q, g.n_params, g.signature
g.fitted_params   # {'omega': ..., 'alpha': [...], 'beta': [...]}
g.persistence()
g.is_stationary()
g.unconditional_variance()
```

### GJRGARCH

```python
gjr = GJRGARCH(1, 1)
gjr = GJRGARCH(auto=True)

gjr.n_params              # 1 + 2p + q
gjr.fitted_params         # includes 'gamma'
gjr.persistence()         # α + 0.5·γ + β
gjr.persistence(p_neg=0.6)
```

### AutoVol

```python
av = AutoVol()
av = AutoVol(candidates=['GJRGARCH'], max_p=2, max_q=2)
av.get_candidates()  # list of (model, p, q) tuples
```

### AutoDensity

```python
ad = AutoDensity()
ad = AutoDensity(candidates=['Normal', 'StudentT'])
```

### EstimationResult

```python
result.params, result.garch_params, result.dist_params
result.loglikelihood, result.aic, result.bic, result.hqic
result.sigma2, result.volatility, result.std_resid
result.std_errors, result.std_errors_robust
result.cov_matrix, result.cov_robust
result.success, result.niter, result.time_elapsed
result.summary(), result.to_dict(), result.diagnostic_tests()
result.selection_summary()
result._selection_candidates  # list of all evaluated models
```

---

## Examples

### Fit GARCH(1,1) and inspect results

```python
import numpy as np
from volkit import GARCH, Normal

returns = np.random.randn(1000) * 0.01

spec = GARCH(1, 1) + Normal()
result = spec.fit(returns)

print(f"Persistence: {result.garch_params.persistence:.4f}")
print(f"Log-likelihood: {result.loglikelihood:.2f}")
result.summary()
```

### Compare MLE and sandwich standard errors

```python
from volkit import GARCH, Normal

spec = GARCH(1, 1) + Normal()
result = spec.fit(returns, method='qmle')

print("Parameter       MLE SE    Robust SE")
print("-" * 40)
for i, name in enumerate(['omega', 'alpha', 'beta']):
    print(f"{name:10}  {result.std_errors[i]:10.6f}  {result.std_errors_robust[i]:10.6f}")
```

### GJR-GARCH leverage effect

```python
from volkit import GJRGARCH, StudentT

spec = GJRGARCH(1, 1) + StudentT()
result = spec.fit(returns)

gp = result.garch_params
print(f"alpha: {gp.alpha[0]:.4f}")
print(f"gamma: {gp.gamma[0]:.4f}")
print(f"beta:  {gp.beta[0]:.4f}")
print(f"nu:    {result.dist_params.nu:.2f}")

if gp.gamma[0] > 0:
    ratio = (gp.alpha[0] + gp.gamma[0]) / gp.alpha[0]
    print(f"Negative shocks hit {ratio:.1f}x harder than positive")
```

### Full automatic search

```python
from volkit import AutoVol, AutoDensity

spec = AutoVol() + AutoDensity()
result = spec.fit(returns, verbose_selection=True)

print(f"Best: {result.spec}")
result.selection_summary()
```

### One-step-ahead variance forecast

```python
from volkit import GARCH, GJRGARCH, Normal
import numpy as np

# GARCH(1,1)
spec = GARCH(1, 1) + Normal()
result = spec.fit(returns)

gp = result.garch_params
h_next = gp.omega + gp.alpha[0] * returns[-1]**2 + gp.beta[0] * result.sigma2[-1]
print(f"GARCH forecast σ: {np.sqrt(h_next):.6f}")

# GJR-GARCH(1,1)
spec_gjr = GJRGARCH(1, 1) + Normal()
result_gjr = spec_gjr.fit(returns)

gp = result_gjr.garch_params
indicator = 1.0 if returns[-1] < 0 else 0.0
h_next = (gp.omega
    + gp.alpha[0] * returns[-1]**2
    + gp.gamma[0] * indicator * returns[-1]**2
    + gp.beta[0] * result_gjr.sigma2[-1])
print(f"GJR forecast σ:   {np.sqrt(h_next):.6f}")
```

---

## License

[Add license information]

## Citation

[Add citation information]
