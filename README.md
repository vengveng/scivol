# scivol

Fast, accurate volatility models for time series in Python.

scivol is a Python library for likelihood-based volatility modeling with
GARCH-family models. It provides composable model specifications such as
`GARCH(1, 1) + StudentT()` and `ARMA(1, 1) + GARCH(1, 1) + SkewT()`. scivol
uses analytical gradients and Hessians for maximum speed and accuracy, and
verifies those derivatives against automatic differentiation. It also provides MLE and 
QMLE estimation, diagnostic tests and automatic model selection.

## Supported models

- Univariate volatility models: `GARCH(p, q)`, `GJRGARCH(p, q)`, `EGARCH(p, q)`, `APARCH(p, q)`
- Conditional densities: `Normal()`, `StudentT()`, `SkewT()`, `GED()`,
  `GeneralizedHyperbolic()`, `Hyperbolic()`, `NormalInverseGaussian()`, and
  `VarianceGamma()`
- Mean model: `ARMA(p, q)` (standalone with `Normal()`, `GED()`, and all GH-family densities)
- Composite univariate specifications: volatility-only models and
  `ARMA(p, q) + GARCH(P, Q)` with any shipped linked density, plus
  `ARMA(p, q) + EGARCH(P, Q)` with any shipped linked density
- Exogenous-mean specifications: standalone and linked `ARX/HARX` surfaces with
  all shipped densities
- Multivariate correlation model: `DCC(p, q)` with Gaussian or common-df Student-t copula dynamics
- Automatic model selection helpers: `AutoMean()`, `AutoVol()`, and `AutoDensity()`

## Performance

Scivol consistently outperforms other libraries in both speed and optimality. Below are benchmarks on synthetic data. Unless noted, each row uses 1,000,000 observations.

GARCH(1,1)-Normal:
| Library | Time | Converged | Reported log-likelihood |
|---|---:|---|---:|
| scivol | 0.300 s | Yes | -1,379,431.12 |
| Python `arch` 8.0.0 | 1.951 s | Yes | -1,379,777.40 |
| MATLAB R2024a `garch(1,1)` | 3.400 s | Yes | -1,379,432.00 |
| R `rugarch` 1.5.5 | 24.029 s | Yes | -1,379,432.00 |
| Manual Python | 6.659 s | Yes | -1,383,160.89 |

GJR-GARCH(5,5)+Student-t:
| Library | Time | Converged | Reported log-likelihood |
|---|---:|---|---:|
| scivol | 7.583 s | Yes | -844,600.85 |
| Python `arch` 8.0.0 | 27.267 s | Yes | -844,604.00 |
| MATLAB R2024a `gjr(5,5)` | 44.125 s | Yes | -844,604.00 |
| R `rugarch` 1.5.5 | 477.904 s | Yes | -844,605.57 |

AR(1)+EGARCH(2,1)+Normal:
| Library | Time | Converged | Reported log-likelihood |
|---|---:|---|---:|
| scivol | 1.895 s | Yes | -318,924.29 |
| Python `arch` 8.0.0 | 9.792 s | Yes | -318,925.65 |
| MATLAB R2024a `arima` + `egarch` | 83.554 s | Yes | -318,925.47 |
| R `rugarch` 1.5.5 | 121.280 s | Yes | -318,925.49 |

## Install

```bash
pip install scivol
```

Prebuilt wheels are the default installation path on supported platforms.
If no compatible wheel is available, `pip` falls back to building from the
source distribution, which requires a working C toolchain.
`tqdm` ships as a normal dependency, so progress bars are available out of the box.

## Quick start

```python
import numpy as np
from scivol import GARCH, Normal, StudentT

np.random.seed(42)
returns = np.random.randn(1000) * 0.01

spec = GARCH(1, 1) + Normal()
result = spec.fit(returns)
result.summary()
```

For workflows that iterate over many fits, progress bars are enabled by default.
You can disable them globally:

```python
import scivol

scivol.settings.show_progress = False
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
Persistence (alpha + beta): 0.993500
Stationary:               Yes
Unconditional Variance:   1.900000e-04
Half-life (periods):      106.3
══════════════════════════════════════════════════════════════════════
```

---

## Contents

1. [Performance](#performance)
2. [Model specification](#model-specification)
3. [Components](#components)
4. [Estimation](#estimation)
5. [Automatic model selection](#automatic-model-selection)
6. [Results](#results)
7. [Display settings](#display-settings)
8. [Diagnostics](#diagnostics)
9. [Method references](#method-references)
10. [API reference](#api-reference)

---

## Model specification

Build models by combining components with `+`. scivol orders them automatically: MEAN, then VOLATILITY, then DENSITY.

```python
from scivol import (
    APARCH, ARMA, DCC, EGARCH, GARCH, GJRGARCH,
    GeneralizedHyperbolic, Hyperbolic, Normal, StudentT, SkewT,
    VarianceGamma,
)

spec = GARCH(1, 1)                          # Normal density by default
spec = GARCH(1, 1) + StudentT()             # Explicit density
spec = GJRGARCH(1, 1) + StudentT()          # Asymmetric volatility
spec = EGARCH(2, 1) + Normal()              # Log-variance volatility model
spec = APARCH(1, 1) + StudentT()            # Asymmetric power volatility
spec = ARMA(1, 1) + GARCH(1, 1) + SkewT()  # Mean + volatility + density
spec = ARMA(1, 1) + EGARCH(2, 1) + Normal() # Linked mean + EGARCH volatility
spec = GARCH(1, 1) + GeneralizedHyperbolic()
spec = GARCH(1, 1) + Hyperbolic()
spec = GARCH(1, 1) + VarianceGamma()
dcc = DCC(1, 1)                             # Dynamic correlations (multivariate)
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

`GARCH(p, q)` is [Bollerslev's (1986)](<https://doi.org/10.1016/0304-4076(86)90063-1>)
generalization of [Engle's (1982)](<https://doi.org/10.2307/1912773>) ARCH
process. It lets conditional variance depend on both squared past shocks and
past conditional variances.

Conditional variance:

$$
\sigma_t^2 = \omega + \sum_i \alpha_i \varepsilon_{t-i}^2 + \sum_j \beta_j \sigma_{t-j}^2
$$

Stationary when $\sum \alpha + \sum \beta < 1$.

```python
spec = GARCH(1, 1)  # most common
spec = GARCH(2, 1)
```

Parameters: `omega` ($\omega > 0$), `alpha[1:p]` (ARCH terms), `beta[1:q]` (GARCH terms).

### GJRGARCH(p, q)

The asymmetric specification of
[Glosten, Jagannathan, and Runkle (1993)](<https://doi.org/10.1111/j.1540-6261.1993.tb05128.x>)
adds a leverage term so that negative shocks raise volatility more than
positive shocks of the same size:

$$
\sigma_t^2 = \omega + \sum_i \left(\alpha_i + \gamma_i I(\varepsilon_{t-i} < 0)\right)\varepsilon_{t-i}^2 + \sum_j \beta_j \sigma_{t-j}^2
$$

A negative shock contributes $(\alpha + \gamma)\varepsilon^2$ to the next period's variance; a positive shock contributes $\alpha \varepsilon^2$.

Stationarity:
- Symmetric densities (Normal, Student-t): $\alpha + 0.5\gamma + \beta < 1$
- Asymmetric densities (Skew-t): $\alpha + \gamma P(z < 0) + \beta < 1$

```python
spec = GJRGARCH(1, 1) + StudentT()
```

### EGARCH(p, q)

[Nelson's (1991)](<https://doi.org/10.2307/2938260>) exponential GARCH model
specifies volatility in logs, so the recursion produces a positive variance
without non-negativity constraints on every shock coefficient:

$$
\log h_t = \omega + \sum_{i=1}^{p}\alpha_i(|z_{t-i}| - \mathbb{E}|z|) + \sum_{i=1}^{p}\gamma_i z_{t-i} + \sum_{j=1}^{q}\beta_j \log h_{t-j}
$$

```python
spec = EGARCH(1, 1) + Normal()
spec = EGARCH(2, 1) + StudentT()
```

`EGARCH` is shipped with every conditional density listed below.

Parameters: `omega`, `alpha[1:p]`, `gamma[1:p]` (leverage), `beta[1:q]`.

### APARCH(p, q)

The asymmetric power ARCH model of
[Ding, Granger, and Engle (1993)](<https://doi.org/10.1016/0927-5398(93)90006-D>)
estimates the power applied to volatility rather than fixing it at two:

$$
\sigma_t^\delta = \omega + \sum_i \alpha_i
(|\varepsilon_{t-i}|-\gamma_i\varepsilon_{t-i})^\delta
+ \sum_j \beta_j\sigma_{t-j}^\delta
$$

```python
spec = APARCH(1, 1) + Normal()
spec = APARCH(2, 1) + GeneralizedHyperbolic()
```

`APARCH` supports orders up to 3 and every shipped density. Its first release is
zero-mean and MLE-only. Parameters are `omega`, `alpha[1:p]`, `gamma[1:p]`,
`beta[1:q]`, and `delta`.

### DCC(p, q)

Dynamic Conditional Correlation for multivariate return series, introduced by
[Engle (2002)](<https://doi.org/10.1198/073500102288618487>).

`DCC.fit()` uses a two-step workflow:
1. Fit a univariate volatility model to each series.
2. Fit Gaussian or Student-t copula DCC dynamics to the standardized residuals.

```python
from scivol import DCC, GARCH, StudentT

dcc = DCC(1, 1)
result = dcc.fit(returns_df, univariate_spec=GARCH(1, 1) + StudentT())

t_dcc = DCC(1, 1, distribution="studentt")
t_result = t_dcc.fit(returns_df, univariate_spec=GARCH(1, 1) + StudentT())
t_result.params.nu
```

Result access focuses on the economically useful correlation objects:
- `result.Rt` for the full time-varying correlation path
- `result.corr(i, j)` for a single pair
- `result.unconditional_corr` for the long-run correlation matrix
- `result.forecast_correlation(horizon)` for the forecast correlation path
- `result.forecast(horizon)` for joint marginal and correlation forecasts

Draw many outcomes from one unchanged forecast origin with `scenarios()`:

```python
scenarios = result.scenarios(50_000, seed=7)
empirical = result.scenarios(50_000, seed=7, innovations="empirical")
```

The fitted DCC distribution supplies the Gaussian or Student-t copula. The
parametric mode maps its draws through each fitted marginal density; the
empirical mode uses standardized-residual quantiles. `simulate()` uses the same
copula and marginals along an evolving time path.

Rolling DCC fits warm-start both the marginals and correlation dynamics:

```python
rolling = dcc.fit_rolling(
    returns_df,
    window=1000,
    step=20,
    univariate_spec=GARCH(1, 1) + StudentT(),
    n_jobs=4,
)
```

### ARMA(p, q)

Conditional mean. Standalone `ARMA(p, q)` supports `Normal()`, `GED()`, GH,
and NIG. Linked ARMA surfaces support every shipped density. Box and Jenkins
(1970) is the standard reference for the ARMA model-building framework.

```python
spec = ARMA(2, 1)                 # standalone ARMA with Normal errors
spec = ARMA(1, 1) + GARCH(1, 1)
spec = ARMA(1, 1) + EGARCH(1, 1) + StudentT()
```

### ARX / HARX

Array-based mean models with explicit regressors via `x=`.
Standalone and linked `ARX`/`HARX` surfaces support every shipped density.
`HARX` uses averages over the requested horizons, following the heterogeneous
autoregressive structure associated with [Corsi's (2009)](<https://doi.org/10.1093/jjfinec/nbp001>)
HAR model.

```python
from scivol import ARX, HARX, EGARCH, GARCH, GED, StudentT

spec = ARX(1)
result = spec.fit(y, x=x)

spec = HARX((1, 5))
result = spec.fit(y, x=x)

spec = ARX(1) + GARCH(1, 1) + StudentT()
result = spec.fit(y, x=x)

spec = HARX((1, 5)) + GARCH(1, 1) + GED()
result = spec.fit(y, x=x)

spec = ARX(1) + StudentT()
result = spec.fit(y, x=x)

spec = HARX((1, 5)) + EGARCH(1, 1) + GED()
result = spec.fit(y, x=x)
```

### Normal()

Gaussian density. No extra parameters. Default when none is specified.

### StudentT()

Heavier tails than Normal. One extra parameter: `nu` ($\nu > 2$). Lower $\nu$ means fatter tails; as $\nu$ grows, the distribution approaches Normal.

### SkewT()

[Hansen's (1994)](<https://doi.org/10.2307/2527081>) zero-mean, unit-variance
skewed Student-t. Two extra parameters: `nu` ($\nu > 2$) and `lambda`
($-1 < \lambda < 1$). $\lambda = 0$ gives a symmetric Student-t; $\lambda < 0$
shifts weight to the left tail.

### GED()

Zero-mean, unit-variance generalized error density. One extra parameter: `nu`.

### GeneralizedHyperbolic()

Zero-mean, unit-variance generalized hyperbolic density from the family
introduced by [Barndorff-Nielsen (1977)](<https://doi.org/10.1098/rspa.1977.0041>).
It estimates `rho`, `zeta`, and `lambda`.

### NormalInverseGaussian()

A separate two-parameter density class developed by
[Barndorff-Nielsen (1997)](<https://doi.org/10.1111/1467-9469.00045>) and backed
by the shared GH core. It estimates `rho` and `zeta` and fixes the GH family
index at `lambda=-0.5`.

### Hyperbolic()

A separate two-parameter density class backed by the shared GH core. It
estimates `rho` and `zeta` and fixes the GH family index at `lambda=1`.

### VarianceGamma()

The standardized Variance-Gamma boundary of the GH family. Madan and Seneta
[(1990)](<https://doi.org/10.1086/296519>) introduced the Variance-Gamma model
for asset returns. scivol estimates `rho` and `lambda`, with `lambda > 1.5` so
the observed Hessian remains finite at the density center.

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
| `"native"` | C++ function-jet SLSQP | Default for fixed MLE GARCH with Normal, Student-t, Skew-t, or GED |
| `"slsqp"` | SciPy SLSQP over the established C kernels | Explicit independent legacy path |
| `"legacy"` | Alias for `"slsqp"` | Keeps the established engine easy to request |
| `"nelder-mead"` | Derivative-free simplex | Slow but dependable |
| `"trust"` | Trust-region (gradient + Hessian) | Fast when it converges |
| `"trust-exact"` | Trust-region in log-space | Most stable for difficult data |

**Log-mode** (`log_mode=True`) transforms constrained parameters into unconstrained space before optimization:

| Parameter | Constraint | Transform |
|-----------|------------|-----------|
| $\omega$ | $\omega > 0$ | `softplus(z)` |
| $\alpha, \beta$ | $\ge 0$, sum $< 1$ | `softmax` interior chart |
| $\gamma$ (GJR) | $\gamma > 0$ | `4-class softmax` |
| $\nu$ | $\nu > 2$ | `2 + softplus(z)` |
| $\lambda$ | $-1 < \lambda < 1$ | `tanh(z)` |

This guarantees stationarity by construction and avoids boundary problems during optimization.

Fixed MLE GARCH models use the native transformed engine by default with Normal,
Student-t, Skew-t, or GED density. Pass `solver="native"` to use it explicitly
for GH-family GARCH or genuine joint ARMA-GARCH models. Pass `solver="slsqp"` or
`solver="legacy"` to run the independent SciPy path. Passing `log_mode=False`
without a solver also selects the legacy theta-space path. Joint ARMA-GARCH,
GH-family GARCH, QMLE, and auto-selection retain legacy defaults pending their
own promotion gates.

Other model families retain their benchmark-driven `log_mode` defaults. Every
fit reports the effective path:

```python
result = spec.fit(data)
result.fit_info.to_dict()
# solver='native-slsqp'/'slsqp', optimization_space='native-z-space'/'z-space'/'theta-space'
```

Volatility orders do not implicitly remove observations. Mean models apply the
lags they require; `hold_back=k` requests at least `k` leading observations be
trimmed. Results preserve pandas alignment, and the effective sample is recorded
in `result.fit_info`.

Standalone volatility fits and native joint ARMA-GARCH fits accept
constrained-space warm starts and explicit optimizer budgets. The same start
works in theta- or z-space:

```python
first = spec.fit(returns, maxiter=500)
updated = spec.fit(new_returns, initial_params=first, maxiter=100)
updated.fit_info.warm_started
```

Current default path policy:

| Family | Default path |
|--------|--------------|
| `GARCH + Normal / StudentT / SkewT / GED` | native `z-space` |
| `GARCH + GH / NIG / Hyperbolic / VarianceGamma` | `theta-space` |
| `GJRGARCH + Normal` | `z-space` |
| `GJRGARCH + StudentT / SkewT` | `theta-space` |
| `EGARCH + Normal` | `theta-space` |
| `EGARCH + StudentT` | `theta-space` |
| `EGARCH + SkewT` | `theta-space` |
| `EGARCH + GED` | `theta-space` |
| `APARCH + any shipped density` | `theta-space` |
| `ARMA + Normal / GED` | `theta-space` |
| `ARX/HARX + Normal / StudentT / SkewT / GED` | `theta-space` |
| `ARMA + GARCH + any shipped density` | `theta-space` |
| `ARMA + EGARCH + Normal / StudentT / SkewT / GED` | `theta-space` |
| `ARX/HARX + GARCH + Normal / StudentT / SkewT / GED` | `theta-space` |
| `ARX/HARX + EGARCH + Normal / StudentT / SkewT / GED` | `theta-space` |

For `EGARCH + Normal`, the current SLSQP policy is based on targeted checks of the shipped `EGARCH(1,1)` and `EGARCH(2,1)` surfaces. Those runs showed theta-space matching z-space on fit quality while usually converging in fewer iterations and less wall time.

### QMLE

scivol follows the Gaussian quasi-maximum likelihood framework of
[Bollerslev and Wooldridge (1992)](<https://doi.org/10.1080/07474939208800229>):
fit under a Normal likelihood, then compute sandwich standard errors that remain
valid under distributional misspecification when the conditional mean and
variance are correctly specified. Pass `method='qmle'`:

```python
spec = GARCH(1, 1) + Normal()
result = spec.fit(data, method='qmle')

result.std_errors        # MLE standard errors
result.std_errors_robust # sandwich (robust) standard errors
```

This works for standalone `GARCH`, standalone `GJRGARCH`, and joint `ARMA + GARCH` specifications. For joint fits, the robust covariance is computed for the full mean-plus-volatility parameter vector under the Normal QMLE step.

For Student-t or Skew-t, QMLE runs a two-step procedure: first it estimates the mean/volatility parameters under Normal likelihood with sandwich errors, then it fixes those parameters and estimates the distribution parameters by MLE.

```python
from scivol import ARMA

spec = GARCH(1, 1) + StudentT()
result = spec.fit(data, method='qmle')

spec = GJRGARCH(1, 1) + Normal()
result = spec.fit(data, method='qmle')

spec = ARMA(2, 1) + GARCH(1, 2) + SkewT()
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

`AutoVol` defaults to GARCH and GJRGARCH. EGARCH and APARCH are available as
explicit candidates:

```python
from scivol import AutoVol

spec = AutoVol() + Normal()
result = spec.fit(returns)

spec = AutoVol(candidates=['GJRGARCH'], max_p=2, max_q=2) + StudentT()
spec = AutoVol(candidates=['APARCH'], max_p=2, max_q=2) + Normal()
```

### By distribution

`AutoDensity` defaults to Normal, StudentT, and SkewT. GH-family densities are
available as explicit candidates:

```python
from scivol import AutoDensity

spec = GARCH(1, 1) + AutoDensity()
spec = GARCH(1, 1) + AutoDensity(candidates=['Normal', 'StudentT'])
spec = GARCH(1, 1) + AutoDensity(
    candidates=['Normal', 'Hyperbolic', 'VarianceGamma']
)
```

### By mean model

`AutoMean` searches a zero mean and ARMA orders through `(2, 2)` by default.
For large three-axis grids it screens every mean against
`GARCH(1,1) + Normal()` and expands only the best mean finalists.

```python
from scivol import AutoMean

spec = AutoMean() + GARCH(1, 1) + Normal()
spec = AutoMean(max_p=3, max_q=2, candidate_budget=150, finalists=3) + AutoVol() + AutoDensity()
```

### Full search

Combine them to search volatility model, order, and distribution at once:

```python
from scivol import AutoVol, AutoDensity

spec = AutoVol() + AutoDensity()
print(len(spec.preview_candidates()))

result = spec.fit(returns, verbose_selection=True)

print(result.spec)
result.selection_summary()
df = result.selection.to_dataframe()
```

### Selection criterion

The default score is:

$$
\mathrm{Score} = \mathrm{AIC} + \mathrm{diagnostic\ penalty} + \mathrm{degeneracy\ penalty}
$$

The degeneracy term flags boundary-hugging and near-integrated fits. Set
`degeneracy_weight=0` to disable it. Candidate fits warm-start from lower-order
neighbors and the Normal fit of the same volatility specification when available.

Use a built-in criterion with `select_by`:

```python
result = spec.fit(returns)                 # "aic+diag" default
result = spec.fit(returns, select_by="aic")
result = spec.fit(returns, select_by="bic")
result = spec.fit(returns, select_by="hqic")
```

The named information criteria refer to Akaike's AIC
[(1974)](<https://doi.org/10.1109/TAC.1974.1100705>), Schwarz's BIC
[(1978)](<https://doi.org/10.1214/aos/1176344136>), and the Hannan-Quinn
criterion [(1979)](<https://doi.org/10.1111/j.2517-6161.1979.tb01072.x>).

Use OOS selection to fit each candidate on the training portion, score the
holdout, then refit the winner on the full sample:

```python
result = spec.fit(returns, select_by="oos")  # log score, last 20%
result = spec.fit(returns, select_by="oos", oos_test_size=500)
result = spec.fit(returns, select_by="oos", oos_refit=20)  # expanding-window refits
```

`oos_loss="qlike"` and `oos_loss="mse"` score variance forecasts, not the full
predictive density. They are blocked with `AutoDensity` unless you pass
`allow_density_blind_loss=True`.

OOS candidates retain aligned per-observation losses. Use them for a bootstrap
model confidence set based on the $T_{\max}$ elimination procedure of
[Hansen, Lunde, and Nason (2011)](<https://doi.org/10.3982/ECTA5771>):

```python
mcs = result.selection.confidence_set(alpha=0.10, random_state=7)
mcs.summary()
mcs.to_dataframe()
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
import pandas as pd

spec = GARCH(1, 1) + Normal()
returns = pd.DataFrame(
    {"asset_1": returns1, "asset_2": returns2, "asset_3": returns3}
)
results = spec.fit(returns, n_jobs=4)

for name, r in results.items():
    print(f"{name}: persistence = {r.garch_params.persistence:.4f}")
```

For repeated portfolio work, `fit_many()` returns structured successes and
failures and supports per-fit limits and prior results:

```python
batch = spec.fit_many(
    returns,
    n_jobs=4,
    maxiter=500,
    timeout=10.0,
    on_error="record",
)

updated = spec.fit_many(new_returns, initial_results=batch, n_jobs=4)
batch.results
batch.failures
batch.warning_summary
batch.to_dataframe()
```

Rolling windows reuse each series' last successful parameters:

```python
rolling = spec.fit_rolling(returns, window=1000, step=20, n_jobs=4)
latest = rolling[-1].batch
```

Auto-selection works here too -- each series gets its own best model:

```python
spec = GARCH(auto=True) + AutoDensity()
results = spec.fit(returns, n_jobs=4)
```

### Inspecting candidates

```python
result.selection_summary()

for c in result.selection.candidates[:5]:
    print(f"{c.spec}: AIC={c.aic:.2f}, BIC={c.bic:.2f}, Score={c.score:.2f}")

result.selection.best
result.selection.to_dataframe()
result.selection.decisive
```

The candidate table includes convergence, fit time, warm-start provenance,
diagnostics, boundary flags, persistence, OOS losses, and winner-comparison
[Diebold-Mariano (1995)](<https://doi.org/10.1080/07350015.1995.10524599>)
statistics where applicable.

QMLE with AutoDensity is redundant (QMLE always uses Normal likelihood). scivol warns and fits Normal only.

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
gp.persistence           # alpha+beta (GARCH) or alpha+0.5*gamma+beta (GJR-GARCH)

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
result.sigma2        # sigma^2_t
result.volatility    # sigma_t
result.std_resid     # epsilon_t / sigma_t
```

### Standard errors

```python
result.std_errors        # MLE (from inverse Hessian)
result.std_errors_robust # sandwich (QMLE only)
result.cov_matrix        # H⁻¹
result.cov_robust        # H⁻¹ @ OPG @ H⁻¹
result.warnings          # persistent fit warnings shown by result.summary()
```

### Output

```python
result.fit_info             # effective solver / theta-vs-z path
result.summary()              # full table
result.summary(robust=True)   # with sandwich SEs
print(result)                 # compact
result.to_dict()              # for programmatic use
```

---

## Display settings

Override how parameter names appear in summaries, print output, and `to_dict()` keys:

```python
import scivol

scivol.settings.names.gamma = "leverage"
scivol.settings.names.nu = "df"
scivol.settings.names.alpha = "a"

# Now result.summary() shows "leverage[1]" instead of "gamma[1]"
# and result.to_dict() uses "leverage" as a key
```

Overrides apply to indexed variants too: renaming `alpha` to `a` turns `alpha[1]`, `alpha[2]` into `a[1]`, `a[2]`.

Reset with:

```python
scivol.settings.names.reset()
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

The probability integral transform (PIT) maps each standardized residual through
its fitted conditional CDF. If the conditional density is correctly specified,
the resulting sequence should be independent and uniformly distributed on
$[0,1]$, as described by
[Diebold, Gunther, and Tay (1998)](<https://doi.org/10.2307/2527342>).

scivol's DGT cell statistic divides $[0,1]$ into `n_cells` equal bins and applies
[Pearson's (1900)](<https://doi.org/10.1080/14786440009463897>) chi-square
goodness-of-fit statistic to their counts. It tests marginal uniformity only. A
non-rejection means that the histogram provides no evidence against uniformity;
it does not prove that the density is correctly specified.

The companion [Ljung-Box (1978)](<https://doi.org/10.1093/biomet/65.2.297>)
statistics test serial correlation through `lags` in the centered PIT powers
$(u_t-0.5)^k$, for $k=1,2,3,4$. These transformations are sensitive, in order,
to misspecification in location, dispersion, asymmetry, and tail shape. Read the
DGT and Ljung-Box results together because a calibrated density needs both
uniform PIT values and no remaining serial structure.

```python
result.diagnostic_tests(alpha=0.01, n_cells=50, lags=20)

diag = result.diagnostic_tests(print_results=False)
diag['dgt']['p_value']
diag['ljung_box'][2]['reject']
```

Auto-selection uses these tests internally to penalize poorly fitting candidates.

Every shipped univariate surface has full theta-space and log-space validation
against independent AD reference implementations. The machine-checked contract
and current verification status live in [MODELS.md](MODELS.md).

---

## Method references

### Models and conditional densities

- Barndorff-Nielsen, O. E. (1977). ["Exponentially decreasing distributions for the logarithm of particle size."](<https://doi.org/10.1098/rspa.1977.0041>) *Proceedings of the Royal Society A*, 353, 401-419.
- Barndorff-Nielsen, O. E. (1997). ["Normal inverse Gaussian distributions and stochastic volatility modelling."](<https://doi.org/10.1111/1467-9469.00045>) *Scandinavian Journal of Statistics*, 24, 1-13.
- Bollerslev, T. (1986). ["Generalized autoregressive conditional heteroskedasticity."](<https://doi.org/10.1016/0304-4076(86)90063-1>) *Journal of Econometrics*, 31, 307-327.
- Box, G. E. P., and Jenkins, G. M. (1970). *Time Series Analysis: Forecasting and Control*. Holden-Day.
- Corsi, F. (2009). ["A simple approximate long-memory model of realized volatility."](<https://doi.org/10.1093/jjfinec/nbp001>) *Journal of Financial Econometrics*, 7, 174-196.
- Ding, Z., Granger, C. W. J., and Engle, R. F. (1993). ["A long memory property of stock market returns and a new model."](<https://doi.org/10.1016/0927-5398(93)90006-D>) *Journal of Empirical Finance*, 1, 83-106.
- Engle, R. F. (1982). ["Autoregressive conditional heteroscedasticity with estimates of the variance of United Kingdom inflation."](<https://doi.org/10.2307/1912773>) *Econometrica*, 50, 987-1007.
- Engle, R. F. (2002). ["Dynamic conditional correlation: A simple class of multivariate generalized autoregressive conditional heteroskedasticity models."](<https://doi.org/10.1198/073500102288618487>) *Journal of Business & Economic Statistics*, 20, 339-350.
- Glosten, L. R., Jagannathan, R., and Runkle, D. E. (1993). ["On the relation between the expected value and the volatility of the nominal excess return on stocks."](<https://doi.org/10.1111/j.1540-6261.1993.tb05128.x>) *Journal of Finance*, 48, 1779-1801.
- Hansen, B. E. (1994). ["Autoregressive conditional density estimation."](<https://doi.org/10.2307/2527081>) *International Economic Review*, 35, 705-730.
- Madan, D. B., and Seneta, E. (1990). ["The Variance Gamma model for share market returns."](<https://doi.org/10.1086/296519>) *Journal of Business*, 63, 511-524.
- Nelson, D. B. (1991). ["Conditional heteroskedasticity in asset returns: A new approach."](<https://doi.org/10.2307/2938260>) *Econometrica*, 59, 347-370.

### Estimation and forecast evaluation

- Akaike, H. (1974). ["A new look at the statistical model identification."](<https://doi.org/10.1109/TAC.1974.1100705>) *IEEE Transactions on Automatic Control*, 19, 716-723.
- Bollerslev, T., and Wooldridge, J. M. (1992). ["Quasi-maximum likelihood estimation and inference in dynamic models with time-varying covariances."](<https://doi.org/10.1080/07474939208800229>) *Econometric Reviews*, 11, 143-172.
- Diebold, F. X., Gunther, T. A., and Tay, A. S. (1998). ["Evaluating density forecasts, with applications to financial risk management."](<https://doi.org/10.2307/2527342>) *International Economic Review*, 39, 863-883.
- Diebold, F. X., and Mariano, R. S. (1995). ["Comparing predictive accuracy."](<https://doi.org/10.1080/07350015.1995.10524599>) *Journal of Business & Economic Statistics*, 13, 253-263.
- Hannan, E. J., and Quinn, B. G. (1979). ["The determination of the order of an autoregression."](<https://doi.org/10.1111/j.2517-6161.1979.tb01072.x>) *Journal of the Royal Statistical Society: Series B*, 41, 190-195.
- Hansen, P. R., Lunde, A., and Nason, J. M. (2011). ["The model confidence set."](<https://doi.org/10.3982/ECTA5771>) *Econometrica*, 79, 453-497.
- Ljung, G. M., and Box, G. E. P. (1978). ["On a measure of lack of fit in time series models."](<https://doi.org/10.1093/biomet/65.2.297>) *Biometrika*, 65, 297-303.
- Pearson, K. (1900). ["On the criterion that a given system of deviations from the probable in the case of a correlated system of variables is such that it can be reasonably supposed to have arisen from random sampling."](<https://doi.org/10.1080/14786440009463897>) *Philosophical Magazine*, 50, 157-175.
- Schwarz, G. (1978). ["Estimating the dimension of a model."](<https://doi.org/10.1214/aos/1176344136>) *Annals of Statistics*, 6, 461-464.

---

## API reference

### Exports

```python
from scivol import (
    APARCH, EGARCH, GARCH, GJRGARCH, ARMA, DCC,
    Normal, StudentT, SkewT, GED,
    GeneralizedHyperbolic, Hyperbolic, NormalInverseGaussian, VarianceGamma,
    AutoDensity, AutoMean, AutoVol,
    BatchFitResult, RollingFitResult,
    Component, CompositeSpec, Role,
    DCCParams, DCCResult, MultivariateScenarioResult,
    settings, __version__,
)
```

### spec.fit()

```python
result = spec.fit(
    data,                      # 1D array, Series, or DataFrame
    hold_back=0,               # explicit leading observations to trim
    scale=None,                # optional estimation scale
    common_sample=None,        # align auto-selection candidates
    method="mle",              # "mle" or "qmle"
    solver="trust",
    log_mode=True,
    initial_params=None,        # prior result or constrained parameter vector
    maxiter=5000,               # standalone volatility fits
    maxfev=50000,               # derivative-free evaluation budget
    verbose=False,
    n_jobs=None,               # parallel workers (selection or multi-series)
    diagnostic_weight=50.0,    # AIC penalty per failed test
    select_by="aic+diag",      # "aic+diag", "aic", "bic", "hqic", or "oos"
    criterion=None,            # custom scoring callable
    diagnostic_kwargs=None,    # forwarded to diagnostic_tests()
    oos_loss="logscore",       # for select_by="oos"
    oos_test_size=0.2,         # float fraction or int holdout count
    oos_refit=None,            # expanding-window refit frequency
    min_train_size=250,
    min_test_size=50,
    selection_warm_start=True,
    degeneracy_weight=50.0,
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
gjr.persistence()         # alpha + 0.5*gamma + beta
gjr.persistence(p_neg=0.6)
```

### APARCH

```python
aparch = APARCH(1, 1)
aparch.n_params            # 2 + 2p + q
aparch.fitted_params       # includes gamma and delta
aparch.persistence()       # alpha + beta proxy
```

### AutoVol

```python
av = AutoVol()
av = AutoVol(candidates=['GJRGARCH'], max_p=2, max_q=2)
av = AutoVol(candidates=['APARCH'], max_p=2, max_q=2)
av.get_candidates()  # list of (model, p, q) tuples
```

### AutoMean

```python
am = AutoMean()
am = AutoMean(max_p=3, max_q=2, candidate_budget=150, finalists=3)
am.get_candidates()
```

### AutoDensity

```python
ad = AutoDensity()
ad = AutoDensity(candidates=['Normal', 'StudentT'])
```

### DCC

```python
dcc = DCC(1, 1)
dcc.p, dcc.q, dcc.n_params, dcc.signature

t_dcc = DCC(1, 1, distribution="studentt")
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
result.selection.candidates   # list of all evaluated models
result.selection.to_dataframe()
result.selection.confidence_set(alpha=0.10)
```

### DCCResult

```python
result.params, result.log_likelihood, result.aic, result.bic
result.params.nu              # Student-t DCC
result.Rt
result.corr(0, 1)
result.unconditional_corr
result.std_errors, result.std_errors_robust
result.scenarios(10_000, seed=7)
result.simulate(250, seed=7)
result.summary()
```

---

## Examples

### Fit GARCH(1,1) and inspect results

```python
import numpy as np
from scivol import GARCH, Normal

returns = np.random.randn(1000) * 0.01

spec = GARCH(1, 1) + Normal()
result = spec.fit(returns)

print(f"Persistence: {result.garch_params.persistence:.4f}")
print(f"Log-likelihood: {result.loglikelihood:.2f}")
result.summary()
```

### Compare MLE and sandwich standard errors

```python
from scivol import GARCH, Normal

spec = GARCH(1, 1) + Normal()
result = spec.fit(returns, method='qmle')

print("Parameter       MLE SE    Robust SE")
print("-" * 40)
for i, name in enumerate(['omega', 'alpha', 'beta']):
    print(f"{name:10}  {result.std_errors[i]:10.6f}  {result.std_errors_robust[i]:10.6f}")
```

### GJR-GARCH leverage effect

```python
from scivol import GJRGARCH, StudentT

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
from scivol import AutoVol, AutoDensity

spec = AutoVol() + AutoDensity()
result = spec.fit(returns, verbose_selection=True)

print(f"Best: {result.spec}")
result.selection_summary()
```

### Fit DCC and inspect correlations

```python
import numpy as np
import pandas as pd
from scivol import DCC, GARCH, Normal

rng = np.random.default_rng(42)
returns = rng.standard_normal(1000) * 0.01
returns_df = pd.DataFrame(
    {
        "stock": returns,
        "bond": returns * 0.3 + rng.standard_normal(len(returns)) * 0.005,
        "commodity": returns * -0.2 + rng.standard_normal(len(returns)) * 0.008,
    }
)

dcc = DCC(1, 1)
result = dcc.fit(returns_df, univariate_spec=GARCH(1, 1) + Normal())

stock_bond = result.corr("stock", "bond")
print(stock_bond.tail())
print(result.unconditional_corr)
```

### One-step-ahead variance forecast

```python
from scivol import GARCH, GJRGARCH, Normal
import numpy as np

# GARCH(1,1)
spec = GARCH(1, 1) + Normal()
result = spec.fit(returns)

gp = result.garch_params
h_next = gp.omega + gp.alpha[0] * returns[-1]**2 + gp.beta[0] * result.sigma2[-1]
print(f"GARCH forecast sigma: {np.sqrt(h_next):.6f}")

# GJR-GARCH(1,1)
spec_gjr = GJRGARCH(1, 1) + Normal()
result_gjr = spec_gjr.fit(returns)

gp = result_gjr.garch_params
indicator = 1.0 if returns[-1] < 0 else 0.0
h_next = (gp.omega
    + gp.alpha[0] * returns[-1]**2
    + gp.gamma[0] * indicator * returns[-1]**2
    + gp.beta[0] * result_gjr.sigma2[-1])
print(f"GJR forecast sigma:   {np.sqrt(h_next):.6f}")
```
