# Mean models and constant variance

Every univariate model in scivol has the same observation equation,

$$
y_t=m_t+e_t,\qquad e_t=\sqrt{h_t}\,z_t,\qquad E[z_t]=0,\quad E[z_t^2]=1,
$$

where $m_t$ is the conditional mean, $h_t$ the conditional variance of the residual $e_t$, and $z_t$ an innovation drawn from the [chosen density](densities.md). This page covers the mean components, which define $m_t$, and the constant-variance model, which applies when no volatility component is given. Without a mean component, $m_t=0$.

The parameter vector is ordered `[mean | volatility | density shapes]`. Parameter names use one-based lags: `const`, `ar[1]`, `har[5]` for the five-day average and `ma[1]`; each regressor's coefficient has the regressor's name. The same names appear in the summary report, as the index of `result.params` and in the parameter mappings passed to `fix` or `simulate`.

There are two mean components, `ARMA` and `HAR`. Both take MA terms and regressors; they differ in how they parameterize the autoregressive part.

## ARMA

`ARMA(p=0, q=0, constant=True)` specifies

$$
m_t=c+\sum_{i=1}^{p}\phi_i y_{t-i}+x_t^{\mathsf T}\beta+\sum_{j=1}^{q}\vartheta_j e_{t-j},
$$

where the regressors $x_t$ are those passed to `fit` (see [Regressors](#regressors)); without them the term $x_t^{\mathsf T}\beta$ is absent. Its parameters are `const`, then `ar[1]` through `ar[p]`, then one coefficient per regressor, then `ma[1]` through `ma[q]`, and `constant=False` removes `const`. Both orders run from 0 to 8. `ARMA(0, 0)` is a constant mean and `ARMA(p)` an autoregression. The AR polynomial $1-\sum_i\phi_iL^i$ must be stationary and the MA polynomial $1+\sum_j\vartheta_jL^j$ invertible, which means that the roots of both lie outside the unit circle. The MA terms enter with a plus sign, which matters when you compare estimates with a library that uses the opposite convention.

```python
from scivol import ARMA, GARCH, StudentT

spec = ARMA(1, 1) + GARCH(1, 1) + StudentT()
```

The first $p$ observations supply lags, and residuals before the sample are zero. With no regressors the unconditional mean is $c/(1-\sum_i\phi_i)$, which the MA coefficients do not affect. Mean forecasts are exact at every horizon. The model is the ARMA model of [Box and Jenkins (1970)](../references.md#box-jenkins-1970). In unconstrained coordinates, scivol represents the stationary and invertible polynomials by their partial autocorrelations, which map the stationary region one-to-one onto a box ([Barndorff-Nielsen and Schou, 1973](../references.md#barndorff-nielsen-schou-1973)).

## HAR

`HAR(windows, q=0, constant=True)` regresses on averages of past observations over several windows:

$$
\overline y_{t-1}^{(w)}=\frac1w\sum_{j=1}^{w}y_{t-j},\qquad
m_t=c+\sum_{w\in\mathcal W}\phi_w\overline y_{t-1}^{(w)}+x_t^{\mathsf T}\beta+\sum_{j=1}^{q}\vartheta_j e_{t-j}.
$$

`windows` is an increasing sequence of distinct window lengths between 1 and 256. `HAR([1, 5, 22])` has parameters `const`, `har[1]`, `har[5]` and `har[22]`, followed by the regressors' coefficients and `ma[1]` through `ma[q]`.

The model is an autoregression of order $\max\mathcal W$ whose coefficient at lag $j$ is $\sum_{w\ge j}\phi_w/w$; this implied polynomial must be stationary. HAR is therefore an autoregression with fewer free coefficients, and AR(p) is HAR(1, 2, ..., p), so a mean has AR lags or HAR windows but not both. Without regressors the unconditional mean is $c/(1-\sum_w\phi_w)$, and mean forecasts are exact at every horizon. The window structure is the heterogeneous autoregressive (HAR) model of [Corsi (2009)](../references.md#corsi-2009). scivol applies it to whatever series you fit, so a realized-volatility HAR needs the realized measure as its input.

```python
from scivol import HAR, GARCH

spec = HAR([1, 5, 22]) + GARCH()
```

## Regressors

Regressors are data, so they are passed with the data: `fit(y, x=X)` adds the columns of `X` to the mean equation, and the same specification without `x` has no regressors. A DataFrame's columns name the coefficients, a named Series is one regressor of that name, and the columns of an array are named `x[1]`, `x[2]` and so on. The example simulates a series with a Monday effect and one more regressor, then fits it:

```python
import numpy as np
import pandas as pd
from scivol import ARMA, GARCH, StudentT

dates = pd.bdate_range("2015-01-01", periods=2000)
x = pd.DataFrame({"monday": (dates.dayofweek == 0).astype(float),
                  "vix": np.random.default_rng(7).standard_normal(2000)}, index=dates)

spec = ARMA(1, 1) + GARCH(1, 1) + StudentT()
truth = {"const": 0.02, "ar[1]": 0.3, "monday": -0.15, "vix": 0.2, "ma[1]": 0.2,
         "omega": 0.05, "alpha[1]": 0.08, "beta[1]": 0.9, "eta": 0.15}
y = pd.Series(spec.simulate(2000, truth, x=x, burn=0, seed=3).data, index=dates)

result = spec.fit(y, x=x)
print(result.params.round(3))
```

```text
parameter
const       0.004
ar[1]       0.303
monday     -0.179
vix         0.217
ma[1]       0.188
omega       0.045
alpha[1]    0.079
beta[1]     0.905
eta         0.136
Name: params, dtype: float64
```

The summary lists the regressors under the model line, so a fit that left them out is easy to spot. scivol does not model the regressors, so forecasts and simulations need their rows over the horizon, and the unconditional mean of a model with regressors is NaN. A DataFrame is matched by column name, so its columns may come in any order; an array is taken by position:

```python
future = pd.DataFrame({"vix": [1.5, 0.0, -0.5], "monday": [1.0, 0.0, 0.0]})
print(result.forecast(3, x=future).mean.round(4))
print(result.forecast(3, x=future[["monday", "vix"]].to_numpy()).mean.round(4))
```

```text
[-0.1602 -0.0448 -0.1183]
[-0.1602 -0.0448 -0.1183]
```

`x` may also name the layer its regressors belong to, as in `x={"mean": X}`, which is the same as `x=X`. A bare `x` goes to the only part of the model that takes regressors. At present that is always the mean: a volatility component takes no regressors, and `x={"volatility": X}` raises. A specification without a mean component also raises when given `x`.

The regressors enter the mean equation itself, as in the ARX model of the Python `arch` package. With AR terms, a change in $x_t$ therefore moves $y_t$ by $\beta$, $y_{t+1}$ by $\phi_1\beta$ and so on, and its long-run effect is $\beta/(1-\sum_i\phi_i)$. This is not the regression with ARMA errors, $y_t=x_t^{\mathsf T}\beta+u_t$ with ARMA disturbances $u_t$, that R's `arima(xreg=)` estimates. That model's $\beta$ is the contemporaneous effect only, and the two give different estimates once the mean has AR terms. Without AR terms they are the same model.

In a [model search](../selection.md), each regressor's coefficient is a restriction the search can test: a candidate drops a regressor by name and keeps the others.

## Constant variance

A specification without a volatility component has a constant conditional variance,

$$h_t=\sigma^2,\qquad \sigma^2>0,$$

and its parameter `sigma2` follows the mean block and precedes any density shapes. There is no separate constant-variance component; you omit the volatility component instead:

```python
from scivol import ARMA, Normal

spec = ARMA() + Normal()  # parameters: const, sigma2
```

Variance forecasts equal `sigma2` at every horizon. `GARCH(0, 0)` is a different model with the same parameter count, because it starts its recursion from the variance seed described below.

## Sample and initialization

The likelihood scores observations from row $\max(\text{hold\_back}, c_0)$ on, counting from zero, where $c_0$ is the number of lags the mean needs ($p$ for ARMA, $\max\mathcal W$ for HAR). Earlier rows supply those lags and nothing else. Dynamic volatility models start their first scored variance at the mean of $y_t^2$ over the scored rows and the $c_0$ lag rows before them, and residuals before the sample are zero. The fitted paths in a result cover the scored rows.

`hold_back` lets several models score the same rows, which their likelihoods need to be comparable. [Model selection](../selection.md) sets this up automatically for all candidates.
