# Mean models and constant variance

Every univariate model in scivol has the same observation equation,

$$
y_t=m_t+e_t,\qquad e_t=\sqrt{h_t}\,z_t,\qquad E[z_t]=0,\quad E[z_t^2]=1,
$$

where $m_t$ is the conditional mean, $h_t$ the conditional variance of the residual $e_t$, and $z_t$ an innovation drawn from the [chosen density](densities.md). This page covers the mean components, which define $m_t$, and the constant-variance model, which applies when no volatility component is given. Without a mean component, $m_t=0$.

The parameter vector is ordered `[mean | volatility | density shapes]`. Parameter names use one-based lags, such as `ar[1]`, and the summary report shows the conventional symbols instead: `phi[1]` for AR coefficients, `theta[1]` for MA coefficients and `x[1]` for regressor coefficients. Parameter mappings passed to `fix` or `simulate` accept the names `ar`, `ma` and `exog`.

## ARMA

`ARMA(p, q, constant=True)` specifies

$$
m_t=c+\sum_{i=1}^{p}\phi_i y_{t-i}+\sum_{j=1}^{q}\vartheta_j e_{t-j}.
$$

Its parameters are `const`, then `ar[1]` through `ar[p]`, then `ma[1]` through `ma[q]`, and `constant=False` removes `const`. Both orders run from 0 to 8. The AR polynomial $1-\sum_i\phi_iL^i$ must be stationary and the MA polynomial $1+\sum_j\vartheta_jL^j$ invertible, which means that the roots of both lie outside the unit circle. The MA terms enter with a plus sign, which matters when you compare estimates with a library that uses the opposite convention.

```python
from scivol import ARMA, GARCH, StudentT

spec = ARMA(1, 1) + GARCH(1, 1) + StudentT()
```

The first $p$ observations supply lags, and residuals before the sample are zero. With no regressors the unconditional mean is $c/(1-\sum_i\phi_i)$, which the MA coefficients do not affect. Mean forecasts are exact at every horizon. The model is the ARMA model of [Box and Jenkins (1970)](../references.md#box-jenkins-1970). In unconstrained coordinates, scivol represents the stationary and invertible polynomials by their partial autocorrelations, which map the stationary region one-to-one onto a box ([Barndorff-Nielsen and Schou, 1973](../references.md#barndorff-nielsen-schou-1973)).

## ARX

`ARX(lags=0, constant=True)` adds contemporaneous regressors $x_t$ to an autoregression of order $k$:

$$
m_t=c+\sum_{i=1}^{k}\phi_i y_{t-i}+x_t^{\mathsf T}\xi.
$$

The parameters are `const`, `ar[1]` through `ar[k]`, then `exog[1]` up to the number of regressor columns, which scivol takes from the `x` you pass to `fit`. The AR polynomial must be stationary, and the intercept and regressor coefficients are unrestricted. Forecasting requires the future regressor values as a `(horizon, n_exog)` array, since scivol does not model the regressors. For the same reason the unconditional mean of an ARX model with regressors is reported as NaN.

```python
import numpy as np
from scivol import ARX, GARCH

rng = np.random.default_rng(17)
y = rng.standard_normal(800)
x = rng.standard_normal((800, 2))
result = (ARX(1) + GARCH()).fit(y, x=x)
forecast = result.forecast(3, x=np.zeros((3, 2)))
```

## HARX

`HARX(horizons, constant=True)` regresses on averages of past observations over several windows:

$$
\overline y_{t-1}^{(w)}=\frac1w\sum_{j=1}^{w}y_{t-j},\qquad
m_t=c+\sum_{w\in\mathcal W}\phi_w\overline y_{t-1}^{(w)}+x_t^{\mathsf T}\xi.
$$

`horizons` is an increasing sequence of distinct window lengths between 1 and 256. `HARX([1, 5, 22])` has parameters `const`, `har[1]`, `har[5]` and `har[22]`, followed by `exog[1]` and so on when regressors are given. Regressors work as in ARX.

The model is an autoregression of order $\max\mathcal W$ whose coefficient at lag $j$ is $\sum_{w\ge j}\phi_w/w$; this implied polynomial must be stationary. Without regressors the unconditional mean is $c/(1-\sum_w\phi_w)$, and mean forecasts are exact at every horizon. The window structure is the heterogeneous autoregressive (HAR) model of [Corsi (2009)](../references.md#corsi-2009). scivol applies it to whatever series you fit, so a realized-volatility HAR needs the realized measure as its input.

```python
from scivol import HARX, GARCH

spec = HARX([1, 5, 22]) + GARCH()
```

## Constant variance

A specification without a volatility component has a constant conditional variance,

$$h_t=\sigma^2,\qquad \sigma^2>0,$$

and its parameter `sigma2` follows the mean block and precedes any density shapes. There is no separate constant-variance component; you omit the volatility component instead:

```python
from scivol import ARX, Normal

spec = ARX(0) + Normal()  # parameters: const, sigma2
```

Variance forecasts equal `sigma2` at every horizon. `GARCH(0, 0)` is a different model with the same parameter count, because it starts its recursion from the variance seed described below.

## Sample and initialization

The likelihood scores observations from row $\max(\text{hold\_back}, c_0)$ on, counting from zero, where $c_0$ is the number of lags the mean needs ($p$ for ARMA, $k$ for ARX, $\max\mathcal W$ for HARX). Earlier rows supply those lags and nothing else. Dynamic volatility models start their first scored variance at the mean of $y_t^2$ over the scored rows and the $c_0$ lag rows before them, and residuals before the sample are zero. The fitted paths in a result cover the scored rows.

`hold_back` lets several models score the same rows, which their likelihoods need to be comparable. [Model selection](../selection.md) sets this up automatically for all candidates.
