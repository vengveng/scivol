# Mean models and constant variance

## Common observation equation

For every univariate composition,

$$
y_t=m_t+e_t,\qquad e_t=\sqrt{h_t}z_t,\qquad E[z_t]=0,\quad E[z_t^2]=1.
$$

Here $m_t$ is the conditional mean and $h_t$ the conditional residual variance. Innovations use the [selected density](densities.md). With no mean component, $m_t=0$. An intercept `const` is a conditional-mean intercept, not the unconditional mean.

The full parameter vector is `[mean | volatility | density shapes]`. Canonical native names use one-based lags, for example `ar[1]`. Default summaries display `phi[1]` for AR, `theta[1]` for MA, and `x[1]` for exogenous coefficients; parameter mappings and component dictionaries retain `ar`, `ma` and `exog`. HAR coefficients use window lengths, for example `har[22]`. Component `unpack()` dictionaries use list keys `ar`, `ma`, `har` and `exog`.

## ARMA

```python
from scivol import ARMA, GARCH, StudentT
spec = ARMA(1, 1, constant=True) + GARCH() + StudentT()
```

`ARMA(p, q, *, constant=True)` specifies

$$
m_t=c+\sum_{i=1}^{p}\phi_i y_{t-i}+\sum_{j=1}^{q}\vartheta_j e_{t-j}.
$$

| Reported parameters, in order | Declared domain |
| --- | --- |
| `const` when included, `ar[1]` through `ar[p]`, `ma[1]` through `ma[q]` | Finite coefficients; stationary AR and invertible MA polynomials |

The AR polynomial is $1-\sum_i\phi_i L^i$ and the MA polynomial is $1+\sum_j\vartheta_jL^j$. Their roots must lie outside the unit circle. The plus sign on the MA contribution matters when transferring parameters from another implementation. Orders are integers from 0 through 8. `constant=False` removes the intercept parameter.

The required conditioning span is the largest observed-data lag, $p$. Pre-sample residuals are zero, including the missing MA lags. With no exogenous variables the stationary mean is $c/(1-\sum_i\phi_i)$; MA coefficients do not change it. Exact conditional-mean forecasts and their psi-weight variance propagation are supported at every horizon, subject to the residual variance forecasts existing.

The standard reference is [Box and Jenkins (1970)](../references.md#box-jenkins-1970). In z coordinates, stationary and invertible polynomials are parameterized by their partial autocorrelations ([Barndorff-Nielsen and Schou, 1973](../references.md#barndorff-nielsen-schou-1973)), which map the stationary region one-to-one onto a box.

## ARX

`ARX(lags=0, *, constant=True)` specifies

$$
m_t=c+\sum_{i=1}^{k}\phi_i y_{t-i}+x_t^{\mathsf T}\xi.
$$

`x_t` contains contemporaneous, user-supplied regressors. The parameter order is `const` when included, `ar[1]` through `ar[k]`, then `exog[1]` through the number of columns. The AR polynomial must be stationary. Intercept and exogenous coefficients are unrestricted finite values. `k` is from 0 through 8; `constant=False` removes `const`.

The required mean conditioning is $k$. Future mean forecasts require a `(horizon, n_exog)` array of future regressors. scivol does not forecast those regressors. The unconditional mean is unavailable, reported as NaN, if exogenous means have not been specified, as is the case for this API.

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

`HARX(horizons, *, constant=True)` uses lag averages:

$$
\overline y_{t-1}^{(w)}=\frac1w\sum_{j=1}^{w}y_{t-j},\qquad
m_t=c+\sum_{w\in\mathcal W}\phi_w\overline y_{t-1}^{(w)}+x_t^{\mathsf T}\xi.
$$

`horizons` is a nonempty, increasing sequence of unique positive integer windows, each at most 256. For `HARX([1, 5, 22])`, names are `const`, `har[1]`, `har[5]`, `har[22]`, then `exog[1]`, etc. Omitting the constant removes that parameter. Exogenous columns have the same input and forecast requirements as ARX.

The implied AR coefficient at lag $j$ is $\sum_{w\ge j}\phi_w/w$. The implied AR polynomial must be stationary. The mean's conditioning span is $\max\mathcal W$. The unconditional mean, when no regressors are present, is $c/(1-\sum_w\phi_w)$. Conditional means and psi-weight propagation are exact at every horizon given future regressors and the necessary residual variances.

The lag averages follow the heterogeneous autoregressive (HAR) model of [Corsi (2009)](../references.md#corsi-2009). scivol applies them to whatever series you fit; it does not compute realized volatility from intraday data.

```python
from scivol import HARX, GARCH
spec = HARX([1, 5, 22]) + GARCH()
```

## Constant variance

A specification with no volatility component estimates

$$h_t=\sigma^2,\qquad \sigma^2>0.$$

The parameter `sigma2` follows the mean block and precedes the density shapes. This variance applies to every scored row, including the first. There is no public constant-variance component constructor; omit the volatility component:

```python
from scivol import ARX, Normal
spec = ARX(0) + Normal()  # const, sigma2
```

Residual variance forecasts equal `sigma2` at every horizon. A zero-order dynamic model such as `GARCH(0, 0)` retains dynamic initialization and is a different finite-sample likelihood.

## Sample and initialization

If $c_0$ is the mean conditioning span, the original first scored row is `max(hold_back, c_0)`, using zero-based indexing. Earlier rows provide mean lags. They do not advance the residual or volatility recursion. For an ordinary Python fit, rows before `max(hold_back-c_0, 0)` are removed before native preparation. Dynamic variance initialization is the mean square of the remaining native input span, including its conditioning rows. [Selection](../selection.md) instead gives all candidates a common comparison span and variance seed.

Dynamic models start the first scored variance at this seed. Pre-sample residuals are zero. Missing GARCH/GJR/APARCH variance lags contribute zero; EGARCH pre-sample log variances equal the log seed. Returned paths cover the scored rows.
