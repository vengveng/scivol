# Volatility models

A volatility component defines the conditional variance $h_t=\mathrm{Var}(e_t\mid\mathcal F_{t-1})$ of the residual $e_t=y_t-m_t$. The recursions below are written in terms of $e_t$ and the standardized innovation $z_t=e_t/\sqrt{h_t}$, which has zero mean and unit variance. Every component takes the number of shock lags `p` and the number of variance lags `q`, each from 0 to 8, and a `stationarity` declaration that sets its parameter domain:

```text
GARCH(p=1, q=1, stationarity="covariance")
```

The default, `"covariance"`, is the textbook domain in which the variance process is covariance stationary. The [last section](#declaring-stationarity) describes the alternatives.

## Comparing two models on the same data

The example simulates a GJR-GARCH series with skewed Student-t innovations and fits both GJR-GARCH and EGARCH to it:

```python
from scivol import GJRGARCH, EGARCH, SkewT

true_model = GJRGARCH(1, 1) + SkewT()
y = true_model.simulate(2000, [0.05, 0.03, 0.1, 0.88, 0.12, -0.15], seed=11).data

gjr = true_model.fit(y)
egarch = (EGARCH(1, 1) + SkewT()).fit(y)
print(gjr.log_likelihood, gjr.persistence, gjr.unconditional_variance)
print(egarch.log_likelihood, egarch.persistence, egarch.unconditional_variance)
```

```text
-2802.2958522410854 0.9528537004160007 1.071142893790459
-2806.2780801843282 0.9556237462212616 inf
```

Both models have six parameters, and the GJR-GARCH model, which generated the data, has the higher likelihood. The two persistence measures are close, but they measure different things: for GJR-GARCH it is the density-weighted sum defined below, for EGARCH the sum of the beta coefficients of the log-variance recursion. EGARCH's unconditional variance is infinite here, because a Student-t innovation has no exponential moments. That is a property of the model, explained in the EGARCH section, and it is not a numerical failure.

## GARCH and ARCH

$$
h_t=\omega+\sum_{i=1}^{p}\alpha_i e_{t-i}^2+\sum_{j=1}^{q}\beta_jh_{t-j}
$$

`GARCH(p, q)` has parameters `omega`, `alpha[1]` through `alpha[p]` and `beta[1]` through `beta[q]`, in that order. Its default domain is

$$
\omega>0,\quad\alpha_i\ge0,\quad\beta_j\ge0,\quad
P=\sum_i\alpha_i+\sum_j\beta_j<1,
$$

and within it the unconditional variance is $E[h_t]=\omega/(1-P)$. `GARCH(p, 0)` is the ARCH model of [Engle (1982)](../references.md#engle-1982). GARCH is due to [Bollerslev (1986)](../references.md#bollerslev-1986), and [Bollerslev (1987)](../references.md#bollerslev-1987) combined it with Student-t innovations. Variance forecasts are exact at every horizon.

## GJR-GARCH

$$
h_t=\omega+\sum_{i=1}^{p}
[\alpha_i+\gamma_i I(e_{t-i}<0)]e_{t-i}^2
+\sum_{j=1}^{q}\beta_jh_{t-j}
$$

`GJRGARCH(p, q)` has parameters `omega`, then the blocks `alpha`, `gamma` and `beta`, each indexed by lag. The model is due to [Glosten, Jagannathan and Runkle (1993)](../references.md#glosten-jagannathan-runkle-1993). Its default domain is

$$
\omega>0,\quad\alpha_i\ge0,\quad\alpha_i+\gamma_i\ge0,\quad\beta_j\ge0,\qquad
P=\sum_i\alpha_i+\kappa\sum_i\gamma_i+\sum_j\beta_j<1,
$$

where $\kappa=E[z^2I(z<0)]$ is the share of the innovation variance that comes from negative values. A negative `gamma` is allowed as long as the response to negative shocks, $\alpha_i+\gamma_i$, stays nonnegative. For a symmetric density $\kappa=1/2$. For a skewed density scivol computes $\kappa$ from the shape parameters, so the persistence condition, the unconditional variance $E[h_t]=\omega/(1-P)$ and the variance forecasts all use the density actually fitted. Variance forecasts are exact at every horizon.

## EGARCH

$$
\log h_t=\omega+\sum_{i=1}^{p}\alpha_i(|z_{t-i}|-m_1)
+\sum_{i=1}^{p}\gamma_i z_{t-i}
+\sum_{j=1}^{q}\beta_j\log h_{t-j},\qquad m_1=E|z|
$$

`EGARCH(p, q)` has parameters `omega`, then the blocks `alpha`, `gamma` and `beta`. The model is due to [Nelson (1991)](../references.md#nelson-1991). Because the recursion is in logarithms, the coefficients need no sign restrictions. Under the default declaration the beta polynomial $1-\sum_j\beta_jL^j$ must be stable, which makes $\log h_t$ stationary. The centering constant $m_1$ is computed for the fitted density, so the density's shape parameters enter the variance path.

A stationary log variance does not by itself give a finite variance. Let $a_k$ and $b_k$ be the coefficients of the alpha and gamma polynomials after division by the beta polynomial. Then

$$
E[h_t]=\exp\!\left(\frac{\omega}{1-\sum_j\beta_j}\right)
\prod_{k\ge1}\exp(-a_km_1)\,M(a_k,b_k),\qquad M(a,b)=E[\exp(a|z|+bz)],
$$

which is finite only when every factor $M(a_k,b_k)$ is. Student-t and skewed-t innovations with `eta > 0` have no positive exponential moments, so with the usual positive `alpha` their EGARCH variance is infinite, as in the example above, and `unconditional_variance` reports `inf`. Normal, GED and generalized hyperbolic innovations have finite moments for slopes inside a range that depends on their shape. The same moments give the variance forecasts, which are exact at every horizon where they exist; a forecast that would need an infinite moment raises an error. The one-step forecast is always available, because it depends only on observed data.

## APARCH

The recursion is for a power $\delta$ of the conditional standard deviation $\sigma_t=\sqrt{h_t}$:

$$
\sigma_t^\delta=\omega+\sum_{i=1}^{p}\alpha_i
(|e_{t-i}|-\gamma_i e_{t-i})^\delta
+\sum_{j=1}^{q}\beta_j\sigma_{t-j}^\delta.
$$

`APARCH(p, q)` has parameters `omega`, the blocks `alpha`, `gamma` and `beta`, then `delta`. The model is due to [Ding, Granger and Engle (1993)](../references.md#ding-granger-engle-1993). Its default domain is

$$
\omega>0,\quad\alpha_i\ge0,\quad\beta_j\ge0,\quad|\gamma_i|<1,\quad\delta>0,\qquad
P=\sum_i\alpha_iE[(|z|-\gamma_i z)^\delta]+\sum_j\beta_j<1.
$$

The condition on $P$ makes $E[\sigma_t^\delta]$ finite. With Student-t or skewed-t innovations the power moment additionally requires $\delta\eta<1$. At general $\delta$ the persistence condition says nothing exact about $E[h_t]$, so `unconditional_variance` is reported only at $\delta=2$, where the recursion is a variance recursion, and is NaN otherwise. One-step variance forecasts are exact; later horizons average simulated paths.

## News impact curves

The news impact curve of [Engle and Ng (1993)](../references.md#engle-ng-1993) shows how the next variance responds to the latest residual $e_t$ when everything else is at its usual level. `result.news_impact(e)` evaluates it, continuing the example at the top of the page:

```python
import numpy as np

shocks = np.array([-3.0, -1.5, 0.0, 1.5, 3.0])
print(gjr.news_impact(shocks).round(3))
print(egarch.news_impact(shocks).round(3))
```

```text
[1.913 1.221 0.991 1.088 1.381]
[1.592 1.171 0.861 1.031 1.235]
```

Both models respond more to negative residuals than to positive ones of the same size. Every lagged variance is held at a level $v$, and the earlier residuals enter through their expectation given it: $e^2$ as $v$ and $e^2I(e<0)$ as $\kappa v$ in GJR-GARCH, $(|e|-\gamma_ie)^\delta$ as $v^{\delta/2}E[(|z|-\gamma_iz)^\delta]$ in APARCH, and the EGARCH shock terms as zero. The EGARCH residual is standardized as $z=e/\sqrt v$. By default $v$ is the stationary level of the recursion, the one a simulation starts from: $E[h_t]$ for GARCH and GJR-GARCH, $\exp(E\log h_t)$ for EGARCH and $(E\sigma_t^\delta)^{2/\delta}$ for APARCH. The EGARCH curve is lower at zero here because $\exp(E\log h_t)$ is below $E[h_t]$. `news_impact(e, variance=v)` holds the lagged variances at a given $v$ instead, which a nonstationary estimate needs, since it has no stationary level.

## Declaring stationarity

The `stationarity` argument is part of the model, because it sets which parameter values the likelihood is maximized over:

| Declaration | GARCH, GJR-GARCH and APARCH | EGARCH |
| --- | --- | --- |
| `"covariance"` (default) | Persistence $P<1$, and $\delta\eta<1$ for APARCH with a t density | Stable beta polynomial |
| `"integrated"` | Persistence $P\le1$, and $\delta\eta<1$ for APARCH with a t density | Not available |
| `"none"` | Only the positivity conditions | Any finite coefficients |

```python
from scivol import GARCH, GJRGARCH, EGARCH, StudentT, SkewT

integrated = GARCH(1, 1, stationarity="integrated") + StudentT()
unrestricted = GJRGARCH(1, 1, stationarity="none") + SkewT()
unrestricted_log_variance = EGARCH(1, 1, stationarity="none")
```

`"integrated"` allows persistence up to and including one, and estimates it; to fix persistence at one, as in the integrated GARCH model of [Engle and Bollerslev (1986)](../references.md#engle-bollerslev-1986), you would impose the restriction on the parameters yourself. Under the default declaration an estimate at persistence one is reported with the status `at_excluded_boundary`, as shown under [estimation](../estimation.md#what-success-means). Outside the covariance-stationary domain, `unconditional_variance` is `inf` for GARCH, GJR-GARCH and APARCH and NaN for an EGARCH model whose beta polynomial is unstable, and `simulate` refuses parameters for which no stationary starting point exists.

## Initialization and kinks

Every dynamic model starts its first scored variance at the mean square described in [mean models](mean.md#sample-and-initialization), and residuals before the sample are zero. For GARCH, GJR-GARCH and APARCH the missing variance lags are zero as well, while EGARCH sets missing log-variance lags to the logarithm of the starting value.

With an estimated mean, a residual can pass through exactly zero as the mean parameters change. EGARCH's absolute value then makes the likelihood non-differentiable at that point, APARCH with $\delta\le1$ has a cusp there, and Hansen's skewed t has no second derivative at the point where its two halves join. [Estimation](../estimation.md#likelihood-kinks) explains how scivol fits and reports such points.
