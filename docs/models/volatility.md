# Volatility models

Throughout, $e_t = y_t - m_t$ is the residual, $h_t = \operatorname{Var}(e_t \mid \mathcal F_{t-1})$ its conditional variance, and $z_t = e_t/\sqrt{h_t}$ the standardized innovation, with zero mean and unit variance. In every component, `p` is the number of shock lags and `q` the number of variance lags; both can be 0 to 8.

Each component is constructed as `Family(p=1, q=1, stationarity="covariance")`. The `stationarity` declaration sets the parameter domain of the model (see [declaring stationarity](#declaring-stationarity)); it is part of the model, not an optimizer setting.

## GARCH and ARCH

$$
h_t=\omega+\sum_{i=1}^{p}\alpha_i e_{t-i}^2+\sum_{j=1}^{q}\beta_jh_{t-j}.
$$

`GARCH(p, q)` reports `omega`, `alpha[1]` through `alpha[p]`, then `beta[1]` through `beta[q]`. The default domain is

$$
\omega>0,\quad\alpha_i\ge0,\quad\beta_j\ge0,\quad
P=\sum_i\alpha_i+\sum_j\beta_j<1.
$$

`GARCH(p, 0)` is the ARCH model. ARCH was introduced by [Engle (1982)](../references.md#engle-1982) and GARCH by [Bollerslev (1986)](../references.md#bollerslev-1986); [Bollerslev (1987)](../references.md#bollerslev-1987) combined GARCH with Student-t innovations.

For $P<1$, $E[h_t]=\omega/(1-P)$. Residual variance forecasts are exact at every horizon, replacing future squared shocks by their conditional variances.

## GJR-GARCH

$$
h_t=\omega+\sum_{i=1}^{p}
[\alpha_i+\gamma_i I(e_{t-i}<0)]e_{t-i}^2
+\sum_{j=1}^{q}\beta_jh_{t-j}.
$$

`GJRGARCH(p, q)` reports `omega`, the `alpha` block, the `gamma` block, then the `beta` block, each indexed by lag. The domain is

$$
\omega>0,\quad\alpha_i\ge0,\quad\alpha_i+\gamma_i\ge0,\quad\beta_j\ge0,
$$

$$
P=\sum_i\alpha_i+\kappa\sum_i\gamma_i+\sum_j\beta_j<1,
\qquad \kappa=E[z^2I(z<0)].
$$

A negative `gamma` is allowed as long as the response to negative shocks, $\alpha_i + \gamma_i$, stays nonnegative. For symmetric densities $\kappa = 1/2$; for skewed densities scivol computes $\kappa$ from the density's shape parameters and includes its derivatives, rather than assuming one half.

For $P<1$, $E[h_t]=\omega/(1-P)$. Variance forecasts are exact at every horizon using the same density-dependent weight. The original model is [Glosten, Jagannathan and Runkle (1993)](../references.md#glosten-jagannathan-runkle-1993).

## EGARCH

$$
\log h_t=\omega+\sum_{i=1}^{p}\alpha_i(|z_{t-i}|-m_1)
+\sum_{i=1}^{p}\gamma_i z_{t-i}
+\sum_{j=1}^{q}\beta_j\log h_{t-j},\qquad m_1=E|z|.
$$

`EGARCH(p, q)` reports `omega`, `alpha`, `gamma`, `beta` in blocks. All coefficients are unrestricted finite values except that the beta polynomial $1-\sum_j\beta_jL^j$ must be stable under the default declaration. In particular, alpha and gamma have no nonnegativity restriction. `m_1` is computed for the fitted density, so density shapes enter the variance path. The model and the GED application originate in [Nelson (1991)](../references.md#nelson-1991).

The declaration named `"covariance"` imposes stability of the log-variance recursion. It does not by itself guarantee a finite $E[h_t]$. In particular, Student-t and skew-t innovations with `eta>0` have no positive tail exponential moments. The usual positive magnitude response can therefore yield infinite unconditional variance despite stable beta coefficients.

EGARCH residual variance forecasts use $M(a,b)=E[\exp(a|z|+bz)]$. They are exact at every horizon where these moments are finite and numerically representable. A divergent required moment produces an explicit failure rather than a finite approximation. One-step variance is determined by the observed history.

For a stable recursion, let $a_k$ and $b_k$ be the filtered coefficients of the alpha and gamma polynomials through the inverse beta polynomial. Then

$$
E[h_t]=\exp\!\left(\frac{\omega}{1-\sum_j\beta_j}\right)
\prod_{k\ge1}\{\exp(-a_km_1)M(a_k,b_k)\}.
$$

The native calculation bounds the omitted log product by `2e-12`. A divergent factor returns infinity; failure to establish the bound or represent the result returns NaN. $\exp(E[\log h_t])$ is not $E[h_t]$.

## APARCH

Set $\sigma_t=\sqrt{h_t}$. The recursion is

$$
\sigma_t^\delta=\omega+\sum_{i=1}^{p}\alpha_i
(|e_{t-i}|-\gamma_i e_{t-i})^\delta
+\sum_{j=1}^{q}\beta_j\sigma_{t-j}^\delta.
$$

`APARCH(p, q)` reports `omega`, `alpha`, `gamma`, `beta`, then `delta`. The default domain is

$$
\omega>0,\quad\alpha_i\ge0,\quad\beta_j\ge0,\quad|\gamma_i|<1,\quad\delta>0,
$$

$$
P=\sum_i\alpha_iE[(|z|-\gamma_i z)^\delta]+\sum_j\beta_j<1.
$$

Student-t and skew-t additionally require $\delta\eta<1$ for the power moment under the covariance and integrated declarations. At `eta=0` this restriction is inactive. The original paper is [Ding, Granger and Engle (1993)](../references.md#ding-granger-engle-1993).

This persistence condition controls the scale-power moment. At general delta it does not give $E[h_t]=\omega/(1-P)$. The exact unconditional residual variance is supplied at exactly `delta=2`, when the recursion becomes a variance recursion with the corresponding power-moment weights. At other powers the unconditional variance method returns NaN. One-step variance forecasts are exact; later variance forecasts average simulated paths.

## Declaring stationarity

| Declaration | GARCH, GJR and APARCH | EGARCH |
| --- | --- | --- |
| `"covariance"` | The persistence above is strictly below one | Stable beta polynomial |
| `"integrated"` | Persistence at most one; equality is admitted | Raises; unavailable |
| `"none"` | Positivity conditions only; no persistence or APARCH power-moment constraint | Any finite beta coefficients |

`"integrated"` admits equality; it does not impose persistence equal to one. Persistence one is a closed face under this declaration and an excluded boundary under `"covariance"`. Integrated GARCH is discussed by [Engle and Bollerslev (1986)](../references.md#engle-bollerslev-1986).

```python
from scivol import GARCH, GJRGARCH, EGARCH, StudentT, SkewT

integrated_domain = GARCH(1, 1, stationarity="integrated") + StudentT()
unrestricted_persistence = GJRGARCH(1, 1, stationarity="none") + SkewT()
unrestricted_log_variance = EGARCH(1, 1, stationarity="none")
```

A nonstationary or divergent unconditional moment must not be interpreted as a finite long-run variance. The native interface returns infinity for an established divergent moment and NaN when unavailable or invalid. New-series simulation needs a stationary initialization; a fit admitted by `"none"` need not be simulatable by that workflow.

## Initialization and nonsmooth points

Every dynamic model initializes its first scored variance at the sample mean-square seed described in [Mean models](mean.md#sample-and-initialization). Missing pre-sample shock terms are zero. Missing variance or scale lags are zero for GARCH, GJR and APARCH. EGARCH instead uses the log seed for missing log-variance lags.

With an estimated mean, zero residuals can create likelihood kinks through EGARCH's absolute value. APARCH powers and GED densities can create residual cusps. Hansen skew-t can lack second derivatives at its branch join. These are model properties; see [Estimation](../estimation.md#likelihood-kinks) for certification limits and withheld derivatives.
