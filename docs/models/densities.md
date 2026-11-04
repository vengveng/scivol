# Innovation densities

A density component describes the standardized innovation $z_t=e_t/\sqrt{h_t}$. Every density is standardized to $E[z_t]=0$ and $E[z_t^2]=1$, so that $h_t$ is the conditional variance whatever the density, and each observation contributes

$$\ell_t=\log f(z_t)-\tfrac12\log h_t$$

to the log-likelihood, including every normalization constant. Shape parameters are constant over time and come last in the parameter vector. The constructors take no arguments; shapes are estimated, or supplied as part of the parameters for `fix` and `simulate`.

| Component | Shape parameters, in order | Domain |
| --- | --- | --- |
| `Normal()` | none | none |
| `StudentT()` | `eta` | $0\le\eta<1/2$ |
| `SkewT()` | `eta`, `lam` | $0\le\eta<1/2$, $\vert\lambda\vert<1$ |
| `GED()` | `nu` | $\nu>0$ |
| `GeneralizedHyperbolic()` | `rho`, `zeta`, `lambda` | $\vert\rho\vert<1$, $\zeta>0$, $\lambda$ real |
| `NormalInverseGaussian()` | `rho`, `zeta` | $\vert\rho\vert<1$, $\zeta>0$ |
| `Hyperbolic()` | `rho`, `zeta` | $\vert\rho\vert<1$, $\zeta>0$ |
| `VarianceGamma()` | `rho`, `lambda` | $\vert\rho\vert<1$, $\lambda>3/2$ |

Normal is the default when a specification has no density component.

## Reading fitted shapes

A fitted result collects the shapes by name in `result.shapes`: the estimated ones, those the density fixes, such as the NIG's `lambda = -1/2`, and derived ones. For the Student-t densities the derived `nu` is the degrees of freedom $1/\eta$; for GED, `nu` is the estimated exponent itself.

```python
from scivol import GJRGARCH, SkewT, GeneralizedHyperbolic, NormalInverseGaussian

true_model = GJRGARCH(1, 1) + SkewT()
y = true_model.simulate(2000, [0.05, 0.03, 0.1, 0.88, 0.12, -0.15], seed=11).data

skew = true_model.fit(y)
print(skew.shapes)

nig = (GJRGARCH(1, 1) + NormalInverseGaussian()).fit(y)
print(nig.shapes)
print(skew.log_likelihood, nig.log_likelihood)
```

```text
{'eta': 0.10725280236603035, 'lam': -0.12973590131998786, 'nu': 9.323765700659445}
{'rho': -0.19626034673602827, 'zeta': 3.256231711049524, 'lambda': -0.5}
-2802.2958522410854 -2803.2378632053415
```

The skewed-t estimates are close to the simulated `eta = 0.12` and `lam = -0.15`. The NIG density, which has the same number of parameters, fits almost as well, with negative skewness in `rho`.

The full generalized hyperbolic density on the same data runs to the edge of its domain. Its estimate has $\zeta$ at the smallest positive double, which is the variance-gamma end of the family, and scivol reports it as such:

```python
gh = (GJRGARCH(1, 1) + GeneralizedHyperbolic()).fit(y)
print(gh.success, gh.fit_info.inference_status, gh.fit_info.excluded_boundary)
```

```text
True at_excluded_boundary ('zeta > 0',)
```

## Normal

$$f(z)=(2\pi)^{-1/2}\exp(-z^2/2)$$

The Normal density has no shape parameters and has every polynomial and exponential moment.

## Student-t

For $\eta>0$, let $\nu=1/\eta>2$. The unit-variance Student-t density is

$$
f(z)=c_\nu\left(1+\frac{z^2}{\nu-2}\right)^{-(\nu+1)/2},\qquad
c_\nu=\frac{\Gamma((\nu+1)/2)}{\sqrt{\pi(\nu-2)}\,\Gamma(\nu/2)},
$$

and at $\eta=0$ it is exactly the Normal density, with `nu` reported as infinity. Parameterizing by $\eta$ makes the Normal a point of the domain rather than a limit that an optimizer can only approach, so a fit to Normal data can reach it, and a test of Normal against Student-t innovations has a well-defined null. An absolute moment of order $r$ exists when $r<\nu$, and positive exponential moments do not exist for any finite $\nu$. The GARCH-t model is due to [Bollerslev (1987)](../references.md#bollerslev-1987).

```python
from scivol import GARCH, StudentT

spec = GARCH() + StudentT()
# omega, alpha[1], beta[1], eta = 1/8, so nu = 8
sample = spec.simulate(1000, [0.1, 0.1, 0.8, 1/8], seed=10)
```

## Hansen's skewed t

The skewed t of [Hansen (1994)](../references.md#hansen-1994) joins two Student-t halves with different scales. With $c=c_\nu$ and $\lambda$ the parameter `lam`,

$$
a=4\lambda c\,\frac{\nu-2}{\nu-1},\qquad
b^2=1+3\lambda^2-a^2,\qquad
v(z)=\frac{bz+a}{1+\lambda\,\mathrm{sign}(bz+a)},
$$

and for $\eta>0$

$$f(z)=bc\left(1+\frac{v(z)^2}{\nu-2}\right)^{-(\nu+1)/2}.$$

`lam = 0` gives the symmetric Student-t, and negative values put more weight in the left tail. At $\eta=0$ the density becomes a two-piece Normal, with $c=(2\pi)^{-1/2}$, $a=4\lambda c$ and $f(z)=bc\exp[-v(z)^2/2]$, which is the Normal density when `lam = 0` as well. Moments exist under the same conditions as for the symmetric t. Where the two halves join, at $z=-a/b$, the density has a continuous first derivative but no second derivative. The skewed t in some other libraries, such as the [Fernández and Steel (1998)](../references.md#fernandez-steel-1998) form in `rugarch`, is a different distribution.

## GED

For the exponent $\nu>0$,

$$
s^2=\frac{\Gamma(1/\nu)}{\Gamma(3/\nu)},\qquad
f(z)=\frac{\nu}{2s\,\Gamma(1/\nu)}\exp\left[-\left|\frac zs\right|^\nu\right].
$$

`nu = 2` is the Normal density, smaller values give heavier tails and larger values lighter ones. [Nelson (1991)](../references.md#nelson-1991) introduced the GED for EGARCH. All polynomial moments exist, and the exponential moments needed by EGARCH exist for a range of slopes that depends on $\nu$. The density has a cusp at zero when $\nu\le1$, and no second derivative there when $\nu<2$.

## Generalized hyperbolic family

The generalized hyperbolic (GH) distribution of [Barndorff-Nielsen (1977)](../references.md#barndorff-nielsen-1977) is usually written with five parameters $(\alpha,\beta,\delta,\mu,\lambda)$. Two of them are fixed by the standardization, which leaves three shapes, and scivol estimates them as $\rho=\beta/\alpha$, which sets the skewness, $\zeta=\delta\sqrt{\alpha^2-\beta^2}$, which sets the tail weight, and the index $\lambda$. These $\alpha$, $\beta$ and $\delta$ are density parameters, unrelated to the volatility coefficients of the same names.

With $K_\lambda$ the modified Bessel function of the second kind, the standardization gives

$$
q=1-\rho^2,\quad
R=\zeta\frac{K_{\lambda+1}(\zeta)}{K_\lambda(\zeta)},\quad
V=\zeta\left[\frac{K_{\lambda+2}(\zeta)}{K_{\lambda+1}(\zeta)}
-\frac{K_{\lambda+1}(\zeta)}{K_\lambda(\zeta)}\right],
$$

$$
\alpha^2=\frac Rq\left(1+\frac{\rho^2V}{q}\right),\quad
\beta=\rho\alpha,\quad
\delta^2=\frac{\zeta^2}{\alpha^2q},\quad
\mu=-\frac{\rho R}{\alpha q}.
$$

With $c=z-\mu$, $r=\sqrt{\delta^2+c^2}$ and $\gamma=\sqrt{\alpha^2-\beta^2}$, the density is

$$
f(z)=\frac{(\gamma/\delta)^\lambda}{\sqrt{2\pi}\,K_\lambda(\delta\gamma)}
\left(\frac r\alpha\right)^{\lambda-1/2}
K_{\lambda-1/2}(\alpha r)\exp(\beta c).
$$

An equivalent description is a normal mixture. Let $U$ follow the generalized inverse Gaussian distribution with density proportional to $u^{\lambda-1}\exp[-\zeta(u+1/u)/2]$, with mean $m$ and variance $v$, let $N$ be an independent standard Normal, and let $D=mq+\rho^2\zeta v$. Then

$$z=\rho\sqrt{\zeta/D}\,(U-m)+\sqrt{qU/D}\,N,$$

which shows the role of $\rho$ directly: it scales the skewing term, and $\rho=0$ gives a symmetric density.

`NormalInverseGaussian()` fixes $\lambda=-1/2$ ([Barndorff-Nielsen, 1997](../references.md#barndorff-nielsen-1997)) and `Hyperbolic()` fixes $\lambda=1$ ([Eberlein and Keller, 1995](../references.md#eberlein-keller-1995)); both estimate $\rho$ and $\zeta$.

## Variance-gamma

The variance-gamma distribution of [Madan and Seneta (1990)](../references.md#madan-seneta-1990) and [Madan, Carr and Chang (1998)](../references.md#madan-carr-chang-1998) is the GH limit $\zeta\to0$ with $\lambda>0$. scivol writes it as a normal-gamma mixture. With

$$
\vartheta=\frac{\sqrt2\,\rho}{\sqrt{1-\rho^2}},\quad
s=\sqrt{\lambda(1+\vartheta^2)},\quad
U\sim\mathrm{Gamma}(\lambda,1),\quad N\sim N(0,1)
$$

independent,

$$z=\frac{\vartheta(U-\lambda)+\sqrt U\,N}{s},$$

which in the GH notation above has $\delta=0$, $\beta=\vartheta s$, $\alpha=s\sqrt{\vartheta^2+2}$, $\mu=-\vartheta\lambda/s$ and Bessel index $\lambda-1/2$. The free parameters are $\rho$ and $\lambda$. The domain $\lambda>3/2$ keeps the density twice differentiable at its center, which the likelihood's Hessian needs. For smaller $\lambda$ the second derivative is unbounded at the center, and for $\lambda\le1/2$ so is the density itself.

The GH, NIG, hyperbolic and variance-gamma densities have exponential tails, so all their polynomial moments exist, and their exponential moments exist for slopes in a bounded interval. Their EGARCH variance and APARCH persistence are therefore finite for moderate coefficients.

## Distributions

`result.innovations` is the fitted standardized distribution as an object, with its density, distribution and quantile functions and its exact moments. For the skewed-t fit above, its skewness and kurtosis are close to those of the standardized residuals:

```python
import numpy as np

law = skew.innovations
print(law)
print(round(law.skewness, 4), round(law.kurtosis, 4))
z = skew.std_resid
print(round(float(np.mean(z**3)), 4), round(float(np.mean(z**4)), 4))
print(law.ppf([0.01, 0.05]).round(4), round(law.cdf(-2.0), 5))
```

```text
Distribution(SkewT, eta=0.107253, lam=-0.129736, nu=9.32377)
-0.3229 4.2317
-0.3552 4.1154
[-2.6643 -1.6951] 0.0303
```

Each density component builds the distribution at shapes you choose, by name. A moment that diverges is infinite, as the fourth moment is for a Student-t with $\eta\ge1/4$:

```python
from scivol import StudentT

t5 = StudentT().distribution(eta=0.2)
print(t5.shapes, round(t5.kurtosis, 6), round(t5.isf(1e-6), 3))
print(StudentT().distribution(eta=0.3).kurtosis)
```

```text
{'eta': 0.2, 'nu': 5.0} 9.0 19.188
inf
```

The methods are `pdf`, `logpdf`, `cdf`, `sf`, `ppf`, `isf`, `moment(k)` and `rvs`, and the moments `mean`, `variance`, `skewness` and `kurtosis` (not excess kurtosis); the [API page](../api.md#innovation-distributions) lists them.

## Quantiles and forecasts

Each density also provides its distribution function, quantile function and moments. A one-step-ahead quantile of the observation is $m_{T+1}+\sqrt{h_{T+1}}\,F^{-1}(u)$; quantiles further ahead mix over future means and variances and are simulated. [Forecasting and simulation](../forecasting-and-simulation.md) covers both.
