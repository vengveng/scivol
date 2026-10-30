# Innovation densities

Every density describes $z_t=e_t/\sqrt{h_t}$, standardized so that $E[z_t]=0$ and $E[z_t^2]=1$. Its observation log density is

$$\ell_t=\log f(z_t)-\tfrac12\log h_t.$$

The likelihood includes every normalization constant. Shapes are constant across observations and follow the mean and volatility blocks in theta. Constructors take no shape arguments; supply shapes through fixed parameters, simulation parameters or estimation starts.

| Component | Shape names in theta order | Declared domain |
| --- | --- | --- |
| `Normal()` | None | No shapes |
| `StudentT()` | `eta` | $0\le\eta<1/2$ |
| `SkewT()` | `eta`, `lam` | $0\le\eta<1/2$, $\vert \mathrm{lam}\vert <1$ |
| `GED()` | `nu` | $\nu>0$ |
| `GeneralizedHyperbolic()` | `rho`, `zeta`, `lambda` | $\vert \rho\vert <1$, $\zeta>0$, $\lambda\in\mathbb R$ |
| `NormalInverseGaussian()` | `rho`, `zeta` | $\vert \rho\vert <1$, $\zeta>0$; fixed GH index $-1/2$ |
| `Hyperbolic()` | `rho`, `zeta` | $\vert \rho\vert <1$, $\zeta>0$; fixed GH index $1$ |
| `VarianceGamma()` | `rho`, `lambda` | $\vert \rho\vert <1$, $\lambda>3/2$ |

All estimated values must be finite. The GH name `lambda` and Hansen canonical name `lam` refer to different parameters. Default summaries display Hansen `lam` as `lambda`; its component dictionary still uses `lam`. `result.dist_params.shape_lambda` stores the GH/VG index; `result.dist_params.lam` stores Hansen skewness. `nu` in Student-t output is derived degrees of freedom, while GED's `nu` is an estimated exponent.

## Normal

$$f(z)=(2\pi)^{-1/2}\exp(-z^2/2).$$

Normal has no shape parameters. It is the default density when none is supplied. It has all polynomial and exponential moments used by the model interface.

## Student-t

For `eta>0`, set $\nu=1/\eta>2$. scivol uses the unit-variance density

$$
f(z)=c_\nu\left(1+\frac{z^2}{\nu-2}\right)^{-(\nu+1)/2},\qquad
c_\nu=\frac{\Gamma((\nu+1)/2)}{\sqrt{\pi(\nu-2)}\Gamma(\nu/2)}.
$$

At `eta=0`, $f$ is exactly Normal and derived `nu` is infinity. This is an admitted closed face, not a large finite degrees-of-freedom cap. A polynomial absolute moment of order $r$ requires $r<\nu$. Positive exponential tail moments diverge for finite nu. See [Bollerslev (1987)](../references.md#bollerslev-1987).

```python
from scivol import GARCH, StudentT
spec = GARCH() + StudentT()
# omega, alpha[1], beta[1], eta; nu=8
sample = spec.simulate(1000, [0.1, 0.1, 0.8, 1/8], seed=10)
```

## Hansen skew-t

Let $c=c_\nu$, $\lambda=\mathrm{lam}$,

$$
a=4\lambda c\frac{\nu-2}{\nu-1},\qquad
b^2=1+3\lambda^2-a^2,\qquad
v(z)=\frac{bz+a}{1+\lambda\operatorname{sign}(bz+a)}.
$$

For `eta>0`,

$$f(z)=bc\left(1+\frac{v(z)^2}{\nu-2}\right)^{-(\nu+1)/2}.$$

This is [Hansen's (1994)](../references.md#hansen-1994) standardized two-piece density. `lam=0` gives the symmetric Student-t. At `eta=0`, use $c=(2\pi)^{-1/2}$, $a=4\lambda c$, the same expression for $b$, and $f(z)=bc\exp[-v(z)^2/2]$. This is a two-piece Normal limit, generally skewed. It is Normal when `lam=0` as well.

The branch join is $z=-a/b$. With nonzero skewness the first derivative joins but the second derivatives need not exist there. Polynomial tail moments have the same existence threshold as the symmetric Student-t. GJR persistence uses the density's actual lower partial second moment. Other libraries' "skewed t" densities, for example the [Fernández and Steel (1998)](../references.md#fernandez-steel-1998) form used by `rugarch`, are different distributions.

## GED

For the reported exponent $\nu>0$, define

$$
s^2=\frac{\Gamma(1/\nu)}{\Gamma(3/\nu)},\qquad
f(z)=\frac{\nu}{2s\Gamma(1/\nu)}\exp\left[-\left|\frac zs\right|^\nu\right].
$$

`nu=2` is Normal. This exponent is not reciprocal Student-t degrees of freedom. The GED used with EGARCH is discussed by [Nelson (1991)](../references.md#nelson-1991). All polynomial moments exist. Exponential moments depend on the exponent and requested slopes; the native moment routine checks their existence.

At zero, raw first derivatives require `nu>1`, and second derivatives require `nu>=2`. Products needed only for variance or shape derivatives can exist when raw residual derivatives do not. scivol does not insert a finite slope at a cusp.

## Generalized hyperbolic parameterization

The public shapes are $(\rho,\zeta,\lambda)$, not $(\alpha,\beta,\delta,\mu,\lambda)$. Let $K_\lambda$ be the modified Bessel function of the second kind. The mapping implemented in `scivolapi/src/density/gh.hpp` is

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

Thus $\rho=\beta/\alpha$ and $\zeta=\delta\sqrt{\alpha^2-\beta^2}$. Location and scale are derived to give zero mean and unit variance. They are not additional free parameters. These density-level alpha, beta and delta must not be confused with volatility parameters.

For $c=z-\mu$, $r=\sqrt{\delta^2+c^2}$ and $\gamma=\sqrt{\alpha^2-\beta^2}$, the equivalent GH density is

$$
f(z)=\frac{(\gamma/\delta)^\lambda}{\sqrt{2\pi}\,K_\lambda(\delta\gamma)}
\left(\frac r\alpha\right)^{\lambda-1/2}
K_{\lambda-1/2}(\alpha r)\exp(\beta c).
$$

The implementation evaluates this expression with scaled Bessel functions and cancellation-free algebra. GH originates in [Barndorff-Nielsen (1977)](../references.md#barndorff-nielsen-1977).

An equivalent standardized mixture, also specified in `gh.hpp`, is useful for identifying the parameterization. Let $U\sim\operatorname{GIG}(\lambda,\zeta,\zeta)$ with density proportional to $u^{\lambda-1}\exp[-\zeta(u+1/u)/2]$ and let $N\sim N(0,1)$ be independent. Write $m=E[U]$, $v=\operatorname{Var}(U)$ and $D=mq+\rho^2\zeta v$. Then

$$z=\rho\sqrt{\zeta/D}(U-m)+\sqrt{qU/D}\,N.$$

It makes the standardization and the role of $\rho$ explicit: $\rho$ scales the skewing term, and $\rho = 0$ gives a symmetric density.

### NIG and Hyperbolic

`NormalInverseGaussian()` fixes $\lambda=-1/2$ in the same standardized GH mapping. See [Barndorff-Nielsen (1997)](../references.md#barndorff-nielsen-1997).

`Hyperbolic()` fixes $\lambda=1$. See [Eberlein and Keller (1995)](../references.md#eberlein-keller-1995) for the finance application. Neither constructor estimates the fixed index. Both estimate `rho, zeta` with the GH domain above.

## Variance-Gamma

scivol uses the standardized normal-gamma mixture and the GH zero-concentration limit. Let

$$
\vartheta=\frac{\sqrt2\rho}{\sqrt{1-\rho^2}},\quad
s=\sqrt{\lambda(1+\vartheta^2)},\quad
U\sim\operatorname{Gamma}(\lambda,\text{rate}=1),\quad N\sim N(0,1),
$$

independently. Then

$$z=\frac{\vartheta(U-\lambda)+\sqrt U\,N}{s}.$$

Equivalently, its GH-type density has

$$
\delta=0,\quad\beta=\vartheta s,\quad
\alpha=s\sqrt{\vartheta^2+2},\quad
\mu=-\vartheta\lambda/s,\quad\text{Bessel index }\lambda-1/2.
$$

The mixture equation defines the normalized density also at its center, where a direct Bessel expression needs a limit. The free parameters are `rho, lambda`; the library deliberately restricts `lambda>3/2` so the observed density Hessian is finite at the center. It does not expose the full positive-index VG family. See [Madan and Seneta (1990)](../references.md#madan-seneta-1990) and [Madan, Carr and Chang (1998)](../references.md#madan-carr-chang-1998).

Interior GH, NIG, Hyperbolic and VG densities have exponentially decreasing tails and all polynomial moments. Their exponential moments still have a finite strip of admissible slopes. Use the model's moment calculations when assessing EGARCH forecasts or APARCH persistence.

## Forecasts, simulation and initial conditions

Densities supply the standardized CDF, quantile and moments. One-step observation quantiles are $m+\sqrt h\,F^{-1}(u)$. Later quantiles mix over future volatility and mean paths and use simulation. Variance-forecast exactness is a property of the full volatility-density composition; see [Forecasting and simulation](../forecasting-and-simulation.md). A density adds no separate recursion initialization; it uses the mean and volatility sample conventions.
