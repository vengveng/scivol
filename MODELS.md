# scivol models

Every combination of a mean, a volatility model and a density below is one
model with one exact likelihood, estimated jointly. Likelihoods, gradients,
Hessians, forecasts and simulations come from the native library; each model's
likelihood and derivatives are tested against an independent JAX reference.

## Parameters and domains

theta = [mean | volatility | density shapes]. The domain is enforced exactly by
the bounds and constraints (theta coordinates) and by the coordinate charts
(z coordinates); trial points outside it are rejected, never repaired.

| Component | Parameters | Domain |
|---|---|---|
| `ARMA(p, q)` | `const, ar[1..p], ma[1..q]` | stationary AR, invertible MA |
| `ARX(k)` | `const, ar[1..k], exog[1..]` | stationary AR |
| `HARX(windows)` | `const, har[w]..., exog[1..]` | stationary implied AR polynomial |
| none (mean only) | `sigma2` | `sigma2 > 0` |
| `GARCH(p, q)` | `omega, alpha[1..p], beta[1..q]` | `omega > 0`, `alpha, beta >= 0`, `sum alpha + sum beta < 1` |
| `GJRGARCH(p, q)` | `omega, alpha, gamma, beta` | `omega > 0`, `alpha >= 0`, `alpha + gamma >= 0`, `beta >= 0`, `sum alpha + kappa sum gamma + sum beta < 1` with `kappa = E[z^2 1(z<0)]` |
| `EGARCH(p, q)` | `omega, alpha, gamma, beta` | `1 - sum beta_j L^j` stable |
| `APARCH(p, q)` | `omega, alpha, gamma, beta, delta` | `omega > 0`, `alpha, beta >= 0`, `abs(gamma) < 1`, `delta > 0`, `sum alpha_i E[(abs(z) - gamma_i z)^delta] + sum beta < 1` |
| `Normal()` | none | |
| `StudentT()` | `eta` | `0 <= eta < 1/2` |
| `SkewT()` (Hansen 1994) | `eta, lam` | `0 <= eta < 1/2`, `abs(lam) < 1` |
| `GED()` | `nu` | `nu > 0` |
| `GeneralizedHyperbolic()` | `rho, zeta, lambda` | `abs(rho) < 1`, `zeta > 0` |
| `NormalInverseGaussian()` | `rho, zeta` | `abs(rho) < 1`, `zeta > 0` |
| `Hyperbolic()` | `rho, zeta` | `abs(rho) < 1`, `zeta > 0` |
| `VarianceGamma()` | `rho, lambda` | `abs(rho) < 1`, `lambda > 3/2` |

All densities are standardized to zero mean and unit variance. Orders are at
most 8 for `p`, `q` and the AR and MA parts; HAR windows at most 256.
For APARCH with StudentT or SkewT, the power moment also requires
`delta * eta < 1`; this constraint is inactive at `eta = 0`.

## Sample conventions

- The likelihood scores the rows from `max(hold_back, lags)` on (`lags`: the
  mean's largest lag). Earlier rows only supply lags.
- The first scored row has variance `mean(y^2)` over the rows passed to the
  model; pre-sample residuals are zero.
- Pre-sample variance lags of GARCH, GJR and APARCH are omitted (zero);
  EGARCH's pre-sample log variances equal the first row's, `log mean(y^2)`.
  Every likelihood is therefore equivariant to the scale of the data.
- The likelihood is the complete negative log-likelihood, constants included.

## Inference

- MLE: inverse observed information; the sandwich `H^-1 OPG H^-1` is reported
  alongside. Standard errors are withheld at the boundary of the domain or when
  the information cannot be inverted reliably.
- QMLE: Gaussian quasi-likelihood for the mean and volatility, sandwich
  covariance; with a shaped density, a staged shape fit with the two-step
  sandwich. Not offered for EGARCH with a shaped density.
- A fit is certified (`result.converged`) when the estimate is finite and
  feasible, the solver converged, and the projected-gradient (KKT) residual is
  at most `max(sqrt(accuracy), sqrt(eps))`.

## Forecasts

| | Exact | Simulated |
|---|---|---|
| Mean and its variance (psi weights) | every horizon | |
| GARCH, GJR variance | every horizon | |
| EGARCH variance | every horizon, through E exp(a abs(z) + b z); Student-t and skew-t permit positive tail exponential moments only at `eta = 0` | |
| APARCH variance | one step | beyond one step |
| Quantiles | one step | beyond one step |

## DCC

`DCC(p, q)` uses a Gaussian or Student-t copula. For each marginal's
standardized residual, fix `u[t,i] = F_i(eps[t,i])` using its fitted innovation
CDF. The copula scores are `x[t,i] = G_eta^{-1}(u[t,i])`, where `G_eta` is the
standardized Student-t CDF, or the Normal CDF for a Gaussian copula. The row
log-likelihood is `log g_{eta,R_t}(x_t) - sum_i log g_eta(x[t,i])`.
Normal marginals with a Gaussian copula preserve `x = eps` exactly.

Correlation targeting uses `Qbar = x'x / T`, `Q_0 = Qbar`, and the recursion
`Q_t = (1 - sum a - sum b) Qbar + sum_i a_i x_{t-i} x_{t-i}' + sum_j b_j Q_{t-j}`.
An explicit target stays fixed. For a Student-t copula, the estimated target,
shocks and scores depend on eta; the likelihood gradient and Hessian include
all these derivatives. Domain: `a, b >= 0`, `sum a + sum b < 1`, `0 <= eta < 1/2`.
The correlation stage's standard errors treat the marginal estimates and their
probabilities as given. They include the estimated target's eta dependence.
`fit_from_residuals` accepts one `InnovationDensity` per series through
`marginals`; its default is Normal marginals. `CCC` uses Gaussian scores and
`R = Qbar` normalized to a correlation matrix.

For StudentT, SkewT and the Student-t copula, `eta = 1/nu`. The closed
face `eta = 0` is the exact Normal limit, or Hansen's two-piece Normal limit
for SkewT. Summaries report derived `nu`, infinite at zero. The scalar eta
chart covers the interior; as with the simplex, excluded closed boundary faces
do not change `chart_covers_domain`.
