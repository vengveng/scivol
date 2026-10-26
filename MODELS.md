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

### Comparison samples and native restrictions

`ComparisonPlan(y, first_scored, x, variance_seed)` owns the input observations
and exogenous columns for one comparison. Build it once using the largest mean
lag in the search, or a larger declared hold-back. Every candidate scores exactly
`y[first_scored:]`. Earlier rows supply mean regressors only; they do not advance
the residual or volatility recursion. All dynamic candidates start those
recursions on the same row, with the same parameter-independent variance seed
and zero pre-sample residuals. The default seed is `mean(y^2)` over the entire
plan input span. An explicit positive finite seed is also supported. Internal
data scaling scales that seed exactly, without recomputing it from a sliced
sample. Exogenous reductions retain leading columns of the same plan design.

The constant-variance model uses `sigma2` on **every** scored row, including the
first. It shares scored rows and mean conditioning with the plan but does not
use its seed as a variance parameter. It remains a separate comparison root.
Zero-order dynamic models retain the dynamic initialization: their first row
uses the seed and subsequent rows follow their recursion. They are not
substituted for the constant-variance root. Single fits keep their existing
sample preparation by default.

`Model::restrictions()` lists exact children expressible as ordinary native
specifications. Each child runs its own compiled kernel. Its affine embedding,
tested coordinate rows, null values, restriction types, and identification
conditions come from the parent model. `embed` and `map_start` reject a map
outside the relevant domain. Warm starts do not fix the retained parameters.

| Parent | Immediate restriction | Tested directions |
|---|---|---|
| Any mean | Drop intercept, last AR, last MA, last exogenous column, or one HAR window | Interior equalities |
| GARCH | Drop last shock or last variance lag | Closed nonnegative boundaries |
| GJR | Drop last shock pair | Two closed boundaries, `c+ = alpha`, `c- = alpha + gamma` |
| GJR | Drop last variance lag | Closed nonnegative boundary |
| GJR | All `gamma = 0`, child GARCH | Interior equalities when common sign loadings are positive |
| EGARCH | Drop last alpha/gamma pair or last variance lag | Interior equalities |
| APARCH | Drop last shock/asymmetry pair | Boundary alpha and signed gamma; unidentified under the null |
| APARCH | Drop last variance lag | Closed nonnegative boundary |
| Skew-t | `lam = 0`, child Student-t | Interior equality |
| Student-t | `eta = 0`, child Normal | Closed nonnegative boundary |
| GED | `nu = 2`, child Normal | Interior equality |

Dropping the last remaining shock while retaining variance lags is listed as
unidentified under the null, including the variance-lag nuisance in a no-shock
GARCH or EGARCH. ARCH can reduce to a seeded zero-order dynamic model; this
edge has no unidentified variance-lag nuisance. APARCH
shock deletion removes an unidentified asymmetry even when other shocks remain.
These edges have no ordinary LR calibration. ARMA reductions require a minimal,
nonsingular representation and are intended for opt-in searches. Non-trailing
volatility lags, separate EGARCH alpha/gamma deletions, APARCH fixed power or
asymmetry, GJR one-sign faces and the skew-Gaussian face are not listed because
they lack the required child representation. GH density reductions are deferred
in this first version; GH closure limits have no ordinary finite restriction.

`restriction_test` requires certified MLE fits from the same immutable plan and
the parent observed information at the embedded child estimate. It profiles
all signed directions and nuisance before computing boundary weights. Interior
equalities use chi-square; closed coordinates use chi-bar-square, with the
interior count shifting its degrees. Weights for one through three boundaries
use closed forms. Larger blocks enumerate orthant faces using deterministic
Cholesky conditioning and adaptive cube subdivision. Independent blocks factor
before integration. The reported absolute error encloses the p-value error;
failure to reach `1e-6`, or an interval crossing the decision level, returns
`unresolved`. The subdivision budget is explicit. Retained boundary nuisance,
singular information, unidentified edges and QMLE do not receive an ordinary
LR p-value. Native resultant-rank checks also reject common-factor and shared
trailing-zero ARMA/GARCH representations whose finite seed transients could
hide asymptotic nonidentification. Enumeration is bounded to twelve boundary
directions; larger blocks report `unresolved`. A negative statistic requests
a parent refit; it is not clamped.

`restrict_fit` reuses an estimate only when a certified parent lies exactly on
the declared face and embedding the projected child reproduces that parent
point exactly. It evaluates the child with its own kernel and runs the common
finalizer, including the child's original-domain KKT check. Otherwise a new
fit is needed. Fit bindings release the GIL; separate Model/Jet instances and
their workspaces can be used by independent Python threads.

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

## Unconditional moments

`Model::unconditional_mean(theta)` returns `intercept / (1 - sum AR - sum HAR)`
for a stable implied AR polynomial, including zero-mean and constant-only models.
MA coefficients do not affect this mean. Exogenous-regressor means are not
specified, so models with exogenous regressors return NaN.

`Model::unconditional_variance(theta)` returns exact stationary `E[h]`:

- Constant variance: `sigma2`.
- GARCH/GJR: `omega / (1 - persistence)`, with GJR's density-weighted persistence.
- APARCH: the same expression at exactly `delta = 2`, with weights
  `E[(|z| - gamma_i z)^2]`. This is the GJR-equivalent variance. At other
  powers the exact variance is unavailable and the method returns NaN.
- EGARCH: let `a_k` and `b_k` be the coefficients of
  `(sum alpha_i L^i)/(1-sum beta_j L^j)` and
  `(sum gamma_i L^i)/(1-sum beta_j L^j)`, and `m = E|z|`. Then
  `E[h] = exp(omega/(1-sum beta)) * product_k exp(-a_k*m) M(a_k,b_k)`,
  where `M(a,b) = E exp(a|z|+bz)` is the native exponential moment.
  The omitted log product is bounded by `2e-12`, using a contracting power
  of the beta companion matrix and a finite exponential envelope moment.
  When `a_k + |b_k| <= 0` is preserved by nonnegative beta, a one-sided Taylor
  bound also permits polynomial tails. A divergent factor returns infinity.
  Student-t and skew-t with `eta > 0` have an infinite variance whenever a
  filtered shock has a positive tail slope, including the usual positive
  magnitude response. `exp(E[log h])` is not the unconditional variance.

NaN also reports an invalid or nonstationary recursion, failure to establish
the truncation bound within the numerical work budget, or a finite moment
outside double precision's range. The Python summary asks the common Model
interface for long-run variance for every family.

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
