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
`unresolved`. The subdivision budget is explicit. Retained boundary nuisance
uses the nested-cone calibration described below. Singular information,
unidentified edges and QMLE do not receive an ordinary LR p-value. Native resultant-rank checks also reject common-factor and shared
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
- A fit is certified (`result.success`) when the estimate is finite and
  feasible, the solver converged, and the projected-gradient (KKT) residual is
  at most `max(sqrt(accuracy), sqrt(eps))`.

The model capability `likelihood_can_have_gradient_kinks` tells optimizers whether
moving residuals can cross a face with a nondifferentiable first derivative.
`residual_kink_decomposition` exposes those residual faces through the jet.
`Model::residual_kink_order(theta, future_rows)` distinguishes finite EGARCH
absolute-value kinks from non-Lipschitz power cusps and ignores unused lag terms.
The theta-only `KINKS` request returns zero-based sample row indices, residual
normals and curvatures, and the lifted derivatives for
`F(theta) = G(theta, {|e_t(theta)|})`. These outputs are computed separately;
ordinary smooth requests keep their existing observation loops.

A residual is roundoff-active when
`abs(e_t) <= 64 * eps * (abs(y_t) + sum abs(r_tj*b_j) + sum abs(ma_j*e_(t-j)))`.
This bounds cancellation in the actual residual subtraction, in its own units.
It is independent of solver accuracy and sample size. The branch calculations
take the residual's limiting value zero; the reported likelihood always comes
from the original parameter point. Ordinary pointwise derivatives are invalid
at a genuine kink and are withheld when a requested decomposition identifies
an unresolved residual sign.

After vanishing steps or a failed smooth refinement, Newton proposes faces
within `sqrt(eps)` on the same scale, retracts onto their equalities, and refines
the original likelihood there. This identification radius never defines an
active certificate or modifies the likelihood. With residual normals `a_t`,
the finalizer minimizes the norm of
`G_theta + sum mu_t*a_t + domain_normal`, subject to `-g_t <= mu_t <= g_t`,
where `g_t = dG/d|e_t|`. The equality Newton system includes residual curvature
in its Lagrangian Hessian. An out-of-interval multiplier releases the face.
Negative `g_t` cannot certify a minimum. Genuine unresolved cases retain
`stalled`; smooth constrained models retain their full refinement budget.
This follows the active-manifold approach of
[Lewis (2002)](https://people.orie.cornell.edu/aslewis/publications/03-active.pdf).

Certified kink fits report `fit_info.kink_point`, `active_kink_rows`, and
`inference_status="at_kink"`. Their ordinary gradient, Hessian and standard
errors are withheld. APARCH with estimated `delta <= 1` and GED with estimated
`nu <= 1` can reach residual cusps; their finite Clarke/manifold certificate is
unsupported. Below one, the one-sided slopes diverge. At one, allowing the
estimated power to vary below one still prevents local Lipschitz continuity.
Neither case is replaced by a finite artificial slope.

Restriction LR tests involving a kink-point fit return `unsupported`. A finite
sample kink alone does not prove a nonstandard asymptotic LR law: a Wilks law
can hold for nonsmooth likelihoods under additional local asymptotic normality
conditions. Those conditions and a calibration for these joint recursive models
are not established by the library's current information/cone calculation.

The automatic z-first path gives its first theta refinement at most 100 steps.
If the resulting estimate is uncertified, it restarts theta-space SLSQP and
Newton from the model's own start, within the original total iteration budget.
Every restart is named in `fit_info.optimizer_phases`. Explicit theta and z
paths retain their separate formulations. Zero-order starts set the intercept
from the persistence of the lags actually present.

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

## Automatic selection

`Search` declares component generators; native restrictions determine the
permitted downward closure. `GeneralToSpecific` uses certified MLE fits, Holm
adequacy checks, multipath LR reductions and joint backtests against each root.
It ranks terminals by free parameters and then BIC, or explicitly by IC alone.
Interior restrictions use chi-square laws; supported closed faces use native
chi-bar-square weights after nuisance profiling. Unidentified, singular,
retained-boundary and unresolved edges cannot authorize a reduction. QMLE LR
selection is rejected pending its own calibration.

The required adequacy family contains standardized-residual Ljung-Box at
`min(10, floor(n_eff/5))` and `min(20, floor(n_eff/5))`, with the mean model's
residual degrees of freedom; `absolute_li_mak` at the same deduplicated lags;
and `absolute_sign_bias`, the joint sign/size-bias test on absolute standardized
residuals. These absolute tests always apply, independently of the fitted
innovation shape. Holm combines the five checks at `diagnostic_level`. A known
Holm rejection establishes inadequacy even if another check is unavailable;
unavailable required checks otherwise leave adequacy unresolved. Berkowitz and
DGT on fitted in-sample PITs are reported with `calibrated=False` and do not
constrain adequacy.

For the absolute tests, let $\mu=E|z|$, $u_t=|z_t|-\mu$,
$s=E[\operatorname{sign}(z)]$, $a_t=(\partial e_t/\partial\theta)/\sqrt{h_t}$
and $b_t=\partial\log h_t/\partial\theta$. The predictable instruments $r_t$
are the centered lagged absolute residuals, or the centered Engle-Ng vector
$(I(z_{t-1}<0),z_{t-1}I(z_{t-1}<0),z_{t-1}I(z_{t-1}\ge0))$. Under the iid null,

$$
D=E\!\left[r_t\{s a_t-\tfrac12\mu b_t\}^{\mathsf T}\right],\qquad
w_t=u_t r_t-D A^{-1}\psi_t,\qquad
Q=n\bar m^{\mathsf T}\widehat\Sigma^{-1}\bar m\ \Rightarrow\ \chi^2_q,
$$

where $\bar m=n^{-1}\sum_t u_t r_t$, $\psi_t$ is the full joint negative-log
likelihood score, $A=E[\partial\psi_t/\partial\theta^{\mathsf T}]$, and
$\widehat\Sigma$ is the centered empirical covariance of $w_t$. The observed
mean Hessian estimates $A$; active closed-face normal coordinates are held
fixed. The estimator expansion is $\sqrt n(\hat\theta-\theta_0)
=-A^{-1}n^{-1/2}\sum_t\psi_t+o_p(1)$. Differentiating the lagged instrument
contributes zero because the current $u_t$ has conditional mean zero. Sample
centering likewise contributes zero to the population derivative. In
particular, skew innovations require the $s a_t$ mean term. Shape scores stay
in the joint information even when the residual path has no shape sensitivity.
The covariance retains the moment-score cross terms rather than treating fitted
residuals as observed innovations.

Only $E z^2=1$ is needed for these innovation moments: $\mu\le1$,
$\operatorname{Var}|z|=1-\mu^2$, $|\operatorname{sign}(z)|\le1$, and every
sign/size instrument has finite second moment. Independence gives
$E[(u_t u_{t-k})^2]=(1-\mu^2)^2$, and $E[u_t^2r_t r_t^{\mathsf T}]
=(1-\mu^2)E[r_t r_t^{\mathsf T}]$. The sample covariance therefore estimates
finite quantities without requiring $E z^4$. The model supplies $\mu$ through
`innovation_power_moment(theta, 0, 1)` and $s$ through its innovation CDF at
zero. These moments exist for Student-t and Skew-t throughout $\nu>2$, including
$2<\nu\le4$. Normal and GED have all polynomial moments; the declared interior
GH, NIG, Hyperbolic and Variance-Gamma domains have exponentially decreasing
tails and also have all polynomial moments. Regular identification, finite
square-integrable joint scores and path sensitivities, and an invertible
conditional information matrix remain estimator assumptions. They do not
follow from the innovation second moment alone. Gaussian QMLE still needs a
finite fourth moment for its own estimator influence function.

The absolute-residual precedent is [Li and Li (2005)](https://doi.org/10.1093/biomet/92.3.691),
which uses LAD estimation. The correction above uses scivol's joint MLE or
Gaussian QMLE influence function. Asymmetric power-model portmanteau theory is
discussed by [Carbon and Francq (2011)](https://www.stat.tugraz.at/AJS/ausg111%2B2/111%2B2Francq.pdf);
their squared-residual QMLE assumptions include a fourth innovation moment.
Finite-second-moment asymptotics can converge slowly near $\nu=2$, so these laws
do not establish accurate finite-sample size uniformly over the density domain.

The standalone squared `li_mak`, `sign_bias` and `arch_lm` statistics remain
available. Model-backed squared diagnostics report `fourth_moment_unavailable`
where necessary; array-only auxiliary regressions require their caller to
justify the innovation moments. Constant variance uses the same absolute
adequacy family. Its absolute ARCH-LM auxiliary regression has the same lagged
moments and limiting Wald test as `absolute_li_mak`, so adding it to Holm would
repeat a test. Its volatility-GUM encompassing test remains unavailable because
variance-lag parameters are unidentified under no ARCH.

`Model.at_open_bound(theta)` identifies floating-point endpoints of open
scalar and order-one stability bounds. Ordinary estimated-residual diagnostic
laws return `at_boundary` there. For example, APARCH at
$\gamma=\operatorname{nextafter}(1,0)$ approaches the excluded leverage endpoint.
For powers below one, derivatives of $(1-\gamma)^\delta$ diverge there; an
indefinite observed Hessian is not evidence of weak identification of $\delta$.
A generalized inverse cannot give this excluded endpoint an interior MLE law.
Genuine singular conditional information elsewhere remains explicit.

When every diagnostically evaluated root rejects, each with a residual
Ljung-Box rejection after Holm, GETS asks each mean generator to `widen` its
AR order by one. ARX preserves its intercept and exogenous columns; ARMA
preserves its MA order. Uncertified roots supply no diagnostic evidence, and an
adequate or unresolved evaluated root prevents widening. Volatility and density
failures alone never trigger widening, and neither does a mean failure in some
roots while another root fails only its volatility or density checks. Each wider root is refitted and counts
against `max_fits`. The new roots and all descendants share a new comparison
sample; likelihoods from different conditioning spans are never compared.
Native highest-AR-lag restrictions then reduce the wider model in the usual way.
`GeneralToSpecific(max_mean_order=5)` and `Auto(max_mean_order=5)` set the default
cap, configurable from 1 through the native limit of 8. `SelectionResult.widenings`
records each new order, triggering test, p-value and root triggers; `summary()`
reports the widening. Records also state its fit count and whether the root
batch completed; an unfitted budget frontier is reported as pending widening.
Persistent mean rejection at the cap returns
`mean_order_cap_reached`; an unavailable template returns
`mean_widening_unavailable`; an exhausted fit budget remains `budget_exhausted`.

The graph currently lacks APARCH fixed-power/asymmetry, skew-Gaussian and finite
GH restrictions. Cross-terminal union tests and equivalence deduplication beyond
native spec identity are not implemented. The defaults and the adaptive
procedure have not been calibrated.

`InformationCriterion` requires an explicit adequacy choice. `ConfidenceSet`
compares frozen training candidates using chronological predictive losses and
the native range-statistic MCS with joint circular-block resampling. A full-data
deployment refit supplies no additional holdout evidence. Selection does not
cover free-power no-shock hypotheses, singular simultaneous order reductions,
GH closure limits, structural breaks, long memory, infinite variance, DCC or
post-selection confidence intervals. Non-rejection does not identify the true
model, and the search budget limits any parsimony claim to the explored graph.

### Retained boundary nuisance in restriction tests

Selection evaluates the parent row scores at the embedded restricted estimate
and uses their outer products as Fisher information. Under a correct-likelihood
null this is consistent. An observed Hessian at a strongly false restricted
point can be indefinite even when information and the parent optimum are regular.
Singular score information and unidentified recursions still preclude calibration.

With retained closed faces, LR uses the difference of squared information-metric
distances of a Gaussian vector to the child and parent tangent cones. This need
not be a chi-bar-square mixture (Self and Liang, 1987, Case 8). Orthogonal retained
blocks reduce to the ordinary chi-square or chi-bar-square law. Other regular
polyhedral cases enumerate the projection QP's independent active sets and use
Gaussian simulation with mt19937_64 seed `0x534349564f4c` and an explicit
Box-Muller conversion. The orthogonal complement of the constraint span is
profiled out before simulation. Up to 1,024, 4,096 and 32,768 draws are used;
`maximum_cells` controls the maximum budget. A decision interval crossing the
requested level is unresolved. The reported error for `cone_projection` is a
99.99% Bernoulli Chernoff Monte Carlo interval radius, not a deterministic
integration bound; three prespecified looks have at least 99.97% simultaneous
coverage by
the union bound. The intervals invert the finite-sample Bernoulli relative-entropy
bound, including zero observed exceedances. The deterministic chi-bar cubature
retains its absolute-error test.
These are pointwise asymptotic laws; estimating the active face does not establish
uniform size over nuisance sequences approaching a boundary.

References: [Self and Liang (1987)](https://doi.org/10.1080/01621459.1987.10478472),
[Andrews (2001)](https://doi.org/10.1111/1468-0262.00210).
