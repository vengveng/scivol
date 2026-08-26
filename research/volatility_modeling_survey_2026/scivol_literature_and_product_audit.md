# What scivol should become next

Literature and product audit, 25 August 2026

## Executive judgment

scivol no longer has an obvious univariate GARCH coverage problem. Its current
combination of four variance recursions, eight standardized densities, three
mean families, analytical derivatives, QMLE, automatic selection, rolling
evaluation, CCC, and DCC is already unusually broad. Adding a dozen lightly used
GARCH acronyms would make the model list longer without making the library much
more useful.

The main gap is that scivol still treats a fitted conditional-variance recursion
as the center of the product. Mature volatility work treats the fitted model as
one stage in a longer chain:

1. construct or choose a volatility target;
2. fit and diagnose one or several models;
3. quantify forecast and parameter uncertainty;
4. evaluate density, variance, tail-risk, and covariance forecasts out of
   sample;
5. turn those forecasts into portfolio, hedging, scenario, or monitoring
   decisions.

The best next move is therefore a workflow release, followed by a realized-data
release, followed by a multivariate scaling release. Long-memory,
score-driven, regime-switching, and stochastic-volatility models matter, but
they should come after the library can test and use its present models as well
as it estimates them.

My recommended order is:

1. diagnostics, backtests, expected shortfall, forecast distributions, and
   bootstrap uncertainty;
2. a realized-measure subsystem, then HAR-RV, Realized GARCH, HEAVY, and
   GARCH-MIDAS;
3. corrected and asymmetric DCC, scalable factor or shrinkage correlation
   models, composite likelihood, and multivariate diagnostics;
4. MF2-GARCH or a component layer, conditional EVT tails, variance regressors,
   and a score-driven scale model;
5. FIGARCH or FIEGARCH, GARCH-in-mean, then selected regime-switching and
   stochastic-volatility models;
6. separate integrations for option-implied data and risk-neutral calibration.

## How the survey was judged

The phrase "most complete" is dangerous unless completeness has a unit. This
audit scores a candidate addition on five questions:

- Does it enable a common empirical task that scivol cannot perform now?
- Is there durable evidence of use across assets, samples, or applications?
- Does it fit scivol's likelihood, derivative, and composition architecture?
- Can scivol offer a clear correctness or speed advantage?
- Does the addition create a coherent workflow, rather than a disconnected
  class name?

This favors Realized GARCH over an obscure one-equation GARCH variant, ADCC over
unrestricted VECH, and a complete VaR and ES backtest object over another
printed statistic in `summary()`.

## What is already strong

The companion [current-surface inventory](current_scivol_surface.md) records the
details. Four points change how the literature should be read.

First, scivol already covers the asymmetric daily-return workhorses. GJR-GARCH,
EGARCH, and APARCH remove the strongest reason to add threshold or power variants
one at a time.

Second, its density layer is deeper than those in the mainstream Python and R
packages. The generalized-hyperbolic family matters more for tail and skew
coverage than adding several nearly interchangeable named skew distributions.

Third, scivol has the beginnings of a proper model-evaluation workflow. It has
rolling and expanding fits, log-score, QLIKE and MSE losses, Diebold-Mariano
comparisons, and a block-bootstrap model confidence set. The missing work is to
turn these pieces into general public evaluation objects rather than keep them
inside automatic selection.

Fourth, CCC and scalar DCC mean scivol is already multivariate. The next question
is not whether to add multivariate support. It is which restrictions retain
positive definiteness, statistical meaning, and usable scaling as the number of
assets grows.

## Three immediate correctness findings

The survey also exposed three issues in current post-fit behavior. These deserve
attention before new model families.

First, `ForecastResult.quantile()` and `.var()` are exact location-scale
calculations only one step ahead. At longer horizons, a GARCH predictive return
integrates over random future variance states and is generally a mixture.
Scaling an innovation quantile by the square root of a point forecast for
expected variance does not produce the mixture quantile. For linked mean models,
the implementation also switches to a Normal quantile when total forecast
variance differs from residual variance, even if the fitted innovation density
is heavy-tailed. Neither approximation is recorded in the result.

Second, QLIKE and MSE automatic selection currently define the realized variance
proxy as each candidate's squared residual. When mean candidates differ, the
target therefore differs by candidate. Pairwise forecast rankings require one
aligned outcome, fixed across all candidates. A user-supplied realized measure
is best; squared returns relative to one declared common mean is a fallback.

Third, multi-step APARCH and EGARCH variance forecasts use 4,000 internal Monte
Carlo paths and a fixed seed, but the public result reports neither the method,
path count, seed, nor Monte Carlo error. A reproducible deterministic-looking
number is not the same as an analytic forecast. The result should state how it
was computed and let the user control accuracy.

These findings do not undermine the analytical estimation kernels. They show
why post-fit semantics and validation deserve the next concentrated release.

## Exact derivatives do not finish the inference problem

scivol's analytical Hessians and AD validation establish that it differentiates
the stated objective correctly. Statistical validity still depends on the
estimating objective, innovation moments, interior parameters, sample design,
and model specification.

Every covariance result should record its provenance: observed Hessian,
expected information, OPG, sandwich, bootstrap, or posterior; theta or z space;
effective sample; rank and condition number; and the assumptions under which
the interval is valid. A QMLE sandwich is robust to a misspecified conditional
density when the relevant conditional moments and higher-moment conditions hold.
It is not valid under arbitrary misspecification ([White,
1982](https://doi.org/10.2307/1912526); [Bollerslev and Wooldridge,
1992](https://doi.org/10.1080/07474939208800229)).

Zero ARCH, GARCH, leverage, and spillover coefficients are boundary points.
Their estimators have cone-projection limits, so ordinary symmetric Wald
intervals and chi-square likelihood-ratio tests are wrong. This also means that
ordinary t-tests are not a sound way to delete extra lags. Add restricted
theta-space fits, analytical one-boundary tests, and shrinkage-bootstrap LR tests
for joint or nuisance-boundary cases ([Francq and Zakoian,
2007](https://doi.org/10.1016/j.spa.2007.01.001); [Cavaliere et al.,
2022](https://doi.org/10.1016/j.jeconom.2020.05.006)).

Heavy tails create a separate failure. A fitted Student-t density with degrees
of freedom at or below four does not provide the usual fourth moment needed for
standard Gaussian-QMLE sandwich inference. scivol should test and report moment
existence, suppress unjustified default z-tests, and later add subsampling or a
heavy-tail robust estimator rather than silently relabel ordinary QMLE as
robust.

Missing values also change the model. Dropping an interior gap joins two returns
that were not adjacent in time. Observation-driven recursions cannot generally
integrate over a missing update by ignoring it, and convenient zero-update rules
can be inconsistent ([Blasques, Gorgi, and Koopman,
2021](https://doi.org/10.1016/j.jeconom.2020.07.043)). Keep rejection as the
safe default, add explicit calendar and time-index policies now, and treat
particle-likelihood or indirect-inference handling as separate estimators.

## The highest-value omission is workflow depth

### Diagnostics should test the model, not only the fitted density

The current PIT cell test and Ljung-Box tests on four PIT moments are useful.
They do not replace the standard layered diagnostic sequence:

- serial dependence in raw and standardized residuals;
- remaining ARCH effects in squared or absolute standardized residuals;
- asymmetric response and sign or size bias;
- conditional-distribution calibration and independence;
- parameter constancy and structural breaks;
- model-specific multivariate correlation and covariance adequacy.

A new `result.diagnostics()` should return typed results with statistics,
p-values, null hypotheses, sample adjustments, and warnings. It should include
Engle ARCH-LM, residual and squared-residual portmanteau tests with the correct
degrees-of-freedom adjustments, [Engle-Ng sign and size
bias](https://doi.org/10.1111/j.1540-6261.1993.tb05127.x), [Berkowitz density
calibration](https://doi.org/10.1198/07350010152596718), and [Nyblom-style
parameter stability](https://doi.org/10.1080/01621459.1989.10478759). DCC and CCC need correlation
constancy and multivariate residual tests. Plots should consume these result
objects rather than recompute hidden variants of the tests.

### Tail risk needs a complete forecast and backtest object

`ForecastResult.var()` is a useful start. The library should add expected
shortfall for every shipped density, empirical and filtered-historical modes,
and a `RiskForecastResult` that retains forecast origin, horizon, level,
realization alignment, and exceedance indicators.

The matching backtest layer should cover unconditional coverage, independence,
conditional coverage, duration and dynamic-quantile tests for VaR, plus modern
joint VaR and ES tests. A loss-based comparison should use strictly consistent
scores for the target functional. This is both a feature and a correctness
requirement. A tail forecast without alignment metadata is easy to backtest one
day off, and an ES comparison with an unsuitable loss can rank models
incorrectly.

### Forecast uncertainty is different from process uncertainty

Path simulation draws future innovations conditional on fitted parameters.
That does not quantify parameter uncertainty. scivol should expose at least
three distinct forecast modes:

- analytic conditional moments where the recursion permits them;
- Monte Carlo or residual-bootstrap paths with parameters fixed;
- full residual or parametric bootstrap with refitting, which propagates
  estimation uncertainty.

Return and variance prediction intervals should label which sources of
uncertainty they contain. The full bootstrap is expensive, which makes it a good
match for scivol's speed and parallel-fitting strengths. [Pascual, Romo, and
Ruiz](https://doi.org/10.1016/j.csda.2004.12.008) give a direct GARCH bootstrap
design that incorporates parameter uncertainty.

### Evaluation should become a first-class public module

The automatic-selection code already contains valuable pieces. Promote aligned
loss panels, rolling forecast origins, Diebold-Mariano output, and model
confidence sets into public objects that can compare arbitrary fitted models.
Add Giacomini-White conditional predictive-ability tests, forecast combinations,
and economic losses for volatility targeting and minimum-variance portfolios.
Use QLIKE when a noisy realized variance proxies for latent variance. Preserve
the distinction between statistical loss, density loss, tail-risk loss, and an
application's economic objective.

## Realized volatility is the largest model-family gap

Daily squared returns are noisy observations of latent daily variance. Intraday
prices permit much more precise realized measures, but only after dealing with
microstructure noise, jumps, asynchronous trading, overnight returns, and
session boundaries. A model class that accepts a precomputed `rv` array without
recording these choices would be convenient and scientifically weak.

Build a measurement subsystem first. Its initial surface should include realized
variance and covariance, quarticity, bipower and median realized variation,
positive and negative semivariance, noise signature plots, realized kernels,
pre-averaging, Hayashi-Yoshida covariance, and explicit synchronization and PSD
policies. Every result should retain the estimand, sampling rule, session rule,
bandwidth, effective observations, noise assumptions, and jump treatment.

Then add models in this order:

1. HAR-RV and its leverage, semivariance, jump, and quarticity extensions;
2. Realized GARCH and realized EGARCH, which combine returns with a measurement
   equation for a realized measure;
3. HEAVY models, particularly when realized covariance is available;
4. GARCH-MIDAS for short-run and long-run variance with mixed-frequency
   regressors;
5. multivariate realized GARCH or a factor-restricted realized covariance model.

This ordering follows the empirical signal in the recent literature. Simple
HAR-type models remain difficult to beat systematically. Nonlinear machine
learning gains are sample- and objective-dependent, while realized inputs and
pooled cross-asset information show more stable value. The core references are
[Corsi's HAR](https://doi.org/10.1093/jjfinec/nbp001), [Realized
GARCH](https://doi.org/10.1002/jae.1234),
[HEAVY](https://doi.org/10.1002/jae.1158), and
[GARCH-MIDAS](https://doi.org/10.1162/REST_a_00300).

## Multivariate development should favor restrictions that scale

### Fix the DCC family before jumping to BEKK

The direct extensions with the best benefit-to-complexity ratio are corrected
DCC and scalar asymmetric DCC. Corrected DCC addresses the finite-sample bias in
the original correlation-targeting recursion. ADCC permits negative aggregate
shocks to change correlations differently from positive shocks. Both preserve
the familiar marginal-plus-correlation workflow. See [Aielli's corrected
DCC](https://doi.org/10.1016/j.jeconom.2011.02.001) and [Cappiello, Engle, and
Sheppard's asymmetric model](https://doi.org/10.1093/jjfinec/nbl005).

Two-stage estimation also needs two-stage inference. Standard errors that treat
estimated marginal parameters as known are too optimistic. A complete result
should state whether covariance estimates use naive second-stage curvature,
the appropriate sandwich correction, or a bootstrap.

### High dimension changes the target

Scalar DCC estimates few dynamic parameters, but it still stores and updates
dense covariance matrices and assumes common dynamics for every pair. The next
scalable tier should include shrinkage targets, block or equicorrelation
restrictions, latent factors, and composite likelihood. These methods trade
pair-specific detail for stable estimation and computation. That trade should be
visible in the model name and result metadata.

Recent work on multivariate realized GARCH, realized covariance graphs, and
factor models points in the same direction. High-dimensional covariance
forecasting is increasingly a structured prediction problem. A useful scivol
model should produce PSD forecasts, factor or block interpretations, and direct
portfolio-loss evaluation.

### BEKK belongs in the library, but with a narrow promise

BEKK is foundational, guarantees positive-definite covariance recursions under
the standard construction, and supports volatility-spillover and news-impact
analysis. It is also expensive and overparameterized. Full BEKK(1,1) is useful
for small systems, not as scivol's answer to a 100-asset covariance problem.
The original construction is [Engle and
Kroner](https://doi.org/10.1017/S0266466600009063).

If implemented, ship scalar, diagonal, asymmetric, and full low-dimensional
BEKK with hard dimension warnings, stationarity diagnostics, news-impact
surfaces, and impulse-response tools. Unrestricted VECH has little claim on
implementation priority because positivity constraints and parameter counts
make it a poor general-purpose public model.

### Dynamic copulas should be separate from correlation dynamics

The present Gaussian and common-t DCC choices conflate a correlation recursion
with a narrow copula set. A later dependence layer could combine fitted
marginals with dynamic copulas, pair-copula vines, or tail-dependence models.
This is valuable for joint extremes and multivariate VaR, but it needs its own
diagnostics, simulation rules, and parameterization. It should not be smuggled
into `DCC(distribution=...)` until the API can distinguish linear correlation
from general dependence.

## Univariate models worth adding

### First group, expected and mature

MF2-GARCH is the strongest near-term univariate candidate. It separates short
and long volatility cycles while keeping returns stationary and long-horizon
forecasts mean reverting. A 2025 study reports gains over GJR-GARCH,
GARCH-MIDAS-RV, and log-HAR at longer horizons across the S&P 500 and more than
2,100 equities. It fits scivol's observation-driven C-kernel architecture and
produces an interpretable component decomposition ([Conrad and Engle,
2025](https://doi.org/10.1002/jae.3118)).

More general component GARCH and GARCH-MIDAS models remain useful for term
structures, macro-finance questions, and option applications. Variance
regressors, often called GARCH-X, are a smaller code change with broad use in
macro, energy, and news-based applications. GARCH-in-mean links conditional
risk to expected return and fits the existing component composition well.

Conditional extreme-value theory is a workflow omission, not another center
density. Fit the ordinary volatility and innovation model first, then fit a
peaks-over-threshold model to standardized residual tails. This directly extends
conditional VaR and ES beyond the center distribution and complements scivol's
strong GH density coverage. [McNeil and
Frey](https://doi.org/10.1016/S0927-5398(00)00012-8) provide the foundational
conditional EVT workflow.

`FIGARCH` or `FIEGARCH` would close the clearest familiar named-model gap. Long
memory is prominent in return volatility and even more prominent in realized
measures. The implementation must expose truncation length, initialization, and
forecast semantics because the infinite ARCH representation is approximated in
finite time. Break diagnostics should accompany it, since level shifts can look
like fractional persistence.

### Second group, a new abstraction rather than one more recursion

Generalized autoregressive score models update time-varying parameters with the
scaled conditional score. They encompass robust heavy-tailed volatility,
time-varying shape and skewness, dynamic correlations, counts, and durations.
A real GAS layer would reuse scivol's density derivatives and open several model
families at once. A Student-t score-driven scale model is a sensible first slice
because extreme observations have bounded influence under the appropriate
update, but it should be designed as the first use of a reusable GAS state
engine.

Markov-switching GARCH addresses abrupt volatility regimes and parameter
instability, but likelihood filtering, label identification, initialization,
and multimodal optimization are harder than in ordinary GARCH. It should follow
the diagnostic and multistart infrastructure needed to tell users when the
answer is fragile.

Stochastic volatility is the largest conceptual omission if "complete" includes
the whole volatility field. It is parameter-driven rather than
observation-driven, so estimation usually requires state-space integration,
particle methods, MCMC, or specialized approximations. It deserves a separate
estimation architecture. Forcing it through the current GARCH routine registry
would harm both APIs.

### Third group, useful in adjacent data types

Multiplicative error models cover nonnegative series such as realized variance,
volume, and duration. Autoregressive conditional duration models cover irregular
event times. These matter for a broad volatility toolkit, especially once a
high-frequency data layer exists, but they are not prerequisites for a better
daily-return workflow.

## Recent developments that should influence design

Several 2024 to 2026 results are more useful as design signals than as immediate
classes:

- multivariate Realized GARCH uses an unconstrained correlation
  parameterization and factor restrictions to make realized correlation models
  practical ([Archakov, Hansen, and Lunde,
  2026](https://doi.org/10.1016/j.jeconom.2025.106040));
- graph-based realized covariance forecasts exploit cross-asset structure and
  improve portfolio outcomes in empirical work;
- "bespoke" realized measures optimize measurement for the downstream forecast
  model, a warning that measurement and prediction cannot always be designed
  independently;
- realized-drift decompositions challenge the usual assumption that all
  high-frequency quadratic variation should be interpreted as volatility;
- large pooled neural networks benefit more from cross-sectional sample scale
  than from model size or architectural novelty;
- broad comparisons still find that nonlinear models do not systematically
  dominate simple leveraged HAR and HARX specifications ([Branco, Rubesam, and
  Zevallos, 2024](https://doi.org/10.1016/j.jempfin.2024.101524));
- rough-volatility evidence remains contested because measurement error and
  observation schemes can make ordinary diffusions appear rough ([Cont and Das,
  2024](https://doi.org/10.1007/s13571-024-00322-2)).

The practical conclusion is conservative. Expose clean data, aligned evaluation,
pooled panels, and extensible forecast interfaces before putting neural or rough
models in the core package. Those foundations let researchers compare new
methods without giving them privileged evaluation rules.

## Applications and the features they require

For equities, foreign exchange, rates, commodities, and crypto, the common
requirements are heavy tails, asymmetric responses, tail-risk forecasts,
time-varying covariance, rolling refits, and calendar-safe alignment. scivol
already covers much of the model layer but lacks the complete tail-risk and
backtesting layer.

Electricity prices and other energy series add spikes, strong seasonality,
exogenous weather and load, regime changes, and multiple intraday horizons.
ARX and HARX are useful pieces, but seasonal mean components, variance
regressors, regime switching, and explicit forecast-origin data alignment would
make the library much more credible in this domain.

Macroeconomic applications often use conditional variance as an uncertainty
measure. They need mixed frequencies, long samples, structural-break tests,
GARCH-in-mean effects, and careful missing-data semantics. Environmental,
insurance, duration, and count applications often change the conditional data
type itself. That is where a future score-driven or multiplicative-error layer
adds more than another return-specific density.

## Options and implied volatility

GARCH option pricing is a substantial field, from Duan's locally risk-neutral
valuation to affine Heston-Nandi GARCH and joint return, VIX, and option
calibration. It also changes the probability measure, pricing kernel, data
layout, and numerical outputs. That is too much to hide inside the historical
estimation classes.

Keep risk-neutral calibration outside the core for now. The core should first
support implied-volatility or VIX series as regressors and mixed-frequency
inputs, physical-measure variance forecasts over arbitrary horizons, and clean
scenario export. A later `scivol-options` integration could own Heston-Nandi,
pricing transforms, risk-neutral parameters, option panels, and calibration
losses while consuming scivol's fitted physical dynamics.

## Software comparison

Python's [`arch`](https://arch.readthedocs.io/en/latest/univariate/univariate.html)
remains the closest mainstream comparator. It has FIGARCH,
HARCH, EWMA and RiskMetrics variants, MIDAS-hyperbolic weights,
ARCH-in-mean, analytic, simulated and residual-bootstrap forecasts, VaR examples,
and a general bootstrap and multiple-comparison toolkit. It does not yet offer a
mature released multivariate volatility layer. scivol is already stronger in
density breadth, analytical Hessians, AD-backed derivative verification, joint
ARMA-GARCH coverage, and shipped CCC/DCC.

R's [`rugarch`](https://cran.r-project.org/web/packages/rugarch/rugarch.pdf)
remains the workflow benchmark. It combines many univariate
families, variance regressors, ARFIMA and ARCH-in-mean, simulation, rolling
backtests, parameter-distribution simulation, bootstrap forecasts, and numerous
diagnostics. [`rmgarch`](https://cran.r-project.org/web/packages/rmgarch/rmgarch.pdf)
adds DCC and GO-GARCH workflows. scivol's opportunity is
to match that empirical completeness with a smaller, typed Python API and much
stronger derivative and kernel verification.

The newer R packages [`tsgarch`](https://tsmodels.r-universe.dev/tsgarch) and
[`tsmarch`](https://tsmodels.r-universe.dev/tsmarch) are more relevant design
comparators than their age suggests. They use automatic differentiation, expose
profiling, filtering, PITs, walk-forward backtests, and explicit multivariate
inference choices. Julia's
[`ARCHModels.jl`](https://s-broda.github.io/ARCHModels.jl/stable/) is the most
interesting high-dimensional comparator: its large-scale DCC route uses
nonlinear shrinkage and pairwise likelihood to avoid full matrix inversions.

MATLAB covers a narrower GARCH model list but integrates it with ARIMA,
state-space, switching, diagnostics, and an interactive econometric workflow.
That is a reminder that users compare complete analyses, not only likelihood
timings.

## A staged roadmap

### Stage 0, correctness and workflow completion

- Typed residual, ARCH, asymmetry, PIT, stability, VaR, ES, and multivariate
  diagnostics.
- Expected shortfall for every density and aligned tail-risk backtests.
- Public rolling loss panels, forecast comparisons, and model confidence sets.
- Simulation and residual-bootstrap forecast distributions, then a full
  refitting bootstrap.
- Forecast-origin, horizon, target-date, annualization, and uncertainty metadata.
- Explicit missing-data policy and calendar alignment checks.
- Covariance provenance, moment-validity flags, boundary-aware tests, KKT and
  curvature diagnostics, and initialization sensitivity.
- A public conformance suite that matches likelihoods, scores, Hessians,
  filtered variances, forecasts, and robust covariance against independent
  packages under identical scaling, backcasts, burn-in, and effective samples.

Definition of done: a user can take any fitted model through diagnostics,
uncertainty-aware forecasting, tail-risk or covariance backtesting, and model
comparison without leaving scivol.

### Stage 1, realized measurement and models

- Clean intraday data container and session-aware sampling.
- Realized variance, covariance, quarticity, semivariance, jump measures,
  realized kernels, pre-averaging, and asynchronous covariance.
- HAR-RV family, Realized GARCH or EGARCH, HEAVY, and GARCH-MIDAS.
- QLIKE and economic evaluation against noisy proxies, with measurement metadata.

Definition of done: results reproduce published benchmark designs on at least
one open high-frequency dataset and remain stable across defensible sampling and
noise treatments.

### Stage 2, scalable multivariate models

- cDCC, ADCC, corrected two-stage inference, and multivariate adequacy tests.
- Shrinkage and factor correlation targets, block or DECO restrictions, and
  composite likelihood.
- Low-dimensional scalar, diagonal, asymmetric, and full BEKK.
- Portfolio variance, beta, hedge-ratio, volatility-spillover, and covariance
  forecast evaluation objects.

Definition of done: the library states and tests its time and memory scaling,
keeps every forecast PSD when promised, and separates low-dimensional structural
models from high-dimensional forecasting models.

### Stage 3, selected univariate extensions

- MF2-GARCH with exposed short and long components, then general component and
  mixed-frequency regressors.
- Conditional EVT tails on filtered standardized residuals.
- Variance regressors, GARCH-in-mean, and a general score-driven state update,
  beginning with robust Student-t scale.
- FIGARCH or FIEGARCH with transparent truncation, break diagnostics, and
  validation.

### Stage 4, different estimation paradigms

- Markov-switching volatility with reproducible multistart and identification
  diagnostics.
- Stochastic volatility in a separate state-space estimation layer.
- Multiplicative-error and duration models after the realized-data subsystem is
  stable.
- Optional neural and rough-volatility adapters judged by the same rolling loss
  and economic-value interfaces as classical models.

## What not to prioritize

Do not implement unrestricted VECH as a flagship multivariate model. Do not add
minor named GARCH variants solely to increase a checklist. Do not put a neural
network in the core before pooled-panel evaluation and reproducibility rules
exist. Do not call simulated innovation bands "forecast uncertainty" without
stating that parameters are fixed. Do not silently repair non-PSD covariance
matrices or silently synchronize asynchronous prices. Do not fold option pricing
into physical-measure estimation without an explicit risk-neutral model.

## The durable product principle

scivol can plausibly become the fastest and most correct likelihood-based
volatility library. "Most complete" should mean that it completes the empirical
job and makes every statistical choice inspectable. The winning sequence is not
more recursions first. It is better diagnostics and forecasts, then realized
information, then multivariate scale, then new estimation paradigms.

The topic notes and source index in this folder document the evidence behind
each recommendation.
