# Univariate volatility omissions

## Bottom line

scivol already covers the workhorse conditional-variance models. The next additions should expand the questions users can answer, not collect more acronyms. Five additions meet that test.

1. **MF2-GARCH, followed by a general component/MIDAS layer.** This is the clearest near-term omission. It separates short and long volatility cycles, improves medium- and long-horizon forecasts, and remains an observation-driven model that fits scivol's C-kernel design.
2. **A score-driven scale model.** GAS supplies a principled density-aware update and gives heavy-tailed observations bounded influence when the chosen density implies it. It also creates a reusable path to time-varying shape, skewness, and jump intensity.
3. **A two-regime independent Markov-switching GARCH.** It addresses abrupt breaks and gives users a serious rival to the long-memory explanation of persistence.
4. **A conditional EVT workflow on standardized residuals.** This adds a distinct tail-risk capability without pretending that a generalized Pareto distribution is a density for the center of returns.
5. **FIGARCH as the first long-memory return model.** It remains widely used and is an expected feature in a complete volatility library. Its implementation must make truncation, initialization, and the long-memory-versus-break ambiguity visible.

Stochastic volatility is a strategic omission, but it is a second estimation paradigm, not another GARCH component. It deserves a separate project with filtering and simulation infrastructure. GARCH-in-mean is cheap and useful for replication work, but it has lower forecasting value. ACD and MEM are mature models for durations and other positive series, yet they expand scivol beyond return volatility. FIEGARCH, HYGARCH, smooth-transition GARCH, and a generic augmented-GARCH grammar should follow evidence of demand. Shipping all of them first would add more maintenance than capability.

## Priority matrix

| Candidate | Distinct applied question | Maturity | Engineering fit | Recommendation |
| --- | --- | --- | --- | --- |
| MF2-GARCH / component GARCH | What are the short and long volatility cycles, and how far ahead can volatility be forecast? | High, including strong 2025 evidence | High | P0 |
| Conditional EVT | What are conditional VaR and ES beyond the fitted center distribution? | High | High as a post-fit workflow | P0 |
| Score-driven scale | How should the full density, including tails, update volatility? | High | High for scale-only GAS | P0/P1 |
| Two-state independent MS-GARCH | Are persistence and crises better described by discrete regimes? | High | Medium | P1 |
| FIGARCH | Does variance have hyperbolic rather than geometric memory? | High | Medium because of infinite recursion | P1 |
| GARCH-MIDAS-X | Which low-frequency macro or financial variables drive long volatility? | High | Medium because data have mixed frequencies | P1 after MF2 |
| Basic stochastic volatility | Is latent log variance better than a deterministic conditional recursion? | High | Low without a state-space engine | Strategic P1/P2 |
| GARCH-in-mean | Does conditional risk enter expected return or another conditional mean? | High historically | High, but cross-role | P2 quick win |
| MEM / ACD | How do volatility proxies, volumes, ranges, and inter-event durations evolve? | High in market microstructure | Medium, requires a positive-response family | P2 if scope expands |
| Rough / fractional SV | Is volatility locally rough, and can this improve realized-volatility forecasts or option work? | Active | Low | Experimental P2/P3 |
| FIEGARCH | Are long memory and leverage needed together in log variance? | Established | Medium/low | P3 after FIGARCH |
| Smooth-transition GARCH | Does the response change continuously with a transition variable? | Established but niche | Medium | P3 |
| HYGARCH | Can a hyperbolic model relax FIGARCH's integration restriction? | Established but niche | Medium/low | P3 |
| Generic augmented GARCH | Can one grammar nest many nonlinear recursions? | Strong theory | Low product value | Do not prioritize |

## 1. Long-memory conditional variance

### FIGARCH

Baillie, Bollerslev, and Mikkelsen introduced FIGARCH to replace geometric shock decay with hyperbolic decay through a fractional-differencing parameter `d`. Their application to daily exchange rates showed why a standard GARCH fit can look close to IGARCH when the data contain slower decay ([Baillie, Bollerslev, and Mikkelsen, 1996](https://doi.org/10.1016/S0304-4076(95)01749-6)). Bollerslev and Mikkelsen developed the log-variance FIEGARCH form and connected long memory to option pricing ([Bollerslev and Mikkelsen, 1996](https://doi.org/10.1016/0304-4076(95)01736-4)). Davidson's HYGARCH places GARCH, IGARCH, and FIGARCH in a broader linear-in-squares class and relaxes some of FIGARCH's moment restrictions ([Davidson, 2004](https://doi.org/10.1198/073500103288619359)).

FIGARCH adds a real empirical hypothesis. Its impulse weights decay like a power law, and `d = 0` recovers short memory. It is useful for daily returns, commodities, exchange rates, and long-horizon risk forecasts. A 2022 commodity study compared FIGARCH with HAR and fractional stochastic volatility, which shows that the model remains an active forecasting benchmark rather than a historical curiosity ([Alfeus and Nikitopoulos, 2022](https://doi.org/10.1016/j.jcomm.2022.100248)).

The cost is substantial:

- The ARCH infinity representation needs truncation. The truncation lag affects both likelihood and gradients, particularly when `d` is large. A fast implementation should generate fractional weights recursively, cache them, and expose the effective truncation.
- Positivity is more restrictive than in ordinary GARCH. Arbitrary optimizer boxes do not guarantee nonnegative impulse weights.
- Initialization dies out slowly. Backcasting and presample choices can affect a large part of a short sample.
- Classical FIGARCH has awkward moment properties. Users should not infer that `0 < d < 1` gives the same covariance-stationary interpretation as ARFIMA in a conditional mean.
- Breaks and regime shifts can mimic long memory. Diebold and Inoue prove that infrequent regime switching can appear as long memory even asymptotically ([Diebold and Inoue, 2001](https://doi.org/10.1016/S0304-4076(01)00073-2)). Earlier evidence showed that ignored variance shifts inflate estimated GARCH persistence ([Lamoureux and Lastrapes, 1990](https://doi.org/10.1080/07350015.1990.10509794)).

Recommended scope: ship `FIGARCH(p, q, truncation="auto")` for small orders and Normal, Student-t, and Skew-t densities. Add `truncation={500, 1000, 2500, "auto"}` and a `presample` option. `fit_info` should report the chosen lag, discarded warm-up observations, and weight-tail mass. A `result.truncation_sensitivity()` method should refit or reevaluate at longer lags. For model comparison, offer a break-aware warning or workflow that compares FIGARCH against component and two-state MS-GARCH models.

Required tests and diagnostics:

- AD-oracle checks of the full truncated objective, including `d`, at identical truncation and presample values.
- Exact nesting checks at `d = 0` against GARCH, with boundary-aware inference or a parametric bootstrap for the null.
- DGP recovery across `d` values of 0.1, 0.3, and 0.45, heavy-tailed innovations, and at least three truncation lengths.
- Forecast convergence checks and impulse-weight nonnegativity checks.
- Standardized residual and squared-residual portmanteau tests, plus a stability or break comparison. A large estimated `d` with unstable subsample fits should be reported as ambiguous evidence.

### FIEGARCH and HYGARCH

FIEGARCH combines fractional memory, log variance, and asymmetric news. It avoids direct positivity constraints and is the logical long-memory companion to scivol's EGARCH. Its infinite filter is still expensive, and parameter identification can be poor because fractional memory, EGARCH persistence, leverage, and tail thickness all compete to explain the same observations. It should follow a correct FIGARCH kernel and share its truncation diagnostics.

HYGARCH has a sound theoretical motivation, but it has far less applied demand. It is best treated as a later model or as an alternate long-memory transform once the common ARCH-infinity engine is stable. FIAPARCH is even less urgent because scivol already has APARCH and would need to combine two numerically difficult mechanisms at once.

Product judgment: FIGARCH is completeness; FIEGARCH is a defensible extension; HYGARCH and FIAPARCH are taxonomy until users ask for them.

## 2. Components, mixed frequencies, and volatility cycles

Engle and Lee's component GARCH decomposes volatility into persistent and transitory components ([Engle and Lee, 1999](https://doi.org/10.1093/oso/9780198296836.003.0020)). This decomposition is easier to explain than fractional integration and often fits the actual forecasting question: which part of today's volatility should survive for weeks or months?

Two later developments make this family a high priority.

First, GARCH-MIDAS links the slow component to monthly or quarterly macroeconomic variables while retaining daily GARCH dynamics. Engle, Ghysels, and Sohn used it to connect stock-market volatility to macroeconomic fundamentals ([Engle, Ghysels, and Sohn, 2013](https://doi.org/10.1162/REST_a_00300)). The framework is now common in equity, commodity, energy, uncertainty-index, and climate-risk applications. Recent work adds variable selection to the slow component and improves recursive forecasts ([Fang, Lee, and Su, 2020](https://doi.org/10.1016/j.jempfin.2020.05.007)). Conrad and Kleen's two-component GARCH-MIDAS study also shows that one slow component may be insufficient ([Conrad and Kleen, 2020](https://doi.org/10.1002/jae.2742)).

Second, Conrad and Engle's 2025 MF2-GARCH builds the long component from moving averages of standardized forecast errors from the short component. Returns remain stationary and long-horizon forecasts mean-revert. On the S&P 500 and more than 2,100 equities, MF2-GARCH outperformed GJR-GARCH, GARCH-MIDAS-RV, and log-HAR at longer horizons ([Conrad and Engle, 2025](https://doi.org/10.1002/jae.3118)). This is unusually direct evidence for a new model that also scales to scivol's intended workload.

Recommended sequence:

1. Implement `MF2GARCH(short=GJRGARCH(1, 1), window=63, long_order=(1, 1))` with Normal and Student-t innovations. Keep the short and long components available in the result.
2. Generalize the internal component engine to additive and multiplicative decompositions. If direct Engle-Lee compatibility is wanted, expose `ComponentGARCH(short_order=(1, 1), long_order=1, form="additive")` rather than hiding it inside MF2.
3. Add `GARCHMIDAS(short=GJRGARCH(1, 1), weights="beta", lags=12)` and pass dated low-frequency regressors through `fit(data, x_long=..., dates=...)`.

The result object should expose `variance_short`, `variance_long`, component-specific persistence, and horizon decompositions. Forecasts must honor data-release dates. A GARCH-MIDAS backtest that uses revised or future macro observations would be a look-ahead bug, so real-time vintages and publication lags need first-class support.

Estimation should start with joint QML and analytic derivatives. The MIDAS beta-weight parameters often form a flat or correlated likelihood surface, so multistart and profile plots are appropriate. Tests should verify positivity of the total path and any component defined as positive, recover short and long half-lives from a component DGP, and confirm that long-horizon forecasts converge to the model's stated limit. Diagnostics should test standardized forecast errors and moving averages of those errors. For MF2-GARCH, the latter is the model's motivating moment condition.

## 3. GARCH-in-mean

ARCH-in-mean allows conditional risk to enter the mean. Engle, Lilien, and Robins introduced it for time-varying term premia ([Engle, Lilien, and Robins, 1987](https://doi.org/10.2307/1913242)). The model remains useful for asset-pricing replication, risk-return studies, interest rates, inflation, and commodity premia.

This is a small feature with a real architectural wrinkle. The mean component consumes the volatility component's current state, so the usual MEAN to VOLATILITY one-way composition no longer holds. A clean API is:

```python
spec = ARMA(1, 0) + GARCHInMean(GARCH(1, 1), loading="variance") + StudentT()
result = spec.fit(returns)
```

Alternatively, `VolatilityInMean(power=2)` can be a mean component that declares a dependency on the single volatility component. The loading choices should be variance, standard deviation, and log variance, with the choice included in the signature. Estimation must be joint. Two-step estimation understates uncertainty and can distort the loading coefficient.

Diagnostics should include mean-residual tests after conditioning on volatility, a profile likelihood for the risk-loading coefficient, and weak-identification warnings when the fitted volatility path has little independent variation. GARCH-M is worth adding for compatibility, but it should not displace models with clearer forecast gains.

## 4. Stochastic volatility and rough volatility

### Basic stochastic volatility

In the canonical stochastic-volatility model, latent log variance follows an autoregression and returns provide a noisy nonlinear measurement. Harvey, Ruiz, and Shephard laid out the state-space treatment and emphasized the natural common-factor extensions ([Harvey, Ruiz, and Shephard, 1994](https://doi.org/10.2307/2297980)). Kim, Shephard, and Chib supplied a practical likelihood and Bayesian framework using a normal-mixture approximation to the log chi-square observation disturbance ([Kim, Shephard, and Chib, 1998](https://doi.org/10.1111/1467-937X.00050)). Durbin and Koopman showed that simulation-based maximum likelihood can approximate the likelihood as accurately as needed and extends cleanly beyond the basic model ([Durbin and Koopman, 1998](https://doi.org/10.1016/S0304-4076(98)00016-5)).

SV adds latent-state uncertainty, leverage through correlated return and variance shocks, heavy-tailed observations, jumps, and a bridge to option-pricing models. It also supplies an independent public solution path. Agreement between a GARCH and SV forecast is informative precisely because the two models do not share the same recursion.

It does not fit scivol's current kernel contract. A credible implementation needs filtering, smoothing, likelihood approximation or particle methods, simulation, and state uncertainty. A C objective alone is insufficient. The minimum serious scope is:

```python
spec = StochasticVolatility(ar=1, leverage=True) + StudentT()
result = spec.fit(returns, method="mle", particles=4096)
smoothed = result.smoothed_volatility
```

Support one maximum-likelihood route and one Bayesian route only if both can be tested independently. Return filtered and smoothed latent variance, state intervals, predictive draws, effective sample size for particle filters, and Monte Carlo error in the likelihood. Fix random-number streams for reproducibility. Tests need simulated-state recovery, unbiased-likelihood checks, particle-count convergence, and comparison with an independent reference implementation.

Recommendation: design an internal `StateSpaceRoutine` before exposing `StochasticVolatility`. Treat this as a strategic project, not a quick addition to `_kernels/`.

### Rough and fractional stochastic volatility

Gatheral, Jaisson, and Rosenbaum found that log volatility behaves locally like fractional Brownian motion with a Hurst exponent near 0.1, motivating rough volatility models ([Gatheral, Jaisson, and Rosenbaum, 2018](https://doi.org/10.1080/14697688.2017.1393551)). Recent econometric work has made the hypothesis more testable. Bolko, Christensen, Pakkanen, and Veliyev developed a GMM estimator that corrects the downward roughness bias caused by noisy realized variance and estimated `H` around 0.05 across major equity indices ([Bolko et al., 2023](https://doi.org/10.1016/j.jeconom.2022.06.009)). Fractional Ornstein-Uhlenbeck models also produce closed-form estimators and competitive realized-volatility forecasts ([Wang, Xiao, and Yu, 2023](https://doi.org/10.1016/j.jeconom.2021.08.001)).

This matters for realized-volatility forecasting and derivatives, but it is a weak first addition for a return-based GARCH library. Measurement error can create "illusory roughness," simulation is expensive, and option calibration introduces risk-neutral dynamics that scivol does not currently model. A contained first step would be an experimental `FractionalOU` model for log realized volatility, not rough Heston or rough Bergomi. It belongs beside realized-volatility models and should require a noise-bias diagnostic for `H`.

## 5. Regime and break models

Hamilton and Susmel's switching ARCH model was motivated by a simple failure of one-regime ARCH: crashes can have a different cause and persistence than ordinary large returns ([Hamilton and Susmel, 1994](https://doi.org/10.1016/0304-4076(94)90067-1)). Gray integrated out the regime path to obtain a tractable conditional distribution for interest rates ([Gray, 1996](https://doi.org/10.1016/0304-405X(96)00875-6)). Haas, Mittnik, and Paolella later proposed independent regime-specific GARCH processes with better understood dynamics and fewer path-dependence problems ([Haas, Mittnik, and Paolella, 2004](https://doi.org/10.1093/jjfinec/nbh020)).

MS-GARCH is useful for crises, policy changes, energy markets, cryptocurrencies, and any long sample that crosses structural shifts. It also provides a required adversary for FIGARCH. Recent forecasting work continues to combine regime switching with MIDAS and finds that the switching model can outperform a non-switching competitor even when a proposed exogenous predictor adds no value ([Segnon, Gupta, and Wilfling, 2024](https://doi.org/10.1016/j.ijforecast.2022.11.007)).

Recommended first API:

```python
spec = MarkovSwitching(
    GARCH(1, 1),
    regimes=2,
    switching=("omega", "alpha", "beta"),
    formulation="independent",
) + StudentT(shared_nu=True)
result = spec.fit(returns)
```

Begin with the Haas independent-process formulation, two regimes, constant transition probabilities, and ordered unconditional or long-run regime variances to control label switching. Return filtered, predicted, and smoothed regime probabilities; transition matrix; expected regime durations; and regime-specific variance paths. Forecasts must integrate over future regimes rather than freeze the most likely state.

Estimation is multimodal and regime-count tests are nonstandard because transition and regime parameters are unidentified under the one-regime null. Use deterministic multistart, record every start, and offer parametric-bootstrap model comparison. Diagnostics should flag near-absorbing transition probabilities, tiny effective regime samples, label instability across starts, and large differences between filtered and smoothed classifications.

Time-varying transition probabilities, hidden semi-Markov duration models, and three or more regimes are later work. A generic `MarkovSwitching(any_spec)` wrapper sounds elegant but multiplies identification and testing problems. Start narrow.

## 6. Score-driven models

Generalized autoregressive score models update a time-varying parameter in the direction of the scaled conditional-density score. The framework contains GARCH, ACD, and many count models as special cases, but its main value is broader: the same principle can update scale, location, skewness, degrees of freedom, correlations, or jump intensity ([Creal, Koopman, and Lucas, 2013](https://doi.org/10.1002/jae.1279)).

For volatility, a Student-t score update automatically downweights a return that is extreme under the current scale instead of feeding its square without bound into tomorrow's variance. This makes GAS more than a renamed GARCH. Recent applications use fat-tailed score dynamics to combine realized daytime variance and overnight returns for VaR and ES forecasting ([Opschoor and Lucas, 2021](https://doi.org/10.1016/j.ijforecast.2020.07.009)). A 2024 model jointly updates volatility and jump intensity and outperforms standard GARCH-with-jumps alternatives for S&P 500 returns and options ([Ballestra, D'Innocenzo, and Guizzardi, 2024](https://doi.org/10.1093/jjfinec/nbad001)).

The first implementation should be intentionally small:

```python
spec = ScoreDrivenScale(order=(1, 1), scaling="fisher") + StudentT()
result = spec.fit(returns)
```

The density owns the score formula; the volatility component owns the autoregressive update and transform that keeps scale positive. `fit_info` must state the scaling convention because identity, inverse square-root information, and inverse information define different models. Analytical derivatives of the likelihood are feasible, but the production gradient must differentiate through the complete score recursion. Validate the whole objective with AD.

Tests should verify score martingale differences, information-scaling identities, nesting relationships where they exist, and bounded influence under Student-t. Diagnostics should plot standardized scores and test them for autocorrelation. Later, a `ScoreDriven(density=..., dynamic=("scale", "shape"))` family can add time-varying tail thickness and skewness. Starting with a fully generic GAS constructor would expose too many unidentified or unstable combinations.

## 7. ACD and MEM for positive series

Engle and Russell's ACD model applies an ARCH-like recursion to irregular intertrade durations ([Engle and Russell, 1998](https://doi.org/10.2307/2999632)). Engle then formulated multiplicative error models for general nonnegative processes such as duration, volume, range, and realized volatility ([Engle, 2002](https://doi.org/10.1002/jae.683)). Engle and Gallo modeled absolute returns, daily ranges, and realized volatility jointly and found useful cross-indicator dynamics ([Engle and Gallo, 2006](https://doi.org/10.1016/j.jeconom.2005.01.018)). Cipollini and Gallo's retrospective documents two decades of continued MEM work and applications ([Cipollini and Gallo, 2025, first circulated in 2021](https://doi.org/10.1016/j.ecosta.2022.05.005)).

These are mature models, but they change scivol's response contract. A MEM fits a positive observation directly:

```python
spec = MEM(1, 1) + GammaMeanOne()
result = spec.fit(realized_variance)

spec = ACD(1, 1) + WeibullMeanOne()
result = spec.fit(durations, timestamps=timestamps)
```

They need positive-support densities parameterized to unit mean, seasonality adjustment, zeros, censoring, and event-time metadata. A vector MEM also needs nonnegative multivariate interactions and a new result shape. If scivol wants to own market-microstructure workflows, MEM and ACD are worthwhile. If its scope stays at return volatility, these models belong in a sibling package or a later positive-series module.

Diagnostics differ from return GARCH. Check generalized residuals through the fitted conditional CDF, overdispersion, remaining duration or positive-series autocorrelation, intraday seasonal patterns, zero mass, and censoring. Treating zero durations or market closures as ordinary positive observations will invalidate the likelihood.

## 8. Nonlinear recursions and the risk of taxonomy padding

Gonzalez-Rivera's smooth-transition GARCH lets the response to news move continuously between regimes and found threshold behavior for stocks but smoother transitions for exchange rates ([Gonzalez-Rivera, 1998](https://doi.org/10.2202/1558-3708.1041)). Duan's augmented GARCH unifies many power and nonlinear transforms and connects them to stochastic-volatility diffusion limits ([Duan, 1997](https://doi.org/10.1016/S0304-4076(97)00009-2)). These papers have sound theory.

Stochastic recurrence equations are more valuable as scivol's internal correctness framework than as a user-facing model. Straumann and Mikosch use contraction conditions to establish existence, uniqueness, invertibility, and QML theory for a broad class that includes GARCH, asymmetric GARCH, and EGARCH ([Straumann and Mikosch, 2006](https://doi.org/10.1214/009053606000000803)). Each new nonlinear recursion should document its contraction or stationarity condition and test that initialization error decays at the predicted rate. A generic `StochasticRecurrence()` constructor would shift those proof obligations to users and weaken scivol's correctness promise.

Neither should be a near-term user-facing family. APARCH, GJR-GARCH, and EGARCH already cover the common sign and power asymmetries. A generic augmented-GARCH API would allow many parameter combinations with obscure positivity and stationarity conditions, while most users need a small list of named, tested specifications. Smooth-transition GARCH is more defensible because it asks whether asymmetry changes gradually and can incorporate a transition variable such as volume. It still has nuisance parameters unidentified under the no-transition null, a flat likelihood when the transition slope is small, and weak empirical demand compared with MS-GARCH.

Recommended policy:

- Do not add TGARCH, ZARCH, NAGARCH, AVGARCH, or other named forms already nested exactly or nearly by APARCH/GJR/EGARCH unless compatibility requires aliases.
- Use internal generalized recursions to share kernels, but expose named components with written constraints and DGP tests.
- Add smooth-transition GARCH only with an LM specification test, bounded transition parameters, and profile diagnostics.
- Treat neural-GARCH hybrids as benchmarks in research workflows, not core statistical models, until their likelihood, uncertainty, and reproducibility contracts are as clear as the econometric models.

## 9. Semiparametric innovations and EVT tails

### Semiparametric density estimation

Engle and Gonzalez-Rivera combined parametric conditional mean and variance recursions with a nonparametric innovation density and recovered part of the efficiency lost by Gaussian QML ([Engle and Gonzalez-Rivera, 1991](https://doi.org/10.2307/1391236)). Drost and Klaassen derived efficient estimation for semiparametric GARCH and showed that the efficiency loss from plain QML can be material ([Drost and Klaassen, 1997](https://doi.org/10.1016/S0304-4076(97)00042-0)).

A `KernelDensity()` component sounds consistent with scivol's density role, but the estimation is iterative: volatility parameters determine standardized residuals, the residual density changes the efficient score, and bandwidth or basis complexity changes the result. This is more than swapping a closed-form likelihood. The result must retain the estimated density and support density, quantile, simulation, and tail diagnostics.

Recommendation: offer filtered historical simulation and an empirical standardized-residual distribution first. Add adaptive semiparametric estimation later if benchmarks show a meaningful efficiency or density-forecast gain over Student-t, Skew-t, generalized hyperbolic, and EVT tails.

### Conditional EVT

McNeil and Frey's conditional EVT procedure fits a volatility filter, then applies peaks over threshold to standardized residual tails. It improved one-day VaR and ES and outperformed square-root-of-time scaling for multiday risk in their study ([McNeil and Frey, 2000](https://doi.org/10.1016/S0927-5398(00)00012-8)). The method remains an active benchmark. Recent work extends GARCH-EVT with covariates and regularization ([James, Leung, Leung, and Prokhorov, 2023](https://doi.org/10.1016/j.jempfin.2023.01.002)).

EVT should be a workflow, not a return density:

```python
fit = (ARMA(1, 0) + GJRGARCH(1, 1) + StudentT()).fit(returns)
tail = fit.fit_evt_tail(side="left", threshold="auto", min_exceedances=100)
tail.var(level=0.995, horizon=1)
tail.expected_shortfall(level=0.975, horizon=1)
```

This preserves the center distribution and estimates a generalized Pareto tail only where asymptotic tail theory applies. Support left, right, and separate two-sided tails. Offer fixed quantile and automatic threshold candidates, but never hide the selected threshold or exceedance count.

Required outputs and diagnostics:

- threshold, number and dates of exceedances, generalized Pareto shape and scale, covariance, and threshold-stability plot;
- mean-excess and tail QQ plots, probability integral transform checks for the center, and residual extremal-dependence checks;
- conditional VaR coverage and independence tests, ES backtests, and rolling exceedance plots;
- parametric or semiparametric bootstrap intervals that propagate first-stage volatility-filter uncertainty;
- simulation for multiday VaR and ES rather than square-root-of-time scaling.

Threshold selection dominates the statistical error. An "auto" mode should return a sensitivity range, not a single unexplained optimum. EVT adds more practical value than several extra closed-form innovation densities because it targets the part of the distribution used in stress and capital calculations.

## 10. Cross-model correctness requirements

The following requirements matter more than another model name.

1. **Make observational equivalence visible.** Long memory, slow components, regimes, and breaks can produce similar persistence. Provide matched forecast comparisons and stability diagnostics instead of declaring one explanation from in-sample likelihood alone.
2. **Keep paths independent.** FIGARCH, MS-GARCH, GAS, and SV should optimize their stated formulations. Do not force agreement by routing them through a common hidden fit.
3. **Expose initialization and approximation.** Report FIGARCH truncation, SV particle count and Monte Carlo error, regime multistarts, MIDAS release lags, and EVT thresholds in `fit_info`.
4. **Return the latent object users care about.** Component models need short and long variance; MS-GARCH needs regime probabilities; SV needs filtered and smoothed states; GAS needs scores; EVT needs tail parameters and exceedances.
5. **Use model-specific DGP tests.** Ordinary convergence tests will miss label switching, filter degeneracy, truncation bias, and false long memory from breaks.
6. **Compare forecasts at the horizon the model targets.** MF2-GARCH and components should be judged at weeks and months; EVT at tail levels; ACD in event time; rough models against realized volatility; one-day QLike alone is not enough.

## Suggested roadmap

### Phase A: high value within the current architecture

- Conditional EVT workflow with uncertainty propagation and VaR/ES backtests.
- MF2-GARCH with decomposed forecasts.
- Score-driven Student-t scale model.
- GARCH-in-mean as a dependency-aware mean component.

### Phase B: alternative persistence explanations

- FIGARCH with truncation sensitivity and break warnings.
- Two-state independent MS-GARCH with deterministic multistart and bootstrap regime comparison.
- General GARCH-MIDAS-X with publication-lag-safe data handling.

### Phase C: a second estimation engine

- Basic univariate SV with leverage and Student-t observations.
- Filtering, smoothing, predictive simulation, and state uncertainty as reusable infrastructure.
- Experimental fractional OU or rough-SV estimation only after realized-volatility inputs are first class.

### Phase D: adjacent positive-series models

- MEM for realized variance, range, and volume.
- ACD with seasonality, censoring, and event-time support.
- Vector MEM only after the univariate positive-response contract is stable.

## Primary source ledger

### Foundational sources

- Baillie, Bollerslev, and Mikkelsen (1996), ["Fractionally Integrated Generalized Autoregressive Conditional Heteroskedasticity"](https://doi.org/10.1016/S0304-4076(95)01749-6). FIGARCH definition, approximate MLE, exchange-rate application.
- Bollerslev and Mikkelsen (1996), ["Modeling and Pricing Long Memory in Stock Market Volatility"](https://doi.org/10.1016/0304-4076(95)01736-4). FIEGARCH, QML evidence, option-pricing application.
- Davidson (2004), ["Moment and Memory Properties of Linear Conditional Heteroscedasticity Models, and a New Model"](https://doi.org/10.1198/073500103288619359). HYGARCH and moment conditions.
- Engle and Lee (1999), ["A Long-Run and Short-Run Component Model of Stock Return Volatility"](https://doi.org/10.1093/oso/9780198296836.003.0020). Component GARCH.
- Engle, Ghysels, and Sohn (2013), ["Stock Market Volatility and Macroeconomic Fundamentals"](https://doi.org/10.1162/REST_a_00300). GARCH-MIDAS and macroeconomic drivers of long volatility.
- Engle, Lilien, and Robins (1987), ["Estimating Time Varying Risk Premia in the Term Structure: The ARCH-M Model"](https://doi.org/10.2307/1913242). ARCH-in-mean.
- Harvey, Ruiz, and Shephard (1994), ["Multivariate Stochastic Variance Models"](https://doi.org/10.2307/2297980). Latent stochastic variance and state-space treatment.
- Kim, Shephard, and Chib (1998), ["Stochastic Volatility: Likelihood Inference and Comparison with ARCH Models"](https://doi.org/10.1111/1467-937X.00050). Mixture-based Bayesian likelihood and diagnostics.
- Durbin and Koopman (1998), ["Estimation of Stochastic Volatility Models via Monte Carlo Maximum Likelihood"](https://doi.org/10.1016/S0304-4076(98)00016-5). Simulation likelihood.
- Hamilton and Susmel (1994), ["Autoregressive Conditional Heteroskedasticity and Changes in Regime"](https://doi.org/10.1016/0304-4076(94)90067-1). Switching ARCH.
- Gray (1996), ["Modeling the Conditional Distribution of Interest Rates as a Regime-Switching Process"](https://doi.org/10.1016/0304-405X(96)00875-6). Tractable regime aggregation.
- Haas, Mittnik, and Paolella (2004), ["A New Approach to Markov-Switching GARCH Models"](https://doi.org/10.1093/jjfinec/nbh020). Independent regime-specific GARCH processes.
- Creal, Koopman, and Lucas (2013), ["Generalized Autoregressive Score Models with Applications"](https://doi.org/10.1002/jae.1279). GAS framework and score scaling.
- Engle and Russell (1998), ["Autoregressive Conditional Duration: A New Model for Irregularly Spaced Transaction Data"](https://doi.org/10.2307/2999632). ACD.
- Engle (2002), ["New Frontiers for ARCH Models"](https://doi.org/10.1002/jae.683). MEM for nonnegative processes.
- Engle and Gallo (2006), ["A Multiple Indicators Model for Volatility Using Intra-Daily Data"](https://doi.org/10.1016/j.jeconom.2005.01.018). Multivariate MEM for volatility proxies.
- Gonzalez-Rivera (1998), ["Smooth-Transition GARCH Models"](https://doi.org/10.2202/1558-3708.1041). Continuous regime transition and specification testing.
- Duan (1997), ["Augmented GARCH(p,q) Process and Its Diffusion Limit"](https://doi.org/10.1016/S0304-4076(97)00009-2). General nonlinear GARCH class.
- Straumann and Mikosch (2006), ["Quasi-Maximum-Likelihood Estimation in Conditionally Heteroscedastic Time Series: A Stochastic Recurrence Equations Approach"](https://doi.org/10.1214/009053606000000803). Stationarity, invertibility, and QML theory for nonlinear recursions.
- Engle and Gonzalez-Rivera (1991), ["Semiparametric ARCH Models"](https://doi.org/10.2307/1391236). Nonparametric innovation density with parametric moments.
- Drost and Klaassen (1997), ["Efficient Estimation in Semiparametric GARCH Models"](https://doi.org/10.1016/S0304-4076(97)00042-0). Semiparametric efficiency.
- McNeil and Frey (2000), ["Estimation of Tail-Related Risk Measures for Heteroscedastic Financial Time Series: An Extreme Value Approach"](https://doi.org/10.1016/S0927-5398(00)00012-8). Conditional GARCH-EVT for VaR and ES.
- Diebold and Inoue (2001), ["Long Memory and Regime Switching"](https://doi.org/10.1016/S0304-4076(01)00073-2). Identification warning for apparent long memory.
- Lamoureux and Lastrapes (1990), ["Persistence in Variance, Structural Change, and the GARCH Model"](https://doi.org/10.1080/07350015.1990.10509794). Structural shifts inflate GARCH persistence.
- Gatheral, Jaisson, and Rosenbaum (2018), ["Volatility Is Rough"](https://doi.org/10.1080/14697688.2017.1393551). Empirical roughness and its market-microstructure motivation.

### Primary developments from 2020 to 2025

- Conrad and Kleen (2020), ["Two Are Better Than One: Volatility Forecasting Using Multiplicative Component GARCH-MIDAS Models"](https://doi.org/10.1002/jae.2742). Multiple slow volatility components and real-time forecasting.
- Fang, Lee, and Su (2020), ["Predicting the Long-Term Stock Market Volatility: A GARCH-MIDAS Model with Variable Selection"](https://doi.org/10.1016/j.jempfin.2020.05.007). Adaptive-lasso selection in the long component.
- Opschoor and Lucas (2021), ["Observation-Driven Models for Realized Variances and Overnight Returns Applied to Value-at-Risk and Expected Shortfall Forecasting"](https://doi.org/10.1016/j.ijforecast.2020.07.009). Robust score updates and tail-risk forecasting.
- Alfeus and Nikitopoulos (2022), ["Forecasting Volatility in Commodity Markets with Long-Memory Models"](https://doi.org/10.1016/j.jcomm.2022.100248). FIGARCH, HAR, and fractional-SV forecasts across 22 commodities.
- Cipollini and Gallo (2025, first circulated in 2021), ["Multiplicative Error Models: 20 Years On"](https://doi.org/10.1016/j.ecosta.2022.05.005). Current MEM scope and empirical evidence.
- Bolko, Christensen, Pakkanen, and Veliyev (2023), ["A GMM Approach to Estimate the Roughness of Stochastic Volatility"](https://doi.org/10.1016/j.jeconom.2022.06.009). Measurement-error-corrected roughness estimation.
- Wang, Xiao, and Yu (2023), ["Modeling and Forecasting Realized Volatility with the Fractional Ornstein-Uhlenbeck Process"](https://doi.org/10.1016/j.jeconom.2021.08.001). Closed-form estimation and realized-volatility forecasts.
- Segnon, Gupta, and Wilfling (2024), ["Forecasting Stock Market Volatility with Regime-Switching GARCH-MIDAS: The Role of Geopolitical Risks"](https://doi.org/10.1016/j.ijforecast.2022.11.007). Long-sample regime and MIDAS forecast comparison.
- James, Leung, Leung, and Prokhorov (2023), ["Forecasting Tail Risk Measures for Financial Time Series: An Extreme Value Approach with Covariates"](https://doi.org/10.1016/j.jempfin.2023.01.002). Regularized conditional EVT.
- Ballestra, D'Innocenzo, and Guizzardi (2024), ["Score-Driven Modeling with Jumps: An Application to S&P 500 Returns and Options"](https://doi.org/10.1093/jjfinec/nbad001). Joint score-driven volatility and jump intensity.
- Conrad and Engle (2025), ["Modelling Volatility Cycles: The MF2-GARCH Model"](https://doi.org/10.1002/jae.3118). Stationary component model with scalable long-horizon forecast evidence.
