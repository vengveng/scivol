# Forecast, risk, portfolio, and production workflows

## Executive finding

scivol already has much of the numerical foundation: fitted and fixed-parameter workflows, conditional variance forecasts, distribution quantiles, univariate simulation, DCC/CCC covariance forecasts and scenarios, rolling fits, rolling out-of-sample selection, Diebold-Mariano winner comparisons, and PIT-based diagnostics. The largest product gap is the layer between a fitted model and a defensible decision. Users still need explicit predictive distributions, cumulative-horizon risk, uncertainty bands, aligned walk-forward evaluation, portfolio transformations, tail-risk backtests, and monitoring artifacts.

The main design principle should be **semantic precision before convenience**. A forecast of expected conditional variance at date \(t+h\), the variance of the cumulative return through \(t+h\), a prediction interval for that return, and a confidence interval for the variance forecast are different objects. A fast library that returns one under the name of another is worse than a slower library.

## 1. Forecast objects: distinguish four targets

For origin \(t\) and horizon \(h\), scivol should distinguish:

1. **State/path forecast:** \(E_t[h_{t+h}]\), the expected conditional innovation variance at the future date.
2. **Marginal predictive distribution:** \(F_t(r_{t+h})\), which integrates over the intervening volatility path.
3. **Aggregate-horizon distribution:** \(F_t(\sum_{j=1}^h r_{t+j})\), required for an \(h\)-day P&L, VaR, or ES.
4. **Estimation uncertainty:** uncertainty in any of the above because parameters were estimated.

These coincide only in special cases. For GARCH-type models, an analytic recursion often gives the first target. Beyond one step, the predictive return distribution is normally a mixture over random future variance states. Scaling the fitted innovation quantile by \(\sqrt{E_t[h_{t+h}]}\) does not generally give its quantile. Aggregating daily variance forecasts also does not automatically give cumulative-return variance when the mean model induces serial dependence.

This matters for the current API. `ForecastResult.quantile()` and `.var()` are defensible as one-step location-scale calculations. At horizons above one, their use of a point variance path can be mistaken for a full predictive quantile. For an ARMA-volatility result, the implementation also switches to a Normal quantile whenever total forecast variance differs from residual variance, even if the fitted innovation density is heavy-tailed; that approximation is not recorded in the result. The API should either restrict analytic quantiles to one step or label the approximation and offer simulation as the exact model-implied route.

Recommended surface:

```python
fc = result.forecast(
    horizon=20,
    method="analytic",          # analytic | simulation | residual_bootstrap
    target="state",             # state | marginal | cumulative
)

pred = result.predictive(
    horizon=20,
    n_paths=100_000,
    innovations="parametric",  # parametric | empirical_iid | empirical_block
    parameter_uncertainty=False,
    seed=7,
)

pred.quantile([0.01, 0.05], at=20)
pred.interval(0.95, target="cumulative_return")
pred.paths                         # optional or lazily stored
pred.monte_carlo_error
pred.metadata                      # origin, target, method, seed, draws, assumptions
```

The result must record whether a forecast is analytic or simulated. This is especially important because current APARCH and multi-step EGARCH forecasts use 4,000 internally simulated paths with a fixed seed, while the public result looks deterministic and gives no Monte Carlo error. Expose `method`, `n_paths`, `seed`, and simulation error; let users trade speed for precision.

### Forecast intervals

Use names that prevent three intervals from being confused:

- `prediction_interval(target="return")`: uncertainty about a future observation.
- `prediction_interval(target="variance_state")`: model-implied distribution of future latent variance, obtained from paths.
- `confidence_interval(target="expected_variance")`: parameter-estimation uncertainty around a point forecast.

Pascual, Romo, and Ruiz propose bootstrap predictive densities for returns and volatility that incorporate parameter uncertainty without imposing a parametric innovation law. Their work supports a bootstrap forecast mode, but parameter resampling must refit each bootstrap sample; merely resampling innovations conditional on fixed estimates captures process risk, not estimation risk ([Pascual, Romo, and Ruiz, 2006](https://doi.org/10.1016/j.csda.2004.12.008)).

### Iterated, direct, and scaled multi-period forecasts

An iterated GARCH forecast and a model estimated directly for an \(h\)-period target answer different questions. A useful API should represent the target frequency and aggregation rule rather than silently apply square-root-of-time scaling. A broad empirical review finds that iterated GARCH often beats direct GARCH, while direct realized-volatility specifications can beat their iterated variants; MIDAS-type models perform well across longer horizons ([Ghysels, Plazzi, Valkanov, Rubia, and Dossani, 2019](https://doi.org/10.1146/annurev-financial-110217-022808)). This argues for supporting direct-horizon model integrations later, not for disguising a scaled daily forecast as a direct forecast.

## 2. Conditional simulation and bootstrap paths

`spec.simulate()` generates an unconditional artificial series after a burn-in. Risk work instead needs many paths conditional on the last filtered state. DCC `result.scenarios()` supplies many one-step draws from one origin, while `result.simulate()` evolves one multivariate path. The missing primitive is a tensor of conditional paths:

```python
paths = result.simulate_forecast(
    horizon=10,
    n_paths=200_000,
    innovations="filtered_historical_rows",
    seed=11,
)
# paths.returns: (n_paths, horizon, n_assets)
# paths.variance, paths.correlation, paths.covariance
```

Required modes:

- **Parametric:** draw from the fitted marginal densities and copula.
- **Residual bootstrap:** resample centered, standardized innovations and recursively re-inflate them through the variance process.
- **Filtered historical simulation (FHS):** in a multivariate portfolio, resample contemporaneous standardized-residual vectors as rows so historical cross-sectional dependence is retained, then update each variance path. Barone-Adesi, Giannopoulos, and Vosper use this framework for nonlinear portfolios and reprice positions at each scenario node ([2002](https://doi.org/10.1111/1468-036X.00175)).
- **Block residual bootstrap:** optional when standardized residual diagnostics still show dependence. It is a robustness mode, not a cure for a misspecified variance model.
- **Parameter bootstrap:** refit within the bootstrap to add estimation risk.

The existing DCC `innovations="empirical"` mode maps fitted-copula uniforms through separate empirical marginal quantiles. That is a useful semiparametric copula simulation, but it is not the row-resampling FHS algorithm and should not be named or documented as such.

Tail calculations need numerical diagnostics: number of draws, effective tail observations, quantile interpolation rule, Monte Carlo standard error or repeated-batch error, and seed. At a 1% tail, a nominally large simulation can still leave few observations for ES.

## 3. Rolling and expanding evaluation

`fit_rolling()` currently returns fitted windows. A production-grade workflow should also return forecasts aligned with the observations that became available after each origin. Add a public walk-forward evaluator:

```python
bt = spec.walk_forward(
    data,
    initial=1000,
    window="rolling",            # rolling | expanding
    refit_every=5,
    horizons=[1, 5, 20],
    method="simulation",
    realized_proxy=realized_var,
)

bt.forecasts
bt.actuals
bt.losses(["qlike", "mse"])
bt.compare(test="dm", hac="auto")
bt.model_confidence_set(level=0.90)
```

Correctness requirements:

- Freeze every input at its real-time availability date. Future regressors require an explicit supplied path; revised realized measures need vintage handling if relevant.
- Separate **refitting parameters** from **updating the filtered state** with newly observed returns between refits.
- Preserve timestamp, forecast origin, target timestamp, horizon, window bounds, parameter vintage, scale transform, and failure status in tidy output.
- Define whether an \(h\)-day target is overlapping or non-overlapping. Overlapping losses are serially dependent, so equal-accuracy tests need horizon-aware HAC or a block bootstrap.
- Keep failed fits and fallback decisions visible. A warm start may change computation time and convergence basin; it must never change what data enter the fit.
- Score variance forecasts against a declared proxy. Squared daily returns are noisy; realized measures can improve evaluation. Patton derives loss functions whose rankings remain valid with conditionally unbiased noisy proxies, including QLIKE ([Patton, 2011](https://doi.org/10.1016/j.jeconom.2010.03.034)).

Diebold and Mariano allow general loss differentials and dependent forecast errors for pairwise equal-accuracy tests ([1995](https://doi.org/10.1080/07350015.1995.10524599)). Giacomini and White evaluate forecasting *methods* under possible misspecification and explicitly preserve finite estimation-window effects ([2006](https://doi.org/10.1111/j.1468-0262.2006.00718.x)). West covers predictive inference when estimated parameters affect forecasts ([1996](https://www.jstor.org/stable/2171956)). The distinctions should appear in method names and documentation; a generic `p_value` is insufficient.

For many candidates, return a model confidence set rather than crown the smallest sample loss. The MCS retains all models that cannot be distinguished from the best at a chosen confidence level ([Hansen, Lunde, and Nason, 2011](https://doi.org/10.3982/ECTA5771)). scivol already uses MCS in selection; the same engine should work on public forecast panels.

## 4. VaR and expected shortfall

VaR and ES belong close to the predictive distribution because they are functionals of it. The low-level methods should remain instrument-neutral:

```python
risk = pred.risk(level=[0.95, 0.975, 0.99], convention="loss_positive")
risk.var
risk.expected_shortfall
risk.horizon
risk.target                 # marginal_return | cumulative_return | portfolio_pnl
risk.method                 # analytic | simulated | FHS
```

One-step analytic VaR and ES are appropriate for supported standardized location-scale densities when their moments exist. Cumulative and nonlinear-portfolio risk should use paths. The library must validate moment existence: a fitted heavy-tailed law can have a finite quantile while ES, variance, or higher moments do not exist.

Do not overload `level`. Some literature uses \(\alpha\) for the lower-tail probability and some uses it for the confidence level. Accept an explicit convention or expose `tail_probability`; serialize both. Also record whether reported VaR is a negative return quantile or a positive loss number. The current `.var()` returns the negative return quantile, which is convenient but ambiguous without the convention.

Expected shortfall is a coherent tail-risk measure under the usual definitions ([Acerbi and Tasche, 2002](https://doi.org/10.1111/1468-0300.00091)). The Basel market-risk framework uses daily 97.5% ES with liquidity-horizon adjustments, stressed calibration, desk-level backtesting, and P&L attribution ([BCBS, 2019](https://www.bis.org/bcbs/publ/d457.htm)). A generic `expected_shortfall()` method is valuable, but scivol should not call it “FRTB compliant”: regulatory ES needs portfolio classification, liquidity horizons, stress-period selection, modellability tests, P&L rules, and governance outside a volatility kernel.

### Tail-risk backtesting

Add a `RiskBacktestResult` with the exception series, sample size, observed and expected counts, test statistics, finite-sample warnings, and plots/data. At minimum:

- Kupiec unconditional coverage. Warn about weak power for rare events and short samples; Kupiec's original study emphasizes this limitation ([1995](https://fraser.stlouisfed.org/title/finance-economics-discussion-series-1491/techniques-verifying-accuracy-risk-measurement-models-717921)).
- Christoffersen independence and conditional coverage, because the right number of clustered breaches still indicates a failed dynamic model ([1998](https://users.ssc.wisc.edu/~behansen/718/Christoffersen1998.pdf)).
- Engle-Manganelli dynamic quantile test against richer information sets ([2004](https://www.simonemanganelli.org/Simone/Research_files/caviarPublished.pdf)).
- Joint VaR/ES strictly consistent scores and comparative tests. ES alone is not elicitable, but VaR and ES are jointly elicitable under mild conditions ([Fissler and Ziegel, 2016](https://doi.org/10.1214/16-AOS1439)).
- Optional Acerbi-Szekely model-independent ES tests ([2014](https://www.msci.com/documents/10199/238916/Research_Insight_Backtesting_Expecting_Shortfall_December_2014.pdf)).

A “pass” is evidence at a specified sample size and power, not certification. Return diagnostics, not a single green/red flag.

## 5. Portfolio covariance, beta, and hedging

DCC already returns the forecast correlation and residual covariance path. Build thin, label-aware portfolio transformations on top:

```python
pf = dcc_result.forecast(20).portfolio(weights)
pf.variance_path
pf.aggregate_variance(horizon=20)
pf.volatility
pf.marginal_risk
pf.component_risk

dcc_result.conditional_beta(asset="AAPL", market="SPY", horizon=1)
dcc_result.minimum_variance_hedge(asset="EUR", hedge="EUR_FUT", horizon=1)
```

For asset \(i\) and hedge \(j\), the one-step conditional minimum-variance hedge ratio is \(H_{ij,t+1}/H_{jj,t+1}\); conditional market beta has the same covariance-over-market-variance form. Dynamic GARCH hedge ratios have a long empirical lineage ([Kroner and Sultan, 1993](https://www.cambridge.org/core/journals/journal-of-financial-and-quantitative-analysis/article/timevarying-distributions-and-dynamic-hedging-with-foreign-currency-futures/889607832350D7FE09163B5438C11122)).

Correctness safeguards:

- Align weights by stored column names and reject missing, duplicate, or extra assets.
- Distinguish a future-date covariance matrix from covariance of the cumulative \(h\)-period return.
- Specify fixed weights, daily rebalancing, or buy-and-hold weights. These generate different multi-period P&L distributions.
- Verify symmetry, unit correlation diagonals, positive variances, and positive semidefiniteness after every forecast or stress transform. Report any eigenvalue repair rather than applying it silently.
- For nonlinear instruments, accept a user repricing callback over risk-factor paths; a delta-normal shortcut cannot represent convexity or volatility-surface exposure.
- Return hedge turnover and let applications add transaction costs, funding, integer contracts, and position limits.

Engle's DCC model was designed to combine flexible univariate volatility models with parsimonious time-varying correlations ([2002](https://archive.nyu.edu/handle/2451/26879)). That decomposition makes scivol a natural covariance engine, but portfolio optimization itself should remain a workflow layer: constraints, estimators of expected return, costs, and business rules extend beyond volatility modeling.

### Options inputs and volatility derivatives

A physical-measure volatility forecast is a useful option-risk input, but it is not an option price or implied volatility. Pricing requires a risk-neutral measure, a variance-risk premium or pricing kernel, rates, dividends, contract conventions, and a pricer. Duan's GARCH option model obtains a pricing relation under explicit preference and distribution assumptions ([1995](https://doi.org/10.1111/j.1467-9965.1995.tb00099.x)); Heston and Nandi derive a closed-form price for a particular GARCH process under its pricing dynamics ([2000](https://doi.org/10.1093/rfs/13.3.585)). Neither result licenses plugging a historical \(P\)-measure forecast directly into Black-Scholes and calling the output model-consistent.

The core forecast object should therefore expose the conditional variance state, physical-measure paths, innovation law, horizon aggregation, and parameter metadata that an option integration needs. It should not embed a universal \(P\)-to-\(Q\) conversion. An integration can then calibrate risk-neutral parameters to an option surface, compute the variance-risk premium, price a contract, and return Greeks and calibration residuals. Portfolio risk paths should also accept a full-revaluation callback so that spot, rate, time decay, and implied-volatility shocks reach option P&L together.

## 6. Scenario and stress testing

Model-based scenario simulation and stress testing should share path machinery but remain conceptually separate. A predictive simulation assigns probabilities under the fitted model; a stress scenario asks what happens under a deliberately adverse state and may have no defensible probability.

Proposed objects:

```python
scenario = StressScenario(
    initial_shock={"SPX": -0.25},
    correlation="historical_2020_peak",
    volatility_multiplier=2.0,
    exogenous_path=macro_path,
    horizon=20,
)
stress = dcc_result.stress(scenario, portfolio=reprice)
```

Support historical replay, hypothetical shocks, parameter/state shocks, hybrid scenarios, sensitivity grids, and reverse stress. Keep all shocks explicit and serializable. Never force stressed correlations by elementwise editing without restoring and reporting positive semidefiniteness.

The Basel Committee treats stress testing as a core risk-management tool and explicitly includes sensitivity, scenario, enterprise-wide, and reverse stress tests ([BCBS, 2018](https://www.bis.org/bcbs/publ/d450.htm)). The Federal Reserve's market-shock framework uses both historical and hypothetical risk-factor shocks and covers equity, FX, rates, implied volatility, commodities, and credit spreads ([Federal Reserve scenario framework](https://www.federalreserve.gov/frrs/regulations/appendix-a-policy-statement-on-the-scenario-design-framework-for-stress-testing.htm)). scivol should provide the statistical state-transition engine and an extensible scenario schema; institutional capital aggregation belongs in integrations.

## 7. Volatility targeting and forecast combinations

### Volatility targeting

A small workflow can turn a one-step portfolio variance forecast into exposure, for example \(w_t=\sigma^*/\hat\sigma_{t+1|t}\), with leverage caps and lagged information. A separate “volatility-managed portfolio” rule scales by inverse variance; Moreira and Muir study that rule across factors ([2017](https://doi.org/10.1111/jofi.12513)). Do not conflate the two.

The workflow must require:

- target-frequency and annualization conventions;
- a forecast formed before the return it scales;
- leverage floor/cap, cash return, financing cost, turnover, and rebalance schedule;
- missing/failed-forecast policy;
- both gross and net results.

This belongs in `scivol.workflows`, not in model kernels. It is useful as an end-to-end example and economic evaluation, but its portfolio assumptions should never contaminate estimation.

### Forecast combinations

Combining forecasts can reduce model-selection risk; the foundational combination result is Bates and Granger ([1969](https://doi.org/10.1057/jors.1969.103)). Offer equal weights first, then nonnegative simplex weights estimated on a past-only validation window, inverse-loss weights, and MCS-restricted averaging.

Combine the object that the loss evaluates. Averaging variances differs from averaging volatilities. A mixture of predictive densities is valid, while averaging quantiles generally does not produce the mixture quantile. Estimate weights separately by horizon and prevent look-ahead. Report weight concentration and turnover; sophisticated estimated weights can be unstable in short samples.

## 8. Production monitoring and audit trail

The current in-sample PIT/DGT and Ljung-Box diagnostics are useful development checks. Production monitoring should operate on the sequence of genuinely out-of-sample forecasts and outcomes.

Recommended `ForecastMonitor` outputs:

- input freshness, gaps, duplicates, nonfinite values, scale changes, and calendar anomalies;
- fit convergence, optimizer path, boundary proximity, persistence, Hessian/OPG condition, elapsed time, warm-start source, and fallback use;
- rolling QLIKE and proper density score against declared benchmarks;
- PIT histogram and serial tests on out-of-sample PITs;
- VaR exception rates, independence, duration/clustering, and joint VaR/ES scores;
- covariance symmetry, minimum eigenvalue, condition number, correlation extremes, and repair count;
- parameter and forecast drift, challenger-model performance, and threshold breaches;
- full provenance: model/spec UID, library and C-extension version, data window/hash, timestamp, seed, method, and configuration.

Diebold, Gunther, and Tay show that correctly specified one-step density forecasts imply i.i.d. uniform PITs ([1998](https://doi.org/10.2307/2527349)). Berkowitz provides a likelihood-based density-forecast test with risk-management applications ([2001](https://doi.org/10.1198/07350010152596718)). Engle and Ng's sign and size-bias tests identify asymmetric volatility responses omitted by a fitted model ([1993](https://doi.org/10.1111/j.1540-6261.1993.tb05127.x)). Add these alongside raw and squared standardized-residual portmanteau tests.

The current `diagnostic_tests()` applies a Pearson cell-uniformity test and Ljung-Box tests in sample. Its p-values do not visibly account for parameters estimated on the same observations, its Ljung-Box degrees of freedom are not adjusted for fitted dynamics, and a fixed 40-cell default can leave sparse expected counts in small samples. Label these as approximate in-sample diagnostics, validate input-size conditions, and prioritize out-of-sample monitoring or a parameter/bootstrap-adjusted calibration mode.

Current U.S. interagency model-risk guidance emphasizes ongoing monitoring, outcomes analysis, input and code verification, benchmarking, change control, limitations, and responses to deterioration ([Federal Reserve, 2026](https://www.federalreserve.gov/frrs/guidance/supervisory-guidance-on-model-risk-management.htm)). A scivol monitor should therefore return machine-readable measurements and decisions, not print-only diagnostics.

## 9. Recommended delivery order

### P0: statistical correctness and reusable primitives

1. Split state, marginal, and cumulative forecast targets; restrict or clearly label multi-step analytic quantiles.
2. Add conditional multi-path simulation for every shipped univariate and multivariate model.
3. Expose analytic/simulation/bootstrap method metadata, path count, seed, and Monte Carlo error.
4. Add one-step analytic and path-based multi-step VaR/ES with explicit sign and tail conventions.
5. Add an aligned `walk_forward()` result with actuals, losses, failures, origins, and no-look-ahead guarantees.
6. Add portfolio covariance aggregation, variance, beta, and hedge-ratio transformations with label and PSD checks.

### P1: evaluation and uncertainty

7. Add prediction intervals and parameter-bootstrap confidence intervals.
8. Add Kupiec, Christoffersen, DQ, and joint VaR/ES backtests with power/sample warnings.
9. Generalize DM and MCS to public forecast panels, overlapping horizons, user losses, and block/HAC inference.
10. Add true multivariate FHS row resampling and nonlinear P&L callbacks.

### P2: decision and operations layer

11. Add stress-scenario schemas, historical replay, and reverse-stress hooks.
12. Add forecast combinations and volatility-targeting workflows.
13. Add production-monitor artifacts, thresholds, provenance, and export.

The core should own model-consistent recursion, prediction, simulation, distribution functionals, covariance, and statistical evaluation. A `workflows` namespace should own portfolio transformations, hedging, targeting, and generic stress orchestration. Market data, instrument pricing, regulatory capital, scheduling, storage, and alerts belong in integrations. This boundary keeps scivol complete as a volatility engine without turning it into an opaque trading or bank-capital platform.

## Source table

| Source | Type | Workflow implication |
|---|---|---|
| [Pascual, Romo, and Ruiz (2006), *Bootstrap prediction for returns and volatilities in GARCH models*](https://doi.org/10.1016/j.csda.2004.12.008) | Primary research | Bootstrap predictive densities; distinguish process and parameter uncertainty. |
| [Ghysels, Plazzi, Valkanov, Rubia, and Dossani (2019), *Direct Versus Iterated Multiperiod Volatility Forecasts*](https://doi.org/10.1146/annurev-financial-110217-022808) | Authoritative review and comparison | Make target horizon and direct/iterated/scaled method explicit. |
| [Barone-Adesi, Giannopoulos, and Vosper (2002), *Backtesting Derivative Portfolios with Filtered Historical Simulation*](https://doi.org/10.1111/1468-036X.00175) | Primary research | FHS with recursive volatility and full repricing of nonlinear portfolios. |
| [Pritsker (2001), *The Hidden Dangers of Historical Simulation*](https://www.federalreserve.gov/pubs/feds/2001/200127/200127abs.html) | Central-bank working paper | Warn about short tail samples, window choice, and time-varying dependence in HS/FHS. |
| [Patton (2011), *Volatility Forecast Comparison Using Imperfect Volatility Proxies*](https://doi.org/10.1016/j.jeconom.2010.03.034) | Primary research | Use proxy-robust loss functions such as QLIKE and declare the volatility proxy. |
| [Diebold and Mariano (1995), *Comparing Predictive Accuracy*](https://doi.org/10.1080/07350015.1995.10524599) | Primary research | Pairwise forecast comparison under general loss and dependent errors. |
| [West (1996), *Asymptotic Inference about Predictive Ability*](https://www.jstor.org/stable/2171956) | Primary research | Account for estimated parameters in forecast evaluation. |
| [Giacomini and White (2006), *Tests of Conditional Predictive Ability*](https://doi.org/10.1111/j.1468-0262.2006.00718.x) | Primary research | Compare forecasting procedures under misspecification and fixed rolling windows. |
| [Hansen, Lunde, and Nason (2011), *The Model Confidence Set*](https://doi.org/10.3982/ECTA5771) | Primary research | Return a set of statistically indistinguishable best methods. |
| [Acerbi and Tasche (2002), *Expected Shortfall: A Natural Coherent Alternative to Value at Risk*](https://doi.org/10.1111/1468-0300.00091) | Primary research | Define and validate ES consistently, including discontinuous distributions. |
| [BCBS (2019), *Minimum capital requirements for market risk*](https://www.bis.org/bcbs/publ/d457.htm) | Current authoritative standard | Regulatory ES needs stress calibration, liquidity horizons, backtesting, and P&L attribution beyond generic ES. |
| [Kupiec (1995), *Techniques for Verifying the Accuracy of Risk Measurement Models*](https://fraser.stlouisfed.org/title/finance-economics-discussion-series-1491/techniques-verifying-accuracy-risk-measurement-models-717921) | Primary central-bank research | VaR unconditional-coverage test and explicit low-power warning. |
| [Christoffersen (1998), *Evaluating Interval Forecasts*](https://users.ssc.wisc.edu/~behansen/718/Christoffersen1998.pdf) | Primary research | Test breach independence as well as unconditional coverage. |
| [Engle and Manganelli (2004), *CAViaR*](https://www.simonemanganelli.org/Simone/Research_files/caviarPublished.pdf) | Primary research | Dynamic quantile test for conditional VaR calibration. |
| [Fissler and Ziegel (2016), *Higher Order Elicitability and Osband's Principle*](https://doi.org/10.1214/16-AOS1439) | Primary research | Evaluate VaR and ES jointly with strictly consistent scores. |
| [Acerbi and Szekely (2014), *Backtesting Expected Shortfall*](https://www.msci.com/documents/10199/238916/Research_Insight_Backtesting_Expecting_Shortfall_December_2014.pdf) | Primary practitioner research | Add model-independent ES backtests as optional diagnostics. |
| [Engle (2002), *Dynamic Conditional Correlation*](https://archive.nyu.edu/handle/2451/26879) | Primary research | Forecast covariance by combining marginal variances with dynamic correlations. |
| [Kroner and Sultan (1993), *Time-Varying Distributions and Dynamic Hedging with Foreign Currency Futures*](https://www.cambridge.org/core/journals/journal-of-financial-and-quantitative-analysis/article/timevarying-distributions-and-dynamic-hedging-with-foreign-currency-futures/889607832350D7FE09163B5438C11122) | Primary research | Conditional covariance supports dynamic minimum-variance hedge ratios. |
| [Duan (1995), *The GARCH Option Pricing Model*](https://doi.org/10.1111/j.1467-9965.1995.tb00099.x) | Primary research | GARCH option pricing needs an explicit risk-neutral valuation relation. |
| [Heston and Nandi (2000), *A Closed-Form GARCH Option Valuation Model*](https://doi.org/10.1093/rfs/13.3.585) | Primary research | A special GARCH specification admits closed-form pricing under defined pricing dynamics. |
| [BCBS (2018), *Stress testing principles*](https://www.bis.org/bcbs/publ/d450.htm) | Current authoritative guidance | Support sensitivity, scenario, enterprise-wide, and reverse stress concepts without assigning forecast probabilities. |
| [Federal Reserve, *Scenario Design Framework for Stress Testing*](https://www.federalreserve.gov/frrs/regulations/appendix-a-policy-statement-on-the-scenario-design-framework-for-stress-testing.htm) | Authoritative supervisory framework | Historical, hypothetical, and hybrid market shocks require broad risk-factor paths. |
| [Moreira and Muir (2017), *Volatility-Managed Portfolios*](https://doi.org/10.1111/jofi.12513) | Primary research | Provide an economic workflow while separating inverse-variance management from target-vol scaling. |
| [Bates and Granger (1969), *The Combination of Forecasts*](https://doi.org/10.1057/jors.1969.103) | Primary research | Start forecast combination with transparent, constrained methods. |
| [Diebold, Gunther, and Tay (1998), *Evaluating Density Forecasts*](https://doi.org/10.2307/2527349) | Primary research | Monitor out-of-sample PIT uniformity and independence. |
| [Berkowitz (2001), *Testing Density Forecasts, with Applications to Risk Management*](https://doi.org/10.1198/07350010152596718) | Primary research | Add likelihood-based density calibration tests. |
| [Engle and Ng (1993), *Measuring and Testing the Impact of News on Volatility*](https://doi.org/10.1111/j.1540-6261.1993.tb05127.x) | Primary research | Add sign, negative-size, positive-size, and joint bias diagnostics. |
| [Federal Reserve/OCC/FDIC (2026), *Supervisory Guidance on Model Risk Management*](https://www.federalreserve.gov/frrs/guidance/supervisory-guidance-on-model-risk-management.htm) | Current authoritative guidance | Build audit-ready monitoring, benchmarking, outcomes analysis, and deterioration responses. |
