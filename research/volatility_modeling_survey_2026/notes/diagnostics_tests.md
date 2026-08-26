# Diagnostics, backtests, and forecast-evaluation workflows

## Scope

This note surveys statistical checks that belong around scivol's univariate and multivariate conditional-volatility models. It separates four questions that software often mixes:

1. Did the fitted model remove predictable structure from its in-sample innovations?
2. Are genuine out-of-sample distribution, quantile, and tail forecasts calibrated?
3. Does one forecasting procedure beat another under a stated loss?
4. Does an operational risk model satisfy a regulatory reporting rule?

These questions need different samples, null hypotheses, reference distributions, and result objects. A single `diagnostic_tests()` method should not pretend that they are interchangeable.

## Current scivol baseline

The public `EstimationResult.diagnostic_tests()` implementation currently provides:

- a Pearson cell test of uniformity for in-sample PIT values, labelled DGT;
- Ljung-Box statistics for four centered PIT powers;
- the PIT series itself;
- automatic-selection penalties based on the number of rejected tests.

The OOS-selection code adds log score, QLIKE, and MSE losses, pairwise HAC Diebold-Mariano statistics against the selected winner, and a circular-block-bootstrap, Tmax-style model confidence set.

This is a useful start, but it leaves the core fitted-model checks absent: whiteness of standardized residuals, remaining ARCH, asymmetry, and parameter stability. It also treats in-sample PIT checks as though their textbook null distributions were unaffected by parameter estimation. The literature does not support that shortcut.

Two implementation details deserve attention before the menu grows:

- The Pearson PIT test always defaults to 40 cells. It neither enforces a minimum expected count nor adjusts the null distribution for estimated parameters. At 100 observations, for example, the expected count is only 2.5. A parametric bootstrap that simulates, refits, and recomputes the statistic is the defensible default for an in-sample composite null.
- QLIKE and MSE currently set realized variance equal to each candidate's squared residual. When mean specifications differ, each candidate is therefore evaluated against a different target. Forecast comparisons require one aligned realized-variance proxy, fixed across candidates. Daily squared demeaned returns may be the fallback. Realized variance or another user-supplied common proxy is much better.

## Recommended diagnostic stack

### 1. Standardized-residual adequacy

Let $z_t = \hat{\epsilon}_t / \sqrt{\hat h_t}$. A correctly specified conditional mean and variance imply that $z_t$ has no predictable first- or second-moment dynamics. The minimum report should show ACFs and formal tests for both $z_t$ and $z_t^2-1$.

| Check | Null and diagnostic value | Implementation requirement | Priority |
|---|---|---|---|
| Ljung-Box on $z_t$ | No remaining linear serial correlation in standardized innovations. Failure points first to the mean specification. | Report a lag path, not one arbitrary lag. Account for fitted mean parameters or offer a refit bootstrap. | P0 |
| Squared-residual portmanteau | No remaining conditional second-moment dependence. McLeod-Li motivated squared-residual autocorrelations, but fitted GARCH residuals need a model-aware covariance rather than a naive chi-square reference. | Implement the Li-Mak or Lundbergh-Terasvirta robust form, or a parametric bootstrap. Include $z_t^2-1$, and optionally absolute-power transforms for heavy-tailed data. | P0 |
| Post-fit ARCH LM | Lagged squared standardized innovations do not explain current squared standardized innovation. | Prefer the robust GARCH adequacy test of Lundbergh and Terasvirta over simply rerunning Engle's original regression. Their robust version handles nonnormal errors. | P0 |
| Engle-Ng sign and size bias | Past negative signs and positive or negative shock sizes do not explain $z_t^2$. This targets omitted leverage and a wrong news-impact curve. | Return the three component tests and the joint test. Use the lagged raw innovation in the sign-size regressors, as the paper specifies. | P0 |
| BDS independence | Standardized residuals are iid against a broad class of nonlinear alternatives. | Make this optional. Results depend on embedding dimension and distance. Use a residual or parametric bootstrap and return the full parameter grid. | P2 |
| Targeted functional-form LM | The fitted GARCH recursion has no neglected smooth-transition or nonlinear variance term. | Follow the robust Lundbergh-Terasvirta framework. This is more actionable than a generic BDS rejection. | P1 |

The classic Ljung-Box statistic was derived for fitted linear time-series models. McLeod and Li established why squared-residual autocorrelations detect nonlinear dependence, while Tse showed that univariate and multivariate conditional-heteroskedasticity diagnostics must account for estimated regressors. The practical conclusion is plain: applying `chi2.sf(Q, lags)` to every transformed residual series is not a generally valid GARCH diagnostic ([Ljung and Box 1978](https://doi.org/10.1093/biomet/65.2.297), [McLeod and Li 1983](https://doi.org/10.1111/j.1467-9892.1983.tb00373.x), [Tse 2002](https://ideas.repec.org/a/ect/emjrnl/v5y2002i2p358-374.html)).

Engle and Ng's regression remains especially valuable because it says how the model is wrong. A sign-bias rejection suggests omitted leverage. Separate negative- and positive-size rejections show where the news-impact curve has the wrong shape ([Engle and Ng 1993](https://doi.org/10.1111/j.1540-6261.1993.tb05127.x)).

### 2. Parameter stability and change detection

High persistence can be a symptom of an omitted level shift or regime change. Stability diagnostics should therefore sit next to persistence and half-life in every fitted-result summary.

| Check | Use | Important qualification | Priority |
|---|---|---|---|
| Nyblom score stability | Joint and parameter-specific constancy tests from cumulative likelihood scores. Individual statistics identify whether mean, persistence, leverage, or density shape moved. | Critical values depend on parameter dimension. A simulation or bootstrap path avoids hard-coded tables and covers newer models. | P0 |
| Robust smooth-change LM | Tests parameter constancy against gradual change within the fitted GARCH recursion. | Use the nonnormality-robust form in Lundbergh and Terasvirta. | P1 |
| Sup-LM or score CUSUM | Detects an unknown abrupt parameter shift and estimates a likely break region. | Report trimmed search range and confidence interval. Boundary parameters need special treatment. | P1 |
| Robust unconditional-variance break test | Detects level changes in unconditional variance before or alongside conditional-volatility modelling. | Do not expose vanilla ICSS as the recommended financial-data default. It assumes independent observations and has large size distortions under kurtosis and conditional heteroskedasticity. Use the Sanso-Arago-Carrion correction. | P1 |
| Multiple-break segmentation | Estimates several regimes for offline analysis. | Separate breaks in unconditional scale from breaks in conditional GARCH parameters. Bai-Perron is a general regression tool; Andreou-Ghysels evaluates procedures tailored to volatility dynamics. | P2 |
| Sequential change monitor | Raises a prospective alarm after a stable training sample. | Keep monitoring separate from retrospective break estimation. Return detection delay, threshold, and restart policy. | P2 |

Nyblom's test is attractive for scivol because analytical per-observation scores already underpin OPG and sandwich covariance calculations. The same score stream can support joint and parameter-specific stability statistics ([Nyblom 1989](https://doi.org/10.1080/01621459.1989.10478759)). Chu adapted score-based parameter-shift testing to GARCH, and Lundbergh and Terasvirta placed no-ARCH, linearity, and constancy tests in one robust LM framework ([Chu 1995](https://doi.org/10.1080/07474939508800318), [Lundbergh and Terasvirta 2002](https://doi.org/10.1016/S0304-4076(02)00096-9)).

Vanilla ICSS should appear only as a legacy option. Sanso, Arago, and Carrion show severe over-rejection for leptokurtic and conditionally heteroskedastic financial series and propose corrected statistics ([Sanso, Arago, and Carrion 2004](https://aefin.es/wp-content/uploads/2019/02/A4-2_443809.pdf)).

### 3. PIT and full-distribution calibration

For a one-step forecast CDF $F_t$, the PIT $u_t=F_t(y_t)$ should be iid uniform under correct conditional calibration. Uniformity alone is insufficient. A misspecified dynamic model can have a uniform marginal PIT distribution while its PIT sequence remains dependent.

The current scivol report should evolve as follows:

| Addition | Purpose | Priority |
|---|---|---|
| PIT histogram plus simultaneous ECDF bands | Shows where calibration fails. Offer KS, Cramer-von Mises, and Anderson-Darling variants, with a refit bootstrap for in-sample residual PITs. | P0 |
| Berkowitz LR tests | Transform $u_t$ to $x_t=\Phi^{-1}(u_t)$, then test zero mean, unit variance, and no AR(1) dependence. Return LR tests for the full joint null and useful decompositions. | P0 |
| PIT moment tests | Test selected standardized PIT moments jointly with a HAC covariance. This is useful for overlapping multi-step forecasts, whose PITs need not be serially independent even under correct calibration. An optional orthogonal basis can reduce redundancy across moment directions. | P1 |
| Tail calibration | Add lower- and upper-tail exceedance plots, quantile coverage across a grid, and tail-weighted proper scores. | P1 |
| Rolling calibration | Show windowed PIT moments, coverage, and log score with uncertainty bands. This often reveals a regime failure hidden by a full-sample pass. | P1 |

Diebold, Gunther, and Tay proposed PIT-based density-forecast evaluation and stressed both uniformity and independence ([DGT 1998](https://www.nber.org/papers/t0215)). Berkowitz's inverse-normal likelihood-ratio approach was designed to improve power in the modest samples common in risk management ([Berkowitz 2001](https://doi.org/10.1198/07350010152596718)).

Knuppel's joint raw-moment test is a better fit for multi-step forecasts than applying ordinary independence tests to overlapping PITs. It uses a long-run covariance matrix and can target the moments that matter for the application ([Knuppel 2015](https://doi.org/10.1080/07350015.2014.948175)).

Three distinctions should be explicit in the API:

- `residual_pit_diagnostics()` evaluates a fitted model on the same observations used for estimation. Its default inference should simulate and refit under the composite null.
- `forecast_calibration()` evaluates stored, timestamped forecasts that were issued before outcomes arrived. Standard forecast-evaluation theory belongs here.
- A calibration test is not a ranking rule. Proper scores compare sharp forecasts while rewarding honest probabilities. A forecast can pass a low-power calibration test and still be much less useful than a competitor.

For full densities, scivol should report negative log score and CRPS. Tail-sensitive comparison should use threshold- or quantile-weighted CRPS, which remains proper. A simple weighted log score can be hedged and should not be the recommended tail score ([Gneiting and Raftery 2007](https://doi.org/10.1198/016214506000001437), [Gneiting and Ranjan 2011](https://doi.org/10.1198/jbes.2010.08110)).

### 4. VaR backtests

Every distributional volatility model can issue VaR forecasts. A complete risk workflow needs several complementary tests because exception frequency and exception dynamics diagnose different faults.

| Test | Null | What a rejection says | Priority |
|---|---|---|---|
| Exact binomial and Kupiec proportion-of-failures | The unconditional exception rate equals the nominal tail probability. | Overall tail thickness or scale is wrong. | P0 |
| Christoffersen independence | Exceptions are independent under a first-order Markov alternative. | Tail risk clusters in time, usually because dynamics adapt too slowly. | P0 |
| Christoffersen conditional coverage | Correct exception rate and independence jointly. | At least one of calibration or exception dynamics is wrong. | P0 |
| Dynamic quantile test | The hit sequence is orthogonal to lagged hits, VaR, and user-selected information variables. | The conditional quantile is predictable from available information. | P0 |
| Duration test | Inter-exception durations have the null exponential or geometric behavior. | Exceptions cluster or arrive too regularly; often more informative in small tail samples. | P1 |
| Multi-quantile multinomial test | Coverage is correct across several tail quantiles. | Locates shape errors inside the tail and offers an indirect ES check. | P1 |

Kupiec also showed the poor power of rare-event verification in realistic samples. scivol should therefore return the observed and expected exception counts, exact confidence intervals, and a power or minimum-detectable-deviation calculation, not only a p-value ([Kupiec 1995](https://fraser.stlouisfed.org/title/finance-economics-discussion-series-1491/techniques-verifying-accuracy-risk-measurement-models-717921)).

Christoffersen decomposes conditional coverage into unconditional coverage and independence, while the DQ test allows a richer information set ([Christoffersen 1998](https://doi.org/10.2307/2527341), [Engle and Manganelli 2004](https://doi.org/10.1198/073500104000000370)). Duration tests examine the waiting times between exceptions and can improve power over binary first-order transition checks ([Christoffersen and Pelletier 2004](https://doi.org/10.1093/jjfinec/nbh004)).

Implementation must handle:

- lower-tail returns versus upper-tail losses without silent sign changes;
- one-step and multi-step horizons;
- overlapping multi-step forecasts through HAC or block-bootstrap inference;
- estimation and refit schedules;
- exact tests when expected exceptions are few;
- aligned missing dates and a declared timezone/calendar;
- clustered standard errors when many assets share market shocks;
- familywise or false-discovery adjustment across assets, levels, and horizons.

### 5. Expected-shortfall backtests

ES describes average loss beyond VaR. It cannot be validated by counting VaR exceptions alone. scivol should support several approaches because each asks a different question.

| Backtest | Inputs | Role | Priority |
|---|---|---|---|
| McNeil-Frey exceedance residual | VaR, ES, outcomes | Tests whether realized tail losses have the predicted conditional mean, usually with bootstrap inference. | P1 |
| Acerbi-Szekely unconditional tests | ES and, depending on variant, VaR or PIT information | Model-independent checks of average tail severity. | P1 |
| Du-Escanciano cumulative-violation tests | VaR, ES, outcomes | Tests conditional ES calibration using cumulative violations and accounts for tail risk. | P1 |
| ESR regression tests | ES forecasts, with optional VaR forecasts | Strict, auxiliary, and intercept tests of whether ES forecast regression intercept and slope equal 0 and 1. Use misspecification-robust covariance. | P0 |
| Joint VaR-ES calibration and score | VaR, ES, outcomes | Uses identification functions and strictly consistent Fissler-Ziegel scores to test and compare joint forecasts. | P0 |

Fissler and Ziegel proved that VaR and ES are jointly elicitable, which permits principled comparative scoring even though ES alone is not elicitable ([Fissler and Ziegel 2016](https://doi.org/10.1214/16-AOS1439)). Any implementation of their general score characterizations should incorporate the authors' later correction ([Fissler and Ziegel 2021](https://doi.org/10.1214/20-AOS2014)). Du and Escanciano provide cumulative-violation tests, and Bayer and Dimitriadis develop regression-based strict and auxiliary ES backtests with misspecification-robust inference ([Du and Escanciano 2017](https://doi.org/10.1287/mnsc.2015.2342), [Bayer and Dimitriadis 2022](https://doi.org/10.1093/jjfinec/nbaa013)). Comparative and traditional calibration backtests answer different questions and should both be available ([Nolde and Ziegel 2017](https://doi.org/10.1214/17-AOAS1041)).

The regulatory detail matters. The current Basel market-risk framework calculates internal-model capital from 97.5% ES, but model backtesting still compares one-day VaR with actual and hypothetical P&L. Bank-wide backtesting uses 99% VaR; trading-desk backtesting uses both 97.5% and 99% VaR over 250 trading days. The framework also requires a P&L attribution test and defines green, amber, and red zones ([Basel MAR32](https://www.bis.org/basel_framework/chapter/MAR/32.htm?inforce=20230101&published=20200327&tldate=20461003), [Basel MAR33](https://www.bis.org/basel_framework/chapter/MAR/33.htm?inforce=20230101&published=20200605&tldate=20230424)). A `basel_backtest()` helper should reproduce those counts and zones exactly, but scivol should not label a research ES backtest as a Basel requirement.

### 6. Forecast losses and comparative tests

#### Losses

Use a loss that matches the forecast target:

| Target | Recommended scores | Notes |
|---|---|---|
| Conditional variance | QLIKE and MSE against one common realized proxy | Patton derives conditions under which rankings survive noise in a conditionally unbiased volatility proxy. Support squared returns, ranges, and user-supplied realized measures. |
| Full univariate density | Negative log score and CRPS | Log score strongly penalizes assigning tiny density to an observation. CRPS integrates quantile loss and is less dominated by a few density values. |
| Tail density | Quantile-weighted or threshold-weighted CRPS | Retains propriety while focusing on the region users care about. |
| Quantile or VaR | Pinball loss | Strictly consistent for the requested quantile. |
| Joint VaR and ES | Fissler-Ziegel score family | Return the exact score specification because rankings can depend on that choice. |
| Multivariate density | Log score when tractable, plus energy and variogram scores | Energy score can have low sensitivity to wrong correlations. Variogram scores target dependence more directly. |

Patton's result is directly relevant to scivol's QLIKE and MSE workflow: noisy realized volatility need not corrupt rankings if the proxy is conditionally unbiased and the loss has the required form ([Patton 2011](https://doi.org/10.1016/j.jeconom.2010.03.034)). The target must still be common to all candidates.

#### Comparative inference

| Method | Best use | Required behavior in scivol | Priority |
|---|---|---|---|
| Diebold-Mariano | Two prespecified forecast procedures under a chosen loss. | Let users set horizon and HAC bandwidth/kernel. Add the Harvey-Leybourne-Newbold small-sample correction. Do not treat p-values against an ex-post selected winner as ordinary pairwise inference. | P0 fix |
| Giacomini-White | Conditional predictive ability with rolling or fixed-window estimation and possible misspecification. | Accept conditioning instruments and preserve the actual estimation-window scheme. | P1 |
| White reality check | Familywise comparison of many searched models against a benchmark. | Provide for reproducibility, but prefer SPA when poor alternatives make the reality check conservative. | P2 |
| Hansen SPA | Tests whether any model in a searched family beats a benchmark while controlling data snooping. | Use stationary or circular block bootstrap; return benchmark, studentization, block rule, and recentering method. | P1 |
| Model confidence set | Produces a set of statistically indistinguishable best models without a privileged benchmark. | scivol already has a Tmax-style version. Document the exact statistic and elimination rule, add the range statistic option, and validate against the paper's reference algorithm. | P0 audit |
| Fluctuation and reversal tests | Shows whether relative forecast performance changed over time. | Plot local loss differentials and simultaneous bands. | P1 |

Diebold-Mariano permits general loss and serially correlated loss differentials ([Diebold and Mariano 1995](https://doi.org/10.1080/07350015.1995.10524599)). The current implementation chooses a generic $T^{1/3}$ bandwidth. That is not a substitute for horizon-aware inference, especially for overlapping forecasts. Giacomini and White preserve finite-window estimation effects and test whether relative performance is conditionally predictable ([Giacomini and White 2006](https://doi.org/10.1111/j.1468-0262.2006.00718.x)).

Once auto-selection searches many models, pairwise DM tests no longer control the search. White's reality check, Hansen's SPA, and the model confidence set address data snooping and multiplicity in different ways ([White 2000](https://doi.org/10.1111/1468-0262.00152), [Hansen 2005](https://doi.org/10.1198/073500105000000063), [Hansen, Lunde, and Nason 2011](https://doi.org/10.3982/ECTA5771)). Giacomini-Rossi fluctuation tests add something the global average cannot: they show when one model overtook another ([Giacomini and Rossi 2010](https://doi.org/10.1002/jae.1177)).

### 7. Multivariate diagnostics

DCC or other conditional-covariance support needs diagnostics that test cross-series dynamics, not a loop over univariate PIT tests.

Recommended additions:

- whiteness tests for the standardized innovation vector;
- portmanteau tests for lagged standardized-residual cross-products;
- tests of remaining conditional cross-correlation, with parameter-estimation adjustment;
- a constant-correlation LM test when a CCC model is fitted;
- eigenvalue, condition-number, and positive-definiteness traces for every forecast covariance matrix;
- correlation calibration plots and realized covariance losses when a common realized covariance proxy is available;
- multivariate Rosenblatt diagnostics with an explicit variable order;
- multivariate proper scores, at least log, energy, and variogram scores.

Tse's residual regressions cover squared standardized residuals and cross-products in univariate and multivariate models. Tse also provides a convenient LM test of constant correlation ([Tse 2000](https://doi.org/10.1016/S0304-4076(99)00080-9), [Tse 2002](https://ideas.repec.org/a/ect/emjrnl/v5y2002i2p358-374.html)). The Rosenblatt transform maps a continuous multivariate distribution to uniforms, but it depends on the conditional ordering ([Rosenblatt 1952](https://doi.org/10.1214/aoms/1177729394)). The report must disclose that order or repeat the diagnostic over selected orderings.

For multivariate forecast comparison, an energy score alone is insufficient. Scheuerer and Hamill show its weak discrimination against wrong correlations and propose variogram scores that respond more strongly to dependence errors ([Scheuerer and Hamill 2015](https://doi.org/10.1175/MWR-D-14-00269.1)).

## Workflow and API design

A clear API would use three result types.

### `FitDiagnostics`

Created from one fitted result and its estimation sample. It should contain:

- aligned residuals, standardized residuals, PITs, scores, and valid-observation mask;
- ACF and lag-path tables for $z_t$, $z_t^2-1$, and selected PIT basis functions;
- robust or bootstrap p-values, test statistics, degrees of freedom, and bootstrap settings;
- Engle-Ng component and joint tests;
- Nyblom joint and individual statistics;
- warnings for clipped PITs, insufficient expected cell counts, boundary estimates, and too few effective observations.

### `ForecastEvaluation`

Built from immutable forecast records with issue time, target time, horizon, distribution or requested functionals, outcome, refit identifier, and model identifier. It should provide:

- PIT and Berkowitz calibration;
- proper score paths and aggregates;
- VaR and ES backtests by level and horizon;
- DM, Giacomini-White, SPA, and MCS comparisons;
- rolling views and regime annotations;
- exportable long-form tables.

### `RiskBacktest`

Focused on tail forecasts and operational reporting. It should preserve the sign convention, nominal level, exception sequence, durations, actual and hypothetical P&L labels, traffic-light rules, and audit metadata. A Basel mode should be versioned to the applicable framework text.

The diagnostic engine should expose low-level pure functions too. Users need to test forecasts from other packages, combine scivol forecasts with external realized measures, and reproduce papers without constructing a fitted scivol model.

## Statistical correctness requirements

### Composite-null inference

In-sample residual diagnostics use estimated parameters. For tests whose asymptotic adjustment is model-specific or uncertain, the default bootstrap should:

1. simulate from the fitted model;
2. refit with the same solver, constraints, hold-back, and initialization policy;
3. recompute the diagnostic;
4. form a finite-sample p-value with the usual plus-one correction.

Offering a faster asymptotic p-value is reasonable, but the result must say which reference distribution it used.

### Multiple testing

The current default adds a fixed penalty for every rejection among one Pearson and four Ljung-Box tests. These tests are correlated, their size varies with lag and cell choices, and the penalty is a heuristic rather than a statistical selection criterion. Keep it only if it is labelled as such.

A better automatic-selection workflow is:

1. define a small, prespecified adequacy family;
2. control familywise error or false discovery within that family;
3. treat surviving models as adequate rather than ranking them by p-value;
4. rank survivors with an OOS proper score;
5. report SPA or MCS uncertainty across the searched family.

P-values should never be converted into an impression that a nonrejected model is true.

### Forecast alignment

Every comparative procedure must operate on the intersection of identical target timestamps and one common realized target. The result should record:

- issue and target times;
- horizon and overlap;
- expanding, rolling, or fixed estimation window;
- refit frequency;
- missing-value policy;
- realized volatility or covariance source;
- loss and all tuning parameters;
- bootstrap type, block length, draws, and random seed.

### Small tail samples

At a 1% VaR level, 250 observations imply only 2.5 expected exceptions. Asymptotic chi-square approximations can be poor and tests have little power. Return exact binomial inference where possible, bootstrap alternatives elsewhere, and a prominent low-power warning.

## Validation plan for new diagnostics

Each test needs size, power, and workflow validation rather than a single numerical example.

### Null-size simulations

- Simulate every supported innovation density and representative model family.
- Cover effective samples near 250, 1,000, and 5,000 observations.
- Include near-boundary persistence, heavy tails, skewness, and leverage.
- Verify asymptotic and refit-bootstrap rejection rates separately.
- For multivariate tests, vary dimension, unconditional correlation, and DCC persistence.

### Targeted power simulations

- omitted ARMA dynamics for $z_t$ whiteness;
- omitted GARCH lags for squared-residual and ARCH tests;
- omitted leverage for sign and size bias;
- wrong density shape for PIT and Berkowitz tests;
- clustered exceptions for VaR independence, duration, and DQ tests;
- wrong tail severity with correct VaR for ES tests;
- abrupt and smooth parameter changes for stability tests;
- wrong dynamic correlation for multivariate tests.

### Invariance and edge cases

- return versus loss sign conventions;
- scale changes in returns;
- different hold-back lengths;
- missing dates and unequal candidate samples;
- clipped PIT values at machine precision;
- zero, one, and all VaR exceptions;
- overlapping horizons;
- singular or nearly singular covariance forecasts;
- deterministic bootstrap reproduction.

### External parity

Compare formulas and seeded examples with at least two independent implementations where available. The purpose is not to clone their defaults. It is to catch sign conventions, degrees-of-freedom errors, test-regression mistakes, and bootstrap recentering differences.

## Prioritized roadmap

### P0: correct the foundation

1. Split in-sample fit diagnostics from out-of-sample forecast evaluation.
2. Replace fixed-bin, unadjusted in-sample PIT inference with validated cell rules and a refit bootstrap.
3. Add standardized-residual Ljung-Box, model-aware squared-residual or robust ARCH tests, Engle-Ng, Berkowitz, and Nyblom.
4. Add complete VaR coverage, independence, conditional coverage, and DQ tests.
5. Add strict and auxiliary ESR tests plus joint VaR-ES scores.
6. Fix OOS variance evaluation to use one common realized target.
7. Make DM horizon- and bandwidth-aware, add the small-sample correction, and audit MCS against the reference algorithm.

### P1: make workflows research-grade

1. Add CRPS, weighted CRPS, pinball, and Fissler-Ziegel score families.
2. Add Giacomini-White, SPA, rolling calibration, and forecast-performance fluctuation tests.
3. Add robust unconditional-variance breaks, score CUSUM or sup-LM, and regime annotations.
4. Add duration, multi-quantile VaR, McNeil-Frey, Du-Escanciano, and Acerbi-Szekely backtests.
5. Add DCC cross-product diagnostics and multivariate proper scores.

### P2: completeness without clutter

1. Optional BDS and targeted smooth-transition tests.
2. Offline multiple-break segmentation and online parameter monitoring.
3. Rosenblatt-order sensitivity tools and high-dimensional multivariate diagnostics.
4. Power calculators and batch false-discovery control across large asset panels.

## Source table

All links below point to primary papers, author or institutional copies, or current regulatory text.

| Topic | Source | Contribution used here |
|---|---|---|
| ARCH LM | [Engle (1982), *Econometrica*](https://www.jstor.org/stable/1912773) | ARCH model and LM test based on squared residual predictability. |
| Portmanteau | [Ljung and Box (1978), *Biometrika*](https://doi.org/10.1093/biomet/65.2.297) | Small-sample-improved portmanteau statistic for residual autocorrelation. |
| Squared residuals | [McLeod and Li (1983), *Journal of Time Series Analysis*](https://doi.org/10.1111/j.1467-9892.1983.tb00373.x) | Squared-residual autocorrelations as diagnostics for nonlinear dependence. |
| News impact | [Engle and Ng (1993), *Journal of Finance*](https://doi.org/10.1111/j.1540-6261.1993.tb05127.x) | Sign, negative-size, positive-size, and joint bias diagnostics. |
| Stability | [Nyblom (1989), *JASA*](https://doi.org/10.1080/01621459.1989.10478759) | Cumulative-score tests of joint and individual parameter constancy. |
| GARCH shifts | [Chu (1995), *Econometric Reviews*](https://doi.org/10.1080/07474939508800318) | Supremum LM tests for GARCH parameter shifts. |
| Unified GARCH checks | [Lundbergh and Terasvirta (2002), *Journal of Econometrics*](https://doi.org/10.1016/S0304-4076(02)00096-9) | Robust LM tests of remaining ARCH, nonlinearity, and parameter constancy. |
| Residual diagnostics | [Tse (2002), *Econometrics Journal*](https://ideas.repec.org/a/ect/emjrnl/v5y2002i2p358-374.html) | Estimation-aware univariate and multivariate squared-residual and cross-product tests. |
| Constant correlation | [Tse (2000), *Journal of Econometrics*](https://doi.org/10.1016/S0304-4076(99)00080-9) | LM test of the constant-correlation restriction in multivariate GARCH. |
| Omnibus nonlinearity | [Broock et al. (1996), *Econometric Reviews*](https://doi.org/10.1080/07474939608800353) | BDS test of iid residuals against broad nonlinear dependence. |
| PIT foundation | [Rosenblatt (1952), *Annals of Mathematical Statistics*](https://doi.org/10.1214/aoms/1177729394) | Multivariate transform to uniforms and the ordering issue. |
| Density calibration | [Diebold, Gunther, and Tay (1998), *International Economic Review*](https://www.nber.org/papers/t0215) | PIT uniformity and independence for density-forecast evaluation. |
| Density calibration | [Berkowitz (2001), *JBES*](https://doi.org/10.1198/07350010152596718) | Inverse-normal PIT likelihood-ratio tests with useful small-sample power. |
| Multi-step calibration | [Knuppel (2015), *JBES*](https://doi.org/10.1080/07350015.2014.948175) | Joint raw-moment test with dependence-robust covariance for overlapping PITs. |
| Calibration and sharpness | [Gneiting, Balabdaoui, and Raftery (2007), *JRSS B*](https://doi.org/10.1111/j.1467-9868.2007.00587.x) | Distinction between probabilistic calibration and sharpness. |
| Proper scores | [Gneiting and Raftery (2007), *JASA*](https://doi.org/10.1198/016214506000001437) | General theory and catalog of strictly proper scoring rules. |
| Tail scores | [Gneiting and Ranjan (2011), *JBES*](https://doi.org/10.1198/jbes.2010.08110) | Proper threshold- and quantile-weighted CRPS. |
| Multivariate scores | [Scheuerer and Hamill (2015), *Monthly Weather Review*](https://doi.org/10.1175/MWR-D-14-00269.1) | Variogram scores and the energy score's weak sensitivity to correlation errors. |
| Multivariate portmanteau | [Hosking (1980), *JASA*](https://doi.org/10.1080/01621459.1980.10477520) | Residual-autocovariance portmanteau statistic for multivariate time series. |
| VaR coverage | [Kupiec (1995), Federal Reserve FEDS / *Journal of Derivatives*](https://fraser.stlouisfed.org/title/finance-economics-discussion-series-1491/techniques-verifying-accuracy-risk-measurement-models-717921) | Proportion-of-failures test and warning about rare-event power. |
| VaR conditional coverage | [Christoffersen (1998), *International Economic Review*](https://doi.org/10.2307/2527341) | Decomposition into unconditional coverage and exception independence. |
| Dynamic quantiles | [Engle and Manganelli (2004), *JBES*](https://doi.org/10.1198/073500104000000370) | Dynamic quantile test using the exception sequence and information variables. |
| VaR durations | [Christoffersen and Pelletier (2004), *Journal of Financial Econometrics*](https://doi.org/10.1093/jjfinec/nbh004) | Duration-based exception-clustering backtests. |
| Tail risk | [McNeil and Frey (2000), *Journal of Empirical Finance*](https://doi.org/10.1016/S0927-5398(00)00012-8) | GARCH-EVT VaR and ES forecasts, exceedance-residual backtesting, and multi-day simulation. |
| ES backtesting | [Acerbi and Szekely (2014), MSCI Research](https://www.msci.com/resources/research/articles/2014/Research_Insight_Backtesting_Expected_Shortfall_December_2014.pdf) | Three model-independent, nonparametric ES backtests. |
| ES backtesting | [Du and Escanciano (2017), *Management Science*](https://doi.org/10.1287/mnsc.2015.2342) | Cumulative-violation ES backtests for tail severity. |
| Joint VaR-ES | [Fissler and Ziegel (2016), *Annals of Statistics*](https://doi.org/10.1214/16-AOS1439) | Joint elicitability and strictly consistent VaR-ES scores. |
| Joint VaR-ES correction | [Fissler and Ziegel (2021), *Annals of Statistics*](https://doi.org/10.1214/20-AOS2014) | Corrected conditions in the general score characterization. |
| Regulatory backtesting | [Nolde and Ziegel (2017), *Annals of Applied Statistics*](https://doi.org/10.1214/17-AOAS1041) | Traditional versus comparative risk backtests. |
| ES regression | [Bayer and Dimitriadis (2022), *Journal of Financial Econometrics*](https://doi.org/10.1093/jjfinec/nbaa013) | Strict, auxiliary, and intercept ESR backtests with robust covariance. |
| Basel eligibility | [Basel Framework MAR32](https://www.bis.org/basel_framework/chapter/MAR/32.htm?inforce=20230101&published=20200327&tldate=20461003) | Current VaR backtesting levels, 250-day zones, desk thresholds, and P&L attribution. |
| Basel ES capital | [Basel Framework MAR33](https://www.bis.org/basel_framework/chapter/MAR/33.htm?inforce=20230101&published=20200605&tldate=20230424) | 97.5% ES capital calculation and VaR-based backtesting add-on. |
| Pairwise comparison | [Diebold and Mariano (1995), *JBES*](https://doi.org/10.1080/07350015.1995.10524599) | Equal predictive-accuracy tests for general dependent loss differentials. |
| Small-sample DM | [Harvey, Leybourne, and Newbold (1997), *International Journal of Forecasting*](https://doi.org/10.1016/S0169-2070(96)00719-4) | Small-sample correction for forecast-accuracy tests. |
| Conditional comparison | [Giacomini and White (2006), *Econometrica*](https://doi.org/10.1111/j.1468-0262.2006.00718.x) | Conditional predictive-ability tests with finite-window estimation effects. |
| Data snooping | [White (2000), *Econometrica*](https://doi.org/10.1111/1468-0262.00152) | Reality check for the best model found in a specification search. |
| SPA | [Hansen (2005), *JBES*](https://doi.org/10.1198/073500105000000063) | Studentized superior-predictive-ability test with less influence from poor alternatives. |
| MCS | [Hansen, Lunde, and Nason (2011), *Econometrica*](https://doi.org/10.3982/ECTA5771) | Sequential model confidence set and elimination rules. |
| Noisy volatility targets | [Patton (2011), *Journal of Econometrics*](https://doi.org/10.1016/j.jeconom.2010.03.034) | Loss functions that preserve rankings under imperfect volatility proxies. |
| Unstable performance | [Giacomini and Rossi (2010), *Journal of Applied Econometrics*](https://doi.org/10.1002/jae.1177) | Fluctuation and one-time-reversal tests for changing relative forecast performance. |
| Variance change points | [Inclan and Tiao (1994), *JASA*](https://doi.org/10.1080/01621459.1994.10476824) | ICSS multiple variance-change algorithm under independent observations. |
| Robust variance changes | [Sanso, Arago, and Carrion (2004), *Revista de Economia Financiera*](https://aefin.es/wp-content/uploads/2019/02/A4-2_443809.pdf) | ICSS size failures under kurtosis and GARCH, plus robust corrections. |
| Multiple breaks | [Bai and Perron (1998), *Econometrica*](https://www.jstor.org/stable/2998540) | Tests and estimators for multiple structural changes in regression models. |
| Volatility breaks | [Andreou and Ghysels (2002), *Journal of Applied Econometrics*](https://doi.org/10.1002/jae.684) | Evaluation of multiple-break tests for ARCH, SV, and realized-volatility processes. |
