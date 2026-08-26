# Volatility-modeling software packages

Research date: 2026-08-25. This note uses package authors' documentation, repositories, and vendor manuals. It treats a model as supported only when the source documents estimation or a complete fit, filter, forecast, or simulation path. A generic optimization or probabilistic-programming framework is not counted as a specialist volatility implementation.

## Main findings

No package combines the widest model catalog, fast native derivatives, multivariate scalability, modern forecast evaluation, and a clean Python API.

The closest broad benchmarks are R's `rugarch`/`rmgarch` and their newer `tsgarch`/`tsmarch` successors, OxMetrics G@RCH, and Kevin Sheppard's MATLAB MFE toolbox. Python's `arch` is the usability benchmark for univariate models, but it has no documented multivariate volatility module. Julia's `ARCHModels.jl` has a smaller univariate catalog, yet its large-dimensional DCC path is unusually relevant. Specialist R packages cover major branches that general-purpose libraries leave out: BEKK, Markov-switching GARCH, stochastic volatility, factor stochastic volatility, score-driven models, realized measures, and DCC-MIDAS.

scivol already has an unusually rich standardized-density catalog, joint mean-volatility specifications, analytical C derivatives, independent AD validation, rolling estimation, simulation, forecasts, VaR, automatic selection, DCC, and model-confidence sets. Its conspicuous gaps are now model families and end-to-end evaluation rather than ordinary GARCH fitting.

## Comparison at a glance

| System | Documented strength | Important limits for comparison with scivol |
|---|---|---|
| Python `arch` | Polished univariate composition; ARCH/GARCH, EGARCH, FIGARCH, APARCH, HARCH, MIDAS Hyperbolic, EWMA/RiskMetrics; Normal, t, skew-t, GED; analytic, simulation, and residual-bootstrap forecasts; VaR examples | No documented multivariate volatility models; no realized-GARCH, stochastic-volatility, regime-switching, or score-driven family |
| R `rugarch` | Very broad univariate ARCH family, ARFIMAX, variance regressors, many densities, rolling estimation, bootstrap, simulation, inference, and diagnostics | Large S4 API and several numerical-derivative or solver paths make behavior harder to audit than a small typed core |
| R `tsgarch` | Partial rewrite of `rugarch` using TMB automatic differentiation; fit, filter, predict, simulate, profile, PIT, walk-forward backtests, parallel multispec estimation | Newer and less battle-tested; package NEWS still records numerical Hessian fallbacks for unstable cases |
| R `rmgarch` / `tsmarch` | DCC, copula-GARCH, GO-GARCH; forecasts, simulation, rolling estimation, portfolio density, VaR/ES, co-moments; `tsmarch` documents two-step and partitioned-Hessian inference | Two-step estimation conditions correlation inference on marginal fits; full standard errors can be expensive |
| Julia `ARCHModels.jl` | Fast typed API for ARMA plus ARCH/GARCH/TGARCH/EGARCH; CCC/DCC; AD; large-scale DCC using nonlinear shrinkage and pairwise likelihood | Smaller model and density catalog; multivariate density is currently Gaussian only |
| MATLAB Econometrics Toolbox | Stable object workflow and GUI for GARCH, EGARCH, GJR with fit, infer, filter, forecast, and simulation | Narrow catalog, only Gaussian or Student-t innovations, and no documented multivariate GARCH estimator |
| MATLAB MFE toolbox | Broad research reference: GARCH variants, HEAVY, CCC/DCC/ADCC, BEKK, OGARCH/GOGARCH/RARCH, realized measures, bootstrap, MCS/SPA | Maintenance signals are mixed; the author's feature page still says last updated in 2013, while a File Exchange package was refreshed in 2024 |
| OxMetrics G@RCH 10 | Broad menu and Ox API: long-memory GARCH, BEKK/DCC/DECO/GO-GARCH, realized volatility, VaR, forecast comparison, many specification tests | Proprietary; fewer innovation distributions than scivol; less suitable as an embeddable Python engine |
| Stan / PyMC | Fully Bayesian custom GARCH and stochastic-volatility models, priors, posterior uncertainty, predictive checks | Examples rather than a standardized volatility library; users must code constraints, initialization, forecasting, and diagnostics; latent SV can be costly |
| Stata / EViews | Mature econometric workflows, robust covariance, forecasting, testing, GUI/reporting; Stata has DVECH/CCC/DCC/VCC; EViews has FIGARCH/FIEGARCH/CGARCH/MIDAS-GARCH | Closed source; derivative and model coverage vary by specification; limited composability compared with scivol |

## Python

### `arch`

`arch` is the Python baseline. Its high-level constructor and lower-level component API combine a mean process, volatility process, and standardized distribution. Documented mean models include zero, constant, least-squares, AR/ARX, and HAR/HARX, plus ARCH-in-mean. Volatility processes include ARCH, power GARCH and threshold variants through the `GARCH(p,o,q,power)` class, EGARCH, FIGARCH, APARCH, HARCH, MIDAS Hyperbolic, EWMA, and RiskMetrics 2006. Built-in distributions are Normal, standardized Student-t, Hansen skew-t, and GED. [`arch` volatility processes](https://arch.readthedocs.io/en/stable/univariate/volatility.html), [`arch` mean models](https://arch.readthedocs.io/en/stable/univariate/mean.html), [`arch` distributions](https://arch.readthedocs.io/en/stable/univariate/distribution.html)

Every volatility model has one-step forecasts. Multi-step analytic forecasts exist when the recursion is linear in squared residuals; other models use Monte Carlo or a residual bootstrap. The forecast API supports origin or target alignment, scenarios, and exogenous variables, and the documentation includes a VaR workflow. [`arch` forecasting](https://arch.readthedocs.io/en/latest/univariate/forecasting.html), [`arch` univariate guide](https://arch.readthedocs.io/en/stable/univariate/univariate.html)

The repository states that Cython or Numba accelerates calculations. Its release page lists 8.0.0 as the latest release and records Python 3.14 wheels plus compatibility work for recent NumPy and pandas versions, which verifies current maintenance. [`arch` repository](https://github.com/bashtage/arch), [`arch` releases](https://github.com/bashtage/arch/releases)

The package does not document CCC, DCC, BEKK, stochastic volatility, score-driven volatility, realized GARCH, or Markov-switching GARCH. This is the main opening for scivol in Python.

### `statsmodels`, `sktime`, and smaller specialists

`statsmodels` is a companion, not a competitor, for conditional volatility. Its current time-series index covers ARIMA, state-space, VAR, and regime-switching mean models but no GARCH estimator. It does provide Engle's ARCH LM test and general residual tools. [`statsmodels` time-series API](https://www.statsmodels.org/stable/tsa), [`het_arch`](https://www.statsmodels.org/dev/generated/statsmodels.stats.diagnostic.het_arch.html)

`sktime` wraps `arch` behind its common forecasting interface. The wrapper matters as a workflow benchmark because it supplies standard temporal splitting, tuning, pipelines, update semantics, and hierarchical or panel containers, but it adds no new volatility likelihood. [`sktime` ARCH wrapper](https://www.sktime.net/en/v0.35.1/api_reference/auto_generated/sktime.forecasting.arch.ARCH.html), [`sktime` forecasting API](https://www.sktime.org/en/stable/api_reference/forecasting.html)

`frds` implements constant-mean, Normal GARCH(1,1), bivariate CCC, bivariate DCC, and GJR-DCC. Its own source says the multivariate estimators calculate parameters but no standard errors. It is useful application code for systemic-risk measures, not a full model library. [`frds` documentation](https://frds.io/), [`frds` multivariate source](https://frds.io/_modules/frds/algorithms/_mgarch/)

## R

### `rugarch` and `rmgarch`

`rugarch` remains the broadest established open-source univariate reference. The official vignette documents sGARCH, iGARCH, EGARCH, GJR-GARCH, APARCH, family GARCH, component GARCH, multiplicative component GARCH, realized GARCH, and FIGARCH. The package also supports ARFIMA/ARFIMAX, ARCH-in-mean, external regressors, fit/filter/forecast/simulation, rolling estimation, bootstraps, parameter uncertainty, and multiple solvers. Its density set includes Normal, t, GED and skewed variants, Johnson SU, generalized hyperbolic, NIG, and GH skew-t. Version 1.5-6 was built in August 2026. [`rugarch` vignette](https://alexiosg.r-universe.dev/rugarch/doc/Introduction_to_the_rugarch_package.pdf), [`rugarch` manual and current build](https://alexiosg.r-universe.dev/rugarch/doc/manual.html)

`rmgarch` supplies DCC, GO-GARCH, and copula-GARCH with specification, fitting, filtering, forecasting, simulation, and rolling methods. DCC supports multivariate Normal, Laplace, and Student distributions; GO-GARCH exposes conditional co-skewness and co-kurtosis and portfolio densities. The current R-universe build is 1.4-2. [`rmgarch` manual](https://alexiosg.r-universe.dev/rmgarch/doc/manual.html), [`rmgarch` model vignette](https://alexiosg.r-universe.dev/rmgarch/doc/The_rmgarch_models.pdf)

### `tsgarch`, `tsmarch`, and `tstests`

The newer `tsmodels` stack is more informative for scivol's design. `tsgarch` 1.0.4 is a partial rewrite and reconsideration of `rugarch` using Template Model Builder automatic differentiation. It exposes estimation, prediction, filtering, simulation, coefficient profiling, PITs, parallel multi-series estimation, walk-forward rolling backtests, sandwich-compatible score and bread methods, and benchmark fixtures. Its May 2026 NEWS also records a numerical Hessian fallback when an analytical step fails, including known instability for some EGARCH-NIG data sets. That candor is worth copying in fit metadata. [`tsgarch` package page](https://tsmodels.r-universe.dev/tsgarch), [`tsgarch` manual](https://tsmodels.r-universe.dev/tsgarch/doc/manual.html), [`tsgarch` NEWS](https://tsmodels.r-universe.dev/tsgarch/NEWS)

`tsmarch` 1.0.3 replaces much of `rmgarch`'s role. It implements DCC, copula-GARCH, and generalized orthogonal GARCH with generalized-hyperbolic factors. It also reports covariance and correlation paths, co-skewness, co-kurtosis, portfolio convolution, VaR, ES, forecasts, simulations, and the Engle-Sheppard constant-correlation test. Its documentation makes the inference tradeoff explicit: two-stage DCC is fast; the partitioned Hessian for standard errors is expensive; OPG is the default if the user does not request the Hessian; and parallel futures can accelerate the full calculation. [`tsmarch` package page](https://tsmodels.r-universe.dev/tsmarch), [`tsmarch` manual](https://tsmodels.r-universe.dev/tsmarch/doc/manual.html), [`tsmarch` multivariate vignette](https://tsmodels.r-universe.dev/tsmarch/doc/feasible_multivariate_garch.html)

`tstests` is a good specification for a library-level diagnostics module. It includes GMM orthogonality, Nyblom-Hansen parameter constancy, Engle-Ng sign bias, Berkowitz and Hong-Li density tests, Mincer-Zarnowitz and directional-accuracy forecast tests, and VaR/ES tests. [`tstests` package page](https://tsmodels.r-universe.dev/tstests)

### Specialist R packages

Several maintained packages show that "complete" extends well beyond GARCH variants.

- `BEKKs` 1.4.7 estimates, simulates, and forecasts full, diagonal, scalar, and asymmetric BEKK(1,1,1) models using a C++ BHHH implementation. It includes portfolio VaR, backtesting, and volatility impulse-response functions. [`BEKKs` CRAN page](https://CRAN.R-project.org/package=BEKKs), [`BEKKs` model reference](https://search.r-project.org/CRAN/refmans/BEKKs/html/BEKKs.html)

- `MSGARCH` fits Markov-switching and mixture GARCH by ML or Bayesian MCMC. It filters regimes and supports forecasts, simulation, PIT, VaR, and ES. Its C++ implementation is limited to one-lag single-regime components and expects the mean to be prefiltered. Version 2.51 remains on CRAN, though its last release was in 2022. [`MSGARCH` CRAN page](https://CRAN.R-project.org/package=MSGARCH), [`MSGARCH` manual](https://stat.ethz.ch/CRAN/web/packages/MSGARCH/MSGARCH.pdf)

- `stochvol` 3.2.9 provides efficient Bayesian MCMC for stochastic volatility with optional leverage. `factorstochvol` 1.1.1 adds sparse latent-factor multivariate SV. Both had 2026 CRAN releases. [`stochvol` CRAN page](https://CRAN.R-project.org/package=stochvol), [`factorstochvol` CRAN page](https://CRAN.R-project.org/package=factorstochvol)

- `gasmodel` supports generic generalized autoregressive score models with higher score and AR orders, exogenous variables, missing values, bounded or fixed coefficients, filtering, forecasting, simulation, and bootstrap inference. Its 36 documented distributions include univariate Normal, t, GED, and multivariate Normal and t. [`gasmodel` repository](https://github.com/vladimirholy/gasmodel)

- `dccmidas` covers corrected DCC, asymmetric DCC, DCC-MIDAS, dynamic equicorrelation, scalar and diagonal BEKK, GARCH-MIDAS, and double-asymmetric GARCH-MIDAS. [`dccmidas` CRAN page](https://CRAN.R-project.org/package=dccmidas)

- `highfrequency` 1.0.3 cleans trades and quotes, constructs univariate and multivariate realized measures, estimates HEAVY models, detects jumps, and handles microstructure noise and intraday periodicity. [`highfrequency` CRAN page](https://CRAN.R-project.org/package=highfrequency), [`HEAVYmodel` reference](https://search.r-project.org/CRAN/refmans/highfrequency/html/HEAVYmodel.html)

- `bmgarch` 2.1.0 uses Stan for Bayesian DCC, CCC, positive-definite BEKK, and BEKK under multivariate Normal or Student-t errors, with weighted mean, variance, and correlation forecasts. [`bmgarch` CRAN page](https://CRAN.R-project.org/package=bmgarch)

## MATLAB and OxMetrics

MathWorks' Econometrics Toolbox has a consistent object workflow for GARCH, EGARCH, and GJR models. Objects share `estimate`, `infer`, `filter`, `forecast`, `simulate`, and `summarize`; the Econometric Modeler app can generate MATLAB code and reports. Innovation distributions are Gaussian or Student-t. The toolbox offers ARCH tests, likelihood-ratio/Wald testing, information criteria, and examples for conditional mean plus variance models. Its documented multivariate models are VAR/VEC and state-space, not multivariate GARCH. [`Econometrics Toolbox`](https://www.mathworks.com/help/econ/), [`garch` object](https://www.mathworks.com/help/econ/garch.html), [model specification](https://www.mathworks.com/help/econ/specify-garch-models-using-garch.html), [function index and tests](https://www.mathworks.com/help/econ/referencelist.html)

The open MFE toolbox is a broader algorithmic benchmark. Its documented catalog includes ARCH/GARCH/GJR/TARCH/ZARCH, EGARCH, APARCH, AGARCH/NAGARCH, IGARCH, FIGARCH, and HEAVY; multivariate CCC, DCC/ADCC, BEKK, OGARCH/GOGARCH, RARCH, and RCC; realized variance/covariance, realized kernels, robust jump measures, and quarticity; block/stationary bootstraps; Reality Check, SPA, and MCS. C MEX code accelerates core recursions. Treat it as reference code rather than a modern package contract: the author's feature page says it was last updated in 2013, the GitHub repository has no formal releases, and a third-party MATLAB File Exchange package was updated in 2024. [MFE toolbox feature list](https://www.kevinsheppard.com/code/matlab/mfe-toolbox/), [official repository](https://github.com/bashtage/mfe-toolbox), [MathWorks File Exchange entry](https://www.mathworks.com/matlabcentral/fileexchange/170381-mfe-toolbox-kevin-sheppard)

OxMetrics G@RCH 10 is the broadest commercial point of comparison. Its univariate set includes GARCH, EGARCH, GJR, APARCH, IGARCH, RiskMetrics, FIGARCH, FIEGARCH, FIAPARCH, and HYGARCH, with mean and variance regressors and Normal, Student-t, GED, or skew-t errors. Its multivariate set includes scalar, diagonal, and full BEKK, CCC, DCC, DECO, RiskMetrics, OGARCH, and GOGARCH. The package also has realized-volatility tools, VaR, multiplicative-error models, multi-step forecasts, and specification tests such as Nyblom stability, sign bias, Pearson goodness of fit, Box-Pierce, conditional-heteroskedasticity diagnostics, multivariate portmanteau tests, and constant-correlation tests. G@RCH 10 integrates MulCom forecast comparison and runs through a GUI, batches, or Ox programs. [`G@RCH 10 overview`](https://oxmetrics.dev/doc/G%40RCH/page_main.html), [`OxMetrics 10`](https://oxmetrics.dev/doc/OxMetrics/page_main.html)

## Julia

`ARCHModels.jl` is the strongest non-R multivariate open-source benchmark. It estimates and simulates ARMA plus ARCH, GARCH, TGARCH/GJR, and EGARCH models under Normal, standardized t, Hansen skew-t, GED, or user-wrapped standardized distributions. It supplies ARCH LM and dynamic-quantile tests, model selection, forecasts, and integration with Julia's statistical-model interface. Its documentation and repository were active in 2026. [`ARCHModels.jl` home](https://s-broda.github.io/ARCHModels.jl/stable/), [univariate type system](https://s-broda.github.io/ARCHModels.jl/stable/univariatetypehierarchy/), [repository](https://github.com/s-broda/ARCHModels.jl)

The package implements CCC and DCC(p,q). The default large-scale DCC estimator replaces the sample covariance target with nonlinear shrinkage and maximizes a sum of pairwise likelihoods, avoiding large matrix inversions. The classical two-step estimator remains available. Forward-mode AD is an explicit fit option. The present multivariate standardized distribution is Gaussian only. [`ARCHModels.jl` multivariate models](https://s-broda.github.io/ARCHModels.jl/stable/multivariatetypehierarchy/), [`ARCHModels.jl` usage](https://s-broda.github.io/ARCHModels.jl/stable/usage/)

## Stan and PyMC

Stan's current user guide contains executable GARCH(1,1) and latent stochastic-volatility models. It gives users exact control over priors and constraints and compiles the log posterior and gradients to C++. The stochastic-volatility chapter also shows why parametrization matters: a non-centered reformulation greatly improves mixing, while dense mass matrices do not scale to long series. [`Stan time-series guide`](https://mc-stan.org/docs/stan-users-guide/time-series.html), [`Stan stochastic volatility`](https://mc-stan.org/docs/2_33/stan-users-guide/stochastic-volatility-models.html)

PyMC's official example uses a latent Gaussian random walk for log volatility with Student-t returns and NUTS inference. PyMC provides posterior predictive sampling and ArviZ diagnostics, but no specialist model catalog or volatility result interface. [`PyMC stochastic-volatility example`](https://www.pymc.io/projects/examples/en/latest/time_series/stochastic_volatility.html), [`PyMC distribution API`](https://www.pymc.io/projects/docs/en/stable/api/distributions.html)

These frameworks make excellent development oracles and Bayesian escape hatches. They are poor default engines for a library whose main claim is the fastest frequentist estimation across many specifications.

## Other commercial benchmarks

Stata 19's univariate `arch` command covers a wide ARCH/GARCH family, including EGARCH, GJR, APARCH and other power or nonlinear forms, ARMA mean terms, ARCH-in-mean, robust covariance, constraints, rolling estimation, and forecasts. `mgarch` implements diagonal VECH, CCC, DCC, and VCC with multivariate Normal or t errors and standard or robust covariance. Its postestimation system and treatment of missing data are valuable workflow references. [`arch` manual](https://www.stata.com/manuals/tsarch.pdf), [`mgarch` manual](https://www.stata.com/manuals/ts.pdf), [current time-series features](https://www.stata.com/features/time-series/)

EViews supports GARCH/TARCH, EGARCH, PARCH, component GARCH, FIGARCH, FIEGARCH, and MIDAS-GARCH, including variance regressors, ARCH-in-mean, variance targeting, and Bollerslev-Wooldridge QML covariance. Its manual states exactly where it uses analytical derivatives: simple GARCH(p,q) with uncomplicated mean equations under Normal or unrestricted t errors. More complex models and distributions use numerical or mixed derivatives. That per-specification disclosure is a useful precedent for scivol's `fit_info`. [EViews model catalog](https://help.eviews.com/content/arch-Additional_ARCH_Models.html), [EViews estimation and derivative methods](https://help.eviews.com/content/arch-Estimating_ARCH_Models_in_EViews.html), [EViews command reference](https://help.eviews.com/content/commandcmd-arch.html)

## What distinguishes scivol now

scivol's strongest technical claim is not raw model count. It is the combination of specialized C recursions, exact observed gradients and Hessians, independent AD oracles, separately implemented constrained and transformed optimization paths, transparent `fit_info`, rich standardized GH-family densities, and one composition API across mean, volatility, density, and correlation components. None of the surveyed systems documents that entire correctness chain.

The main competitive risks are different:

1. `tsgarch` and `tsmarch` make AD, profiling, scoring, filtering, rolling backtests, and inference choices part of the public workflow.
2. `ARCHModels.jl` has a credible high-dimensional DCC design that avoids full matrix inversions.
3. `rugarch`, OxMetrics, EViews, and specialist R packages cover long memory, realized measures, mixed-frequency inputs, regime changes, and score-driven models.
4. `tstests`, Stata, and OxMetrics offer a fuller route from fit to model criticism and regulatory risk backtesting.

## Actionable implications for scivol

The next work should favor gaps that strengthen the whole library rather than inflate the model list.

1. **Complete risk evaluation and diagnostics first.** Add Expected Shortfall to forecast results, joint VaR/ES scoring, Kupiec and Christoffersen coverage tests, duration and dynamic-quantile tests, Engle-Ng sign bias, ARCH LM and squared-residual portmanteau tests, Nyblom parameter stability, Berkowitz, Hong-Li, and forecast calibration tests. scivol already has PIT/DGT tests, VaR, rolling losses, and MCS, so this is a bounded extension with large practical value.

2. **Add a realized and mixed-frequency branch.** The most defensible sequence is HEAVY, Realized GARCH, then GARCH-MIDAS. These models open high-frequency volatility forecasting and macro-finance applications. They also require explicit measurement equations, missing/irregular observation handling, and realized-measure input contracts, all reusable infrastructure.

3. **Fill the long-memory and component gap.** FIGARCH is the obvious compatibility target because `arch`, `rugarch`, OxMetrics, EViews, and MFE all ship it. Component GARCH should follow. FIEGARCH, FIAPARCH, and HYGARCH are lower priority unless benchmarks show demand. FIGARCH's truncation, initialization, stationarity interpretation, and forecast approximation must be visible in results.

4. **Broaden multivariate structure in layers.** Corrected DCC, asymmetric DCC, and DECO fit the existing DCC decomposition and should precede BEKK. Add pairwise-composite or other large-scale DCC estimation, with shrinkage targets, before claiming high-dimensional support. Then add scalar/diagonal BEKK and GO-GARCH or factor covariance models. Report whether inference is full, partitioned, OPG, or conditional on first-stage marginal estimates.

5. **Treat copulas as their own component family.** A common-df Student-t correlation likelihood is useful but narrower than copula-GARCH. Gaussian, Student-t, and selected asymmetric or tail-dependent copulas would separate marginal density choice from dependence and support portfolio tail risk without forcing identical marginals.

6. **Reserve separate architecture for regime-switching, GAS, and latent SV.** Markov-switching GARCH and score-driven volatility are plausible medium-term additions. Bayesian stochastic volatility is a different computational product. It should enter through an optional backend or companion package rather than compromise scivol's small, fast frequentist core.

7. **Publish cross-package conformance and speed suites.** Match likelihood, score, Hessian, filtered variance, forecasts, and robust covariance against `arch`, `rugarch`/`tsgarch`, `ARCHModels.jl`, MFE, Stata, or OxMetrics under identical scaling, backcasts, burn-in, and effective samples. Report disagreements rather than comparing only final parameter estimates. This would turn scivol's internal derivative discipline into a visible reason to trust the package.
