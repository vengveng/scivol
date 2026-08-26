# Realized volatility and high-frequency models

## Bottom line

High-frequency volatility is the largest coherent addition scivol could make without abandoning its current identity. The missing piece is not one more GARCH recursion. It is a measurement layer that converts trades or quotes into daily realized measures, carries measurement uncertainty forward, and feeds a family of daily models.

The most useful first release would contain:

1. A trustworthy realized-measure engine: realized variance and covariance, subsampling, realized kernels, pre-averaging, quarticity, bipower and threshold measures, semivariances, jumps, and overnight variation.
2. HAR, log-HAR, HARQ, and signed/jump HAR specifications with direct and iterated multi-horizon forecasts.
3. Realized GARCH and Realized EGARCH, including several realized measures and full likelihood diagnostics.
4. HEAVY models, followed by DCC-HEAVY and DECO-HEAVY for multivariate work.
5. GARCH-MIDAS for short-run and long-run decomposition with release-aware macroeconomic regressors.
6. Positive-definite realized covariance estimators and forecasts, then Multivariate Realized GARCH.

This order matters. Model estimates are only as defensible as the synchronization, cleaning, session, noise, and jump decisions used to construct their realized inputs. A fast likelihood on a vaguely defined `rv` column would fall short of scivol's correctness goal.

## 1. What realized measures estimate

For intraday log returns \(r_{t,i}=p_{t,i}-p_{t,i-1}\), realized variance is

\[
RV_t=\sum_{i=1}^{M_t}r_{t,i}^2.
\]

As the observation mesh shrinks, and in the absence of market microstructure noise, this estimates quadratic variation:

\[
[p,p]_t=\int_t^{t+1}\sigma_s^2\,ds+\sum_{t<s\leq t+1}(\Delta p_s)^2.
\]

The first term is integrated variance. The second is jump variation. This distinction explains much of the realized-measure literature. Plain RV estimates total variation. Bipower, multipower, median, and threshold estimators seek the continuous component. The difference between total and continuous estimates supplies jump estimates and tests. Andersen et al. established the empirical usefulness of realized volatility for exchange rates, and Barndorff-Nielsen and Shephard developed the associated power-variation and jump theory ([Andersen et al., 2001](https://doi.org/10.1198/016214501750332965); [Barndorff-Nielsen and Shephard, 2004](https://doi.org/10.1093/jjfinec/nbh001)).

At very high sampling frequencies the observed price is better written as \(p^{obs}_{t_i}=p_{t_i}+\epsilon_{t_i}\). Bid-ask bounce, discrete ticks, stale quotes, latency, and price discovery then contaminate RV. Hansen and Lunde show that the noise may be dependent and correlated with efficient-price increments, not benign independent error ([Hansen and Lunde, 2006](https://doi.org/10.1198/073500106000000071)). Consequently, "use every tick" is generally wrong for plain RV.

The package should distinguish three objects throughout its API:

- `variance`: quadratic or integrated variance, in squared-return units.
- `volatility`: the square root of variance, with annualization stored separately.
- `measure_uncertainty`: an estimator of the realized measure's sampling variance, often derived from realized quarticity.

Conflating variance and volatility remains a common source of incorrect annualization, loss functions, and regressors.

## 2. The realized-measure catalog

### 2.1 Baselines that should always be present

**Sparse and subsampled realized variance.** Five-minute RV remains a strong operational benchmark. It gives up some data to reduce noise, but it is transparent and often hard to beat out of sample. Subsampling averages RV over displaced time grids to reduce dependence on an arbitrary grid origin. The library should make grid width, offset set, session endpoints, price field, and annualization explicit.

**Realized quarticity.** \(RQ_t=(M_t/3)\sum_i r_{t,i}^4\) estimates integrated quarticity under a continuous diffusion. It supports confidence intervals, feasible central limit theorems, HARQ, bandwidth selection, and measurement-error diagnostics. Jump-robust quarticity variants are also needed because a few large returns can dominate the fourth power.

**Realized range.** Range-based measures use intraperiod highs and lows and can improve efficiency when only a modest number of intraday blocks is available. They are useful for OHLC data, but cannot replace noise-aware tick estimators. The realized-range estimator of Christensen and Podolskij is the natural primary reference ([Christensen and Podolskij, 2007](https://doi.org/10.1016/j.jeconom.2006.06.012)).

### 2.2 Noise-robust estimators

**Realized kernels.** A realized kernel is a weighted sum of return autocovariances,

\[
RK_t=\hat\gamma_0+\sum_{h=1}^{H}k(h/H)(\hat\gamma_h+\hat\gamma_{-h}).
\]

The kernel and bandwidth offset the serial dependence induced by market microstructure noise. The non-negative Parzen form, endpoint jittering, plug-in bandwidth estimation, and diagnostics in Barndorff-Nielsen et al. are the most practical default ([design theory, 2008](https://doi.org/10.3982/ECTA6495); [trades and quotes implementation, 2009](https://doi.org/10.1111/j.1368-423X.2008.00275.x)). Bandwidth selection must be part of the estimator result, not a hidden implementation detail.

**Two-scale and multiscale realized variance.** Zhang, Mykland, and Ait-Sahalia combine fast and sparse scales to correct noise bias ([Zhang et al., 2005](https://doi.org/10.1198/016214505000000169)). These estimators remain useful as independent checks and are simpler to explain than a full kernel.

**Pre-averaging.** Pre-averaging smooths local returns before constructing variation and subtracts a noise-bias term. It reaches the optimal \(n^{-1/4}\) rate under standard noisy-observation asymptotics and extends naturally to covariance and power variation ([Jacod et al., 2009](https://doi.org/10.1016/j.spa.2008.11.004)). A package implementation should accept the weight function and window, expose the bias correction, and return feasible standard errors.

**Noise QMLE.** Under additive independent noise, observed returns have an MA(1)-type covariance structure. Gaussian quasi-likelihood can estimate integrated variance and noise variance jointly at the optimal noisy-data rate ([Xiu, 2010](https://doi.org/10.1016/j.jeconom.2010.07.002)). This is an attractive scivol fit: it supplies an independent, likelihood-based public estimator and admits fast banded calculations.

These methods should coexist. Their assumptions differ, and agreement across sparse RV, kernel, and pre-averaged estimates is a useful empirical diagnostic. The goal should not be one universal "best" measure.

Recent work relaxes the convenient assumption that noise is independent, identically distributed, and stationary. Li, Laeven, and Vellekoop derive finite-sample corrections for pre-averaging under serially dependent noise ([2020](https://doi.org/10.1016/j.jeconom.2019.10.004)). Da and Xiu use an adaptively selected MA(\(q\)) quasi-likelihood and obtain inference that remains valid as noise magnitude and dependence change ([2021](https://doi.org/10.3982/ECTA15593)). Li and Linton's ReMeDI estimates general noise moments while allowing serial dependence, endogeneity, and nonstationarity ([2022](https://doi.org/10.3982/ECTA17505)). These papers support a public `MicrostructureNoiseResult` containing variance, autocovariances, dependence order, intraday pattern, endogeneity diagnostics, and uncertainty. Noise estimates should not remain hidden bandwidth inputs.

At high dimension, robust pre-averaging and structured regularization have become central. Shin, Kim, and Fan combine pre-averaging, adaptive truncation, and POET regularization to handle nonsynchronicity, jumps, heavy tails, and many assets ([2023](https://doi.org/10.1016/j.jeconom.2023.105514)). This belongs after the core multivariate estimators, but it supplies the right long-run design: covariance estimation and regularization should be composable stages with separate diagnostics.

A signature plot, RV over a range of sampling intervals, remains a useful first noise check. An eventual formal option should implement the Ait-Sahalia-Xiu Hausman comparison of a no-noise efficient estimator and a noise-robust estimator ([2019](https://doi.org/10.1016/j.jeconom.2018.12.013)).

### 2.3 Jumps, signed variation, and quarticity

**Bipower and multipower variation.** Realized bipower variation,

\[
BV_t=\mu_1^{-2}\sum_{i=2}^{M_t}|r_{t,i}||r_{t,i-1}|,
\]

estimates continuous variation under finite-activity jumps. Tripower and higher multipower measures extend the idea and provide jump-robust quarticity. Zero returns, staleness, and microstructure noise can create meaningful finite-sample bias, so they need diagnostic counts and warnings rather than blind formulas.

**Threshold and truncated measures.** Truncation removes increments above a local, volatility-scaled threshold. It can estimate continuous variation more efficiently than multipower measures when tuning is sound, but the answer depends on threshold choice. The result object should retain the threshold series and fraction of excluded returns. Nearest-neighbor truncation and median realized volatility offer alternatives with different robustness properties ([Andersen, Dobrev, and Schaumburg, 2012](https://doi.org/10.1016/j.jeconom.2012.01.011)).

**Jump estimates and tests.** At minimum, provide the Barndorff-Nielsen-Shephard standardized jump test, threshold tests, and finite-sample bootstrap options. Return continuous variation, jump variation, the test statistic, p-value, and all tuning inputs. Never silently replace negative \(RV-BV\) estimates with zero without also returning the untruncated estimate.

**Realized semivariances.** The decomposition

\[
RS_t^- = \sum_i r_{t,i}^2\mathbf 1(r_{t,i}<0),\qquad
RS_t^+ = \sum_i r_{t,i}^2\mathbf 1(r_{t,i}\geq 0)
\]

separates "bad" and "good" intraday variation. This is useful in downside-risk forecasts, leverage analysis, beta decomposition, and portfolio allocation. Recent work extends the idea to semicovariances and multiple return thresholds. The broad empirical lesson is restrained: the zero threshold is hard to improve upon in high dimensions, despite the flexibility of data-chosen partial-variation thresholds ([Bollerslev, 2022](https://doi.org/10.1093/jjfinec/nbab025); [Bollerslev et al., 2022](https://doi.org/10.1016/j.jeconom.2021.04.013)).

### 2.4 Covariance, asynchrony, and positive definiteness

For synchronized return vectors \(r_{t,i}\), realized covariance is

\[
RCov_t=\sum_i r_{t,i}r_{t,i}^{\mathsf T}.
\]

Real data are asynchronous. Previous-tick interpolation can induce the Epps effect, pairwise refresh times can produce an indefinite matrix, and global refresh time discards data as the asset count rises. A complete package should therefore provide:

- calendar-time previous-tick sampling;
- pairwise and all-refresh-time synchronization;
- Hayashi-Yoshida covariance for nonsynchronous observations ([Hayashi and Yoshida, 2005](https://doi.org/10.3150/bj/1116340299));
- multivariate realized kernels;
- pre-averaged or modulated realized covariance ([Christensen, Kinnebrock, and Podolskij, 2010](https://doi.org/10.1016/j.jeconom.2010.05.001));
- nearest positive-definite projection and shrinkage as explicit post-processing steps.

The multivariate realized kernel is particularly important because it handles noise and nonsynchronous trading while guaranteeing a positive semidefinite covariance estimate ([Barndorff-Nielsen et al., 2011](https://doi.org/10.1016/j.jeconom.2010.07.009)). The estimator should report effective observations by pair, refresh-time losses, eigenvalues before and after any projection, condition number, and whether the matrix is positive semidefinite only or strictly positive definite.

For large asset sets, factor and shrinkage estimators are not optional extras. Raw realized covariance becomes noisy and poorly conditioned as dimension approaches the effective number of intraday observations.

## 3. Daily models built on realized measures

### 3.1 HAR and HARQ

The heterogeneous autoregressive model is the essential baseline:

\[
y_{t+1}=\beta_0+\beta_d y_t+\beta_w\bar y_{t,5}+\beta_m\bar y_{t,22}+u_{t+1},
\]

where \(y_t\) is usually RV, realized volatility, or log RV. HAR is a short-memory linear model whose daily, weekly, and monthly averages approximate the persistent autocorrelation of realized volatility with few parameters ([Corsi, 2009](https://doi.org/10.1093/jjfinec/nbp001)).

The implementation should include:

- levels, square-root, and log specifications, with correct back-transformation and optional bias correction;
- arbitrary horizon blocks rather than hard-coded 1, 5, and 22;
- direct and iterated forecasts;
- overlapping-horizon HAC covariance;
- rolling, expanding, and fixed-window estimation;
- HAR-J and HAR-CJ with continuous and jump terms;
- HAR-RS with positive and negative semivariances;
- leverage terms and external regressors;
- pooled, panel, and vector HAR variants.

HARQ allows the impact of lagged RV to vary with realized quarticity, directly addressing time-varying measurement error. It improved forecasts in the original S&P 500 and Dow stock applications ([Bollerslev, Patton, and Quaedvlieg, 2016](https://doi.org/10.1016/j.jeconom.2015.10.007)). It is a high-priority feature because it joins the measurement and forecasting layers cleanly. The package should also expose errors-in-variables diagnostics, since later work finds that measurement-error modeling remains consequential ([Clements and Preve, 2021](https://doi.org/10.1016/j.ijforecast.2020.02.009)).

HAR's simplicity is a virtue. A 2024 comparison across ten equity indices found no general statistical dominance of nonlinear machine-learning forecasts over linear models; leveraged HAR and HAR with predictors were among the strongest statistical models, while simple HAR-family models produced high economic value ([Branco, Rubesam, and Zevallos, 2024](https://doi.org/10.1016/j.jempfin.2024.101524)). Any advanced model in scivol should therefore benchmark itself against carefully tuned HAR, HARQ, and signed HAR, not only GARCH(1,1).

### 3.2 Realized GARCH and Realized EGARCH

Realized GARCH jointly models the daily return and an observed realized measure. A representative linear form is

\[
r_t=\sqrt{h_t}z_t,\qquad
h_t=\omega+\beta h_{t-1}+\gamma x_{t-1},
\]

\[
x_t=\xi+\phi h_t+\tau_1 z_t+\tau_2(z_t^2-1)+u_t.
\]

The measurement equation links noisy realized variance \(x_t\) to latent conditional variance \(h_t\) and captures leverage through contemporaneous standardized-return terms. Log-linear Realized GARCH keeps variance positive and often behaves better numerically. Hansen, Huang, and Shek supply the core model and likelihood ([Hansen et al., 2012](https://doi.org/10.1002/jae.1234)).

Realized EGARCH generalizes the leverage function and allows several realized measures. In the original 27-stock and ETF application, multiple-measure specifications dominated single-measure models ([Hansen and Huang, 2016](https://doi.org/10.1080/07350015.2015.1038543)). This argues for an API that accepts an \(n\times k\) matrix of aligned measures, not a single special `rv` array.

Recommended density and inference support:

- Normal, standardized Student-t, skew-t, and a user-supplied return density;
- Gaussian and Student-t measurement errors, plus robust sandwich covariance;
- separate likelihood contributions for returns and each measurement equation;
- standardized return and measurement residual diagnostics;
- one-step and multi-step return variance, realized-measure, VaR, and ES forecasts;
- missing realized measures without discarding the corresponding daily return;
- parameter restrictions, stationarity, and unconditional-moment reporting;
- exact gradients and Hessians checked against the project's AD oracle.

### 3.3 HEAVY models

HEAVY models use yesterday's realized measure to update today's conditional return variance and model the realized measure with its own multiplicative-error recursion. They respond to volatility shocks faster than daily-return-only GARCH because intraday variation is informative before many large daily squared returns accumulate ([Shephard and Sheppard, 2010](https://doi.org/10.1002/jae.1158)).

HEAVY deserves its own component rather than being treated as a generic GARCH-X option. The model's forecasting recursion, targeting, persistence, and multi-step properties depend on the paired return and realized-measure equations. Expose targeting and non-targeting parameterizations and provide return-side, measure-side, and joint diagnostics.

Multivariate HEAVY is a practical bridge from scivol's DCC support to realized covariance. DCC-HEAVY and DECO-HEAVY use realized variances and correlations, include asymmetric variants, and outperformed daily-data BEKK, DCC, and DECO benchmarks in the authors' application ([Bauwens and Xu, 2023](https://doi.org/10.1016/j.ijforecast.2022.03.005)).

### 3.4 MIDAS and GARCH-MIDAS

MIDAS regressions compress many high-frequency lags into a low-dimensional beta or exponential-Almon weighting scheme. GARCH-MIDAS decomposes conditional variance into a short-run GARCH component and a slowly moving component driven by low-frequency macroeconomic or financial regressors. The model answers a different question from realized GARCH: it attributes long-run volatility to business-cycle or policy variables while retaining daily dynamics ([Engle, Ghysels, and Sohn, 2013](https://doi.org/10.1162/REST_a_00300)).

A correct implementation must handle data vintages. Using revised macroeconomic values or future releases at historical forecast origins leaks information. The API should accept `observation_time`, `release_time`, and optional `vintage_time`, and it should construct each design matrix as of the forecast origin. It should support daily, weekly, monthly, and quarterly regressors, ragged edges, multiple MIDAS terms, fixed-span and rolling weights, and GARCH-MIDAS with realized measures.

The empirical evidence is horizon-dependent. A multiplicative-component comparison found housing-starts GARCH-MIDAS strongest at two- and three-month horizons, not uniformly at short horizons ([Conrad and Kleen, 2020](https://doi.org/10.1002/jae.2742)). Forecast reports should therefore separate horizons and statistical from portfolio or risk-management loss.

### 3.5 Multivariate Realized GARCH and realized beta

Realized Beta GARCH combines daily returns with realized variances and covariances to model dynamic conditional beta. Its estimates moved more than conventional rolling daily-return betas during crises, which makes it useful for hedging, factor exposure, and systemic-risk work ([Hansen, Lunde, and Voev, 2014](https://doi.org/10.1002/jae.2389)).

The newer Multivariate Realized GARCH model incorporates realized volatilities and correlations and parameterizes correlation matrices through off-diagonal elements of their matrix logarithm. The transformed vector is unconstrained, while the inverse map returns a valid correlation matrix. Factor and block-correlation forms reduce dimension. The paper was revised through 2025 and appears in the March 2026 *Journal of Econometrics* ([Archakov, Hansen, and Lunde, 2026](https://doi.org/10.1016/j.jeconom.2025.106040)). This is the clearest recent multivariate omission for scivol.

An implementation should expose:

- full, block, equicorrelation, and factor correlation dynamics;
- matrix-log and inverse transforms as tested public utilities;
- realized variance and correlation measurement equations;
- two-step and joint estimation;
- PSD input validation and controlled regularization;
- analytical scores and observed Hessians where tractable;
- likelihood contributions by asset, factor, and date;
- portfolio variance, beta, correlation, VaR, and ES forecasts.

## 4. Recent developments, 2020 to 2026

The period's useful developments refine the core realized framework rather than overturn it.

1. **Signed and partial covariation matured.** Semicovariance and partial covariance now support richer downside-risk, beta, and network decompositions. Yet the zero cutoff often remains the most stable high-dimensional choice. scivol should implement the general measure but default to zero-threshold semivariation.
2. **Multivariate models became more usable.** DCC/DECO-HEAVY reached journal publication in 2023. Multivariate Realized GARCH reached final publication in 2026 with a correlation transform that avoids awkward direct constraints.
3. **Forecast reconciliation reached intraday decompositions.** Recent work forecasts non-overlapping intraday, sign, or quantile components and reconciles them so they add to daily volatility ([Caporin, Di Fonzo, and Girolimetto, 2024](https://doi.org/10.1093/jjfinec/nbae014)). This suggests an eventual `RealizedDecomposition` object with aggregation constraints and coherent forecasts.
4. **Outlier resistance moved into covariance forecasting.** Robust realized covariance inputs and least-trimmed multivariate HAR estimation improve stability when isolated days dominate ordinary least squares ([Li, Drovandi, and Clements, 2024](https://doi.org/10.1016/j.ijforecast.2023.04.004)). Robust loss and influence diagnostics belong in multivariate HAR.
5. **Simple linear baselines remain hard to displace.** The best recent broad comparison does not find general nonlinear-model dominance over linear HAR variants. scivol should support external ML predictors and benchmark adapters, but it should not make a neural model the center of the realized module.
6. **Rough volatility created a useful measurement warning.** Gatheral, Jaisson, and Rosenbaum estimate very low Hurst exponents for log volatility and link roughness to high-frequency market activity ([2018](https://doi.org/10.1080/14697688.2017.1393551)). Later work shows that realized-volatility estimation error itself can create apparent roughness ([Cont and Das, 2024](https://doi.org/10.1007/s13571-024-00322-2)). A `roughness_test` should therefore work with several volatility proxies, simulate the complete observation and estimation pipeline under the null, and avoid interpreting a log-log slope as direct evidence about latent spot volatility.

Rough volatility is adjacent to this module, not a replacement for it. Realized measures provide calibration and validation targets for rough stochastic-volatility models, while rough models address latent spot dynamics and option smiles. A later scivol addition could offer rough fractional stochastic-volatility forecasting, Hurst estimation with measurement-error correction, and realized-variance DGPs. It should arrive after the realized-measure engine can reproduce the bias debates.

## 5. Data workflow that a correct library should enforce

### 5.1 Inputs and sessions

Accept long-form ticks or pre-sampled bars with a schema such as:

```text
asset, event_time, sequence, trade_price, trade_size,
bid, ask, bid_size, ask_size, exchange, condition
```

Require timezone-aware event times. A trading calendar must define session opens, closes, half-days, breaks, auctions, and daylight-saving transitions. Store the calendar and timezone in the output metadata. For assets with near-continuous trading, define the day boundary explicitly.

Price choice should be an option: trade, bid, ask, midpoint, weighted midpoint, or user-supplied efficient-price estimate. Trade and quote cleaning rules differ. The result should preserve counts before and after each filter.

### 5.2 Cleaning

A reproducible default pipeline should:

1. sort by event time and sequence number;
2. resolve duplicates under a named rule;
3. remove invalid prices, crossed or locked quotes according to policy, and disallowed trade conditions;
4. apply corporate-action adjustments consistently across intraday and close prices;
5. flag outliers with local-median or quote-relative filters;
6. retain, flag, or remove zero returns under an explicit rule;
7. separate regular sessions, auctions, and overnight returns;
8. emit a cleaning report with counts and reasons.

Barndorff-Nielsen et al. show that cleaning and realized-kernel behavior interact materially ([2009](https://doi.org/10.1111/j.1368-423X.2008.00275.x)). The cleaning report is therefore part of the statistical result, not incidental data engineering.

### 5.3 Sampling and synchronization

Support calendar-time bars, business-time sampling, previous tick, nearest tick, linear interpolation only when requested, and refresh time. Default resampling should never cross a session boundary. Return the timestamp selected for every grid point when `audit=True`.

For covariance, provide pairwise observation counts and synchronization loss. Users need to see when one illiquid asset reduces an all-refresh-time grid from thousands of observations to dozens.

### 5.4 Overnight variation

Open-to-close intraday RV omits close-to-open information. Models of close-to-close returns need an explicit overnight policy:

- keep intraday and overnight squared returns as separate measures;
- add the raw overnight squared return;
- scale the overnight term using an estimated coefficient;
- fit a joint intraday/overnight model.

The separate-components design is best. Overnight news has different dynamics and microstructure, and treating a single overnight return as equivalent to hundreds of intraday increments is rarely neutral.

### 5.5 Result objects and provenance

A `RealizedMeasureResult` should contain:

```text
value, continuous, jump, quarticity, standard_error,
n_raw, n_clean, n_used, sampling, synchronization,
kernel, bandwidth, preaverage_window, threshold,
session, timezone, annualization, cleaning_report,
is_psd, eigenvalues, condition_number, provenance
```

`provenance` should include the estimator version, input hash, schema, calendar version, and all tuning values. Realized measures are derived data. Two columns both named `rv5` need not be comparable if their sessions, prices, corrections, or grids differ.

## 6. Tests and validation

### 6.1 Deterministic unit tests

- Constant prices produce zero variation.
- A single known jump produces its squared size in RV and the expected signed bucket.
- A hand-computed return sequence matches RV, covariance, bipower, quarticity, and semivariance formulas.
- Scaling all log returns by \(c\) scales variance and covariance by \(c^2\), quarticity by \(c^4\), and volatility by \(|c|\).
- Reversing asset order permutes covariance rows and columns without changing values.
- Identical assets have correlation one and covariance equal to variance.
- Session resampling never carries a price across an overnight boundary.
- Daylight-saving and half-day fixtures produce the documented grid counts.

### 6.2 Estimator Monte Carlo tests

Simulate latent Itô prices with configurable stochastic volatility, leverage, finite- and infinite-activity jumps, endogenous noise, bid-ask bounce, irregular arrival times, and nonsynchronous assets. Test:

- consistency as the mesh shrinks without noise;
- plain RV's noise bias at ultra-high frequency;
- bias and RMSE of sparse RV, two-scale, kernel, and pre-averaging estimators;
- empirical coverage of feasible confidence intervals;
- jump-test size under no jumps and power under several jump regimes;
- PSD frequency, smallest eigenvalue, and Frobenius loss for covariance estimators;
- sensitivity to bandwidth, threshold, grid origin, zero returns, and staleness.

Published asymptotics are not enough. Many jump-robust estimators struggle in finite samples with active jumps or stale prices. The test suite should retain adversarial parameter sets, not only a clean diffusion.

### 6.3 Model tests

Each model needs a DGP with known parameters and tests for recovery, convergence, and forecast recursion. For Realized GARCH and HEAVY, simulate both returns and realized measures. Missing-measure tests must confirm that the return likelihood remains usable. Validate analytical theta- and log-space derivatives against an independent AD objective at matched parameter points, in line with scivol's existing derivative policy.

For multivariate models test symmetry, unit diagonal, positive definiteness, transform round trips, permutation equivariance, factor/block reductions, and equality of specialized and generic code paths.

### 6.4 Forecast evaluation

Use losses consistent with variance forecasts:

- QLIKE, which is comparatively robust to noise in a volatility proxy;
- squared and absolute error on variance and log variance;
- multivariate Gaussian QLIKE or Stein loss for covariance matrices;
- realized portfolio variance and turnover-adjusted portfolio utility;
- VaR coverage, independence, dynamic-quantile, and joint VaR-ES tests.

Forecast comparisons should include Diebold-Mariano tests with horizon-aware HAC errors, encompassing or Mincer-Zarnowitz regressions, the Superior Predictive Ability test, and the Model Confidence Set. Evaluate fixed, rolling, and expanding schemes. Preserve publication lags for macro and option predictors. Report results by asset, horizon, volatility regime, and liquidity group.

## 7. Proposed scivol API

The following design preserves component composition while separating measurement from modeling:

```python
from scivol.realized import (
    RealizedVariance, RealizedKernel, PreAveragedVariance,
    BipowerVariation, RealizedSemivariance, RealizedCovariance,
)
from scivol import HAR, HARQ, RealizedGARCH, RealizedEGARCH, HEAVY, StudentT

rv = RealizedKernel(kernel="parzen", bandwidth="plug_in").compute(ticks)
rq = rv.companion("quarticity")

forecast = HARQ(horizons=(1, 5, 22)).fit(rv.value, quarticity=rq.value)

spec = RealizedGARCH(1, 1, measurement="log") + StudentT()
result = spec.fit(daily_returns, realized=rv)

multi = RealizedCovariance(
    synchronization="refresh",
    estimator="multivariate_kernel",
).compute(ticks)
```

Important behavioral rules:

- Models accept either arrays or typed realized-measure results. Typed results allow automatic unit, alignment, and provenance checks.
- A model never silently annualizes, logs, square-roots, fills missing values, clips jumps, or projects a covariance matrix.
- Every result reports the effective estimator, solver, transform, sample, and realized-input construction.
- C kernels should operate on preallocated contiguous arrays and release the GIL. Tick parsing, grouping, and resampling need streaming or chunked execution so a year of TAQ data need not fit in memory.
- Fixed-bandwidth kernel, multiscale, and pre-averaging estimators can be expressed as quadratic forms and computed with bounded memory ([Holy and Tomanova, 2023](https://doi.org/10.1007/s10614-021-10210-w)). Batch and streaming modes should produce the same result for fixed tuning parameters.
- Parallelism should be across days and assets where estimates are independent. Deterministic reductions and reproducible ordering matter more than maximum thread count.

## 8. Recommended implementation sequence

| Priority | Feature | Why it belongs here | Exit criterion |
|---|---|---|---|
| P0 | Realized-data schema, calendars, cleaning report, resampling | Every later feature depends on it | Reproducible tick-to-grid fixtures across DST, half-days, duplicates, and bad quotes |
| P0 | RV, covariance, quarticity, subsampling, semivariances | Baseline measurement and forecast proxies | Formula tests, unit invariants, and cross-checks against a second implementation |
| P0 | Realized kernel and pre-averaging | Noise-robust production measures | Monte Carlo bias, RMSE, coverage, bandwidth/window diagnostics |
| P0 | Bipower, threshold, median measures and jump tests | Continuous/jump separation | Size and power grids with finite-sample reporting |
| P1 | HAR, log-HAR, HARQ, signed and jump HAR | Essential transparent baselines | Direct/iterated forecasts, HAC inference, rolling evaluation |
| P1 | Realized GARCH and Realized EGARCH | Natural extension of current component architecture | All densities, multi-measure likelihood, AD derivative parity, DGP recovery |
| P1 | HEAVY | Fast response to realized shocks and simple QMLE | Targeted/non-targeted fits and multi-step forecast tests |
| P2 | Multivariate kernel covariance plus DCC/DECO-HEAVY | Immediate use of current DCC experience | PSD guarantees, realized correlation diagnostics, portfolio losses |
| P2 | GARCH-MIDAS | Long-run macro-volatility attribution | Vintage-safe mixed-frequency alignment and leakage tests |
| P2 | Realized Beta GARCH and Multivariate Realized GARCH | Major recent multivariate omission | Correlation-transform round trips, factor/block forms, DGP recovery |
| P3 | Intraday forecast reconciliation and robust multivariate HAR | Coherent component forecasts and outlier resistance | Aggregation identities and contamination Monte Carlo |
| P3 | Roughness diagnostics and rough-SV links | Research and option-model bridge | Measurement-error-aware simulation and proxy sensitivity report |

## 9. Source table

All links below point to primary papers or author and institutional copies when available.

| Topic | Source | What it supports |
|---|---|---|
| Realized volatility | [Andersen, Bollerslev, Diebold, and Labys (2001), JASA](https://doi.org/10.1198/016214501750332965) | Empirical distribution and use of high-frequency realized exchange-rate volatility |
| Realized-volatility modeling | [Andersen, Bollerslev, Diebold, and Labys (2003), Econometrica](https://doi.org/10.1111/1468-0262.00418) | Reduced-form modeling and forecasting of realized volatility |
| Power and bipower variation | [Barndorff-Nielsen and Shephard (2004), JFE](https://doi.org/10.1093/jjfinec/nbh001) | Continuous variation, jumps, bipower theory |
| Noise and RV | [Hansen and Lunde (2006), JBES](https://doi.org/10.1198/073500106000000071) | Dependent/endogenous microstructure noise and kernel evidence |
| Two-scale RV | [Zhang, Mykland, and Ait-Sahalia (2005), JASA](https://doi.org/10.1198/016214505000000169) | Noise-bias correction using two time scales |
| Realized kernels | [Barndorff-Nielsen, Hansen, Lunde, and Shephard (2008), Econometrica](https://doi.org/10.3982/ECTA6495) | Noise-robust kernel theory and non-negative kernels |
| Kernel implementation | [Barndorff-Nielsen, Hansen, Lunde, and Shephard (2009), Econometrics Journal](https://doi.org/10.1111/j.1368-423X.2008.00275.x) | Cleaning, jittering, bandwidth selection, trades-versus-quotes checks |
| Pre-averaging | [Jacod, Li, Mykland, Podolskij, and Vetter (2009), SPA](https://doi.org/10.1016/j.spa.2008.11.004) | Rate-optimal noise-robust integrated variance and feasible inference |
| Noise QMLE | [Xiu (2010), Journal of Econometrics](https://doi.org/10.1016/j.jeconom.2010.07.002) | Joint likelihood estimation of integrated variance and noise variance |
| Realized range | [Christensen and Podolskij (2007), Journal of Econometrics](https://doi.org/10.1016/j.jeconom.2006.06.012) | Range-based integrated-variance estimation |
| Robust jump measures | [Andersen, Dobrev, and Schaumburg (2012), Journal of Econometrics](https://doi.org/10.1016/j.jeconom.2012.01.011) | Nearest-neighbor truncation and finite-sample jump robustness |
| Dependent-noise pre-averaging | [Li, Laeven, and Vellekoop (2020), Journal of Econometrics](https://doi.org/10.1016/j.jeconom.2019.10.004) | Finite-sample correction and optimal-rate estimation under dependent noise |
| Adaptive noise QMLE | [Da and Xiu (2021), Econometrica](https://doi.org/10.3982/ECTA15593) | Data-selected MA noise dependence and uniform inference as noise changes |
| General noise diagnostics | [Li and Linton (2022), Econometrica](https://doi.org/10.3982/ECTA17505) | Noise moments under dependence, endogeneity, and nonstationarity |
| Noise Hausman test | [Ait-Sahalia and Xiu (2019), Journal of Econometrics](https://doi.org/10.1016/j.jeconom.2018.12.013) | Formal test of whether microstructure noise materially affects volatility estimation |
| Asynchronous covariance | [Hayashi and Yoshida (2005), Bernoulli](https://doi.org/10.3150/bj/1116340299) | Pairwise covariance without interpolating irregular observations |
| Modulated realized covariance | [Christensen, Kinnebrock, and Podolskij (2010), Journal of Econometrics](https://doi.org/10.1016/j.jeconom.2010.05.001) | Pre-averaged covariance under noise and nonsynchronicity |
| Multivariate realized kernels | [Barndorff-Nielsen, Hansen, Lunde, and Shephard (2011), Journal of Econometrics](https://doi.org/10.1016/j.jeconom.2010.07.009) | PSD covariance under noise and nonsynchronous trading |
| Robust high-dimensional covariance | [Shin, Kim, and Fan (2023), Journal of Econometrics](https://doi.org/10.1016/j.jeconom.2023.105514) | Pre-averaging, adaptive truncation, and POET under jumps and heavy tails |
| Streaming realized measures | [Holy and Tomanova (2023), Computational Economics](https://doi.org/10.1007/s10614-021-10210-w) | Bounded-memory quadratic-form computation for several noise-robust estimators |
| HAR | [Corsi (2009), Journal of Financial Econometrics](https://doi.org/10.1093/jjfinec/nbp001) | Parsimonious heterogeneous-horizon realized-volatility forecast model |
| HARQ | [Bollerslev, Patton, and Quaedvlieg (2016), Journal of Econometrics](https://doi.org/10.1016/j.jeconom.2015.10.007) | Quarticity-driven correction for time-varying RV measurement error |
| Measurement error | [Clements and Preve (2021), International Journal of Forecasting](https://doi.org/10.1016/j.ijforecast.2020.02.009) | Robustness of realized-volatility forecasts to multiplicative measurement errors |
| Signed realized variation | [Bollerslev (2022), Journal of Financial Econometrics](https://doi.org/10.1093/jjfinec/nbab025) | Review and applications of semivariation and semicovariation |
| Partial covariation | [Bollerslev, Medeiros, Patton, and Quaedvlieg (2022), Journal of Econometrics](https://doi.org/10.1016/j.jeconom.2021.04.013) | Multiple thresholds and the empirical stability of zero-threshold semivariation |
| Realized GARCH | [Hansen, Huang, and Shek (2012), Journal of Applied Econometrics](https://doi.org/10.1002/jae.1234) | Joint return and realized-measure model with measurement equation |
| Realized EGARCH | [Hansen and Huang (2016), JBES](https://doi.org/10.1080/07350015.2015.1038543) | Flexible leverage and several realized measures |
| HEAVY | [Shephard and Sheppard (2010), Journal of Applied Econometrics](https://doi.org/10.1002/jae.1158) | Return-variance and realized-measure multiplicative recursions |
| MIDAS volatility | [Ghysels, Santa-Clara, and Valkanov (2006), Journal of Econometrics](https://doi.org/10.1016/j.jeconom.2005.01.004) | Mixed-frequency lag weighting for volatility forecasts |
| GARCH-MIDAS | [Engle, Ghysels, and Sohn (2013), Review of Economics and Statistics](https://doi.org/10.1162/REST_a_00300) | Short- and long-run volatility decomposition with macro variables |
| Component GARCH-MIDAS | [Conrad and Kleen (2020), Journal of Applied Econometrics](https://doi.org/10.1002/jae.2742) | Monte Carlo properties and horizon-specific forecast evidence |
| Realized Beta GARCH | [Hansen, Lunde, and Voev (2014), Journal of Applied Econometrics](https://doi.org/10.1002/jae.2389) | Dynamic beta from returns and realized variance/covariance |
| DCC/DECO-HEAVY | [Bauwens and Xu (2023), International Journal of Forecasting](https://doi.org/10.1016/j.ijforecast.2022.03.005) | Multivariate variance and correlation forecasts from realized measures |
| Multivariate Realized GARCH | [Archakov, Hansen, and Lunde (2026), Journal of Econometrics](https://doi.org/10.1016/j.jeconom.2025.106040) | Unconstrained log-correlation vector and factor/block realized GARCH |
| Robust covariance HAR | [Li, Drovandi, and Clements (2024), International Journal of Forecasting](https://doi.org/10.1016/j.ijforecast.2023.04.004) | Outlier-robust covariance inputs and least-trimmed MHAR estimation |
| Intraday reconciliation | [Caporin, Di Fonzo, and Girolimetto (2024), Journal of Financial Econometrics](https://doi.org/10.1093/jjfinec/nbae014) | Coherent forecasts of temporal and signed intraday volatility components |
| Rough volatility | [Gatheral, Jaisson, and Rosenbaum (2018), Quantitative Finance](https://doi.org/10.1080/14697688.2017.1393551) | Empirical roughness, RFSV forecasting, and market-microstructure interpretation |
| Roughness critique | [Cont and Das (2024), Sankhya B](https://doi.org/10.1007/s13571-024-00322-2) | Realized-volatility estimation error can mimic latent volatility roughness |
| Linear versus ML forecasts | [Branco, Rubesam, and Zevallos (2024), Journal of Empirical Finance](https://doi.org/10.1016/j.jempfin.2024.101524) | No general nonlinear dominance; strong HAR-family statistical and economic results |

## 10. Assessment

The realized-volatility literature does reveal an obvious gap in scivol. It also suggests a disciplined boundary. The library should first become excellent at deriving, auditing, and modeling daily realized measures. It need not become an exchange-feed handler, limit-order-book simulator, or option-pricing system in the same release.

The best near-term identity is: raw or pre-sampled intraday prices in, statistically defined realized measures out, then fast joint and reduced-form volatility models with exact inference and honest forecast comparisons. That would connect scivol's current GARCH strengths to the most widely used high-frequency workflows and open a credible path to multivariate volatility without sacrificing correctness.
