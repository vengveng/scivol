# Multivariate volatility models: literature survey and scivol roadmap

Survey date: 2026-08-25

## Bottom line

scivol already has a credible multivariate core: closed-form Gaussian `CCC()` and two-step `DCC(p,q)` with Gaussian or common-degrees-of-freedom Student-t copulas, arbitrary fitted univariate marginals, exact C gradients and Hessians, `(1,1)` specializations, sandwich covariance conditional on the standardized residuals, rolling fits, covariance forecasts, simulation, and scenarios. This is stronger computational coverage than many libraries provide.

The next work should deepen this core before adding every named MGARCH model. The highest-value gaps are:

1. **A multivariate validation and comparison layer.** Add CCC constancy tests, whitened-residual and cross-product diagnostics, copula diagnostics, matrix forecast losses, and economic backtests. A fast estimator without model checking is not a complete volatility system.
2. **ADCC/AG-DCC.** One extra scalar asymmetry coefficient captures the empirically important rise in equity correlations after joint negative shocks. It fits the current DCC design and should be the first new correlation process.
3. **Corrected and high-dimensional DCC.** Offer cDCC semantics, configurable correlation targets, linear and nonlinear shrinkage, pairwise composite likelihood, and inference appropriate to composite and two-step estimation. Plain sample targeting becomes ill-conditioned and biased when the number of series is not small relative to the sample.
4. **A realized-covariance input contract and a small realized multivariate family.** Start with precomputed positive-semidefinite daily realized covariance matrices. Implement DCC-HEAVY or DECO-HEAVY, then multivariate Realized GARCH. The literature finds consistent short-horizon gains from intraday information.
5. **A factor/eigenvalue covariance layer.** OGARCH with a positive idiosyncratic covariance term can reuse every scivol univariate component and scale to large portfolios. Add observed-factor and sparse/shrunk residual options before GO-GARCH.
6. **One low-dimensional full-covariance model.** Diagonal and scalar BEKK(1,1), followed by full BEKK(1,1) behind explicit dimension limits, would cover spillover, hedging, and cross-market applications that DCC cannot represent.
7. **One scalable alternative correlation parameterization.** The best candidates are the log-correlation score model or the 2024 dynamic partial-correlation model. Both guarantee positive-definite correlations and relax scalar DCC dynamics. Treat the July 2026 dynamic spectral model as a research prototype until independent evidence accumulates.

Full VECH, unrestricted high-order BEKK, and a general dynamic vine engine are poor near-term investments. They matter in a historical catalogue, but their parameter growth, constraints, or implementation burden offer little practical value relative to BEKK, shrinkage DCC, factors, and realized models.

## What the current implementation does and does not do

The repository states and code confirm the following baseline.

| Capability | Current scivol status | Consequence |
|---|---|---|
| Constant correlation | `CCC()`; closed-form Gaussian correlation fit | Useful baseline, but no formal CCC-versus-dynamic test is exposed |
| Dynamic correlation | Scalar-coefficient `DCC(p,q)` | Parsimonious and positive definite; all pairs share the same news and persistence parameters |
| Dependence distributions | Gaussian and common-df Student-t copulas | Covers symmetric tail dependence, not asymmetric upper/lower-tail dependence |
| Marginals | Univariate specifications fitted separately; fitted marginal CDFs used in scenarios | Good modular design; the multivariate layer is still a two-step estimator |
| Derivatives | Exact C gradients/Hessians and per-observation scores, checked against JAX | Strong base for new recursions and sandwich inference |
| Uncertainty | MLE and sandwich covariance for DCC parameters conditional on supplied standardized residuals | Does not propagate first-stage marginal estimation uncertainty |
| Forecast workflows | Correlation/covariance forecasts, rolling warm starts, scenarios, simulation | Good API foundation for all new correlation processes |
| Large-system protection | Sample `Qbar`, dense likelihood, no shrinkage/composite mode | Weak when `N/T` is large or `Qbar` is ill-conditioned |
| Correlation asymmetry | None | Misses joint-bad-news effects documented in equities and bonds |
| Full covariance dynamics | None | Cannot estimate cross-shock/cross-persistence structures used in BEKK spillover work |
| Realized covariance | None | Cannot use intraday covariance measures or HEAVY/Realized GARCH models |
| Dynamic non-elliptical copulas/vines | None | Cannot model asymmetric tail dependence or pair-specific copula families |

The exact derivative infrastructure reduces the cost of extending DCC, but it does not resolve the statistical issues of correlation targeting, two-step inference, or large-system matrix conditioning.

## Model families

### VECH and BEKK

Let `m = N(N+1)/2`. The unrestricted VECH model regresses `vech(H_t)` on its own lags and `vech(e_{t-1}e_{t-1}')`. It is historically important because it models every variance and covariance directly. It is also a bad general-purpose API: even VECH(1,1) has an intercept of length `m` and two `m x m` coefficient matrices, so parameter count is `m + 2m^2 = O(N^4)`. Ordinary coefficient restrictions do not make positive definiteness transparent.

BEKK writes covariance updates as quadratic forms, for example

```text
H_t = C C' + A' e_(t-1)e_(t-1)' A + B' H_(t-1) B.
```

This construction guarantees positive semidefiniteness. Full BEKK(1,1) still carries roughly `2.5N^2` parameters and a dense Gaussian or Student-t likelihood requires a matrix factorization per observation. Diagonal and scalar BEKK sharply reduce dynamic parameters while retaining a full unconditional covariance component. Full BEKK is therefore a small-system model, not a path to hundreds of assets.

BEKK earns a place in scivol for a different reason from DCC. Its coefficient matrices can represent shock and volatility transmission across series, so researchers use it for volatility spillovers, hedge ratios, futures and spot markets, international transmission, and variance-causality restrictions. The raw entries are not automatically economic spillover measures: users need news-impact functions, impulse responses, and Wald or likelihood-ratio restrictions. Labeling coefficients as direct spillovers would invite bad applied work.

Recommended surface:

```python
BEKK(1, 1, structure="scalar")
BEKK(1, 1, structure="diagonal")
BEKK(1, 1, structure="full", max_dimension=...)
```

Ship `(1,1)` first. Require explicit opt-in above a conservative dimension threshold. Provide Gaussian and multivariate Student-t likelihoods, exact scores/Hessians, covariance targeting as an explicit option, news-impact matrices, volatility impulse responses, cross-dynamic restriction tests, and forecast/simulation parity with DCC. Do not implement unrestricted VECH except perhaps as a development oracle and tiny-dimensional benchmark.

### CCC, DCC, cDCC, and asymmetry

CCC separates univariate conditional variances from a fixed correlation matrix. It remains a useful baseline because calm samples often do not reject constant correlation. Tse's LM test only requires the CCC estimate and was designed to be more robust to nonnormality than the older information-matrix alternative. scivol should make `CCC()` a formal diagnostic gate, not merely another result object.

DCC applies a GARCH-like recursion to a quasi-correlation matrix and normalizes it to a correlation matrix. Scalar DCC has only `p+q` correlation-dynamic parameters, so it avoids the parameter explosion of VECH and BEKK. Dense likelihood evaluation still costs `O(TN^3)` because each `R_t` needs factorization, and retaining every `R_t` costs `O(TN^2)`.

Aielli showed that conventional DCC correlation targeting can make the large-system estimator inconsistent and that the usual interpretation of DCC parameters can mislead. cDCC corrects the recursion. Simulation evidence in later work suggests small practical differences in ordinary low-dimensional financial samples, so cDCC should not silently replace the current public path. It should be a named, independently fitted model or an explicit `correlation_update="corrected"` option. This preserves the library's rule that public estimation paths remain genuinely independent.

ADCC/AG-DCC adds a matrix of negative standardized-residual products to the recursion. The scalar version adds only one dynamic coefficient, while the generalized version permits series-specific news, smoothing, and asymmetry. Cappiello, Engle, and Sheppard found that joint bad news raises correlations, especially for equities. That makes scalar ADCC the clearest immediate model addition. Generalized ADCC is useful later but requires non-scalar constraints and rapidly becomes hard to identify.

The asymmetric intercept must include the unconditional negative-shock covariance, rather than use a naive `1-a-b-g` leftover. Stationarity and persistence also depend on the innovation distribution. In particular, a hard-coded one-half sign probability is wrong for skewed copulas or marginals. The C kernel, transform, forecast recursion, and DGP oracle must share the same convention.

Other useful DCC variants divide into two groups:

- **DECO and block DECO** assume one equicorrelation, or a small number of within- and between-block correlations, at each date. The restriction permits analytic determinant/inverse formulas and very large systems. It is useful for sector panels and portfolio covariance estimation, but a single equicorrelation is too rigid for pair-specific interpretation.
- **Flexible, generalized, and clustered DCC** allow asset-specific or group-specific dynamics. They sit between scalar DCC and full correlation dynamics. A block/group component with user-supplied labels and later data-driven clustering would offer more value than an unconstrained `N x N` parameter matrix.

Two recent variants belong on the benchmark watchlist. Projected DCC replaces diagonal normalization with a nearest-correlation projection. CLIP-DCC, published in 2026, pools part of the cross-sectional innovation while preserving a heterogeneous long-run target; its reported 1,000-stock rolling tests improve global-minimum-variance risk over DCC and DCC with nonlinear shrinkage. These should remain named models. Neither is a numerical patch to ordinary DCC.

### High-dimensional DCC, shrinkage, and factors

High-dimensional DCC has two separate problems. First, the sample correlation target becomes noisy or singular as `N/T` grows. Second, evaluating a full Gaussian or Student-t likelihood requires repeated dense factorizations.

The literature has practical answers:

- shrink the correlation target toward identity, constant correlation, or a factor target;
- use analytic linear shrinkage as the simple default and nonlinear eigenvalue shrinkage as the high-accuracy option;
- fit dynamics with pairwise or block composite likelihood to avoid full-dimensional inversions;
- report Godambe/sandwich covariance for composite estimators;
- decompose covariance into low-rank factors plus diagonal, sparse, or block residual covariance.

Engle, Ledoit, and Wolf combine composite likelihood with nonlinear shrinkage and report handling at least 1,000 assets. Pakel, Shephard, Sheppard, and Engle formalize composite estimation for vast-dimensional DCC and BEKK models. These methods are a more natural continuation of scivol's current DCC than adding old, unrestricted MGARCH forms.

Factor models trade covariance flexibility for scale. Factor-GARCH specifies a small number of common shocks with GARCH dynamics plus idiosyncratic variance. Orthogonal GARCH applies PCA, fits univariate GARCH models to retained components, and rotates covariance forecasts back. GO-GARCH replaces orthogonality with a general invertible mixing matrix, often identified with independent component analysis, and can combine non-Gaussian factor densities through convolution.

The tradeoffs matter:

- OGARCH is fast and useful as a benchmark, but fixed principal-component directions cannot follow changing eigenvectors, truncation can discard risk, and signs/order can change across rolling samples.
- GO-GARCH permits richer mixing and independent factor distributions, but identification, initialization, and rotation stability complicate inference.
- observed-factor and latent-factor GARCH models scale better and map naturally to market/sector factors. Their residual covariance must still be regularized.
- shrinkage often matters as much as the dynamic model. Dynamic-volatility data violate the iid assumptions behind naive covariance cleaning, and recent theory proposes time-variation-adjusted nonlinear shrinkage.

A low-rank-plus-sparse decomposition such as POET is preferable to a truncated low-rank matrix alone. It preserves full rank and economically meaningful residual links. Factor count must also be visible and testable: Bai-Ng information criteria, eigenvalue-ratio methods, eigenvalue-cluster methods, or cross-validation are better choices than a hard-coded explained-variance cutoff. Portfolio inverses are often most sensitive to the small eigenvalue directions that an explained-variance rule discards.

Recommended order: add shrinkage targets and composite DCC first; then an observed-factor or PCA-GARCH covariance process with diagonal, sparse, and shrinkage-residual options. OGARCH is worth shipping as a transparent baseline, provided it never returns a singular covariance silently. GO-GARCH is lower priority unless scivol also commits to ICA diagnostics, explicit sign/permutation/scale conventions, and stable rolling alignment. The 2023 lambda-GARCH model, which fixes conditional eigenvectors and drives eigenvalues with scores, is a useful modern bridge between OGARCH and general score-driven covariance models.

### Dynamic copulas and vines

The current Gaussian and Student-t copulas are elliptical. The Student-t copula supplies symmetric tail dependence, but upper and lower tails remain tied. That is restrictive for equity, FX, commodity, and systemic-risk applications where joint crashes and rallies differ.

Dynamic copula models separate marginal volatility from time-varying dependence. A link function or likelihood score can update a copula parameter, while rotations of Clayton, Gumbel, Joe, BB1, or BB7 families distinguish upper and lower tails. A vine combines bivariate pair copulas into a high-dimensional density, allowing family and parameter choices to vary by edge. Regular vines are flexible but bring `N(N-1)/2` pair copulas, structure selection, conditional-distribution recursion, and difficult multi-step simulation. Dynamic parameters on every vine edge compound the cost.

For scivol, the right first step is a reusable copula protocol and a bivariate dynamic-copula module, not a general vine engine:

```python
copula.logpdf(u_t, state_t)
copula.score(u_t, state_t)
copula.update(state_t, u_t)
copula.simulate(state_t, n_draws, rng)
copula.rosenblatt(u, state_path)
```

Add static and score-driven Gaussian, Student-t, and one lower/upper-tail-asymmetric family with rotations. Every family needs `cdf`, `logpdf`, conditional CDF and inverse conditional CDF, simulation, Kendall-tau conversion, and tail-dependence coefficients. Then add a factor copula, which scales closer to `O(NK)` for `K` factors, before a static R-vine. A static R-vine needs sequential estimation, maximum-spanning-tree selection, truncation or thresholding, and a sparsity-aware criterion. Dynamic vines should remain optional until the library has reliable structure selection, simplifying-assumption tests, edge-wise diagnostics, bootstrap uncertainty, and fast conditional-CDF kernels. They are central for density and systemic-tail-risk modeling, but not the best route to fast covariance forecasting.

### Correlation matrices through logs, partial correlations, and spectra

Several recent models avoid DCC's quasi-correlation recursion by parameterizing the correlation matrix directly.

Archakov and Hansen map the `N(N-1)/2` off-diagonal elements of the matrix logarithm to an unconstrained real vector and uniquely recover a positive-definite correlation matrix. This generalized Fisher transform underlies a 2023 score-driven log-correlation model and the 2025 multivariate Realized GARCH model. It gives clean unconstrained optimization but still has `O(N^2)` states unless the dynamics impose factors or shrinkage.

D'Innocenzo and Lucas build a correlation matrix recursively from bivariate dynamic partial correlations. Their 2024 model has stationarity, ergodicity, filter-invertibility, consistency, and asymptotic-normality results based on bivariate slices. It beat Student-t cDCC and multivariate Student-t GAS benchmarks in simulations and a U.S. stock application. Partial correlations also connect naturally to vine structures and sparse conditional-dependence graphs. The result depends on variable ordering unless the model or selection method handles ordering explicitly.

The 2026 dynamic spectral conditional correlation model parameterizes eigenvalues and eigenvectors to span the full correlation elliptope and reports portfolios of up to 1,000 stocks. This is promising and unusually relevant to scivol's speed goal. It is also too new to displace cDCC or shrinkage DCC without independent replications. Implement it first in `localdev/`, reproduce the authors' public code results, and compare matched-point derivatives and portfolio backtests.

These parameterizations suggest a durable architecture: separate the multivariate marginal layer, the correlation state representation, the state update, and the dependence likelihood. `DCC`, `ADCC`, `LogCorrelationGAS`, and `DynamicPartialCorrelation` should share forecast, simulation, diagnostic, and result protocols without sharing hidden optimizers.

### Realized covariance and matrix-valued models

Daily returns provide one rank-one covariance observation per day. Intraday data can estimate the day's integrated covariance much more precisely, but microstructure noise and nonsynchronous trading make naive realized covariance unreliable. Multivariate realized kernels were designed to be positive semidefinite, consistent under common forms of noise, and tolerant of nonsynchronous trading.

Models then use realized covariance in three ways:

1. **HEAVY-type models** feed lagged realized covariance into the conditional covariance of daily returns and separately model realized covariance. The original multivariate HEAVY paper found gains over MGARCH, especially at short horizons. DCC-HEAVY and DECO-HEAVY use realized variances and correlations and outperformed daily-data BEKK, DCC, and DECO benchmarks in the reported application.
2. **Transformation models** model Cholesky factors, matrix logarithms, or generalized Fisher coordinates with VAR, HAR, or factor dynamics. They guarantee positive-definite forecasts after inverse transformation, but Cholesky models depend on series order and log/exponential transformations can introduce forecast bias.
3. **Matrix-distribution models** use Wishart/CAW, matrix-F, Riesz, or F-Riesz conditional densities. CAW is a natural matrix analogue of GARCH. Matrix-F and F-Riesz variants handle the heavy tails and heterogeneous measurement errors that a single-degree-of-freedom Wishart law cannot. They offer full density forecasts but require special-function derivatives, careful degrees-of-freedom constraints, and `O(N^2)` parameter control.

Archakov, Hansen, and Lunde's multivariate Realized GARCH, published online in 2025 for the 2026 Journal of Econometrics volume, is the most direct recent omission. It uses the generalized Fisher parameterization and incorporates realized correlation measures without extra covariance restrictions. A practical scivol sequence is:

1. accept a `(T,N,N)` realized-covariance cube, dates, sampling metadata, and optional measurement-error weights;
2. validate symmetry, eigenvalues, missingness, and return/realized-date alignment;
3. ship scalar DCC-HEAVY/DECO-HEAVY with asymmetry;
4. add log-HAR or generalized-Fisher HAR as a transparent forecast benchmark;
5. implement multivariate Realized GARCH;
6. consider scalar/diagonal CAW and matrix-F only after the return-plus-realized workflows and evaluation suite are stable.

Do not make raw-tick cleaning part of the first model release. Accept precomputed realized measures first. A separate estimator layer can later add refresh-time sampling, synchronization, multivariate realized kernels, subsampling, jump-robust measures, and PSD repair.

The input contract must record how each realized matrix was built. At minimum, preserve sampling frequency, synchronization rule, kernel or estimator, effective intraday observation count, market-hours convention, and any noise or jump correction. Wishart, matrix-F, Cholesky, and matrix-log likelihoods need strictly positive-definite inputs. A realized covariance formed from fewer synchronized observations than assets is rank deficient by construction. scivol should reject it, apply a named regularizer, or switch to an explicitly rank-tolerant loss. It should never add invisible jitter. Realized quarticity or an equivalent precision estimate can downweight especially noisy days.

Matrix-model residuals also differ from ordinary return residuals. For fitted conditional mean matrix `V_t` and observed realized covariance `X_t`, inspect `E_t = V_t^(-1/2) X_t V_t^(-1/2)`, whose conditional mean should be identity under the basic matrix models. Useful checks include lagged Frobenius inner products, trace, log determinant, eigenvalues, and condition number. Forecast comparisons should add affine-invariant or log-Euclidean distances to QLIKE and Frobenius loss.

## Diagnostics, tests, and evaluation

These features are more important than several additional model acronyms.

### In-sample diagnostics

- **CCC gate:** Tse's LM test for constant conditional correlation, with bootstrap calibration for smaller samples or heavy tails. The paper's Monte Carlo evidence supports the asymptotic approximation mainly around samples of 1,000 or more.
- **Whitened residuals:** compute `z_t = L_t^{-1} e_t`, where `L_t L_t' = H_t`. Test autocorrelation in `z_t`, `z_t^2`, and `vech(z_t z_t' - I)`. Use tests whose reference distribution accounts for estimated parameters. A naive Ljung-Box chi-square reference on cross-products can be wrong.
- **Modern residual test:** implement the 2021 transformed-quadratic-residual test, which explicitly accounts for estimation uncertainty, alongside transparent graphical ACF diagnostics.
- **Distribution and copula checks:** marginal PIT uniformity, serial independence of PIT moments, Rosenblatt-transform uniformity/independence for copulas, multivariate tail exceedance counts, and energy-distance or characteristic-function checks for simulated innovations.
- **Vine-specific checks:** test the simplifying assumption, compare empirical and fitted lower/upper joint-exceedance curves, and bootstrap the whole margin-plus-vine procedure. An energy score alone can be insensitive to dependence errors; pair it with variogram and tail-weighted scores.
- **Dynamics:** LR/Wald/score tests for `a=0`, ADCC asymmetry `g=0`, group-equality restrictions, and BEKK cross-dynamic restrictions. Report persistence distance from the stationarity boundary.
- **Numerical validity:** minimum eigenvalue and condition number of every fitted and forecast covariance/correlation matrix, Cholesky failures or jitter, target conditioning, effective rank, optimizer path, gradient norm, and boundary-active parameters.
- **Breaks:** rolling parameter/correlation plots and formal change diagnostics. Crisis-driven breaks can look like extreme persistence or asymmetric news response.

### Out-of-sample evaluation

No single loss answers every use case. scivol should evaluate:

- matrix Gaussian QLIKE or log score;
- Frobenius and elementwise variance/covariance losses against a realized-covariance proxy;
- correlation-only loss after separating marginal variance forecasts;
- realized variance of global-minimum-variance and constrained portfolios;
- hedge-ratio loss for spot-futures or cross-asset hedges;
- VaR/ES and joint-tail backtests for simulated portfolio returns;
- turnover, leverage, short constraints, and transaction costs;
- Diebold-Mariano or Giacomini-White comparisons and the existing model-confidence-set workflow extended to matrix losses.

The 2012 comparison of 125 MGARCH forecasts found that model rankings changed across calm and crisis periods. DCC with leverage in marginal variances remained difficult to beat. This argues for regime-stratified backtests and simple benchmarks, not selection by in-sample likelihood alone.

### Inference

For current two-step DCC, distinguish three covariance estimates in the API:

1. conditional-on-fitted-marginals Hessian covariance;
2. conditional sandwich covariance using correlation scores;
3. full two-step covariance that propagates marginal estimation, or a block/time-series bootstrap when the analytic correction is unavailable.

Composite likelihood needs Godambe covariance, not inverse Hessian covariance. Covariance targeting also changes standard errors because the target is estimated. These labels should appear in `fit_info` and summaries.

## Computational comparison

The table gives dominant orders for dense implementations. Constants and special structure matter.

| Model | Dynamic parameter scale | Typical per-time likelihood cost | Main limitation |
|---|---:|---:|---|
| Full VECH(1,1) | `O(N^4)` | at least `O(N^3)` | parameter explosion; positivity constraints |
| Full BEKK(1,1) | `O(N^2)` | `O(N^3)` | nonconvex estimation; only small systems |
| Diagonal/scalar BEKK | `O(N)` / `O(1)` dynamics plus covariance target | `O(N^3)` | full likelihood still dense |
| Scalar DCC | `O(1)` dynamics plus `O(N^2)` target | `O(N^3)` | homogeneous dynamics; noisy target |
| Pairwise-composite DCC | `O(1)` dynamics plus target | `O(N^2)` for all pairs; lower if sampled/blocked | efficiency loss; Godambe SE required |
| DECO | `O(1)` | `O(N)` to `O(N^2)` | equicorrelation restriction |
| Factor/OGARCH with `K` factors | roughly `O(NK + K^2)` | `O(NK^2 + K^3)` | fixed/estimated factor structure |
| Static vine | `O(N^2)` pair copulas | `O(N^2)` density | selection and conditional-CDF recursion |
| Fully dynamic vine | `O(N^2)` states and parameters | `O(TN^2)` plus optimization | high implementation and diagnostic burden |
| CAW/full matrix realized model | commonly `O(N^2)` | `O(N^3)` | matrix density and high-dimensional estimation |

For all families, storing `T` dense covariance paths costs `O(TN^2)`. Add iterator/chunked output, pair extraction without materializing the full path, and optional disk-backed arrays before advertising thousand-asset support.

## Proposed public architecture

Keep model semantics visible and preserve independent public paths.

```python
# Low-dimensional full covariance
BEKK(1, 1, structure="diagonal", distribution="studentt").fit(returns)

# Correlation processes with fitted marginals
ADCC(1, 1, distribution="studentt").fit(returns, univariate_spec=...)
CDCC(1, 1, target="nonlinear_shrinkage").fit(...)
DCC(1, 1, estimator="pairwise_composite").fit(...)
BlockDECO(groups=sector_labels).fit(...)

# Factors
FactorGARCH(factors=market_factors, residual="shrinkage").fit(returns)
OGARCH(n_factors="cv", residual="poet").fit(returns)

# Precomputed realized covariance
DCCHEAVY(1, 1).fit(returns, realized_cov=rcov)
MultivariateRealizedGARCH(...).fit(returns, realized_cov=rcov)
```

Each result should expose `conditional_covariance`, `conditional_correlation`, `forecast`, `simulate`, `scenarios`, `whitened_residuals`, `diagnostics`, `fit_info`, and a common matrix-loss interface. Store representation-specific states such as `Q_t`, factor covariance, partial correlations, or log-correlation coordinates without pretending they are interchangeable.

## Priority classification

### Must-have for the stated goal

| Priority | Feature | Why |
|---|---|---|
| P0 | Multivariate diagnostics and covariance/correlation forecast evaluation | Correctness claim requires specification checks and fair model comparison |
| P0 | cDCC as an independent path; fixed/sample/linear-shrinkage/nonlinear-shrinkage targets | Addresses known targeting inconsistency and large-`N` conditioning |
| P0 | Full two-step or bootstrap inference; Godambe SE for composite modes | Current conditional sandwich does not cover all generated-residual uncertainty |
| P1 | Scalar ADCC with exact value/gradient/Hessian and AD oracle | High empirical value for little extra parameter cost |
| P1 | Pairwise-composite DCC and block/pair subsampling | Removes dense likelihood bottleneck for large systems |
| P1 | Scalar and diagonal BEKK(1,1); full BEKK(1,1) for small `N` | Covers full covariance transmission and canonical MGARCH applications |
| P1 | Realized-covariance data contract plus DCC/DECO-HEAVY | Largest information-set omission; strong short-horizon forecast evidence |
| P1 | Factor-GARCH or OGARCH plus positive sparse/shrunk residual covariance | Practical path to large portfolios and a required benchmark |
| P2 | Generalized-Fisher/log-correlation state utilities | Reusable positive-definite foundation for score and realized models |
| P2 | Static and bivariate dynamic asymmetric copulas | Adds asymmetric tail dependence without committing to a whole vine framework |

### Valuable after the core is complete

- block DECO and user-defined group DCC;
- log-correlation GAS and dynamic partial correlations;
- multivariate Realized GARCH;
- scalar/diagonal CAW and matrix-F realized-covariance density models;
- static R-vines with selection and complete diagnostics;
- BEKK news-impact surfaces, impulse responses, and variance-causality tests;
- robust bounded-innovation DCC for isolated jumps and contamination;
- regime-switching correlations when break diagnostics justify them.

### Low-value completeness or research-only for now

- unrestricted VECH;
- arbitrary-order full BEKK beyond tiny systems;
- fully generalized `N x N` DCC dynamic coefficient matrices;
- a fully dynamic R-vine with an independent state for every edge;
- GO-GARCH before stable ICA, factor-alignment, and rolling-identification diagnostics exist;
- every matrix-variate density before one realized-covariance workflow is polished;
- neural covariance models without hard PSD guarantees, reproducible probabilistic scoring, and strong econometric benchmarks;
- the July 2026 DSCC model as a production default before replication.

## Suggested benchmark matrix

Every new multivariate model should enter one benchmark harness with controlled DGPs and real data.

| Dimension regime | Required models | Required outputs |
|---|---|---|
| `N=2-5`, long `T` | CCC, DCC, cDCC, ADCC, scalar/diagonal/full BEKK | recovery, AD derivatives, Hessian/sandwich/bootstrapped SE, LR/Wald tests |
| `N=20-100` | DCC, shrinkage DCC, composite DCC, block DECO, factor-GARCH | runtime, peak memory, conditioning, forecast loss, portfolio risk |
| `N=250-1000` | nonlinear-shrinkage composite DCC, DECO/block DECO, factors | streaming feasibility, no dense-path default, economic backtests |
| realized covariance | DCC-HEAVY, DECO-HEAVY, log-HAR, multivariate Realized GARCH | matrix QLIKE, density score, horizon curves, PSD and bias checks |
| asymmetric tails | t-DCC, ADCC, asymmetric dynamic copula, static vine | joint-tail calibration, portfolio VaR/ES, Rosenblatt diagnostics |

Use calm, crisis, and structural-break periods. Report both statistical and economic losses, since a better likelihood need not yield better constrained portfolio decisions.

## Source table

The table prioritizes primary papers and publisher or institutional copies. DOI links are stable even when full text requires access.

| Year | Source | What it establishes | Relevance to scivol |
|---:|---|---|---|
| 1988 | Bollerslev, Engle, and Wooldridge, [A Capital Asset Pricing Model with Time-Varying Covariances](https://doi.org/10.1086/261527) | VECH-style MGARCH applied to bills, bonds, and equities; time-varying covariance affects risk premia | Historical full-covariance baseline and asset-pricing application |
| 1990 | Engle, Ng, and Rothschild, [Asset Pricing with a Factor-ARCH Covariance Structure](https://doi.org/10.1016/0304-4076(90)90099-F) | Low-dimensional factor structure for conditional covariance | Foundation for factor-GARCH |
| 1990 | Bollerslev, [Modelling the Coherence in Short-Run Nominal Exchange Rates](https://doi.org/10.2307/2109358) | Constant conditional correlation MGARCH | Current `CCC()` foundation |
| 1993 | Kroner and Sultan, [Time-Varying Distributions and Dynamic Hedging with Foreign Currency Futures](https://doi.org/10.2307/2331164) | MGARCH hedge ratios and out-of-sample risk reduction | Canonical hedging workflow |
| 1995 | Engle and Kroner, [Multivariate Simultaneous Generalized ARCH](https://doi.org/10.1017/S0266466600009063) | BEKK representation and theory | Canonical small-system full-covariance model |
| 2000 | Tse, [A Test for Constant Correlations in a Multivariate GARCH Model](https://doi.org/10.1016/S0304-4076(99)00080-9) | Convenient LM test for CCC; studies size, power, and nonnormality | Immediate diagnostic for `CCC()` |
| 2002 | Engle, [Dynamic Conditional Correlation](https://doi.org/10.1198/073500102288618487) | Two-step scalar DCC | Current scivol baseline |
| 2002 | Engle and Sheppard, [Theoretical and Empirical Properties of DCC](https://doi.org/10.3386/w8554) | Theory, large-system motivation, and up-to-100-asset examples | Inference and specification reference |
| 2002 | Tse and Tsui, [A Multivariate GARCH Model with Time-Varying Correlations](https://ideas.repec.org/a/bes/jnlbes/v20y2002i3p351-62.html) | Alternative positive-definite time-varying correlation recursion | Independent benchmark for DCC semantics |
| 2002 | van der Weide, [GO-GARCH: A Multivariate Generalized Orthogonal GARCH Model](https://doi.org/10.1002/jae.688) | Generalized orthogonal factor mixing | GO-GARCH foundation |
| 2002 | Alexander, [Principal Component Models for Generating Large GARCH Covariance Matrices](https://doi.org/10.1111/1468-0300.00089) | Practical OGARCH construction for large covariance matrices | Fast factor baseline and source of PCA caveats |
| 2006 | Bauwens, Laurent, and Rombouts, [Multivariate GARCH Models: A Survey](https://doi.org/10.1002/jae.842) | Model, estimation, application, and diagnostic survey | Broad foundational map |
| 2006 | Cappiello, Engle, and Sheppard, [Asymmetric Dynamics in the Correlations of Global Equity and Bond Returns](https://doi.org/10.1093/jjfinec/nbl005) | AG-DCC and empirical correlation asymmetry | Basis for scalar ADCC priority |
| 2011 | Barndorff-Nielsen, Hansen, Lunde, and Shephard, [Multivariate Realised Kernels](https://doi.org/10.1016/j.jeconom.2010.07.009) | PSD realized covariance under noise and nonsynchronous trading | Future realized-measure estimator layer |
| 2011 | Chiriac and Voev, [Modelling and Forecasting Multivariate Realized Volatility](https://doi.org/10.1002/jae.1152) | Cholesky/long-memory realized covariance forecasts | Transformation benchmark and ordering caveat |
| 2011 | Bauer and Vorkink, [Forecasting Multivariate Realized Stock Market Volatility](https://doi.org/10.1016/j.jeconom.2010.03.021) | Matrix-log latent-factor model with PSD forecasts | Log-covariance/factor benchmark |
| 2012 | Golosnoy, Gribisch, and Liesenfeld, [The Conditional Autoregressive Wishart Model](https://doi.org/10.1016/j.jeconom.2011.11.004) | CAW density and autoregressive matrix mean | Foundational realized matrix model |
| 2012 | Noureldin, Shephard, and Sheppard, [Multivariate HEAVY Models](https://doi.org/10.1002/jae.1260) | Uses realized covariance; short-horizon gains over MGARCH | High-priority realized extension |
| 2012 | Engle and Kelly, [Dynamic Equicorrelation](https://doi.org/10.1080/07350015.2011.652048) | DECO and block DECO; consistency under misspecified equicorrelation; portfolio results | Large-system structured correlation |
| 2012 | Laurent, Rombouts, and Violante, [On the Forecasting Accuracy of Multivariate GARCH Models](https://ideas.repec.org/a/wly/japmet/v27y2012i6p934-955.html) | 125-model forecast comparison; rankings change by regime; leverage-DCC strong | Benchmark and selection design |
| 2012 | Hafner and Reznikova, [On the Estimation of Dynamic Conditional Correlation Models](https://doi.org/10.1016/j.csda.2010.09.022) | High-dimensional bias and shrinkage targeting | Target options and conditioning checks |
| 2011 | Boswijk and van der Weide, [Method of Moments Estimation of GO-GARCH Models](https://doi.org/10.1016/j.jeconom.2010.11.011) | Joint-diagonalization estimator avoids fragile full conditional ML | Independent GO-GARCH estimation path |
| 2013 | Aielli, [Dynamic Conditional Correlation: On Properties and Estimation](https://doi.org/10.1080/07350015.2013.771027) | DCC targeting inconsistency and cDCC correction | Correctness-critical independent path |
| 2013 | Boudt, Danielsson, and Laurent, [Robust Forecasting of DCC-GARCH Models](https://doi.org/10.1016/j.ijforecast.2012.06.003) | Bounded innovation propagation and robust targeting/estimation | Later robust DCC option |
| 2013 | Fan, Liao, and Mincheva, [Large Covariance Estimation by Thresholding Principal Orthogonal Complements](https://doi.org/10.1111/rssb.12016) | POET low-rank-plus-sparse covariance estimator | Full-rank idiosyncratic covariance for factor models |
| 2014 | Hansen, Lunde, and Voev, [Realized Beta GARCH](https://doi.org/10.1002/jae.2389) | Uses realized variance/covariance to model time-varying beta | Realized beta and hedging workflow |
| 2019 | Engle, Ledoit, and Wolf, [Large Dynamic Covariance Matrices](https://doi.org/10.1080/07350015.2017.1345683) | Composite likelihood plus nonlinear shrinkage; systems of at least 1,000 assets | Primary large-`N` design |
| 2018 | Opschoor, Janus, Lucas, and van Dijk, [New HEAVY Models for Fat-Tailed Realized Covariances and Returns](https://doi.org/10.1080/07350015.2016.1245622) | Matrix-F realized measurement and Student-t returns with score robustness | Heavy-tailed realized-density option |
| 2019 | Gorgi and coauthors, [Realized Wishart-GARCH](https://doi.org/10.1093/jjfinec/nby007) | Joint model for daily returns and realized covariance | Alternative joint return-matrix density |
| 2020 | Barthel, Czado, and Okhrin, [A Partial Correlation Vine Based Approach](https://doi.org/10.1016/j.csda.2019.106810) | Partial-correlation vine for multivariate volatility forecasting | Bridge between vines and scalable correlation states |
| 2020 | Pakel, Shephard, Sheppard, and Engle, [Fitting Vast Dimensional Time-Varying Covariance Models](https://doi.org/10.1080/07350015.2020.1713795) | Composite likelihood for vast-dimensional DCC/BEKK | Estimator and Godambe-inference blueprint |
| 2020 | Ledoit and Wolf, [Analytical Nonlinear Shrinkage of Large-Dimensional Covariance Matrices](https://doi.org/10.1214/19-AOS1921) | Fast nonlinear eigenvalue shrinkage, including `N>T` | Production-quality high-dimensional target utility |
| 2021 | Archakov and Hansen, [A New Parametrization of Correlation Matrices](https://doi.org/10.3982/ECTA16910) | Unconstrained generalized Fisher/log-correlation coordinates with guaranteed PD | Reusable transform layer |
| 2021 | Ke, [A Residual-Based Test for Multivariate GARCH Models Using Transformed Quadratic Residuals](https://doi.org/10.1016/j.econlet.2021.109978) | Diagnostic test accounting for estimated-parameter uncertainty | Modern multivariate residual diagnostic |
| 2006 | Patton, [Modelling Asymmetric Exchange Rate Dependence](https://doi.org/10.1111/j.1468-2354.2006.00387.x) | Conditional copulas with asymmetric dependence dynamics | Foundation for non-elliptical dynamic copulas |
| 2013 | Creal, Koopman, and Lucas, [Generalized Autoregressive Score Models](https://doi.org/10.1002/jae.1279) | Likelihood-score observation-driven state updates | Common engine for dynamic copulas and correlations |
| 2017 | Oh and Patton, [Modeling Dependence in High Dimensions with Factor Copulas](https://doi.org/10.1080/07350015.2015.1062384) | Scalable high-dimensional non-elliptical dependence | Factor copula should precede dynamic vines |
| 2017 | Brownlees and Engle, [SRISK: A Conditional Capital Shortfall Measure of Systemic Risk](https://doi.org/10.1093/rfs/hhw060) | GJR-GARCH/DCC simulation in systemic-risk measurement | Concrete DCC scenario and tail-risk application |
| 2016 | Almeida, Czado, and Manner, [Modeling High-Dimensional Time-Varying Dependence Using Dynamic D-Vines](https://doi.org/10.1002/asmb.2182) | Dynamic pair-copula construction | Reference for a later dynamic-vine module |
| 2019 | Nagler, Bumann, and Czado, [Model Selection in Sparse High-Dimensional Vine Copula Models](https://doi.org/10.1016/j.jmva.2019.03.004) | Sparse vine selection | Truncation and sparsity are mandatory at scale |
| 2022 | Zhou, Jiang, Zhu, and Li, [Time Series Models for Realized Covariance Matrices Based on the Matrix-F Distribution](https://doi.org/10.5705/ss.202019.0424) | Heavy-tailed matrix-F dynamics for realized covariance | Alternative to restrictive Wishart tails |
| 2020 | Bollerslev and coauthors, [Realized Semicovariances](https://doi.org/10.1016/j.jeconom.2019.12.011) | Signed realized covariance measures for multivariate leverage | Asymmetric realized-covariance input and model |
| 2021 | Vassallo and coauthors, [A DCC-Type Approach for Realized Covariance Modeling with Score-Driven Dynamics](https://doi.org/10.1016/j.ijforecast.2020.07.006) | Wishart/matrix-F score updates in a two-step correlation framework | Matrix-density competitor to DCC-HEAVY |
| 2023 | Bauwens and Xu, [DCC- and DECO-HEAVY](https://doi.org/10.1016/j.ijforecast.2022.03.005) | Realized variance/correlation DCC and DECO, including asymmetry; forecast gains | Best first realized multivariate family |
| 2023 | Bauwens and Otranto, [Hadamard-Exponential CAW](https://doi.org/10.1093/jjfinec/nbac007) | Pair-specific covariance impacts with controlled parameter growth | Flexible CAW extension after the scalar baseline |
| 2023 | Hafner and Wang, [A Dynamic Conditional Score Model for the Log Correlation Matrix](https://doi.org/10.1016/j.jeconom.2021.09.004) | Score dynamics in log-correlation coordinates with Student-t copula | Scalable alternative correlation process |
| 2023 | Llorens-Terrazas and Brownlees, [Projected Dynamic Conditional Correlations](https://doi.org/10.1016/j.ijforecast.2022.06.003) | Projects covariance state onto the correlation set rather than diagonally rescaling | Named DCC challenger for large systems |
| 2023 | Hetland, Pedersen, and Rahbek, [Dynamic Conditional Eigenvalue GARCH](https://doi.org/10.1016/j.jeconom.2021.09.003) | Score-driven conditional eigenvalues with fixed eigenvectors and reduced-rank tests | Modern eigenvalue-volatility model |
| 2023 | Vogler and Golosnoy, [Unrestricted Maximum Likelihood Estimation of Multivariate Realized Volatility Models](https://doi.org/10.1016/j.ejor.2022.05.029) | Analytic CAW gradient makes unrestricted estimation feasible around 30 assets | Evidence for analytic derivative investment |
| 2023 | Gribisch and Hartkopf, [Modeling Realized Covariance Measures with Heterogeneous Liquidity](https://doi.org/10.1016/j.jeconom.2022.01.007) | Generalized Wishart/Riesz state space for heterogeneous measurement error | Liquidity-aware realized input metadata |
| 2024 | D'Innocenzo and Lucas, [Dynamic Partial Correlation Models](https://doi.org/10.1016/j.jeconom.2024.105747) | Scalable recursion, full asymptotics, strong U.S. equity results | High-value post-DCC candidate |
| 2024 | Ding and Zheng, [High-Dimensional Covariance Matrices Under Dynamic Volatility Models](https://doi.org/10.1214/24-AOS2381) | Time-variation-adjusted spectral theory and nonlinear shrinkage | Warns against iid shrinkage assumptions |
| 2025 | Opschoor, Lucas, and Rossini, [Conditional Autoregressive F-Riesz Model](https://doi.org/10.1093/jjfinec/nbae023) | Flexible heavy-tail and heterogeneous matrix measurement errors | Later matrix-density extension |
| 2025 | Ruan, Zhang, and Liu, [A Semiparametric Factor-GARCH Model for High-Dimensional Covariance Estimation](https://doi.org/10.1007/s42952-025-00324-4) | Latent factors/loadings with high-dimensional asymptotics | Recent factor route; replication needed |
| 2025 | Archakov, Hansen, and Lunde, [A Multivariate Realized GARCH Model](https://doi.org/10.1016/j.jeconom.2025.106040) | Realized variances and correlations through generalized Fisher coordinates | Major recent model omission |
| 2026 | van Os and van Dijk, [CLIP-DCC](https://doi.org/10.1080/07350015.2025.2526431) | Pooled correlation innovation with heterogeneous long-run target; tested to 1,000 stocks | Recent bridge between DCC and DECO |
| 2026 | Abadir and Rockinger, [Dynamic Spectral Conditional Correlations](https://doi.org/10.1093/ectj/utag017) | Full elliptope coverage and portfolios up to 1,000 stocks | Research prototype and benchmark candidate |
| 2026 | Otranto, [A Matrix-Variate Log-Normal Model for Covariance Matrices](https://doi.org/10.1016/j.spl.2026.110782) | Matrix-log-normal likelihood with retransformation correction | Recent transformation-model benchmark |

## Source caveats

- Publication year can differ from online-first year. The table uses the journal year where clear. The multivariate Realized GARCH DOI was assigned in 2025 and appears in a 2026 volume.
- The 2026 DSCC article was published in July 2026 and is new enough that its portfolio claims need independent replication.
- Computational orders above describe dense leading costs, not measured scivol performance. Specialized equicorrelation, low-rank, sparse, and composite paths can change them materially.
- Empirical superiority statements refer to the cited paper's datasets and losses. They are reasons to benchmark, not universal rankings.
