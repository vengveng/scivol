# Recent machine-learning and rough-volatility developments, 2020-2026

Research cutoff: 25 August 2026.

## Bottom line

Recent work does not justify replacing scivol's likelihood-based models with a deep-learning stack. It does justify making scivol a better experimental base for comparing them. The strongest recurring findings are these:

1. Roughness is a live empirical hypothesis, not a settled fact. Noise-aware GMM and semiparametric estimators still find very small Hurst exponents, while other studies show that realized-volatility measurement error can create apparent roughness. A 2025 SPX option study also finds that parsimonious Markovian or non-rough path-dependent models can fit smiles better than rough models.
2. Neural gains often come from the training design rather than the architecture. Pooling many assets, adding market-wide volatility, using intraday observations, and giving the network a GARCH or HAR starting structure matter more consistently than adding layers.
3. Hybrid models are the most credible neural extension for scivol. RECH, GARCH-LSTM, HARNet, and Deep Realized GARCH retain a known econometric recursion and ask a network to learn a correction. They offer nesting, interpretability, and a meaningful classical baseline.
4. Transformers have positive case studies, but no stable dominance result. Recent large comparisons find that well-fitted linear HAR models remain competitive, and a July 2026 foundation-model comparison finds only narrow and uneven gains over Log-HAR.
5. Distribution and quantile forecasts are more valuable than another point-forecast architecture. A library that claims correctness should expose predictive densities or samples, proper scores, calibration diagnostics, and tail-risk backtests.

The immediate opportunity is therefore an architecture-neutral forecasting and evaluation layer, followed by an optional hybrid-model interface. A full rough-Bergomi or rough-Heston pricing stack belongs later, and only if scivol deliberately expands into derivative pricing.

## What changed in rough volatility

The rough-volatility claim began from the observation that log volatility behaves locally like fractional Brownian motion with a small Hurst exponent, often near 0.1. The practical distinction is important. Roughness concerns local path regularity. Long memory concerns slow autocorrelation decay. A process may have either property or both. Bennedsen, Lunde, and Pakkanen build Brownian semistationary models that decouple the two and report forecast gains when both are modeled ([Bennedsen et al., 2022](https://doi.org/10.1093/jjfinec/nbaa049)). A future scivol component should therefore avoid treating `H < 0.5` as a synonym for long memory.

Inference has improved, but it has also exposed fragility. Bolko et al. derive a measurement-error correction for GMM estimation from realized measures and estimate H near 0.05 for equity indices ([Bolko et al., 2023](https://doi.org/10.1016/j.jeconom.2022.06.009)). Chong et al. construct a semiparametric estimator with a central limit theorem directly from high-frequency prices ([Chong et al., 2024](https://doi.org/10.1214/23-AAP2002)). Damian and Frey use finite-dimensional OU approximations and nested particle filters; their simulation and single-stock application suggest filtering can detect roughness while reducing sensitivity to microstructure noise ([Damian and Frey, 2024](https://doi.org/10.1080/14697688.2024.2399284)).

The opposing evidence is serious. Cont and Das show by simulation that realized volatility can look rough even when latent spot volatility is a Brownian diffusion. Their estimated H for the proxy need not estimate the H of the latent process ([Cont and Das, 2022](https://arxiv.org/abs/2203.13820)). The disagreement is not resolved by a naive log-log regression on a realized-volatility series.

Recent work also weakens the claim that roughness is required for option smiles. Abi Jaber and Li compare Volterra-Bergomi models with Markovian and non-rough path-dependent alternatives on SPX options from 2011 through 2022. Rough models fit the global maturity shape poorly and fail to beat parsimonious competitors consistently ([Abi Jaber and Li, 2025](https://doi.org/10.1111/mafi.12463)). This does not make rough models useless. It means scivol should expose model comparison rather than declare them structurally superior.

The computational direction is clearer than the empirical verdict. Bayer and Breneis approximate fractional-kernel stochastic Volterra equations by finite-dimensional diffusions and prove rapid convergence in the number of factors ([Bayer and Breneis, 2023](https://doi.org/10.1080/14697688.2022.2139193)). This turns a non-Markovian history dependence into weighted OU states. It is the right implementation strategy for fast simulation, filtering, or pricing. Neural surrogates can then accelerate the repeated pricing map rather than replace the stochastic model; Horvath, Muguruza, and Tomas demonstrate millisecond-scale surface calibration after offline training ([Horvath et al., 2021](https://doi.org/10.1080/14697688.2020.1817974)).

### Implication for scivol

Rough volatility should enter in two stages.

- First add research-grade roughness diagnostics: scale-dependent H estimates, confidence intervals, sensitivity to sampling interval, and a measurement-error warning. The result must identify whether its input is a latent-volatility estimate, realized variance, or raw high-frequency prices.
- If empirical demand follows, add a physical-measure Brownian semistationary or fractional log-volatility component with explicit short-memory and long-memory parameters.
- Add rough Bergomi or rough Heston only with a derivative-pricing module. Their objects, calibration data, loss functions, and simulation requirements differ from scivol's current return-likelihood workflow.
- Use exponential-sum or OU-factor approximations in production. Keep direct fractional convolution as a slow oracle for validation.

## Neural, hybrid, and global models

The best recent neural models impose volatility structure rather than learn it again from scratch. Nguyen, Tran, and Kohn's RECH family adds a deterministic recurrent state to GARCH-type conditional variance equations and preserves familiar GARCH, GJR, or EGARCH effects ([Nguyen et al., 2022](https://doi.org/10.1002/jae.2902)). Zhao et al. express GARCH recursions as neural kernels and insert them into an LSTM cell ([Zhao et al., 2024](https://doi.org/10.1609/aaai.v38i15.29643)). HARNet starts from an exact HAR forecast, then trains a dilated convolutional extension; QLIKE training and HAR initialization stabilize optimization ([Reisenhofer et al., 2022](https://arxiv.org/abs/2205.07719)). These constructions make the network's incremental value measurable because setting the correction to zero recovers a known model.

The same lesson appears in data organization. Christensen, Siggaard, and Veliyev find that several machine-learning methods beat HAR variants for Dow 30 realized variance, especially at longer horizons and when extra predictors add information ([Christensen et al., 2023](https://doi.org/10.1093/jjfinec/nbac020)). Zhang et al. pool intraday data across 100 liquid US stocks, add market realized volatility, and obtain gains that transfer to stocks excluded from training ([Zhang et al., 2024](https://doi.org/10.1093/jjfinec/nbad005)). A new 2026 study covering more than 10,000 stocks reports that sample-size scaling and global estimation matter more than model-size scaling or architectural choice ([Liu et al., 2026](https://doi.org/10.1093/jjfinec/nbag019)).

DeepVol provides a related high-frequency result. Dilated causal convolutions consume intraday returns directly, share parameters across assets, and outperform daily realized-measure baselines in the authors' NASDAQ-100 exercise ([Moreno-Pino and Zohren, 2024](https://doi.org/10.1080/14697688.2024.2387222)). The useful feature is not merely deep learning. It is a global causal map over the intraday path, trained on far more observations than a daily single-asset model sees.

These findings point to two additions that are more useful than shipping a fixed LSTM class. First, scivol needs pooled estimation over panels with asset identifiers, market factors, masks, and leakage-safe time splits. Second, it needs a generic correction interface around an econometric base model.

A conservative hybrid contract could use one of two forms:

```text
log h_t = log h_t(base; theta) + g_phi(x_{1:t-1})

h_t = h_t(base; theta) * exp(w * tanh(g_phi(x_{1:t-1})))
```

The first form is simpler and guarantees positivity after exponentiation. The second has an explicit gate, so `w = 0` recovers the base model. The hybrid must expose the base forecast, correction, final forecast, and correction-off nested model. Tests should confirm that a zero correction reproduces the production GARCH kernel exactly.

## Transformers and foundation models

Temporal Fusion Transformers combine recurrent local processing, variable selection, attention, static covariates, and direct multi-horizon quantile forecasts. The original paper includes a five-day-ahead experiment on 31 realized-volatility series and provides useful ideas for mixed covariates and horizon-specific output ([Lim et al., 2021](https://doi.org/10.1016/j.ijforecast.2021.03.012)). A volatility-specific working paper reports good weekly and monthly results for S&P 500 stocks, especially with sector pooling ([Frank, 2023](https://www.econstor.eu/bitstream/10419/268951/1/1837499578.pdf)). Both again point toward pooling and direct multi-horizon training.

The case for embedding a transformer in scivol is weak. Branco, Rubesam, and Zevallos compare linear and nonlinear methods on ten global indices and find no systematic statistical advantage for nonlinear models; simpler models can also deliver more economic value at the monthly horizon ([Branco et al., 2024](https://doi.org/10.1016/j.jempfin.2024.101524)). Brini compares nine zero-shot time-series foundation models with eight econometric benchmarks across 50 assets. Only one small foundation model narrowly beats Log-HAR at every horizon on an asset-balanced measure. Much of the short-horizon pooled gain disappears after Mincer-Zarnowitz recalibration, while a simple foundation-model and Log-HAR average enters the model confidence set for 98 to 100 percent of assets ([Brini, 2026](https://arxiv.org/abs/2607.05291)). This is a recent preprint, so it is evidence to monitor rather than a final verdict.

scivol should support transformers as external estimators under a common protocol. It should not take a hard dependency on PyTorch or select a transformer as a default. Architecture churn is fast; forecast contracts and proper evaluation are durable.

## Probabilistic forecasts and tail risk

Most volatility papers still report point forecasts of variance. Risk applications need more. At minimum they need uncertainty over future variance, a conditional return distribution, or direct tail quantiles. Temporal Fusion Transformers already illustrate multi-horizon quantile output. Chronopoulos, Raftapostolos, and Kapetanios use deep quantile regression for VaR and report gains over linear quantile alternatives, with time-series cross-validation and conditional quantile tests ([Chronopoulos et al., 2024](https://doi.org/10.1093/jjfinec/nbad014)). Bayesian neural VIX forecasts also show why post-training calibration matters: modern neural uncertainty estimates can be miscalibrated even when point RMSE is good ([Hortúa and Mora-Valencia, 2025](https://doi.org/10.1007/s41060-024-00562-5)).

For scivol, probabilistic should have a precise contract. A forecast may return one or more of:

- a conditional variance path;
- distribution parameters for future returns;
- draws from the joint predictive distribution over horizons;
- quantiles of returns, variance, or realized variance;
- epistemic intervals produced by parameter draws or ensembles.

The result must name the target and scale. A 5 percent return quantile, a 5 percent volatility quantile, and a 95 percent interval for conditional variance are different objects. The evaluation layer should match each object to a proper score or calibration test: QLIKE for positive variance point forecasts, log score and CRPS for distributions, pinball loss for quantiles, probability-integral-transform diagnostics for continuous densities, and coverage plus independence tests for VaR.

## Evidence quality and reproducibility

Claims of neural dominance remain hard to compare. A systematic review of 35 post-2015 neural-volatility studies found that different targets, proxies, assets, horizons, splits, and metrics prevented easy meaningful comparisons and called for a shared task ([Ge et al., 2022](https://doi.org/10.1145/3483596)). The conflicting 2022-2026 horse races support that warning. Neural methods win in some pooled or high-frequency designs, while linear HAR models win or remain statistically tied in others.

Common failure modes are:

- random train-test splits or scaling fitted using future data;
- weak baselines, especially omitting Log-HAR, HARQ, GJR-GARCH, or a simple forecast average;
- reporting RMSE alone instead of QLIKE and an economic or tail-risk criterion;
- comparing one trained neural seed with deterministic baselines;
- tuning on the reported test period;
- pooled loss dominated by a few high-variance assets;
- no recalibration check, forecast-comparison test, code, environment lock, or per-origin forecast archive.

scivol already has rolling evaluation and model confidence sets. Extend that machinery into a reproducibility contract. Each experiment should persist data hashes, transformation state, forecast origins, window rules, hyperparameter search ranges, seeds, dependency versions, per-series losses, aggregate weighting, and every raw forecast. Neural candidates should report the distribution across seeds. Panel evaluations should include equal-asset aggregation alongside pooled loss.

## Recommended scivol roadmap

| Priority | Addition | Why now | Minimum correctness bar |
|---|---|---|---|
| P0 | Unified forecast result and scoring API | It lets every classical, hybrid, and external model compete on the same target | Explicit target, scale, horizon, origin, information cutoff, QLIKE, log score, CRPS, pinball loss, calibration output |
| P0 | Leakage-safe rolling and panel experiment runner | Pooling is one of the most credible sources of ML gains | Expanding and rolling windows, per-asset and pooled summaries, seed replication, immutable forecast archive |
| P0 | External forecaster protocol | Supports fast-moving ML models without making them core dependencies | `fit`, `update`, `predict`, `predict_dist`, serialization metadata, deterministic seed handling |
| P1 | Structured hybrid correction wrapper | RECH, GARCH-LSTM, and HARNet provide a defensible nesting pattern | Positivity, correction-off identity, base and correction decomposition, gradient checks for the differentiable wrapper |
| P1 | Global or hierarchical multi-series fitting | Evidence from 100 to 10,000 stocks favors shared estimation | Asset IDs, static and dynamic covariates, masking, new-asset evaluation, no future cross-sectional leakage |
| P1 | Predictive simulation and quantile interface | Tail-risk workflows need distributions rather than point variance | Reproducible draws, horizon dependence, calibrated quantiles, VaR and ES backtests |
| P2 | Roughness estimation and diagnostic report | Roughness is important but empirically disputed | Noise-aware estimator, confidence interval, sampling-scale plot, proxy warning, simulation calibration |
| P2 | OU-factor fractional-kernel engine | This is the practical route to rough filtering and simulation | Direct-convolution oracle, convergence tests by factor count, stable near `H = 0` and `H = 0.5` |
| P3 | Rough Bergomi or rough Heston pricing and neural calibration surrogate | Valuable only if derivative pricing enters scope | Arbitrage-aware surface inputs, Monte Carlo error accounting, surrogate error bounds, independent pricing benchmark |

Two omissions are deliberate. First, a generic `TransformerVolatility` class would age quickly and duplicate mature ML frameworks. The protocol is the stable contribution. Second, rough volatility should not be squeezed into the current `VOLATILITY` role until scivol decides whether that role covers conditional return variance, latent continuous-time variance, or risk-neutral option dynamics. Those are separate statistical objects.

## Source table

| Source | Status | Main use in this note |
|---|---|---|
| [Gatheral, Jaisson, and Rosenbaum, Volatility is rough](https://doi.org/10.1080/14697688.2017.1393551) | Quantitative Finance, 2018 | Foundational rough-volatility evidence and RFSV forecast motivation |
| [Bennedsen, Lunde, and Pakkanen, Decoupling the short- and long-term behavior of stochastic volatility](https://doi.org/10.1093/jjfinec/nbaa049) | Journal of Financial Econometrics, 2022 | Separates roughness from persistence and tests both in forecasting |
| [Cont and Das, Rough volatility: fact or artefact?](https://arxiv.org/abs/2203.13820) | Working paper, 2022 | Shows realized-volatility noise can induce apparent roughness |
| [Bolko et al., A GMM approach to estimate the roughness of stochastic volatility](https://doi.org/10.1016/j.jeconom.2022.06.009) | Journal of Econometrics, 2023 | Noise-corrected GMM and empirical H estimates near 0.05 |
| [Chong et al., Statistical inference for rough volatility: central limit theorems](https://doi.org/10.1214/23-AAP2002) | Annals of Applied Probability, 2024 | Semiparametric H estimator and inferential theory |
| [Damian and Frey, Detecting rough volatility: a filtering approach](https://doi.org/10.1080/14697688.2024.2399284) | Quantitative Finance, 2024 | OU approximation and nested particle-filter inference |
| [Bayer and Breneis, Markovian approximations of stochastic Volterra equations with the fractional kernel](https://doi.org/10.1080/14697688.2022.2139193) | Quantitative Finance, 2023 | Fast finite-factor approximation for rough models |
| [Abi Jaber and Li, Volatility models in practice: rough, path-dependent or Markovian?](https://doi.org/10.1111/mafi.12463) | Mathematical Finance, 2025 | Large SPX option comparison that challenges rough-model dominance |
| [Horvath, Muguruza, and Tomas, Deep learning volatility](https://doi.org/10.1080/14697688.2020.1817974) | Quantitative Finance, 2021 | Neural surrogate for fast classical and rough-model calibration |
| [Bucci, Realized volatility forecasting with neural networks](https://doi.org/10.1093/jjfinec/nbaa008) | Journal of Financial Econometrics, 2020 | Early recent evidence for RNN and LSTM realized-volatility gains |
| [Nguyen, Tran, and Kohn, Recurrent conditional heteroskedasticity](https://doi.org/10.1002/jae.2902) | Journal of Applied Econometrics, 2022 | RNN augmentation of GARCH-type conditional variance; replication data available |
| [Christensen, Siggaard, and Veliyev, A machine learning approach to volatility forecasting](https://doi.org/10.1093/jjfinec/nbac020) | Journal of Financial Econometrics, 2023 | ML versus HAR on Dow 30 realized variance |
| [Zhang et al., Volatility forecasting with machine learning and intraday commonality](https://doi.org/10.1093/jjfinec/nbad005) | Journal of Financial Econometrics, 2024 | Pooled multi-asset training, market RV, and transfer to unseen stocks |
| [Reisenhofer, Bayer, and Hautsch, HARNet](https://arxiv.org/abs/2205.07719) and [official code](https://github.com/mdsunivie/HARNet) | Working paper and replication code, 2022 | HAR-initialized dilated network and reproducible baselines |
| [Moreno-Pino and Zohren, DeepVol](https://doi.org/10.1080/14697688.2024.2387222) | Quantitative Finance, 2024 | Global dilated convolution over raw intraday returns |
| [Zhao et al., From GARCH to neural network for volatility forecast](https://doi.org/10.1609/aaai.v38i15.29643) | AAAI, 2024 | GARCH kernels embedded in LSTM cells |
| [Lim et al., Temporal Fusion Transformers for interpretable multi-horizon time series forecasting](https://doi.org/10.1016/j.ijforecast.2021.03.012) | International Journal of Forecasting, 2021 | Transformer design, direct quantiles, and a realized-volatility application |
| [Frank, Forecasting realized volatility in turbulent times using Temporal Fusion Transformers](https://www.econstor.eu/bitstream/10419/268951/1/1837499578.pdf) | Discussion paper, 2023 | Volatility-specific TFT comparison and sector pooling |
| [Branco, Rubesam, and Zevallos, Forecasting realized volatility: does anything beat linear models?](https://doi.org/10.1016/j.jempfin.2024.101524) | Journal of Empirical Finance, 2024 | No systematic nonlinear dominance across ten indices |
| [Liu et al., Neural-network volatility forecasting](https://doi.org/10.1093/jjfinec/nbag019) | Journal of Financial Econometrics, 2026 | More than 10,000 stocks; sample scaling and global training dominate architecture scaling |
| [Chronopoulos, Raftapostolos, and Kapetanios, Forecasting Value-at-Risk using deep neural network quantile regression](https://doi.org/10.1093/jjfinec/nbad014) | Journal of Financial Econometrics, 2024 | Direct neural quantiles, mixed-frequency inputs, and VaR tests |
| [Hortúa and Mora-Valencia, Forecasting VIX using Bayesian deep learning](https://doi.org/10.1007/s41060-024-00562-5) | International Journal of Data Science and Analytics, 2025 | Probabilistic neural forecasts and post-hoc uncertainty calibration |
| [Ge et al., Neural network-based financial volatility forecasting: a systematic review](https://doi.org/10.1145/3483596) | ACM Computing Surveys, 2022 | Documents comparison problems across 35 studies and proposes a shared task |
| [Brini, Forecasting realized volatility with time series foundation models](https://arxiv.org/abs/2607.05291) and [official code](https://github.com/Alessiobrini/tsfm-rv) | Working paper and code, 2026 | Nine foundation models versus eight econometric models, recalibration, and model confidence sets |
