# Applications and the library requirements they create

## Why applications matter to the roadmap

The same conditional-variance equation can appear in a portfolio-risk paper, an
electricity-price forecast, and a macroeconomic uncertainty study. The data and
decisions around it are different. A complete library should not encode every
industry's business rules, but it should support the statistical operations that
recur across serious applications.

## Traded financial assets

Equities, foreign exchange, government bonds, rates, commodities, and crypto
remain the central use cases. The recurring outputs are one- and multi-period
variance forecasts, complete predictive densities, tail risk, covariance and
correlation paths, hedge ratios, conditional beta, scenarios, and volatility
spillovers.

The application horizon matters. A review using equity, Treasury, currency, and
commodity indices finds that the best construction differs between GARCH and
realized-volatility models and across horizons. Iterated GARCH often beats direct
GARCH, while direct realized-volatility models often beat iterated variants;
MIDAS designs perform well at longer horizons ([Ghysels et al.,
2019](https://doi.org/10.1146/annurev-financial-110217-022808)). A forecast API
therefore needs an explicit target, aggregation rule, and origin. A generic
`forecast(20)` cannot imply whether it returns the variance state on day 20 or
the distribution of a 20-day P&L.

Multivariate work is used for portfolio covariance, dynamic hedging, contagion,
and shock transmission. Recent structural volatility impulse-response work
applies BEKK to U.S. equity, government-bond, and foreign-exchange markets
([Fengler and Herwartz,
2025](https://doi.org/10.1093/jjfinec/nbae036)). This supports small-system BEKK,
news-impact surfaces, and impulse-response tools. Large portfolios instead need
shrinkage, factors, block restrictions, or composite likelihood.

Crypto does not require a separate GARCH family. It strengthens the case for
continuous calendars, explicit day boundaries, heavy-tailed and skewed
innovations, structural-break tests, jumps, missing and stale observations, and
rolling evaluation across market regimes.

## Market and regulatory risk

VaR, expected shortfall, filtered historical simulation, and scenario analysis
turn a volatility model into a risk workflow. The Basel market-risk framework
uses 97.5% expected shortfall for internal-model capital but still uses one-day
VaR backtests and P&L attribution for model eligibility ([BCBS,
2019](https://www.bis.org/bcbs/publ/d457.htm)). scivol should expose generic,
well-defined risk functionals and backtests. It should not claim regulatory
compliance, which also depends on instrument mapping, liquidity horizons,
stress-period selection, modellability, P&L definitions, and governance.

Options and volatility derivatives add risk-neutral dynamics. Historical
physical-measure volatility is an input, not an option price. Duan's GARCH option
model and Heston-Nandi's affine construction both require explicit pricing
assumptions ([Duan,
1995](https://doi.org/10.1111/j.1467-9965.1995.tb00099.x); [Heston and Nandi,
2000](https://doi.org/10.1093/rfs/13.3.585)). This work belongs in an integration
with option panels, rates, dividends, pricing kernels, calibration residuals,
and Greeks.

## Energy and electricity

Wholesale electricity prices add deterministic seasonality, weather and load
regressors, scarcity spikes, negative prices, regime changes, and horizons from
hours to months. Weron's review finds mixed evidence for ordinary GARCH and
better performance in some crisis or high-volatility periods. It also stresses
that model rankings are hard to compare when calibration samples and
implementations differ ([Weron,
2014](https://doi.org/10.1016/j.ijforecast.2014.08.008)).

The library implications are concrete:

- seasonal and calendar-safe mean terms;
- future paths for weather, load, and other regressors;
- variance regressors and mixed-frequency inputs;
- spikes, jumps, and regime-change diagnostics;
- probabilistic forecasts and proper scores, not only point errors;
- rolling evaluation by horizon and market regime.

ARX and HARX already supply part of this. Variance-X, GARCH-MIDAS, stability
tests, and explicit future-regressor alignment would close more of the practical
gap than another daily-equity density.

## Macroeconomic uncertainty

ARCH began with inflation uncertainty as well as finance. Macroeconomic uses
often treat the conditional variance itself as an explanatory variable or
uncertainty index. They need GARCH-in-mean, mixed frequencies, long-run and
short-run components, structural breaks, vintage-safe exogenous data, and
missing-observation rules.

GARCH-MIDAS directly separates daily short-run variance from a long-run component
driven by lower-frequency macro variables ([Engle, Ghysels, and Sohn,
2013](https://doi.org/10.1162/REST_a_00300)). The newer MF2-GARCH offers a
stationary two-cycle construction and reports long-horizon gains on a large
equity panel ([Conrad and Engle,
2025](https://doi.org/10.1002/jae.3118)). Both strengthen the case for a general
component result that exposes short and long variance paths.

## High-frequency trading and market microstructure

Intraday applications require more than applying daily GARCH to every tick.
Trades arrive irregularly, prices are discrete, bid-ask bounce creates serially
dependent noise, assets trade asynchronously, and sessions contain auctions and
breaks. Realized measures, autoregressive conditional duration, and
score-driven count or price-change models answer different parts of this
problem.

The immediate shared need is a provenance-aware high-frequency data layer. It
should record cleaning, price type, session calendar, sampling, synchronization,
noise treatment, and jump treatment. ACD and multiplicative-error models become
more natural once that layer exists. Engle and Russell's ACD model is the
foundational event-time reference ([1998](https://doi.org/10.2307/2999632)).

## Positive-valued and nonfinancial series

Multiplicative-error and score-driven models extend conditional dynamics to
positive measurements, counts, durations, and changing distributional
parameters. This is a better route into insurance, environmental, epidemiologic,
and operational data than forcing every series through a return-GARCH equation.

Spatial and spatiotemporal volatility models are also a growing field. A recent
review covers spatial extensions of GARCH and stochastic volatility and notes
the dimensional limits of VECH and BEKK constructions ([Otto,
2025](https://doi.org/10.1111/joes.12643)). These applications need adjacency or
distance structures, missing locations, panels, and sparse computation. They are
strategically interesting, but they require a new indexed-state abstraction and
should follow the core time-series workflow and high-dimensional covariance
work.

## Product conclusion

Financial applications justify completing the present forecast, risk,
diagnostic, and covariance workflows. Energy and macro applications justify
variance regressors, mixed frequencies, components, regimes, and calendar-safe
alignment. High-frequency applications justify realized measures and explicit
irregular-time semantics. Broader positive, count, duration, and spatial data
justify a later score-driven or multiplicative-error architecture.

That progression expands the range of valid applications without turning
scivol into a collection of industry-specific wrappers.

## Source table

| Source | Application signal |
|---|---|
| [Engle (2001), GARCH 101](https://doi.org/10.1257/jep.15.4.157) | Applied GARCH motivation and portfolio risk. |
| [Ghysels et al. (2019), multiperiod forecasts](https://doi.org/10.1146/annurev-financial-110217-022808) | Cross-asset and cross-horizon volatility forecast design. |
| [Fengler and Herwartz (2025), structural volatility impulse responses](https://doi.org/10.1093/jjfinec/nbae036) | Small-system BEKK shock transmission across equity, bonds, and FX. |
| [BCBS (2019), market-risk framework](https://www.bis.org/bcbs/publ/d457.htm) | ES capital, VaR backtesting, and P&L attribution distinction. |
| [Duan (1995), GARCH option pricing](https://doi.org/10.1111/j.1467-9965.1995.tb00099.x) | Physical-to-risk-neutral assumptions for GARCH pricing. |
| [Heston and Nandi (2000), affine GARCH option valuation](https://doi.org/10.1093/rfs/13.3.585) | Closed-form pricing for a specified GARCH risk-neutral model. |
| [Weron (2014), electricity price forecasting review](https://doi.org/10.1016/j.ijforecast.2014.08.008) | Seasonality, exogenous drivers, spikes, regimes, and evaluation design. |
| [Engle, Ghysels, and Sohn (2013), GARCH-MIDAS](https://doi.org/10.1162/REST_a_00300) | Mixed-frequency macro drivers and long-run variance. |
| [Conrad and Engle (2025), MF2-GARCH](https://doi.org/10.1002/jae.3118) | Stationary short and long volatility cycles and long-horizon forecasts. |
| [Engle and Russell (1998), ACD](https://doi.org/10.2307/2999632) | Event-time duration dynamics for irregular transactions. |
| [Otto (2025), spatial and spatiotemporal volatility review](https://doi.org/10.1111/joes.12643) | Nonfinancial and spatial extensions, with multivariate scaling limits. |
