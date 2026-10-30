# scivol

scivol fits univariate conditional volatility models and dynamic conditional correlation (DCC) models in Python. Models are built from components (a mean, a volatility recursion and an innovation density) and estimated jointly by exact maximum likelihood, with analytical gradients and Hessians computed in a standalone C++20 library.

```python
from scivol import ARMA, GJRGARCH, SkewT

result = (ARMA(1, 0) + GJRGARCH(1, 1) + SkewT()).fit(returns)
result.summary()
forecast = result.forecast(10)
```

Daily returns of an equal-weighted portfolio of 1,381 US stocks, 2015–2024, with ±2 conditional standard deviations from a fitted GJR-GARCH model with skewed-t innovations:

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/vengveng/scivol/main/docs/images/returns-volatility-dark.svg">
  <img alt="Daily portfolio returns, 2015 to 2024, inside a band of plus and minus two fitted conditional standard deviations that widens sharply in March 2020" src="https://raw.githubusercontent.com/vengveng/scivol/main/docs/images/returns-volatility-light.svg">
</picture>

Conditional correlations among five of those stocks from a DCC model; the line is their average and the band the middle half of the ten pairwise correlations:

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/vengveng/scivol/main/docs/images/dcc-correlation-dark.svg">
  <img alt="Average conditional correlation among five US stocks, 2015 to 2024, rising from about 0.3 to nearly 0.6 in March 2020" src="https://raw.githubusercontent.com/vengveng/scivol/main/docs/images/dcc-correlation-light.svg">
</picture>

**Models.** ARMA, ARX and HARX means; GARCH, GJR-GARCH, EGARCH and APARCH volatility; Normal, Student-t, Hansen's skewed t, GED, generalized hyperbolic, NIG, hyperbolic and variance-gamma innovations; DCC with Gaussian or Student-t copulas. Every combination is one model with one likelihood.

**Estimation you can check.** Each likelihood and its derivatives are tested against an automatic-differentiation reference. A fit is reported as successful only when the first-order conditions hold in the model's parameter domain; otherwise the estimate comes back with a status that says why. Standard errors are withheld where ordinary inference does not apply. Several starting points are tried by default, because volatility likelihoods often have more than one local maximum.

**Speed.** On 500 liquid US stocks and ten models, a fit took a median 1–6 ms, 4.9 times faster than `arch` and 53 times faster than `rugarch` per fit, while missing its own likelihood maximum less often than either; DCC fits were 20–50 times faster than `rmgarch`. See [the benchmarks](https://github.com/vengveng/scivol/blob/main/docs/benchmarks.md) for the method and its limits.

## Installation

```bash
pip install scivol
```

Wheels for CPython 3.10–3.12 on Linux (x86-64), macOS (Apple Silicon) and Windows (x64); elsewhere pip builds from source, which needs a C++20 compiler, CMake and Ninja.

## Documentation

[Getting started](https://github.com/vengveng/scivol/blob/main/docs/getting-started.md) · [Models](https://github.com/vengveng/scivol/blob/main/docs/models/volatility.md) · [Estimation](https://github.com/vengveng/scivol/blob/main/docs/estimation.md) · [Inference](https://github.com/vengveng/scivol/blob/main/docs/inference.md) · [Forecasting](https://github.com/vengveng/scivol/blob/main/docs/forecasting-and-simulation.md) · [Diagnostics](https://github.com/vengveng/scivol/blob/main/docs/diagnostics.md) · [Model selection](https://github.com/vengveng/scivol/blob/main/docs/selection.md) · [API](https://github.com/vengveng/scivol/blob/main/docs/api.md) · [References](https://github.com/vengveng/scivol/blob/main/docs/references.md)

The defaults of the automatic model-selection procedures have not yet been calibrated by simulation.

## License

MIT.
