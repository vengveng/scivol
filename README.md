# scivol

scivol fits univariate conditional-volatility models and DCC correlations.
Likelihoods, analytical gradients and Hessians, optimization, inference,
forecasting, simulation and statistical tests live in a standalone C++20 library.
Python supplies component composition and orchestration.

Supported volatility models are GARCH, GJR-GARCH, EGARCH and APARCH. Means may
include ARMA, ARX or HARX terms. Innovations may be Normal, Student-t, Hansen
skew-t, GED, generalized hyperbolic, NIG, Hyperbolic or Variance-Gamma.
See [MODELS.md](MODELS.md) for equations, domains, initialization and exact
forecast horizons.

## Install

```bash
pip install .
```

A source build requires a C++20 compiler, CMake, Ninja, nanobind and
scikit-build-core. The native library also builds independently of Python.

## Fit and evaluate

Components combine in mean, volatility, density order. A missing density is
Normal; a missing volatility component gives constant variance. `+` composes
components.

```python
import numpy as np
from scivol import ARMA, ARX, GARCH, GJRGARCH, Normal, SkewT, StudentT

rng = np.random.default_rng(17)
y = (GARCH() + StudentT()).simulate(
    2600, [.1, .15, .7, .15], seed=62,
).data
spec = ARMA(1, 0) + GJRGARCH(1, 1) + StudentT()
result = spec.fit(y)
print(result.log_likelihood, result.bic, result.success)
result.summary()
```

`fit` accepts `method="mle"` or `"qmle"`, `coordinates="automatic"`, `"z"` or
`"theta"`, `maxiter`, `accuracy`, `initial_params` and `starts`. Data options are
`x`, `hold_back` and `scale`. Unknown options raise. A warm start is a parameter
array or a previous result for the same specification.

Volatility likelihoods can have several local maxima: daily returns often have
one at moderate persistence and a higher one near integration. Each model offers
candidate starts in structurally different regions. With the default
`starts="auto"`, the first runs in full and each other is probed for a few
iterations, then finished only if the probe has already climbed above the best
finished start. `starts=n` runs the first `n` candidates in full (`starts=1` is a
single start); a list of parameter vectors runs each of them in full. Every start
and its outcome is recorded in `fit_info.starts`, and `fit_info.solver` is the path
of the start that produced the estimate. DCC's `fit_from_residuals` takes the same
`starts`.

QMLE estimates mean and volatility with a Gaussian quasi-likelihood and sandwich
covariance. A shaped density adds a staged shape fit and two-step sandwich
covariance. Inspect `result.fit_info` for optimizer phases, stationarity,
inference status and warnings. A finite estimate can remain uncertified;
`result.success` reports certification.
An estimate at a boundary the declared domain excludes, such as persistence
one under covariance stationarity, has inference status `at_excluded_boundary`;
`fit_info.excluded_boundary` names the conditions.

Volatility components ship with the textbook covariance-stationary domain.
`stationarity="integrated"` admits persistence one (IGARCH), and
`stationarity="none"` drops the stationarity condition (positivity only; any
finite beta for EGARCH, which has no integrated declaration):

```python
GARCH(1, 1, stationarity="integrated") + StudentT()
GJRGARCH(1, 1, stationarity="none") + SkewT()
```

At a certified residual kink, `fit_info.kink_point` and `active_kink_rows`
identify the nonsmooth estimate. Its inference status is `at_kink`; ordinary
Hessian standard errors and restriction LR calibration are unavailable.

```python
robust = (GARCH() + Normal()).fit(y, method="qmle")
fixed = spec.fix(y, result.params)
print(fixed.log_likelihood)
state = spec.filter(y, result.params)
forecast = result.forecast(5, seed=5)
print(forecast.residual_variance)
```

`spec.loglikelihood`, `spec.score`, `spec.hessian`, `spec.filter` and `spec.fix`
evaluate fixed parameters. `spec.simulate` draws a series at supplied parameters.
`result.forecast` returns conditional means, observation variances and residual
variances. Some multi-step expectations require simulation, as documented in
MODELS.md. Parameter values use native theta order: mean, volatility, shapes.
Student-t uses `eta=1/nu`, including the exact Normal boundary `eta=0`.

ARX accepts contemporaneous regressors through `x=`. HARX uses averages over its
specified lag windows. ARMA and ARX accept `constant=False` to omit the intercept.

```python
x = rng.normal(size=(len(y), 2))
regression = (ARX(1) + GARCH()).fit(y, x=x)
print(regression.params)
```

## Model selection

`Search` declares component generators. Native `Model.restrictions()` supplies
their downward closure, including mean, volatility and density reductions.
`up_to` declares a maximal generator; it does not enumerate an order grid in
Python. With `closure=False`, only explicitly listed candidates may be selected.
All candidates use one native comparison plan with identical scored rows,
regressor span, initialization and variance seed.

```python
from scivol import (APARCH, EGARCH, Search, GeneralToSpecific,
                    InformationCriterion, ConfidenceSet, Auto)

search = Search(
    mean=[ARX.up_to(1)],
    volatility=[GJRGARCH.up_to(1, 1), EGARCH.up_to(1, 1), APARCH.up_to(1, 1)],
    density=[SkewT()],
    constant_variance=True,
)
selection = search.fit(y, procedure=GeneralToSpecific(max_fits=96), n_jobs=2)
print(selection.summary())
print(selection.table())
```

General-to-specific selection requires certified, adequate roots. Supported
reductions use native LR tests, adequacy checks with Holm's rule, and joint
backtests against the original root. It retains multiple accepted paths and
memoizes fits. Terminals rank by free parameters, then BIC; set
`prefer="information_criterion"` for IC ranking alone. Defaults are 0.01 for
reduction, encompassing and diagnostics. `method="qmle"` raises because robust
LR calibration is unavailable.

Adequacy requires residual Ljung-Box, estimation-corrected absolute-residual
Li-Mak and absolute sign/size-bias checks, combined with Holm at the diagnostic
level; their laws need only a finite innovation variance. In-sample Berkowitz
and DGT appear in each result's diagnostics and never constrain adequacy. See [MODELS.md](MODELS.md#automatic-selection)
for assumptions and current limits.

`Auto().fit(y, x=None)` declares AR(1) with a selectable intercept crossed with
GJR(1,1), EGARCH(1,1), APARCH(1,1), and skew-t, plus a constant-variance root.
It uses the same `GeneralToSpecific` procedure.

An IC search must explicitly enable or disable adequacy. Disabling it gives an
IC ranking of certified fits without an adequacy claim.

```python
comparison = Search(
    volatility=[GARCH()], density=[Normal(), StudentT()],
    constant_variance=False, closure=False,
)
ranked = comparison.fit(y, procedure=InformationCriterion("bic", adequacy=False))
print(ranked.selected_model, ranked.selected.log_likelihood)
```

A `SelectionResult` exposes `selected`, `selected_model`, `terminals`, `paths`,
`tests`, `candidates`, `confidence_set`, `status`, `search_complete`, `frontier`
and `fit_counts`. `selected` is an ordinary `EstimationResult` on the full sample.
An identical comparison-plan fit is reused; otherwise the normal fit path refits
it. `table()` returns a pandas DataFrame when pandas is installed, or records
otherwise. `summary()` returns text.

Every attempted LR test records its restriction, parent/child likelihoods,
statistic, reference law and mixture weights, p-value, numerical error, decision
level, diagnostics, assumptions and failure reason. Unsupported or unidentified
edges are never accepted. On optimizer-budget exhaustion, the result reports
`search_complete=False`, its frontier and the best certified adequate fit found.
Independent fits use a thread pool and deterministic node ordering.

## Predictive confidence sets

`ConfidenceSet` freezes the training procedure's terminal specifications and
compares one-step predictive losses on a chronological holdout. GETS training
requires the adequacy checks; explicit IC training can disable adequacy.
The holdout must contain at least 500 observations.

```python
predictive = comparison.fit(
    y,
    procedure=ConfidenceSet(
        reduction=InformationCriterion("bic", adequacy=False),
        holdout=.2, confidence=.90, loss="log_score",
        refit_every=100, bootstrap_draws=4999, seed=44,
    ),
    n_jobs=2,
)
print(predictive.summary())
```

Losses are negative predictive log densities or QLIKE. Expanding refits use only
observations available at each origin; the training variance seed stays fixed.
The native Hansen–Lunde–Nason MCS uses the range statistic and a circular block
bootstrap of jointly aligned loss rows. The block-length rule is native
`mcs_block_length`; sensitivity to that rule remains the caller's responsibility.
NumPy PCG64 owns the seeded bootstrap uniforms. MCS resampling needs no refits.

The whole confidence set is returned. The deployment decision takes its smallest
model, breaking ties by training BIC, and refits on all observations. That refit
has no new holdout performance estimate. Coverage concerns the best member of
the frozen candidate set under the MCS assumptions.

## Multiple series and rolling windows

```python
panel = np.column_stack([y, .8*y])
batch = GARCH().fit_many(panel, n_jobs=2)
rolling = GARCH().fit_rolling(panel, window=2400, step=200, n_jobs=2)
print(batch.succeeded, len(rolling))
```

`fit_many` returns successful results and per-series failures.
`fit_rolling` warm-starts each series from its preceding window. A multi-column
array or pandas DataFrame passed directly to `.fit()` returns a result dictionary.
Pandas indices are retained. Each worker receives its own component copy and
native model. A batch timeout records a failure after the native call returns;
threads cannot interrupt an in-flight native fit.

## Diagnostics and results

`result.diagnostic_tests(print_results=False)` returns native residual and PIT
tests, estimation-corrected Li–Mak, sign bias and Berkowitz results. Every test
reports applicability/status. Innovation moment failures and boundary estimates
can make a reference law unavailable. In-sample PIT p-values are uncalibrated.

Results expose parameters, likelihood, AIC/BIC/HQIC, residuals, conditional
variances, Hessian, covariance and robust covariance, timing and fit metadata.
Use `result.garch_params.persistence` and
`result.garch_params.unconditional_variance` for fitted model quantities.
The component constructors themselves do not estimate persistence.

## Development

```bash
make native
pytest tests/
pytest tests/ --run-slow
python benchmark.py
```

The native tests build the standalone library. Fast tests skip the full oracle
matrices and marked simulation cases. Benchmarks report time alongside
likelihood and certification so comparisons retain equal solution quality.
