# Model selection

scivol selects among volatility models in three ways: a general-to-specific search that tests restrictions, ranking by an information criterion, and a model confidence set built from out-of-sample forecast losses. All three start from a `Search`, which declares the candidate models, and differ in the procedure passed to `Search.fit`.

## Declaring the candidates

`Search` takes lists of mean, volatility and density components. `up_to` declares a largest model, and the search reaches the smaller ones by restricting it:

```python
from scivol import ARX, GJRGARCH, EGARCH, SkewT, Search, GeneralToSpecific

true_model = GJRGARCH(1, 1) + SkewT()
y = true_model.simulate(2000, [0.05, 0.03, 0.1, 0.88, 0.12, -0.15], seed=11).data

search = Search(mean=[ARX.up_to(1)],
                volatility=[GJRGARCH.up_to(1, 1), EGARCH.up_to(1, 1)],
                density=[SkewT()])
selection = search.fit(y, procedure=GeneralToSpecific())
print(selection.summary())
```

```text
Selection: selected; selected: GJR-GARCH(1,1)+SkewT
Search complete: True; terminals: 2; optimizer invocations: 38
```

The search starts from the two largest models, ARX(1) with GJR-GARCH(1,1) and with EGARCH(1,1), and from a constant-variance model, then removes terms while the data allow it. It ends at two models, and selects the one that generated the data:

```python
table = selection.table()
print(table[table.terminal][["model", "parameters", "log_likelihood", "bic", "adequacy"]].to_string(index=False))
```

```text
               model  parameters  log_likelihood         bic adequacy
GJR-GARCH(1,1)+SkewT           6    -2800.897821 5647.398057 adequate
   EGARCH(1,1)+SkewT           6    -2804.972070 5655.546554 adequate
```

`table()` has one row for each of the 37 models fitted. `selection.selected` is the selected model fitted to the full sample, an ordinary result with `summary()`, `forecast()` and the rest.

`Search` has the defaults `mean=[None]` (zero mean), no volatility components, `density=[Normal()]`, `constant_variance=True` and `closure=True`. With `closure=False`, only the declared models are candidates. Every candidate scores the same observations, from the largest mean lag or `hold_back`, whichever is later, with the same starting variance, so that their likelihoods can be compared. For the same reason a constant-variance model and `GARCH(0, 0)` are different candidates: the first estimates one variance for every row, the second starts from the common starting variance.

## General-to-specific search

`GeneralToSpecific` starts from the largest models that pass the adequacy checks described below. At each model it tries every supported one-step restriction, such as dropping the last lag or setting a skewness parameter to zero, fits the restricted model, and accepts the restriction when a likelihood-ratio test does not reject it and the restricted model is still adequate. Each accepted path is also tested jointly against its starting model. Paths can branch, and the search stops at models that admit no further accepted restriction. Fits are shared between paths, and restricted models start from their parent's estimates.

| Option | Default | Meaning |
| --- | --- | --- |
| `reduction_level` | `0.01` | Level of the restriction tests |
| `encompassing_level` | `0.01` | Level of the joint test of each path against its starting model |
| `diagnostic_level` | `0.01` | Level of the adequacy checks |
| `prefer` | `"fewest_parameters"` | How to choose among final models; or `"information_criterion"` |
| `tie_break` | `InformationCriterion("bic")` | Criterion that breaks ties in the parameter count |
| `max_fits` | `256` | Budget of optimizer runs |
| `max_mean_order` | `5` | Largest AR order that mean widening may reach, from 1 to 8 |

A model is adequate when none of three checks rejects under Holm's correction ([Holm, 1979](references.md#holm-1979)) at `diagnostic_level`: the Ljung-Box test on the standardized residuals and the absolute-residual Li-Mak test, each at lags $\min(10, n/5)$ and $\min(20, n/5)$, and the absolute sign-bias test. These tests need only finite second moments of the innovations; [diagnostics](diagnostics.md#absolute-residual-tests) gives their formulas. When a check cannot be evaluated, the model's adequacy is unresolved and it cannot serve as a starting point.

Independent fits run in parallel with `n_jobs` workers, by default up to four, and the result does not depend on the number of workers. The search requires maximum-likelihood fits, because the restriction tests use likelihood ratios, so `method="qmle"` raises.

### Widening the mean

If every starting model fails the adequacy checks, and each fails the residual Ljung-Box test, the remaining autocorrelation points to the mean. The search then adds one AR lag to each mean, keeping any intercept, regressors and MA terms, and starts again, up to `max_mean_order`. A starting model that is adequate, or whose adequacy is unresolved, prevents widening. Each wider model is fitted on a new common sample and counts against the budget. `selection.widenings` records each step and the test that triggered it.

### Automatic selection

`Auto(max_mean_order=5).fit(y, x=None)` runs the general-to-specific search over a fixed set of starting models: an ARX(1) mean, with regressors if you pass `x`, combined with GJR-GARCH(1,1), EGARCH(1,1) and APARCH(1,1), each with skewed-t innovations, plus a constant-variance model. To change the starting models or the procedure, use `Search` directly.

## Restriction tests

The likelihood-ratio statistic for a parent model with log-likelihood $\ell_P$ and a restricted child with $\ell_C$ is $2(\ell_P-\ell_C)$, computed on the common sample. Both fits must have `success=True`. A negative statistic means that the parent's fit stopped short of its maximum, and scivol then refits the parent rather than setting the statistic to zero.

| Restriction | Reference distribution |
| --- | --- |
| Drop the intercept, the last AR, MA or regressor coefficient, or one HAR window | chi-square |
| GARCH: drop the last shock or variance lag | chi-bar-square (the coefficient is on its bound under the null) |
| GJR-GARCH: drop the last shock lag | chi-bar-square, for the two sign responses `alpha` and `alpha + gamma` |
| GJR-GARCH: drop the last variance lag | chi-bar-square |
| GJR-GARCH to GARCH: all `gamma` zero | chi-square, when both sign responses are positive |
| EGARCH: drop the last shock pair or variance lag | chi-square |
| APARCH: drop the last shock pair | none: the asymmetry is not identified under the null |
| APARCH: drop the last variance lag | chi-bar-square |
| Skewed t to Student-t: `lam = 0` | chi-square |
| Student-t to Normal: `eta = 0` | chi-bar-square |
| GED to Normal: `nu = 2` | chi-square |

When the null puts a parameter on the boundary of its domain, the statistic follows a chi-bar-square distribution, a mixture of chi-squares whose weights depend on the information matrix ([Self and Liang, 1987](references.md#self-liang-1987); [Andrews, 2001](references.md#andrews-2001)). scivol computes the weights in closed form for up to three boundary parameters and by numerical integration, to an absolute error of $10^{-6}$, for more. When other parameters also sit on their bounds, the distribution is no longer a chi-bar-square, and scivol simulates it with a fixed seed and reports the simulation error. A test whose p-value cannot be separated from the level within these errors is reported as unresolved.

A fit at a likelihood kink, such as EGARCH with an estimated mean whose residual sits exactly at zero, is tested with the same laws. Its likelihood is not differentiable, but its population counterpart is, and the few residuals at zero move the statistic by a vanishing amount; in simulations of EGARCH with an AR(1) mean, the 5% tests at kink points rejected in 5–7% of cases (9% in one of eight settings), close to their rate at smooth fits. The search accepts only restrictions that have a reference distribution, which excludes those whose information matrix is singular and those that remove a parameter not identified under the null, such as the APARCH asymmetry. Each entry of `selection.tests` records the statistic, the distribution used, its weights or simulation error, the assumptions, the decision, and the reason for any failure.

## Information criteria

`InformationCriterion(name="bic", adequacy=...)` fits every declared model and ranks them by AIC, BIC or HQIC. You must choose whether the adequacy checks apply: `adequacy=True` ranks only the models that pass them, and `adequacy=False` ranks every model with `success=True`.

```python
from scivol import GARCH, GJRGARCH, Normal, StudentT, SkewT, InformationCriterion

comparison = Search(volatility=[GARCH(), GJRGARCH()],
                    density=[Normal(), StudentT(), SkewT()],
                    constant_variance=False, closure=False)
ranked = comparison.fit(y, procedure=InformationCriterion("bic", adequacy=False))
print(ranked.table()[["model", "parameters", "log_likelihood", "bic"]].sort_values("bic").to_string(index=False))
```

```text
                  model  parameters  log_likelihood         bic
       GARCH(1,1)+SkewT           5    -2805.828435 5649.661382
   GJR-GARCH(1,1)+SkewT           6    -2802.295852 5650.197119
GJR-GARCH(1,1)+StudentT           5    -2811.082195 5660.168902
    GARCH(1,1)+StudentT           4    -2814.982518 5660.368645
  GJR-GARCH(1,1)+Normal           4    -2832.625323 5695.654256
      GARCH(1,1)+Normal           3    -2838.548752 5699.900211
```

BIC narrowly prefers GARCH over the true GJR-GARCH model here, because the leverage effect in the simulation is small. With $k$ parameters and $n$ scored observations, the criteria are

$$\mathrm{AIC}=-2\ell+2k,\qquad \mathrm{BIC}=-2\ell+k\log n,\qquad \mathrm{HQIC}=-2\ell+2k\log\log n,$$

from [Akaike (1974)](references.md#akaike-1974), [Schwarz (1978)](references.md#schwarz-1978) and [Hannan and Quinn (1979)](references.md#hannan-quinn-1979). `diagnostic_level=0.01` works as for the general-to-specific search, and `max_fits=1000`, since a ranking needs every candidate fitted.

## Predictive confidence sets

`ConfidenceSet` compares models by their one-step-ahead forecasts on a holdout sample at the end of the data. A training procedure, by default a general-to-specific search, selects candidates on the data before the holdout; scivol then forecasts through the holdout, refitting every `refit_every` observations on all data up to that point, and computes the model confidence set of [Hansen, Lunde and Nason (2011)](references.md#hansen-lunde-nason-2011) from the forecast losses. The holdout must contain at least 500 observations.

```python
from scivol import ConfidenceSet

y = true_model.simulate(2500, [0.05, 0.03, 0.1, 0.88, 0.12, -0.15], seed=11).data
procedure = ConfidenceSet(reduction=InformationCriterion("bic", adequacy=False),
                          holdout=0.2, confidence=0.90, loss="log_score", seed=44)
predictive = comparison.fit(y, procedure=procedure)
print([str(c.specification) for c in predictive.confidence_set])
print(predictive.summary())
```

```text
['GARCH(1,1)+SkewT', 'GJR-GARCH(1,1)+SkewT', 'GJR-GARCH(1,1)+Normal', 'GARCH(1,1)+Normal']
Selection: predictive_confidence_set; selected: GARCH(1,1)+Normal
Search complete: True; terminals: 6; optimizer invocations: 151
```

With 500 forecasts, the bootstrap eliminates only the two Student-t models, and four models remain in the 90% set. scivol then deploys the smallest model in the set, breaking ties by training BIC, and refits it on the full sample. Here that is GARCH-Normal, the model with the fewest parameters that the holdout cannot reject. A longer holdout, or a different `loss`, would separate the models more sharply.

`loss="log_score"` is the negative log predictive density, and `loss="qlike"` is $\log h_t+e_t^2/h_t$. Within a refit block, each new observation updates the variance recursion before the next forecast. The confidence set uses the range statistic and a circular block bootstrap of the aligned loss rows, with `bootstrap_draws=4999` resamples from NumPy's PCG64 generator seeded by `seed`. `selection.predictive` holds the losses, the bootstrap block length, the p-values and the included and eliminated models.

## Reading the result

| Attribute | Contents |
| --- | --- |
| `status` | `selected`, `predictive_confidence_set`, `no_adequate_model`, `adequacy_unresolved`, `budget_exhausted`, ... |
| `selected`, `selected_model` | The selected model fitted to the full sample, and its specification; `None` when nothing qualifies |
| `terminals` | The models at which the search stopped |
| `candidates`, `table()` | Every model fitted, with likelihood, criteria, `success`, adequacy and failure reason |
| `tests` | Every restriction test, with its decision and reference distribution |
| `paths` | The accepted restriction paths |
| `confidence_set`, `predictive` | The model confidence set and its holdout results |
| `widenings` | The mean-widening steps |
| `search_complete`, `frontier` | Whether the budget sufficed, and the models left unexplored if it did not |
| `fit_counts` | Optimizer runs, reused fits, warm starts, refits |

When the budget runs out, `search_complete` is `False`, `frontier` lists the models not yet explored, and the best eligible model found so far is still returned. `table()` returns a pandas DataFrame, or a list of dictionaries when pandas is not installed.

## How the defaults perform

The defaults were checked by simulation. The search covered an AR(1) mean, GJR-GARCH(1,1) and EGARCH(1,1) with a skewed t, and constant variance. Six models inside it generated 120 series each, at n = 1000 and at n = 3000. The table gives the share of series on which each procedure selected the generating model. A GARCH(0,0) recursion counts as constant variance:

| Data-generating model | GETS, n=1000 | BIC, n=1000 | GETS, n=3000 | BIC, n=3000 |
| --- | --- | --- | --- | --- |
| GARCH(1,1), Normal | 90% | 94% | 99% | 98% |
| GARCH(1,1), Student-t | 82% | 91% | 95% | 98% |
| GJR-GARCH(1,1), skewed t | 37% | 44% | 89% | 91% |
| EGARCH(1,1), Normal | 52% | 79% | 88% | 90% |
| AR(1) mean, GARCH(1,1), Student-t | 68% | 72% | 98% | 98% |
| iid Normal | 93% | 95% | 98% | 98% |

The general-to-specific search used its defaults, and BIC ranked the whole search with adequacy checks. At n = 1000, most misses drop a small leverage effect or an AR term. Overfitting was rare for both procedures, at 2–3% of series. Restriction tests whose null held rejected in 0.5–2% of cases at the 1% level. The true model failed the adequacy checks in at most 1% of series. With all levels at 5%, the search overfitted 12% of the time and selected the true model less often at n = 3000 (82%), so the defaults stay at 1%. AIC overfitted about 40% of the time. The study's scripts are not part of the package.

## Limitations

These simulations cover models inside the search space at two sample sizes. Error rates for misspecified models, much larger searches or shorter series are not established. Some restrictions are missing from the search: fixing the APARCH power or removing its asymmetry, a skewed Normal, and the limits of the generalized hyperbolic family. ARMA order searches need a minimal, nonsingular representation and must be requested explicitly. The procedures do not cover DCC models, structural breaks, long memory or infinite-variance models, and they report no confidence intervals that account for the selection itself.
