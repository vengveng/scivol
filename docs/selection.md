# Model selection

The selection defaults and the complete adaptive procedure have not been calibrated. Results report the explored graph, test assumptions, unresolved decisions and fit budget. Non-rejection does not establish a true model.

## Declare the candidates

`Search` takes iterables of mean, volatility and density generators. Their cross-product declares specifications. Native restrictions supply downward closure; `up_to` constructs a maximal generator, not a Python enumeration of every order.

```python
from scivol import ARX, GJRGARCH, EGARCH, APARCH, SkewT
from scivol import Search, GeneralToSpecific

search = Search(
    mean=[ARX.up_to(1)],
    volatility=[GJRGARCH.up_to(1, 1), EGARCH.up_to(1, 1), APARCH.up_to(1, 1)],
    density=[SkewT()], constant_variance=True,
)
# selection = search.fit(y, procedure=GeneralToSpecific(max_fits=96), n_jobs=2)
```

Defaults are `mean=[None]`, no dynamic volatilities, `density=[Normal()]`, `constant_variance=True`, `closure=True`. `None` in the mean list means zero mean. With `closure=False`, only explicitly declared specifications can be selected, although a compound native restriction may connect them.

The comparison plan scores the same rows for every candidate. Its first scored row is the maximum declared mean conditioning span or the requested `hold_back`, whichever is larger. It preserves the input/regressor span, one parameter-independent variance seed, and zero pre-sample residuals. A constant-variance root uses its own `sigma2` on every row; a seeded zero-order dynamic model remains a distinct specification.

## General-to-specific procedure

`GeneralToSpecific` starts at certified, adequate roots. It tries supported native restrictions, fits children, tests the reductions, checks child adequacy, and backtests each cumulative reduction jointly against its original root. Accepted paths can branch. Fits are memoized by native specification identity. A warm start does not hold retained parameters fixed.

| Option | Default |
| --- | --- |
| `reduction_level` | `0.01` |
| `encompassing_level` | `0.01` |
| `diagnostic_level` | `0.01` |
| `prefer` | `"fewest_parameters"` |
| `tie_break` | `InformationCriterion(name="bic")` |
| `max_fits` | `96` optimizer invocations, including required deployment reserve |
| `max_mean_order` | `5`, configurable from 1 through 8 |

Terminals rank first by free parameter count, then the chosen tie-break IC, default BIC. `prefer="information_criterion"` ranks on that IC alone. Independent fits use a thread pool with deterministic node ordering. `n_jobs=None` or `-1` selects up to four workers. `method="qmle"` raises for GETS because the required LR calibration is unavailable.

Adequacy combines residual Ljung-Box, estimation-corrected absolute Li-Mak and absolute sign/size-bias checks with Holm's rule. It does not use in-sample PIT tests to certify a root or child. [Diagnostics](diagnostics.md#absolute-residual-adequacy-checks) gives formulas and assumptions.

## Automatic mean widening

When every diagnostically evaluated root rejects and each has a Holm-rejected residual Ljung-Box test, GETS asks each widenable mean template for one more AR lag. ARX retains its intercept and regressors. ARMA retains its MA order. An adequate or unresolved root prevents widening. Uncertified roots supply no diagnostic evidence. Volatility or density rejection alone does not trigger it.

Each wider root is refitted and counts against the fit budget. Its descendants share a new common comparison span; likelihoods across different spans are not compared. Widening stops at `max_mean_order`. `widenings` records order, triggering test, p-value, root evidence, counts and completion. Statuses can include `mean_order_cap_reached`, `mean_widening_unavailable` and `budget_exhausted`.

`Auto(max_mean_order=5).fit(y, x=None, **options)` constructs an ARX(1) mean with selectable intercept, crossed with GJR, EGARCH and APARCH(1,1) and SkewT, plus constant variance. It runs GETS with the chosen mean cap. Configure a different procedure through `Search`, rather than passing another `procedure` to `Auto.fit`.

## Restriction tests

For a parent likelihood $\ell_P$ and child likelihood $\ell_C$, the LR statistic is $2(\ell_P-\ell_C)$. Parent and child must be certified MLE fits on the same immutable comparison plan. A negative statistic requests parent refitting; it is not clamped to zero.

| Native restriction | Calibration/limit |
| --- | --- |
| Remove intercept, last AR/MA/exogenous coefficient, or one HAR window | Interior equality when identification holds |
| GARCH last shock or variance lag | Closed nonnegative face |
| GJR last shock pair | Two nonnegative sign-loadings, `alpha` and `alpha+gamma` |
| GJR last variance lag | Nonnegative face |
| All GJR gamma coefficients zero, child GARCH | Interior equality when both common sign-loadings are positive |
| EGARCH last alpha/gamma pair or variance lag | Interior equality; no-shock variance-lag nuisance can be unidentified |
| APARCH last shock/asymmetry pair | Listed, but asymmetry is unidentified under the null; no ordinary LR law |
| APARCH last variance lag | Nonnegative face when identified |
| SkewT `lam=0` | StudentT child, interior equality |
| StudentT `eta=0` | Normal child, closed nonnegative face |
| GED `nu=2` | Normal child, interior equality |

Interior equalities use chi-square. Supported closed faces use chi-bar-square after profiling signed directions and nuisance. One through three boundary weights use closed forms. Larger blocks use deterministic orthant integration, factoring independent blocks. The requested absolute probability error is `1e-6`; a failed error bound or decision interval crossing the requested level yields `unresolved`. Enumeration is bounded to twelve boundary directions.

With retained boundary nuisance, the LR law is a difference of squared information-metric distances to parent and child tangent cones. It need not be a chi-bar mixture. Orthogonal blocks can reduce to simpler laws; other regular polyhedral cases use fixed-seed Gaussian projection simulation with explicit error intervals and draw budgets. The reported `cone_projection` error is a 99.99% Bernoulli Chernoff Monte Carlo interval radius at each prespecified look. The three looks have at least 99.97% simultaneous coverage by the union bound. These are numerical error statements, not finite-sample size guarantees for the LR test. See [Self and Liang (1987)](references.md#self-liang-1987) and [Andrews (2001)](references.md#andrews-2001).

Singular information, nonminimal ARMA/GARCH representations, unidentified nuisance and kink fits cannot authorize an ordinary calibrated reduction. The implementation uses parent row-score information at the embedded child for retained-boundary calibration. Tests record the law, weights or projection method, error, assumptions and failure reason.

## Information-criterion ranking

An IC procedure requires an explicit adequacy choice.

```python
from scivol import Search, GARCH, Normal, StudentT, InformationCriterion

comparison = Search(
    volatility=[GARCH()], density=[Normal(), StudentT()],
    constant_variance=False, closure=False,
)
# ranked = comparison.fit(y, procedure=InformationCriterion("bic", adequacy=False))
```

`adequacy=False` ranks certified fits without an adequacy claim. `adequacy=True` also requires the adequacy battery. Options are `name="aic" | "bic" | "hqic"`, `max_fits=96`, and `diagnostic_level=0.01`. For $k$ free parameters and $n$ scored rows,

$$AIC=-2\ell+2k,\qquad BIC=-2\ell+k\log n,\qquad HQIC=-2\ell+2k\log\log n.$$

The underlying criteria are [Akaike (1974)](references.md#akaike-1974), [Schwarz (1978)](references.md#schwarz-1978) and [Hannan and Quinn (1979)](references.md#hannan-quinn-1979). Only likelihoods on a common comparison plan enter this ranking.

## Predictive confidence sets

`ConfidenceSet` freezes the training procedure's terminal specifications, then compares chronological one-step holdout losses. The holdout must have at least 500 observations.

```python
from scivol import GARCH, Normal, StudentT, Search, InformationCriterion, ConfidenceSet

source = GARCH() + StudentT()
y = source.simulate(2600, [0.1, 0.15, 0.7, 0.15], seed=62).data
search = Search(volatility=[GARCH()], density=[Normal(), StudentT()],
                constant_variance=False, closure=False)
selected = search.fit(
    y, procedure=ConfidenceSet(
        reduction=InformationCriterion("bic", adequacy=False),
        holdout=0.2, confidence=0.90, loss="log_score", refit_every=100,
        bootstrap_draws=4999, seed=44,
    ), n_jobs=1,
)
print(selected.summary())
```

`loss="log_score"` uses negative predictive log density. `loss="qlike"` uses $\log h_t+e_t^2/h_t$. Expanding refits use observations available at each forecast origin; within each block, realized observations update the filter before subsequent one-step losses. The training variance seed stays fixed. A predictive refit must certify or the workflow reports `predictive_fit_failed`.

The native [Hansen-Lunde-Nason (2011)](references.md#hansen-lunde-nason-2011) MCS uses the range statistic and a circular block bootstrap of jointly aligned loss rows. NumPy `Generator(PCG64(seed))` supplies uniforms. Resampling requires no additional fits. The block-length rule comes from `mcs_block_length`; sensitivity to that rule remains the caller's responsibility.

The complete confidence set is returned. Deployment chooses its smallest model, breaking ties with training BIC, then refits on all data. That refit provides no new holdout performance estimate. Coverage concerns the best member of the frozen candidate set under MCS assumptions, not all possible models.

## Result inspection

`SelectionResult.selected` is an ordinary full-sample `EstimationResult` or `None`. `selected_model` is its specification. An identical comparison-plan fit can be reused; otherwise deployment uses an explicit full-sample fit. A failed deployment certificate reports `full_sample_refit_uncertified`.

`terminals`, `paths`, `tests`, `candidates`, `confidence_set`, `predictive`, `widenings`, `status`, `search_complete`, `frontier` and `fit_counts` retain the decisions and remaining work. `summary()` returns text. `table()` returns a pandas DataFrame when available, otherwise a list of records. When no eligible fit is found, `selected` remains `None` and status can be `no_adequate_model` or `adequacy_unresolved`.

On budget exhaustion, `search_complete=False` identifies an unexplored frontier and the best eligible fit found so far can still be returned. Counts distinguish optimizer calls, unique fits, exact child reuse, warm starts, parent refits, widening and full-sample refits. A count of optimizer invocations is not a count of individual candidate-start runs.

## Limitations

The defaults and the adaptive procedure have not been calibrated. Missing graph edges include APARCH fixed-power/asymmetry, skew-Gaussian and finite GH restrictions. Cross-terminal union tests and equivalence deduplication beyond native specification identity are absent. ARMA order searches require a minimal nonsingular representation and are opt-in.

The selection procedures do not cover DCC, structural breaks, long memory, infinite-variance models, GH closure limits, unidentified no-shock hypotheses, singular simultaneous reductions or post-selection confidence intervals. A budget limits parsimony claims to the explored graph. An unresolved edge is not an accepted restriction.
