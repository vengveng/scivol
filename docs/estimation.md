# Estimation

## The likelihood

For a model with parameters $\theta = [\text{mean} \mid \text{volatility} \mid \psi]$, where $\psi$ are the density's shape parameters, scivol minimizes the full negative log-likelihood over the scored observations $\mathcal S$:

$$
F(\theta) = -\sum_{t \in \mathcal S} \left[ \log f_\psi\!\left(\frac{e_t}{\sqrt{h_t}}\right) - \tfrac12 \log h_t \right].
$$

Every normalization constant is included, so reported log-likelihoods and information criteria are comparable across models fitted to the same observations. The gradient and Hessian are analytical and exact; in a joint fit they include every cross derivative between the mean, volatility and shape parameters.

Different libraries start their variance recursions differently, so their log-likelihoods for the same model and data generally differ by an amount unrelated to fit quality; the [benchmark page](benchmarks.md#how-likelihoods-are-compared) explains how it compares them.

## Fit options

```python
result = spec.fit(y, method="mle", coordinates="automatic", starts="auto", maxiter=1000, accuracy=1e-10)
```

| Option | Default | Meaning |
| --- | --- | --- |
| `method` | `"mle"` | `"mle"` (maximum likelihood) or `"qmle"` (Gaussian quasi-maximum likelihood) |
| `coordinates` | `"automatic"` | Which formulation the optimizer works in: `"automatic"`, `"z"` or `"theta"` |
| `starts` | `"auto"` | Starting points: `"auto"`, a number of model candidates, or explicit parameter vectors |
| `initial_params` | `None` | A single starting point: a parameter vector or a previous result for the same model |
| `maxiter` | `1000` | Iteration budget for each start |
| `accuracy` | `1e-10` | Solver accuracy; also sets the tolerance of the first-order check |
| `x` | `None` | Regressors for ARX and HARX means, one row per observation |
| `hold_back` | `0` | Number of initial observations used only as lags, not scored |
| `scale` | `None` | A multiplier applied to the data before fitting, such as 100 for percent returns |

Unknown options raise a `TypeError`.

## Maximum likelihood and Gaussian QMLE

`method="mle"` estimates all parameters jointly under the chosen density. At an interior estimate, the covariance is the inverse observed information, and a sandwich (robust) covariance is reported alongside it.

`method="qmle"` estimates the mean and volatility parameters by Gaussian quasi-maximum likelihood ([Bollerslev and Wooldridge, 1992](references.md#bollerslev-wooldridge-1992)), which remains consistent when the innovation density is misspecified, provided the conditional mean and variance are correctly specified. With the Normal density, that is the whole fit, and the covariance is the sandwich. With a shaped density, the shape parameters are then estimated by maximum likelihood with the mean and volatility parameters held at their Gaussian estimates, and the covariance is the two-step sandwich of the stacked score equations. This is a staged estimator, labeled `"QMLE"` in results, not a joint fit of the final density. A staged fit with a shaped density is not available for EGARCH, whose variance recursion depends on the density through $E|z|$.

## Coordinate paths

Most parameters have a restricted domain: variances must be positive, GARCH persistence below one, and so on. scivol can search either directly in these original parameters ("theta") or in unconstrained coordinates z that map smoothly onto the domain. The map changes only the coordinates; the likelihood and its domain are the same, and derivatives in z include the curvature of the map.

| `coordinates` | What runs |
| --- | --- |
| `"automatic"` | A trust-region Newton method with the exact Hessian in z, then a Newton refinement and the first-order check in theta. Where the z map does not cover the whole domain (HAR means), the theta path is used. |
| `"theta"` | A trust-region method in the original parameters with the domain's bounds and constraints, then the theta refinement. |
| `"z"` | SLSQP in z coordinates with the whole budget, without a theta refinement. |

The theta and z paths are independent formulations; scivol never runs one and reports it under the other's name. When they disagree on a data set, that is information: a local maximum, a boundary, or a numerically difficult likelihood.

A trust region that stalls, typically against a likelihood kink (see below), hands over to SLSQP, whose line search tolerates kinks. On the automatic path, an uncertified fit is restarted once in theta from the model's default start, unless its refinement has already shown the likelihood to be flat.

`fit_info.solver` and `fit_info.optimizer_phases` record exactly what ran. The trust-region method follows [Moré and Sorensen (1983)](references.md#more-sorensen-1983) with the scaling of [Moré (1978)](references.md#more-1978); SLSQP is [Kraft's (1988)](references.md#kraft-1988) algorithm; the first-order check projects onto the domain by least-distance programming ([Lawson and Hanson, 1995](references.md#lawson-hanson-1995)).

## Multi-start

Volatility likelihoods often have more than one local maximum. On daily stock returns, GARCH-type likelihoods frequently have one at moderate persistence and a higher one close to integration, and the Gaussian-copula DCC likelihood often has two in its persistence parameter. A single starting point finds whichever maximum it happens to be near.

Each model therefore offers a few candidate starts in structurally different regions:

| Model | Candidate starts |
| --- | --- |
| GARCH, GJR-GARCH, APARCH | persistence 0.9 (the default start), 0.98 and 0.995, the latter two carried mostly by beta |
| EGARCH | beta 0.9, 0.98 and 0.995 |
| DCC | $(a, b) = (0.05, 0.90)$ and $(0.02, 0.80)$ |
| Constant variance, ARCH | the default start only |

The `starts` option decides how they are used:

| `starts` | Behaviour |
| --- | --- |
| `"auto"` (default) | The first candidate runs in full. Each other candidate is probed for 5 iterations and finished only if the probe has already reached a higher likelihood than the best finished start. |
| `n` | The first `n` candidates each run in full; `starts=1` is a single start. |
| a list of parameter vectors | Each vector runs in full. |

Because the optimizer never lowers the likelihood, a finished start always ends at least as high as the start it beat. The reported estimate is the best finished start, preferring a certified one. On 500 liquid stocks and ten models, the automatic race reduced the number of fits that stopped below the best known maximum from 114 to 38 out of 5,000, at 17% more total time ([benchmarks](benchmarks.md)). It does not guarantee the global maximum: a region the candidates do not cover, or a start whose probe does not yet look promising, can still be missed.

`initial_params` is a single explicit start and replaces the candidates; it cannot be combined with another `starts` choice. A previous result must come from the same model. Starts are in the units of the data as fitted, including any `scale`.

```python
first = spec.fit(y, starts=1)
warm = spec.fit(y, initial_params=first)
every = spec.fit(y, starts=3)
mine = spec.fit(y, starts=[first.params, [0.05, 0.05, 0.9, 0.1]])
print(every.fit_info.starts)
```

`fit_info.starts` lists every start with its starting point (`theta`), whether it is a model candidate, whether it was finished, the log-likelihood it reached, whether it was certified, and whether it was selected.

## What `success` means

`result.success` is `True` when the estimate is finite and in the model's domain, the optimizer converged, and the first-order (Karush-Kuhn-Tucker) conditions hold: the gradient, projected onto the domain at any active bound or constraint, is below the tolerance

$$\tau = \max\{\sqrt{\text{accuracy}}, \sqrt{\epsilon_{\text{machine}}}\} = 10^{-5} \text{ by default}.$$

The check is made in the original parameters on the per-observation objective $F/n$ of the internally scaled data, which is why `fit_info.kkt_residual` is not the raw gradient in your units. A certified estimate is a local maximum over the closure of the declared domain; it is not a proof of global optimality.

An estimate can sit on a boundary that the declared domain excludes, for example GARCH persistence exactly one under the default covariance-stationary domain, or a Student-t with $\eta \to 1/2$. Such a fit can still be certified as the maximum over the closure; scivol reports it with `fit_info.inference_status = "at_excluded_boundary"` and names the conditions in `fit_info.excluded_boundary`, for example `("persistence < 1",)`. Standard errors are withheld there. If you want persistence one to be part of the model, declare it on the volatility component with `stationarity="integrated"` (or `"none"` to drop the condition); see [volatility models](models/volatility.md#declaring-stationarity).

A likelihood can also keep rising toward a limit outside the parameter space without any single parameter reaching a bound, as when a generalized hyperbolic density approaches its skewed-t limit ($|\rho| \to 1$ and $\zeta \to 0$ together). An optimizer phase that is not certified stops once its objective has changed by no more than the solver accuracy over ten iterations, reports `stalled`, and returns the estimate uncertified rather than after the whole budget.

An uncertified fit is returned with warnings that explain what failed. If no finite estimate in the domain exists at all, the fit raises `scivol.FitError`, whose message gives the reasons; if many observations are exactly zero, it says so, since exact zeros are the usual cause of likelihoods without a maximum on real returns.

## Data scaling and hold-back

Internally, scivol divides the data by the square root of its mean square before optimizing and maps the parameters and likelihood back afterwards; this changes nothing about the model, and `fit_info.internal_data_scale` records the factor.

The `scale` option is different: `scale=100` fits `100 * y`, and all parameters, paths and forecasts are then in those units.

The likelihood scores observations from `max(hold_back, mean lags)` onward. Earlier observations supply lags for the mean but do not enter the variance recursion. Giving several models the same `hold_back` makes them score the same observations, which is needed to compare their likelihoods; [model selection](selection.md) does this automatically.

## Likelihood kinks

With an estimated mean, a residual can pass through exactly zero as the mean parameters change. EGARCH's $|z|$ term then makes the likelihood non-differentiable at that point, and APARCH with $\delta \le 1$ or a GED with $\nu \le 1$ create sharper cusps. scivol does not smooth these. Its theta refinement identifies the residuals sitting exactly at zero, optimizes on that face, and checks the first-order conditions there, following the active-manifold approach of [Lewis (2002)](references.md#lewis-2002). A certified kink is reported with `fit_info.kink_point = True`, the affected observations in `fit_info.active_kink_rows`, and inference status `at_kink`; the ordinary gradient, Hessian and standard errors are withheld because they do not exist there. For APARCH with $\delta \le 1$ and GED with $\nu \le 1$, the one-sided slopes are unbounded and no finite certificate is available.
