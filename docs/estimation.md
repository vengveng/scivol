# Estimation

## The likelihood

For a model with parameters $\theta = [\text{mean} \mid \text{volatility} \mid \psi]$, where $\psi$ are the density's shape parameters, scivol minimizes the full negative log-likelihood over the scored observations $\mathcal S$:

$$
F(\theta) = -\sum_{t \in \mathcal S} \left[ \log f_\psi\!\left(\frac{e_t}{\sqrt{h_t}}\right) - \tfrac12 \log h_t \right].
$$

Every normalization constant is included, so reported log-likelihoods and information criteria are comparable across models fitted to the same observations. The gradient and Hessian are analytical and exact; in a joint fit they include every cross derivative between the mean, volatility and shape parameters.

Different libraries start their variance recursions differently, so their log-likelihoods for the same model and data generally differ by an amount unrelated to fit quality; the [benchmark page](benchmarks.md#how-likelihoods-are-compared) explains how it compares them.

## Fit options

```text
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
| `boundary` | `"withhold"` | Standard errors at an estimate where closed bounds or constraints bind: `"withhold"` or `"restricted"`; see [inference](inference.md#standard-errors-at-a-boundary) |

Unknown options raise a `TypeError`.

## Maximum likelihood and Gaussian QMLE

`method="mle"` estimates all parameters jointly under the chosen density. At an interior estimate, the covariance is the inverse observed information, and a sandwich (robust) covariance is reported alongside it.

`method="qmle"` estimates the mean and volatility parameters by Gaussian quasi-maximum likelihood ([Bollerslev and Wooldridge, 1992](references.md#bollerslev-wooldridge-1992)), which remains consistent when the innovation density is misspecified, provided the conditional mean and variance are correctly specified. With the Normal density, that is the whole fit, and the covariance is the sandwich. With a shaped density, the shape parameters are then estimated by maximum likelihood with the mean and volatility parameters held at their Gaussian estimates, and the covariance is the two-step sandwich of the stacked score equations. This is a staged estimator, labeled `"QMLE"` in results, not a joint fit of the final density. EGARCH centres $|z|$ with the density's $E|z|$, so its Gaussian stage fits the same model with $\omega$ shifted by $\sum_i\alpha_i(E|z|-\sqrt{2/\pi})$. The shape stage keeps the Gaussian variance path, and the reported $\omega$ carries the shift.

## Coordinate paths

Most parameters have a restricted domain: variances must be positive, GARCH persistence below one, and so on. scivol can search either directly in these original parameters ("theta") or in unconstrained coordinates z that map smoothly onto the domain. The map changes only the coordinates; the likelihood and its domain are the same, and derivatives in z include the curvature of the map.

| `coordinates` | What runs |
| --- | --- |
| `"automatic"` | A trust-region Newton method with the exact Hessian in z, then a Newton refinement and the first-order check in theta. Where the z map does not cover the whole domain (HAR means), the theta path is used. |
| `"theta"` | A trust-region method in the original parameters with the domain's bounds and constraints, then the theta refinement. |
| `"z"` | SLSQP in z coordinates with the whole budget, without a theta refinement. |

The theta and z paths are independent formulations; scivol never runs one and reports it under the other's name. When they disagree on a data set, that is information: a local maximum, a boundary, or a numerically difficult likelihood.

A trust region that stalls, typically against a likelihood kink (see below), hands over to SLSQP, whose line search tolerates kinks. On the automatic path, a fit that fails its first-order check is restarted once in theta from the model's default start, unless its refinement has already shown the likelihood to be flat.

`fit_info.solver` and `fit_info.optimizer_phases` record exactly what ran. The trust-region method follows [Moré and Sorensen (1983)](references.md#more-sorensen-1983) with the scaling of [Moré (1978)](references.md#more-1978); SLSQP is [Kraft's (1988)](references.md#kraft-1988) algorithm; the first-order check projects onto the domain by least-distance programming ([Lawson and Hanson, 1995](references.md#lawson-hanson-1995)).

## Multi-start

Volatility likelihoods often have more than one local maximum. On daily stock returns, GARCH-type likelihoods frequently have one at moderate persistence and a higher one close to integration, and the Gaussian-copula DCC likelihood often has two in its persistence parameter. A single starting point finds whichever maximum it happens to be near.

Each model therefore offers a few candidate starts in structurally different regions:

| Model | Candidate starts |
| --- | --- |
| GARCH, GJR-GARCH, APARCH | persistence 0.9 (the default start), 0.98 and 0.995, the latter two carried mostly by beta |
| EGARCH | beta 0.9, 0.98, 0.995 and −0.9 |
| DCC | $(a, b) = (0.05, 0.90)$ and $(0.02, 0.80)$ |
| Constant variance, ARCH | the default start only |

The `starts` option decides how they are used:

| `starts` | Behaviour |
| --- | --- |
| `"auto"` (default) | The first candidate runs in full. Each other candidate is probed for 5 iterations and finished only if the probe has already reached a higher likelihood than the best finished start. While the best finished estimate is on the persistence face, every candidate is finished. |
| `n` | The first `n` candidates each run in full; `starts=1` is a single start. |
| a list of parameter vectors | Each vector runs in full. |

Because the optimizer never lowers the likelihood, a finished start always ends at least as high as the start it beat. The reported estimate is the best finished start, preferring one that passes its first-order check. On 500 liquid stocks and ten models, the automatic race reduced the number of fits that stopped below the best known maximum from 114 to 38 out of 5,000, at 17% more total time ([benchmarks](benchmarks.md)). EGARCH's last candidate covers the negative half of its stable region for beta: with little volatility dynamics, as in a series of independent draws, the likelihood can have a separate maximum there, where the log variance alternates. On 40 simulated series of independent Normal draws, this candidate moved 5 of 40 AR(1)-EGARCH-skew-t fits to a higher maximum at negative beta; on daily stock returns it never won, and it costs EGARCH fits about 8% more time. A maximum in a region the candidates do not cover, or one whose probe does not yet look promising after five iterations, can still be missed.

On the persistence face, where persistence reaches the limit of the declared stationarity, the likelihood can have several separate maxima, and a five-iteration probe says little about them. On daily stock returns, the probe of the start that led to the higher maximum was still 5 to 60 below the finished value. So while the best finished estimate is on that face, the remaining candidates are all finished. On 400 daily stock-return series and 14 models, this cut the automatic fits that stopped below the best known maximum from 50 to 23 of 5,600, at 25% more total time, and no fit ended lower.

`initial_params` is a single explicit start and replaces the candidates; it cannot be combined with another `starts` choice. A previous result must come from the same model. Starts are in the units of the data as fitted, including any `scale`.

```python
from scivol import GARCH, StudentT

spec = GARCH(1, 1) + StudentT()
y = spec.simulate(2000, [0.05, 0.08, 0.9, 0.1], seed=4).data

first = spec.fit(y, starts=1)
warm = spec.fit(y, initial_params=first)
every = spec.fit(y, starts=3)
mine = spec.fit(y, starts=[first.params, [0.05, 0.05, 0.9, 0.1]])
print(every.fit_info.starts)
```

`fit_info.starts` lists every start with its starting point (`theta`), whether it is a model candidate, whether it was finished, the log-likelihood it reached, whether it passed the first-order check (`success`), and whether it was selected.

## What `success` means

`result.success` is `True` when the estimate is finite and in the model's domain, the optimizer converged, and the first-order (Karush-Kuhn-Tucker) conditions hold: the gradient, projected onto the domain at any active bound or constraint, is below the tolerance

$$\tau = \max\{\sqrt{\text{accuracy}}, \sqrt{\epsilon_{\text{machine}}}\} = 10^{-5} \text{ by default}.$$

The check is made in the original parameters on the per-observation objective $F/n$ of the internally scaled data, which is why `fit_info.kkt_residual` is not the raw gradient in your units. A successful fit is therefore a local maximum over the closure of the declared domain.

An estimate can sit on a boundary that the declared domain excludes, for example GARCH persistence exactly one under the default covariance-stationary domain, or a Student-t with $\eta \to 1/2$. Such a fit can still be the maximum over the closure of the domain, and scivol reports it as found. In the example below the variance level of the data shifts halfway through the sample, a common way for GARCH persistence to run to one:

```python
import numpy as np
from scivol import GARCH

rng = np.random.default_rng(3)
y = np.concatenate([rng.standard_normal(1000), 2.5 * rng.standard_normal(1000)])
result = GARCH(1, 1).fit(y)
print(result.success, result.fit_info.inference_status, result.fit_info.excluded_boundary)
print(result.warnings)
```

```text
True at_excluded_boundary ('persistence < 1',)
('the estimate is at the excluded boundary of the declared domain (persistence < 1); standard errors are not reported; to estimate on it, declare stationarity="integrated" (persistence <= 1) and fit with boundary="restricted"', 'persistence is very close to one')
```

The fit succeeds, `fit_info.excluded_boundary` names the condition the estimate reaches, and standard errors are not reported, since the estimate is not an interior point of the model. If persistence one belongs in your model, declare it on the volatility component: `GARCH(1, 1, stationarity="integrated")` fits the same likelihood with persistence one inside the domain and reports this estimate with status `at_boundary`. `stationarity="none"` drops the condition altogether. [Volatility models](models/volatility.md#declaring-stationarity) describes the three declarations. Along the persistence-one face the likelihood can have several separate maxima, and the paths can end at different ones: on daily stock returns, GJR-GARCH fits with a skewed t on that face differed by up to 140 in log-likelihood between the automatic and theta paths, whether or not the face was declared. A fit on the face therefore finishes all its candidate starts ([multi-start](#multi-start)), which reaches most of these maxima, though not all. Refitting with `coordinates="theta"` and comparing the likelihoods is a cheap check.

A likelihood can also keep rising toward a limit outside the parameter space without any single parameter reaching a bound, as when a generalized hyperbolic density approaches its skewed-t limit ($|\rho| \to 1$ and $\zeta \to 0$ together). An optimizer phase that has not met the first-order conditions stops once its objective has changed by no more than the solver accuracy over ten iterations, and reports `stalled` in `fit_info.optimizer_phases`. The estimate is then returned with `success=False` without spending the rest of the iteration budget.

Whenever `success` is `False`, `result.warnings` explains what failed. If no finite estimate in the domain exists at all, the fit raises `scivol.FitError`, whose message gives the reasons. When many observations are exactly zero, the message says so, since exact zeros are the usual cause of likelihoods without a maximum on real returns.

## Data scaling and hold-back

Internally, scivol divides the data by the square root of its mean square before optimizing and maps the parameters and likelihood back afterwards; this changes nothing about the model, and `fit_info.internal_data_scale` records the factor.

The `scale` option is different: `scale=100` fits `100 * y`, and all parameters, paths and forecasts are then in those units.

The likelihood scores observations from `max(hold_back, mean lags)` onward. Earlier observations supply lags for the mean but do not enter the variance recursion. Giving several models the same `hold_back` makes them score the same observations, which is needed to compare their likelihoods; [model selection](selection.md) does this automatically.

## Likelihood kinks

With an estimated mean, a residual can pass through exactly zero as the mean parameters change. EGARCH's $|z|$ term then makes the likelihood non-differentiable at that point, and APARCH with $\delta \le 1$ or a GED with $\nu \le 1$ create sharper cusps. scivol does not smooth these. Its theta refinement identifies the residuals sitting exactly at zero, optimizes on that face, and checks the first-order conditions there, following the active-manifold approach of [Lewis (2002)](references.md#lewis-2002). A kink point that passes this check is reported with `fit_info.kink_point = True`, the affected observations in `fit_info.active_kink_rows`, and inference status `at_kink`; the ordinary gradient, Hessian and standard errors do not exist there and are not reported. For APARCH with $\delta \le 1$ and GED with $\nu \le 1$ the one-sided slopes are unbounded, so a fit that ends at such a cusp reports `success=False`. The z path has no theta refinement and so does not identify kink points: a z fit that ends at one fails its first-order check. On 400 daily stock-return series, this left 45% of HARX-EGARCH-t fits uncertified on the z path, against 9% on the automatic path.
