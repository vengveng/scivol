# Estimation, inference, and uncertainty

## Bottom line

scivol already has an unusually strong computational base: constrained and transformed optimizers remain independent, analytical scores and Hessians are checked against AD, fit results identify the effective path, and QMLE exposes a sandwich covariance. The largest remaining gaps are statistical contracts around that machinery.

The most valuable additions are not new optimizers. They are methods that tell a user when familiar optimizer output and normal standard errors stop being trustworthy.

1. **Make inference assumptions explicit in every result.** A covariance matrix should identify the estimating objective, bread, meat, effective sample, parameter space, moment requirements, and any boundary or rank problem. “Robust” must never mean “valid under anything.”
2. **Add boundary-aware inference.** A zero ARCH, GARCH, leverage, or spillover coefficient has a cone-projection limit, not an ordinary normal limit. Standard Wald, score, and likelihood-ratio reference distributions are wrong there. Shrinkage-bootstrap LR inference is the most general mature remedy.
3. **Add forecast uncertainty as a first-class workflow.** Point forecasts, expected future variance paths, predictive return distributions, and confidence intervals for latent conditional variance are different objects. The API should keep parameter uncertainty and future-shock uncertainty separate.
4. **Add bootstrap inference with named validity domains.** Recursive residual bootstrap is the useful general baseline under iid innovations; fixed-design bootstrap is attractive for some one-step risk estimators. Neither is universally valid. Heavy tails, boundary parameters, and dependent higher moments each need a different scheme.
5. **Treat missing and irregular observations as a model, not a cleaning option.** Dropping gaps and joining the remaining values changes the recursion. Recent likelihood and indirect-inference results show that several convenient missing-data treatments are inconsistent.
6. **Expose finite-sample initialization and optimizer evidence.** Presample variance is asymptotically negligible under standard conditions but can matter sharply in short, persistent samples. A solver success flag is not enough evidence of a maximum.
7. **Keep native Bayesian inference behind the frequentist correctness work.** Bayesian GARCH is mature and useful for nonlinear uncertainty and predictive densities, but it adds prior design, sampling diagnostics, and a second inferential contract. First expose a stable log-density and transform interface for external samplers; add a native sampler only when scivol can test it as rigorously as its MLE paths.

## Priority matrix

| Capability | Statistical maturity | Product value | Engineering cost | Recommendation |
| --- | --- | --- | --- | --- |
| Covariance provenance, validity flags, rank and conditioning diagnostics | High | Very high | Low | P0 |
| Boundary-aware one-parameter and joint tests | High | Very high | Medium | P0 |
| Delta-method one-step forecast covariance | High | High | Low/medium | P0 |
| Recursive residual bootstrap for parameters and forecasts under iid innovations | High | High | Medium | P0/P1 |
| Moment-existence test and heavy-tail inference warning | High | High | Medium | P0/P1 |
| Profile likelihood for smooth interior parameters and derived functions | High | High | Medium | P1 |
| Initialization policy and sensitivity report | High | Medium/high | Low | P1 |
| Missing-data SMC for GARCH(1,1) | High but specialized | Medium | High | P1/P2 |
| Bootstrap LR with shrinkage for unknown nuisance parameters on boundaries | High | Medium/high | High | P1 |
| Recursive block bootstrap for dependent higher moments | New, 2026 | Medium | High | Experimental P2 |
| Heavy-tail robust M-estimation or tail-trimmed QML | Established, less standardized in software | Medium/high | High | P2 |
| Native Bayesian SMC or HMC | High | Medium | High | P2 after external interface |
| Generic irregular-time or COGARCH likelihood | Established niche | Medium/low for current scope | Very high | P3 |

## 1. What a statistically honest fit result should say

White's misspecified-likelihood theory establishes the general sandwich principle, while Bollerslev and Wooldridge specialize QML inference to dynamic mean and covariance models. Their result is powerful but conditional: if the parameterized conditional moments are correct, Gaussian QMLE can remain consistent under a nonnormal conditional density, and the sandwich covariance replaces inverse information ([White, 1982](https://doi.org/10.2307/1912526); [Bollerslev and Wooldridge, 1992](https://doi.org/10.1080/07474939208800229)). Lee and Hansen provide the foundational GARCH(1,1) QMLE theory under martingale-difference innovations and allow integrated and mildly explosive cases under their assumptions ([Lee and Hansen, 1994](https://doi.org/10.1017/S0266466600008215)).

Those results do not justify a single unlabeled `std_errors` array. Every covariance should carry at least:

- `estimator`: full MLE, Gaussian QMLE, two-step QMLE, or another estimating equation;
- `cov_type`: inverse observed Hessian, expected information, OPG, sandwich, bootstrap, or posterior;
- `parameter_space`: constrained theta space or transformed z space, with covariance mapped to reported parameters;
- `n_eff` and the exact likelihood start index;
- `score_definition`: per-observation score contributions and whether cross-time score covariance is assumed zero;
- `validity`: correct density, correct conditional first two moments, iid standardized innovations, finite fourth moment, interior parameters, or another stated condition;
- `rank`, condition number, smallest eigenvalue, and any regularization used;
- `boundary_parameters` and `near_integrated` status;
- the covariance of any two-step block, including whether cross-step covariance terms were estimated or deliberately omitted.

The result summary should state the inferential target. Under misspecification, an MLE targets a pseudo-true parameter. A sandwich covariance describes repeated-sample uncertainty around that pseudo-true value; it does not make the fitted density true. Ordinary LR tests also cease to have their usual reference distribution under misspecification, even when sandwich Wald inference remains available ([White, 1982](https://doi.org/10.2307/1912526)).

### Heavy tails are a separate failure mode

“QMLE robust” and “heavy-tail robust” are not synonyms. Hall and Yao show that Gaussian QMLE can have a nonnormal, slow-rate limit when innovations are sufficiently heavy-tailed; ordinary parametric bootstrap also fails in this case, while a studentized subsample bootstrap can work ([Hall and Yao, 2003](https://doi.org/10.1111/1468-0262.00396)). Hill develops tail-trimmed QML and redescending estimating equations that recover asymptotic normality under weaker tail conditions ([Hill, 2015](https://doi.org/10.3150/14-BEJ616)). Hallin, Liu, and Mukherjee give a more recent class of GARCH M-estimators and valid weighted bootstraps under weaker higher-moment assumptions ([Hallin, Liu, and Mukherjee, 2023](https://doi.org/10.1007/978-981-99-0803-5_8)).

scivol should therefore:

- warn when a fitted innovation law implies no fourth moment, for example Student-t with `nu <= 4`, and say which reported normal or sandwich intervals lose their standard justification;
- provide a formal moment-existence workflow rather than infer validity only from the fitted density. Francq and Zakoian jointly account for uncertainty in volatility parameters and empirical innovation moments and add a bootstrap refinement ([Francq and Zakoian, 2022](https://doi.org/10.1016/j.jeconom.2020.05.009));
- keep a future heavy-tail estimator under a distinct `method`, such as `method="trimmed_qmle"` or `method="m_estimation"`. Do not silently alter the score of `method="qmle"`;
- suppress default z-tests when the covariance's declared assumptions fail. Returning a number with a warning is too easy to misuse in automated workflows.

## 2. Boundary and near-boundary inference

Positivity restrictions place zero ARCH and GARCH coefficients on the boundary of the parameter space. At such points the QMLE converges to the projection of a normal vector onto a convex cone, so ordinary symmetric confidence intervals and chi-square tests are invalid ([Francq and Zakoian, 2007](https://doi.org/10.1016/j.spa.2007.01.001)). Francq and Zakoian derive corrected Wald, score, and QLR tests for zero GARCH coefficients, including the no-ARCH null ([Francq and Zakoian, 2009](https://doi.org/10.1198/jasa.2009.0117)). The same issue matters in multivariate models when a zero spillover coefficient lies on a boundary ([Pedersen and Rahbek, 2017](https://doi.org/10.1016/j.jeconom.2016.09.004)).

This has immediate consequences for scivol.

- A log transform that maps coefficients to strictly positive values is excellent for optimization but cannot represent the null coefficient exactly. Boundary tests need a constrained theta-space evaluation or a dedicated restricted fit.
- Model-order selection cannot use ordinary parameter t-tests to delete lags. A zero extra lag is precisely the nonregular case.
- A `boundary_hugging` warning is diagnostic evidence, not a valid p-value correction.
- Symmetric Wald intervals that extend below zero should not be clipped. Clipping changes coverage without fixing the reference law.

Cavaliere, Nielsen, Pedersen, and Rahbek propose a shrinkage bootstrap for LR tests when both tested and unknown nuisance parameters may be on the boundary. It consistently reproduces the cone-dependent null distribution and has good finite-sample behavior in ARCH applications ([Cavaliere et al., 2022](https://doi.org/10.1016/j.jeconom.2020.05.006)). This should anchor a general implementation:

```python
test = result.test_boundary(
    {"alpha[2]": 0.0},
    method="shrinkage_bootstrap_lr",
    n_bootstrap=1999,
    random_state=7,
)
```

The returned object should include the restricted and unrestricted fits, bootstrap shrinkage rate, empirical critical values, Monte Carlo standard error of the p-value, failed-replication count, and the exact parameter cone. For common one-coefficient cases, a fast analytical chi-bar-square implementation is worthwhile, but the bootstrap should remain the reference for joint restrictions and nuisance boundaries.

Near integration is related but different. A persistence estimate near one can make Wald intervals highly asymmetric and initialization slow to decay. scivol should report strict-stationarity and finite-moment conditions as derived functions with uncertainty, not merely `sum(alpha) + sum(beta)`. Francq and Zakoian give strict-stationarity inference that covers stationary and explosive GARCH(1,1) without imposing the stationarity constraint during estimation ([Francq and Zakoian, 2012](https://doi.org/10.3982/ECTA9405)).

## 3. Profile likelihood

Profile likelihood is mature and especially useful when a persistence, leverage, tail-shape, half-life, or derived risk parameter has an asymmetric likelihood or weak local identification. It fixes the scalar target and reoptimizes every nuisance parameter. Unlike a Wald interval, the result is invariant to a smooth reparameterization and reveals flat or disconnected support regions. Modern trust-region algorithms can trace profile endpoints more reliably than a fixed grid ([Fischer and Lewis, 2021](https://doi.org/10.1007/s11222-021-10012-y)).

Recommended API:

```python
profile = result.profile(
    "persistence",
    level=0.95,
    calibration="likelihood_ratio",
)
profile.interval
profile.plot()
```

The target should accept a named parameter or a differentiable scalar function of theta. Profiles of persistence and half-life are more informative than separate profiles for highly correlated alpha and beta.

There are two hard limits.

1. Ordinary Wilks calibration assumes a correctly specified likelihood and an interior regular point. Under QMLE misspecification, use a robust LR construction or bootstrap calibration; a sandwich covariance does not repair an ordinary profile LR cutoff. A robust LR test specific to GARCH illustrates this distinction ([Busch, 2005](https://doi.org/10.1016/j.econlet.2005.03.006)).
2. At a boundary, use cone-aware or bootstrap critical values. A profile plot remains diagnostically useful, but a naive chi-square cutoff does not become valid merely because the interval is asymmetric.

Profile likelihood is a P1 feature. It should follow covariance provenance and boundary tests because those features determine which profile calibration is legal.

## 4. Bootstrap methods need named assumptions

There is no single “GARCH bootstrap.” The resampling design must match the inferential problem.

### Parameters and conditional variance under iid innovations

A recursive residual bootstrap resamples standardized residuals, generates a new return and variance path recursively, refits the model, and recomputes the statistic. It naturally includes parameter uncertainty and is the best general baseline for iid innovations. Pascual, Romo, and Ruiz use this design to obtain prediction densities and intervals for both returns and volatility ([Pascual, Romo, and Ruiz, 2006](https://doi.org/10.1016/j.csda.2004.12.008)). Residual-based GARCH bootstrap can also deliver a second-order asymptotic refinement under its regularity conditions ([Jeong, 2017](https://doi.org/10.1017/S0266466616000104)).

The implementation should offer percentile, basic, and studentized intervals, but choose defaults per target using coverage evidence. Store the bootstrap distribution and failed fits. Refit failures are part of the answer, especially near a boundary.

### Conditional VaR

Beutner, Heinemann, and Smeekes establish a fixed-design residual bootstrap for a two-step conditional VaR estimator across a general class of volatility models. Their simulations favor reversed-tail intervals over equal-tailed percentile intervals for coverage ([Beutner, Heinemann, and Smeekes, 2024](https://doi.org/10.1016/j.jeconom.2023.105554)). This is mature and actionable for one-step risk intervals, but its validity domain should appear in the method name and result metadata.

### Dependent innovations

A 2026 paper shows that a fixed-design residual bootstrap is inconsistent when standardized innovations have dependent higher moments, even if blocks rather than individual residuals are resampled. The proposed remedy is a recursive-design residual block bootstrap that preserves dependence between the generated volatility path and innovations ([Beutner, Schaumburg, and Spanjers, 2026](https://doi.org/10.1007/s10182-026-00560-x)). This is an important recent result, but it is too new for scivol to make it the only default. Add it as experimental after diagnostics can test residual dependence and after simulation coverage spans the shipped model families.

### Heavy tails and boundaries

Heavy tails call for subsampling or a robust estimator-specific bootstrap, not a routine residual bootstrap ([Hall and Yao, 2003](https://doi.org/10.1111/1468-0262.00396)). Boundary hypotheses call for the shrinkage bootstrap described above. The user should select the inferential target; scivol should select or reject incompatible resampling designs.

A clean entry point is:

```python
boot = result.bootstrap(
    target="params",              # params, volatility, forecast, var, es
    design="recursive_residual",  # fixed_residual, recursive_block, subsample
    n_bootstrap=1999,
    interval="studentized",
    random_state=7,
)
```

The object should record innovation centering and scaling, block-length rule, presample policy, refit method, parameter constraints, random seed, parallel backend, and p-value Monte Carlo error. Reproducibility across thread counts requires counter-based or per-replication random streams.

## 5. Missing and irregular observations

scivol currently rejects NaN input. That is safer than silently compacting a time series, but a complete library should distinguish four cases:

1. prices are absent on dates when the market is closed, and one model step intentionally means one trading interval;
2. a return is missing within an otherwise regular trading calendar;
3. assets trade nonsynchronously in a multivariate series;
4. observations arrive in genuinely irregular physical time.

Only the first case can often be handled by an explicit trading-time index without a missing-data model. In the other cases, `dropna()` changes lag spacing and therefore changes the statistical model.

Blasques, Gorgi, and Koopman show that common zero-update treatments of missing observations in observation-driven models can produce inconsistent inference. They propose indirect inference that reproduces the missingness pattern during simulation ([Blasques, Gorgi, and Koopman, 2021](https://doi.org/10.1016/j.jeconom.2020.07.043)). Wee, Chen, and Dunsmuir approximate the intractable observed-data likelihood for GARCH(1,1) with smooth sequential Monte Carlo and then optimize it. Their simulations report small bias and near-nominal interval coverage even with substantial missingness ([Wee, Chen, and Dunsmuir, 2023](https://doi.org/10.1214/23-AOAS1734)).

Recommended sequence:

- Add a `TimeIndexPolicy` before adding estimation. It should validate monotonicity, duplicates, timezone, calendar, aggregation, and the meaning of one recursion step.
- Continue rejecting interior missing returns by default. The error should suggest explicit policies rather than `dropna()`.
- First support `missing="smc"` only for GARCH(1,1) plus Normal or Student-t, with a passed boolean observation mask. Return particle count, effective sample size, likelihood Monte Carlo error, seed, and convergence across particle counts.
- Treat `missing="indirect"` as a separate estimator if implemented. Do not call it the same MLE path.
- For truly irregular physical time, use a model designed for elapsed time. ACD-GARCH and continuous-time GARCH are distinct models, not switches on discrete GARCH. The exact-discretization argument is described by Meddahi, Renault, and Werker ([Meddahi, Renault, and Werker, 2006](https://doi.org/10.1016/j.econlet.2005.07.027)); SMC likelihood for irregularly observed COGARCH is developed by Wee, Chen, and Dunsmuir ([Wee, Chen, and Dunsmuir, 2019](https://doi.org/10.1093/jjfinec/nby012)).

Native missing-data estimation is P1/P2 because it requires stochastic likelihood optimization and a new test harness. Calendar semantics and explicit rejection are P0 correctness work.

## 6. Initialization is part of the finite-sample model

A GARCH likelihood requires presample shocks and variances. Under contraction conditions, their effect vanishes asymptotically, but finite samples and near-integrated fits can retain it for a long time. This matters for comparisons with other libraries and for rolling windows.

scivol's current convention, including `h0 = mean(y^2)` and the model-specific likelihood start, should remain the default because changing it would silently change estimates and comparisons. The omission is configurability and evidence.

Recommended additions:

- `initial_variance={"sample", "backcast", "unconditional", float, array}`;
- explicit presample residual and mean-state options for linked ARMA models;
- `burn_in` or `likelihood_start`, separate from the recursive state start;
- a result record containing the actual initial state, data used to compute it, and whether it depends on theta;
- `result.initialization_sensitivity()` that reevaluates or refits a small set of justified policies and reports likelihood and parameter movement;
- exact reuse of the terminal filtered state in update and forecast workflows.

An unconditional-variance start is undefined or unstable at integration and can depend strongly on theta. It should never be an unconditional default. A backward exponentially weighted backcast is a defensible optional start. The sensitivity report matters most when persistence is high, the sample is short, or different public paths disagree.

## 7. Numerical optimization and derivative correctness

Fiorentini, Calzolari, and Panattoni show that analytical first and second derivatives materially improve GARCH estimation and study the finite-sample behavior of Hessian, information, OPG, and robust covariance estimators ([Fiorentini, Calzolari, and Panattoni, 1996](https://doi.org/10.1002/(SICI)1099-1255(199607)11:4%3C399::AID-JAE401%3E3.0.CO;2-R)). The likelihood can contain long, nearly flat regions in familiar coordinates, which explains why parameter transformations and starting values can change convergence in practice. Large-sample asymptotic convexity does not imply that every finite-sample optimizer run found the relevant maximum ([Ip et al., 2006](https://doi.org/10.1016/j.csda.2004.08.012)).

scivol's independent theta-space and z-space paths are therefore a scientific feature. Preserve them. Add the following completion criteria to every fit:

- finite objective, parameters, state path, score, and Hessian;
- exact constraint feasibility in reported theta space;
- small gradient norm in the optimized coordinates;
- KKT or projected-gradient residual for constrained theta-space fits;
- negative-definite likelihood Hessian, or positive-definite NLL Hessian, on the estimable tangent space;
- Hessian symmetry error, numerical rank, condition number, and minimum eigenvalue;
- objective agreement when reevaluated by an independent routine;
- deterministic multistart evidence for complex models, including all start objectives and exit states;
- matched-point equality between theta and z objectives and chain-rule derivatives, without forcing their optimized outputs to agree.

Do not silently replace an indefinite covariance with a pseudoinverse and then print ordinary p-values. A pseudoinverse can be useful for diagnosis, but the result should mark affected directions as weakly identified and suppress standard inference.

The AD oracle should remain the main development reference. Automatic differentiation evaluates derivatives of the executed program without finite-difference subtraction and can reach machine precision for smooth objectives ([Baydin et al., 2018](https://jmlr.org/papers/v18/17-468.html)). Production validation should cover:

- values, gradients, Hessians, and per-observation scores at randomized interior points;
- points close to positivity, stationarity, degrees-of-freedom, and skewness limits without evaluating undefined points;
- all order combinations, including orders beyond `(1, 1)` specializations;
- both constrained objectives and full transformed objectives `L(theta(z))`;
- score-sum equals total gradient, Hessian symmetry, and expected nesting identities;
- identical initialization, variance floors, scaling, and effective sample between C and oracle paths.

Complex-step differentiation is a useful third check for analytic, branch-free kernels, but it cannot replace AD when clipping, indicators, or nonanalytic absolute values enter the program ([Martins, Sturdza, and Alonso, 2003](https://doi.org/10.1145/838250.838251)). Finite differences should remain a coarse smoke test.

## 8. Forecast uncertainty

A point variance forecast hides several different sources of uncertainty:

- **parameter uncertainty:** theta was estimated from a finite sample;
- **future-shock uncertainty:** multi-step conditional variance depends on shocks that have not occurred;
- **innovation uncertainty:** future returns remain random even if the variance path were known;
- **model and density uncertainty:** another volatility recursion or innovation law may forecast differently;
- **state or measurement uncertainty:** relevant for missing data, realized measures, and latent-state models.

The API should not collapse these into one “confidence interval.”

### P0: delta-method uncertainty for one-step conditional objects

For a smooth scalar forecast `g(theta, state_n)`, compute

`Var(g(theta_hat)) = grad(g)' Cov(theta_hat) grad(g)`.

Analytical forecast sensitivities make this cheap. Return intervals for variance, volatility, quantiles, VaR, and derived persistence on a transformed scale that respects support. Label these as parameter-confidence intervals conditional on the observed terminal state. They do not describe the distribution of tomorrow's return.

### P1: recursive predictive simulation and bootstrap

For horizons above one, simulate future innovations and recursively evolve variance. A predictive return distribution must draw future shocks. A parameter-aware predictive distribution must also draw or bootstrap theta. Pascual, Romo, and Ruiz provide the classical recursive-bootstrap construction ([Pascual, Romo, and Ruiz, 2006](https://doi.org/10.1016/j.csda.2004.12.008)).

Return at least:

- `expected_variance`, the model expectation of each future conditional variance;
- simulated conditional-variance quantiles;
- return predictive quantiles and draws;
- intervals with `uncertainty="innovation"`, `"parameter"`, or `"both"`;
- the covariance estimator or bootstrap design used for theta;
- simulation and refit seeds, draw counts, and Monte Carlo error.

Conditional VaR and ES need their own interval methods. Recent work provides asymptotically valid intervals for extreme VaR and ES in covariate-augmented GARCH-type location-scale models ([Hoga, 2021](https://doi.org/10.1016/j.ijforecast.2020.08.009)) and simultaneous confidence bands across tail levels ([Li, Peng, and Song, 2023](https://doi.org/10.1017/S0266466622000275)). These are later risk-workflow features, but the forecast object should be designed now so it can hold pointwise and simultaneous uncertainty without an API break.

Model uncertainty is already partly represented by scivol's model confidence set. A later forecast ensemble should retain between-model dispersion separately from within-model parameter uncertainty.

## 9. Bayesian estimation

Bayesian GARCH is not a research novelty. Bauwens and Lubrano showed Gibbs-style Bayesian inference for asymmetric Student-t GARCH and predictive option prices, while also demonstrating that an apparently innocuous flat prior on degrees of freedom can produce an improper posterior ([Bauwens and Lubrano, 1998](https://doi.org/10.1111/1368-423X.11003)). The prior is therefore part of correctness, not decoration.

Recent work makes two points relevant to scivol. Li, Clements, and Drovandi show that sequential Monte Carlo can perform estimation, model selection, and full predictive inference for GARCH-type models, and that parameter posteriors may remain nonnormal even in long series ([Li, Clements, and Drovandi, 2021](https://doi.org/10.1016/j.ecosta.2020.02.002)). Hamiltonian Monte Carlo is also feasible for differentiable GARCH mixtures and supplies volatility predictive intervals ([Liang, Qin, and Xia, 2022](https://doi.org/10.1007/s10614-022-10337-4)).

Bayesian output adds real capability:

- credible intervals can follow curved, constrained posterior geometry;
- predictive draws integrate parameter uncertainty directly;
- marginal likelihood or predictive criteria support model comparison;
- priors can stabilize weakly identified tail and persistence parameters.

It also creates new ways to be wrong. A native implementation needs proper default priors on transformed parameters, Jacobians, multiple-chain convergence diagnostics, effective sample size, Monte Carlo standard error, divergent-transition checks for HMC, tempering diagnostics for SMC, posterior predictive checks, and reproducible random streams. Under model misspecification, an ordinary posterior does not automatically have sandwich-calibrated frequentist coverage ([Muller, 2013](https://doi.org/10.3982/ECTA9097)).

Recommended sequence:

1. Expose a stable internal or developer API for log likelihood, log-Jacobian, gradients, filtering state, and simulation. This lets PyMC, Stan, NumPyro, or BlackJAX users build external models without duplicating recursions.
2. Publish reference prior recipes and transformation examples, not a claim of universal priors.
3. If demand justifies native inference, implement one method well. SMC fits multimodal and model-comparison workflows; NUTS/HMC fits smooth low-dimensional models and can reuse scivol's derivatives. Do not ship two poorly diagnosed samplers for checklist completeness.
4. Keep `method="bayes"` results distinct from MLE and QMLE results. Never populate frequentist `std_errors` with posterior standard deviations.

Bayesian estimation is P2 because the current frequentist gaps are cheaper and affect more users. The external log-density interface is a P1 interoperability feature.

## 10. Recommended end-to-end workflow

A complete fit should support the following sequence without making all stages automatic.

1. **Validate time semantics.** Declare trading time, calendar time, aggregation, missingness, and any exogenous-data release lag.
2. **Fit through the requested public path.** Record initialization, parameter transform, optimizer, starts, tolerances, and effective sample.
3. **Audit the optimum.** Check feasibility, gradients or KKT residual, curvature, state finiteness, and objective agreement.
4. **Classify inference.** Decide whether the target is correct-density MLE, conditional-moment QMLE, heavy-tail robust estimation, a boundary problem, or a two-step estimator.
5. **Compute legal uncertainty only.** Select inverse information, sandwich, profile, bootstrap, subsampling, or posterior output according to the classification.
6. **Test the assumptions that determine inference.** Run moment-existence, residual, innovation-distribution, and stability checks. The separate diagnostics survey specifies the broader test suite.
7. **Issue forecasts with explicit uncertainty layers.** Separate expected variance, variance-path draws, return prediction, parameter intervals, and model dispersion.
8. **Preserve an audit bundle.** Include software version, data index hash, model UID, initialization, objective scaling, random seeds, warnings, and enough filtered state to reproduce the forecast origin.

An `InferenceReport` could make this concrete:

```python
report = result.inference_report()
report.estimator
report.covariance_assumptions
report.boundary_status
report.moment_status
report.optimization_status
report.available_methods
report.invalid_methods
```

The report should say why a method is invalid. For example: “Wald inference suppressed because `alpha[2]` is on the positivity boundary,” or “Gaussian QMLE sandwich inference requires a finite innovation fourth moment; fitted Student-t `nu=3.6` does not provide it.” This is the difference between exposing formulas and providing correct inference.

## 11. Concrete implementation order

### Phase A: correctness metadata and cheap uncertainty

- Add covariance provenance, rank, eigenvalue, condition number, and assumption fields.
- Add optimizer completion checks and KKT residuals.
- Add moment-validity warnings based on fitted density plus a general result slot for tested moment conditions.
- Add one-step delta-method intervals for variance, volatility, quantiles, and VaR.
- Add explicit time-index and missing-data policy errors.
- Add initialization metadata and sensitivity reevaluation.

### Phase B: mature frequentist extensions

- Add one-coefficient chi-bar-square tests and restricted theta-space fits.
- Add generic shrinkage-bootstrap LR boundary tests.
- Add profile likelihood for parameters and smooth scalar functions.
- Add recursive residual bootstrap for parameter and forecast targets under iid innovations.
- Add the Francq-Zakoian moment-existence tests.
- Add fixed-design residual bootstrap for the supported two-step VaR workflow.

### Phase C: specialized robustness

- Add studentized subsample bootstrap and at least one heavy-tail robust estimator.
- Add experimental recursive block bootstrap for dependent higher moments.
- Add SMC likelihood for missing GARCH(1,1), with particle convergence diagnostics.
- Expose the external Bayesian log-density interface.

### Phase D: second inferential paradigm

- Add one native Bayesian engine only after prior, Jacobian, sampler, and predictive validation are complete.
- Consider COGARCH or a dedicated irregular-time family only if scivol expands beyond regularly indexed return models.

## Source catalogue

### Foundational estimation and misspecification

- Halbert White, “Maximum Likelihood Estimation of Misspecified Models” (1982), [DOI 10.2307/1912526](https://doi.org/10.2307/1912526).
- Tim Bollerslev and Jeffrey Wooldridge, “Quasi-Maximum Likelihood Estimation and Inference in Dynamic Models with Time-Varying Covariances” (1992), [DOI 10.1080/07474939208800229](https://doi.org/10.1080/07474939208800229).
- Sang-Won Lee and Bruce Hansen, “Asymptotic Theory for the GARCH(1,1) Quasi-Maximum Likelihood Estimator” (1994), [DOI 10.1017/S0266466600008215](https://doi.org/10.1017/S0266466600008215).
- Gabriele Fiorentini, Giorgio Calzolari, and Lorenzo Panattoni, “Analytic Derivatives and the Computation of GARCH Estimates” (1996), [DOI 10.1002/(SICI)1099-1255(199607)11:4%3C399::AID-JAE401%3E3.0.CO;2-R](https://doi.org/10.1002/(SICI)1099-1255(199607)11:4%3C399::AID-JAE401%3E3.0.CO;2-R).
- Wing Chi Ip, Wai Keung Li, and Philip L. H. Yu, “The Asymptotic Convexity of the Negative Likelihood Function of GARCH Models” (2006), [DOI 10.1016/j.csda.2004.08.012](https://doi.org/10.1016/j.csda.2004.08.012).

### Boundaries, persistence, and moments

- Christian Francq and Jean-Michel Zakoian, “Quasi-Maximum Likelihood Estimation in GARCH Processes when Some Coefficients Are Equal to Zero” (2007), [DOI 10.1016/j.spa.2007.01.001](https://doi.org/10.1016/j.spa.2007.01.001).
- Christian Francq and Jean-Michel Zakoian, “Testing the Nullity of GARCH Coefficients: Correction of the Standard Tests and Relative Efficiency Comparisons” (2009), [DOI 10.1198/jasa.2009.0117](https://doi.org/10.1198/jasa.2009.0117).
- Christian Francq and Jean-Michel Zakoian, “Strict Stationarity Testing and Estimation of Explosive and Stationary Generalized Autoregressive Conditional Heteroscedasticity Models” (2012), [DOI 10.3982/ECTA9405](https://doi.org/10.3982/ECTA9405).
- Rasmus Sondergaard Pedersen and Anders Rahbek, “Inference and Testing on the Boundary in Extended Constant Conditional Correlation GARCH Models” (2017), [DOI 10.1016/j.jeconom.2016.09.004](https://doi.org/10.1016/j.jeconom.2016.09.004).
- Giuseppe Cavaliere, Heino Bohn Nielsen, Rasmus Sondergaard Pedersen, and Anders Rahbek, “Bootstrap Inference on the Boundary of the Parameter Space, with Application to Conditional Volatility Models” (2022), [DOI 10.1016/j.jeconom.2020.05.006](https://doi.org/10.1016/j.jeconom.2020.05.006).
- Christian Francq and Jean-Michel Zakoian, “Testing the Existence of Moments for GARCH Processes” (2022), [DOI 10.1016/j.jeconom.2020.05.009](https://doi.org/10.1016/j.jeconom.2020.05.009).

### Bootstrap and forecast uncertainty

- Lorenzo Pascual, Juan Romo, and Esther Ruiz, “Bootstrap Prediction for Returns and Volatilities in GARCH Models” (2006), [DOI 10.1016/j.csda.2004.12.008](https://doi.org/10.1016/j.csda.2004.12.008).
- Ki-Hoon Jeong, “Residual-Based GARCH Bootstrap and Second-Order Asymptotic Refinement” (2017), [DOI 10.1017/S0266466616000104](https://doi.org/10.1017/S0266466616000104).
- Yannick Hoga, “The Uncertainty in Extreme Risk Forecasts from Covariate-Augmented Volatility Models” (2021), [DOI 10.1016/j.ijforecast.2020.08.009](https://doi.org/10.1016/j.ijforecast.2020.08.009).
- Shuo Li, Liuhua Peng, and Xiaojun Song, “Simultaneous Confidence Bands for Conditional Value-at-Risk and Expected Shortfall” (2023), [DOI 10.1017/S0266466622000275](https://doi.org/10.1017/S0266466622000275).
- Eric Beutner, Alexander Heinemann, and Stephan Smeekes, “A Residual Bootstrap for Conditional Value-at-Risk” (2024), [DOI 10.1016/j.jeconom.2023.105554](https://doi.org/10.1016/j.jeconom.2023.105554).
- Eric Beutner, Julia Schaumburg, and Barend Spanjers, “Bootstrapping GARCH Models under Dependent Innovations” (2026), [DOI 10.1007/s10182-026-00560-x](https://doi.org/10.1007/s10182-026-00560-x).

### Heavy-tail inference

- Peter Hall and Qiwei Yao, “Inference in ARCH and GARCH Models with Heavy-Tailed Errors” (2003), [DOI 10.1111/1468-0262.00396](https://doi.org/10.1111/1468-0262.00396).
- Jonathan B. Hill, “Robust Estimation and Inference for Heavy Tailed GARCH” (2015), [DOI 10.3150/14-BEJ616](https://doi.org/10.3150/14-BEJ616).
- Marc Hallin, Hang Liu, and Kanchan Mukherjee, “M-Estimation in GARCH Models in the Absence of Higher-Order Moments” (2023), [DOI 10.1007/978-981-99-0803-5_8](https://doi.org/10.1007/978-981-99-0803-5_8).

### Missing and irregular observations

- Tarek Meddahi, Eric Renault, and Bas Werker, “GARCH and Irregularly Spaced Data” (2006), [DOI 10.1016/j.econlet.2005.07.027](https://doi.org/10.1016/j.econlet.2005.07.027).
- Damien C. H. Wee, Feng Chen, and William T. M. Dunsmuir, “Likelihood Inference for a COGARCH Process Using Sequential Monte Carlo” (2019), [DOI 10.1093/jjfinec/nby012](https://doi.org/10.1093/jjfinec/nby012).
- Francisco Blasques, Paolo Gorgi, and Siem Jan Koopman, “Missing Observations in Observation-Driven Time Series Models” (2021), [DOI 10.1016/j.jeconom.2020.07.043](https://doi.org/10.1016/j.jeconom.2020.07.043).
- Damien C. H. Wee, Feng Chen, and William T. M. Dunsmuir, “Estimating GARCH(1,1) in the Presence of Missing Data” (2023), [DOI 10.1214/23-AOAS1734](https://doi.org/10.1214/23-AOAS1734).

### Profiles, derivatives, and Bayesian inference

- Samuel M. Fischer and Mark A. Lewis, “A Robust and Efficient Algorithm to Find Profile Likelihood Confidence Intervals” (2021), [DOI 10.1007/s11222-021-10012-y](https://doi.org/10.1007/s11222-021-10012-y).
- Thomas Busch, “A Robust LR Test for the GARCH Model” (2005), [DOI 10.1016/j.econlet.2005.03.006](https://doi.org/10.1016/j.econlet.2005.03.006).
- Atılım Gunes Baydin, Barak Pearlmutter, Alexey Radul, and Jeffrey Siskind, “Automatic Differentiation in Machine Learning: a Survey” (2018), [JMLR article](https://jmlr.org/papers/v18/17-468.html).
- Joaquim Martins, Peter Sturdza, and Juan Alonso, “The Complex-Step Derivative Approximation” (2003), [DOI 10.1145/838250.838251](https://doi.org/10.1145/838250.838251).
- Luc Bauwens and Michel Lubrano, “Bayesian Inference on GARCH Models Using the Gibbs Sampler” (1998), [DOI 10.1111/1368-423X.11003](https://doi.org/10.1111/1368-423X.11003).
- Yifei Li, Michael Clements, and Christopher Drovandi, “Efficient Bayesian Estimation for GARCH-Type Models via Sequential Monte Carlo” (2021), [DOI 10.1016/j.ecosta.2020.02.002](https://doi.org/10.1016/j.ecosta.2020.02.002).
- Rubing Liang, Binbin Qin, and Qiang Xia, “Bayesian Inference for Mixed Gaussian GARCH-Type Model by Hamiltonian Monte Carlo Algorithm” (2022), [DOI 10.1007/s10614-022-10337-4](https://doi.org/10.1007/s10614-022-10337-4).
- Ulrich Muller, “Risk of Bayesian Inference in Misspecified Models, and the Sandwich Covariance Matrix” (2013), [DOI 10.3982/ECTA9097](https://doi.org/10.3982/ECTA9097).
