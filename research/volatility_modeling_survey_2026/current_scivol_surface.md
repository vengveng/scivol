# Current scivol surface

Snapshot date: 2026-08-25

This inventory is based on `MODELS.md`, `README.md`, the public exports, and the
implementation in `scivol/`. It records what exists before the literature is
used to identify gaps.

## Models

The univariate volatility layer has `GARCH`, `GJRGARCH`, `EGARCH`, and
`APARCH`. The conditional-density layer has Normal, standardized Student-t,
Hansen skew-t, GED, generalized hyperbolic, hyperbolic, normal-inverse-Gaussian,
and variance-gamma distributions. All eight densities can be combined with the
four zero-mean volatility models.

The mean layer has `ARMA`, `ARX`, and `HARX`. Linked ARMA, ARX, and HARX models
cover GARCH, GJR-GARCH, and EGARCH. APARCH is currently zero-mean only.

The multivariate layer has `CCC` and scalar `DCC(p, q)`. DCC supports Gaussian
and common-degrees-of-freedom Student-t copulas after separate univariate
marginal fits.

## Estimation and verification

scivol exposes MLE and QMLE, constrained theta-space and transformed z-space
optimization, explicit optimization-path metadata, fixed-parameter evaluation,
and analytical likelihood derivatives. The shipped derivative paths are checked
against independent automatic-differentiation objectives. The C extension owns
the performance-sensitive likelihood recursions.

The package also has warm starts, fit budgets and timeouts, multi-series fitting,
rolling and expanding fits, and automatic search across mean, volatility,
density, and order choices. Automatic selection can use AIC, BIC, HQIC, or an
out-of-sample log-score, QLIKE, or MSE loss. The out-of-sample path includes a
Diebold-Mariano comparison and a block-bootstrap model confidence set.

## Post-fit workflows

Univariate results expose filtered states, multistep conditional mean and
variance forecasts, predictive quantiles, VaR, simulation, classical and robust
covariance estimates, and PIT-based diagnostics. The diagnostic method currently
implements a binned uniformity test and Ljung-Box tests for the first four
centered PIT moments.

CCC and DCC results expose conditional correlations and covariances, joint
forecasts, parametric and empirical scenarios, path simulation, pairwise access,
and rolling fits.

## Important boundaries of the current surface

The following are absent or materially narrower than the corresponding mature
literature:

- realized-measure construction and models such as HAR-RV, Realized GARCH,
  HEAVY, and GARCH-MIDAS;
- forecast intervals that include parameter uncertainty, residual-bootstrap
  forecasts, expected shortfall, and formal VaR or ES backtests;
- standard residual, sign-bias, stability, structural-break, and multivariate
  adequacy tests;
- asymmetric and high-dimensional correlation dynamics beyond scalar DCC;
- long-memory, component, score-driven, regime-switching, stochastic-volatility,
  and multiplicative-error families;
- missing-observation and irregular-time likelihood semantics;
- option-implied information, risk-neutral dynamics, and option calibration.

These boundaries do not all have equal priority. The synthesis report ranks them
by user value, empirical maturity, architectural fit, and the chance that scivol
can implement them faster and more correctly than existing software.
