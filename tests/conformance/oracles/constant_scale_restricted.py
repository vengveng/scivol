"""Differentiate the restricted scalar function, never a projected ambient jet."""

import numpy as np
import mpmath as mp

from .garch import jax_modules, prepare_logpdf
from .mean import mean_count, mp_residuals, residual_path, theta_objective


def restricted_objective(case, indices, *, request=7, count=256):
    _, jnp, jsp = jax_modules()
    point = np.asarray(case["parameters"], dtype=float)
    indices = np.asarray(indices, dtype=int)
    index = mean_count(case)
    rows = np.asarray(case["scored_rows"], dtype=int) - case["hold_back"]
    all_mean_fixed = not any(i < index for i in indices)
    # This second proof covers the all-zero MA regression. Every forcing term is
    # fixed zero and every presample innovation is zero, so induction gives e=0
    # for every admissible free MA coefficient, not only at the anchor point.
    ma_only_zero = (case["model"] == "arma" and np.all(np.asarray(case["y"]) == 0)
                    and not any(i < int(case["intercept"]) + case["p"] for i in indices)
                    and (not case["intercept"] or point[0] == 0))
    fixed_zero = []
    if case["density"] == "ged":
        with mp.workdps(90):
            residuals = mp_residuals(case, point)
        for row in rows:
            constant = all_mean_fixed or ma_only_zero
            if case["model"] == "linear":
                features = np.r_[1.0, case["design"][row+case["hold_back"]]] if case["intercept"] else case["design"][row+case["hold_back"]]
                constant = not any(i < index and features[i] != 0 for i in indices)
            if constant and residuals[row] == 0:
                fixed_zero.append(row)
    anchor = jnp.asarray(point)
    selected = jnp.asarray(indices)
    if not fixed_zero:
        ambient = theta_objective(case, point, request=request, count=count)
        return lambda free: ambient(anchor.at[selected].set(free))
    ordinary = np.asarray([row for row in rows if row not in fixed_zero], dtype=int)

    def objective(free):
        theta = anchor.at[selected].set(free)
        variance, nu = theta[index], theta[index+1]
        residuals = residual_path(theta, case)
        logpdf = prepare_logpdf(theta[index+1:], "ged", count=count)
        value = jnp.sum(jnp.log(variance)/2 - logpdf(residuals[ordinary]/jnp.sqrt(variance)))
        # Exact restriction at a structurally constant zero. No abs/log-zero AD,
        # smoothing, synthetic residual or finite-difference convention is used.
        log_a = (jsp.special.gammaln(1/nu)-jsp.special.gammaln(3/nu))/2
        centre = jnp.log(2.0)-jnp.log(nu)+jsp.special.gammaln(1/nu)+log_a+jnp.log(variance)/2
        return value + len(fixed_zero)*centre

    return objective


def restricted_jet(case, indices, request=7):
    jax, jnp, _ = jax_modules()
    indices = np.asarray(indices, dtype=int)
    objective = restricted_objective(case, indices, request=request)
    free = jnp.asarray(np.asarray(case["parameters"])[indices])
    if not len(indices):
        value = float(objective(free))
        return value if request & 1 else None, np.empty(0) if request & 2 else None, np.empty((0,0)) if request & 4 else None
    return (float(objective(free)) if request & 1 else None,
            np.asarray(jax.grad(objective)(free)) if request & 2 else None,
            np.asarray(jax.jacfwd(jax.grad(objective))(free)) if request & 4 else None)
