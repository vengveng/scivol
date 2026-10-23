"""AD of the physical restricted scalar, not selection of an ambient jet."""

import numpy as np
import mpmath as mp

from .garch import jax_modules, prepare_logpdf, variance_path, volatility_width
from .joint import state_path, mean_width, shape_start, mean_case
from .mean import mp_residuals


def restricted_jet(case, indices, request=7):
    jax, jnp, jsp = jax_modules()
    indices = np.asarray(indices, dtype=int)
    point = np.asarray(case["parameters"], dtype=float)
    joint = case["model"] not in ("garch", "gjr")
    mean = mean_width(case) if joint else 0
    shape = shape_start(case) if joint else volatility_width(case)
    rows = np.asarray(case.get("scored_positions", case["scored_rows"]), dtype=int)-case["hold_back"]
    fixed_zero = []
    if case["density"] == "ged" and not any(axis < mean for axis in indices):
        if joint:
            with mp.workdps(90):
                residuals = mp_residuals(mean_case(case), point)
        else:
            residuals = case["y"][case["hold_back"]:case["stop"]]
        fixed_zero = [row for row in rows if residuals[row] == 0]
    ordinary = np.asarray([row for row in rows if row not in fixed_zero], dtype=int)
    anchor, selected = jnp.asarray(point), jnp.asarray(indices)

    def objective(free):
        theta = anchor.at[selected].set(free)
        if joint:
            residuals, variances = state_path(theta, case)
        else:
            residuals = jnp.asarray(case["y"][case["hold_back"]:case["stop"]])
            variances = variance_path(theta, case)
        logpdf = prepare_logpdf(theta[shape:], case["density"])
        value = jnp.sum(jnp.log(variances[ordinary])/2-logpdf(residuals[ordinary]/jnp.sqrt(variances[ordinary])))
        if fixed_zero:
            # Exact fixed-residual zero law. No abs/log-zero AD convention.
            nu = theta[shape]
            log_a = (jsp.special.gammaln(1/nu)-jsp.special.gammaln(3/nu))/2
            center = jnp.log(2.0)-jnp.log(nu)+jsp.special.gammaln(1/nu)+log_a
            value += len(fixed_zero)*center+jnp.sum(jnp.log(variances[jnp.asarray(fixed_zero)])/2)
        return value

    free = jnp.asarray(point[indices])
    if not len(indices):
        return (float(objective(free)) if request & 1 else None,
                np.empty(0) if request & 2 else None, np.empty((0,0)) if request & 4 else None)
    return (float(objective(free)) if request & 1 else None,
            np.asarray(jax.grad(objective)(free)) if request & 2 else None,
            np.asarray(jax.jacfwd(jax.grad(objective))(free)) if request & 4 else None)
