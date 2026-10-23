"""Independent AD reference for joint linear-mean volatility likelihoods.

Development only. No production jet, recursion, density cache or derivative is
used to construct this objective. Conditioning variance is fixed at observation 0;
all observations are scored, as in the linear-mean sample contract.
"""

import numpy as np

from .ad_oracle import (
    _jax_modules,
    _jax_aparch_power_moment,
    _jax_gh_logpdf,
    _jax_vg_logpdf,
    _jax_vg_abs_moment,
    _jax_student_log_normalizer,
    _jax_student_log_power,
    _jax_hansen_skewt_abs_moment,
    _egarch_skewt_quadrature_rule,
    _native_shape_coordinates,
    _native_reflection_coefficients,
    native_coordinate_chain_rule_value_grad_hess,
)

NAMES = ("normal", "studentt", "skewt", "ged", "gh", "nig", "hyperbolic", "vg")


def _density_nll(jnp, jsp, z, shape, density, fixed_zero):
    if density == 0:
        return 0.5 * jnp.sum(jnp.log(2 * jnp.pi) + z * z)
    if density in (1, 2):
        nu = shape[0]
        c = _jax_student_log_normalizer(jnp, jsp, nu)
        if density == 2:
            lam = shape[1]
            a = 4 * lam * jnp.exp(c) / (1+1/(nu-2))
            b = jnp.sqrt(1 + 3 * lam * lam - a * a)
            u = b * z + a
            z = u / (1 + lam * jnp.where(u >= 0, 1.0, -1.0))
            c += jnp.log(b)
        return -len(z) * c + jnp.sum(_jax_student_log_power(jnp, nu, z*z))
    if density == 3:
        nu = shape[0]
        log_scale = 0.5 * (jsp.special.gammaln(1 / nu) - jsp.special.gammaln(3 / nu))
        c = jnp.log(nu) - jnp.log(2.0) - log_scale - jsp.special.gammaln(1 / nu)
        return -len(z) * c + jnp.sum(
            (jnp.abs(z[~fixed_zero]) * jnp.exp(-log_scale)) ** nu
        )
    if density == 7:
        return -jnp.sum(_jax_vg_logpdf(jnp, jsp, z, *shape))
    lam = shape[2] if density == 4 else (-0.5 if density == 5 else 1.0)
    return -jnp.sum(_jax_gh_logpdf(jnp, jsp, z, shape[0], shape[1], lam))


def joint_value_grad_hess(
    theta,
    y,
    x,
    h0,
    p,
    q,
    family,
    density,
    *,
    coordinates=None,
    request=7,
    kink_limit=None,
    kink_pivot=0,
    exact_mean_anchor=False,
):
    """Differentiate the objective or an explicitly requested limiting branch.

    A limiting branch is anchored at the exact affine residual-zero point,
    not the supplied doubles. Its backward error must be reported separately.
    The selected mean coefficient absorbs the exact rational correction.
    Request 3 omits the Hessian for theta-space diagnostics.
    An exact mean anchor instead preserves the supplied theta point. Rational
    residuals are rounded once to binary64; AD still differentiates the full
    affine mean, recursion and density. This option requires theta coordinates.
    """
    jax, jnp, jsp = _jax_modules()
    if request not in (3, 7) or (request == 3 and coordinates is not None):
        raise ValueError(
            "The oracle supports full jets or theta first-order diagnostics."
        )
    if exact_mean_anchor and (coordinates is not None or kink_limit is not None):
        raise ValueError("An exact mean anchor requires theta space without a kink limit.")
    m = x.shape[1]
    beta_start = m + 1 + (1 if family == 0 else 2) * p
    end = beta_start + q + int(family == 3)
    fixed_zero = (y == 0.0) & np.all(x == 0.0, axis=1)
    name = NAMES[density]
    anchor_residuals = None
    anchor_mean = None
    limits = {}
    if exact_mean_anchor or kink_limit is not None:
        from fractions import Fraction

    if exact_mean_anchor:
        anchor_mean = [Fraction(float(v)) for v in theta[:m]]
    if kink_limit is not None:
        # One (row, side) pair with an integer pivot, or equal-length lists of both.
        several = isinstance(kink_limit, list)
        pairs = list(kink_limit) if several else [kink_limit]
        pivots = list(kink_pivot) if several and isinstance(kink_pivot, (list, tuple)) else [kink_pivot]
        if (
            family != 2
            or density not in (0, 1, 2, 4, 5, 6, 7)
            or coordinates is not None
            or not pairs
            or len(pivots) != len(pairs)
            or any(len(pair) != 2 or pair[1] not in (-1, 1) for pair in pairs)
        ):
            raise ValueError(
                "Limiting branches require theta Normal, Student-t, Skew-t or GH-family EGARCH."
            )
        rows = [pair[0] for pair in pairs]
        if (
            len(set(rows)) != len(rows) or len(set(map(str, pivots))) != len(pivots)
            or any(not isinstance(row, (int, np.integer)) or not 0 <= row < len(y) for row in rows)
            or any(not isinstance(pivot, (int, np.integer)) or not 0 <= pivot < m for pivot in pivots)
            or (len(rows) == 1 and x[rows[0], pivots[0]] == 0)
        ):
            raise ValueError(
                "A limiting branch requires a sample row and a nonzero mean feature."
            )
        limits = {int(row): side for row, side in pairs}
        mean = [Fraction(float(v)) for v in theta[:m]]
        # Solve the active rows exactly for the pivot coefficients.
        system = [
            [Fraction(float(x[row, pivot])) for pivot in pivots]
            + [Fraction(float(y[row])) - sum(
                Fraction(float(x[row, j])) * mean[j] for j in range(m) if j not in pivots)]
            for row in rows
        ]
        for column in range(len(rows)):
            lead = next((i for i in range(column, len(rows)) if system[i][column] != 0), None)
            if lead is None:
                raise ValueError("Limiting branches require linearly independent faces.")
            system[column], system[lead] = system[lead], system[column]
            for i in range(len(rows)):
                if i != column and system[i][column] != 0:
                    ratio = system[i][column] / system[column][column]
                    system[i] = [a - ratio * b for a, b in zip(system[i], system[column])]
        for i, pivot in enumerate(pivots):
            mean[pivot] = system[i][-1] / system[i][i]
        anchor_mean = mean
    if anchor_mean is not None:
        anchor_residuals = jnp.asarray(
            [
                float(
                    Fraction(float(v))
                    - sum(Fraction(float(a)) * b for a, b in zip(features, anchor_mean))
                )
                for v, features in zip(y, x)
            ]
        )

    def pack_power(z):
        shape = _native_shape_coordinates(jax, jnp, z[end:], name)
        if family == 2:
            return jnp.concatenate(
                [
                    z[:beta_start],
                    _native_reflection_coefficients(jnp, z[beta_start:end], 1.0),
                    shape,
                ]
            )
        gamma = jnp.tanh(z[m + 1 + p : beta_start])
        delta = (
            shape[0] * jax.nn.sigmoid(z[end - 1])
            if density in (1, 2)
            else jax.nn.softplus(z[end - 1])
        )
        logits = jnp.r_[z[m + 1 : m + 1 + p], z[beta_start : beta_start + q], 0.0]
        weights = (1 - 8 * (p + q + 1) * np.finfo(float).eps) * jax.nn.softmax(logits)[
            :-1
        ]
        moments = jnp.stack(
            [
                _jax_aparch_power_moment(jnp, jsp, gamma[i], delta, shape, name)
                for i in range(p)
            ]
        )
        return jnp.concatenate(
            [
                z[:m],
                jax.nn.softplus(z[m : m + 1]),
                weights[:p] / moments,
                gamma,
                weights[p:],
                delta[None],
                shape,
            ]
        )

    def objective(parameters):
        if coordinates is not None and family >= 2:
            parameters = pack_power(parameters)
        residuals = (
            jnp.asarray(y) - jnp.asarray(x) @ parameters[:m]
            if anchor_residuals is None
            else anchor_residuals
            - jnp.asarray(x) @ (parameters[:m] - jnp.asarray(theta[:m]))
        )
        alpha = parameters[m + 1 : m + 1 + p]
        gamma = parameters[m + 1 + p : beta_start]
        beta, shape = parameters[beta_start : beta_start + q], parameters[end:]
        initial = jnp.asarray(h0)
        if family == 3:
            delta = parameters[end - 1]
            history = [initial ** (delta / 2)]
        elif family == 2:
            if density == 1:
                moment = 2*jnp.exp(_jax_student_log_normalizer(jnp, jsp, shape[0]))/(1+1/(shape[0]-2))
            elif density == 2:
                nodes, weights = _egarch_skewt_quadrature_rule()
                moment = _jax_hansen_skewt_abs_moment(jnp, jsp, *shape, jnp.asarray(nodes), jnp.asarray(weights))
            elif density == 7:
                moment = _jax_vg_abs_moment(jnp, jsp, *shape)
            else:
                moment = _jax_aparch_power_moment(jnp, jsp, 0.0, 1.0, shape, name)
            history = [jnp.log(initial)]
        else:
            history = [initial]
        variance = [initial]
        for t in range(1, len(y)):
            state = parameters[m]
            for lag in range(min(t, p)):
                e = residuals[t - 1 - lag]
                if family == 0:
                    state += alpha[lag] * e * e
                elif family == 1:
                    state += (alpha[lag] + gamma[lag] * (e < 0)) * e * e
                elif family == 2:
                    z = e / jnp.sqrt(variance[t - 1 - lag])
                    absolute = (
                        limits[t - 1 - lag] * z
                        if t - 1 - lag in limits
                        else jnp.abs(z)
                    )
                    state += alpha[lag] * (absolute - moment) + gamma[lag] * z
                elif not fixed_zero[t - 1 - lag]:
                    shock = (jnp.where(e >= 0, 1.0, -1.0) - gamma[lag]) * e
                    state += alpha[lag] * shock**delta
            for lag in range(min(t, q)):
                state += beta[lag] * history[t - 1 - lag]
            history.append(state)
            variance.append(
                jnp.exp(state)
                if family == 2
                else (state ** (2 / delta) if family == 3 else state)
            )
        variance = jnp.stack(variance)
        return 0.5 * jnp.sum(jnp.log(variance)) + _density_nll(
            jnp, jsp, residuals / jnp.sqrt(variance), shape, density, fixed_zero
        )

    point = jnp.asarray(
        coordinates if coordinates is not None and family >= 2 else theta
    )
    result = (
        float(objective(point)),
        np.asarray(jax.grad(objective)(point)),
        np.asarray(jax.hessian(objective)(point)) if request & 4 else None,
    )
    if coordinates is not None and family < 2:
        return native_coordinate_chain_rule_value_grad_hess(
            coordinates,
            theta,
            *result,
            p,
            q,
            name,
            leverage=family == 1,
            linear_mean_parameters=m,
        )
    return result
