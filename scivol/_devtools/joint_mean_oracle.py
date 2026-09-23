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
        c = (
            jsp.special.gammaln((nu + 1) / 2)
            - jsp.special.gammaln(nu / 2)
            - 0.5 * jnp.log(jnp.pi * (nu - 2))
        )
        if density == 2:
            lam = shape[1]
            a = 4 * lam * jnp.exp(c) * (nu - 2) / (nu - 1)
            b = jnp.sqrt(1 + 3 * lam * lam - a * a)
            u = b * z + a
            z = u / (1 + lam * jnp.where(u >= 0, 1.0, -1.0))
            c += jnp.log(b)
        return -len(z) * c + (nu + 1) / 2 * jnp.sum(jnp.log1p(z * z / (nu - 2)))
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
):
    """Differentiate the objective or an explicitly requested limiting branch.

    A limiting branch is anchored at the exact affine residual-zero point,
    not the supplied doubles. Its backward error must be reported separately.
    Request 3 omits the Hessian for theta-space diagnostics.
    """
    jax, jnp, jsp = _jax_modules()
    if request not in (3, 7) or (request == 3 and coordinates is not None):
        raise ValueError(
            "The oracle supports full jets or theta first-order diagnostics."
        )
    m = x.shape[1]
    beta_start = m + 1 + (1 if family == 0 else 2) * p
    end = beta_start + q + int(family == 3)
    fixed_zero = (y == 0.0) & np.all(x == 0.0, axis=1)
    name = NAMES[density]
    anchor_residuals = None
    if kink_limit is not None:
        from fractions import Fraction

        row, side = kink_limit
        if (
            family != 2
            or density not in (0, 7)
            or coordinates is not None
            or side not in (-1, 1)
        ):
            raise ValueError(
                "Limiting branches are validated for theta Normal/VG EGARCH."
            )
        if not 0 <= row < len(y) or m == 0 or not np.all(x[:, 0] == 1):
            raise ValueError(
                "A limiting branch requires a sample row and an intercept."
            )
        mean = [Fraction(float(v)) for v in theta[:m]]
        mean[0] = Fraction(float(y[row])) - sum(
            Fraction(float(v)) * b for v, b in zip(x[row, 1:], mean[1:])
        )
        anchor_residuals = jnp.asarray(
            [
                float(
                    Fraction(float(v))
                    - sum(Fraction(float(a)) * b for a, b in zip(features, mean))
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
            moment = (
                _jax_vg_abs_moment(jnp, jsp, *shape)
                if density == 7
                else _jax_aparch_power_moment(jnp, jsp, 0.0, 1.0, shape, name)
            )
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
                        kink_limit[1] * z
                        if kink_limit is not None and t - 1 - lag == kink_limit[0]
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
