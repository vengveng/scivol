"""Independent development oracle for constant-scale recursive mean likelihoods."""

import numpy as np
from .ad_oracle import (
    _jax_modules,
    _jax_gh_logpdf,
    _jax_vg_logpdf,
    _native_shape_coordinates,
)


def mean_value_grad_hess(
    theta, y, x, q, first, density, *, coordinates=False, ar_order=0, ar_start=1
):
    jax, jnp, jsp = _jax_modules()
    columns = x.shape[1]
    mean_count = columns + q

    def reflection(z):
        values = jnp.zeros(len(z))
        for stage in range(len(z)):
            k = jnp.tanh(z[stage])
            previous = values
            for index in range(stage):
                values = values.at[index].set(
                    previous[index] - k * previous[stage - 1 - index]
                )
            values = values.at[stage].set(k)
        return values

    def pack(z):
        mean = z[:mean_count]
        mean = mean.at[ar_start : ar_start + ar_order].set(
            reflection(z[ar_start : ar_start + ar_order])
        )
        mean = mean.at[columns:mean_count].set(-reflection(z[columns:mean_count]))
        names = ["normal", "studentt", "skewt", "ged", "gh", "nig", "hyperbolic", "vg"]
        return jnp.concatenate(
            [
                mean,
                jax.nn.softplus(z[mean_count : mean_count + 1]),
                _native_shape_coordinates(
                    jax, jnp, z[mean_count + 1 :], names[density]
                ),
            ]
        )

    def objective(parameters):
        if coordinates:
            parameters = pack(parameters)

        def advance(history, observation):
            observed, features = observation
            residual = observed - features @ parameters[:columns]
            residual -= parameters[columns:mean_count] @ history
            if q:
                history = jnp.concatenate([residual[None], history[:-1]])
            return history, residual

        _, residuals = jax.lax.scan(
            advance, jnp.zeros(q), (jnp.asarray(y[first:]), jnp.asarray(x[first:]))
        )
        variance = parameters[mean_count]
        z = residuals / jnp.sqrt(variance)
        shape = parameters[mean_count + 1 :]
        scale = 0.5 * len(residuals) * jnp.log(variance)
        if density == 0:
            return scale + 0.5 * jnp.sum(jnp.log(2 * jnp.pi) + z * z)
        if density in (1, 2):
            nu = shape[0]
            normalizer = (
                jsp.special.gammaln((nu + 1) / 2)
                - jsp.special.gammaln(nu / 2)
                - 0.5 * jnp.log(jnp.pi * (nu - 2))
            )
            if density == 2:
                lam = shape[1]
                a = 4 * lam * jnp.exp(normalizer) * (nu - 2) / (nu - 1)
                b = jnp.sqrt(1 + 3 * lam * lam - a * a)
                transformed = b * z + a
                z = transformed / (1 + lam * jnp.where(transformed >= 0, 1.0, -1.0))
                normalizer += jnp.log(b)
            return (
                scale
                - len(residuals) * normalizer
                + (nu + 1) / 2 * jnp.sum(jnp.log1p(z * z / (nu - 2)))
            )
        if density == 3:
            nu = shape[0]
            log_scale = 0.5 * (
                jsp.special.gammaln(1 / nu) - jsp.special.gammaln(3 / nu)
            )
            normalizer = (
                jnp.log(nu) - jnp.log(2.0) - log_scale - jsp.special.gammaln(1 / nu)
            )
            if mean_count == 0:
                # With no mean parameters, an observed zero stays exactly zero
                # for every variance and shape. Avoid differentiating log(0).
                nonzero = jnp.asarray(np.asarray(y[first:]) != 0)
                log_absolute = jnp.log(jnp.abs(jnp.where(nonzero, z, 1.0)))
                powers = jnp.where(
                    nonzero, jnp.exp(nu * (log_absolute - log_scale)), 0.0
                )
            else:
                powers = (jnp.abs(z) * jnp.exp(-log_scale)) ** nu
            return scale - len(residuals) * normalizer + jnp.sum(powers)
        if density == 7:
            return scale - jnp.sum(_jax_vg_logpdf(jnp, jsp, z, shape[0], shape[1]))
        lam = shape[2] if density == 4 else (-0.5 if density == 5 else 1.0)
        return scale - jnp.sum(_jax_gh_logpdf(jnp, jsp, z, shape[0], shape[1], lam))

    point = jnp.asarray(theta)
    return (
        float(objective(point)),
        np.asarray(jax.grad(objective)(point)),
        np.asarray(jax.hessian(objective)(point)),
    )
