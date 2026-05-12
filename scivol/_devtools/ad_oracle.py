"""
JAX-based AD oracles for development-time derivative validation.

These helpers are intentionally separate from the production kernels. They are
used to validate analytical C gradients and Hessians against an independent AD
reference without introducing any runtime dependency in the fast path.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Tuple

import numpy as np
from numpy.typing import NDArray


GJR_H_FLOOR = 1e-12
DCC_H_FLOOR = 1e-12
SOFTPLUS_THRESHOLD = 20.0


@lru_cache(maxsize=1)
def _jax_modules():
    try:
        import jax
        import jax.numpy as jnp
        import jax.scipy as jsp
        from jax import config
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "JAX is required for AD-based derivative validation. "
            "Install development extras via `pip install -e .[dev]`."
        ) from exc

    config.update("jax_enable_x64", True)
    return jax, jnp, jsp


def _value_grad_hess(
    objective,
    theta: NDArray[np.float64],
) -> Tuple[float, NDArray[np.float64], NDArray[np.float64]]:
    jax, jnp, _ = _jax_modules()
    theta_j = jnp.asarray(theta, dtype=jnp.float64)
    value = float(objective(theta_j))
    grad = np.asarray(jax.grad(objective)(theta_j), dtype=np.float64)
    hess = np.asarray(jax.hessian(objective)(theta_j), dtype=np.float64)
    return value, grad, hess


def _softplus_production(jnp, x):
    """Mirror the production softplus thresholding used in transform code."""
    return jnp.where(x > SOFTPLUS_THRESHOLD, x, jnp.log1p(jnp.exp(x)))


@lru_cache(maxsize=1)
def _egarch_skewt_quadrature_rule() -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    nodes, weights = np.polynomial.legendre.leggauss(256)
    return nodes.astype(np.float64), weights.astype(np.float64)


def garch_value_grad_hess(
    theta: NDArray[np.float64],
    resid2: NDArray[np.float64],
    p: int,
    q: int,
    dist: str = "normal",
    resid: NDArray[np.float64] | None = None,
) -> Tuple[float, NDArray[np.float64], NDArray[np.float64]]:
    """Reference NLL, gradient, and Hessian for GARCH(p,q)."""
    jax, jnp, jsp = _jax_modules()
    resid2_j = jnp.asarray(resid2, dtype=jnp.float64)
    resid_j = None if resid is None else jnp.asarray(resid, dtype=jnp.float64)
    n = int(resid2.shape[0])

    def objective(theta_j):
        omega = theta_j[0]
        alpha = theta_j[1:1 + p]
        beta = theta_j[1 + p:1 + p + q]
        sigma0 = jnp.mean(resid2_j)
        sigma_hist0 = jnp.full((max(q, 1),), sigma0, dtype=theta_j.dtype)

        def step(sigma_hist, t):
            sigma_t = omega
            for j in range(p):
                sigma_t = sigma_t + jnp.where(t > j, alpha[j] * resid2_j[t - 1 - j], 0.0)
            for k in range(q):
                sigma_t = sigma_t + jnp.where(t > k, beta[k] * sigma_hist[k], 0.0)
            sigma_hist_next = jnp.concatenate([sigma_t[None], sigma_hist[:-1]])
            return sigma_hist_next, sigma_t

        _, sigma_tail = jax.lax.scan(step, sigma_hist0, jnp.arange(1, n))
        sigma2 = jnp.concatenate([sigma0[None], sigma_tail])

        if dist == "normal":
            return 0.5 * jnp.sum(jnp.log(sigma2) + resid2_j / sigma2)

        if dist == "studentt":
            nu = theta_j[1 + p + q]
            inv_nu_m2 = 1.0 / (nu - 2.0)
            constant = n * (
                jsp.special.gammaln(0.5 * (nu + 1.0))
                - jsp.special.gammaln(0.5 * nu)
                - 0.5 * jnp.log(jnp.pi / inv_nu_m2)
            )
            tail = jnp.log1p(resid2_j / sigma2 * inv_nu_m2)
            return 0.5 * (jnp.sum(jnp.log(sigma2)) + (nu + 1.0) * jnp.sum(tail)) - constant

        if dist == "ged":
            nu = theta_j[1 + p + q]
            log_scale = 0.5 * (jsp.special.gammaln(1.0 / nu) - jsp.special.gammaln(3.0 / nu))
            log_const = jnp.log(nu) - jnp.log(2.0) - log_scale - jsp.special.gammaln(1.0 / nu)
            q_terms = jnp.exp(
                nu * (0.5 * jnp.log(jnp.maximum(resid2_j, 1e-300)) - 0.5 * jnp.log(sigma2) - log_scale)
            )
            return jnp.sum(-log_const + 0.5 * jnp.log(sigma2) + q_terms)

        if dist == "skewt":
            if resid_j is None:
                raise ValueError("Raw residuals are required for GARCH+SkewT AD validation.")
            nu = theta_j[1 + p + q]
            lam = theta_j[2 + p + q]
            c_log = (
                jsp.special.gammaln(0.5 * (nu + 1.0))
                - jsp.special.gammaln(0.5 * nu)
                - 0.5 * jnp.log(jnp.pi * (nu - 2.0))
            )
            a = 4.0 * lam * jnp.exp(c_log) * (nu - 2.0) / (nu - 1.0)
            b = jnp.sqrt(1.0 + 3.0 * lam * lam - a * a)
            z = resid_j / jnp.sqrt(sigma2)
            bz_plus_a = b * z + a
            sign_term = jnp.where(bz_plus_a >= 0.0, 1.0, -1.0)
            z_adj = bz_plus_a / (1.0 - lam * sign_term)
            ll = (
                n * c_log
                - 0.5 * jnp.sum(jnp.log(sigma2))
                + jnp.log(b)
                - 0.5 * (nu + 1.0) * jnp.sum(jnp.log1p(z_adj * z_adj / (nu - 2.0)))
            )
            return -ll

        raise ValueError(f"Unsupported GARCH distribution: {dist}")

    return _value_grad_hess(objective, theta)


def linear_mean_garch_value_grad_hess(
    theta: NDArray[np.float64],
    y: NDArray[np.float64],
    features: NDArray[np.float64],
    p: int,
    q: int,
    dist: str = "normal",
) -> Tuple[float, NDArray[np.float64], NDArray[np.float64]]:
    """Reference NLL, gradient, and Hessian for linked linear-mean GARCH(p,q)."""
    jax, jnp, jsp = _jax_modules()
    y_j = jnp.asarray(y, dtype=jnp.float64)
    features_j = jnp.asarray(features, dtype=jnp.float64)
    n = int(y.shape[0])
    n_mean = int(features.shape[1])

    def objective(theta_j):
        beta_mean = theta_j[:n_mean]
        omega = theta_j[n_mean]
        alpha = theta_j[n_mean + 1:n_mean + 1 + p]
        beta = theta_j[n_mean + 1 + p:n_mean + 1 + p + q]

        mean = features_j @ beta_mean
        resid = y_j - mean
        resid2 = resid * resid
        if p == 0 and q == 0:
            sigma2 = jnp.full((n,), omega, dtype=theta_j.dtype)
            dist_offset = n_mean + 1
        else:
            sigma0 = jnp.mean(y_j * y_j)
            sigma_hist0 = jnp.full((max(q, 1),), sigma0, dtype=theta_j.dtype)

            def step(sigma_hist, t):
                sigma_t = omega
                for j in range(p):
                    sigma_t = sigma_t + jnp.where(t > j, alpha[j] * resid2[t - 1 - j], 0.0)
                for k in range(q):
                    sigma_t = sigma_t + jnp.where(t > k, beta[k] * sigma_hist[k], 0.0)
                sigma_hist_next = jnp.concatenate([sigma_t[None], sigma_hist[:-1]])
                return sigma_hist_next, sigma_t

            _, sigma_tail = jax.lax.scan(step, sigma_hist0, jnp.arange(1, n))
            sigma2 = jnp.concatenate([sigma0[None], sigma_tail])
            dist_offset = n_mean + 1 + p + q

        if dist == "normal":
            return 0.5 * jnp.sum(jnp.log(sigma2) + resid2 / sigma2)

        if dist == "studentt":
            nu = theta_j[dist_offset]
            inv_nu_m2 = 1.0 / (nu - 2.0)
            constant = n * (
                jsp.special.gammaln(0.5 * (nu + 1.0))
                - jsp.special.gammaln(0.5 * nu)
                - 0.5 * jnp.log(jnp.pi / inv_nu_m2)
            )
            tail = jnp.log1p(resid2 / sigma2 * inv_nu_m2)
            return 0.5 * (jnp.sum(jnp.log(sigma2)) + (nu + 1.0) * jnp.sum(tail)) - constant

        if dist == "ged":
            nu = theta_j[dist_offset]
            log_scale = 0.5 * (jsp.special.gammaln(1.0 / nu) - jsp.special.gammaln(3.0 / nu))
            log_const = jnp.log(nu) - jnp.log(2.0) - log_scale - jsp.special.gammaln(1.0 / nu)
            q_terms = jnp.exp(
                nu * (0.5 * jnp.log(jnp.maximum(resid2, 1e-300)) - 0.5 * jnp.log(sigma2) - log_scale)
            )
            return jnp.sum(-log_const + 0.5 * jnp.log(sigma2) + q_terms)

        if dist == "skewt":
            nu = theta_j[dist_offset]
            lam = theta_j[dist_offset + 1]
            c_log = (
                jsp.special.gammaln(0.5 * (nu + 1.0))
                - jsp.special.gammaln(0.5 * nu)
                - 0.5 * jnp.log(jnp.pi * (nu - 2.0))
            )
            a = 4.0 * lam * jnp.exp(c_log) * (nu - 2.0) / (nu - 1.0)
            b = jnp.sqrt(1.0 + 3.0 * lam * lam - a * a)
            z = resid / jnp.sqrt(sigma2)
            bz_plus_a = b * z + a
            sign_term = jnp.where(bz_plus_a >= 0.0, 1.0, -1.0)
            z_adj = bz_plus_a / (1.0 - lam * sign_term)
            ll = (
                n * c_log
                - 0.5 * jnp.sum(jnp.log(sigma2))
                + n * jnp.log(b)
                - 0.5 * (nu + 1.0) * jnp.sum(jnp.log1p(z_adj * z_adj / (nu - 2.0)))
            )
            return -ll

        raise ValueError(f"Unsupported linked linear-mean GARCH distribution: {dist}")

    return _value_grad_hess(objective, theta)


def linear_mean_garch_normal_value_grad_hess(
    theta: NDArray[np.float64],
    y: NDArray[np.float64],
    features: NDArray[np.float64],
    p: int,
    q: int,
) -> Tuple[float, NDArray[np.float64], NDArray[np.float64]]:
    return linear_mean_garch_value_grad_hess(theta, y, features, p, q, dist="normal")


def linear_mean_garch_studentt_value_grad_hess(
    theta: NDArray[np.float64],
    y: NDArray[np.float64],
    features: NDArray[np.float64],
    p: int,
    q: int,
) -> Tuple[float, NDArray[np.float64], NDArray[np.float64]]:
    return linear_mean_garch_value_grad_hess(theta, y, features, p, q, dist="studentt")


def linear_mean_garch_skewt_value_grad_hess(
    theta: NDArray[np.float64],
    y: NDArray[np.float64],
    features: NDArray[np.float64],
    p: int,
    q: int,
) -> Tuple[float, NDArray[np.float64], NDArray[np.float64]]:
    return linear_mean_garch_value_grad_hess(theta, y, features, p, q, dist="skewt")


def linear_mean_garch_ged_value_grad_hess(
    theta: NDArray[np.float64],
    y: NDArray[np.float64],
    features: NDArray[np.float64],
    p: int,
    q: int,
) -> Tuple[float, NDArray[np.float64], NDArray[np.float64]]:
    return linear_mean_garch_value_grad_hess(theta, y, features, p, q, dist="ged")


def linear_mean_normal_value_grad_hess(
    theta: NDArray[np.float64],
    y: NDArray[np.float64],
    features: NDArray[np.float64],
) -> Tuple[float, NDArray[np.float64], NDArray[np.float64]]:
    """Reference concentrated-NLL, gradient, and Hessian for standalone linear-mean + Normal."""
    _, jnp, _ = _jax_modules()
    y_j = jnp.asarray(y, dtype=jnp.float64)
    features_j = jnp.asarray(features, dtype=jnp.float64)
    n = int(y.shape[0])

    def objective(theta_j):
        resid = y_j - features_j @ theta_j
        sigma2 = jnp.maximum(jnp.mean(resid * resid), 1e-12)
        return 0.5 * n * (1.0 + jnp.log(2.0 * jnp.pi) + jnp.log(sigma2))

    return _value_grad_hess(objective, theta)


def linear_mean_normal_log_value_grad_hess(
    z: NDArray[np.float64],
    y: NDArray[np.float64],
    features: NDArray[np.float64],
    scales: NDArray[np.float64],
) -> Tuple[float, NDArray[np.float64], NDArray[np.float64]]:
    """Reference transformed objective for standalone linear-mean + Normal."""
    _, jnp, _ = _jax_modules()
    z_j = jnp.asarray(z, dtype=jnp.float64)
    y_j = jnp.asarray(y, dtype=jnp.float64)
    features_j = jnp.asarray(features, dtype=jnp.float64)
    scales_j = jnp.asarray(scales, dtype=jnp.float64)
    n = int(y.shape[0])

    def objective(z_local):
        theta_j = scales_j * jnp.tanh(z_local)
        resid = y_j - features_j @ theta_j
        sigma2 = jnp.maximum(jnp.mean(resid * resid), 1e-12)
        return 0.5 * n * (1.0 + jnp.log(2.0 * jnp.pi) + jnp.log(sigma2))

    return _value_grad_hess(objective, np.asarray(z_j, dtype=np.float64))


def linear_mean_gjr_garch_value_grad_hess(
    theta: NDArray[np.float64],
    y: NDArray[np.float64],
    features: NDArray[np.float64],
    p: int,
    q: int,
    dist: str = "normal",
) -> Tuple[float, NDArray[np.float64], NDArray[np.float64]]:
    """Reference NLL, gradient, and Hessian for linked linear-mean GJR-GARCH(p,q)."""
    jax, jnp, jsp = _jax_modules()
    y_j = jnp.asarray(y, dtype=jnp.float64)
    features_j = jnp.asarray(features, dtype=jnp.float64)
    n = int(y.shape[0])
    n_mean = int(features.shape[1])

    def objective(theta_j):
        beta_mean = theta_j[:n_mean]
        omega = theta_j[n_mean]
        alpha = theta_j[n_mean + 1:n_mean + 1 + p]
        gamma = theta_j[n_mean + 1 + p:n_mean + 1 + 2 * p]
        beta = theta_j[n_mean + 1 + 2 * p:n_mean + 1 + 2 * p + q]

        mean = features_j @ beta_mean
        resid = y_j - mean
        resid2 = resid * resid
        sigma0 = jnp.mean(y_j * y_j)
        sigma_hist0 = jnp.full((max(q, 1),), sigma0, dtype=theta_j.dtype)

        def step(sigma_hist, t):
            sigma_t = omega
            for j in range(p):
                e_lag = jnp.where(t > j, resid[t - 1 - j], 0.0)
                e2_lag = e_lag * e_lag
                ind = jnp.where(e_lag < 0.0, 1.0, 0.0)
                sigma_t = sigma_t + jnp.where(t > j, alpha[j] * e2_lag + gamma[j] * ind * e2_lag, 0.0)
            for k in range(q):
                sigma_t = sigma_t + jnp.where(t > k, beta[k] * sigma_hist[k], 0.0)
            sigma_hist_next = jnp.concatenate([sigma_t[None], sigma_hist[:-1]])
            return sigma_hist_next, sigma_t

        _, sigma_tail = jax.lax.scan(step, sigma_hist0, jnp.arange(1, n))
        sigma2 = jnp.concatenate([sigma0[None], sigma_tail])

        dist_offset = n_mean + 1 + 2 * p + q
        if dist == "normal":
            return 0.5 * jnp.sum(jnp.log(sigma2) + resid2 / sigma2)

        if dist == "studentt":
            nu = theta_j[dist_offset]
            inv_nu_m2 = 1.0 / (nu - 2.0)
            constant = n * (
                jsp.special.gammaln(0.5 * (nu + 1.0))
                - jsp.special.gammaln(0.5 * nu)
                - 0.5 * jnp.log(jnp.pi / inv_nu_m2)
            )
            tail = jnp.log1p(resid2 / sigma2 * inv_nu_m2)
            return 0.5 * (jnp.sum(jnp.log(sigma2)) + (nu + 1.0) * jnp.sum(tail)) - constant

        if dist == "skewt":
            nu = theta_j[dist_offset]
            lam = theta_j[dist_offset + 1]
            c_log = (
                jsp.special.gammaln(0.5 * (nu + 1.0))
                - jsp.special.gammaln(0.5 * nu)
                - 0.5 * jnp.log(jnp.pi * (nu - 2.0))
            )
            a = 4.0 * lam * jnp.exp(c_log) * (nu - 2.0) / (nu - 1.0)
            b = jnp.sqrt(1.0 + 3.0 * lam * lam - a * a)
            z = resid / jnp.sqrt(sigma2)
            bz_plus_a = b * z + a
            sign_term = jnp.where(bz_plus_a >= 0.0, 1.0, -1.0)
            z_adj = bz_plus_a / (1.0 - lam * sign_term)
            ll = (
                n * c_log
                - 0.5 * jnp.sum(jnp.log(sigma2))
                + n * jnp.log(b)
                - 0.5 * (nu + 1.0) * jnp.sum(jnp.log1p(z_adj * z_adj / (nu - 2.0)))
            )
            return -ll

        if dist == "ged":
            nu = theta_j[dist_offset]
            log_scale = 0.5 * (jsp.special.gammaln(1.0 / nu) - jsp.special.gammaln(3.0 / nu))
            log_const = jnp.log(nu) - jnp.log(2.0) - log_scale - jsp.special.gammaln(1.0 / nu)
            z = jnp.abs(resid) / (jnp.sqrt(sigma2) * jnp.exp(log_scale))
            return -n * log_const + 0.5 * jnp.sum(jnp.log(sigma2)) + jnp.sum(jnp.power(z, nu))

        raise ValueError(f"Unsupported linked linear-mean GJR-GARCH distribution: {dist}")

    return _value_grad_hess(objective, theta)


def linear_mean_gjr_garch_normal_value_grad_hess(
    theta: NDArray[np.float64],
    y: NDArray[np.float64],
    features: NDArray[np.float64],
    p: int,
    q: int,
) -> Tuple[float, NDArray[np.float64], NDArray[np.float64]]:
    return linear_mean_gjr_garch_value_grad_hess(theta, y, features, p, q, dist="normal")


def linear_mean_gjr_garch_studentt_value_grad_hess(
    theta: NDArray[np.float64],
    y: NDArray[np.float64],
    features: NDArray[np.float64],
    p: int,
    q: int,
) -> Tuple[float, NDArray[np.float64], NDArray[np.float64]]:
    return linear_mean_gjr_garch_value_grad_hess(theta, y, features, p, q, dist="studentt")


def linear_mean_gjr_garch_skewt_value_grad_hess(
    theta: NDArray[np.float64],
    y: NDArray[np.float64],
    features: NDArray[np.float64],
    p: int,
    q: int,
) -> Tuple[float, NDArray[np.float64], NDArray[np.float64]]:
    return linear_mean_gjr_garch_value_grad_hess(theta, y, features, p, q, dist="skewt")


def linear_mean_egarch_value_grad_hess(
    theta: NDArray[np.float64],
    y: NDArray[np.float64],
    features: NDArray[np.float64],
    p: int,
    q: int,
    dist: str = "normal",
) -> Tuple[float, NDArray[np.float64], NDArray[np.float64]]:
    """Reference NLL, gradient, and Hessian for linked linear-mean EGARCH(p,q)."""
    jax, jnp, jsp = _jax_modules()
    y_j = jnp.asarray(y, dtype=jnp.float64)
    features_j = jnp.asarray(features, dtype=jnp.float64)
    n = int(y.shape[0])
    n_mean = int(features.shape[1])
    quad_nodes_np, quad_weights_np = _egarch_skewt_quadrature_rule()
    quad_nodes = jnp.asarray(quad_nodes_np, dtype=jnp.float64)
    quad_weights = jnp.asarray(quad_weights_np, dtype=jnp.float64)

    def studentt_abs_moment(nu):
        c_log = (
            jsp.special.gammaln(0.5 * (nu + 1.0))
            - jsp.special.gammaln(0.5 * nu)
            - 0.5 * jnp.log(jnp.pi * (nu - 2.0))
        )
        return 2.0 * jnp.exp(c_log) * (nu - 2.0) / (nu - 1.0)

    def ged_abs_moment(nu):
        log_scale = 0.5 * (jsp.special.gammaln(1.0 / nu) - jsp.special.gammaln(3.0 / nu))
        return jnp.exp(log_scale + jsp.special.gammaln(2.0 / nu) - jsp.special.gammaln(1.0 / nu))

    def skewt_abs_moment(nu, lam):
        bound = 50.0
        c_log = (
            jsp.special.gammaln(0.5 * (nu + 1.0))
            - jsp.special.gammaln(0.5 * nu)
            - 0.5 * jnp.log(jnp.pi * (nu - 2.0))
        )
        a = 4.0 * lam * jnp.exp(c_log) * (nu - 2.0) / (nu - 1.0)
        b = jnp.sqrt(1.0 + 3.0 * lam * lam - a * a)
        split = -a / b

        def panel(lo, hi):
            center = 0.5 * (hi + lo)
            half_width = 0.5 * (hi - lo)
            z = center + half_width * quad_nodes
            w = half_width * quad_weights
            bz_plus_a = b * z + a
            denom = jnp.where(bz_plus_a < 0.0, 1.0 + lam, 1.0 - lam)
            z_adj = bz_plus_a / denom
            log_pdf = c_log + jnp.log(b) - 0.5 * (nu + 1.0) * jnp.log1p(z_adj * z_adj / (nu - 2.0))
            return jnp.sum(w * jnp.abs(z) * jnp.exp(log_pdf))

        return jnp.where(
            jnp.logical_and(split > -bound, split < bound),
            panel(-bound, split) + panel(split, bound),
            panel(-bound, bound),
        )

    def objective(theta_j):
        beta_mean = theta_j[:n_mean]
        omega = theta_j[n_mean]
        alpha = theta_j[n_mean + 1:n_mean + 1 + p]
        gamma = theta_j[n_mean + 1 + p:n_mean + 1 + 2 * p]
        beta = theta_j[n_mean + 1 + 2 * p:n_mean + 1 + 2 * p + q]
        dist_offset = n_mean + 1 + 2 * p + q

        mean = features_j @ beta_mean
        resid = y_j - mean
        resid2 = resid * resid
        sigma0 = jnp.maximum(jnp.mean(y_j * y_j), GJR_H_FLOOR)
        logh0 = jnp.log(sigma0)

        if dist == "normal":
            abs_moment = jnp.sqrt(2.0 / jnp.pi)
        elif dist == "studentt":
            nu = theta_j[dist_offset]
            abs_moment = studentt_abs_moment(nu)
        elif dist == "ged":
            nu = theta_j[dist_offset]
            abs_moment = ged_abs_moment(nu)
        elif dist == "skewt":
            nu = theta_j[dist_offset]
            lam = theta_j[dist_offset + 1]
            abs_moment = skewt_abs_moment(nu, lam)
        else:
            raise ValueError(f"Unsupported linked linear-mean EGARCH distribution: {dist}")

        z0 = resid[0] / jnp.sqrt(sigma0)
        x_buf0 = jnp.concatenate(
            [jnp.array([logh0], dtype=theta_j.dtype), jnp.zeros((max(q - 1, 0),), dtype=theta_j.dtype)]
        )
        z_buf0 = jnp.concatenate(
            [jnp.array([z0], dtype=theta_j.dtype), jnp.zeros((max(p - 1, 0),), dtype=theta_j.dtype)]
        )

        def step(carry, t):
            x_buf, z_buf = carry
            logh_t = omega
            for i in range(p):
                term = alpha[i] * (jnp.abs(z_buf[i]) - abs_moment) + gamma[i] * z_buf[i]
                logh_t = logh_t + jnp.where(t > i, term, 0.0)
            for j in range(q):
                logh_t = logh_t + jnp.where(t > j, beta[j] * x_buf[j], 0.0)

            h_t = jnp.maximum(jnp.exp(logh_t), GJR_H_FLOOR)
            z_t = resid[t] / jnp.sqrt(h_t)
            if q > 0:
                x_buf = jnp.concatenate([jnp.array([jnp.log(h_t)], dtype=theta_j.dtype), x_buf[: q - 1]], axis=0)
            if p > 0:
                z_buf = jnp.concatenate([jnp.array([z_t], dtype=theta_j.dtype), z_buf[: p - 1]], axis=0)
            return (x_buf, z_buf), h_t

        (_, _), sigma_tail = jax.lax.scan(step, (x_buf0, z_buf0), jnp.arange(1, n))
        sigma2 = jnp.concatenate([sigma0[None], sigma_tail])

        if dist == "normal":
            return 0.5 * jnp.sum(jnp.log(sigma2) + resid2 / sigma2)

        if dist == "studentt":
            nu = theta_j[dist_offset]
            inv_nu_m2 = 1.0 / (nu - 2.0)
            constant = n * (
                jsp.special.gammaln(0.5 * (nu + 1.0))
                - jsp.special.gammaln(0.5 * nu)
                - 0.5 * jnp.log(jnp.pi / inv_nu_m2)
            )
            tail = jnp.log1p(resid2 / sigma2 * inv_nu_m2)
            return 0.5 * (jnp.sum(jnp.log(sigma2)) + (nu + 1.0) * jnp.sum(tail)) - constant

        if dist == "ged":
            nu = theta_j[dist_offset]
            log_scale = 0.5 * (jsp.special.gammaln(1.0 / nu) - jsp.special.gammaln(3.0 / nu))
            log_const = jnp.log(nu) - jnp.log(2.0) - log_scale - jsp.special.gammaln(1.0 / nu)
            z_std = jnp.abs(resid) / (jnp.sqrt(sigma2) * jnp.exp(log_scale))
            return -n * log_const + 0.5 * jnp.sum(jnp.log(sigma2)) + jnp.sum(jnp.power(z_std, nu))

        nu = theta_j[dist_offset]
        lam = theta_j[dist_offset + 1]
        c_log = (
            jsp.special.gammaln(0.5 * (nu + 1.0))
            - jsp.special.gammaln(0.5 * nu)
            - 0.5 * jnp.log(jnp.pi * (nu - 2.0))
        )
        a = 4.0 * lam * jnp.exp(c_log) * (nu - 2.0) / (nu - 1.0)
        b = jnp.sqrt(1.0 + 3.0 * lam * lam - a * a)
        z = resid / jnp.sqrt(sigma2)
        bz_plus_a = b * z + a
        sign_term = jnp.where(bz_plus_a >= 0.0, 1.0, -1.0)
        z_adj = bz_plus_a / (1.0 - lam * sign_term)
        ll = (
            n * c_log
            - 0.5 * jnp.sum(jnp.log(sigma2))
            + n * jnp.log(b)
            - 0.5 * (nu + 1.0) * jnp.sum(jnp.log1p(z_adj * z_adj / (nu - 2.0)))
        )
        return -ll

    return _value_grad_hess(objective, theta)


def gjr_garch_value_grad_hess(
    theta: NDArray[np.float64],
    resid: NDArray[np.float64],
    p: int,
    q: int,
    dist: str = "normal",
) -> Tuple[float, NDArray[np.float64], NDArray[np.float64]]:
    """Reference NLL, gradient, and Hessian for GJR-GARCH(p,q)."""
    jax, jnp, jsp = _jax_modules()
    resid_j = jnp.asarray(resid, dtype=jnp.float64)
    resid2_j = resid_j * resid_j
    n = int(resid.shape[0])

    def objective(theta_j):
        omega = theta_j[0]
        alpha = theta_j[1:1 + p]
        gamma = theta_j[1 + p:1 + 2 * p]
        beta = theta_j[1 + 2 * p:1 + 2 * p + q]
        sigma0 = jnp.mean(resid2_j)
        sigma_hist0 = jnp.full((max(q, 1),), sigma0, dtype=theta_j.dtype)

        def step(sigma_hist, t):
            sigma_t = omega
            for j in range(p):
                e_lag = resid_j[t - 1 - j]
                e2_lag = e_lag * e_lag
                ind = jnp.where(e_lag < 0.0, 1.0, 0.0)
                sigma_t = sigma_t + jnp.where(
                    t > j,
                    alpha[j] * e2_lag + gamma[j] * ind * e2_lag,
                    0.0,
                )
            for k in range(q):
                sigma_t = sigma_t + jnp.where(t > k, beta[k] * sigma_hist[k], 0.0)
            sigma_t = jnp.maximum(sigma_t, GJR_H_FLOOR)
            sigma_hist_next = jnp.concatenate([sigma_t[None], sigma_hist[:-1]])
            return sigma_hist_next, sigma_t

        _, sigma_tail = jax.lax.scan(step, sigma_hist0, jnp.arange(1, n))
        sigma2 = jnp.concatenate([sigma0[None], sigma_tail])

        if dist == "normal":
            return 0.5 * jnp.sum(jnp.log(sigma2) + resid2_j / sigma2)

        if dist == "studentt":
            nu = theta_j[1 + 2 * p + q]
            inv_nu_m2 = 1.0 / (nu - 2.0)
            constant = n * (
                jsp.special.gammaln(0.5 * (nu + 1.0))
                - jsp.special.gammaln(0.5 * nu)
                - 0.5 * jnp.log(jnp.pi / inv_nu_m2)
            )
            tail = jnp.log1p(resid2_j / sigma2 * inv_nu_m2)
            return 0.5 * (jnp.sum(jnp.log(sigma2)) + (nu + 1.0) * jnp.sum(tail)) - constant

        if dist == "skewt":
            nu = theta_j[1 + 2 * p + q]
            lam = theta_j[2 + 2 * p + q]
            c_log = (
                jsp.special.gammaln(0.5 * (nu + 1.0))
                - jsp.special.gammaln(0.5 * nu)
                - 0.5 * jnp.log(jnp.pi * (nu - 2.0))
            )
            a = 4.0 * lam * jnp.exp(c_log) * (nu - 2.0) / (nu - 1.0)
            b = jnp.sqrt(1.0 + 3.0 * lam * lam - a * a)
            z = resid_j / jnp.sqrt(sigma2)
            bz_plus_a = b * z + a
            sign_term = jnp.where(bz_plus_a >= 0.0, 1.0, -1.0)
            z_adj = bz_plus_a / (1.0 - lam * sign_term)
            ll = (
                n * c_log
                - 0.5 * jnp.sum(jnp.log(sigma2))
                + jnp.log(b)
                - 0.5 * (nu + 1.0) * jnp.sum(jnp.log1p(z_adj * z_adj / (nu - 2.0)))
            )
            return -ll

        if dist == "ged":
            nu = theta_j[1 + 2 * p + q]
            log_scale = 0.5 * (jsp.special.gammaln(1.0 / nu) - jsp.special.gammaln(3.0 / nu))
            log_const = jnp.log(nu) - jnp.log(2.0) - log_scale - jsp.special.gammaln(1.0 / nu)
            z_std = jnp.abs(resid_j) / (jnp.sqrt(sigma2) * jnp.exp(log_scale))
            return -n * log_const + 0.5 * jnp.sum(jnp.log(sigma2)) + jnp.sum(jnp.power(z_std, nu))

        raise ValueError(f"Unsupported GJR-GARCH distribution: {dist}")

    return _value_grad_hess(objective, theta)


def egarch_value_grad_hess(
    theta: NDArray[np.float64],
    resid: NDArray[np.float64],
    dist: str = "normal",
    p: int = 1,
    q: int = 1,
) -> Tuple[float, NDArray[np.float64], NDArray[np.float64]]:
    """Reference NLL, gradient, and Hessian for EGARCH(p,q)."""
    jax, jnp, jsp = _jax_modules()
    resid_j = jnp.asarray(resid, dtype=jnp.float64)
    resid2_j = resid_j * resid_j
    n = int(resid.shape[0])
    quad_nodes_np, quad_weights_np = _egarch_skewt_quadrature_rule()
    quad_nodes = jnp.asarray(quad_nodes_np, dtype=jnp.float64)
    quad_weights = jnp.asarray(quad_weights_np, dtype=jnp.float64)

    def studentt_abs_moment(nu):
        c_log = (
            jsp.special.gammaln(0.5 * (nu + 1.0))
            - jsp.special.gammaln(0.5 * nu)
            - 0.5 * jnp.log(jnp.pi * (nu - 2.0))
        )
        return 2.0 * jnp.exp(c_log) * (nu - 2.0) / (nu - 1.0)

    def skewt_abs_moment(nu, lam):
        bound = 50.0
        c_log = (
            jsp.special.gammaln(0.5 * (nu + 1.0))
            - jsp.special.gammaln(0.5 * nu)
            - 0.5 * jnp.log(jnp.pi * (nu - 2.0))
        )
        a = 4.0 * lam * jnp.exp(c_log) * (nu - 2.0) / (nu - 1.0)
        b = jnp.sqrt(1.0 + 3.0 * lam * lam - a * a)
        split = -a / b

        def panel(lo, hi):
            center = 0.5 * (hi + lo)
            half_width = 0.5 * (hi - lo)
            z = center + half_width * quad_nodes
            w = half_width * quad_weights
            bz_plus_a = b * z + a
            denom = jnp.where(bz_plus_a < 0.0, 1.0 + lam, 1.0 - lam)
            z_adj = bz_plus_a / denom
            log_pdf = c_log + jnp.log(b) - 0.5 * (nu + 1.0) * jnp.log1p(z_adj * z_adj / (nu - 2.0))
            return jnp.sum(w * jnp.abs(z) * jnp.exp(log_pdf))

        return jnp.where(
            jnp.logical_and(split > -bound, split < bound),
            panel(-bound, split) + panel(split, bound),
            panel(-bound, bound),
        )

    def objective(theta_j):
        k_vol = 1 + 2 * p + q
        omega = theta_j[0]
        alpha = theta_j[1 : 1 + p]
        gamma = theta_j[1 + p : 1 + 2 * p]
        beta = theta_j[1 + 2 * p : k_vol]
        sigma0 = jnp.maximum(jnp.mean(resid2_j), GJR_H_FLOOR)
        logh0 = jnp.log(sigma0)

        if dist == "normal":
            abs_moment = jnp.sqrt(2.0 / jnp.pi)
        elif dist == "studentt":
            nu = theta_j[k_vol]
            abs_moment = studentt_abs_moment(nu)
        elif dist == "skewt":
            nu = theta_j[k_vol]
            lam = theta_j[k_vol + 1]
            abs_moment = skewt_abs_moment(nu, lam)
        else:
            raise ValueError(f"Unsupported EGARCH distribution: {dist}")

        z0 = resid_j[0] / jnp.sqrt(sigma0)
        x_buf0 = jnp.concatenate([jnp.array([logh0], dtype=theta_j.dtype), jnp.zeros((max(q - 1, 0),), dtype=theta_j.dtype)])
        z_buf0 = jnp.concatenate([jnp.array([z0], dtype=theta_j.dtype), jnp.zeros((max(p - 1, 0),), dtype=theta_j.dtype)])

        def step(carry, t):
            x_buf, z_buf = carry
            logh_t = omega
            for i in range(p):
                term = alpha[i] * (jnp.abs(z_buf[i]) - abs_moment) + gamma[i] * z_buf[i]
                logh_t = logh_t + jnp.where(t > i, term, 0.0)
            for j in range(q):
                logh_t = logh_t + jnp.where(t > j, beta[j] * x_buf[j], 0.0)
            h_t = jnp.maximum(jnp.exp(logh_t), GJR_H_FLOOR)
            z_t = resid_j[t] / jnp.sqrt(h_t)
            if q > 0:
                x_buf = jnp.concatenate([jnp.array([jnp.log(h_t)], dtype=theta_j.dtype), x_buf[: q - 1]], axis=0)
            if p > 0:
                z_buf = jnp.concatenate([jnp.array([z_t], dtype=theta_j.dtype), z_buf[: p - 1]], axis=0)
            return (x_buf, z_buf), h_t

        (_, _), sigma_tail = jax.lax.scan(step, (x_buf0, z_buf0), jnp.arange(1, n))
        sigma2 = jnp.concatenate([sigma0[None], sigma_tail])

        if dist == "normal":
            return 0.5 * jnp.sum(jnp.log(sigma2) + resid2_j / sigma2)

        if dist == "studentt":
            nu = theta_j[k_vol]
            inv_nu_m2 = 1.0 / (nu - 2.0)
            constant = n * (
                jsp.special.gammaln(0.5 * (nu + 1.0))
                - jsp.special.gammaln(0.5 * nu)
                - 0.5 * jnp.log(jnp.pi / inv_nu_m2)
            )
            tail = jnp.log1p(resid2_j / sigma2 * inv_nu_m2)
            return 0.5 * (jnp.sum(jnp.log(sigma2)) + (nu + 1.0) * jnp.sum(tail)) - constant

        nu = theta_j[k_vol]
        lam = theta_j[k_vol + 1]
        c_log = (
            jsp.special.gammaln(0.5 * (nu + 1.0))
            - jsp.special.gammaln(0.5 * nu)
            - 0.5 * jnp.log(jnp.pi * (nu - 2.0))
        )
        a = 4.0 * lam * jnp.exp(c_log) * (nu - 2.0) / (nu - 1.0)
        b = jnp.sqrt(1.0 + 3.0 * lam * lam - a * a)
        z = resid_j / jnp.sqrt(sigma2)
        bz_plus_a = b * z + a
        sign_term = jnp.where(bz_plus_a >= 0.0, 1.0, -1.0)
        z_adj = bz_plus_a / (1.0 - lam * sign_term)
        ll = (
            n * c_log
            - 0.5 * jnp.sum(jnp.log(sigma2))
            + jnp.log(b)
            - 0.5 * (nu + 1.0) * jnp.sum(jnp.log1p(z_adj * z_adj / (nu - 2.0)))
        )
        return -ll

    return _value_grad_hess(objective, theta)


def dcc_value_grad_hess(
    theta: NDArray[np.float64],
    eps: NDArray[np.float64],
    qbar: NDArray[np.float64],
    p: int,
    q: int,
) -> Tuple[float, NDArray[np.float64], NDArray[np.float64]]:
    """Reference Gaussian DCC(p,q) NLL, gradient, and Hessian."""
    jax, jnp, jsp = _jax_modules()
    eps_j = jnp.asarray(eps, dtype=jnp.float64)
    qbar_j = jnp.asarray(qbar, dtype=jnp.float64)
    t_count = int(eps.shape[0])
    n_series = int(eps.shape[1])

    def objective(theta_j):
        a = theta_j[:p]
        b = theta_j[p:]
        coeff = 1.0 - jnp.sum(theta_j)

        q_buf0 = jnp.tile(qbar_j[None, :, :], (max(q, 1), 1, 1))
        eps_buf0 = jnp.zeros((max(p, 1), n_series), dtype=theta_j.dtype)

        def step(carry, e):
            q_buf, eps_buf = carry
            q_t = coeff * qbar_j
            for i in range(p):
                q_t = q_t + a[i] * jnp.outer(eps_buf[i], eps_buf[i])
            for j in range(q):
                q_t = q_t + b[j] * q_buf[j]

            d = jnp.sqrt(jnp.maximum(jnp.diag(q_t), DCC_H_FLOOR))
            r_t = q_t / jnp.outer(d, d)
            chol = jnp.linalg.cholesky(r_t)
            y = jsp.linalg.solve_triangular(chol, e, lower=True)
            v = jsp.linalg.solve_triangular(chol.T, y, lower=False)
            contrib = 0.5 * (
                2.0 * jnp.sum(jnp.log(jnp.diag(chol)))
                + e @ v
                - e @ e
            )

            if p > 0:
                eps_buf = jnp.concatenate([e[None, :], eps_buf[:p - 1]], axis=0)
            if q > 0:
                q_buf = jnp.concatenate([q_t[None, :, :], q_buf[:q - 1]], axis=0)
            return (q_buf, eps_buf), contrib

        (_, _), contribs = jax.lax.scan(step, (q_buf0, eps_buf0), eps_j)
        return jnp.sum(contribs) / t_count

    return _value_grad_hess(objective, theta)


def arma_normal_value_grad_hess(
    theta: NDArray[np.float64],
    y: NDArray[np.float64],
    p: int,
    q: int,
) -> Tuple[float, NDArray[np.float64], NDArray[np.float64]]:
    """Reference concentrated-NLL, gradient, and Hessian for ARMA(p,q)+Normal."""
    jax, jnp, _ = _jax_modules()
    y_j = jnp.asarray(y, dtype=jnp.float64)
    n = int(y.shape[0])
    max_lag = max(p, q, 1)

    def objective(theta_j):
        c = theta_j[0]
        phi = theta_j[1:1 + p]
        theta_ma = theta_j[1 + p:1 + p + q]

        invalid = jnp.logical_or(jnp.any(jnp.abs(phi) >= 1.0), jnp.any(jnp.abs(theta_ma) >= 1.0))

        resid_hist0 = jnp.zeros((max_lag,), dtype=theta_j.dtype)

        def step(resid_hist, t):
            e_t = y_j[t] - c
            for i in range(p):
                e_t = e_t - phi[i] * y_j[t - 1 - i]
            for j in range(q):
                e_t = e_t - theta_ma[j] * resid_hist[j]
            resid_hist_next = jnp.concatenate([e_t[None], resid_hist[:-1]])
            return resid_hist_next, e_t

        _, resid_tail = jax.lax.scan(step, resid_hist0, jnp.arange(max_lag, n))
        n_eff = n - max_lag
        sigma2 = jnp.maximum(jnp.sum(resid_tail * resid_tail) / n_eff, 1e-20)
        nll = 0.5 * (jnp.log(sigma2) + 1.0)
        return jnp.where(invalid, 1e10, nll)

    return _value_grad_hess(objective, theta)


def arma_normal_logspace_value_grad_hess(
    z: NDArray[np.float64],
    y: NDArray[np.float64],
    p: int,
    q: int,
) -> Tuple[float, NDArray[np.float64], NDArray[np.float64]]:
    """Reference value/gradient/Hessian for the transformed ARMA(p,q)+Normal objective."""
    jax, jnp, _ = _jax_modules()
    z_j = jnp.asarray(z, dtype=jnp.float64)
    y_j = jnp.asarray(y, dtype=jnp.float64)
    n = int(y.shape[0])
    max_lag = max(p, q, 1)

    def objective(z_theta):
        c = z_theta[0]
        phi = 0.99 * jnp.tanh(z_theta[1:1 + p])
        theta_ma = 0.99 * jnp.tanh(z_theta[1 + p:1 + p + q])

        resid_hist0 = jnp.zeros((max_lag,), dtype=z_theta.dtype)

        def step(resid_hist, t):
            e_t = y_j[t] - c
            for i in range(p):
                e_t = e_t - phi[i] * y_j[t - 1 - i]
            for j in range(q):
                e_t = e_t - theta_ma[j] * resid_hist[j]
            resid_hist_next = jnp.concatenate([e_t[None], resid_hist[:-1]])
            return resid_hist_next, e_t

        _, resid_tail = jax.lax.scan(step, resid_hist0, jnp.arange(max_lag, n))
        n_eff = n - max_lag
        sigma2 = jnp.maximum(jnp.sum(resid_tail * resid_tail) / n_eff, 1e-20)
        return 0.5 * (jnp.log(sigma2) + 1.0)

    return _value_grad_hess(objective, np.asarray(z_j, dtype=np.float64))


def arma_ged_value_grad_hess(
    theta: NDArray[np.float64],
    y: NDArray[np.float64],
    p: int,
    q: int,
) -> Tuple[float, NDArray[np.float64], NDArray[np.float64]]:
    """Reference NLL, gradient, and Hessian for ARMA(p,q)+GED with explicit sigma2."""
    jax, jnp, jsp = _jax_modules()
    y_j = jnp.asarray(y, dtype=jnp.float64)
    n = int(y.shape[0])
    max_lag = max(p, q, 1)

    def objective(theta_j):
        c = theta_j[0]
        phi = theta_j[1:1 + p]
        theta_ma = theta_j[1 + p:1 + p + q]
        sigma2 = theta_j[1 + p + q]
        nu = theta_j[2 + p + q]

        invalid = jnp.logical_or(jnp.any(jnp.abs(phi) >= 1.0), jnp.any(jnp.abs(theta_ma) >= 1.0))
        invalid = jnp.logical_or(invalid, sigma2 <= 1e-12)
        invalid = jnp.logical_or(invalid, nu <= 1.01)

        resid_hist0 = jnp.zeros((max_lag,), dtype=theta_j.dtype)

        def step(resid_hist, t):
            e_t = y_j[t] - c
            for i in range(p):
                e_t = e_t - phi[i] * y_j[t - 1 - i]
            for j in range(q):
                e_t = e_t - theta_ma[j] * resid_hist[j]
            resid_hist_next = jnp.concatenate([e_t[None], resid_hist[:-1]])
            return resid_hist_next, e_t

        _, resid_tail = jax.lax.scan(step, resid_hist0, jnp.arange(max_lag, n))
        log_scale = 0.5 * (jsp.special.gammaln(1.0 / nu) - jsp.special.gammaln(3.0 / nu))
        log_const = jnp.log(nu) - jnp.log(2.0) - log_scale - jsp.special.gammaln(1.0 / nu)
        q_terms = jnp.exp(nu * (jnp.log(jnp.maximum(jnp.abs(resid_tail), 1e-300)) - 0.5 * jnp.log(sigma2) - log_scale))
        nll = jnp.mean(-log_const + 0.5 * jnp.log(sigma2) + q_terms)
        return jnp.where(invalid, 1e10, nll)

    return _value_grad_hess(objective, theta)


def arma_ged_logspace_value_grad_hess(
    z: NDArray[np.float64],
    y: NDArray[np.float64],
    p: int,
    q: int,
) -> Tuple[float, NDArray[np.float64], NDArray[np.float64]]:
    """Reference value/gradient/Hessian for the transformed ARMA(p,q)+GED objective."""
    jax, jnp, jsp = _jax_modules()
    z_j = jnp.asarray(z, dtype=jnp.float64)
    y_j = jnp.asarray(y, dtype=jnp.float64)
    n = int(y.shape[0])
    max_lag = max(p, q, 1)

    def objective(z_theta):
        c = z_theta[0]
        phi = 0.99 * jnp.tanh(z_theta[1:1 + p])
        theta_ma = 0.99 * jnp.tanh(z_theta[1 + p:1 + p + q])
        sigma2 = _softplus_production(jnp, z_theta[1 + p + q])
        nu = 1.01 + _softplus_production(jnp, z_theta[2 + p + q])

        resid_hist0 = jnp.zeros((max_lag,), dtype=z_theta.dtype)

        def step(resid_hist, t):
            e_t = y_j[t] - c
            for i in range(p):
                e_t = e_t - phi[i] * y_j[t - 1 - i]
            for j in range(q):
                e_t = e_t - theta_ma[j] * resid_hist[j]
            resid_hist_next = jnp.concatenate([e_t[None], resid_hist[:-1]])
            return resid_hist_next, e_t

        _, resid_tail = jax.lax.scan(step, resid_hist0, jnp.arange(max_lag, n))
        log_scale = 0.5 * (jsp.special.gammaln(1.0 / nu) - jsp.special.gammaln(3.0 / nu))
        log_const = jnp.log(nu) - jnp.log(2.0) - log_scale - jsp.special.gammaln(1.0 / nu)
        q_terms = jnp.exp(nu * (jnp.log(jnp.maximum(jnp.abs(resid_tail), 1e-300)) - 0.5 * jnp.log(sigma2) - log_scale))
        return jnp.mean(-log_const + 0.5 * jnp.log(sigma2) + q_terms)

    return _value_grad_hess(objective, np.asarray(z_j, dtype=np.float64))


def garch_logspace_value_grad_hess(
    z: NDArray[np.float64],
    resid2: NDArray[np.float64],
    p: int,
    q: int,
    dist: str = "normal",
    resid: NDArray[np.float64] | None = None,
) -> Tuple[float, NDArray[np.float64], NDArray[np.float64]]:
    """Reference value/gradient/Hessian for transformed GARCH(p,q) objectives."""
    jax, jnp, jsp = _jax_modules()
    z_j = jnp.asarray(z, dtype=jnp.float64)
    resid2_j = jnp.asarray(resid2, dtype=jnp.float64)
    resid_j = None if resid is None else jnp.asarray(resid, dtype=jnp.float64)
    n = int(resid2.shape[0])

    def objective(z_theta):
        z_garch = z_theta[:1 + p + q]
        z_joint = jnp.concatenate([z_garch[1:], jnp.array([0.0], dtype=z_theta.dtype)])
        lse = jsp.special.logsumexp(z_joint)
        omega = _softplus_production(jnp, z_garch[0])
        alpha = jnp.exp(z_garch[1:1 + p] - lse)
        beta = jnp.exp(z_garch[1 + p:1 + p + q] - lse)
        sigma0 = jnp.mean(resid2_j)
        sigma_hist0 = jnp.full((max(q, 1),), sigma0, dtype=z_theta.dtype)

        def step(sigma_hist, t):
            sigma_t = omega
            for j in range(p):
                sigma_t = sigma_t + jnp.where(t > j, alpha[j] * resid2_j[t - 1 - j], 0.0)
            for k in range(q):
                sigma_t = sigma_t + jnp.where(t > k, beta[k] * sigma_hist[k], 0.0)
            sigma_hist_next = jnp.concatenate([sigma_t[None], sigma_hist[:-1]])
            return sigma_hist_next, sigma_t

        _, sigma_tail = jax.lax.scan(step, sigma_hist0, jnp.arange(1, n))
        sigma2 = jnp.concatenate([sigma0[None], sigma_tail])

        if dist == "normal":
            return 0.5 * jnp.sum(jnp.log(sigma2) + resid2_j / sigma2)

        if dist == "studentt":
            nu = 2.0 + _softplus_production(jnp, z_theta[-1])
            inv_nu_m2 = 1.0 / (nu - 2.0)
            constant = n * (
                jsp.special.gammaln(0.5 * (nu + 1.0))
                - jsp.special.gammaln(0.5 * nu)
                - 0.5 * jnp.log(jnp.pi / inv_nu_m2)
            )
            tail = jnp.log1p(resid2_j / sigma2 * inv_nu_m2)
            return 0.5 * (jnp.sum(jnp.log(sigma2)) + (nu + 1.0) * jnp.sum(tail)) - constant

        if dist == "ged":
            nu = 1.01 + _softplus_production(jnp, z_theta[-1])
            log_scale = 0.5 * (jsp.special.gammaln(1.0 / nu) - jsp.special.gammaln(3.0 / nu))
            log_const = jnp.log(nu) - jnp.log(2.0) - log_scale - jsp.special.gammaln(1.0 / nu)
            q_terms = jnp.exp(
                nu * (0.5 * jnp.log(jnp.maximum(resid2_j, 1e-300)) - 0.5 * jnp.log(sigma2) - log_scale)
            )
            return jnp.sum(-log_const + 0.5 * jnp.log(sigma2) + q_terms)

        if dist == "skewt":
            if resid_j is None:
                raise ValueError("Raw residuals are required for GARCH+SkewT log-space AD validation.")
            nu = 2.0 + _softplus_production(jnp, z_theta[-2])
            lam = jnp.tanh(z_theta[-1])
            c_log = (
                jsp.special.gammaln(0.5 * (nu + 1.0))
                - jsp.special.gammaln(0.5 * nu)
                - 0.5 * jnp.log(jnp.pi * (nu - 2.0))
            )
            a = 4.0 * lam * jnp.exp(c_log) * (nu - 2.0) / (nu - 1.0)
            b = jnp.sqrt(1.0 + 3.0 * lam * lam - a * a)
            z_std = resid_j / jnp.sqrt(sigma2)
            bz_plus_a = b * z_std + a
            sign_term = jnp.where(bz_plus_a >= 0.0, 1.0, -1.0)
            z_adj = bz_plus_a / (1.0 - lam * sign_term)
            ll = (
                n * c_log
                - 0.5 * jnp.sum(jnp.log(sigma2))
                + jnp.log(b)
                - 0.5 * (nu + 1.0) * jnp.sum(jnp.log1p(z_adj * z_adj / (nu - 2.0)))
            )
            return -ll

        raise ValueError(f"Unsupported GARCH distribution: {dist}")

    return _value_grad_hess(objective, np.asarray(z_j, dtype=np.float64))


def gjr_garch_logspace_value_grad_hess(
    z: NDArray[np.float64],
    resid: NDArray[np.float64],
    p: int,
    q: int,
    dist: str = "normal",
) -> Tuple[float, NDArray[np.float64], NDArray[np.float64]]:
    """Reference value/gradient/Hessian for transformed GJR-GARCH(p,q) objectives."""
    jax, jnp, jsp = _jax_modules()
    z_j = jnp.asarray(z, dtype=jnp.float64)
    resid_j = jnp.asarray(resid, dtype=jnp.float64)
    resid2_j = resid_j * resid_j
    n = int(resid.shape[0])
    m = 2 * p + q

    def objective(z_theta):
        z_gjr = z_theta[:1 + m]
        z_joint = jnp.concatenate([z_gjr[1:], jnp.array([0.0], dtype=z_theta.dtype)])
        lse = jsp.special.logsumexp(z_joint)
        omega = _softplus_production(jnp, z_gjr[0])
        params = jnp.exp(z_gjr[1:] - lse)
        alpha = params[:p]
        gamma = params[p:2 * p]
        beta = params[2 * p:]
        sigma0 = jnp.mean(resid2_j)
        sigma_hist0 = jnp.full((max(q, 1),), sigma0, dtype=z_theta.dtype)

        def step(sigma_hist, t):
            sigma_t = omega
            for j in range(p):
                e_lag = resid_j[t - 1 - j]
                e2_lag = e_lag * e_lag
                ind = jnp.where(e_lag < 0.0, 1.0, 0.0)
                sigma_t = sigma_t + jnp.where(
                    t > j,
                    alpha[j] * e2_lag + gamma[j] * ind * e2_lag,
                    0.0,
                )
            for k in range(q):
                sigma_t = sigma_t + jnp.where(t > k, beta[k] * sigma_hist[k], 0.0)
            sigma_t = jnp.maximum(sigma_t, GJR_H_FLOOR)
            sigma_hist_next = jnp.concatenate([sigma_t[None], sigma_hist[:-1]])
            return sigma_hist_next, sigma_t

        _, sigma_tail = jax.lax.scan(step, sigma_hist0, jnp.arange(1, n))
        sigma2 = jnp.concatenate([sigma0[None], sigma_tail])

        if dist == "normal":
            return 0.5 * jnp.sum(jnp.log(sigma2) + resid2_j / sigma2)

        if dist == "studentt":
            nu = 2.0 + _softplus_production(jnp, z_theta[-1])
            inv_nu_m2 = 1.0 / (nu - 2.0)
            constant = n * (
                jsp.special.gammaln(0.5 * (nu + 1.0))
                - jsp.special.gammaln(0.5 * nu)
                - 0.5 * jnp.log(jnp.pi / inv_nu_m2)
            )
            tail = jnp.log1p(resid2_j / sigma2 * inv_nu_m2)
            return 0.5 * (jnp.sum(jnp.log(sigma2)) + (nu + 1.0) * jnp.sum(tail)) - constant

        if dist == "ged":
            nu = 1.01 + _softplus_production(jnp, z_theta[-1])
            log_scale = 0.5 * (jsp.special.gammaln(1.0 / nu) - jsp.special.gammaln(3.0 / nu))
            log_const = jnp.log(nu) - jnp.log(2.0) - log_scale - jsp.special.gammaln(1.0 / nu)
            z_std = jnp.abs(resid_j) / (jnp.sqrt(sigma2) * jnp.exp(log_scale))
            return -n * log_const + 0.5 * jnp.sum(jnp.log(sigma2)) + jnp.sum(jnp.power(z_std, nu))

        if dist == "skewt":
            nu = 2.0 + _softplus_production(jnp, z_theta[-2])
            lam = jnp.tanh(z_theta[-1])
            c_log = (
                jsp.special.gammaln(0.5 * (nu + 1.0))
                - jsp.special.gammaln(0.5 * nu)
                - 0.5 * jnp.log(jnp.pi * (nu - 2.0))
            )
            a = 4.0 * lam * jnp.exp(c_log) * (nu - 2.0) / (nu - 1.0)
            b = jnp.sqrt(1.0 + 3.0 * lam * lam - a * a)
            z_std = resid_j / jnp.sqrt(sigma2)
            bz_plus_a = b * z_std + a
            sign_term = jnp.where(bz_plus_a >= 0.0, 1.0, -1.0)
            z_adj = bz_plus_a / (1.0 - lam * sign_term)
            ll = (
                n * c_log
                - 0.5 * jnp.sum(jnp.log(sigma2))
                + jnp.log(b)
                - 0.5 * (nu + 1.0) * jnp.sum(jnp.log1p(z_adj * z_adj / (nu - 2.0)))
            )
            return -ll

        raise ValueError(f"Unsupported GJR-GARCH distribution: {dist}")

    return _value_grad_hess(objective, np.asarray(z_j, dtype=np.float64))


def arma_garch_value_grad_hess(
    theta: NDArray[np.float64],
    y: NDArray[np.float64],
    p_ar: int,
    q_ma: int,
    p_arch: int,
    q_garch: int,
    dist: str = "normal",
) -> Tuple[float, NDArray[np.float64], NDArray[np.float64]]:
    """Reference NLL, gradient, and Hessian for ARMA(p,q)+GARCH(P,Q)."""
    jax, jnp, jsp = _jax_modules()
    y_j = jnp.asarray(y, dtype=jnp.float64)
    n = int(y.shape[0])
    max_lag = max(p_ar, q_ma, p_arch, q_garch, 1)

    def objective(theta_j):
        idx = 0
        c = theta_j[idx]
        idx += 1
        phi = theta_j[idx:idx + p_ar]
        idx += p_ar
        theta_ma = theta_j[idx:idx + q_ma]
        idx += q_ma
        omega = theta_j[idx]
        idx += 1
        alpha = theta_j[idx:idx + p_arch]
        idx += p_arch
        beta = theta_j[idx:idx + q_garch]
        idx += q_garch

        invalid = jnp.logical_or(
            jnp.logical_or(omega <= 0.0, jnp.any(alpha < 0.0)),
            jnp.any(beta < 0.0),
        )
        invalid = jnp.logical_or(invalid, jnp.sum(alpha) + jnp.sum(beta) >= 1.0)
        invalid = jnp.logical_or(invalid, jnp.any(jnp.abs(phi) >= 1.0))
        invalid = jnp.logical_or(invalid, jnp.any(jnp.abs(theta_ma) >= 1.0))

        nu = None
        lam = None
        if dist == "studentt":
            nu = theta_j[idx]
            invalid = jnp.logical_or(invalid, nu <= 2.001)
        elif dist == "ged":
            nu = theta_j[idx]
            invalid = jnp.logical_or(invalid, nu <= 1.01)
        elif dist == "skewt":
            nu = theta_j[idx]
            lam = theta_j[idx + 1]
            invalid = jnp.logical_or(invalid, nu <= 2.001)
            invalid = jnp.logical_or(invalid, jnp.abs(lam) >= 0.999)

        h0 = jnp.mean(y_j * y_j)
        resid_hist0 = jnp.zeros((max_lag,), dtype=theta_j.dtype)
        sigma_hist0 = jnp.full((max_lag,), h0, dtype=theta_j.dtype)

        def step(carry, t):
            resid_hist, sigma_hist = carry

            mu_t = c
            for i in range(p_ar):
                mu_t = mu_t + phi[i] * y_j[t - 1 - i]
            for j in range(q_ma):
                mu_t = mu_t + theta_ma[j] * resid_hist[j]
            e_t = y_j[t] - mu_t

            h_t = omega
            for i in range(p_arch):
                h_t = h_t + alpha[i] * resid_hist[i] * resid_hist[i]
            for j in range(q_garch):
                h_t = h_t + beta[j] * sigma_hist[j]
            h_t = jnp.maximum(h_t, GJR_H_FLOOR)

            resid_hist_next = jnp.concatenate([e_t[None], resid_hist[:-1]])
            sigma_hist_next = jnp.concatenate([h_t[None], sigma_hist[:-1]])
            return (resid_hist_next, sigma_hist_next), (e_t, h_t)

        (_, _), out = jax.lax.scan(step, (resid_hist0, sigma_hist0), jnp.arange(max_lag, n))
        resid_tail = out[0]
        sigma_tail = out[1]
        n_eff = n - max_lag

        if dist == "normal":
            nll = jnp.sum(0.5 * (jnp.log(sigma_tail) + resid_tail * resid_tail / sigma_tail)) / n_eff
        elif dist == "studentt":
            assert nu is not None
            constant = n_eff * (
                jsp.special.gammaln(0.5 * (nu + 1.0))
                - jsp.special.gammaln(0.5 * nu)
                - 0.5 * jnp.log(nu * jnp.pi)
            )
            tail = jnp.log1p((resid_tail * resid_tail / sigma_tail) / nu)
            nll = (0.5 * (jnp.sum(jnp.log(sigma_tail)) + (nu + 1.0) * jnp.sum(tail)) - constant) / n_eff
        elif dist == "skewt":
            assert nu is not None
            assert lam is not None
            lgamma_c = jsp.special.gammaln(0.5 * (nu + 1.0)) - jsp.special.gammaln(0.5 * nu)
            a = 4.0 * lam * jnp.exp(lgamma_c) * ((nu - 2.0) / (nu - 1.0)) / jnp.sqrt((nu - 2.0) / jnp.pi)
            b2 = 1.0 + 3.0 * lam * lam - a * a
            invalid = jnp.logical_or(invalid, b2 <= 1e-12)
            b = jnp.sqrt(jnp.maximum(b2, 1e-12))
            zstar = b * resid_tail / jnp.sqrt(sigma_tail) + a
            scale = jnp.where(zstar < 0.0, 1.0 - lam, 1.0 + lam)
            z_adj = zstar / scale
            base_cnst = jnp.log(b) + lgamma_c - 0.5 * jnp.log((nu - 2.0) * jnp.pi)
            nll = (
                jnp.sum(
                    jnp.log(scale)
                    + 0.5 * jnp.log(sigma_tail)
                    + 0.5 * (nu + 1.0) * jnp.log1p((z_adj * z_adj) / (nu - 2.0))
                )
                - n_eff * base_cnst
            ) / n_eff
        elif dist == "ged":
            assert nu is not None
            log_scale = 0.5 * (jsp.special.gammaln(1.0 / nu) - jsp.special.gammaln(3.0 / nu))
            log_const = jnp.log(nu) - jnp.log(2.0) - log_scale - jsp.special.gammaln(1.0 / nu)
            q_terms = jnp.exp(
                nu * (jnp.log(jnp.maximum(jnp.abs(resid_tail), 1e-300)) - 0.5 * jnp.log(sigma_tail) - log_scale)
            )
            nll = jnp.mean(-log_const + 0.5 * jnp.log(sigma_tail) + q_terms)
        else:
            raise ValueError(f"Unsupported ARMA-GARCH distribution: {dist}")

        return jnp.where(invalid, 1e10, nll)

    return _value_grad_hess(objective, theta)


def arma_garch_logspace_value_grad_hess(
    z: NDArray[np.float64],
    y: NDArray[np.float64],
    p_ar: int,
    q_ma: int,
    p_arch: int,
    q_garch: int,
    dist: str = "normal",
) -> Tuple[float, NDArray[np.float64], NDArray[np.float64]]:
    """Reference value/gradient/Hessian for transformed ARMA(p,q)+GARCH(P,Q) objectives."""
    jax, jnp, jsp = _jax_modules()
    z_j = jnp.asarray(z, dtype=jnp.float64)
    y_j = jnp.asarray(y, dtype=jnp.float64)
    n = int(y.shape[0])
    max_lag = max(p_ar, q_ma, p_arch, q_garch, 1)

    def objective(z_theta):
        n_mean = 1 + p_ar + q_ma
        n_vol = 1 + p_arch + q_garch

        c = z_theta[0]
        phi = 0.99 * jnp.tanh(z_theta[1:1 + p_ar])
        theta_ma = 0.99 * jnp.tanh(z_theta[1 + p_ar:n_mean])

        z_garch = z_theta[n_mean:n_mean + n_vol]
        z_joint = jnp.concatenate([z_garch[1:], jnp.array([0.0], dtype=z_theta.dtype)])
        lse = jsp.special.logsumexp(z_joint)
        omega = _softplus_production(jnp, z_garch[0])
        alpha = jnp.exp(z_garch[1:1 + p_arch] - lse)
        beta = jnp.exp(z_garch[1 + p_arch:1 + p_arch + q_garch] - lse)

        nu = None
        lam = None
        if dist == "studentt":
            nu = 2.0 + _softplus_production(jnp, z_theta[n_mean + n_vol])
        elif dist == "ged":
            nu = 1.01 + _softplus_production(jnp, z_theta[n_mean + n_vol])
        elif dist == "skewt":
            nu = 2.0 + _softplus_production(jnp, z_theta[n_mean + n_vol])
            lam = jnp.tanh(z_theta[n_mean + n_vol + 1])

        h0 = jnp.mean(y_j * y_j)
        resid_hist0 = jnp.zeros((max_lag,), dtype=z_theta.dtype)
        sigma_hist0 = jnp.full((max_lag,), h0, dtype=z_theta.dtype)

        def step(carry, t):
            resid_hist, sigma_hist = carry

            mu_t = c
            for i in range(p_ar):
                mu_t = mu_t + phi[i] * y_j[t - 1 - i]
            for j in range(q_ma):
                mu_t = mu_t + theta_ma[j] * resid_hist[j]
            e_t = y_j[t] - mu_t

            h_t = omega
            for i in range(p_arch):
                h_t = h_t + alpha[i] * resid_hist[i] * resid_hist[i]
            for j in range(q_garch):
                h_t = h_t + beta[j] * sigma_hist[j]
            h_t = jnp.maximum(h_t, GJR_H_FLOOR)

            resid_hist_next = jnp.concatenate([e_t[None], resid_hist[:-1]])
            sigma_hist_next = jnp.concatenate([h_t[None], sigma_hist[:-1]])
            return (resid_hist_next, sigma_hist_next), (e_t, h_t)

        (_, _), out = jax.lax.scan(step, (resid_hist0, sigma_hist0), jnp.arange(max_lag, n))
        resid_tail = out[0]
        sigma_tail = out[1]
        n_eff = n - max_lag

        if dist == "normal":
            return jnp.sum(0.5 * (jnp.log(sigma_tail) + resid_tail * resid_tail / sigma_tail)) / n_eff

        if dist == "studentt":
            assert nu is not None
            constant = n_eff * (
                jsp.special.gammaln(0.5 * (nu + 1.0))
                - jsp.special.gammaln(0.5 * nu)
                - 0.5 * jnp.log(nu * jnp.pi)
            )
            tail = jnp.log1p((resid_tail * resid_tail / sigma_tail) / nu)
            return (0.5 * (jnp.sum(jnp.log(sigma_tail)) + (nu + 1.0) * jnp.sum(tail)) - constant) / n_eff

        if dist == "skewt":
            assert nu is not None
            assert lam is not None
            lgamma_c = jsp.special.gammaln(0.5 * (nu + 1.0)) - jsp.special.gammaln(0.5 * nu)
            a = 4.0 * lam * jnp.exp(lgamma_c) * ((nu - 2.0) / (nu - 1.0)) / jnp.sqrt((nu - 2.0) / jnp.pi)
            b2 = 1.0 + 3.0 * lam * lam - a * a
            b = jnp.sqrt(jnp.maximum(b2, 1e-12))
            zstar = b * resid_tail / jnp.sqrt(sigma_tail) + a
            scale = jnp.where(zstar < 0.0, 1.0 - lam, 1.0 + lam)
            z_adj = zstar / scale
            base_cnst = jnp.log(b) + lgamma_c - 0.5 * jnp.log((nu - 2.0) * jnp.pi)
            return (
                jnp.sum(
                    jnp.log(scale)
                    + 0.5 * jnp.log(sigma_tail)
                    + 0.5 * (nu + 1.0) * jnp.log1p((z_adj * z_adj) / (nu - 2.0))
                )
                - n_eff * base_cnst
            ) / n_eff

        if dist == "ged":
            assert nu is not None
            log_scale = 0.5 * (jsp.special.gammaln(1.0 / nu) - jsp.special.gammaln(3.0 / nu))
            log_const = jnp.log(nu) - jnp.log(2.0) - log_scale - jsp.special.gammaln(1.0 / nu)
            q_terms = jnp.exp(
                nu * (jnp.log(jnp.maximum(jnp.abs(resid_tail), 1e-300)) - 0.5 * jnp.log(sigma_tail) - log_scale)
            )
            return jnp.mean(-log_const + 0.5 * jnp.log(sigma_tail) + q_terms)

        raise ValueError(f"Unsupported ARMA-GARCH distribution: {dist}")

    return _value_grad_hess(objective, np.asarray(z_j, dtype=np.float64))


def arma_gjr_garch_value_grad_hess(
    theta: NDArray[np.float64],
    y: NDArray[np.float64],
    p_ar: int,
    q_ma: int,
    p_arch: int,
    q_garch: int,
    dist: str = "normal",
) -> Tuple[float, NDArray[np.float64], NDArray[np.float64]]:
    """Reference NLL, gradient, and Hessian for ARMA(p,q)+GJR-GARCH(P,Q)."""
    jax, jnp, jsp = _jax_modules()
    y_j = jnp.asarray(y, dtype=jnp.float64)
    n = int(y.shape[0])
    max_lag = max(p_ar, q_ma, p_arch, q_garch, 1)

    def objective(theta_j):
        idx = 0
        c = theta_j[idx]
        idx += 1
        phi = theta_j[idx:idx + p_ar]
        idx += p_ar
        theta_ma = theta_j[idx:idx + q_ma]
        idx += q_ma
        omega = theta_j[idx]
        idx += 1
        alpha = theta_j[idx:idx + p_arch]
        idx += p_arch
        gamma = theta_j[idx:idx + p_arch]
        idx += p_arch
        beta = theta_j[idx:idx + q_garch]
        idx += q_garch

        invalid = jnp.logical_or(
            jnp.logical_or(omega <= 0.0, jnp.any(alpha < 0.0)),
            jnp.any(gamma < 0.0),
        )
        invalid = jnp.logical_or(invalid, jnp.any(beta < 0.0))
        invalid = jnp.logical_or(
            invalid,
            jnp.sum(alpha) + jnp.sum(gamma) + jnp.sum(beta) >= 1.0,
        )
        invalid = jnp.logical_or(invalid, jnp.any(jnp.abs(phi) >= 1.0))
        invalid = jnp.logical_or(invalid, jnp.any(jnp.abs(theta_ma) >= 1.0))

        nu = None
        lam = None
        if dist == "studentt":
            nu = theta_j[idx]
            invalid = jnp.logical_or(invalid, nu <= 2.001)
        elif dist == "ged":
            nu = theta_j[idx]
            invalid = jnp.logical_or(invalid, nu <= 1.01)
        elif dist == "skewt":
            nu = theta_j[idx]
            lam = theta_j[idx + 1]
            invalid = jnp.logical_or(invalid, nu <= 2.001)
            invalid = jnp.logical_or(invalid, jnp.abs(lam) >= 0.999)

        h0 = jnp.mean(y_j * y_j)
        resid_hist0 = jnp.zeros((max_lag,), dtype=theta_j.dtype)
        sigma_hist0 = jnp.full((max_lag,), h0, dtype=theta_j.dtype)

        def step(carry, t):
            resid_hist, sigma_hist = carry

            mu_t = c
            for i in range(p_ar):
                mu_t = mu_t + phi[i] * y_j[t - 1 - i]
            for j in range(q_ma):
                mu_t = mu_t + theta_ma[j] * resid_hist[j]
            e_t = y_j[t] - mu_t

            h_t = omega
            for i in range(p_arch):
                e_lag = resid_hist[i]
                e2_lag = e_lag * e_lag
                ind = jnp.where(e_lag < 0.0, 1.0, 0.0)
                h_t = h_t + alpha[i] * e2_lag + gamma[i] * ind * e2_lag
            for j in range(q_garch):
                h_t = h_t + beta[j] * sigma_hist[j]
            h_t = jnp.maximum(h_t, GJR_H_FLOOR)

            resid_hist_next = jnp.concatenate([e_t[None], resid_hist[:-1]])
            sigma_hist_next = jnp.concatenate([h_t[None], sigma_hist[:-1]])
            return (resid_hist_next, sigma_hist_next), (e_t, h_t)

        (_, _), out = jax.lax.scan(step, (resid_hist0, sigma_hist0), jnp.arange(max_lag, n))
        resid_tail = out[0]
        sigma_tail = out[1]
        n_eff = n - max_lag

        if dist == "normal":
            nll = jnp.sum(0.5 * (jnp.log(sigma_tail) + resid_tail * resid_tail / sigma_tail)) / n_eff
        elif dist == "studentt":
            assert nu is not None
            constant = n_eff * (
                jsp.special.gammaln(0.5 * (nu + 1.0))
                - jsp.special.gammaln(0.5 * nu)
                - 0.5 * jnp.log(nu * jnp.pi)
            )
            tail = jnp.log1p((resid_tail * resid_tail / sigma_tail) / nu)
            nll = (0.5 * (jnp.sum(jnp.log(sigma_tail)) + (nu + 1.0) * jnp.sum(tail)) - constant) / n_eff
        elif dist == "skewt":
            assert nu is not None
            assert lam is not None
            lgamma_c = jsp.special.gammaln(0.5 * (nu + 1.0)) - jsp.special.gammaln(0.5 * nu)
            a = 4.0 * lam * jnp.exp(lgamma_c) * ((nu - 2.0) / (nu - 1.0)) / jnp.sqrt((nu - 2.0) / jnp.pi)
            b2 = 1.0 + 3.0 * lam * lam - a * a
            invalid = jnp.logical_or(invalid, b2 <= 1e-12)
            b = jnp.sqrt(jnp.maximum(b2, 1e-12))
            zstar = b * resid_tail / jnp.sqrt(sigma_tail) + a
            scale = jnp.where(zstar < 0.0, 1.0 - lam, 1.0 + lam)
            z_adj = zstar / scale
            base_cnst = jnp.log(b) + lgamma_c - 0.5 * jnp.log((nu - 2.0) * jnp.pi)
            nll = (
                jnp.sum(
                    jnp.log(scale)
                    + 0.5 * jnp.log(sigma_tail)
                    + 0.5 * (nu + 1.0) * jnp.log1p((z_adj * z_adj) / (nu - 2.0))
                )
                - n_eff * base_cnst
            ) / n_eff
        elif dist == "ged":
            assert nu is not None
            log_scale = 0.5 * (jsp.special.gammaln(1.0 / nu) - jsp.special.gammaln(3.0 / nu))
            log_const = jnp.log(nu) - jnp.log(2.0) - log_scale - jsp.special.gammaln(1.0 / nu)
            z_std = jnp.abs(resid_tail) / (jnp.sqrt(sigma_tail) * jnp.exp(log_scale))
            nll = (-n_eff * log_const + 0.5 * jnp.sum(jnp.log(sigma_tail)) + jnp.sum(jnp.power(z_std, nu))) / n_eff
        else:
            raise ValueError(f"Unsupported ARMA-GJR-GARCH distribution: {dist}")

        return jnp.where(invalid, 1e10, nll)

    return _value_grad_hess(objective, theta)


def arma_gjr_garch_logspace_value_grad_hess(
    z: NDArray[np.float64],
    y: NDArray[np.float64],
    p_ar: int,
    q_ma: int,
    p_arch: int,
    q_garch: int,
    dist: str = "normal",
) -> Tuple[float, NDArray[np.float64], NDArray[np.float64]]:
    """Reference value/gradient/Hessian for transformed ARMA+GJR-GARCH objectives."""
    jax, jnp, jsp = _jax_modules()
    z_j = jnp.asarray(z, dtype=jnp.float64)
    y_j = jnp.asarray(y, dtype=jnp.float64)
    n = int(y.shape[0])
    max_lag = max(p_ar, q_ma, p_arch, q_garch, 1)

    def objective(z_theta):
        n_mean = 1 + p_ar + q_ma
        m = 2 * p_arch + q_garch

        c = z_theta[0]
        phi = 0.99 * jnp.tanh(z_theta[1:1 + p_ar])
        theta_ma = 0.99 * jnp.tanh(z_theta[1 + p_ar:n_mean])

        z_gjr = z_theta[n_mean:n_mean + 1 + m]
        z_joint = jnp.concatenate([z_gjr[1:], jnp.array([0.0], dtype=z_theta.dtype)])
        lse = jsp.special.logsumexp(z_joint)
        omega = _softplus_production(jnp, z_gjr[0])
        params = jnp.exp(z_gjr[1:] - lse)
        alpha = params[:p_arch]
        gamma = params[p_arch:2 * p_arch]
        beta = params[2 * p_arch:]

        nu = None
        lam = None
        if dist == "studentt":
            nu = 2.0 + _softplus_production(jnp, z_theta[-1])
        elif dist == "ged":
            nu = 1.01 + _softplus_production(jnp, z_theta[-1])
        elif dist == "skewt":
            nu = 2.0 + _softplus_production(jnp, z_theta[-2])
            lam = jnp.tanh(z_theta[-1])

        h0 = jnp.mean(y_j * y_j)
        resid_hist0 = jnp.zeros((max_lag,), dtype=z_theta.dtype)
        sigma_hist0 = jnp.full((max_lag,), h0, dtype=z_theta.dtype)

        def step(carry, t):
            resid_hist, sigma_hist = carry

            mu_t = c
            for i in range(p_ar):
                mu_t = mu_t + phi[i] * y_j[t - 1 - i]
            for j in range(q_ma):
                mu_t = mu_t + theta_ma[j] * resid_hist[j]
            e_t = y_j[t] - mu_t

            h_t = omega
            for i in range(p_arch):
                e_lag = resid_hist[i]
                e2_lag = e_lag * e_lag
                ind = jnp.where(e_lag < 0.0, 1.0, 0.0)
                h_t = h_t + alpha[i] * e2_lag + gamma[i] * ind * e2_lag
            for j in range(q_garch):
                h_t = h_t + beta[j] * sigma_hist[j]
            h_t = jnp.maximum(h_t, GJR_H_FLOOR)

            resid_hist_next = jnp.concatenate([e_t[None], resid_hist[:-1]])
            sigma_hist_next = jnp.concatenate([h_t[None], sigma_hist[:-1]])
            return (resid_hist_next, sigma_hist_next), (e_t, h_t)

        (_, _), out = jax.lax.scan(step, (resid_hist0, sigma_hist0), jnp.arange(max_lag, n))
        resid_tail = out[0]
        sigma_tail = out[1]
        n_eff = n - max_lag

        if dist == "normal":
            return jnp.sum(0.5 * (jnp.log(sigma_tail) + resid_tail * resid_tail / sigma_tail)) / n_eff

        if dist == "studentt":
            assert nu is not None
            constant = n_eff * (
                jsp.special.gammaln(0.5 * (nu + 1.0))
                - jsp.special.gammaln(0.5 * nu)
                - 0.5 * jnp.log(nu * jnp.pi)
            )
            tail = jnp.log1p((resid_tail * resid_tail / sigma_tail) / nu)
            return (0.5 * (jnp.sum(jnp.log(sigma_tail)) + (nu + 1.0) * jnp.sum(tail)) - constant) / n_eff

        if dist == "ged":
            assert nu is not None
            log_scale = 0.5 * (jsp.special.gammaln(1.0 / nu) - jsp.special.gammaln(3.0 / nu))
            log_const = jnp.log(nu) - jnp.log(2.0) - log_scale - jsp.special.gammaln(1.0 / nu)
            z_std = jnp.abs(resid_tail) / (jnp.sqrt(sigma_tail) * jnp.exp(log_scale))
            return (-n_eff * log_const + 0.5 * jnp.sum(jnp.log(sigma_tail)) + jnp.sum(jnp.power(z_std, nu))) / n_eff

        if dist == "skewt":
            assert nu is not None
            assert lam is not None
            lgamma_c = jsp.special.gammaln(0.5 * (nu + 1.0)) - jsp.special.gammaln(0.5 * nu)
            a = 4.0 * lam * jnp.exp(lgamma_c) * ((nu - 2.0) / (nu - 1.0)) / jnp.sqrt((nu - 2.0) / jnp.pi)
            b = jnp.sqrt(jnp.maximum(1.0 + 3.0 * lam * lam - a * a, 1e-12))
            zstar = b * resid_tail / jnp.sqrt(sigma_tail) + a
            scale = jnp.where(zstar < 0.0, 1.0 - lam, 1.0 + lam)
            z_adj = zstar / scale
            base_cnst = jnp.log(b) + lgamma_c - 0.5 * jnp.log((nu - 2.0) * jnp.pi)
            return (
                jnp.sum(
                    jnp.log(scale)
                    + 0.5 * jnp.log(sigma_tail)
                    + 0.5 * (nu + 1.0) * jnp.log1p((z_adj * z_adj) / (nu - 2.0))
                )
                - n_eff * base_cnst
            ) / n_eff

        raise ValueError(f"Unsupported ARMA-GJR-GARCH distribution: {dist}")

    return _value_grad_hess(objective, np.asarray(z_j, dtype=np.float64))


def arma_egarch_value_grad_hess(
    theta: NDArray[np.float64],
    y: NDArray[np.float64],
    p_ar: int,
    q_ma: int,
    p_arch: int,
    q_egarch: int,
    dist: str = "normal",
) -> Tuple[float, NDArray[np.float64], NDArray[np.float64]]:
    """Reference NLL, gradient, and Hessian for ARMA(p,q)+EGARCH(P,Q)."""
    jax, jnp, jsp = _jax_modules()
    y_j = jnp.asarray(y, dtype=jnp.float64)
    n = int(y.shape[0])
    max_lag = max(p_ar, q_ma, p_arch, q_egarch, 1)

    def studentt_abs_moment(nu):
        c_log = (
            jsp.special.gammaln(0.5 * (nu + 1.0))
            - jsp.special.gammaln(0.5 * nu)
            - 0.5 * jnp.log(jnp.pi * (nu - 2.0))
        )
        return 2.0 * jnp.exp(c_log) * (nu - 2.0) / (nu - 1.0)

    def objective(theta_j):
        idx = 0
        c = theta_j[idx]
        idx += 1
        phi = theta_j[idx:idx + p_ar]
        idx += p_ar
        theta_ma = theta_j[idx:idx + q_ma]
        idx += q_ma
        omega = theta_j[idx]
        idx += 1
        alpha = theta_j[idx:idx + p_arch]
        idx += p_arch
        gamma = theta_j[idx:idx + p_arch]
        idx += p_arch
        beta = theta_j[idx:idx + q_egarch]
        idx += q_egarch

        invalid = jnp.logical_or(jnp.any(jnp.abs(phi) >= 1.0), jnp.any(jnp.abs(theta_ma) >= 1.0))
        invalid = jnp.logical_or(invalid, jnp.any(jnp.abs(beta) >= 0.999))

        if dist == "normal":
            abs_moment = jnp.sqrt(2.0 / jnp.pi)
            nu = None
        elif dist == "studentt":
            nu = theta_j[idx]
            invalid = jnp.logical_or(invalid, nu <= 2.001)
            abs_moment = studentt_abs_moment(nu)
        else:
            raise ValueError(f"Unsupported ARMA-EGARCH distribution: {dist}")

        h0 = jnp.maximum(jnp.mean(y_j * y_j), GJR_H_FLOOR)
        x0 = jnp.log(h0)
        resid_hist0 = jnp.zeros((max_lag,), dtype=theta_j.dtype)
        x_hist0 = jnp.full((max_lag,), x0, dtype=theta_j.dtype)

        def step(carry, t):
            resid_hist, x_hist = carry

            mu_t = c
            for i in range(p_ar):
                mu_t = mu_t + phi[i] * y_j[t - 1 - i]
            for j in range(q_ma):
                mu_t = mu_t + theta_ma[j] * resid_hist[j]
            e_t = y_j[t] - mu_t

            x_t = omega
            for i in range(p_arch):
                z_lag = resid_hist[i] * jnp.exp(-0.5 * x_hist[i])
                x_t = x_t + alpha[i] * (jnp.abs(z_lag) - abs_moment) + gamma[i] * z_lag
            for j in range(q_egarch):
                x_t = x_t + beta[j] * x_hist[j]

            h_t = jnp.maximum(jnp.exp(x_t), GJR_H_FLOOR)
            x_t = jnp.log(h_t)

            resid_hist_next = jnp.concatenate([e_t[None], resid_hist[:-1]])
            x_hist_next = jnp.concatenate([x_t[None], x_hist[:-1]])
            return (resid_hist_next, x_hist_next), (e_t, x_t, h_t)

        (_, _), out = jax.lax.scan(step, (resid_hist0, x_hist0), jnp.arange(max_lag, n))
        resid_tail = out[0]
        x_tail = out[1]
        sigma_tail = out[2]
        n_eff = n - max_lag

        if dist == "normal":
            nll = jnp.sum(0.5 * (x_tail + resid_tail * resid_tail / sigma_tail)) / n_eff
        else:
            assert nu is not None
            constant = (
                jsp.special.gammaln(0.5 * (nu + 1.0))
                - jsp.special.gammaln(0.5 * nu)
                - 0.5 * jnp.log(jnp.pi * (nu - 2.0))
            )
            tail = jnp.log1p((resid_tail * resid_tail / sigma_tail) / (nu - 2.0))
            nll = (
                jnp.sum(0.5 * x_tail + 0.5 * (nu + 1.0) * tail) - n_eff * constant
            ) / n_eff

        return jnp.where(invalid, 1e10, nll)

    return _value_grad_hess(objective, theta)


def arma_egarch_logspace_value_grad_hess(
    z: NDArray[np.float64],
    y: NDArray[np.float64],
    p_ar: int,
    q_ma: int,
    p_arch: int,
    q_egarch: int,
    dist: str = "normal",
) -> Tuple[float, NDArray[np.float64], NDArray[np.float64]]:
    """Reference value/gradient/Hessian for transformed ARMA(p,q)+EGARCH(P,Q) objectives."""
    jax, jnp, jsp = _jax_modules()
    z_j = jnp.asarray(z, dtype=jnp.float64)
    y_j = jnp.asarray(y, dtype=jnp.float64)
    n = int(y.shape[0])
    max_lag = max(p_ar, q_ma, p_arch, q_egarch, 1)

    def studentt_abs_moment(nu):
        c_log = (
            jsp.special.gammaln(0.5 * (nu + 1.0))
            - jsp.special.gammaln(0.5 * nu)
            - 0.5 * jnp.log(jnp.pi * (nu - 2.0))
        )
        return 2.0 * jnp.exp(c_log) * (nu - 2.0) / (nu - 1.0)

    def objective(z_theta):
        n_mean = 1 + p_ar + q_ma
        k_vol = 1 + 2 * p_arch + q_egarch

        c = z_theta[0]
        phi = 0.99 * jnp.tanh(z_theta[1:1 + p_ar])
        theta_ma = 0.99 * jnp.tanh(z_theta[1 + p_ar:n_mean])

        z_egarch = z_theta[n_mean:n_mean + k_vol]
        omega = z_egarch[0]
        alpha = z_egarch[1:1 + p_arch]
        gamma = z_egarch[1 + p_arch:1 + 2 * p_arch]
        beta = jnp.tanh(z_egarch[1 + 2 * p_arch:1 + 2 * p_arch + q_egarch])

        if dist == "normal":
            abs_moment = jnp.sqrt(2.0 / jnp.pi)
            nu = None
        elif dist == "studentt":
            nu = 2.0 + _softplus_production(jnp, z_theta[n_mean + k_vol])
            abs_moment = studentt_abs_moment(nu)
        else:
            raise ValueError(f"Unsupported ARMA-EGARCH distribution: {dist}")

        h0 = jnp.maximum(jnp.mean(y_j * y_j), GJR_H_FLOOR)
        x0 = jnp.log(h0)
        resid_hist0 = jnp.zeros((max_lag,), dtype=z_theta.dtype)
        x_hist0 = jnp.full((max_lag,), x0, dtype=z_theta.dtype)

        def step(carry, t):
            resid_hist, x_hist = carry

            mu_t = c
            for i in range(p_ar):
                mu_t = mu_t + phi[i] * y_j[t - 1 - i]
            for j in range(q_ma):
                mu_t = mu_t + theta_ma[j] * resid_hist[j]
            e_t = y_j[t] - mu_t

            x_t = omega
            for i in range(p_arch):
                z_lag = resid_hist[i] * jnp.exp(-0.5 * x_hist[i])
                x_t = x_t + alpha[i] * (jnp.abs(z_lag) - abs_moment) + gamma[i] * z_lag
            for j in range(q_egarch):
                x_t = x_t + beta[j] * x_hist[j]

            h_t = jnp.maximum(jnp.exp(x_t), GJR_H_FLOOR)
            x_t = jnp.log(h_t)

            resid_hist_next = jnp.concatenate([e_t[None], resid_hist[:-1]])
            x_hist_next = jnp.concatenate([x_t[None], x_hist[:-1]])
            return (resid_hist_next, x_hist_next), (e_t, x_t, h_t)

        (_, _), out = jax.lax.scan(step, (resid_hist0, x_hist0), jnp.arange(max_lag, n))
        resid_tail = out[0]
        x_tail = out[1]
        sigma_tail = out[2]
        n_eff = n - max_lag

        if dist == "normal":
            return jnp.sum(0.5 * (x_tail + resid_tail * resid_tail / sigma_tail)) / n_eff

        assert nu is not None
        constant = (
            jsp.special.gammaln(0.5 * (nu + 1.0))
            - jsp.special.gammaln(0.5 * nu)
            - 0.5 * jnp.log(jnp.pi * (nu - 2.0))
        )
        tail = jnp.log1p((resid_tail * resid_tail / sigma_tail) / (nu - 2.0))
        return (jnp.sum(0.5 * x_tail + 0.5 * (nu + 1.0) * tail) - n_eff * constant) / n_eff

    return _value_grad_hess(objective, np.asarray(z_j, dtype=np.float64))
