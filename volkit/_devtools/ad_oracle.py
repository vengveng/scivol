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


def garch_value_grad_hess(
    theta: NDArray[np.float64],
    resid2: NDArray[np.float64],
    p: int,
    q: int,
    dist: str = "normal",
) -> Tuple[float, NDArray[np.float64], NDArray[np.float64]]:
    """Reference NLL, gradient, and Hessian for GARCH(p,q)."""
    jax, jnp, jsp = _jax_modules()
    resid2_j = jnp.asarray(resid2, dtype=jnp.float64)
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

        raise ValueError(f"Unsupported GARCH distribution: {dist}")

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

        raise ValueError(f"Unsupported GJR-GARCH distribution: {dist}")

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
        if dist == "studentt":
            nu = theta_j[idx]
            invalid = jnp.logical_or(invalid, nu <= 2.001)

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
        else:
            raise ValueError(f"Unsupported ARMA-GARCH distribution: {dist}")

        return jnp.where(invalid, 1e10, nll)

    return _value_grad_hess(objective, theta)
