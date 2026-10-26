"""Independent AD checks for zero-order kernels on explicit comparison plans."""
from __future__ import annotations

import numpy as np
import pytest
from numpy.testing import assert_allclose

from scivol import _scivolapi as api
from tests._oracle import _jax_modules
from tests.test_native_restrictions import model


@pytest.mark.parametrize("kind", ["garch", "gjr", "egarch", "aparch"])
@pytest.mark.parametrize("p,q", [(0, 0), (0, 1), (1, 0), (2, 0)])
def test_zero_orders_and_common_plan_against_ad(kind, p, q):
    jax, jnp, _ = _jax_modules()
    y = np.random.default_rng(20).normal(size=20)
    first, seed = 5, 1.37
    native = model(kind, p, q, ar=1, ma=1, constant=True)
    plan = api.ComparisonPlan(y, first, variance_seed=seed)
    theta = native.start(plan)
    theta[:3] = [.03, .2, -.1]
    if kind == "aparch":
        theta[-1] = 1.6
    def oracle(t):
        e, h = [], []
        m, gamma, beta = 3, 4+p, 4+(p if kind == "garch" else 2*p)
        for row in range(first, len(y)):
            k = row-first
            e.append(y[row] - t[0] - t[1]*y[row-1] - (t[2]*e[-1] if k else 0))
            if k == 0:
                h.append(jnp.asarray(seed))
                continue
            value = t[m]
            for i in range(1, min(p, k)+1):
                if kind in ("garch", "gjr"):
                    a = t[m+i] + (t[gamma+i-1]*(e[k-i] < 0) if kind == "gjr" else 0)
                    value += a*e[k-i]**2
                elif kind == "egarch":
                    z = e[k-i]/jnp.sqrt(h[k-i])
                    value += t[m+i]*(jnp.abs(z)-jnp.sqrt(2/jnp.pi)) + t[gamma+i-1]*z
                else:
                    value += t[m+i]*(jnp.abs(e[k-i])-t[gamma+i-1]*e[k-i])**t[-1]
            for j in range(1, min(q, k)+1):
                past = jnp.log(h[k-j]) if kind == "egarch" else h[k-j]**(t[-1]/2) if kind == "aparch" else h[k-j]
                value += t[beta+j-1]*past
            if kind == "egarch":
                for j in range(k+1, q+1):
                    value += t[beta+j-1]*np.log(seed)
            h.append(jnp.exp(value) if kind == "egarch" else value**(2/t[-1]) if kind == "aparch" else value)
        h, e = jnp.array(h), jnp.array(e)
        return .5*jnp.sum(jnp.log(2*jnp.pi)+jnp.log(h)+e*e/h)
    expected = [oracle(theta), jax.grad(oracle)(theta), jax.hessian(oracle)(theta)]
    jet = api.Jet(native, plan)
    for mask in range(1, 8):
        out = jet.theta(theta, mask)
        assert out["status"] == "ok"
        for bit, name, reference in zip([1, 2, 4], ["value", "gradient", "hessian"], expected):
            if bit & mask:
                assert_allclose(out[name], reference, rtol=3e-10, atol=3e-10)
    if kind == "garch" or p == q == 0:
        z = native.to_z(theta)
        bound, tiny = np.nextafter(1.0, 0.0), np.finfo(float).tiny
        def chart(u):
            mean = jnp.stack([u[0], bound*jnp.tanh(u[1]), -bound*jnp.tanh(u[2])])
            omega = u[3] if kind == "egarch" else tiny+jnp.exp(u[3])
            if kind == "garch":
                shares = jnp.exp(u[4:])
                coefficients = (1-8*(p+q+1)*np.finfo(float).eps)*shares/(1+jnp.sum(shares))
                return jnp.concatenate([mean, omega[None], coefficients])
            tail = jnp.array([tiny+jnp.exp(u[4])]) if kind == "aparch" else jnp.array([])
            return jnp.concatenate([mean, omega[None], tail])
        def transformed(u):
            return oracle(chart(u))
        assert_allclose(chart(z), theta, rtol=2e-14, atol=1e-14)
        expected_z = [transformed(z), jax.grad(transformed)(z), jax.hessian(transformed)(z)]
        for mask in range(1, 8):
            out = jet.z(z, mask)
            assert out["status"] == "ok"
            for bit, name, reference in zip([1, 2, 4], ["value", "gradient", "hessian"], expected_z):
                if bit & mask:
                    assert_allclose(out[name], reference, rtol=3e-10, atol=3e-10)
