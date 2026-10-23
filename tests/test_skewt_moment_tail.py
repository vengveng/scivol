"""Independent density-integral controls for Hansen's finite moment boundary."""

import numpy as np
import pytest

from scivol import _native_core
from scivol._devtools.ad_oracle import _jax_modules


def native_moment(point):
    shape = np.ascontiguousarray(point[2:])
    results = []
    for request in range(1, 8):
        f, g, h = np.full(1, np.nan), np.full(4, np.nan), np.full((4, 4), np.nan)
        assert _native_core._aparch_asymmetric_power_moment(
            shape.ctypes.data, 2, point[0], point[1], request,
            f.ctypes.data, g.ctypes.data, h.ctypes.data) == 0
        results.append((f, g, h))
    for request, result in enumerate(results, 1):
        for bit, actual, full in zip((1, 2, 4), result, results[-1], strict=True):
            if request & bit:
                np.testing.assert_allclose(actual, full, rtol=2e-10, atol=2e-11)
            else:
                assert np.isnan(actual).all()
    f, g, h = results[-1]
    return f[0], g, h


def integrated_density_beta_jet(point):
    """Integrate skew derivatives of the original density, then differentiate."""
    mp = pytest.importorskip("mpmath")
    with mp.workdps(90):
        def local(gamma, delta, nu, skew):
            k = nu-2
            c = mp.gamma((nu+1)/2)/(mp.gamma(nu/2)*mp.sqrt(mp.pi*k))
            a = 4*c*k/(nu-1)
            def q(power, order):
                return c/2*k**((power+1)/2-order)*mp.beta(
                    (power+1)/2, (nu+1)/2+order-(power+1)/2)
            half = q(delta, 0)
            first = (nu+1)*(q(delta+2, 1)-a*q(delta+1, 1))
            second = (3-a*a)*half
            second -= (nu+1)*(a*a*k*q(delta, 2)-2*a*k*q(delta+1, 2)
                +k*q(delta+2, 2)-a*a*q(delta+2, 2)+2*a*q(delta+3, 2)-q(delta+4, 2))
            second -= (nu+1)*((5-a*a)*q(delta+2, 1)-2*a*q(delta+1, 1))
            second += (nu+1)**2*(a*a*q(delta+2, 2)-2*a*q(delta+3, 2)+q(delta+4, 2))
            plus, minus = (1-gamma)**delta, (1+gamma)**delta
            return (plus+minus)*half+skew*(plus-minus)*first+skew*skew/2*(plus+minus)*second
        x = list(map(mp.mpf, point))
        gradient = [mp.diff(local, x, tuple(int(k==i) for k in range(4))) for i in range(4)]
        hessian = [[mp.diff(local, x, tuple(int(k==i)+int(k==j) for k in range(4)))
                    for j in range(4)] for i in range(4)]
        return float(local(*x)), np.array(gradient, dtype=float), np.array(hessian, dtype=float)


@pytest.mark.parametrize("point", [(.25, .2, 8., 0.), (.25, 3.7, 3.700001, 0.),
    (.25, 2.0001, 2.0002, 0.), (.25, 1e-20, 8., 0.), (1e-20, .7, 8., 0.)])
def test_zero_skew_retains_all_shape_crosses(point):
    actual = native_moment(point)
    expected = integrated_density_beta_jet(point)
    for left, right in zip(actual, expected, strict=True):
        np.testing.assert_allclose(left, right, rtol=3e-9, atol=3e-10)
    if min(point[:2]) < 1e-15:
        assert actual[1][3] == pytest.approx(expected[1][3], rel=2e-12, abs=0)


def recurrence_oracle(point, nodes_count):
    """Lower-power integration by parts, independent of the Euler subtraction."""
    jax, jnp, jsp = _jax_modules()
    nodes, weights = np.polynomial.legendre.leggauss(nodes_count)
    u, weights = jnp.asarray((nodes+1)/2), jnp.asarray(weights/2)

    def moment(x):
        gamma, delta, nu, skew = x
        s, k = jnp.abs(skew), nu-2
        c = jnp.exp(jsp.special.gammaln((nu+1)/2)-jsp.special.gammaln(nu/2))/jnp.sqrt(jnp.pi*k)
        a = 4*s*c*k/(nu-1)
        b = jnp.sqrt(1+3*s*s-a*a)
        mu = a/b
        sl, sr = jnp.sqrt(k)*(1-s)/b, jnp.sqrt(k)*(1+s)/b
        radius = (u/(1-u))**4

        def lower(power, positive):
            scale = sr if positive else sl
            z = scale*radius+(0. if positive else mu)
            t = (z+mu)/scale if positive else radius
            log_jacobian = jnp.log(scale*4)+3*jnp.log(u)-5*jnp.log1p(-u)
            return jnp.sum(weights*jnp.exp(power*jnp.log(z)+jnp.log(b*c)
                -(nu+1)/2*jnp.log1p(t*t)+log_jacobian))

        positive = (mu*(2*delta-nu-1)*lower(delta-1, True)
            +(delta-1)*(mu*mu+sr*sr)*lower(delta-2, True))/(nu-delta)
        negative = (mu*(nu-2*delta+1)*lower(delta-1, False)
            +(delta-1)*(mu*mu+sl*sl)*lower(delta-2, False)
            +b*c*sl*sl*mu**(delta-1))/(nu-delta)
        v = u**4
        negative += c*a**(delta+1)/b**delta*jnp.sum(weights*4*u**3
            *v**delta*(1+(a*(1-v)/(1+s))**2/k)**(-(nu+1)/2))
        p, n = jnp.where(skew<0, negative, positive), jnp.where(skew<0, positive, negative)
        return (1-gamma)**delta*p+(1+gamma)**delta*n

    x = jnp.asarray(point)
    return float(moment(x)), np.asarray(jax.grad(moment)(x)), np.asarray(jax.hessian(moment)(x))


@pytest.mark.parametrize("point", [(-.3, 3.7, 3.701, .4), (.2, 3.7, 3.700001, -.8),
    (.2, 2.0001, 2.0002, .7), (.2, 3.7, 3.700001, .9999), (.2, 2.7, 2.701, 1e-8)])
def test_nonzero_skew_tail_pole_matches_independent_recurrence_ad(point):
    actual = native_moment(point)
    coarse, expected = recurrence_oracle(point, 256), recurrence_oracle(point, 512)
    for left, right in zip(coarse, expected, strict=True):
        np.testing.assert_allclose(left, right, rtol=2e-8, atol=2e-9)
    for left, right in zip(actual, expected, strict=True):
        np.testing.assert_allclose(left, right, rtol=2e-8, atol=2e-9)
