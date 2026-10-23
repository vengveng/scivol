"""Direct DCC calls preserve the Gaussian-limit shape jet without fit caps."""

import numpy as np
import pytest

from scivol import _dcc_kernels as kernels


@pytest.mark.parametrize("dimension", [2, 3, 8])
@pytest.mark.parametrize("nu", [8., 128., 200., 1e7, 1e15])
def test_complete_dcc_student_shape_jet_matches_mp(dimension, nu):
    mp = pytest.importorskip("mpmath")
    eps = np.random.default_rng(270).normal(size=(5, dimension))
    qbar = np.eye(dimension)
    theta = np.array([0., 0., nu])
    actual = kernels.dcc_studentt_nll_grad_hess(theta, eps, qbar, 1, 1,
                                              return_scores=True)
    with mp.workdps(100):
        observations = [[mp.mpf(float(e)) for e in row] for row in eps]

        def objective(degrees):
            constant = -mp.loggamma((degrees+dimension)/2)
            constant += (1-dimension)*mp.loggamma(degrees/2)
            constant += dimension*mp.loggamma((degrees+1)/2)
            total = mp.mpf(0)
            for row in observations:
                total += constant+(degrees+dimension)/2*mp.log1p(
                    sum(e*e for e in row)/(degrees-2))
                total -= (degrees+1)/2*sum(mp.log1p(e*e/(degrees-2)) for e in row)
            return total/len(observations)

        expected = [float(objective(mp.mpf(nu))), float(mp.diff(objective, mp.mpf(nu))),
                    float(mp.diff(objective, mp.mpf(nu), 2))]
    np.testing.assert_allclose(actual[0], expected[0], rtol=3e-11, atol=3e-13)
    np.testing.assert_allclose([actual[1][-1], actual[2][-1, -1]], expected[1:],
                               rtol=3e-8 if nu < 128 else 3e-12, atol=0)
    assert kernels.dcc_studentt_nll(theta, eps, qbar, 1, 1) == pytest.approx(actual[0], abs=1e-13)
    np.testing.assert_allclose(-actual[3].mean(axis=0), actual[1], rtol=3e-12, atol=0)


@pytest.mark.parametrize("orders", [(1, 1), (2, 1)])
def test_dcc_large_nu_keeps_correlation_shape_crosses(orders):
    mp = pytest.importorskip("mpmath")
    p, q = orders
    eps = np.random.default_rng(271).normal(size=(5, 2))
    qbar = np.array([[1., .2], [.2, 1.3]])
    theta = np.r_[np.full(p, .1/p), .7, 1e15]
    actual = kernels.dcc_studentt_nll_grad_hess(theta, eps, qbar, p, q)
    with mp.workdps(100):
        rows, base = [mp.matrix(list(map(mp.mpf, row))) for row in eps], mp.matrix(qbar.tolist())

        def objective(*point):
            degrees, previous, shocks = point[-1], base, [mp.zeros(2, 1) for _ in range(p)]
            constant = -mp.loggamma((degrees+2)/2)-mp.loggamma(degrees/2)
            constant += 2*mp.loggamma((degrees+1)/2)
            total = mp.mpf(0)
            for residual in rows:
                current = (1-sum(point[:-1]))*base+point[p]*previous
                for weight, shock in zip(point[:p], shocks, strict=True):
                    current += weight*shock*shock.T
                rho = current[0, 1]/mp.sqrt(current[0, 0]*current[1, 1])
                squared = (residual[0]**2+residual[1]**2-2*rho*residual[0]*residual[1])/(1-rho*rho)
                total += constant+mp.log1p(-rho*rho)/2+(degrees+2)/2*mp.log1p(squared/(degrees-2))
                total -= (degrees+1)/2*sum(mp.log1p(e*e/(degrees-2)) for e in residual)
                previous, shocks = current, [residual, *shocks[:-1]]
            return total/len(rows)

        point = list(map(mp.mpf, theta))
        gradient = [float(mp.diff(objective, point, tuple(int(i==k) for i in range(len(point)))))
                    for k in range(len(point))]
        hessian = [[float(mp.diff(objective, point, tuple(int(i==k)+int(i==j) for i in range(len(point)))))
                    for j in range(len(point))] for k in range(len(point))]
    np.testing.assert_allclose(actual[1], gradient, rtol=3e-11, atol=0)
    np.testing.assert_allclose(actual[2], hessian, rtol=3e-11, atol=0)
