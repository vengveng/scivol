"""Five exact header cases from the nine-case compound old node."""
from pathlib import Path
import json
import os
import subprocess
import pytest

@pytest.mark.parametrize("source", (
    "tests/native/transfer_006_location_scale_assembly.cpp", "tests/native/transfer_006_reflection_composition.cpp",
    "tests/native/transfer_006_mean_workspace.cpp", "tests/native/transfer_006_joint_garch_workspace.cpp", "tests/native/transfer_006_gh_quadrature.cpp"))
def test_historical_mean_header_cases(source):
    directory=os.environ.get("SCIVOLAPI_NUMERICAL_EXECUTABLE_DIR")
    if not directory:
        raise RuntimeError("Chief must build the staged header probes before running this test")
    executable=Path(directory)/Path(source).stem
    output=subprocess.run([str(executable)],check=True,capture_output=True,text=True,timeout=10).stdout
    if source=="tests/native/transfer_006_mean_workspace.cpp": verify_native_range_primitives(output)
    if source=="tests/native/transfer_006_gh_quadrature.cpp": verify_packed_density_oracles(output)

def verify_native_range_primitives(output):
    from fractions import Fraction
    import numpy as np
    mp = __import__("mpmath")
    affine_rows = 0
    with mp.workdps(100):
        for line in output.splitlines():
            row = json.loads(line)
            if row["kind"] == "aparch-affine":
                affine_rows += 1
                inputs, actual, reference = row["inputs"], row["actual"], row["reference"]
                assert np.isfinite(actual).all() and np.isfinite(reference).all()
                a, al, c, b, bl = inputs
                if 2.**-900 <= c*b <= 2.**900:
                    aa, aal, cc, bb, bbl = map(Fraction.from_float, inputs)
                    exact = aa + aal + cc*(bb+bbl)
                    error = abs(sum(map(Fraction.from_float, actual))-exact)
                    bound = 8*Fraction(1, 2**53)**2*(abs(aa)+abs(cc*bb))+16*Fraction(1, 2**1074)
                    assert error <= bound
                else:
                    np.testing.assert_array_equal(np.array(actual).view(np.uint64),
                                                  np.array(reference).view(np.uint64))
            elif row["kind"] == "ged-log":
                verify_ged_log_oracle(row)
            elif row["kind"] == "tail-map":
                x = mp.mpf(row["x"])
                function = (lambda z: 1 / (1 + mp.exp(-z))) if row["logistic"] else mp.tanh
                np.testing.assert_allclose([row["value"], row["g"], row["h"]],
                    [float(function(x)), float(mp.diff(function,x)),
                     0.0 if x == 0 else float(mp.diff(function,x,2))],
                    rtol=4e-15, atol=0)
            elif row["kind"] == "reflection-map":
                x = mp.mpf(row["x"])
                np.testing.assert_allclose([row["g"], row["h"]],
                    [float(mp.diff(mp.tanh, x)),
                     0.0 if x == 0 else float(mp.diff(mp.tanh, x, 2))],
                    rtol=2e-12, atol=0)
            elif row["kind"] == "bounded-map":
                lower, upper, theta, x = (mp.mpf(row[key]) for key in ("lower","upper","theta","x"))
                function = lambda z: lower+(upper-lower)/(1+mp.exp(-z))
                np.testing.assert_allclose([row["g"],row["h"]],
                    [float(mp.diff(function,x)),float(mp.diff(function,x,2))],rtol=4e-15,atol=0)
                inverse = mp.log(theta-lower)-mp.log(upper-theta)
                np.testing.assert_allclose([row["theta_g"],row["theta_h"]],
                    [float(mp.diff(function,inverse)),float(mp.diff(function,inverse,2))],rtol=4e-15,atol=0)
            else:
                nu, e = mp.mpf(row["nu"]),mp.mpf(row["e"])
                scale=(mp.loggamma(1/nu)-mp.loggamma(3/nu))/2
                scale_prime=(-mp.digamma(1/nu)+3*mp.digamma(3/nu))/(2*nu**2)
                power=mp.exp(nu*(mp.log(abs(e))-scale))
                gradient=nu*power/e
                expected=[gradient,nu*(nu-1)*power/e**2,-nu*gradient/2,
                          gradient*(1/nu+mp.log(abs(e))-scale-nu*scale_prime)]
                np.testing.assert_allclose([row[key] for key in ("g","h","eh","enu")],
                    [float(value) for value in expected],rtol=3e-12,atol=0)
    assert affine_rows == 9



def verify_ged_log_oracle(row):
    import numpy as np
    from .oracle_ad import _jax_modules, _value_grad_hess

    _, jnp, jsp = _jax_modules()
    residual, log_variance, nu = (row[key] for key in ("e", "logh", "nu"))

    def log_scale(shape):
        return (jsp.special.gammaln(1 / shape) - jsp.special.gammaln(3 / shape)) / 2

    def normalizer(point):
        _, logh, shape = point
        return -jnp.log(shape) + jnp.log(2.) + log_scale(shape) + jsp.special.gammaln(1 / shape) + logh / 2

    point = np.array([0., log_variance, nu])
    value, gradient, hessian = _value_grad_hess(normalizer, point)
    if residual:
        log_abs = np.log(abs(residual))
        anchor = float(nu * (log_abs - log_variance / 2 - log_scale(nu)))

        def scaled_power(point):
            relative_residual, logh, shape = point
            return jnp.exp(shape * (log_abs + jnp.log1p(relative_residual)
                                   - logh / 2 - log_scale(shape)) - anchor)

        power, power_gradient, power_hessian = _value_grad_hess(scaled_power, point)

        def restore(entry, residual_partials):
            # Undo the constant likelihood scale and relative coordinate together;
            # the density value can underflow while its residual curvature is finite.
            if entry == 0:
                return 0.
            magnitude = np.exp(np.log(abs(entry)) + anchor - residual_partials * log_abs)
            sign = np.sign(entry) * (np.sign(residual) if residual_partials == 1 else 1)
            return sign * magnitude

        gradient, hessian = gradient.copy(), hessian.copy()
        value += restore(power, 0)
        for index in range(3):
            gradient[index] += restore(power_gradient[index], int(index == 0))
            for column in range(3):
                hessian[index, column] += restore(
                    power_hessian[index, column], int(index == 0) + int(column == 0))
    elif nu == 2:
        def gaussian_power(point):
            r, logh, shape = point
            return r ** 2 * jnp.exp(-2 * log_scale(shape) - logh)
        power, power_gradient, power_hessian = _value_grad_hess(gaussian_power, point)
        value, gradient, hessian = value + power, gradient + power_gradient, hessian + power_hessian
    np.testing.assert_allclose(row["f"], value, rtol=0, atol=2e-12)
    np.testing.assert_allclose(row["g"], gradient, rtol=2e-8, atol=2e-9)
    np.testing.assert_allclose(row["h"], hessian[np.triu_indices(3)], rtol=2e-7, atol=2e-8)



def verify_packed_density_oracles(output):
    __import__("jax")
    import numpy as np
    from .oracle_ad import (
        _jax_modules, _jax_gh_logpdf, _jax_vg_logpdf, _value_grad_hess,
        vg_observation_mixture_value_grad_hess,
    )

    _, jnp, jsp = _jax_modules()
    rows = [json.loads(line) for line in output.splitlines()]
    assert len(rows) == 16
    for row in rows:
        if row.get("near_center"):
            point = np.asarray(row["point"])
            expected = vg_observation_mixture_value_grad_hess(point)
            refined = vg_observation_mixture_value_grad_hess(point, nodes_per_interval=24)
            for left, right in zip(expected, refined):
                np.testing.assert_allclose(left, right, rtol=2e-10, atol=2e-10)
            value, gradient, hessian = expected
            np.testing.assert_allclose(row["value"], value, rtol=2e-11, atol=2e-11)
            np.testing.assert_allclose(row["gradient"], gradient, rtol=2e-9, atol=2e-9)
            packed = hessian[np.triu_indices(4)]
            np.testing.assert_allclose(row["hessian"], packed, rtol=2e-8, atol=2e-8)
            continue
        family, residual_derivatives = row["family"], row["residual_derivatives"]
        shape = [.07, 3.85] if family == 7 else [.17, 1.8]
        if family == 4:
            shape += [-.7]
        point = np.array(([-1.3] if residual_derivatives else [])
                         + [.33287108369807955] + shape)

        def objective(local):
            residual = local[0] if residual_derivatives else -1.3
            variance = local[int(residual_derivatives)]
            shapes = local[1 + int(residual_derivatives):]
            standardized = jnp.array([residual / jnp.sqrt(variance)])
            if family == 7:
                logpdf = _jax_vg_logpdf(jnp, jsp, standardized, *shapes)
            else:
                order = shapes[2] if family == 4 else (-.5 if family == 5 else 1.)
                logpdf = _jax_gh_logpdf(jnp, jsp, standardized, *shapes[:2], order)
            return .5 * jnp.log(variance) - logpdf[0]

        value, gradient, hessian = _value_grad_hess(objective, point)
        np.testing.assert_allclose(row["value"], value, rtol=2e-10, atol=2e-11)
        np.testing.assert_allclose(row["gradient"], gradient, rtol=2e-9, atol=2e-10)
        np.testing.assert_allclose(row["hessian"], hessian[np.triu_indices(len(point))],
                                   rtol=2e-8, atol=2e-9)
