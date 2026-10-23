"""Compile shared header contracts independently of an installed extension."""

from pathlib import Path
import json
import re
import shutil
import subprocess

import pytest


@pytest.mark.parametrize(
    "source",
    [
        "native_location_scale_assembly.cpp",
        "native_reflection_composition.cpp",
        "native_mean_workspace.cpp",
        "native_joint_garch_workspace.cpp",
        "native_linear_power_workspace.cpp",
        "native_runtime_equalities.cpp",
        "native_nnls_workspace.cpp",
        "native_portable_linalg.cpp",
        "native_gh_quadrature.cpp",
    ],
)
def test_mean_primitives(source, tmp_path):
    compiler = shutil.which("clang++") or shutil.which("g++")
    if compiler is None:
        pytest.skip("A C++20 compiler is required for the header contract test")
    root = Path(__file__).resolve().parents[1]
    executable = tmp_path / "header_contract"
    frame_flags = (
        ["-Wframe-larger-than=16384", "-Werror=frame-larger-than"]
        if "clang" in Path(compiler).name
        else ["-Werror=frame-larger-than=16384"]
    )
    extra_sources = (
        [str(root / "scivol/_native/gh_math_translation.cpp")]
        if source
        in {
            "native_mean_workspace.cpp",
            "native_joint_garch_workspace.cpp",
            "native_gh_quadrature.cpp",
        }
        else []
    )
    if source in {"native_runtime_equalities.cpp", "native_nnls_workspace.cpp"}:
        extra_sources = [
            str(root / "scivol/_native" / name)
            for name in (
                "slsqp_translation.cpp",
                "portable_linalg.cpp",
                "portable_nnls.cpp",
            )
        ]
    if source == "native_portable_linalg.cpp":
        extra_sources = [str(root / "scivol/_native/portable_linalg.cpp")]
    optimization = (
        ["-O3", "-ffp-contract=fast", "-funroll-loops"]
        if source
        in {
            "native_location_scale_assembly.cpp",
            "native_reflection_composition.cpp",
            "native_mean_workspace.cpp",
            "native_linear_power_workspace.cpp",
            "native_runtime_equalities.cpp",
            "native_nnls_workspace.cpp",
            "native_portable_linalg.cpp",
            "native_gh_quadrature.cpp",
        }
        else ["-O2"]
    )
    if source == "native_linear_power_workspace.cpp":
        extra_sources = build_aparch_moment_backend(compiler, root, tmp_path, frame_flags)
    if source in {"native_mean_workspace.cpp", "native_linear_power_workspace.cpp"}:
        optimization += ["-fno-exceptions", "-fno-rtti"]
    subprocess.run(
        [
            compiler,
            "-std=c++20",
            *optimization,
            *frame_flags,
            "-I",
            str(root / "scivol/_native"),
            "-I",
            str(root / "scivol/_csrc"),
            "-I",
            str(root / "scivol/_native/third_party/boost"),
            "-I",
            str(root / "scivol/_native/third_party/tlfloat"),
            str(root / "tests" / source),
            *extra_sources,
            "-o",
            str(executable),
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=120 if source == "native_joint_garch_workspace.cpp" else 60,
    )
    completed = subprocess.run([str(executable)], check=True, timeout=10,
                               capture_output=True, text=True)
    if source == "native_gh_quadrature.cpp":
        verify_packed_density_oracles(completed.stdout)
    if source == "native_mean_workspace.cpp":
        verify_native_range_primitives(completed.stdout)
    if source == "native_linear_power_workspace.cpp":
        verify_aparch_precision_moments(completed.stdout)
        verify_aparch_batch_derivatives(completed.stdout)
        verify_aparch_weighted_consumers(completed.stdout)
        verify_aparch_range_derivatives(completed.stdout)
        verify_skew_tail_precision(completed.stdout)


def verify_native_range_primitives(output):
    from fractions import Fraction
    import numpy as np
    mp = pytest.importorskip("mpmath")
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
    from scivol._devtools.ad_oracle import _jax_modules, _value_grad_hess

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


def verify_aparch_precision_moments(output):
    """Independent moment identities, not the density quadrature under test."""
    import mpmath as mp

    rows = [json.loads(line) for line in output.splitlines()]
    rows = [row for row in rows if not any(row.get(key) for key in
            ("moment_jet", "weighted_map", "student_tail_map", "scaled_log_moment", "skew_tail_precision"))]
    assert len(rows) == 82
    with mp.workdps(160):
        for row in rows:
            positive = mp.fsum(map(mp.mpf, row["positive"]))
            negative = mp.fsum(map(mp.mpf, row["negative"]))
            if row.get("variance"):
                assert abs(positive+negative-1) < mp.mpf("1e-24"), row
                continue
            power = mp.mpf(row["power"])
            shape = list(map(mp.mpf, row["shape"]))
            normal = mp.power(2, power/2)*mp.gamma((power+1)/2)/mp.sqrt(mp.pi)
            code = row["density"]
            if code == 0:
                moment = normal
            elif code in (1, 2):
                nu = shape[0]
                moment = normal*mp.exp(power/2*mp.log((nu-2)/2)
                    +mp.loggamma((nu-power)/2)-mp.loggamma(nu/2))
            elif code == 3:
                nu = shape[0]
                scale = mp.sqrt(mp.gamma(1/nu)/mp.gamma(3/nu))
                moment = scale**power*mp.gamma((power+1)/nu)/mp.gamma(1/nu)
            elif code in (4, 5, 6):
                zeta = shape[1]
                order = shape[2] if code == 4 else mp.mpf("-.5") if code == 5 else mp.mpf(1)
                k0 = mp.besselk(order, zeta)
                moment = normal*mp.besselk(order+power/2, zeta)/k0
                moment *= mp.power(k0/mp.besselk(order+1, zeta), power/2)
            else:
                order = shape[1]
                moment = normal*mp.gamma(order+power/2)/(mp.gamma(order)*mp.power(order, power/2))
            for half in (positive, negative):
                assert abs(2*half/moment-1) < mp.mpf("1e-24"), row


def verify_skew_tail_precision(output):
    """Wide scalar halves against independently integrated lower moments."""
    import mpmath as mp

    rows = [json.loads(line) for line in output.splitlines()]
    rows = [row for row in rows if row.get("skew_tail_precision")]
    assert len(rows) == 5
    with mp.workdps(80):
        for row in rows:
            d, nu, skew = map(mp.mpf, row["point"])
            k, s = nu-2, abs(skew)
            c = mp.gamma((nu+1)/2)/(mp.sqrt(mp.pi*k)*mp.gamma(nu/2))
            a = 4*s*c*k/(nu-1)
            b = mp.sqrt(1+3*s*s-a*a)
            mu, sl, sr = a/b, mp.sqrt(k)*(1-s)/b, mp.sqrt(k)*(1+s)/b
            if not s:
                p = n = k**(d/2)*mp.beta((d+1)/2,(nu-d)/2)/(2*mp.beta(mp.mpf('.5'),nu/2))
            else:
                def lower(power, positive):
                    scale = sr if positive else sl
                    def integrand(x):
                        z = scale*x+(0 if positive else mu)
                        t = (z+mu)/scale if positive else x
                        return z**power*b*c*(1+t*t)**(-(nu+1)/2)*scale
                    return mp.quad(integrand, [0, 1, mp.inf])
                p = (mu*(2*d-nu-1)*lower(d-1, True)
                    +(d-1)*(mu*mu+sr*sr)*lower(d-2, True))/(nu-d)
                n = (mu*(nu-2*d+1)*lower(d-1, False)
                    +(d-1)*(mu*mu+sl*sl)*lower(d-2, False)
                    +b*c*sl*sl*mu**(d-1))/(nu-d)
                n += c*a**(d+1)/b**d*mp.quad(
                    lambda u: u**d*(1+(a*(1-u)/(1+s))**2/k)**(-(nu+1)/2), [0, 1])
                if skew < 0:
                    p, n = n, p
            for actual, expected in zip((row["positive"], row["negative"]), (p, n), strict=True):
                assert abs(mp.fsum(map(mp.mpf, actual))/expected-1) < mp.mpf('1e-24'), row


def verify_aparch_batch_derivatives(output):
    pytest.importorskip("jax")
    import numpy as np
    from scivol._devtools.ad_oracle import aparch_power_moment_value_grad_hess

    rows = [json.loads(line) for line in output.splitlines()]
    rows = [row for row in rows if row.get("moment_jet")]
    assert len(rows) == 30
    names = {2: "skewt", 4: "gh", 5: "nig", 6: "hyperbolic", 7: "vg"}
    for row in rows:
        point = np.asarray(row["point"])
        value, gradient, hessian = aparch_power_moment_value_grad_hess(
            point, names[row["density"]])
        assert row["value"] == pytest.approx(value, rel=2e-9, abs=2e-11)
        np.testing.assert_allclose(row["gradient"], gradient, rtol=2e-8, atol=2e-10)
        np.testing.assert_allclose(row["hessian"], hessian[np.triu_indices(len(point))],
                                   rtol=2e-7, atol=1e-8)


def verify_aparch_weighted_consumers(output):
    """Differentiate the scaled map without forming overflowing raw moments."""
    pytest.importorskip("jax")
    import numpy as np
    from scivol._devtools.ad_oracle import (
        _jax_modules, _jax_gh_logpdf, _native_shape_coordinates,
    )

    jax, jnp, jsp = _jax_modules()
    rows = [json.loads(line) for line in output.splitlines()]
    rows = [row for row in rows if row.get("weighted_map")]
    assert len(rows) == 2
    for row in rows:
        alpha, gamma, delta, shape = (row[name] for name in ("alpha", "gamma", "delta", "shape"))
        beta = gamma+row["p"]
        z = jnp.array(row["z"])
        references = []
        for count in (512, 1024):
            nodes, weights = np.polynomial.legendre.leggauss(count)
            angle = jnp.array((nodes+1)*np.pi/4)
            magnitude = jnp.tan(angle)
            log_weight = jnp.log(jnp.array(weights*np.pi/4)/jnp.cos(angle)**2)

            def log_moment(point):
                asymmetry, power, rho, zeta, lam = point
                positive = _jax_gh_logpdf(jnp, jsp, magnitude, rho, zeta, lam)
                negative = _jax_gh_logpdf(jnp, jsp, -magnitude, rho, zeta, lam)
                positive += log_weight+power*(jnp.log(magnitude)+jnp.log1p(-asymmetry))
                negative += log_weight+power*(jnp.log(magnitude)+jnp.log1p(asymmetry))
                return jsp.special.logsumexp(jnp.concatenate([positive, negative]))

            def scaled_alpha(coordinates):
                shape_theta = _native_shape_coordinates(jax, jnp, coordinates[shape:], "gh")
                point = jnp.r_[jnp.tanh(coordinates[gamma]),
                               jax.nn.softplus(coordinates[delta]), shape_theta]
                contributions = jnp.r_[coordinates[alpha:gamma], coordinates[beta:delta], 0.]
                log_contribution = coordinates[alpha]-jsp.special.logsumexp(contributions)
                return jnp.exp(jnp.log(1e307)+log_contribution-log_moment(point))

            references.append(np.r_[float(scaled_alpha(z)), np.asarray(jax.grad(scaled_alpha)(z)),
                                     np.asarray(jax.hessian(scaled_alpha)(z)).ravel()])
        np.testing.assert_allclose(references[0], references[1], rtol=2e-8, atol=2e-10)
        actual = np.r_[row["value"], row["gradient"], row["hessian"]]
        np.testing.assert_allclose(actual, references[1], rtol=2e-7, atol=2e-10)
        theta = np.array(row["theta"])
        jacobian = np.zeros_like(theta)
        for lag in range(row["p"]):
            point = jnp.array(np.r_[theta[gamma+lag], theta[delta], theta[shape:]])
            log_value = float(log_moment(point))
            normalized_gradient = np.asarray(jax.grad(log_moment)(point))
            contribution = np.exp(np.log(theta[alpha+lag])+log_value)
            jacobian[alpha+lag] = -np.exp(log_value)
            jacobian[gamma+lag] = -contribution*normalized_gradient[0]
            jacobian[delta] -= contribution*normalized_gradient[1]
            jacobian[shape:] -= contribution*normalized_gradient[2:]
        jacobian[beta:delta] = -1.
        np.testing.assert_allclose(row["constraint_jacobian"], jacobian, rtol=2e-8, atol=2e-10)


def verify_aparch_range_derivatives(output):
    """Anchor independent chart derivatives to the actual mapped doubles."""
    import mpmath as mp
    import numpy as np
    from scivol._devtools.ad_oracle import _jax_modules

    jax, jnp, jsp = _jax_modules()
    rows = [json.loads(line) for line in output.splitlines()]
    maps = [row for row in rows if row.get("student_tail_map")]
    assert len(maps) == 16
    for row in maps:
        with mp.workdps(100):
            gamma, delta, nu = map(mp.mpf, row["theta"])
            g0, u0, v0 = map(mp.mpf, row["coordinates"])
            q0 = 1/(1+mp.exp(u0))
            start = [mp.mpf(0)]*3

            def alpha(g, u, v):
                skew = gamma+mp.tanh(g0+g)-mp.tanh(g0)
                tail = nu+mp.log1p(mp.exp(v0+v))-mp.log1p(mp.exp(v0))
                gap = nu-delta+tail/(1+mp.exp(u0+u))-nu*q0
                power = tail-gap
                moment = ((1-skew)**power+(1+skew)**power)/2
                moment *= (tail-2)**(power/2)*mp.gamma((power+1)/2)
                moment *= mp.gamma((tail-power)/2)/(mp.sqrt(mp.pi)*mp.gamma(tail/2))
                return 1/moment

            value = alpha(*start)
            expected_g, expected_h = [], []
            for i in range(3):
                order = [int(j == i) for j in range(3)]
                expected_g.append(float(mp.diff(alpha, start, order)/value))
                for j in range(3):
                    order = [int(k == i)+int(k == j) for k in range(3)]
                    expected_h.append(float(mp.diff(alpha, start, order)/value))
        np.testing.assert_allclose(row["gradient"], expected_g, rtol=3e-11, atol=3e-12)
        np.testing.assert_allclose(row["hessian"], expected_h, rtol=3e-11, atol=3e-12)
        gamma0, delta0, nu0 = row["theta"]
        g0, u0, v0 = row["coordinates"]
        q0 = float(q0)

        def chart_log_moment(displacement):
            # Differentiate the gap itself, not a subtraction of coupled nu/delta.
            g = gamma0+(jnp.tanh(g0+displacement[0])-jnp.tanh(g0))
            dn = jax.nn.softplus(v0+displacement[2])-jax.nn.softplus(v0)
            n = nu0+dn
            log_ratio = jax.nn.softplus(u0)-jax.nn.softplus(u0+displacement[1])
            gap = (nu0-delta0)+q0*dn+n*q0*jnp.expm1(log_ratio)
            log_gap = jnp.log(gap)
            d = n-gap
            leverage = jnp.logaddexp(d*jnp.log1p(-g),d*jnp.log1p(g))-jnp.log(2.)
            return leverage+d/2*jnp.log(n-2)+jsp.special.gammaln((d+1)/2) \
                +jsp.special.gammaln(1+gap/2)-log_gap+jnp.log(2.) \
                -.5*jnp.log(jnp.pi)-jsp.special.gammaln(n/2)

        zero = jnp.zeros(3)
        anchor = float(chart_log_moment(zero))
        normalized_alpha = lambda displacement: jnp.exp(anchor-chart_log_moment(displacement))
        np.testing.assert_allclose(jax.grad(normalized_alpha)(zero), expected_g, rtol=3e-11, atol=3e-12)
        np.testing.assert_allclose(np.asarray(jax.hessian(normalized_alpha)(zero)).ravel(),
                                   expected_h, rtol=3e-11, atol=3e-12)

    moment, = [row for row in rows if row.get("scaled_log_moment")]
    point = jnp.array([.9, 1200., 300.])

    def log_moment(theta):
        gamma, delta, nu = theta
        leverage = jnp.logaddexp(delta*jnp.log1p(-gamma), delta*jnp.log1p(gamma))-jnp.log(2.)
        log_scale = .5*(jsp.special.gammaln(1/nu)-jsp.special.gammaln(3/nu))
        return leverage+delta*log_scale+jsp.special.gammaln((delta+1)/nu)-jsp.special.gammaln(1/nu)

    anchor = float(log_moment(point))
    normalized = lambda theta: jnp.exp(log_moment(theta)-anchor)
    gradient = np.asarray(jax.grad(normalized)(point))
    hessian = np.asarray(jax.hessian(normalized)(point))
    np.testing.assert_allclose(np.asarray(moment["gradient"])/moment["value"], gradient, rtol=3e-11, atol=1e-12)
    np.testing.assert_allclose(np.asarray(moment["hessian"])/moment["value"],
                               hessian[np.triu_indices(3)], rtol=3e-11, atol=1e-12)
    with mp.workdps(90):
        g, d, n = map(mp.mpf, [.9, 1200., 300.])
        scale = mp.sqrt(mp.gamma(1/n)/mp.gamma(3/n))
        reference = ((1-g)**d+(1+g)**d)/2 * scale**d * mp.gamma((d+1)/n)/mp.gamma(1/n)
        reference *= mp.exp(mp.mpf(moment["log_weight"]))
        assert abs(mp.mpf(moment["value"])/reference-1) < mp.mpf("5e-12")


def build_aparch_moment_backend(compiler, root, directory, frame_flags):
    """Keep no-exception numerical objects separate from the allocation probe."""
    native = root / "scivol/_native"
    includes = [native, root / "scivol/_csrc",
                native / "third_party/boost", native / "third_party/tlfloat"]
    flags = [compiler, "-std=c++20", "-O3", "-ffp-contract=fast", "-funroll-loops",
             "-fno-exceptions", "-fno-rtti", *frame_flags]
    for path in includes:
        flags.extend(["-I", str(path)])
    objects = []
    for name in ("aparch_moment_precision", "gh_math_translation"):
        obj = directory / f"{name}.o"
        subprocess.run([*flags, "-c", str(native / f"{name}.cpp"), "-o", str(obj)],
                       check=True, capture_output=True, text=True, timeout=120)
        objects.append(str(obj))
    return objects


def verify_packed_density_oracles(output):
    pytest.importorskip("jax")
    import numpy as np
    from scivol._devtools.ad_oracle import (
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


def test_nnls_scratch_has_no_stack_or_heap_allocator():
    root = Path(__file__).resolve().parents[1]
    source = (root / "scivol/_native/portable_nnls.cpp").read_text()
    assert "WorkspaceArena" in source
    assert not re.search(
        r"\b(?:alloca|_alloca|__builtin_alloca|malloc|calloc|realloc)\s*\(|\bnew\s+",
        source,
    )


def test_gh_preparation_shares_equations_and_reuses_scalar_bessel_values():
    root = Path(__file__).resolve().parents[1]
    source = (root / "scivol/_csrc/gh_math.c").read_text()
    preparation = source.split("static int gh_shape_cache_init_order(", 1)[1].split(
        "int gh_shape_cache_init(", 1
    )[0]
    assert '#include "../_native/gh_equations.hpp"' in preparation
    assert "kappa" not in preparation
    for name in ("gh_first_order.hpp", "aparch_moment_precision.cpp"):
        consumer = (root / "scivol/_native" / name).read_text()
        recipe = consumer.split("#define SCIVOL_GH_PREPARE_EQUATIONS", 1)[1].split(
            "#undef SCIVOL_GH_PREPARE_EQUATIONS", 1
        )[0]
        assert '#include "gh_equations.hpp"' in recipe
    scalar = source.split("static int scalar_standardized_parameters(", 1)[1].split(
        "static int gh_shape_cache_init_order(", 1
    )[0]
    for intention in ("STANDARDIZE", "NORMALIZER"):
        selector = f"SCIVOL_GH_SCALAR_{intention}_EQUATIONS"
        recipe = scalar.split(f"#define {selector}", 1)[1].split(
            f"#undef {selector}", 1
        )[0]
        assert recipe.count('#include "../_native/gh_equations.hpp"') == 1
    assert scalar.count("if (!scalar_standardized_parameters(") == 1
    assert "log_bessel_k_scalar(" not in scalar
    equations = (root / "scivol/_native/gh_equations.hpp").read_text()
    preparation_equations = equations.split(
        "#elif defined(SCIVOL_GH_OBSERVATION_EQUATIONS)", 1
    )[0]
    assert "#define SCIVOL_GH_LOG_BESSEL(a,b) log_bessel_k_scalar(a,b)" in preparation_equations
    bessel_k1 = (
        "SCIVOL_GH_LOG_BESSEL(\n"
        "    SCIVOL_GH_ADD(SCIVOL_GH_FIELD(lambda), SCIVOL_GH_ONE), SCIVOL_GH_FIELD(zeta))"
    )
    assert preparation_equations.count(bessel_k1) == 1
    assert f"const N log_k1 = {bessel_k1};" in preparation_equations
    assert "SCIVOL_GH_SUB(log_k1, log_k0)" in preparation_equations
    assert "SCIVOL_GH_SUB(SCIVOL_GH_LOG_K2, log_k1)" in preparation_equations


def test_gh_observation_loops_do_not_rebuild_point_normalizers():
    root = Path(__file__).resolve().parents[1]
    first = (root / "scivol/_native/gh_first_order.hpp").read_text()
    observations = first.split("variance_gamma_logpdf(", 1)[1]
    assert "ad::lgamma(" not in observations
    assert "ad::log(cache.alpha)" not in observations
    assert "ad::log(cache.vg_argument_scale)" not in observations
    equations = (root / "scivol/_native/gh_equations.hpp").read_text()
    observations = equations.split("#elif defined(SCIVOL_GH_OBSERVATION_EQUATIONS)", 1)[1]
    assert "ad::lgamma(" not in observations
    assert "ad::log(cache.alpha)" not in observations
    assert "log_bessel_k<" not in observations
    equations = (root / "scivol/_native/vg_equations.hpp").read_text()
    observations = equations.split("#elif defined(SCIVOL_VG_OBSERVATION_EQUATIONS)", 1)[1]
    assert "ad::lgamma(" not in observations
    assert "ad::log(cache.vg_argument_scale)" not in observations
    scalar = (root / "scivol/_csrc/gh_math.c").read_text()
    observations = scalar.split("static gh_jet_t vg_logpdf_jet(", 1)[1].split(
        "int gh_abs_moment_derivatives(", 1
    )[0]
    assert "lgamma(" not in observations
    assert "jet_log(cache->alpha)" not in observations
    assert "log(cache->alpha" not in observations
    assert "log(cache->argument_scale)" not in observations
    assert "vg_standardized_logpdf_scalar(" not in observations
