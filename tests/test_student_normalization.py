"""Regression for gamma-ratio cancellation near the Gaussian limit."""

from pathlib import Path
import shutil
import subprocess

import numpy as np
import pytest


@pytest.mark.parametrize("nu", [7.,128.,1e10,1e15])
def test_legacy_student_density_reuses_stable_normalization_and_curvature(nu):
    from scivol import _core
    mp=pytest.importorskip("mpmath")
    variance=np.array([.5,1.,2.,3.,.2])
    square=np.array([.1,.5,1.,4.,10.])
    with mp.workdps(100):
        def nll(degrees):
            c=mp.loggamma((degrees+1)/2)-mp.loggamma(degrees/2)-mp.log(mp.pi*(degrees-2))/2
            return sum(mp.log(float(h))/2-c+(degrees+1)*mp.log1p(mp.mpf(float(q))/(degrees-2))/2
                       for h,q in zip(variance,square))
        expected=float(nll(mp.mpf(nu)))
        curvature=float(mp.diff(nll,mp.mpf(nu),2))
    actual=-_core._studentt_ll(variance.ctypes.data,square.ctypes.data,len(square),nu)
    hessian=_core._studentt_shape_hessian(square.ctypes.data,len(square),nu)
    assert actual==pytest.approx(expected,abs=3e-13)
    np.testing.assert_allclose(hessian,curvature,rtol=3e-11 if nu<128 else 3e-14,atol=0)


@pytest.fixture(scope="module")
def student_executable(tmp_path_factory):
    compiler = shutil.which("clang++") or shutil.which("g++")
    if compiler is None:
        pytest.skip("A C++20 compiler is required.")
    root = Path(__file__).resolve().parents[1]
    executable = tmp_path_factory.mktemp("student_normalizer") / "normalizer"
    frame_flags = (["-Wframe-larger-than=16384", "-Werror=frame-larger-than"]
                   if "clang" in Path(compiler).name else ["-Werror=frame-larger-than=16384"])
    subprocess.run(
        [
            compiler,
            "-std=c++20",
            "-O3",
            "-ffp-contract=fast",
            "-funroll-loops",
            *frame_flags,
            "-I",
            str(root / "scivol/_native"),
            "-I",
            str(root / "scivol/_csrc"),
            str(root / "tests/native_student_normalization.cpp"),
            "-o",
            str(executable),
        ],
        check=True,
        capture_output=True,
        timeout=60,
    )
    return executable


def test_student_normalizer_high_precision(student_executable):
    mp = pytest.importorskip("mpmath")
    points = [2.001, 7.0, 127.999999, 128.0, 128.000001, 256.0, 1e4, 1e6, 1e10, 1e15]
    output = subprocess.run(
        [str(student_executable), *map(str, points)],
        check=True,
        capture_output=True,
        text=True,
        timeout=10,
    ).stdout
    actual = np.array([list(map(float, line.split())) for line in output.splitlines()])
    with mp.workdps(80):

        def reference(nu):
            return (
                mp.loggamma((nu + 1) / 2)
                - mp.loggamma(nu / 2)
                - mp.log(mp.pi * (nu - 2)) / 2
            )

        for point, row in zip(points, actual):
            nu = mp.mpf(point)
            expected = [
                float(reference(nu)),
                float(mp.diff(reference, nu)),
                float(mp.diff(reference, nu, 2)),
            ]
            np.testing.assert_allclose(row, expected, rtol=2e-10, atol=3e-13)
            if point >= 128:
                np.testing.assert_allclose(row[1:], expected[1:], rtol=2e-14, atol=0)


@pytest.mark.parametrize("variance", [1e200, 1e308])
@pytest.mark.parametrize("sign", [-1., 1.])
def test_student_finite_local_jet_avoids_raw_denominator_overflow(student_executable, variance, sign):
    mp = pytest.importorskip("mpmath")
    residual = sign*np.sqrt(variance)
    output = subprocess.run([str(student_executable), "--student-range", str(variance),
                             str(residual), "8"], check=True, capture_output=True,
                            text=True, timeout=10).stdout
    actual = np.fromstring(output, sep=" ")
    with mp.workdps(100):
        root, base = mp.sqrt(mp.mpf(variance)), mp.mpf(variance)
        def objective(e_unit, h_unit, nu):
            e, h = e_unit*root, h_unit*base
            return (mp.log(h)/2-mp.loggamma((nu+1)/2)+mp.loggamma(nu/2)
                    +mp.log(mp.pi*(nu-2))/2+(nu+1)*mp.log1p(e*e/(h*(nu-2)))/2)
        point = (mp.mpf(residual)/root, mp.mpf(1), mp.mpf(8))
        scales = (root, base, mp.mpf(1))
        expected = [objective(*point)]
        expected += [mp.diff(objective, point, tuple(int(i==j) for i in range(3)))/scales[j]
                     for j in range(3)]
        for i, j in [(0,0), (0,1), (0,2), (1,1), (1,2), (2,2)]:
            expected.append(mp.diff(objective, point, tuple(int(k==i)+int(k==j) for k in range(3)))
                            /scales[i]/scales[j])
    assert np.isfinite(actual).all()
    np.testing.assert_allclose(actual, list(map(float, expected)), rtol=3e-9, atol=0)


def test_student_absolute_moment_high_precision(student_executable):
    mp = pytest.importorskip("mpmath")
    points = [np.nextafter(2., np.inf), 2.001, 3., 7., 31., 127.999999,
              128., 128.000001, 256., 1e4, 1e6, 1e10, 1e15]
    output = subprocess.run(
        [str(student_executable), "--moment", *map(str, points)],
        check=True, capture_output=True, text=True, timeout=10,
    ).stdout
    rows = np.array([list(map(float, line.split())) for line in output.splitlines()])
    with mp.workdps(100):
        for point, row in zip(points, rows, strict=True):
            nu = mp.mpf(float(point))
            value = mp.sqrt((nu-2)/mp.pi)*mp.exp(mp.loggamma((nu-1)/2)-mp.loggamma(nu/2))
            slope = (mp.digamma((nu-1)/2)-mp.digamma(nu/2)+1/(nu-2))/2
            curvature = (mp.polygamma(1, (nu-1)/2)-mp.polygamma(1, nu/2))/4-1/(2*(nu-2)**2)
            expected = np.array(list(map(float, (value, value*slope, value*(curvature+slope*slope)))))
            jets = row[:9].reshape(3, 3)
            for order in range(3):
                np.testing.assert_allclose(jets[order, :order+1], jets[2, :order+1],
                                           rtol=4*np.finfo(float).eps, atol=0)
                assert np.all(jets[order, order+1:] == 0)
            np.testing.assert_allclose(row[9:], jets[2, [0, 2]], rtol=4*np.finfo(float).eps, atol=0)
            np.testing.assert_allclose(jets[2, 0], expected[0], rtol=8*np.finfo(float).eps, atol=0)
            np.testing.assert_allclose(jets[2, 1:], expected[1:],
                                       rtol=2e-14 if point>=128 else 2e-9, atol=0)


def test_truncated_normal_tail_keeps_small_positive_moments(student_executable):
    mp = pytest.importorskip("mpmath")
    points = [-10., -1., 0., 2., np.nextafter(4., 0.), 4., np.nextafter(4., np.inf),
              8., 16., 32., 38., 40.]
    output = subprocess.run([str(student_executable), "--normal-tail", *map(str, points)],
                            check=True, capture_output=True, text=True, timeout=10).stdout
    rows = np.array([np.fromstring(row, sep=" ") for row in output.splitlines()])
    with mp.workdps(100):
        for point, row in zip(points, rows, strict=True):
            t = mp.mpf(float(point))
            phi = mp.exp(-t*t/2)/mp.sqrt(2*mp.pi)
            cdf = mp.erfc(t/mp.sqrt(2))/2
            expected = list(map(float, ((1+t*t)*cdf-t*phi, 2*(t*cdf-phi), 2*cdf)))
            np.testing.assert_allclose(row, expected, rtol=6e-13, atol=2*np.nextafter(0., 1.))


@pytest.mark.parametrize("power", [.2, .7, 1., 1.7, 2., 4., 20.])
def test_student_power_moment_gaussian_limit(student_executable, power):
    mp = pytest.importorskip("mpmath")
    switch = 64 + power
    ratio_switch = max(switch, 16 * power)
    points = [np.nextafter(switch, 0.), switch, np.nextafter(switch, np.inf),
              np.nextafter(ratio_switch, 0.), ratio_switch,
              np.nextafter(ratio_switch, np.inf), 128., 1e4, 1e8, 1e15, 1e100]
    output = subprocess.run(
        [str(student_executable), "--power", str(power), *map(str, points)],
        check=True, capture_output=True, text=True, timeout=10,
    ).stdout
    rows = np.array([list(map(float, line.split())) for line in output.splitlines()])
    with mp.workdps(400):
        d = mp.mpf(power)
        for point, row in zip(points, rows, strict=True):
            nu = mp.mpf(float(point))
            a, b, c = (d + 1) / 2, (nu - d) / 2, nu / 2
            value = mp.exp(d / 2 * mp.log(nu - 2) + mp.loggamma(a)
                           + mp.loggamma(b) - mp.loggamma(c) - mp.log(mp.pi) / 2)
            slope = [mp.log(nu - 2) / 2 + (mp.digamma(a) - mp.digamma(b)) / 2,
                     d / (2 * (nu - 2)) + (mp.digamma(b) - mp.digamma(c)) / 2]
            curvature = [(mp.polygamma(1, a) + mp.polygamma(1, b)) / 4,
                         1 / (2 * (nu - 2)) - mp.polygamma(1, b) / 4,
                         -d / (2 * (nu - 2)**2)
                         + (mp.polygamma(1, b) - mp.polygamma(1, c)) / 4]
            expected = np.array(list(map(float, [value, *(value * g for g in slope),
                *(value * (curvature[k] + slope[i] * slope[j])
                  for k, (i, j) in enumerate([(0, 0), (0, 1), (1, 1)]))])))
            jets = row.reshape(3, 6)
            scale = np.array([1., 1., point**2, 1., point**2, point**3])
            for order, length in enumerate([1, 3, 6]):
                np.testing.assert_allclose(jets[order, :length] * scale[:length],
                                           jets[2, :length] * scale[:length],
                                           rtol=4e-14, atol=4e-14)
                assert np.all(jets[order, length:] == 0)
            # Scale tiny tail-shape derivatives before comparing. A broad absolute
            # tolerance on the raw gradient would accept the old erroneous zeros.
            np.testing.assert_allclose(jets[2] * scale, expected * scale,
                                       rtol=2e-10, atol=2e-10)
            np.testing.assert_allclose(jets[2, 0], expected[0], rtol=5e-13, atol=0)


def test_joint_student_moment_integration_matches_ad(student_executable):
    pytest.importorskip("jax")
    from scivol._devtools.joint_mean_oracle import joint_value_grad_hess

    points = (7., 128., 1e4)
    output = subprocess.run(
        [str(student_executable), "--joint", *map(str, points)],
        check=True, capture_output=True, text=True, timeout=10,
    ).stdout
    rows = np.array([list(map(float, line.split())) for line in output.splitlines()])
    rows = rows.reshape(len(points), 2, 7, 158)
    y = np.array([0.5,-0.8,1.2,-0.3,0.7,-1.1,0.2,0.9])
    x = np.column_stack((np.ones(8), [0.2,-0.3,0.5,-0.1,0.4,-0.5,0.3,-0.2]))
    for nu, paths in zip(points, rows, strict=True):
        theta = np.array([0.1,-0.04,-0.1,0.08,0.04,0.02,-0.05,0.01,0.03,0.3,0.2,nu])
        expected = joint_value_grad_hess(theta, y, x, 1.1, 3, 2, 2, 1)
        for masks in paths:
            full = masks[-1]
            np.testing.assert_allclose(full[1], expected[0], rtol=0, atol=1e-9)
            np.testing.assert_allclose(full[2:14], expected[1], rtol=2e-6, atol=3e-6)
            np.testing.assert_allclose(full[14:].reshape(12,12), expected[2], rtol=2e-6, atol=3e-6)
            for mask, row in enumerate(masks, 1):
                assert row[0] == mask
                for bit, section in ((1,slice(1,2)), (2,slice(2,14)), (4,slice(14,None))):
                    if mask & bit:
                        np.testing.assert_allclose(row[section], full[section], rtol=2e-12, atol=2e-12)
                    else:
                        assert np.isnan(row[section]).all()
        np.testing.assert_allclose(paths[0], paths[1], rtol=2e-12, atol=2e-12, equal_nan=True)


@pytest.mark.parametrize("skew", [False, True])
def test_joint_gaussian_limit_shape_derivatives(student_executable, skew):
    pytest.importorskip("jax")
    from scivol._devtools.joint_mean_oracle import joint_value_grad_hess

    points = (128., 1e4, 19958224.45994093, 1e10)
    width = 12 + int(skew)
    output = subprocess.run(
        [str(student_executable), "--skew-joint" if skew else "--joint", *map(str, points)],
        check=True, capture_output=True, text=True, timeout=10,
    ).stdout
    rows = np.array([np.fromstring(line, sep=" ") for line in output.splitlines()])
    rows = rows.reshape(len(points), 2, 7, 2+width+width*width)
    y = np.array([0.5,-0.8,1.2,-0.3,0.7,-1.1,0.2,0.9])
    x = np.column_stack((np.ones(8), [0.2,-0.3,0.5,-0.1,0.4,-0.5,0.3,-0.2]))
    for nu, paths in zip(points, rows, strict=True):
        theta = np.array([0.1,-0.04,-0.1,0.08,0.04,0.02,-0.05,0.01,0.03,0.3,0.2,nu]
                         + ([0.08728423370553595] if skew else []))
        f, g, h = joint_value_grad_hess(theta, y, x, 1.1, 3, 2, 2, 2 if skew else 1)
        for masks in paths:
            full = masks[-1]
            np.testing.assert_allclose(full[1], f, rtol=0, atol=2e-11)
            # Relative checks on the entire joint jet include O(nu^-2) gradients,
            # O(nu^-3) curvature and every mixed block. No absolute escape hatch.
            np.testing.assert_allclose(full[2:2+width], g, rtol=2e-9, atol=0)
            np.testing.assert_allclose(full[2+width:].reshape(width,width), h, rtol=2e-9, atol=0)
            for mask, row in enumerate(masks, 1):
                for bit, section in ((1,slice(1,2)), (2,slice(2,2+width)), (4,slice(2+width,None))):
                    if mask & bit:
                        np.testing.assert_allclose(row[section], full[section], rtol=2e-12, atol=0)
                    else:
                        assert np.isnan(row[section]).all()
        np.testing.assert_allclose(paths[0], paths[1], rtol=2e-12, atol=0, equal_nan=True)


def test_skew_density_gaussian_limit_local_ad(student_executable):
    jax = pytest.importorskip("jax")
    from scivol._devtools.ad_oracle import _jax_modules
    from scivol._devtools.joint_mean_oracle import _density_nll
    _, jnp, jsp = _jax_modules()

    def objective(point, base):
        residual, log_h, log_nu, lam = point
        nu = 2+(base-2)*jnp.exp(log_nu)
        z = jnp.array([residual*jnp.exp(-log_h/2)])
        return log_h/2+_density_nll(jnp,jsp,z,jnp.array([nu,lam]),2,None)

    fg = jax.jit(jax.value_and_grad(objective))
    hessian = jax.jit(jax.hessian(objective))
    for nu in (128.,1e4,19958224.45994093,1e10,1e15):
        for lam in (-0.6,0.,0.08728423370553595):
            for z in (-3.,-.31,1.4):
                rows = subprocess.run([str(student_executable),"--skew",str(nu),str(lam),str(z)],
                                      check=True,capture_output=True,text=True,timeout=10).stdout.splitlines()
                components, full = (np.fromstring(rows[i],sep=" ") for i in (0,2))
                scale = np.array([1,1,nu-2,1])
                gradient = full[:4]*scale
                curvature = full[4:].reshape(4,4)*np.outer(scale,scale)
                curvature[1,1] += gradient[1]
                curvature[2,2] += gradient[2]
                f,g = fg(jnp.array([z,0.,0.,lam]),nu)
                h = hessian(jnp.array([z,0.,0.,lam]),nu)
                np.testing.assert_allclose(components[0],f,rtol=2e-14,atol=2e-11)
                np.testing.assert_allclose(gradient,g,rtol=2e-10,atol=0)
                np.testing.assert_allclose(curvature,h,rtol=2e-10,atol=0)


def test_skew_gaussian_limit_against_multiprecision(student_executable):
    mp = pytest.importorskip("mpmath")

    def reference(nu, lam, z):
        c = mp.loggamma((nu+1)/2)-mp.loggamma(nu/2)-mp.log(mp.pi*(nu-2))/2
        half = mp.exp(c)*(nu-2)/(nu-1)
        a, left, right = 4*lam*half, 1-lam, 1+lam
        b = mp.sqrt(1+3*lam*lam-a*a)
        u = b*z+a
        density = -c-mp.log(b)+(nu+1)*mp.log1p((u/(1+mp.sign(u)*lam))**2/(nu-2))/2
        threshold = a/right
        # Independent incomplete-beta CDF, not the native quadrature plan.
        cdf = mp.betainc(mp.mpf(".5"),nu/2,0,threshold**2/(nu-2+threshold**2),regularized=True)/2
        partial = -half*(1+threshold**2/(nu-2))**(-(nu-1)/2)
        moment = (a*left/2+left*left*half+a*right*(2*cdf-mp.mpf(".5"))
                  +right*right*(-half-2*partial))/b
        return density, moment

    for nu in (128.,1e4,19958224.45994093,1e10,1e15):
        for lam in (0.08728423370553595,0.6):
            z = -0.31
            output = subprocess.run([str(student_executable),"--skew",str(nu),str(lam),str(z)],
                                    check=True,capture_output=True,text=True,timeout=10).stdout
            actual = np.fromstring(output.splitlines()[0],sep=" ")
            with mp.workdps(80):
                n,l,v = map(mp.mpf,(nu,lam,z))
                density = lambda n: reference(n,l,v)[0]
                moment = lambda n: reference(n,l,v)[1]
                expected = [density(n),mp.diff(density,n),mp.diff(density,n,2),
                            moment(n),mp.diff(moment,n),mp.diff(moment,n,2)]
            np.testing.assert_allclose(actual,list(map(float,expected)),rtol=2e-10,atol=0)


@pytest.mark.parametrize("nu", [128., 1e7, 1e15])
@pytest.mark.parametrize("square", [6., np.nextafter(6., 0.), np.nextafter(6., np.inf)])
def test_student_tail_secondary_cancellation(student_executable, nu, square):
    mp = pytest.importorskip("mpmath")
    output = subprocess.run([str(student_executable), "--tail-cancellation", str(nu), str(square)],
                            check=True, capture_output=True, text=True, timeout=10).stdout
    with mp.workdps(100):
        y, n = mp.mpf(square), mp.mpf(nu)
        objective = lambda degrees: (degrees+1)/2*mp.log1p(y/(degrees-2))
        f, g, h = float(objective(n)), float(mp.diff(objective,n)), float(mp.diff(objective,n,2))
    np.testing.assert_allclose(np.fromstring(output, sep=" "), [f,g,h,g,h], rtol=3e-14, atol=0)


@pytest.mark.parametrize("family", ["GARCH", "GJRGARCH", "EGARCH"])
@pytest.mark.parametrize("skew", [False, True])
@pytest.mark.parametrize("nu", [128.0, 5.82483924e7, 1e10])
def test_joint_legacy_student_tail_likelihood_high_precision(family, skew, nu):
    mp = pytest.importorskip("mpmath")
    import scivol

    rng = np.random.default_rng(2931)
    data = rng.normal(size=31)
    x = rng.normal(size=(len(data), 2))
    mean = scivol.HARX([1, 3])
    mean.set_n_exog(2)
    volatility = getattr(scivol, family)(3, 2)
    density = scivol.SkewT() if skew else scivol.StudentT()
    if family == "EGARCH":
        vol = [-0.1, 0.03, 0.02, 0.01, -0.02, 0.01, 0.02, 0.4, 0.3]
    elif family == "GJRGARCH":
        vol = [0.1, 0.03, 0.02, 0.01, 0.02, 0.01, 0.02, 0.4, 0.3]
    else:
        vol = [0.1, 0.03, 0.02, 0.01, 0.4, 0.3]
    lam = -0.038898954 if skew else 0.0
    theta = np.r_[np.full(mean.n_params, 0.04), vol, [nu, lam] if skew else [nu]]
    state = (mean + volatility + density).filter(data, theta, x=x)
    with mp.workdps(80):
        v, l = mp.mpf(nu), mp.mpf(lam)
        c = mp.loggamma((v + 1) / 2) - mp.loggamma(v / 2) - mp.log(mp.pi * (v - 2)) / 2
        a = 4 * l * mp.exp(c) * (v - 2) / (v - 1)
        b = mp.sqrt(1 + 3 * l * l - a * a)
        total = mp.mpf(0)
        for residual, variance in zip(state.residuals, state.sigma2):
            h = mp.mpf(float(variance))
            u = b * mp.mpf(float(residual)) / mp.sqrt(h) + a
            adjusted = u / (1 + l * (1 if u >= 0 else -1))
            total += (
                c
                + mp.log(b)
                - mp.log(h) / 2
                - (v + 1) / 2 * mp.log1p(adjusted**2 / (v - 2))
            )
    assert state.log_likelihood == pytest.approx(float(total), rel=2e-13, abs=2e-12)
