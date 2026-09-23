"""Regression for gamma-ratio cancellation near the Gaussian limit."""

from pathlib import Path
import shutil
import subprocess

import numpy as np
import pytest


def test_student_normalizer_high_precision(tmp_path):
    mp = pytest.importorskip("mpmath")
    compiler = shutil.which("clang++") or shutil.which("g++")
    if compiler is None:
        pytest.skip("A C++20 compiler is required.")
    root = Path(__file__).resolve().parents[1]
    executable = tmp_path / "normalizer"
    subprocess.run(
        [
            compiler,
            "-std=c++20",
            "-O3",
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
    points = [2.001, 7.0, 127.999999, 128.0, 128.000001, 256.0, 1e4, 1e6, 1e10, 1e15]
    output = subprocess.run(
        [str(executable), *map(str, points)],
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
