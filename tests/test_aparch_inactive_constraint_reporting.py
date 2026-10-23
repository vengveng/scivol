"""Inactive APARCH constraints need values, not unavailable derivatives."""

from types import SimpleNamespace

import numpy as np
import pytest

from scivol import _native_fit as native


@pytest.fixture(params=[(p, mean, density) for p in (1, 3) for mean in (0, 3)
                       for density in (6, 7)])
def problem(request):
    p, mean, density = request.param
    theta = np.r_[np.full(mean, 0.1), 0.1, np.full(p, 0.08 / p),
                  np.full(p, 0.15), 0.3, 0.3, 1.7, 0.1, 3.0]
    model = SimpleNamespace(n_mean=mean, p_vol=p, q_vol=2,
                            n_params=theta.size, density_code=density)
    moments = np.linspace(0.8, 1.1, p)
    derivatives = np.arange(1, 4*p+1).reshape(p, 4) / 100
    gradient = np.linspace(-0.2, 0.2, theta.size)
    scales = np.linspace(0.8, 1.2, theta.size)
    return model, theta, moments, derivatives, gradient, scales


@pytest.mark.parametrize("active", (False, True))
def test_request_schedule_preserves_exact_active_normal(problem, monkeypatch, active):
    model, theta, moments, derivatives, gradient, scales = problem
    alpha = model.n_mean + 1
    gamma, beta = alpha + model.p_vol, alpha + 2*model.p_vol
    power, shape = beta + model.q_vol, beta + model.q_vol + 1
    calls = []
    if active:
        theta[beta+1] = 1-3e-6-theta[alpha:gamma] @ moments-theta[beta]

    def moment_request(model, point, *, gradient):
        calls.append(gradient)
        return moments, derivatives if gradient else None

    monkeypatch.setattr(native, "_aparch_moment_jets", moment_request)
    saved = theta.copy()
    actual = native._aparch_kkt_residual(model, theta, gradient, scales, 1e-10)
    # The assessment interface receives G already scaled into optimizer units.
    residual = gradient.copy()
    if active:
        normal = np.zeros(theta.size)
        normal[alpha:gamma] = -moments
        normal[gamma:beta] = -theta[alpha:gamma]*derivatives[:, 0]
        normal[beta:power] = -1
        normal[power] = -theta[alpha:gamma] @ derivatives[:, 1]
        normal[shape:] = -theta[alpha:gamma] @ derivatives[:, 2:]
        normal *= scales
        multiplier = max(float(normal @ residual / (normal @ normal)), 0.0)
        residual -= normal*multiplier
    assert actual == pytest.approx(np.max(np.abs(residual)), rel=1e-13, abs=1e-14)
    assert calls == ([False, True] if active else [False])
    np.testing.assert_array_equal(theta, saved)


def test_unavailable_active_derivatives_are_not_silently_zeroed(problem, monkeypatch):
    model, theta, moments, _, gradient, scales = problem
    alpha = model.n_mean+1
    beta = alpha+2*model.p_vol
    theta[beta+1] = 1-3e-6-theta[alpha:alpha+model.p_vol] @ moments-theta[beta]

    def moment_request(model, point, *, gradient):
        if gradient:
            raise RuntimeError("unavailable active constraint derivative")
        return moments, None

    monkeypatch.setattr(native, "_aparch_moment_jets", moment_request)
    with pytest.raises(RuntimeError, match="unavailable active constraint derivative"):
        native._aparch_kkt_residual(model, theta, gradient, scales, 1e-10)


def test_unavailable_inactive_derivatives_do_not_reject_a_finite_score(problem, monkeypatch):
    model, theta, moments, _, gradient, scales = problem

    def moment_request(model, point, *, gradient):
        if gradient:
            raise AssertionError("Inactive constraint requested a derivative")
        return moments, None

    monkeypatch.setattr(native, "_aparch_moment_jets", moment_request)
    residual = native._aparch_kkt_residual(model, theta, gradient, scales, 1e-10)
    assert np.isfinite(residual) and residual > 1e-5
