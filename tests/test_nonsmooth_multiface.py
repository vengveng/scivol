"""Several simultaneous residual kinks: evidence, refinement and reporting.

Linear-mean EGARCH likelihoods are piecewise smooth in the mean coefficients,
with kinks on the hyperplanes e_t(b) = 0. The first-order conditions at several
active kinks are tangential stationarity plus normal growth (Griewank and
Walther 2016). These tests keep that evidence honest and uncertified.
"""

from types import SimpleNamespace

import numpy as np
import pytest

from benchmark_native_joint import comparison_sample
from scivol import HARX, EGARCH, SkewT
import scivol._joint_mean_fit as joint
from scivol._joint_mean_fit import JointMeanModel, fit_joint_mean_spec, solve_joint
from scivol._nonsmooth_fit import bracket_residual_faces


def separable_model(kappas, slopes, *, dependent=False, crossing=False):
    """F = sum kappa_i |e_i| + slopes'b + theta_2^2 / 2 with e_i = y_i - b_i."""
    observations = [0.0, 0.0]
    design = [[1.0, 0.0], [0.0, 1.0]]
    if dependent:
        observations.append(0.0)
        design.append([2.0, 0.0])
    if crossing:
        observations.append(0.0)
        design.append([1.0, 2.0])  # crosses zero between the neighbor patterns
    observations.append(5.0)  # last row: no successor shock
    design.append([1.0, 1.0])
    observations, design = np.array(observations), np.array(design)

    def evaluate(theta, mask, *, paths=False):
        residuals = observations - design @ theta[:2]
        value = sum(k*abs(e) for k, e in zip(kappas, residuals[:2])) + slopes @ theta[:2] + theta[2]**2/2
        gradient = None
        if mask & 2:
            gradient = np.r_[np.asarray(slopes, float)
                             - design[:2].T @ (np.asarray(kappas)*np.sign(residuals[:2])), theta[2]]
        return value, gradient, None, residuals, np.ones(len(observations))

    def assess(theta, gradient):
        return True, np.inf if gradient is None else float(np.max(abs(gradient))), 0.0

    return SimpleNamespace(family=2, p_vol=1, data=observations, features=design, columns=2,
                           size=3, units=np.ones(3), evaluate=evaluate, assess=assess)


def test_normal_growth_and_tangential_residual_are_measured_separately():
    model = separable_model([2.0, 3.0], [1.0, -1.5])
    theta = np.array([1e-9, -2e-9, 0.2])
    evidence = bracket_residual_faces(model, theta, [0, 1])
    np.testing.assert_array_equal(theta, [1e-9, -2e-9, 0.2])
    # Kink sizes are reported per observation, in the production gradient units.
    np.testing.assert_allclose(np.asarray(evidence["kappas"]) * len(model.data), [2.0, 3.0], rtol=1e-12)
    np.testing.assert_allclose(evidence["weights"], [0.5, -0.5], rtol=1e-12)
    assert evidence["normal_minimum"] and evidence["face_set_exact"]
    assert evidence["mixed_kkt"] == pytest.approx(0.2)  # the smooth theta_2 direction remains
    assert not evidence["certified"] and not any(evidence["saturated"])


def test_concave_kinks_and_saturated_weights_cannot_support_a_minimum():
    concave = bracket_residual_faces(separable_model([2.0, -3.0], [1.0, 0.0]),
                                     np.array([0.0, 0.0, 0.0]), [0, 1])
    assert not concave["normal_minimum"] and concave["mixed_kkt"] == np.inf
    steep = bracket_residual_faces(separable_model([2.0, 3.0], [5.0, 0.0]),
                                   np.array([0.0, 0.0, 0.0]), [0, 1])
    assert steep["saturated"] == [True, False]
    assert steep["mixed_kkt"] == pytest.approx(3.0)


@pytest.mark.parametrize("release_reason", ["concave", "saturated"])
@pytest.mark.parametrize("gradient_state", ["retained", "missing", "nonfinite", "wrong_shape"])
def test_required_face_release_precedes_nearby_face_entry(monkeypatch, release_reason, gradient_state):
    from scipy.optimize import OptimizeResult
    import scivol._nonsmooth_fit as nonsmooth

    model = SimpleNamespace(code=0, columns=3, size=4, units=np.ones(4),
        features=np.eye(3), data=np.zeros(3), request_counts=[0]*8)
    model.assess = lambda theta, gradient: (True, 1.0, 0.0)

    def evaluate(theta, mask, *, paths=False):
        model.request_counts[mask] += 1
        return 1.0, np.zeros(4)

    def candidate():
        gradient = {"retained": np.zeros(4), "missing": None,
                    "nonfinite": np.full(4, np.nan), "wrong_shape": np.zeros(5)}[gradient_state]
        return OptimizeResult(x=np.zeros(4), fun=1.0, nit=1, status=9, success=False,
            curvature_prepared=True, affine_equality=None, kernel_dispatch=2,
            stationarity_bound=1.0, jac=gradient)

    visited = []

    def solve(model, point, *, affine, **options):
        visited.append(tuple(np.argmax(affine[0], axis=1)))
        return candidate()

    def evidence(model, point, rows, **options):
        count = len(rows)
        return dict(feasible=True, face_set_exact=True,
            normal_minimum=release_reason!="concave", mixed_kkt=1.0, tangent_jump=0.0,
            maximum_value_discrepancy=0.0, backward_error=0.0,
            smooth_gradient=np.zeros(4), face_gradients=np.zeros((count,4)),
            kappas=np.full(count, -1.0 if release_reason=="concave" else 1.0),
            weights=np.ones(count), saturated=[release_reason=="saturated"]*count)

    model.evaluate = evaluate
    monkeypatch.setattr(joint, "_solve_joint_phase", solve)
    monkeypatch.setattr(joint, "_near_faces", lambda *args: [0,1])
    monkeypatch.setattr(nonsmooth, "bracket_residual_faces", evidence)
    phases = []
    initial = candidate()
    result, remaining, stop = joint._refine_terminated_faces(
        model, initial, 20, phases, 1e-12, False, [0,1])
    assert visited == [(0,1),(0,1),(1,),(1,)]
    assert remaining == 16 and sum(phase["iterations"] for phase in phases)==4
    assert all([p["accuracy"] for p in phase["refinement_attempts"]] ==
               [1e-12,np.finfo(float).eps] for phase in phases)
    assert not stop and not result.success
    assert model.request_counts[2] == (0 if gradient_state == "retained" else 2)
    np.testing.assert_array_equal(result.x, initial.x)
    assert result.fun == initial.fun


@pytest.mark.parametrize("multiple", [False, True])
def test_two_radius_evidence_shares_only_its_base_request(monkeypatch, multiple):
    import scivol._nonsmooth_fit as nonsmooth

    model = separable_model([2., 3.], [1., -1.5])
    theta = np.array([1e-9, -2e-9, .2])
    evaluate = model.evaluate
    requests = []

    def recorded(theta, mask, **options):
        requests.append(mask)
        return evaluate(theta, mask, **options)

    model.evaluate = recorded
    name = "bracket_residual_faces" if multiple else "bracket_residual_face"
    bracket = getattr(nonsmooth, name)
    rows = [0, 1] if multiple else [0]
    expected = [bracket(model, theta, rows, radius=r) if multiple
                else bracket(model, theta, row=0, radius=r) for r in (1, 2)]
    separate_requests = requests.copy()
    requests.clear()
    bases = []

    def checked(*args, radius, _base, **kwargs):
        bases.append(_base)
        result = bracket(*args, radius=radius, _base=_base, **kwargs)
        np.testing.assert_equal(result, expected[radius - 1])
        return result

    monkeypatch.setattr(nonsmooth, name, checked)
    evidence = nonsmooth.residual_face_evidence(model, theta, rows, 1e-12, multiple=multiple)
    assert len(bases) == 2 and bases[0] is bases[1]
    assert requests.count(1) == 1 and separate_requests.count(1) == 2
    assert [mask for mask in requests if mask != 1] == [mask for mask in separate_requests if mask != 1]
    assert evidence["nll"] == expected[0]["nll"] and not evidence["certified"]
    np.testing.assert_array_equal(theta, [1e-9, -2e-9, .2])


def test_subnormal_neighbor_steps_are_formed_without_binary64_underflow():
    model = separable_model([2.0, 3.0], [0.0, 0.0])
    model.features[:] *= 0.25
    evidence = bracket_residual_faces(model, np.zeros(3), [0, 1])
    assert evidence["normal_minimum"] and evidence["mixed_kkt"] < 1e-15
    assert evidence["neighborhood_radius"] > 0


def test_neighbor_evidence_rejects_unresolved_jet_residual_signs():
    model = separable_model([2.0, 3.0], [0.0, 0.0])
    original = model.evaluate

    def unresolved(theta, mask, **kwargs):
        value, gradient, hessian, residuals, variance = original(theta, mask, **kwargs)
        residuals[:2] = 0.0
        return value, gradient, hessian, residuals, variance

    model.evaluate = unresolved
    with pytest.raises(ValueError, match="residual signs"):
        bracket_residual_faces(model, np.zeros(3), [0, 1])


def test_dependent_faces_and_extra_crossings_are_rejected_or_reported():
    with pytest.raises(ValueError, match="linearly dependent"):
        bracket_residual_faces(separable_model([2.0, 3.0], [0.0, 0.0], dependent=True),
                               np.zeros(3), [0, 2])
    crossing = bracket_residual_faces(separable_model([2.0, 3.0], [0.0, 0.0], crossing=True),
                                      np.zeros(3), [0, 1])
    assert not crossing["face_set_exact"]
    with pytest.raises(ValueError, match="successor"):
        bracket_residual_faces(separable_model([2.0, 3.0], [0.0, 0.0]), np.zeros(3), [0, 2])
    with pytest.raises(ValueError, match="distinct"):
        bracket_residual_faces(separable_model([2.0, 3.0], [0.0, 0.0]), np.zeros(3), [0, 0])


def persistent_wide(density=SkewT, seed=2785):
    y, x = comparison_sample(seed, True, "persistent")
    return HARX([1, 3]) + EGARCH(3, 2) + density(), y, x


@pytest.mark.parametrize("transformed", [False, True])
@pytest.mark.parametrize("orders", [(3, 2), (4, 3)])
def test_several_affine_rows_are_enforced_by_the_native_phase(transformed, orders):
    spec, y, x = persistent_wide()
    spec = HARX([1, 3]) + EGARCH(*orders) + SkewT()
    model = JointMeanModel(spec, y, x)
    start = model.initialize()
    rows = [10, 20, 30]
    phase = joint._solve_joint_phase(model, start, transformed=transformed, accuracy=1e-12,
        maxiter=40, affine=(model.features[rows], model.data[rows]))
    residuals = model.data[rows] - model.features[rows] @ phase.x[: model.columns]
    assert np.max(abs(residuals)) <= 1e-12 * max(1.0, np.max(abs(model.data)))
    assert phase.affine_equality["violation"] <= 1e-12
    assert len(phase.affine_equality["normals"]) == 3
    for bad in ((model.features[rows], model.data[rows[:2]]),
                (model.features[rows[:1]] * 0.0, model.data[rows[:1]]),
                (1.0, 0.0),
                (np.ones((model.columns + 1, model.columns)), np.zeros(model.columns + 1))):
        with pytest.raises(ValueError, match="affine"):
            joint._solve_joint_phase(model, start, transformed=False, accuracy=1e-12,
                                     maxiter=5, affine=bad)
    for count in (-1, model.columns + 1, 2**32):
        with pytest.raises((ValueError, OverflowError)):
            joint._native_core._linear_joint_sizes(model.descriptor, 3, False, model.domain, count)


def test_multiface_stop_is_quieter_cheaper_and_explicitly_uncertified(capsys):
    spec, y, x = persistent_wide()
    single = JointMeanModel(spec, y, x)
    original = joint._refine_terminated_kinks
    joint._refine_terminated_kinks = joint._refine_terminated_face
    try:
        baseline = solve_joint(single, single.initialize(), transformed=False,
            accuracy=1e-12, maxiter=300, allow_chart_refinement=True)
    finally:
        joint._refine_terminated_kinks = original
    result = fit_joint_mean_spec(spec, y, x=x, maxiter=300)
    info = result.fit_info
    evidence = info.nonsmooth_evidence
    assert evidence["supports_kink_stop"] and not evidence["certified"]
    assert len(evidence["rows"]) >= 2 and evidence["mixed_kkt"] <= info.kkt_tolerance
    assert not result.converged and info.stationarity_status == "uncertified"
    assert info.inference_status == "unavailable" and result.std_errors is None
    assert -result.log_likelihood <= baseline.fun + 1e-9
    requests = sum(info.jet_requests[k] for k in "FGH")
    assert requests < baseline.nfev + baseline.njev + baseline.nhev
    faces = [p for p in info.phase_diagnostics if p.get("algorithm") == "native-slsqp-affine-shock-faces"]
    assert faces and faces[-1]["rows"] == evidence["rows"]
    assert sum(p["iterations"] for p in info.phase_diagnostics) <= 300
    captured = capsys.readouterr()
    assert not captured.out and not captured.err


def test_multiface_neighbor_gradients_match_independent_limiting_ad():
    from scivol._devtools.joint_mean_oracle import joint_value_grad_hess

    spec, y, x = persistent_wide()
    model = JointMeanModel(spec, y, x)
    result = solve_joint(model, model.initialize(), transformed=False, accuracy=1e-12,
                         maxiter=300, allow_chart_refinement=True)
    rows = result.nearby_kink_evidence["rows"]
    assert len(rows) >= 2
    evidence = bracket_residual_faces(model, result.x, rows)
    scale = model.units / len(model.data)
    for gradient, pattern in zip(evidence["gradients"], evidence["patterns"]):
        limits = [(row, int(side)) for row, side in zip(rows, pattern)]
        value, reference, _ = joint_value_grad_hess(result.x, model.data, model.features,
            model.h0, 3, 2, 2, 2, request=3, kink_limit=limits, kink_pivot=evidence["pivots"])
        np.testing.assert_allclose(value, result.fun, rtol=0, atol=2e-9)
        np.testing.assert_allclose(gradient*scale, reference*scale, rtol=2e-7, atol=2e-9)


def test_limiting_oracle_rejects_malformed_face_lists():
    from scivol._devtools.joint_mean_oracle import joint_value_grad_hess

    theta = np.array([0.1, 0.2, -0.1, 0.08, -0.04, 0.7])
    args = (theta, np.array([0.3, 0.6, 0.2]), np.eye(3, 2) + 0.5, 1.0, 1, 1, 2, 0)
    with pytest.raises(ValueError, match="EGARCH"):
        joint_value_grad_hess(*args, request=3, kink_limit=[(0, 1), (1, 1)], kink_pivot=[0])
    with pytest.raises(ValueError, match="nonzero mean feature"):
        joint_value_grad_hess(*args, request=3, kink_limit=[(0, 1), (0, -1)], kink_pivot=[0, 1])
    with pytest.raises(ValueError, match="linearly independent"):
        joint_value_grad_hess(*args[:2], np.ones((3, 2)), *args[3:], request=3,
                              kink_limit=[(0, 1), (1, 1)], kink_pivot=[0, 1])


def test_limiting_oracle_allows_nonsingular_pivots_with_a_zero_diagonal():
    from scivol._devtools.joint_mean_oracle import joint_value_grad_hess

    theta = np.array([0.1, 0.2, -0.1, 0.08, -0.04, 0.7])
    args = (theta, np.array([0.3, 0.6, 0.2]), np.array([[0., 1.], [2., 0.], [0.5, 1.]]),
            1.0, 1, 1, 2, 0)
    left = joint_value_grad_hess(*args, request=3, kink_limit=[(0, 1), (1, -1)], kink_pivot=[0, 1])
    right = joint_value_grad_hess(*args, request=3, kink_limit=[(0, 1), (1, -1)], kink_pivot=[1, 0])
    for actual, expected in zip(left[:2], right[:2]):
        np.testing.assert_array_equal(actual, expected)
