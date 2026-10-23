"""Neighboring-gradient diagnostics must not manufacture convergence."""

import json

import numpy as np
import pytest

from scivol import ARX, EGARCH, Normal, StudentT, SkewT, GeneralizedHyperbolic, VarianceGamma
from scivol._joint_mean_fit import JointMeanModel, assemble_joint_fit, fit_joint_mean_spec, solve_joint
from scivol.result import FitMetadata
from scivol._nonsmooth_fit import neighboring_residual_proposals


class QuadraticRegionModel:
    columns, size = 2, 3
    features = np.array([[1., 0.], [0., 1.], [1., 1.], [2., -1.], [.5, .2], [0., 0.]])
    data = np.array([.12, .25, .6, -.1, .2, .5])
    hessian = np.array([[4., 1., .8], [1., 3., -.5], [.8, -.5, 2.]])

    def __init__(self, units=None, feasible=True):
        self.units = np.ones(3) if units is None else np.array(units)
        self.feasible, self.requests = feasible, []

    def evaluate(self, theta, mask):
        self.requests.append(mask)
        return float(.5*theta@self.hessian@theta), None, self.hessian, None, None

    def assess(self, theta, gradient):
        return self.feasible, 0., 0.


def test_region_reflections_use_joint_curvature_and_are_unit_invariant():
    original, rescaled = QuadraticRegionModel(), QuadraticRegionModel([.125, 4., 32.])
    point = np.array([.1, .2, .3])
    before = point.copy()
    left = neighboring_residual_proposals(original, point)
    right = neighboring_residual_proposals(rescaled, point)
    np.testing.assert_array_equal(point, before)
    assert original.requests == rescaled.requests == [4, 1, 1, 1, 1]
    assert [p["row"] for p in left["proposals"]] == [p["row"] for p in right["proposals"]]
    for a, b in zip(left["proposals"], right["proposals"]):
        normal = np.r_[original.features[a["row"]], 0.]
        dual = np.linalg.solve(original.hessian, normal)
        residual = original.data[a["row"]]-normal@point
        expected = 2*residual*dual/(normal@dual)
        np.testing.assert_allclose(np.array(a["theta"])-point, expected, rtol=1e-12, atol=1e-14)
        np.testing.assert_allclose(a["theta"], b["theta"], rtol=1e-12, atol=1e-14)
        assert a["proposed_residual"] == pytest.approx(-residual, abs=1e-14)
        assert a["predicted_barrier"] == pytest.approx(.125*expected@original.hessian@expected)
        assert expected[-1] != 0  # Mean/volatility cross curvature is retained.


def test_region_probes_reuse_prepared_curvature():
    model = QuadraticRegionModel()
    neighboring_residual_proposals(model, np.array([.1, .2, .3]), hessian=model.hessian)
    assert model.requests == [1, 1, 1, 1]


@pytest.mark.parametrize("metric", [np.diag([1., -1., 2.]), np.full((3, 3), np.nan)])
def test_region_probes_reject_unsupported_curvature(metric):
    model = QuadraticRegionModel()
    result = neighboring_residual_proposals(model, np.zeros(3), hessian=metric)
    assert result["proposals"] == [] and model.requests == []


def test_region_probes_check_feasibility_before_likelihood():
    model = QuadraticRegionModel(feasible=False)
    result = neighboring_residual_proposals(model, np.array([.1, .2, .3]))
    assert model.requests == [4] and len(result["proposals"]) == 4
    assert all(not p["feasible"] and "nll" not in p for p in result["proposals"])


@pytest.mark.parametrize("count", [0, -1, 1.5, True, np.nan, np.inf])
def test_region_probes_reject_bad_budgets_before_work(count):
    model = QuadraticRegionModel()
    with pytest.raises(ValueError):
        neighboring_residual_proposals(model, np.zeros(3), candidates=count)
    assert model.requests == []


def sample(seed):
    rng = np.random.default_rng(seed)
    return rng.normal(size=400), rng.normal(size=(400, 2))


@pytest.mark.parametrize("seed,density", [(2785, SkewT), (2785, GeneralizedHyperbolic),
                                         (2785, VarianceGamma), (2786, Normal), (2786, StudentT)])
def test_kink_refinement_is_quiet_and_explicitly_uncertified(seed, density, capsys):
    y, x = sample(seed)
    result = fit_joint_mean_spec(ARX(1) + EGARCH() + density(), y, x=x, maxiter=300)
    info = result.fit_info
    evidence = info.nonsmooth_evidence
    assert evidence["supports_kink_stop"] and not evidence["certified"]
    assert not result.converged
    assert info.stationarity_status == "uncertified"
    assert info.inference_status == "unavailable"
    assert result.std_errors is None and result.cov_matrix is None
    assert info.kkt_residual > info.kkt_tolerance
    assert evidence["mixed_kkt"] <= info.kkt_tolerance
    assert any("not a stationarity certificate" in warning for warning in info.warnings)
    phases = info.phase_diagnostics
    face = next(p for p in phases if p.get("algorithm") == "native-slsqp-affine-shock-face")
    assert face["nll"] <= phases[0]["nll"]
    assert face["requested_accuracy"] == 1e-12
    assert face["accuracy"] == face["refinement_attempts"][0]["accuracy"]
    assert face["iterations"] == sum(p["charged_iterations"] for p in face["refinement_attempts"])
    assert sum(p["iterations"] for p in phases) <= 300
    captured = capsys.readouterr()
    assert not captured.out and not captured.err
    restored = FitMetadata(**json.loads(json.dumps(info.to_dict())))
    assert restored.nonsmooth_evidence == evidence


@pytest.mark.parametrize("multiple", [False,True])
@pytest.mark.parametrize("failure", [None,"feasible","face","normal_minimum","mixed_kkt",
    "tangent_jump","maximum_value_discrepancy","backward_error","stability"])
def test_residual_face_evidence_keeps_every_stopping_check(monkeypatch,multiple,failure):
    from types import SimpleNamespace
    import scivol._nonsmooth_fit as nonsmooth

    model = SimpleNamespace(data=np.zeros(4),units=np.ones(2))
    model.evaluate = lambda *args, **kwargs: (0., None, None, np.zeros(4), np.ones(4))
    theta = np.ones(2)
    calls = []
    def bracket(*args,radius,**kwargs):
        calls.append(radius)
        face_key = "face_set_exact" if multiple else "single_face"
        result = dict(certified=False,feasible=True,normal_minimum=True,mixed_kkt=0.,
            tangent_jump=0.,maximum_value_discrepancy=0.,backward_error=0.,**{face_key:True})
        if failure in ("feasible","normal_minimum"):
            result[failure] = False
        elif failure=="face":
            result[face_key] = False
        elif failure in ("mixed_kkt","tangent_jump"):
            result[failure] = 2e-6
        elif failure=="maximum_value_discrepancy":
            result[failure] = 8e-12
        elif failure=="backward_error":
            result[failure] = 16*np.finfo(float).eps
        gradient = np.full(2,1e-3 if failure=="stability" and radius==2 else 0.)
        result.update(gradients=[gradient,gradient],smooth_gradient=gradient,
                      face_gradients=np.array([gradient]))
        return result
    monkeypatch.setattr(nonsmooth,"bracket_residual_face",bracket)
    monkeypatch.setattr(nonsmooth,"bracket_residual_faces",bracket)
    evidence = nonsmooth.residual_face_evidence(model,theta,[0],1e-12,multiple=multiple)
    assert evidence["supports_kink_stop"]==(failure is None)
    assert not evidence["certified"] and evidence["diagnostic_threshold"]==1e-6
    assert calls==[1,2]
    assert not {"gradients","smooth_gradient","face_gradients"}.intersection(evidence)
    np.testing.assert_array_equal(theta,np.ones(2))


@pytest.mark.parametrize("support", [True,False,"invalid"])
@pytest.mark.parametrize("strict_better", [False,True])
def test_affine_refinement_uses_requested_accuracy_then_strict_evidence_fallback(
    monkeypatch,support,strict_better
):
    from types import SimpleNamespace
    from scipy.optimize import OptimizeResult
    import scivol._joint_mean_fit as joint
    import scivol._nonsmooth_fit as nonsmooth

    model = SimpleNamespace(code=0,features=np.eye(2),data=np.zeros(2),request_counts=[0]*8)
    calls, evidence_calls = [], []
    def solver(model,point,**options):
        calls.append((point.copy(),options))
        model.request_counts[3] += 1
        second = len(calls)==2
        return OptimizeResult(x=np.full(2,float(second)),fun=(.5 if strict_better else 2.) if second else 1.,
            nit=3 if second else 2,status=9,success=False,
            native_jet_requests={"F":0,"G":0,"F|G":1},constraint_curvature_resets=0)
    def evidence(model,point,*args,**kwargs):
        evidence_calls.append(point.copy())
        model.request_counts[3] += 2
        if support=="invalid":
            raise ValueError("Unresolved limiting evidence")
        return dict(supports_kink_stop=support,certified=False)
    monkeypatch.setattr(joint,"_solve_joint_phase",solver)
    monkeypatch.setattr(nonsmooth,"residual_face_evidence",evidence)
    start = np.ones(2)
    result = joint._solve_affine_phase(model,start,[0],multiple=False,accuracy=1e-12,
        maxiter=10,transformed=True,speculate=False,curvature_reset=False,
        invalid_trial_policy="minimum-step")
    count = 1 if support is True else 2
    assert len(calls)==count and calls[0][1]["accuracy"]==1e-12
    assert all(c[1]["transformed"] and not c[1]["speculate"] for c in calls)
    assert all(c[1]["invalid_trial_policy"]=="minimum-step" for c in calls)
    if count==2:
        np.testing.assert_array_equal(calls[1][0],np.zeros(2))
        assert calls[1][1]["maxiter"]==8 and calls[1][1]["accuracy"]==np.finfo(float).eps
    assert len(evidence_calls)==(2 if strict_better and count==2 else 1)
    assert result.nit==sum(p["charged_iterations"] for p in result.refinement_attempts)==(2 if count==1 else 5)
    assert result.nfev==result.njev==count+2*len(evidence_calls)
    assert result.native_jet_requests["F|G"]==count
    assert result.fun==(.5 if strict_better and count==2 else 1.) and not result.success
    np.testing.assert_array_equal(start,np.ones(2))


@pytest.mark.parametrize("failure", ["setup","overrun"])
def test_affine_refinement_preserves_finite_work_and_cannot_overrun(monkeypatch,failure):
    from types import SimpleNamespace
    from scipy.optimize import OptimizeResult
    import scivol._joint_mean_fit as joint
    import scivol._nonsmooth_fit as nonsmooth

    model = SimpleNamespace(code=0,features=np.eye(2),data=np.zeros(2),request_counts=[0]*8)
    calls = []
    def solver(model,point,**options):
        calls.append(options)
        if len(calls)==2 and failure=="setup":
            model.request_counts[4] += 1
            raise ValueError("Strict phase setup failed")
        return OptimizeResult(x=np.zeros(2),fun=1.,nit=2,status=9,success=False)
    monkeypatch.setattr(joint,"_solve_joint_phase",solver)
    monkeypatch.setattr(nonsmooth,"residual_face_evidence",lambda *a,**k:dict(supports_kink_stop=False))
    def run():
        return joint._solve_affine_phase(model,np.ones(2),[0],multiple=False,accuracy=1e-12,
            maxiter=3,transformed=False,speculate=False,curvature_reset=False,
            invalid_trial_policy="interpolate")
    if failure=="overrun":
        with pytest.raises(RuntimeError,match="iteration budget"):
            run()
    else:
        result = run()
        assert result.fun==1. and result.nit==2 and not result.success
        assert result.refinement_attempts[1]["error"]=="Strict phase setup failed"
        assert result.refinement_attempts[1]["charged_iterations"]==0
        assert result.nhev==1
        np.testing.assert_array_equal(result.x,np.zeros(2))
    assert [c["maxiter"] for c in calls]==[3,1]


def test_gh_affine_request_plan_keeps_one_strict_phase(monkeypatch):
    from types import SimpleNamespace
    from scipy.optimize import OptimizeResult
    import scivol._joint_mean_fit as joint
    import scivol._nonsmooth_fit as nonsmooth

    model = SimpleNamespace(code=4,features=np.eye(2),data=np.zeros(2),request_counts=[0]*8)
    seen = []
    def solve(*args,**options):
        seen.append(options)
        return OptimizeResult(x=np.zeros(2),fun=1.,nit=2,status=9,success=False)
    monkeypatch.setattr(joint,"_solve_joint_phase",solve)
    monkeypatch.setattr(nonsmooth,"residual_face_evidence",lambda *a,**k:dict(supports_kink_stop=False))
    result = joint._solve_affine_phase(model,np.ones(2),[0],multiple=True,accuracy=1e-12,
        maxiter=10,transformed=True,speculate=False,curvature_reset=False,
        invalid_trial_policy="minimum-step")
    assert len(seen)==1 and seen[0]["accuracy"]==np.finfo(float).eps
    assert seen[0]["maxiter"]==10 and seen[0]["transformed"]
    assert result.nit==2 and not result.success


def test_nearby_evidence_cannot_turn_into_a_smooth_certificate():
    import time

    y, x = sample(2785)
    model = JointMeanModel(ARX(1) + EGARCH() + Normal(), y, x)
    solved = solve_joint(model, model.initialize(), transformed=False, accuracy=1e-12,
                         maxiter=300, allow_chart_refinement=True)
    assert solved.success
    solved.nearby_kink_evidence = dict(certified=False, supports_kink_stop=True)
    fitted = assemble_joint_fit(model, solved, False, 1e-12, 300, time.perf_counter())
    assert fitted.fit_info.kkt_residual <= fitted.fit_info.kkt_tolerance
    assert not fitted.converged
    assert fitted.fit_info.stationarity_status == "uncertified"
    assert fitted.fit_info.inference_status == "unavailable"


def test_smooth_fit_retains_ordinary_inference_and_no_kink_warning():
    y, x = sample(2785)
    fitted = fit_joint_mean_spec(ARX(1) + EGARCH() + Normal(), y, x=x, maxiter=300)
    assert fitted.converged
    assert fitted.fit_info.nonsmooth_evidence is None
    assert fitted.fit_info.inference_status == "available"
    assert not any("kink" in warning for warning in fitted.fit_info.warnings)


@pytest.mark.parametrize("sign,duplicates,constrained", [(1, False, False), (-1, False, False),
                                                        (1, True, False), (1, False, True)])
def test_neighboring_gradients_do_not_confuse_maxima_multiple_faces_or_infeasibility(
    sign, duplicates, constrained
):
    from types import SimpleNamespace
    from scivol._nonsmooth_fit import bracket_residual_face

    observations = np.array([0.0, 0.0, 1.0]) if duplicates else np.array([0.0, 1.0])
    design = np.ones((len(observations), 1))

    def evaluate(theta, mask, *, paths):
        value = sign*abs(theta[0]) + theta[1]**2/2
        gradient = np.array([sign*np.sign(theta[0]), theta[1]]) if mask & 2 else None
        return value, gradient, None, observations-theta[0], np.ones(len(observations))

    def assess(theta, gradient):
        return ((not constrained or theta[0] >= 0),
                np.inf if gradient is None else np.max(abs(gradient)), max(-theta[0], 0))

    model = SimpleNamespace(family=2, p_vol=1, data=observations, features=design,
        columns=1, size=2, units=np.ones(2), evaluate=evaluate, assess=assess)
    theta = np.array([1e-8, 0.2])
    if constrained:
        with pytest.raises(ValueError, match="outside the model domain"):
            bracket_residual_face(model, theta)
        return
    evidence = bracket_residual_face(model, theta)
    np.testing.assert_array_equal(theta, [1e-8, 0.2])
    assert evidence["normal_minimum"] == (sign > 0)
    assert evidence["single_face"] == (not duplicates)
    assert evidence["feasible"] == (not constrained)
    assert evidence["mixed_kkt"] == 0.2


@pytest.mark.parametrize("seed,density", [(2785, 2), (2785, 4), (2785, 7), (2786, 0), (2786, 1)])
def test_neighboring_gradients_match_independent_limiting_ad(seed, density):
    from scivol._nonsmooth_fit import bracket_residual_face
    from scivol._joint_mean_fit import DENSITIES
    from scivol._devtools.joint_mean_oracle import joint_value_grad_hess

    y, x = sample(seed)
    model = JointMeanModel(ARX(1) + EGARCH() + DENSITIES[density](), y, x)
    result = solve_joint(model, model.initialize(), transformed=False,
        accuracy=1e-12, maxiter=300, allow_chart_refinement=True)
    assert not result.success
    assert result.nearby_kink_evidence["supports_kink_stop"]
    evidence = bracket_residual_face(model, result.x)
    orientation = int(np.sign(model.features[evidence["row"], evidence["pivot"]]))
    for actual, side in zip(evidence["gradients"], (orientation, -orientation)):
        value, gradient, _ = joint_value_grad_hess(result.x, model.data, model.features,
            model.h0, 1, 1, 2, density, request=3, kink_limit=(evidence["row"], side),
            kink_pivot=evidence["pivot"])
        np.testing.assert_allclose(value, result.fun, rtol=0, atol=2e-9)
        np.testing.assert_allclose(actual*model.units/len(model.data),
                                   gradient*model.units/len(model.data), rtol=2e-7, atol=2e-9)
    assert result.nit == sum(p["iterations"] for p in result.phases) <= 300
    assert model.assess(result.x, model.evaluate(result.x, 2)[1])[1] > 1e-6


@pytest.mark.parametrize("gradient_state", ["retained", "missing", "nonfinite", "wrong_shape"])
def test_affine_refinement_obeys_its_budget_and_explicit_solver_policies(monkeypatch, gradient_state):
    import scivol._joint_mean_fit as joint

    y, x = sample(2785)
    model = JointMeanModel(ARX(1) + EGARCH() + SkewT(), y, x)
    prefix = joint._solve_joint_phase(model, model.initialize(), transformed=False,
        accuracy=1e-12, maxiter=150, use_curvature=False)
    prefix.jac = model.evaluate(prefix.x, 2)[1]
    prefix.stationarity_bound = model.assess(prefix.x, prefix.jac)[1]
    initial = prefix.x.copy()
    original = joint._solve_joint_phase
    seen, gradient_requests = [], []
    evaluate = model.evaluate

    def recorded_evaluation(theta, request=7, **options):
        if request == 2:
            gradient_requests.append(theta.copy())
        return evaluate(theta, request, **options)

    monkeypatch.setattr(model, "evaluate", recorded_evaluation)

    def recorded(*args, **options):
        seen.append(options)
        candidate = original(*args, **options)
        if gradient_state == "missing":
            candidate.jac = None
        elif gradient_state == "nonfinite":
            candidate.jac = np.full(model.size, np.nan)
        elif gradient_state == "wrong_shape":
            candidate.jac = np.zeros(model.size + 1)
        return candidate

    monkeypatch.setattr(joint, "_solve_joint_phase", recorded)
    phases = []
    _, remaining, _ = joint._refine_terminated_face(model, prefix, 4, phases, 1e-12,
        False, speculate=False, curvature_reset=False)
    assert len(seen) == 1 and seen[0]["maxiter"] == 2
    assert not seen[0]["speculate"] and not seen[0]["curvature_reset"]
    assert remaining + sum(p["iterations"] for p in phases) == 4
    assert phases[-1]["native_jet_requests"] is not None
    assert len(gradient_requests) == (0 if gradient_state == "retained" else 1)
    np.testing.assert_array_equal(prefix.x, initial)


@pytest.mark.parametrize("transformed", [False, True])
def test_refinement_preserves_independent_coordinate_paths(transformed):
    y, x = sample(2785)
    model = JointMeanModel(ARX(1) + EGARCH() + SkewT(), y, x)
    result = solve_joint(model, model.initialize(), transformed=transformed,
        accuracy=1e-12, maxiter=300, allow_chart_refinement=False)
    expected = "z" if transformed else "theta"
    assert result.selected_space == expected
    assert all(p["space"] == expected for p in result.phases)
    assert result.nit <= 300


def test_kink_refinement_keeps_the_terminal_reserve():
    y, x = sample(2785)
    model = JointMeanModel(ARX(1) + EGARCH() + SkewT(), y, x)
    result = solve_joint(model, model.initialize(), transformed=False, accuracy=1e-12,
                         maxiter=300, terminal_reserve=200)
    assert result.nearby_kink_evidence["supports_kink_stop"]
    assert result.nit <= 100
    face = next(p for p in result.phases if p.get("algorithm") == "native-slsqp-affine-shock-face")
    assert face["iteration_budget"] == (100-result.phases[0]["iterations"])//2


@pytest.mark.parametrize("pivot", [-1, 2, 1.5, np.nan])
def test_limiting_oracle_rejects_invalid_mean_pivots(pivot):
    from scivol._devtools.joint_mean_oracle import joint_value_grad_hess

    with pytest.raises(ValueError, match="nonzero mean feature"):
        joint_value_grad_hess(np.array([0.1, 0.2, -0.1, 0.08, -0.04, 0.7]),
            np.array([0.3, 0.6, 0.2]), np.ones((3, 2)), 1.0, 1, 1, 2, 0,
            request=3, kink_limit=(1, 1), kink_pivot=pivot)
