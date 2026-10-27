"""Search contracts, native LR traversal, and explicit calibration limits."""
from __future__ import annotations

import numpy as np
import pytest

from scivol import (APARCH, ARMA, ARX, Auto, ConfidenceSet, EGARCH, GARCH, Role,
                    GJRGARCH, GeneralToSpecific, InformationCriterion, Normal,
                    Search, SkewT, StudentT)
from scivol import _scivolapi as api
from scivol._fit import native_spec
from scivol.result import EstimationResult


def search(**kwargs):
    return Search(volatility=[GJRGARCH.up_to(1, 1)], density=[SkewT()], **kwargs)


@pytest.fixture
def garch_data():
    return (GARCH()+Normal()).simulate(3000, [.1, .15, .7], seed=17).data


def test_templates_declare_generators_and_native_closure():
    for cls in [GARCH, GJRGARCH, EGARCH, APARCH]:
        part = cls.up_to(2, 1)
        assert (part.p, part.q) == (2, 1)
    assert ARX.up_to(2).lags == 2
    assert (ARMA.up_to(2, 1).p, ARMA.up_to(2, 1).q) == (2, 1)
    with pytest.raises(TypeError):
        GARCH(auto=True)
    with pytest.raises(ValueError):
        GARCH('auto')


def test_gets_reduces_native_graph_and_records_joint_tests(garch_data):
    result = search(constant_variance=False).fit(garch_data, n_jobs=2)
    assert isinstance(result.selected, EstimationResult)
    assert str(result.selected_model) == 'GARCH(1,1)+Normal'
    assert result.status == 'selected' and result.search_complete
    assert len(result.terminals) == 1
    assert len(result.paths) > 1
    assert any(t['purpose'] == 'gum_backtest' and ';' in t['restriction'] for t in result.tests)
    assert any(t['decision'] == 'unsupported' and 'Unidentified' in t['failure_reason'] for t in result.tests)
    assert all(t['law'] in ('chi_square', 'chi_bar_square')
               for t in result.tests if t['decision'] == 'accepted')
    for t in result.tests:
        assert {'parent', 'child', 'restriction', 'parent_log_likelihood',
                'child_log_likelihood', 'statistic', 'law', 'weights', 'p_value',
                'level', 'decision', 'failure_reason'} <= t.keys()
    assert result.fit_counts['warm_started_fits'] > 0
    assert not result.table().empty


def test_in_sample_pits_are_reported_without_blocking_adequacy(garch_data):
    result = search(constant_variance=False).fit(garch_data)
    assert result.status == 'selected'
    for candidate in result.candidates:
        if not candidate.certified:
            continue
        reported = [c for c in candidate.diagnostics['checks'] if c['name'] in ('berkowitz', 'dgt')]
        assert len(reported) == 2
        assert all(not c['required'] and not c['calibrated'] for c in reported)
    assert not hasattr(result, 'limitations')


def test_thread_scheduling_does_not_change_paths_or_decisions(garch_data):
    serial = search(constant_variance=False).fit(garch_data, n_jobs=1)
    parallel = search(constant_variance=False).fit(garch_data, n_jobs=4)
    assert serial.paths == parallel.paths
    assert serial.fit_counts == parallel.fit_counts
    assert [(t['parent'], t['child'], t['decision'], t['p_value']) for t in serial.tests] == [
        (t['parent'], t['child'], t['decision'], t['p_value']) for t in parallel.tests]
    np.testing.assert_array_equal(serial.selected.params, parallel.selected.params)


def test_budget_returns_best_certified_candidate_and_frontier(garch_data):
    result = search(constant_variance=False).fit(garch_data,
                                               procedure=GeneralToSpecific(max_fits=1))
    assert result.status == 'budget_exhausted'
    assert not result.search_complete and result.frontier
    assert result.selected.success
    assert result.fit_counts['optimizer_invocations'] == 1
    assert not result.terminals
    assert any(t['decision'] == 'budget_exhausted' for t in result.tests)


def test_qmle_and_unknown_options_raise(garch_data):
    with pytest.raises(ValueError, match='requires MLE'):
        search().fit(garch_data, method='qmle')
    with pytest.raises(TypeError):
        search().fit(garch_data, diagnostic_weight=50)
    with pytest.raises(TypeError):
        GARCH().fit(garch_data, select_by='bic')
    with pytest.raises(ValueError, match='explicit adequacy'):
        search().fit(garch_data, procedure=InformationCriterion('bic'))


def test_closure_false_fits_only_listed_candidates(garch_data):
    result = Search(volatility=[GARCH()], density=[Normal(), StudentT()],
                    constant_variance=False, closure=False).fit(
                        garch_data, procedure=InformationCriterion('bic', adequacy=False))
    assert len(result.candidates) == 2
    assert not result.tests
    expected = min(result.candidates, key=lambda c: c.information_criteria['bic'])
    assert result.selected_model is expected.specification
    assert all(c.diagnostics['status'] == 'disabled' for c in result.candidates)
    assert result.selected.log_likelihood == expected.fit.log_likelihood


def test_common_plan_survives_mean_and_regressor_reductions(garch_data):
    x = np.random.default_rng(7).normal(size=(len(garch_data), 2))
    result = Search(mean=[ARX(2)], volatility=[GARCH()], density=[Normal()],
                    constant_variance=False).fit(garch_data, x,
                    procedure=InformationCriterion('bic', adequacy=False, max_fits=96), n_jobs=2)
    assert {c.fit.scored for c in result.candidates if c.fit is not None} == {len(garch_data)-2}
    assert result.selected._native()[0].spec.mean.exog <= 2
    assert len(result.selected.params) == result.selected._native()[0].dimension


def test_constant_root_is_independent_and_encompassing_unavailable(garch_data):
    result = Search(density=[Normal()]).fit(garch_data)
    assert len(result.candidates) == 1
    constant = result.candidates[0]
    assert constant.model.constant_variance
    assert constant.encompassing['volatility_gum'] == 'unavailable'
    arch = next(c for c in constant.diagnostics['checks'] if c['name'] == 'absolute_li_mak[10]')
    assert arch['status'] == 'ok' and arch['required'] and arch['holm_reject']
    assert result.selected is None


def test_native_composition_preserves_likelihood_and_derivatives(garch_data):
    parent = api.Model(native_spec(ARX(1)+GJRGARCH()+SkewT()))
    a = next(r for r in parent.restrictions() if r.name == 'volatility.symmetry')
    middle = api.Model(a.child)
    b = next(r for r in middle.restrictions() if r.name == 'density.symmetry')
    ab = a.then(b)
    child = api.Model(ab.child)
    c = next(r for r in child.restrictions() if r.name == 'density.normal')
    joint = ab.then(c)
    child = api.Model(joint.child)
    plan = api.ComparisonPlan(garch_data, 3)
    theta = child.start(plan)
    lifted = joint.embed(theta)
    np.testing.assert_array_equal(lifted, a.embed(b.embed(c.embed(theta))))
    np.testing.assert_allclose(joint.map_start(lifted), theta)
    po = api.Jet(parent, plan).theta(lifted, 7)
    co = api.Jet(child, plan).theta(theta, 7)
    embedding = np.asarray(joint.embedding).reshape(parent.dimension, child.dimension)
    assert po['value'] == pytest.approx(co['value'], abs=1e-10)
    np.testing.assert_allclose(embedding.T @ po['gradient'], co['gradient'], atol=1e-9)
    np.testing.assert_allclose(embedding.T @ po['hessian'] @ embedding, co['hessian'], atol=1e-8)
    assert joint.types.count(api.RestrictionType.closed_boundary) == 1
    with pytest.raises(ValueError, match='does not connect'):
        b.then(a)


@pytest.mark.parametrize('loss', ['log_score', 'qlike'])
def test_native_predictive_losses(garch_data, loss):
    model = api.Model(native_spec(GARCH().spec))
    jet = api.Jet(model, garch_data)
    theta = np.array([.1, .15, .7])
    out = jet.theta(theta, api.TERMS | api.PATHS)
    expected = out['terms'] if loss == 'log_score' else np.log(out['variance'])+out['residual']**2/out['variance']
    np.testing.assert_allclose(api.predictive_losses(jet, theta, loss), expected, atol=1e-13)


@pytest.mark.slow
def test_auto_garch_normal_dgp():
    y = (GARCH()+Normal()).simulate(5000, [.1, .15, .7], seed=51).data
    result = Auto().fit(y)
    assert result.status == 'selected', result.summary()
    volatility = result.selected_model.get_component(Role.VOLATILITY)
    assert type(volatility) is GARCH and (volatility.p, volatility.q) == (1, 1), result.summary()
    law = result.selected._native()[0].innovation_density(result.selected.params)
    assert law.density == api.DensityKind.normal or (law.density == api.DensityKind.student_t and law.shape[0] == 0), result.summary()


@pytest.mark.slow
def test_gjr_dgp_keeps_real_leverage():
    y = (GJRGARCH()+Normal()).simulate(5000, [.05, .04, .1, .85], seed=81).data
    result = search(constant_variance=False).fit(y)
    assert result.status == 'selected', result.summary()
    assert isinstance(result.selected_model.get_component(Role.VOLATILITY), GJRGARCH), result.summary()
    assert result.selected.params[2] > 0
    assert any(t['restriction'] == 'volatility.symmetry' and t['decision'] == 'rejected' for t in result.tests)


@pytest.mark.slow
def test_student_dgp_keeps_tail_parameter():
    y = (GARCH()+StudentT()).simulate(5000, [.1, .15, .7, 1/6], seed=51).data
    result = search(constant_variance=False).fit(y)
    assert result.status == 'selected', result.summary()
    law = result.selected._native()[0].innovation_density(result.selected.params)
    assert law.density in (api.DensityKind.student_t, api.DensityKind.skew_t)
    assert law.shape[0] > 0


@pytest.mark.slow
@pytest.mark.parametrize('phi', [0., .3])
def test_mean_dgp_keeps_only_real_ar_term(phi):
    y = (ARX(1, constant=False)+GARCH()+Normal()).simulate(
        4000, [phi, .1, .15, .7], seed=92).data
    result = Search(mean=[ARX(1, constant=False)], volatility=[GJRGARCH()],
                    density=[SkewT()], constant_variance=False).fit(y)
    assert result.status == 'selected', result.summary()
    assert result.selected._native()[0].spec.mean.ar == int(phi != 0), result.summary()


@pytest.mark.slow
def test_predictive_confidence_set_is_aligned_and_reproducible():
    y = (GARCH()+StudentT()).simulate(2600, [.1, .15, .7, .15], seed=62).data
    space = Search(volatility=[GARCH()], density=[Normal(), StudentT()],
                   closure=False, constant_variance=False)
    procedure = ConfidenceSet(reduction=InformationCriterion('bic', adequacy=False),
                              refit_every=100, bootstrap_draws=199, seed=44)
    a = space.fit(y, procedure=procedure, n_jobs=1)
    b = space.fit(y, procedure=procedure, n_jobs=2)
    assert a.status == b.status == 'predictive_confidence_set'
    np.testing.assert_array_equal(a.predictive['losses'], b.predictive['losses'])
    assert a.predictive['losses'].shape == (520, 2)
    assert [c.key for c in a.confidence_set] == [c.key for c in b.confidence_set]
    assert a.selected.n_obs == len(y)


@pytest.mark.slow
def test_auto_iid_dgp_selects_constant_variance():
    y = np.random.default_rng(74).normal(size=5000)
    result = Auto().fit(y)
    # Calibrating retained faces opens additional paths. The fixed fit budget
    # may leave a frontier while still selecting the certified constant model.
    assert result.status in ('selected', 'budget_exhausted'), result.summary()
    if result.status == 'budget_exhausted':
        assert not result.search_complete and result.frontier
        assert result.fit_counts['optimizer_invocations'] == GeneralToSpecific().max_fits
    assert result.selected._native()[0].constant_variance, result.summary()
    assert result.selected.success


def test_selected_model_supports_threaded_batches(garch_data):
    result = Search(volatility=[GARCH()], density=[Normal()], closure=False,
                    constant_variance=False).fit(garch_data,
                    procedure=InformationCriterion('bic', adequacy=False))
    panel = np.column_stack([garch_data, .8*garch_data])
    batch = result.selected_model.fit_many(panel, n_jobs=2)
    assert not batch.failures
    assert batch.succeeded == ('0', '1')


def test_budget_includes_full_sample_deployment_refit(garch_data):
    result = Search(mean=[ARX(2)], volatility=[GARCH()], density=[Normal()],
                    constant_variance=False).fit(garch_data,
                    procedure=InformationCriterion('bic', adequacy=False, max_fits=8))
    assert result.fit_counts['optimizer_invocations'] <= 8
    assert result.selected is not None
    assert result.fit_counts['full_sample_refits'] == 1


def test_budget_does_not_promote_a_rejected_child():
    y = (GJRGARCH()+Normal()).simulate(3000, [.05, .025, .22, .82], seed=81).data
    result = Search(volatility=[GJRGARCH()], density=[Normal()],
                    constant_variance=False).fit(y,
                    procedure=GeneralToSpecific(max_fits=2))
    assert not result.search_complete
    assert str(result.selected_model) == 'GJR-GARCH(1,1)+Normal'
    assert result.fit_counts['optimizer_invocations'] <= 2


def test_exact_tail_face_saves_a_fit_without_spurious_parent_refit():
    y = np.random.default_rng(2).normal(size=600)
    result = Search(density=[StudentT()]).fit(y)
    assert result.status == 'selected'
    assert result.fit_counts['free_children'] == 1
    assert result.fit_counts['optimizer_invocations'] == 1
    assert result.fit_counts['parent_refits'] == 0
    reused = next(c for c in result.candidates if c.free)
    assert reused.fit.success
    assert [p.name for p in reused.fit.phases] == ['restriction:projection']
    assert any(t['statistic'] == 0 and t['p_value'] == 1 for t in result.tests)


def test_auto_boundary_tests_and_kink_stopping():
    y = (GARCH()+Normal()).simulate(3000, [.1, .15, .7], seed=101).data
    result = Auto().fit(y)
    assert result.status == 'selected'
    assert str(result.selected_model) == 'GARCH(1,1)+Normal'
    assert not any(t['decision'] in ('retained_boundary', 'singular_information') for t in result.tests)
    kink = next(c for c in result.candidates
                if str(c.specification) == 'ARX(1,constant=False)+EGARCH(1,1)+Normal')
    assert not kink.fit.success
    assert kink.fit.termination in (api.Termination.stalled, api.Termination.line_search_failed)
    assert kink.fit.warnings
    assert sum(p.iterations for p in kink.fit.phases) < 100
    # At the finite cusp, opposing one-sided gradients explain why the smooth
    # KKT check cannot certify it. No claim of nonexistence of an MLE is made.
    theta = np.array(kink.fit.theta)
    plan = api.ComparisonPlan(y, 1)
    jet = api.Jet(kink.model, plan)
    residual = jet.theta(theta, api.PATHS)['residual']
    index = np.argmin(np.abs(residual)) + 1
    root = y[index]/y[index-1]
    assert abs(theta[0]-root) < 1e-7
    left, right = theta.copy(), theta.copy()
    left[0], right[0] = root-1e-8, root+1e-8
    assert jet.theta(left, api.GRADIENT)['gradient'][0] < 0
    assert jet.theta(right, api.GRADIENT)['gradient'][0] > 0


@pytest.mark.parametrize('template', [ARX.up_to(1), ARMA.up_to(1, 0)])
def test_mean_widening_refits_roots_then_reduces_on_common_sample(template):
    y = np.random.default_rng(680).normal(size=5000)
    for t in range(2, len(y)):
        y[t] += .35*y[t-2]
    result = Search(mean=[template], density=[Normal()]).fit(y, n_jobs=1)
    assert result.status == 'selected'
    assert result.selected_model.get_component(Role.MEAN).autoregressive_order == 2
    assert [w['order'] for w in result.widenings] == [2]
    widening = result.widenings[0]
    assert widening['triggering_test'].startswith('residual_ljung_box[')
    assert widening['p_value'] < .01
    assert result.fit_counts['mean_widening_fits'] == 1
    assert {c.fit.scored for c in result.candidates if c.fit} == {len(y)-2}
    assert 'Mean widened to AR(2)' in result.summary()
    assert any(t['restriction'] == 'mean.ar' for t in result.tests)


def test_mean_widening_cap_budget_and_nonmean_rejections(garch_data):
    y = np.random.default_rng(680).normal(size=5000)
    for t in range(2, len(y)):
        y[t] += .35*y[t-2]
    search = Search(mean=[ARX.up_to(1)], density=[Normal()])
    capped = search.fit(y, procedure=GeneralToSpecific(max_mean_order=1))
    assert capped.status == 'mean_order_cap_reached' and capped.selected is None
    assert not capped.widenings
    budget = search.fit(y, procedure=GeneralToSpecific(max_fits=1))
    assert budget.status == 'budget_exhausted' and not budget.search_complete
    assert budget.fit_counts['optimizer_invocations'] == 1
    assert budget.frontier and budget.widenings[0]['order'] == 2
    assert budget.widenings[0]['fit_count'] == 0 and not budget.widenings[0]['completed']
    assert 'Mean widening pending' in budget.summary()
    volatility = search.fit(garch_data)
    assert volatility.selected is None and not volatility.widenings
    assert not any(c['holm_reject'] for candidate in volatility.candidates
                   for c in candidate.diagnostics.get('checks', [])
                   if c['name'].startswith('residual_ljung_box['))


def test_mean_widening_template_preserves_structure_and_nested_lag_restrictions():
    arx = ARX(2, constant=False)
    arx.set_n_exog(2)
    wider = arx.widen(5)
    assert wider.lags == 3 and wider.n_exog == 2 and not wider.constant
    arma = ARMA(2, 1, constant=False).widen(5)
    assert (arma.p, arma.q, arma.constant) == (3, 1, False)
    assert arx.widen(2) is None and Normal().widen(5) is None
    parent = api.Model(native_spec(wider.spec, 2))
    edge = next(r for r in parent.restrictions() if r.name == 'mean.ar')
    child = api.Model(edge.child)
    assert child.spec.mean.ar == 2 and child.spec.mean.exog == 2
    # Mean coefficients, two exogenous coefficients, then sigma2.
    theta = child.start(api.ComparisonPlan(np.random.default_rng(8).normal(size=100), 3,
                                         np.random.default_rng(9).normal(size=(100, 2))))
    embedded = edge.embed(theta)
    assert embedded[2] == 0
    np.testing.assert_array_equal(edge.map_start(embedded), theta)
    with pytest.raises(ValueError, match='max_mean_order'):
        GeneralToSpecific(max_mean_order=9)
    with pytest.raises(ValueError, match='max_mean_order'):
        Auto(max_mean_order=0)
