"""Multi-start fits: the model's candidate starts, the automatic race, explicit starts and their record."""

from __future__ import annotations

import numpy as np
import pytest

api = pytest.importorskip("scivol._scivolapi")
from scivol import DCC, GARCH, Normal  # noqa: E402


@pytest.fixture(scope="module")
def bimodal() -> np.ndarray:
    # A persistent GARCH whose likelihood has a second, lower maximum at
    # moderate persistence, where the single model start leads.
    return (GARCH() + Normal()).simulate(2000, [0.05, 0.02, 0.975], seed=229).data


def test_model_candidates_start_with_the_model_start_and_lie_in_the_domain(bimodal):
    from scivol._fit import native_spec

    model = api.Model(native_spec(GARCH() + Normal(), 0))
    result = (GARCH() + Normal()).fit(bimodal, starts=3)
    starts = [s["theta"] for s in result.fit_info.starts]
    assert len(starts) == 3 and all(model.feasible(s) for s in starts)
    assert np.allclose(starts[0], model.start(bimodal, None))
    assert [s[1] + s[2] for s in starts[1:]] == pytest.approx([0.98, 0.995])
    assert all(s["candidate"] and s["finished"] for s in result.fit_info.starts)


def test_the_race_finishes_a_start_whose_probe_beats_the_first(bimodal):
    single = (GARCH() + Normal()).fit(bimodal, starts=1)
    auto = (GARCH() + Normal()).fit(bimodal)
    assert single.success and auto.success
    assert auto.log_likelihood > single.log_likelihood + 0.5
    assert auto.params[2] > 0.95 > single.params[2]
    record = auto.fit_info.starts
    chosen = [k for k, s in enumerate(record) if s["selected"]]
    assert len(chosen) == 1 and chosen[0] > 0 and record[chosen[0]]["finished"]
    assert record[chosen[0]]["log_likelihood"] == pytest.approx(auto.log_likelihood, rel=1e-13)
    # The reported path is the selected start's: its probe, then its continuation.
    assert auto.fit_info.solver.startswith("probe z:trust") and auto.fit_info.solver.endswith("z:trust+theta:newton")
    assert auto.fit_info.optimization_space == "z+theta+z+theta"
    assert any(p.startswith(f"start {chosen[0] + 1}: probe ") for p in auto.fit_info.optimizer_phases)


def test_a_start_not_pursued_is_only_probed(bimodal):
    y = np.random.default_rng(0).standard_t(5, 2000)
    result = (GARCH() + Normal()).fit(y)
    first, *others = result.fit_info.starts
    assert first["finished"] and first["selected"]
    for s in others:
        assert not s["finished"] and not s["selected"] and s["log_likelihood"] <= first["log_likelihood"]


def test_explicit_starts_each_run_in_full_and_the_best_is_reported(bimodal):
    single = (GARCH() + Normal()).fit(bimodal, starts=1)
    auto = (GARCH() + Normal()).fit(bimodal)
    result = (GARCH() + Normal()).fit(bimodal, starts=[single.params, auto.params])
    assert [s["candidate"] for s in result.fit_info.starts] == [False, False]
    assert all(s["finished"] for s in result.fit_info.starts)
    assert result.log_likelihood == pytest.approx(auto.log_likelihood, abs=1e-8)
    warm = (GARCH() + Normal()).fit(bimodal, initial_params=single)
    assert len(warm.fit_info.starts) == 1 and not warm.fit_info.starts[0]["candidate"]


@pytest.mark.parametrize("bad, message", [(0, "at least 1"), ("fast", "starts must be"), ([[1.0, 2.0]], "needs 3 values")])
def test_invalid_start_choices_raise(bimodal, bad, message):
    with pytest.raises(ValueError, match=message):
        (GARCH() + Normal()).fit(bimodal, starts=bad)


def test_initial_params_and_a_start_choice_are_exclusive(bimodal):
    with pytest.raises(ValueError, match="initial_params"):
        (GARCH() + Normal()).fit(bimodal, initial_params=[0.05, 0.05, 0.9], starts=2)


def test_correlation_fits_race_their_candidate_starts():
    eps = np.random.default_rng(1).standard_normal((800, 3))
    auto = DCC().fit_from_residuals(eps)
    assert [s["theta"].tolist() for s in auto.starts] == [[0.05, 0.9], [0.02, 0.8]]
    assert sum(s["selected"] for s in auto.starts) == 1
    single = DCC().fit_from_residuals(eps, starts=1)
    assert len(single.starts) == 1 and single.path == auto.path
    assert auto.nll <= single.nll + 1e-12


def test_a_fit_without_an_estimate_raises_a_fit_error_with_its_reasons():
    from scivol import FitError
    from scivol._fit import failure_message

    assert issubclass(FitError, RuntimeError)
    y = np.r_[np.zeros(30), np.ones(70)]
    message = failure_message(["no phase produced a finite, feasible estimate"], y)
    assert message.startswith("no finite, feasible estimate: no phase produced")
    assert "30% of the observations are exactly zero" in message
    assert "exactly zero" not in failure_message(["x"], np.ones(100))
