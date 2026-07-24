from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

from scivol import BatchFitResult, EGARCH, GED, GARCH, GJRGARCH, Normal, SkewT, StudentT
from scivol._kernels.routine import Routine
from scivol.result import FitMetadata


class DummyBatchResult:
    def __init__(self, data_len: int) -> None:
        self.n_obs = data_len
        self.fit_info = FitMetadata()
        self.time_elapsed = 0.0
        self._index = None
        self._name = None

    @property
    def warnings(self) -> tuple[str, ...]:
        return self.fit_info.warnings

    @property
    def converged(self) -> bool:
        return True

    @property
    def log_likelihood(self) -> float:
        return -1.0

    @property
    def n_iter(self) -> int:
        return 0


def _install_dummy_garch_normal(monkeypatch: pytest.MonkeyPatch) -> None:
    import scivol._kernels as _k

    def fake_fit(y, **_kw):
        return DummyBatchResult(len(y))

    monkeypatch.setitem(
        _k._ROUTINES,
        "GARCH(1,1)+Normal",
        Routine(uid="GARCH(1,1)+Normal", n_params=3, fit=fake_fit),
    )


@pytest.fixture
def returns_frame() -> pd.DataFrame:
    rng = np.random.default_rng(20260711)
    return pd.DataFrame(
        {
            "a": rng.standard_normal(320) * 0.010,
            "b": rng.standard_normal(320) * 0.015,
            "c": rng.standard_normal(320) * 0.012,
        },
        index=pd.date_range("2020-01-01", periods=320),
    )


def test_fit_many_records_failures_without_losing_successes(returns_frame: pd.DataFrame) -> None:
    data = returns_frame.copy()
    data.loc[data.index[10], "c"] = np.nan

    batch = (GARCH(1, 1) + Normal()).fit_many(
        data, n_jobs=1, maxiter=100, verbose=False
    )

    assert isinstance(batch, BatchFitResult)
    assert batch.succeeded == ("a", "b")
    assert batch.failed == ("c",)
    assert batch.failures["c"].error_type == "ValueError"
    assert batch.to_dataframe().loc["c", "status"] == "failed"


def test_fit_many_summarizes_recorded_fit_warnings(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_dummy_garch_normal(monkeypatch)
    data = pd.DataFrame({"a": np.arange(10.0), "b": np.arange(10.0) + 1.0})

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        batch = (GARCH(1, 1) + Normal()).fit_many(data, n_jobs=1)

    runtime_warnings = [w for w in caught if issubclass(w.category, RuntimeWarning)]
    assert runtime_warnings == []
    expected = "10 obs for 3 parameters; recommend >15."
    assert batch.warning_summary == (f"2 series: {expected}",)
    assert batch.warnings == {"a": (expected,), "b": (expected,)}
    assert batch.to_dataframe().loc["a", "warning_count"] == 1


def test_fit_many_warm_starts_each_series(returns_frame: pd.DataFrame) -> None:
    spec = GARCH(1, 1) + Normal()
    cold = spec.fit_many(returns_frame, n_jobs=1, maxiter=100, verbose=False)
    warm = spec.fit_many(
        returns_frame,
        initial_results=cold,
        n_jobs=1,
        maxiter=2,
        verbose=False,
    )

    assert not cold.failures
    assert not warm.failures
    assert all(result.fit_info.warm_started is True for result in warm.values())
    assert all(result.fit_info.maxiter == 2 for result in warm.values())
    assert all(result.n_iter <= 2 for result in warm.values())


def test_fit_many_timeout_is_reported_per_series(returns_frame: pd.DataFrame) -> None:
    batch = (GARCH(1, 1) + Normal()).fit_many(
        returns_frame[["a"]],
        n_jobs=1,
        timeout=1e-6,
        verbose=False,
    )

    assert batch.failed == ("a",)
    assert batch.failures["a"].timed_out is True


def test_fit_rolling_reuses_successful_previous_window(returns_frame: pd.DataFrame) -> None:
    rolling = (GARCH(1, 1) + Normal()).fit_rolling(
        returns_frame[["a", "b"]],
        window=280,
        step=20,
        n_jobs=1,
        maxiter=50,
        verbose=False,
    )

    assert len(rolling) == 3
    assert rolling[0].batch["a"].fit_info.warm_started is False
    assert all(window.batch["a"].fit_info.warm_started is True for window in rolling[1:])
    assert [window.stop for window in rolling] == [280, 300, 320]


def test_warm_start_rejects_a_different_specification(returns_frame: pd.DataFrame) -> None:
    source = (GARCH(1, 1) + Normal()).fit(
        returns_frame["a"], maxiter=50, verbose=False
    )

    with pytest.raises(ValueError, match="Warm-start result"):
        (GJRGARCH(1, 1) + Normal()).fit(
            returns_frame["a"], initial_params=source, maxiter=2, verbose=False
        )


@pytest.mark.parametrize("volatility", [GARCH, GJRGARCH, EGARCH])
@pytest.mark.parametrize("density", [Normal, StudentT, SkewT, GED])
@pytest.mark.parametrize("log_mode", [False, True])
def test_all_marginal_surfaces_accept_theta_warm_starts(
    returns_frame: pd.DataFrame,
    volatility,
    density,
    log_mode: bool,
) -> None:
    spec = volatility(1, 1) + density()
    cold = spec.fit(
        returns_frame["a"],
        solver="slsqp",
        log_mode=log_mode,
        maxiter=100,
        verbose=False,
    )
    warm = spec.fit(
        returns_frame["a"],
        solver="slsqp",
        log_mode=log_mode,
        initial_params=cold,
        maxiter=2,
        verbose=False,
    )

    assert warm.fit_info.warm_started is True
    assert warm.fit_info.maxiter == 2
    assert warm.n_iter <= 2
    assert np.all(np.isfinite(warm.params))
