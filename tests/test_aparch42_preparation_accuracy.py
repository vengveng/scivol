"""Separate preparation tolerances must be explicit and local to their phase."""

import numpy as np
import pytest

from scivol import _aparch_fit_policy as policy
from test_aparch_fit_policy import make_point, outcome, run_policy


@pytest.mark.parametrize("preparation_accuracy", [None, 1e-12, 1e-8])
def test_preparation_accuracy_does_not_change_joint_or_chart_accuracy(preparation_accuracy):
    requests = []

    def callback(request):
        requests.append(request)
        power = request.physical_power if request.role == "power" else 2.0
        point = make_point(10.0, power=power, mode=-105 if request.role in
                           {"power", "complementary"} else 9)
        return outcome(point)

    selected = run_policy(callback, released=callback, phases=policy.ControllerPhases(
        power=callback, power_accuracy=preparation_accuracy,
        qualified_power_densities=(0,), complementary=(("chart", callback),)))
    for request, diagnostic in zip(requests, selected.diagnostics):
        expected = (preparation_accuracy if request.role == "power"
                    and preparation_accuracy is not None else 1e-10)
        assert request.accuracy == diagnostic["accuracy"] == expected
    assert {request.role for request in requests} == {"baseline", "power", "released", "complementary"}


@pytest.mark.parametrize("accuracy", [0, -1e-12, np.nan, np.inf, -np.inf, True, "1e-12"])
def test_invalid_preparation_accuracy_is_rejected_before_any_work(accuracy):
    def forbidden(request):
        pytest.fail("Invalid policy must not enter the solver.")

    with pytest.raises(ValueError, match="Preparation accuracy"):
        run_policy(forbidden, phases=policy.ControllerPhases(power_accuracy=accuracy))
