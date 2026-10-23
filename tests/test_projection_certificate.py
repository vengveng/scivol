"""Pure-array certificates against exact binary-input rational mathematics."""

from fractions import Fraction

import numpy as np
import pytest

import scivol._fit_credibility as credibility


ACCUMULATORS = (np.float64, np.longdouble)


def rational(value):
    numerator, denominator = value.as_integer_ratio()
    return Fraction(numerator, denominator)


def exact_dot(left, right):
    return sum((rational(x) * rational(y) for x, y in zip(left, right)), Fraction(0))


def exact_proposal_gap(gradient, normals, gaps, point, dual):
    slacks = [exact_dot(row, point) + rational(gap) for row, gap in zip(normals, gaps)]
    assert all(slack >= 0 for slack in slacks)
    residuals = [
        rational(p) + rational(g) - exact_dot(column, dual)
        for p, g, column in zip(point, gradient, normals.T)
    ]
    return sum((r * r for r in residuals), Fraction(0)) + 2 * sum(
        (rational(weight) * slack for weight, slack in zip(dual, slacks)), Fraction(0)
    )


def exact_halfspace_projection(gradient, normal, gap):
    g = [rational(value) for value in gradient]
    a = [rational(value) for value in normal]
    multiplier = max(
        Fraction(0), (sum(x * y for x, y in zip(a, g)) - rational(gap))
        / sum(x * x for x in a),
    )
    return [-x + multiplier * y for x, y in zip(g, a)]


@pytest.fixture
def cancellation_proposal():
    # This feasible boundary proposal can be discarded by a last-bit slack
    # sign after an unguarded retraction. No false certificate is asserted.
    normals = np.array([[-1.0, -0.31947327002541953, -0.3491435533291031]])
    point = np.array([0.43033909376589663, -0.780672173628628, -0.41470282404706027])
    gaps = np.array([0.03614438407550234])
    gradient = np.array([-2.4303390937658964, 0.1417256335777889, -0.2835842826111459])
    return gradient, normals, gaps, point, np.array([2.0])


@pytest.mark.parametrize("accumulator", ACCUMULATORS)
def test_cancellation_proposal_retains_a_tight_conservative_certificate(
    cancellation_proposal, accumulator
):
    gradient, normals, gaps, point, dual = cancellation_proposal
    assert exact_dot(normals[0], point) + rational(gaps[0]) >= 0
    bound = credibility._projection_certificate_bound(
        *cancellation_proposal, accumulator=accumulator
    )
    projection = exact_halfspace_projection(gradient, normals[0], gaps[0])
    assert rational(bound) >= max(map(abs, projection))
    assert bound < 0.781
    origin_gap = exact_proposal_gap(gradient, normals, gaps, np.zeros_like(point), dual)
    # The zero-point certificate is genuinely looser, not a decimal-level change.
    assert rational(bound) ** 2 < 3 * origin_gap / 4


def test_public_certificate_uses_supplied_proposal_without_origin_fallback(
    monkeypatch, cancellation_proposal
):
    gradient, normals, gaps, point, dual = cancellation_proposal
    point_before, dual_before = point.copy(), dual.copy()
    monkeypatch.setattr(credibility, "_projection_multipliers", lambda *args: (point, dual))
    bound = credibility.projected_stationarity_bound(
        gradient, np.full(3, -np.inf), np.full(3, np.inf), normals, gaps
    )
    assert rational(bound) >= max(map(abs, exact_halfspace_projection(
        gradient, normals[0], gaps[0]
    )))
    assert bound < 0.781
    np.testing.assert_array_equal(point, point_before)
    np.testing.assert_array_equal(dual, dual_before)


@pytest.mark.parametrize("accumulator", ACCUMULATORS)
def test_slack_dot_error_enters_the_gap_before_the_square_root(accumulator):
    normals = np.array([[1.0, 1.0]], dtype=accumulator)
    gaps = np.array([1.0], dtype=accumulator)
    point = np.array([-1.0, 2.0**-54], dtype=accumulator)
    gradient = np.array([2.0, 1.0], dtype=accumulator)
    dual = np.array([1.0], dtype=accumulator)
    assert exact_dot(normals[0], point) + rational(gaps[0]) == Fraction(1, 2**54)
    if np.finfo(accumulator).eps == np.finfo(float).eps:
        assert (normals @ point + gaps)[0] == 0
    primal, upper_slack = credibility._retract_projection_proposal(normals, gaps, point)
    exact_slack = exact_dot(normals[0], primal) + rational(gaps[0])
    assert 0 <= exact_slack <= rational(upper_slack[0])
    bound = credibility._projection_certificate_bound(
        gradient, normals, gaps, point, dual, accumulator=accumulator
    )
    distance_bound = rational(bound) - max(rational(abs(value)) for value in primal)
    assert bound < 1.01  # The gradient norm cap cannot mask the gap accounting.
    assert distance_bound > Fraction(1, 10**9)
    assert distance_bound ** 2 >= exact_proposal_gap(gradient, normals, gaps, primal, dual)


@pytest.mark.parametrize("accumulator", ACCUMULATORS)
@pytest.mark.parametrize("gap", [0.0, np.nextafter(0.0, 1.0), 2.0**-54])
def test_retraction_certifies_exact_feasibility_and_upper_slacks(accumulator, gap):
    normals = np.array([[1.0, 1.0, 1.0]], dtype=accumulator)
    point = np.array([1.0, -2.0**-54, -1.0], dtype=accumulator)
    gaps = np.array([gap], dtype=accumulator)
    primal, upper_slack = credibility._retract_projection_proposal(normals, gaps, point)
    exact_slack = exact_dot(normals[0], primal) + rational(gaps[0])
    assert 0 <= exact_slack <= rational(upper_slack[0])
    assert np.isfinite(upper_slack).all() and np.all(upper_slack >= 0)
    assert np.all(np.abs(primal) <= np.abs(point))
    if gap == 0:
        np.testing.assert_array_equal(primal, np.zeros(3))


@pytest.mark.parametrize("accumulator", ACCUMULATORS)
def test_cancelled_force_keeps_an_outward_residual_allowance(accumulator):
    normals = np.array([[1.0, 0.0], [1.0, 0.0], [-1.0, 0.0]], dtype=accumulator)
    gaps = np.zeros(3, dtype=accumulator)
    point = np.array([0.0, -0.5], dtype=accumulator)
    gradient = np.array([1.0, 0.5], dtype=accumulator)
    dual = np.array([1024.0, 2.0**-45, 1023.0], dtype=accumulator)
    gap = exact_proposal_gap(gradient, normals, gaps, point, dual)
    assert gap == Fraction(1, 2**90)
    bound = credibility._projection_certificate_bound(
        gradient, normals, gaps, point, dual, accumulator=accumulator
    )
    assert 0.5 < bound < 0.501
    assert (rational(bound) - Fraction(1, 2)) ** 2 >= gap


@pytest.mark.parametrize("accumulator", ACCUMULATORS)
@pytest.mark.parametrize("magnitude", [np.nextafter(0.0, 1.0), 1e-200, 1.0, 1e200])
def test_gradient_norm_cap_is_outward_even_with_subnormals(accumulator, magnitude):
    gradient = np.array([magnitude, magnitude, magnitude])
    bound = credibility._projection_certificate_bound(
        gradient, np.empty((0, 3)), np.empty(0), np.ones(3), np.empty(0),
        accumulator=accumulator,
    )
    assert np.isfinite(bound)
    assert rational(bound) ** 2 >= exact_dot(gradient, gradient)


@pytest.mark.parametrize("accumulator", ACCUMULATORS)
@pytest.mark.parametrize("gap", [0.0, np.nextafter(0.0, 1.0), 0.125, 8.0])
@pytest.mark.parametrize("proposal", ["origin", "infeasible", "boundary"])
def test_halfspace_projection_is_bounded_for_nonoptimal_proposals(accumulator, gap, proposal):
    gradient = np.array([3.0, -1.0])
    normals = np.array([[1.0, -1.0]])
    gaps = np.array([gap])
    point = {
        "origin": np.zeros(2), "infeasible": np.array([-9.0, 4.0]),
        "boundary": np.array([-gap, 0.0]),
    }[proposal]
    projection = exact_halfspace_projection(gradient, normals[0], gaps[0])
    for dual in (np.array([0.0]), np.array([2.0]), np.array([16.0])):
        bound = credibility._projection_certificate_bound(
            gradient, normals, gaps, point, dual, accumulator=accumulator
        )
        assert np.isfinite(bound)
        assert rational(bound) >= max(map(abs, projection))


@pytest.mark.parametrize("accumulator", ACCUMULATORS)
@pytest.mark.parametrize("gap", [0.0, np.nextafter(0.0, 1.0), 0.125, 8.0])
def test_coupled_domain_has_an_independently_known_projection(accumulator, gap):
    # P((-10,1)) on x>=-gap, y<=x is (-gap,-gap) until gap=9/2,
    # and (-9/2,-9/2) afterward. These two constraints share the vertex.
    gradient = np.array([10.0, -1.0])
    normals = np.array([[1.0, 0.0], [1.0, -1.0]])
    gaps = np.array([gap, 0.0])
    projected_norm = min(rational(gap), Fraction(9, 2))
    for point, dual in (
        (np.zeros(2), np.zeros(2)),
        (np.array([-gap, -gap]), np.array([9.0, 1.0])),
        (np.array([-10.0, 1.0]), np.array([0.0, 5.5])),
    ):
        bound = credibility._projection_certificate_bound(
            gradient, normals, gaps, point, dual, accumulator=accumulator
        )
        assert np.isfinite(bound) and rational(bound) >= projected_norm


@pytest.mark.parametrize("accumulator", ACCUMULATORS)
@pytest.mark.parametrize("gap", [0.0, np.nextafter(0.0, 1.0), 1e-300, 1e-161])
def test_subnormal_complementarity_has_a_nonnegative_upper_gap(accumulator, gap):
    gradient, dual = np.array([1e-160]), np.array([1e-160])
    bound = credibility._projection_certificate_bound(
        gradient, np.array([[1.0]]), np.array([gap]), np.zeros(1), dual,
        accumulator=accumulator,
    )
    exact_gap = 2 * rational(dual[0]) * rational(gap)
    assert np.isfinite(bound) and bound > 0
    assert rational(bound) ** 2 >= exact_gap
    assert rational(bound) >= min(rational(gradient[0]), rational(gap))


@pytest.mark.parametrize("accumulator", ACCUMULATORS)
@pytest.mark.parametrize("exponent", [-500, -40, 0, 40, 500])
def test_positive_row_scaling_preserves_exact_projection(
    cancellation_proposal, accumulator, exponent
):
    gradient, normals, gaps, point, dual = cancellation_proposal
    scaled_normals, scaled_gaps = np.ldexp(normals, exponent), np.ldexp(gaps, exponent)
    scaled_dual = np.ldexp(dual, -exponent)
    projection = exact_halfspace_projection(gradient, normals[0], gaps[0])
    assert projection == exact_halfspace_projection(gradient, scaled_normals[0], scaled_gaps[0])
    bound = credibility._projection_certificate_bound(
        gradient, scaled_normals, scaled_gaps, point, scaled_dual, accumulator=accumulator
    )
    assert rational(bound) >= max(map(abs, projection))
    assert bound < 0.781


@pytest.mark.parametrize("exponent", [-500, 0, 500])
def test_public_row_normalization_does_not_change_certified_rows(
    monkeypatch, cancellation_proposal, exponent
):
    gradient, normals, gaps, point, dual = cancellation_proposal
    scale = np.ldexp(3.0, exponent)
    supplied_normals, supplied_gaps = normals * scale, gaps * scale

    def proposal(g, a, b):
        np.testing.assert_array_equal(a, supplied_normals / scale)
        np.testing.assert_array_equal(b, supplied_gaps / scale)
        return point, dual

    seen = []
    certificate = credibility._projection_certificate_bound

    def certify(g, a, b, p, weights):
        seen.append((a.copy(), b.copy()))
        return certificate(g, a, b, p, weights)

    monkeypatch.setattr(credibility, "_projection_multipliers", proposal)
    monkeypatch.setattr(credibility, "_projection_certificate_bound", certify)
    bound = credibility.projected_stationarity_bound(
        gradient, np.full(3, -np.inf), np.full(3, np.inf), supplied_normals, supplied_gaps
    )
    np.testing.assert_array_equal(seen[0][0], supplied_normals)
    np.testing.assert_array_equal(seen[0][1], supplied_gaps)
    projection = exact_halfspace_projection(gradient, supplied_normals[0], supplied_gaps[0])
    assert rational(bound) >= max(map(abs, projection))
    assert bound < 0.781


@pytest.mark.parametrize("index", range(5))
@pytest.mark.parametrize("invalid", [np.nan, np.inf, -np.inf])
def test_nonfinite_proposals_are_uncertified(index, invalid):
    values = [np.array([1.0]), np.array([[1.0]]), np.array([0.0]),
              np.array([0.0]), np.array([1.0])]
    values[index].flat[0] = invalid
    assert credibility._projection_certificate_bound(*values) == np.inf


@pytest.mark.parametrize("replacement", [
    (0, np.empty(0)), (0, np.ones((1, 1))), (1, np.ones(1)),
    (1, np.ones((2, 1))), (2, np.array([-1.0])), (2, np.ones((1, 1))),
    (3, np.zeros(2)), (3, np.array(0.0)), (4, np.array([-1.0])),
    (4, np.array([-np.nextafter(0.0, 1.0)])), (4, np.zeros(2)),
    (4, np.array(1.0)), (4, "invalid"), (4, {}),
])
def test_malformed_proposals_are_uncertified(replacement):
    values = [np.array([1.0]), np.array([[1.0]]), np.array([0.0]),
              np.array([0.0]), np.array([1.0])]
    index, value = replacement
    values[index] = value
    assert credibility._projection_certificate_bound(*values) == np.inf


@pytest.mark.parametrize("point, dual", [
    (np.array([np.nan, 0.0]), np.array([1.0, 0.0])),
    (np.zeros(2), np.array([-1.0, 0.0])), (np.zeros(2), np.array([np.inf, 0.0])),
    (np.zeros(2), np.zeros(3)), (np.zeros(2), {}),
    (np.array([np.finfo(float).max, 0.0]), np.full(2, np.finfo(float).max)),
])
def test_unusable_proposals_preserve_the_public_box_fallback(monkeypatch, point, dual):
    gradient = np.array([3.0, 0.125])
    lower, upper = np.array([0.0, -np.inf]), np.full(2, np.inf)
    expected = credibility._box_projection_bound(gradient, lower, upper)
    monkeypatch.setattr(credibility, "_projection_multipliers", lambda *args: (point, dual))
    assert credibility.projected_stationarity_bound(
        gradient, lower, upper, np.array([[2.0, 0.0]]), np.array([0.0])
    ) == expected


@pytest.mark.parametrize("accumulator", ACCUMULATORS)
def test_overflow_is_uncertified_not_a_zero_bound(accumulator):
    huge = np.finfo(accumulator).max
    assert credibility._projection_certificate_bound(
        np.array([1.0]), np.array([[huge]], dtype=accumulator), np.array([0.0]),
        np.array([2.0]), np.array([2.0]), accumulator=accumulator,
    ) == np.inf


@pytest.mark.parametrize("side", [-1.0, 1.0])
@pytest.mark.parametrize("gap", [0.0, np.nextafter(0.0, 1.0), 1e-8])
def test_unavailable_jacobian_preserves_exact_public_box_bound(side, gap):
    gradient = np.array([side * 1e100, 1e-5])
    lower = np.array([-gap if side > 0 else -np.inf, -np.inf])
    upper = np.array([np.inf if side > 0 else gap, np.inf])
    expected = credibility._box_projection_bound(gradient, lower, upper)
    assert credibility.projected_stationarity_bound(
        gradient, lower, upper, None, np.array([0.0])
    ) == expected
