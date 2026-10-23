"""Independent VG half-power moments from the existing normalized density Jet.

The split and positive-base power factorization follow the GH MP moment oracle.
There is no native call, JAX density, finite difference, or moment normalization
repair. Gamma factors are composed after the two common half-moment integrals.
"""

from __future__ import annotations

from functools import lru_cache

import mpmath as mp

from test_aparch42_gh_complete_stress import compose_partials, legendre_rule
from test_native_anchored_mean_chart import Jet
from test_native_aparch_subset_curvature import exp_jet, log_jet
from test_vg_retained_value_accuracy import scalar_vg_nll, vg_density_jet


def half_power_moment(point, sign, *, digits, outer, inner, map_power=8):
    """Differentiate one fixed-sign integral in (delta, rho, lambda)."""
    assert sign in (-1, 1) and len(point) == 3
    with mp.workdps(digits):
        delta, rho, lam = [Jet.variable(mp.mpf(float(v)), 3, i) for i, v in enumerate(point)]
        one = Jet.constant(1, 3)
        cuts = [mp.mpf(0)]+[2/mp.pi*mp.atan(mp.mpf(x)**(mp.mpf(1)/map_power))
            for x in (".1", ".25", ".5", 1, 2, 3, 4, 5, 6, 7, 9, 12, 20, 40)]+[mp.mpf(1)]
        sums = {name: Jet.constant(0, 3) for name in ("power", "mass", "mean", "second")}
        scalar_error = mp.mpf(0)
        rho_second_absolute = mp.mpf(0)
        worst = None
        for panel, (left, right) in enumerate(zip(cuts, cuts[1:])):
            half = (right-left)/2
            for index, (node, weight) in enumerate(legendre_rule(outer, digits)):
                tangent = mp.tan(mp.pi*(left+half*(node+1))/2)
                magnitude = tangent**map_power
                jacobian = mp.pi*map_power/2*tangent**(map_power-1)*(1+tangent*tangent)
                density = exp_jet(-vg_density_jet(Jet.constant(sign*magnitude, 3),
                    one, rho, lam, count=inner, digits=digits))
                measure = half*weight*jacobian
                powered = exp_jet(delta*mp.log(magnitude))*density
                sums["power"] += powered*measure
                sums["mass"] += density*measure
                sums["mean"] += density*(sign*magnitude*measure)
                sums["second"] += density*(magnitude*magnitude*measure)
                term = abs(powered.hessian[4]*measure)
                rho_second_absolute += term
                if worst is None or term > worst["term"]:
                    worst = dict(magnitude=magnitude, term=term, panel=panel, node=index)
                if index in (0, outer//2, outer-1):
                    exact = mp.exp(-scalar_vg_nll(sign*magnitude, mp.mpf(1), rho.value, lam.value))
                    scalar_error = max(scalar_error, abs(density.value-exact)*measure)
        return sums, dict(weighted_scalar_density_error=scalar_error,
            rho_second_absolute_integral=rho_second_absolute, worst_rho_second_term=worst)


def compose_gamma_moment(point, positive, negative):
    """Full four-parameter AD, including each endpoint gamma sensitivity."""
    gamma, delta, rho, lam = [Jet.variable(v, 4, i) for i, v in enumerate(point)]
    arguments = (delta, rho, lam)
    halves = [compose_partials(value.value, value.gradient, value.hessian, arguments)
              for value in (positive, negative)]
    return (exp_jet(delta*log_jet(1-gamma))*halves[0]
            +exp_jet(delta*log_jet(1+gamma))*halves[1])


@lru_cache(maxsize=8)
def refined_half_moments(point, digits, outer, inner):
    halves = tuple(half_power_moment(point, sign, digits=digits, outer=outer, inner=inner)[0]
                   for sign in (1, -1))
    with mp.workdps(digits):
        tolerance = mp.mpf("1e-20")
        for name, target in (("mass", 1), ("mean", 0), ("second", 1)):
            total = halves[0][name]+halves[1][name]
            assert abs(total.value-target) <= tolerance, "VG normalization value failed"
            assert all(abs(value) <= tolerance for value in (*total.gradient, *total.hessian)), "VG normalization derivatives failed"
    return tuple(half["power"] for half in halves)


def normalized_vg_moment(point, *, digits, outer, inner):
    with mp.workdps(digits):
        values = tuple(mp.mpf(float(v)) for v in point)
        halves = refined_half_moments(tuple(map(float, point[1:])), digits, outer, inner)
        return compose_gamma_moment(values, *halves)
