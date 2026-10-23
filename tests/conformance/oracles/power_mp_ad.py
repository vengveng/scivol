"""MP arithmetic adapter for the existing independent density/moment owners.

Forward AD and Bessel integration are numerical_006 owners, not legacy imports.
This module supplies arithmetic dispatch and a true-infinite-support quadrature
plan. It does not restate the GH density or APARCH moment equations.
"""

import mpmath as mp

from numerical_006.oracle_mp import Jet, exp_jet, log_jet, loggamma_jet, power_jet, legendre_rule
from numerical_006.test_vg_retained_value_accuracy import log_bessel_jet
from .power import mp_power_density, mp_power_moment
from .values import mp_logpdf, mp_nig_reduced_logpdf


class Number:
    """Operator dispatch to the existing MP Jet, without new AD rules."""

    def __init__(self, jet, arithmetic):
        self.jet, self.arithmetic = jet, arithmetic

    @property
    def value(self):
        return self.jet.value

    def _other(self, other):
        return self.arithmetic.lift(other)

    def __add__(self, other):
        return Number(self.jet + self._other(other).jet, self.arithmetic)

    __radd__ = __add__

    def __neg__(self):
        return Number(-self.jet, self.arithmetic)

    def __sub__(self, other):
        return self + -self._other(other)

    def __rsub__(self, other):
        return self._other(other) + -self

    def __mul__(self, other):
        return Number(self.jet * self._other(other).jet, self.arithmetic)

    __rmul__ = __mul__

    def __truediv__(self, other):
        return Number(self.jet / self._other(other).jet, self.arithmetic)

    def __rtruediv__(self, other):
        return self._other(other) / self

    def __pow__(self, exponent):
        return self.arithmetic.power(self, exponent)

    def __rpow__(self, base):
        return self.arithmetic.power(base, self)

    def __lt__(self, other):
        return self.value < self._other(other).value

    def __le__(self, other):
        return self.value <= self._other(other).value

    def __gt__(self, other):
        return self.value > self._other(other).value

    def __ge__(self, other):
        return self.value >= self._other(other).value


class Arithmetic:
    def __init__(self, width, *, digits, bessel_count):
        self.width, self.digits, self.bessel_count = width, digits, bessel_count
        self._bessel = {}

    @property
    def pi(self):
        return mp.pi

    def lift(self, value):
        if isinstance(value, Number):
            if value.arithmetic is not self:
                raise ValueError("Cannot mix derivative contexts")
            return value
        return Number(Jet.constant(value, self.width), self)

    def mpf(self, value):
        return value if isinstance(value, Number) else mp.mpf(value)

    def variables(self, point):
        return tuple(Number(Jet.variable(value, self.width, index), self)
                     for index, value in enumerate(point))

    def exp(self, value):
        return Number(exp_jet(value.jet), self) if isinstance(value, Number) else mp.exp(value)

    def log(self, value):
        return Number(log_jet(value.jet), self) if isinstance(value, Number) else mp.log(value)

    def log1p(self, value):
        return self.log(1 + value) if isinstance(value, Number) else mp.log1p(value)

    def sqrt(self, value):
        return Number(power_jet(value.jet, mp.mpf(".5")), self) if isinstance(value, Number) else mp.sqrt(value)

    def loggamma(self, value):
        return Number(loggamma_jet(value.jet), self) if isinstance(value, Number) else mp.loggamma(value)

    def power(self, base, exponent):
        if not isinstance(base, Number) and not isinstance(exponent, Number):
            return mp.power(base, exponent)
        if not isinstance(exponent, Number) and exponent in (0, 1, 2, 3, 4):
            result = self.lift(1)
            for _ in range(int(exponent)):
                result = result * base
            return result
        return self.exp(self.lift(exponent) * self.log(self.lift(base)))

    def besselk(self, order, argument):
        if not isinstance(order, Number) and not isinstance(argument, Number):
            return mp.besselk(order, argument)
        order, argument = self.lift(order), self.lift(argument)
        key = order.jet, argument.jet
        if key not in self._bessel:
            self._bessel[key] = exp_jet(log_bessel_jet(
                order.jet, argument.jet, self.bessel_count, self.digits))
        return Number(self._bessel[key], self)


def density_jet(point, *, reduced=False, digits=80, bessel_count=40):
    with mp.workdps(digits):
        arithmetic = Arithmetic(3, digits=digits, bessel_count=bessel_count)
        z, rho, zeta = arithmetic.variables(tuple(mp.mpf(float(v)) for v in point))
        result = (mp_nig_reduced_logpdf(z, rho, zeta, arithmetic=arithmetic) if reduced else
                  mp_logpdf(z, "gh", (rho, zeta, mp.mpf("-.5")), arithmetic=arithmetic))
        return result.jet


def gh_moment_jet(point, *, digits=60, outer=32, bessel_count=32):
    with mp.workdps(digits):
        arithmetic = Arithmetic(5, digits=digits, bessel_count=bessel_count)
        gamma, delta, *shape = arithmetic.variables(tuple(mp.mpf(float(v)) for v in point))
        result = mp_power_moment(gamma, delta, shape, "gh", arithmetic=arithmetic,
            half_integral=lambda power, parameters, density: integrate_halves(
                power, parameters, density, arithmetic, outer))
        return result.jet


def integrate_halves(delta, shape, density, arithmetic, count):
    # Only the integration coordinates live here. The shared owner forms x^delta*p(x).
    magnitudes = (".25", "1", "2", "3", "4", "4.5", "5", "5.5", "6", "7", "9", "12", "20", "40")
    cuts = [mp.mpf(0)] + [2 / mp.pi * mp.atan(mp.mpf(x) ** mp.mpf(".25"))
                         for x in magnitudes] + [mp.mpf(1)]
    halves = [arithmetic.lift(0), arithmetic.lift(0)]
    for lower, upper in zip(cuts, cuts[1:], strict=False):
        half = (upper - lower) / 2
        for node, weight in legendre_rule(count, arithmetic.digits):
            unit = lower + half * (node + 1)
            tangent = mp.tan(mp.pi * unit / 2)
            magnitude = tangent ** 4
            jacobian = 2 * mp.pi * tangent ** 3 * (1 + tangent * tangent)
            for index, sign in enumerate((1, -1)):
                halves[index] += mp_power_density(magnitude, sign, delta, shape, density,
                    arithmetic=arithmetic) * (half * weight * jacobian)
    return tuple(halves)
