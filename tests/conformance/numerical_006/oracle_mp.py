"""Extracted independent MP forward AD, with no native imports."""
from __future__ import annotations
from dataclasses import dataclass
from functools import lru_cache
import mpmath as mp
import numpy as np

@dataclass(frozen=True)
class Jet:
    """Dense second-order forward AD with independent MP scalar arithmetic."""
    value: mp.mpf
    gradient: tuple
    hessian: tuple

    @classmethod
    def constant(cls, value, width):
        zero = mp.mpf(0)
        return cls(mp.mpf(value), (zero,) * width, (zero,) * (width * width))

    @classmethod
    def variable(cls, value, width, index):
        result = cls.constant(value, width)
        return cls(result.value, tuple(mp.mpf(i == index) for i in range(width)), result.hessian)

    def lift(self, other):
        return other if isinstance(other, Jet) else Jet.constant(other, len(self.gradient))

    def __add__(self, other):
        other = self.lift(other)
        return Jet(self.value + other.value,
                   tuple(a + b for a, b in zip(self.gradient, other.gradient)),
                   tuple(a + b for a, b in zip(self.hessian, other.hessian)))

    __radd__ = __add__

    def __neg__(self):
        return Jet(-self.value, tuple(-a for a in self.gradient), tuple(-a for a in self.hessian))

    def __sub__(self, other):
        return self + -self.lift(other)

    def __rsub__(self, other):
        return self.lift(other) + -self

    def __mul__(self, other):
        if not isinstance(other, Jet):
            scalar = mp.mpf(other)
            return Jet(self.value * scalar, tuple(a * scalar for a in self.gradient),
                       tuple(a * scalar for a in self.hessian))
        other = self.lift(other)
        width = len(self.gradient)
        return Jet(self.value * other.value,
                   tuple(a * other.value + self.value * b
                         for a, b in zip(self.gradient, other.gradient)),
                   tuple(self.hessian[i * width + j] * other.value
                         + self.value * other.hessian[i * width + j]
                         + self.gradient[i] * other.gradient[j]
                         + self.gradient[j] * other.gradient[i]
                         for i in range(width) for j in range(width)))

    __rmul__ = __mul__

    def reciprocal(self):
        width = len(self.gradient)
        first, second = -1 / self.value**2, 2 / self.value**3
        return Jet(1 / self.value, tuple(first * a for a in self.gradient),
                   tuple(first * self.hessian[i * width + j]
                         + second * self.gradient[i] * self.gradient[j]
                         for i in range(width) for j in range(width)))

    def __truediv__(self, other):
        return self * self.lift(other).reciprocal()


def log_jet(x):
    width = len(x.gradient)
    return Jet(mp.log(x.value), tuple(g / x.value for g in x.gradient),
               tuple(x.hessian[i * width + j] / x.value
                     - x.gradient[i] * x.gradient[j] / x.value**2
                     for i in range(width) for j in range(width)))



def power_jet(x, exponent):
    """Scalar power chain rule on a strictly positive MP Jet."""
    assert x.value > 0
    exponent = mp.mpf(exponent)
    value = x.value**exponent
    first = exponent * value / x.value
    second = exponent * (exponent - 1) * value / x.value**2
    width = len(x.gradient)
    return Jet(value, tuple(first * g for g in x.gradient),
               tuple(first * x.hessian[i * width + j]
                     + second * x.gradient[i] * x.gradient[j]
                     for i in range(width) for j in range(width)))



def exp_jet(x):
    value, width = mp.exp(x.value), len(x.gradient)
    return Jet(value, tuple(value * g for g in x.gradient),
               tuple(value * (x.hessian[i * width + j] + x.gradient[i] * x.gradient[j])
                     for i in range(width) for j in range(width)))



def loggamma_jet(x):
    width = len(x.gradient)
    first, second = mp.digamma(x.value), mp.polygamma(1, x.value)
    return Jet(mp.loggamma(x.value), tuple(first * g for g in x.gradient),
               tuple(first * x.hessian[i * width + j] + second * x.gradient[i] * x.gradient[j]
                     for i in range(width) for j in range(width)))



def unscaled_theta_oracle(orders, theta, data, h0, *, digits, density="normal", remove_seed=False):
    """Full-theta AD of unscaled equations, with zero residual and h0 histories."""
    ar, ma, p, q = orders
    start, offset, width = max(orders), 1 + ar + ma, theta.size
    with mp.workdps(digits):
        point = [Jet.variable(float(value), width, index) for index, value in enumerate(theta)]
        delta = point[offset + 1 + 2*p + q]
        seed = (Jet.constant(0, width) if remove_seed else
                exp_jet(delta * (mp.log(mp.mpf(float(h0))) / 2)))
        residuals, powers = [Jet.constant(0, width)] * data.size, [seed] * data.size
        total = Jet.constant(0, width)
        assert density in ("normal", "studentt", "skewt")
        if density in ("studentt", "skewt"):
            nu = point[-2] if density == "skewt" else point[-1]
            constant = (loggamma_jet(nu / 2) - loggamma_jet((nu + 1) / 2)
                        + log_jet((nu - 2) * mp.pi) / 2)
            if density == "skewt":
                skew = point[-1]
                shift = 4 * skew * exp_jet(-constant) * (nu - 2) / (nu - 1)
                stretch = exp_jet(log_jet(1 + 3 * skew * skew - shift * shift) / 2)
        for t in range(start, data.size):
            residual = mp.mpf(float(data[t])) - point[0]
            for lag in range(ar):
                residual -= point[1 + lag] * mp.mpf(float(data[t - 1 - lag]))
            for lag in range(ma):
                residual -= point[1 + ar + lag] * residuals[t - 1 - lag]
            residuals[t] = residual
            power = point[offset]
            for lag in range(p):
                if t - 1 - lag < start:
                    continue
                error = residuals[t - 1 - lag]
                assert error.value != 0, "Seed fixtures must avoid residual kinks"
                magnitude = error if error.value > 0 else -error
                base = magnitude - point[offset + 1 + p + lag] * error
                power += point[offset + 1 + lag] * exp_jet(delta * log_jet(base))
            for lag in range(q):
                power += point[offset + 1 + 2*p + lag] * powers[t - 1 - lag]
            powers[t] = power
            log_variance = 2 * log_jet(power) / delta
            standardized_square = residual * residual * exp_jet(-log_variance)
            if density == "normal":
                total += (mp.log(2 * mp.pi) + log_variance + standardized_square) * mp.mpf(".5")
            elif density == "studentt":
                total += constant + log_variance / 2 + (nu + 1) * log_jet(
                    1 + standardized_square / (nu - 2)) / 2
            else:
                branch = stretch * residual * exp_jet(-log_variance / 2) + shift
                adjusted = branch / (1 + skew if branch.value >= 0 else 1 - skew)
                total += constant - log_jet(stretch) + log_variance / 2 + (nu + 1) * log_jet(
                    1 + adjusted * adjusted / (nu - 2)) / 2
        return total



def jet_outputs(jet):
    width = len(jet.gradient)
    return (np.array([float(jet.value)]), np.array(list(map(float, jet.gradient))),
            np.array(list(map(float, jet.hessian))).reshape(width, width))


@lru_cache(maxsize=12)
def legendre_rule(count, digits):
    with mp.workdps(digits):
        nodes, weights = mp.gauss_quadrature(count, "legendre")
        return tuple((nodes[i], weights[i]) for i in range(count))



def compose_partials(value, gradient, hessian, arguments):
    """Compose exact local derivatives at the arguments' current scalar point."""
    width, local = len(arguments[0].gradient), len(arguments)
    return Jet(value,
        tuple(sum(gradient[a]*arguments[a].gradient[i] for a in range(local))
              for i in range(width)),
        tuple(sum(gradient[a]*arguments[a].hessian[i*width+j] for a in range(local))
              + sum(hessian[a*local+b]*arguments[a].gradient[i]*arguments[b].gradient[j]
                    for a in range(local) for b in range(local))
              for i in range(width) for j in range(width)))
