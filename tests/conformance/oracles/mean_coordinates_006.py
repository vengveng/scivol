"""Independent MP forward AD and inverse-polynomial design filtering.

Partitioned from the retained anchored-chart oracle; no core math imported.
"""
from dataclasses import dataclass
import mpmath as mp

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


def inverse_lag_polynomial(ma, length):
    # Coefficients of (1 + psi_1 z + ... + psi_q z^q)^-1.
    weights = [ma[0] * 0 + 1 if ma else mp.mpf(1)]
    for degree in range(1, length):
        weights.append(-sum((coefficient * weights[degree - lag]
                             for lag, coefficient in enumerate(ma, 1) if lag <= degree), 0))
    return weights


def filtered_rows(spec, coordinates, data):
    weights = inverse_lag_polynomial(coordinates[1 + spec.p_ar:], spec.count - spec.start)
    rows = []
    for t in range(spec.start, spec.count):
        a = sum((weights[t - u] * mp.mpf(float(data[u]))
                 for u in range(spec.start, t + 1)), 0)
        b = sum(weights[:t - spec.start + 1], 0)
        for lag, coefficient in enumerate(coordinates[1:1 + spec.p_ar], 1):
            a -= coefficient * sum((weights[t - u] * mp.mpf(float(data[u - lag]))
                                    for u in range(spec.start, t + 1)), 0)
        rows.append((a, b))
    return rows
