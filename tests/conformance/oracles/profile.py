"""Independent exact Normal variance profile on the declared scored mean sample."""

from __future__ import annotations

import math
from fractions import Fraction

import mpmath as mp
import numpy as np

from .garch import jax_modules
from .mean import mean_count, mp_residuals, numpy_residuals, reflection_coefficients, residual_path


def nuisance(case, theta=None, *, digits=90):
    theta = case["parameters"] if theta is None else theta
    with mp.workdps(digits):
        residuals = mp_residuals(case, theta)
        rows = np.asarray(case["scored_rows"], dtype=int) - case["hold_back"]
        sse = mp.fsum(residuals[row] ** 2 for row in rows)
        count = len(rows)
        if not count:
            raise ValueError("A profile needs a nonempty scored sample")
        if sse == 0:
            raise ValueError("Exact zero SSE has an unbounded variance profile")
        variance = sse / count
        value = count * (mp.log(2 * mp.pi) + 1 + mp.log(variance)) / 2
        return {"sse": sse, "variance": variance, "n_eff": count, "f": value}


def theta_objective(case, theta):
    _, jnp, _ = jax_modules()
    rows = np.asarray(case["scored_rows"], dtype=int) - case["hold_back"]
    residuals = numpy_residuals(case, theta)[rows]
    largest = float(np.max(np.abs(residuals)))
    if not largest or not math.isfinite(largest):
        raise ValueError("Oracle anchor must have finite nonzero scored residuals")
    # A point-constant power-of-two scale changes only algebra, not the objective.
    scale = math.ldexp(1.0, math.frexp(largest)[1] - 1)
    constant = math.log(2 * math.pi) + 1 + 2 * math.log(scale) - math.log(len(rows))

    def objective(point):
        standardized = residual_path(point, case)[rows] / scale
        return len(rows) * (constant + jnp.log(jnp.sum(standardized ** 2))) / 2

    return objective


def jet(case, *, theta=None):
    if theta is None:
        if case.get("transformed", False):
            raise ValueError("Original-z validation must anchor to actual mapped theta")
        theta = case["parameters"]
    theta = np.asarray(theta, dtype=np.float64)
    if len(theta) != mean_count(case):
        raise ValueError("Profile derivative axes contain mean parameters only")
    if not len(theta):
        return float(nuisance(case, theta)["f"]), np.empty(0), np.empty((0, 0))
    jax, jnp, _ = jax_modules()
    point = jnp.asarray(theta)
    objective = theta_objective(case, theta)
    return (float(objective(point)), np.asarray(jax.grad(objective)(point)),
            np.asarray(jax.jacfwd(jax.grad(objective))(point)))


def coordinate_map(case):
    _, jnp, _ = jax_modules()
    intercept = int(case["intercept"])
    ar_order = case["p"] if case["model"] == "arma" else case.get("linear_ar_order", 0)

    def mapping(z):
        result = jnp.asarray(z)
        ar_end = intercept + ar_order
        if ar_order:
            result = result.at[intercept:ar_end].set(reflection_coefficients(z[intercept:ar_end]))
        if case["model"] == "arma":
            result = result.at[ar_end:].set(reflection_coefficients(z[ar_end:], -1.0))
        return result

    return mapping


def pullback(case, theta_jet):
    if not mean_count(case):
        return theta_jet
    jax, jnp, _ = jax_modules()
    z = jnp.asarray(case["original_z"])
    mapping = coordinate_map(case)
    jacobian = np.asarray(jax.jacfwd(mapping)(z))
    curvature = np.asarray(jax.jacfwd(jax.jacfwd(mapping))(z))
    value, gradient, hessian = theta_jet
    return (value, jacobian.T @ gradient,
            jacobian.T @ hessian @ jacobian + np.einsum("i,ijk->jk", gradient, curvature))


def full_normal_jet(case, theta, variance):
    jax, jnp, _ = jax_modules()
    rows = np.asarray(case["scored_rows"], dtype=int) - case["hold_back"]
    point = jnp.asarray(np.r_[theta, variance])

    def objective(parameters):
        residuals = residual_path(parameters[:-1], case)[rows]
        return (len(rows) * (math.log(2 * math.pi) + jnp.log(parameters[-1])) / 2
                + jnp.sum(residuals ** 2) / (2 * parameters[-1]))

    return (float(objective(point)), np.asarray(jax.grad(objective)(point)),
            np.asarray(jax.jacfwd(jax.grad(objective))(point)))


def linear_closed_jet(case, theta=None):
    theta = case["parameters"] if theta is None else np.asarray(theta)
    rows = np.asarray(case["scored_rows"], dtype=int)
    design = case["design"][rows]
    if case["intercept"]:
        design = np.column_stack([np.ones(len(rows)), design])
    residuals = case["y"][rows] - design @ theta
    sse = float(residuals @ residuals)
    if not sse:
        raise ValueError("Exact zero SSE has no finite profile")
    a = -design.T @ residuals
    b = design.T @ design
    return (float(nuisance(case, theta)["f"]), len(rows) * a / sse,
            len(rows) * (b / sse - 2 * np.outer(a / sse, a / sse)))


def rounded_variance_backward_error(case, rounded_variance, theta=None):
    exact = nuisance(case, theta)
    with mp.workdps(90):
        ratio = mp.mpf(float(rounded_variance)) / exact["variance"]
        return {"relative_variance_error": ratio - 1,
                "full_minus_profile_f": exact["n_eff"] * (mp.log(ratio) + 1 / ratio - 1) / 2}


def mp_jet(case, theta=None, *, digits=80):
    """Differentiate a scalar MP recursion without rounding perturbed parameters."""
    theta = case["parameters"] if theta is None else theta
    with mp.workdps(digits):
        point = tuple(mp.mpf(float(value)) for value in theta)
        y = tuple(mp.mpf(float(value)) for value in case["y"][case["hold_back"]:case["stop"]])
        design = [[mp.mpf(float(value)) for value in row]
                  for row in case["design"][case["hold_back"]:case["stop"]]]
        first = case["scored_rows"][0] - case["hold_back"]
        intercept = int(case["intercept"])

        def objective(*parameters):
            residuals = [mp.mpf(0)] * len(y)
            for time in range(case["recursion_begin"], len(y)):
                residual = y[time] - (parameters[0] if intercept else 0)
                if case["model"] == "linear":
                    residual -= mp.fsum(x * b for x, b in zip(design[time], parameters[intercept:]))
                else:
                    residual -= mp.fsum(parameters[intercept + lag] * y[time - lag - 1]
                                        for lag in range(case["p"]))
                    residual -= mp.fsum(parameters[intercept + case["p"] + lag] * residuals[time - lag - 1]
                                        for lag in range(case["q"]))
                residuals[time] = residual
            sse = mp.fsum(value ** 2 for value in residuals[first:])
            count = len(y) - first
            return count * (mp.log(2 * mp.pi) + 1 + mp.log(sse / count)) / 2

        gradient, hessian = [], []
        for i in range(len(point)):
            orders = [0] * len(point)
            orders[i] = 1
            gradient.append(mp.diff(objective, point, tuple(orders)))
            row = []
            for j in range(len(point)):
                second = orders.copy()
                second[j] += 1
                row.append(mp.diff(objective, point, tuple(second)))
            hessian.append(row)
        return (float(objective(*point)), np.asarray(gradient, dtype=float),
                np.asarray(hessian, dtype=float).reshape(len(point), len(point)))


def linear_exact_jet(case):
    """Exact dyadic-input moments and rational derivatives, rounded only at outputs."""
    if case["model"] != "linear" or case.get("transformed", False):
        raise ValueError("Exact range fixtures use theta-space affine residuals")
    rows = case["scored_rows"]
    parameters = [Fraction.from_float(float(value)) for value in case["parameters"]]
    design = [[Fraction.from_float(float(value)) for value in case["design"][row]] for row in rows]
    if case["intercept"]:
        design = [[Fraction(1)] + row for row in design]
    residuals = [Fraction.from_float(float(case["y"][row])) - sum(x * b for x, b in zip(values, parameters))
                 for row, values in zip(rows, design)]
    sse = sum(value ** 2 for value in residuals)
    if not sse:
        raise ValueError("Exact zero SSE has no finite profile")
    count, width = len(rows), len(parameters)
    a = [-sum(values[i] * residual for values, residual in zip(design, residuals)) for i in range(width)]
    b = [[sum(values[i] * values[j] for values in design) for j in range(width)] for i in range(width)]
    exact_g = [count * value / sse for value in a]
    exact_h = [[count * (b[i][j] / sse - 2 * a[i] * a[j] / sse ** 2)
                for j in range(width)] for i in range(width)]
    variance = sse / count
    with mp.workdps(100):
        mp_variance = mp.mpf(variance.numerator) / variance.denominator
        value = count * (mp.log(2 * mp.pi) + 1 + mp.log(mp_variance)) / 2
    return {"f": float(value), "g": np.asarray(exact_g, dtype=float),
            "h": np.asarray(exact_h, dtype=float).reshape(width, width),
            "exact_g": exact_g, "exact_h": exact_h, "sse": sse, "variance": variance,
            "a": a, "b": b, "n_eff": count}


def mp_linear_ar_pullback(case, theta, *, digits=100):
    """Exact affine profile moments at actual theta, MP chart derivatives at original z."""
    order, start = int(case["linear_ar_order"]), int(case["intercept"])
    if case["model"] != "linear" or not case["transformed"] or order < 1:
        raise ValueError("This MP target requires a positive linear AR original chart")
    if len(theta) != mean_count(case) or len(case["original_z"]) != len(theta) or start + order > len(theta):
        raise ValueError("AR chart axes must fit the reduced mean layout")
    anchored = dict(case, parameters=np.asarray(theta, dtype=np.float64), transformed=False)
    exact = linear_exact_jet(anchored)
    width = len(theta)
    with mp.workdps(digits):
        coordinates = tuple(mp.mpf(float(value)) for value in case["original_z"][start:start + order])

        def reflection(*point):
            coefficients = [mp.mpf(0)] * order
            for stage, coordinate in enumerate(point):
                previous = coefficients.copy()
                kappa = mp.tanh(coordinate)  # MP perturbations must not pass through float.
                for row in range(stage):
                    coefficients[row] = previous[row] - kappa * previous[stage - row - 1]
                coefficients[stage] = kappa
            return coefficients

        jacobian = mp.eye(width)
        curvature = [[[mp.mpf(0) for _ in range(width)] for _ in range(width)] for _ in range(width)]
        for row in range(order):
            component = lambda *point, row=row: reflection(*point)[row]
            for column in range(order):
                first = [0] * order
                first[column] = 1
                jacobian[start + row, start + column] = mp.diff(component, coordinates, tuple(first))
                for other in range(order):
                    second = first.copy()
                    second[other] += 1
                    curvature[start + row][start + column][start + other] = mp.diff(component, coordinates, tuple(second))
        gradient = mp.matrix([mp.mpf(value.numerator) / value.denominator for value in exact["exact_g"]])
        hessian = mp.matrix([[mp.mpf(value.numerator) / value.denominator for value in row]
                             for row in exact["exact_h"]])
        transformed_gradient = jacobian.T * gradient
        transformed_hessian = jacobian.T * hessian * jacobian
        for row in range(width):
            for column in range(width):
                transformed_hessian[row, column] += mp.fsum(
                    gradient[parameter] * curvature[parameter][row][column] for parameter in range(width))
        return {"jet": (exact["f"], np.asarray(list(transformed_gradient), dtype=float),
                        np.asarray(transformed_hessian.tolist(), dtype=float)),
                "jacobian": np.asarray(jacobian.tolist(), dtype=float),
                "curvature": np.asarray(curvature, dtype=float),
                "variance": float(exact["variance"]), "n_eff": exact["n_eff"]}
