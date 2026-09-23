"""Conservative stationarity and inference checks, separate from solver status."""

from __future__ import annotations

import numpy as np
from scipy.optimize import nnls


def jet_agreement(
    previous_value, previous_gradient, value, gradient, units, n_eff, accuracy
):
    """Compare retained and reporting requests at one unchanged parameter point.

    Agreement is necessary evidence, not a numerical accuracy certificate. The
    errors use the same per-observation objective and parameter units as the
    stationarity check; no additional jet evaluation is performed here.
    """
    gradient_error = None
    gradient_tolerance = max(np.sqrt(accuracy), np.sqrt(np.finfo(float).eps))
    try:
        with np.errstate(over="raise", invalid="raise", divide="raise"):
            value_error = abs(value - previous_value) / n_eff
            if not np.isfinite(value_error):
                value_error = np.inf
    except FloatingPointError:
        value_error = np.inf
    if previous_gradient is not None and gradient is not None:
        try:
            with np.errstate(over="raise", invalid="raise", divide="raise"):
                difference = (gradient - previous_gradient) * units / n_eff
                gradient_error = float(np.hypot.reduce(difference, initial=0.0))
                if not np.isfinite(gradient_error):
                    gradient_error = np.inf
        except FloatingPointError:
            gradient_error = np.inf
    inconsistent = value_error > accuracy or (
        gradient_error is not None and gradient_error > gradient_tolerance
    )
    return {
        "status": (
            "inconsistent"
            if inconsistent
            else "unavailable" if gradient_error is None else "consistent"
        ),
        "value_error": float(value_error),
        "value_tolerance": float(accuracy),
        "gradient_error": gradient_error,
        "gradient_tolerance": float(gradient_tolerance),
        "scope": "same-point request agreement in optimizer units, not an accuracy certificate",
    }


def diagonal_descent_prediction(
    gradient, diagonal, lower_step, upper_step, jacobian, slack
):
    """Estimate remaining decrease along individually feasible coordinate steps.

    This is a refinement diagnostic, not an optimality certificate. Absolute
    diagonal curvature defines a positive local metric; nonlinear constraints
    are only linearized here. The optimizer must validate every actual trial.
    None means the diagnostic could not be evaluated, not that the fit is precise.
    """
    if (
        not np.isfinite(gradient).all()
        or not np.isfinite(diagonal).all()
        or not np.isfinite(jacobian).all()
        or not np.isfinite(slack).all()
        or np.any(slack < 0)
        or np.any(lower_step > 0)
        or np.any(upper_step < 0)
        or np.isnan(lower_step).any()
        or np.isnan(upper_step).any()
    ):
        return None
    curvature = np.abs(diagonal)
    if np.any(curvature == 0):
        return None
    try:
        with np.errstate(over="raise", invalid="raise", divide="raise"):
            step = np.clip(-gradient / curvature, lower_step, upper_step)
            change = jacobian * step
            fractions = np.ones_like(change)
            np.divide(slack[:, None], -change, out=fractions, where=change < 0)
            step *= np.minimum(1, fractions.min(axis=0, initial=1))
            gain = -gradient * step - 0.5 * curvature * step**2
            return float(np.max(gain, initial=0))
    except FloatingPointError:
        return None


def _projection_multipliers(gradient, normals, gaps):
    """Find a working set; only the independent dual bound certifies it."""
    point = np.zeros_like(gradient)
    active = []
    tolerance = (
        64
        * np.finfo(float).eps
        * max(1, gradient.size)
        * max(1.0, np.linalg.norm(gradient))
    )
    for _ in range(10 * (len(gaps) + gradient.size + 1)):
        score = point + gradient
        if active:
            q, r = np.linalg.qr(normals[active].T, mode="complete")
            if np.min(np.abs(np.diag(r))) < 64 * np.finfo(float).eps:
                return np.zeros_like(gradient), np.zeros(len(gaps))
            null = q[:, len(active) :]
            direction = -null @ (null.T @ score)
        else:
            direction = -score
        if np.linalg.norm(direction) <= tolerance:
            weights = (
                np.linalg.lstsq(normals[active].T, score, rcond=None)[0]
                if active
                else np.empty(0)
            )
            if weights.size and weights.min() < -tolerance:
                active.pop(int(np.argmin(weights)))
                continue
            dual = np.zeros(len(gaps))
            if active:
                rank = len(active)
                point = q[:, :rank] @ np.linalg.solve(r[:rank].T, -gaps[active])
                point -= null @ (null.T @ gradient)
                dual[active], _ = nnls(
                    normals[active].T, point + gradient, maxiter=3 * len(active)
                )
            return point, dual
        slack = normals @ point + gaps
        change = normals @ direction
        eligible = change < -64 * np.finfo(float).eps * max(
            1.0, np.linalg.norm(direction)
        )
        eligible[active] = False
        ratios = np.full(len(gaps), np.inf)
        ratios[eligible] = np.maximum(slack[eligible], 0) / -change[eligible]
        blocking = int(np.argmin(ratios))
        step = min(1.0, ratios[blocking])
        point += step * direction
        if step < 1:
            active.append(blocking)
    return np.zeros_like(gradient), np.zeros(len(gaps))


def projected_stationarity_bound(gradient, lower_step, upper_step, jacobian, slack):
    """Return an uncertified bound when certificate arithmetic is unrepresentable."""
    try:
        with np.errstate(over="raise", invalid="raise", divide="raise"):
            bound = _compute_projected_stationarity_bound(
                gradient, lower_step, upper_step, jacobian, slack
            )
        return bound if np.isfinite(bound) else np.inf
    except (FloatingPointError, np.linalg.LinAlgError):
        return np.inf


def _compute_projected_stationarity_bound(
    gradient, lower_step, upper_step, jacobian, slack
):
    """Bound the norm of the projection onto the linearized feasible domain.

    A feasible primal point and nonnegative dual vector bound the projection
    error through their primal-dual gap. An imperfect multiplier solve can
    cause a warning, never certify a point by discarding its constraint slack
    or a free gradient.
    """
    gradient = np.asarray(gradient, dtype=np.float64)
    if (
        not np.isfinite(gradient).all()
        or not np.isfinite(jacobian).all()
        or not np.isfinite(slack).all()
        or np.any(slack < 0)
        or np.any(lower_step > 0)
        or np.any(upper_step < 0)
        or np.isnan(lower_step).any()
        or np.isnan(upper_step).any()
    ):
        return np.inf
    # At the origin, outward forces on exactly active box faces have exact
    # nonnegative multipliers and zero complementarity. Their residual is zero
    # without subtracting large numbers; all other gradient components remain.
    supported = ((lower_step == 0) & (gradient > 0)) | (
        (upper_step == 0) & (gradient < 0)
    )
    residual = np.where(supported, 0.0, gradient)
    box_bound = float(np.hypot.reduce(residual, initial=0.0))
    if box_bound == 0:
        return 0.0
    box_bound = np.nextafter(
        box_bound * (1 + 8 * gradient.size * np.finfo(float).eps), np.inf
    )
    try:
        constrained = _linearized_projection_bound(
            gradient, lower_step, upper_step, jacobian, slack
        )
    except (FloatingPointError, np.linalg.LinAlgError, RuntimeError, ValueError):
        return box_bound
    return min(box_bound, constrained) if np.isfinite(constrained) else box_bound


def _linearized_projection_bound(gradient, lower_step, upper_step, jacobian, slack):
    """Try a tighter primal-dual certificate using every linearized constraint."""
    width = gradient.size
    identity = np.eye(width)
    low, high = np.isfinite(lower_step), np.isfinite(upper_step)
    normals = np.concatenate((jacobian, identity[low], -identity[high]))
    gaps = np.concatenate((slack, -lower_step[low], upper_step[high]))
    scales = np.max(np.abs(normals), axis=1, initial=0)
    nonzero = scales > 0
    normals = normals[nonzero] / scales[nonzero, None]
    gaps = gaps[nonzero] / scales[nonzero]
    if not len(gaps):
        return float(np.linalg.norm(gradient))
    # Inactive constraints may improve the multiplier fit but their full slack
    # still enters the independent certificate below.
    try:
        point, dual = _projection_multipliers(gradient, normals, gaps)
    except (RuntimeError, ValueError):
        return float(np.linalg.norm(gradient))
    extended = np.longdouble
    a, g, weights = (
        normals.astype(extended),
        gradient.astype(extended),
        dual.astype(extended),
    )
    point = point.astype(extended)
    gaps = gaps.astype(extended)
    movement = a @ point
    violating = movement + gaps < 0
    if np.any(violating):
        fraction = max(
            0.0, min(1.0, float(np.min(gaps[violating] / -movement[violating])))
        )
        point *= np.nextafter(fraction, 0.0)
    primal_slack = a @ point + gaps
    if np.any(primal_slack < 0):
        point.fill(0.0)
        primal_slack = gaps
    force = a.T @ weights
    residual = point + g - force
    complementarity = weights @ primal_slack
    bound = np.max(np.abs(point)) + np.sqrt(residual @ residual + 2 * complementarity)
    # Account for the summation error even where active forces nearly cancel.
    roundoff = (
        8
        * np.finfo(extended).eps
        * (1 + len(gaps))
        * (
            np.linalg.norm(np.abs(g) + np.abs(a).T @ weights)
            + np.sqrt(2 * abs(complementarity))
        )
    )
    certified_bound = float(bound + roundoff)
    return min(float(np.linalg.norm(gradient)), np.nextafter(certified_bound, np.inf))


def observed_covariance(hessian, *, stationary):
    """Invert information only when its positivity survives numerical error.

    Diagonal equilibration makes this check independent of parameter units.
    The eigenvalue resolution follows the dimension and machine precision;
    it is not a restriction on the model or its fitted parameters.
    """
    if not stationary or hessian is None or not np.isfinite(hessian).all():
        return None
    information = 0.5 * hessian + 0.5 * hessian.T
    diagonal = information.diagonal()
    if np.any(diagonal <= 0):
        return None
    scales = np.sqrt(diagonal)
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        normalized = information / scales[:, None] / scales[None, :]
    if not np.isfinite(normalized).all():
        return None
    try:
        spectrum = np.linalg.eigvalsh(normalized)
        resolution = len(spectrum) * np.finfo(float).eps * spectrum[-1]
        if spectrum[0] <= resolution:
            return None
        factor = np.linalg.cholesky(normalized)
        inverse = np.linalg.solve(factor, np.eye(len(factor))) / scales[None, :]
    except np.linalg.LinAlgError:
        return None
    with np.errstate(over="ignore", invalid="ignore"):
        covariance = inverse.T @ inverse
    return covariance if np.isfinite(covariance).all() else None
