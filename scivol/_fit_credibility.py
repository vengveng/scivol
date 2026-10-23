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
    stationarity check. A componentwise eight-epsilon allowance covers last-bit
    differences between request plans, without letting a large derivative hide
    disagreement in another component. Both stationarity checks remain separate;
    this allowance is not an accuracy bound for either gradient.
    """
    gradient_error = None
    gradient_excess = None
    gradient_roundoff = None
    roundoff_rtol = 8 * np.finfo(float).eps
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
                allowance = (
                    roundoff_rtol
                    * np.maximum(np.abs(gradient), np.abs(previous_gradient))
                    * np.abs(units)
                    / n_eff
                )
                gradient_roundoff = float(np.hypot.reduce(allowance, initial=0.0))
                gradient_excess = float(np.hypot.reduce(
                    np.maximum(np.abs(difference) - allowance, 0.0), initial=0.0
                ))
                if not np.isfinite([gradient_error, gradient_roundoff, gradient_excess]).all():
                    gradient_excess = np.inf
        except FloatingPointError:
            gradient_excess = np.inf
        if gradient_error is None or not np.isfinite(gradient_error):
            gradient_error = np.inf
    inconsistent = value_error > accuracy or (
        gradient_excess is not None and gradient_excess > gradient_tolerance
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
        "gradient_error_above_roundoff": gradient_excess,
        "gradient_roundoff_allowance": gradient_roundoff,
        "gradient_roundoff_rtol": float(roundoff_rtol),
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
    flat = curvature == 0
    if np.any(flat & (gradient != 0)):
        return None
    try:
        with np.errstate(over="raise", invalid="raise", divide="raise"):
            # A zero slope/curvature contributes no coordinate-model decrease;
            # it must not suppress the predictions for other coordinates.
            step = np.zeros_like(gradient, dtype=float)
            np.divide(-gradient, curvature, out=step, where=~flat)
            step = np.clip(step, lower_step, upper_step)
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
    """Use a box-only dual certificate if the constraint Jacobian is unavailable.

    This is an upper bound for the full problem, not the norm of its box-only
    projection. All constraint values must still establish a feasible origin.
    Unrepresentable certificate arithmetic returns an uncertified bound.
    """
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
        or (jacobian is not None and not np.isfinite(jacobian).all())
        or not np.isfinite(slack).all()
        or np.any(slack < 0)
        or np.any(lower_step > 0)
        or np.any(upper_step < 0)
        or np.isnan(lower_step).any()
        or np.isnan(upper_step).any()
    ):
        return np.inf
    box_bound = _box_projection_bound(gradient, lower_step, upper_step)
    if box_bound == 0 or jacobian is None:
        return box_bound
    try:
        constrained = _linearized_projection_bound(
            gradient, lower_step, upper_step, jacobian, slack
        )
    except (FloatingPointError, np.linalg.LinAlgError, RuntimeError, ValueError):
        return box_bound
    return min(box_bound, constrained) if np.isfinite(constrained) else box_bound


def _box_projection_bound(gradient, lower_step, upper_step):
    """Bound the projection by a zero-step dual certificate without cancellation.

    An opposing box force lambda=abs(g_i) cancels g_i exactly and contributes
    2*lambda*slack to the squared bound. A zero force contributes g_i**2 instead.
    Keep the cheaper certificate term; no nearly active slack is discarded.
    """
    magnitude = np.abs(gradient)
    distance = np.where(gradient > 0, -lower_step, upper_step)
    supported = np.isfinite(distance) & ((distance == 0) | (distance < magnitude / 2))
    terms = magnitude.copy()
    terms[supported] = (
        np.sqrt(magnitude[supported]) * np.sqrt(distance[supported]) * np.sqrt(2.0)
    )
    bound = float(np.hypot.reduce(terms, initial=0.0))
    return np.nextafter(
        bound * (1 + 8 * gradient.size * np.finfo(float).eps), np.inf
    ) if bound else 0.0


def _linearized_projection_bound(gradient, lower_step, upper_step, jacobian, slack):
    """Try a tighter primal-dual certificate using every linearized constraint."""
    width = gradient.size
    identity = np.eye(width)
    low, high = np.isfinite(lower_step), np.isfinite(upper_step)
    normals = np.concatenate((jacobian, identity[low], -identity[high]))
    gaps = np.concatenate((slack, -lower_step[low], upper_step[high]))
    scales = np.max(np.abs(normals), axis=1, initial=0)
    nonzero = scales > 0
    normals, gaps, scales = normals[nonzero], gaps[nonzero], scales[nonzero]
    if not len(gaps):
        return np.inf
    # Inactive constraints may improve the multiplier fit but their full slack
    # still enters the independent certificate below.
    try:
        point, dual = _projection_multipliers(
            gradient, normals / scales[:, None], gaps / scales
        )
        # Certify the original rows, not their rounded solver normalization.
        weights = np.asarray(dual, dtype=np.longdouble) / scales
        return _projection_certificate_bound(gradient, normals, gaps, point, weights)
    except (RuntimeError, TypeError, ValueError, OverflowError):
        return np.inf


def _projection_certificate_bound(
    gradient, normals, gaps, point, dual, *, accumulator=np.longdouble
):
    """Bound the infinity norm of P(-g) on the supplied domain A*x+b >= 0.

    For a feasible p and lambda >= 0, strong convexity gives
    ||p-P(-g)||_2**2 <= ||p+g-A.T*lambda||_2**2 + 2*lambda.T*(A*p+b).
    The origin-feasible domain also gives ||P(-g)||_2 <= ||g||_2. Neither
    certificate requires an optimal proposal. Retraction and outward error
    accounting follow _native/active_face_refinement.hpp::primal_dual_bound.
    The accumulator option permits binary64 regression coverage on wider hosts.
    Invalid proposals or unrepresentable arithmetic return an uncertified bound.
    """
    try:
        with np.errstate(over="raise", invalid="raise", divide="raise", under="ignore"):
            g, a, b, p, weights = (
                np.asarray(values, dtype=accumulator)
                for values in (gradient, normals, gaps, point, dual)
            )
            if (
                g.ndim != 1 or not g.size or p.shape != g.shape
                or b.ndim != 1 or weights.shape != b.shape
                or a.shape != (b.size, g.size)
                or not all(np.isfinite(values).all() for values in (g, a, b, p, weights))
                or np.any(b < 0) or np.any(weights < 0)
            ):
                return np.inf
            primal, upper_slack = _retract_projection_proposal(a, b, p)
            precision = np.finfo(g.dtype)
            eps, tiny = precision.eps, precision.smallest_subnormal
            rows, width = a.shape
            complementarity = np.sum(weights * upper_slack)
            complementarity = (
                complementarity * (1 + 8 * rows * eps) + 8 * rows * tiny
            )
            if not np.isfinite(complementarity) or complementarity < 0:
                return np.inf
            force_terms = a * weights[:, None]
            force = np.sum(force_terms, axis=0)
            residual_norm = np.hypot.reduce(primal + g - force, initial=0)
            force_scale = np.hypot.reduce(
                np.abs(primal) + np.abs(g) + np.sum(np.abs(force_terms), axis=0),
                initial=0,
            )
            slack_error = np.sqrt(complementarity) * np.sqrt(g.dtype.type(2))
            roundoff = 8 * (1 + rows + width) * (
                eps * (force_scale + slack_error) + tiny
            )
            bound = (
                np.max(np.abs(primal)) + np.hypot(residual_norm, slack_error) + roundoff
            )
            gradient_norm = np.hypot.reduce(g, initial=0)
            gradient_cap = gradient_norm * (1 + 8 * width * eps) + 8 * width * tiny
            if not np.isfinite([bound, gradient_cap]).all():
                return np.inf
            upper = np.nextafter(float(min(bound, gradient_cap)), np.inf)
            return upper if np.isfinite(upper) else np.inf
    except (FloatingPointError, TypeError, ValueError, OverflowError):
        return np.inf


def _retract_projection_proposal(normals, gaps, point):
    """Retract toward the feasible origin and return nonnegative upper slacks."""
    precision = np.finfo(point.dtype)
    eps, tiny = precision.eps, precision.smallest_subnormal

    def movement(fraction):
        primal = point * fraction
        terms = normals * primal
        change = np.sum(terms, axis=1)
        error = 8 * (1 + point.size) * (
            eps * (np.sum(np.abs(terms), axis=1) + gaps) + tiny
        )
        return primal, change, error

    _, change, error = movement(1)
    # Reserve a second dot-error allowance for the subsequent feasibility check.
    conservative_change = change - 2 * error
    violating = conservative_change + gaps < 0
    fraction = 1.0
    if np.any(violating):
        fraction = max(0.0, min(1.0, float(np.min(
            gaps[violating] / -conservative_change[violating]
        ))))
        fraction = np.nextafter(fraction, 0.0)
    primal, change, error = movement(fraction)
    if np.any(change - error + gaps < 0):
        primal, change, error = movement(0)
    # Slack error belongs inside the gap before sqrt, not in an O(eps) tail.
    upper_slack = np.nextafter(np.maximum(change + gaps + error, 0), np.inf)
    return primal, upper_slack


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
