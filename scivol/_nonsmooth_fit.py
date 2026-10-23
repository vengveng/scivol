"""Diagnostics and search proposals for linear residual kinks, not certificates.

These helpers never change the caller's estimate or smooth its objective.
The optimizer owns whether to refine an identified affine face or stop.
"""

from fractions import Fraction

import numpy as np


def neighboring_residual_proposals(model, point, *, candidates=4, hessian=None):
    """Reflect through nearby affine faces using the full joint Hessian metric.

    The positive metric ranks a bounded set of proposals; exact feasible jet
    values decide whether to refine them. This is not a global-optimality test.
    """
    if (isinstance(candidates, bool) or not np.isfinite(candidates)
            or int(candidates) != candidates or candidates < 1):
        raise ValueError("Expected a positive integer candidate count.")
    candidates = int(candidates)
    if not model.columns:
        return dict(status="no affine residual switches", proposals=[])
    point = np.asarray(point, dtype=float)
    hessian = model.evaluate(point, 4)[2] if hessian is None else np.asarray(hessian)
    scale = model.units
    metric = hessian * scale[:, None] * scale[None, :]
    if not np.isfinite(metric).all():
        return dict(status="nonfinite metric", proposals=[])
    try:
        factor = np.linalg.cholesky(metric)
    except np.linalg.LinAlgError:
        return dict(status="indefinite metric", proposals=[])
    normals = np.zeros((len(model.data)-1, model.size))
    normals[:, :model.columns] = model.features[:-1] * scale[:model.columns]
    dual = np.linalg.solve(factor.T, np.linalg.solve(factor, normals.T)).T
    denominator = np.einsum("ij,ij->i", normals, dual)
    residual = model.data[:-1] - model.features[:-1] @ point[:model.columns]
    barrier = np.divide(residual**2, 2*denominator, out=np.full(len(residual), np.inf),
                        where=denominator > 0)
    proposals = []
    for row in np.argsort(barrier, kind="stable")[:candidates]:
        if not np.isfinite(barrier[row]):
            continue
        direction = 2*residual[row]*dual[row]/denominator[row]*scale
        trial = point + direction
        record = dict(row=int(row), predicted_barrier=float(barrier[row]),
            theta=trial.tolist(), feasible=bool(model.assess(trial, None)[0]),
            original_residual=float(residual[row]),
            proposed_residual=float(model.data[row]-model.features[row]@trial[:model.columns]))
        if record["feasible"]:
            try:
                record["nll"] = model.evaluate(trial, 1)[0]
            except ValueError as error:
                record["error"] = str(error)
        proposals.append(record)
    return dict(status="positive metric", proposals=proposals)


def bracket_residual_face(model, theta, *, row=None, radius=1, _base=None):
    """Evaluate smooth neighbors of one exact rational residual-zero anchor.

    The mean is linear in the first ``model.columns`` parameters. The last
    residual is excluded because it has no successor shock in the likelihood.
    Report the anchor displacement separately from the neighbor radius. Neither
    nearby gradients nor their convex combination certify the returned doubles.
    """
    if not model.columns or radius not in (1, 2):
        raise ValueError("A residual bracket needs mean parameters and radius 1 or 2.")
    theta = np.array(theta, dtype=float, copy=True)
    value, _, _, residuals, _ = model.evaluate(theta, 1, paths=True) if _base is None else _base
    with np.errstate(over="ignore", invalid="ignore"):
        normals = model.features[:-1] * model.units[:model.columns]
    if not np.isfinite(normals).all():
        raise ValueError("Residual-face scaling is not representable.")
    sensitivity = np.max(abs(normals), axis=1)
    distances = np.divide(
        abs(residuals[:-1]), sensitivity,
        out=np.full(len(sensitivity), np.inf), where=sensitivity > 0,
    )
    row = int(np.argmin(distances)) if row is None else int(row)
    if not 0 <= row < len(model.data)-1 or not np.isfinite(distances[row]):
        raise ValueError("No effective affine shock face is available.")
    normal = model.features[row]
    pivot = int(np.argmax(abs(normals[row])))
    exact = lambda number: Fraction(float(number))
    anchor = (exact(model.data[row]) - sum(
        exact(a)*exact(b) for j, (a, b) in enumerate(zip(normal, theta)) if j != pivot
    )) / exact(normal[pivot])
    rounded = float(anchor)
    if not np.isfinite(rounded):
        raise ValueError("The residual-zero anchor is not representable.")
    neighbors = []
    for direction in (-1, 1):
        coordinate = rounded
        if direction * (exact(coordinate)-anchor) <= 0:
            coordinate = np.nextafter(coordinate, direction*np.inf)
        for _ in range(radius-1):
            coordinate = np.nextafter(coordinate, direction*np.inf)
        point = theta.copy()
        point[pivot] = coordinate
        if not model.assess(point, None)[0]:
            raise ValueError("A residual-face neighbor is outside the model domain.")
        f, g, _, e, _ = model.evaluate(point, 3, paths=True)
        feasible, bound, violation = model.assess(point, g)
        neighbors.append(dict(theta=point, nll=f, gradient=g, residuals=e,
                              feasible=feasible, kkt=bound, violation=violation))
    lower, upper = neighbors
    changed = np.flatnonzero(np.signbit(lower["residuals"]) != np.signbit(upper["residuals"]))
    with np.errstate(over="ignore", invalid="ignore"):
        scale = model.units / len(model.data)
        a, b = lower["gradient"]*scale, upper["gradient"]*scale
    if not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError("Neighbor-gradient scaling is not representable.")
    magnitude = max(np.max(abs(a)), np.max(abs(b)), np.finfo(float).tiny)
    jump = b/magnitude - a/magnitude
    square = float(jump @ jump)
    weight = float(np.clip(-(a/magnitude @ jump)/square, 0, 1)) if square else 0.0
    mixed = (1-weight)*lower["gradient"] + weight*upper["gradient"]
    feasible, bound, violation = model.assess(theta, mixed)
    full_normal = np.r_[normals[row]/sensitivity[row], np.zeros(model.size-model.columns)]
    tangent_jump = jump - full_normal * (full_normal @ jump)/(full_normal @ full_normal)
    return dict(
        classification="neighbor-gradient evidence; not a stationarity or inference certificate",
        certified=False, row=row, pivot=pivot, radius=radius, anchor=str(anchor),
        backward_error=float(abs(anchor-exact(theta[pivot])))/model.units[pivot],
        neighborhood_radius=max(float(abs(exact(p["theta"][pivot])-anchor))
                                for p in neighbors)/model.units[pivot],
        nll=value, maximum_value_discrepancy=max(abs(p["nll"]-value) for p in neighbors),
        single_face=bool(np.array_equal(changed, [row])),
        feasible=bool(feasible and all(p["feasible"] for p in neighbors)),
        weight=weight, mixed_kkt=bound, violation=violation,
        coordinate_slopes=[float(a[pivot]), float(b[pivot])],
        normal_minimum=bool(a[pivot] < 0 < b[pivot]),
        tangent_jump=float(np.max(abs(tangent_jump)))*magnitude,
        neighboring_kkt=[lower["kkt"], upper["kkt"]],
        gradients=[lower["gradient"], upper["gradient"]], mixed_gradient=mixed,
    )


def residual_face_evidence(model, theta, rows, accuracy, *, multiple=False):
    """Check stable neighboring gradients without certifying the returned point."""
    if not multiple and len(rows) != 1:
        raise ValueError("Single-face evidence requires one row.")
    theta = np.array(theta, dtype=float, copy=True)
    # This base request belongs only to these two radii at the same model point.
    base = model.evaluate(theta, 1, paths=True)
    if not multiple:
        probes = [bracket_residual_face(model, theta, row=rows[0], radius=r, _base=base) for r in (1, 2)]
        pairs = zip(probes[0]["gradients"], probes[1]["gradients"])
        face_key = "single_face"
    else:
        probes = [bracket_residual_faces(model, theta, rows, radius=r, _base=base) for r in (1, 2)]
        pairs = [(probes[0][key], probes[1][key]) for key in ("smooth_gradient", "face_gradients")]
        face_key = "face_set_exact"
    stability = max(float(np.max(abs(a-b)*model.units/len(model.data))) for a, b in pairs)
    tolerance = max(np.sqrt(accuracy), np.sqrt(np.finfo(float).eps))
    backward_limit = 8*np.finfo(float).eps*max(1.0, np.max(abs(theta/model.units)))
    supported = all(
        p["feasible"] and p[face_key] and p["normal_minimum"]
        and p["mixed_kkt"] <= tolerance and p["tangent_jump"] <= tolerance
        and p["maximum_value_discrepancy"] <= accuracy*len(model.data)
        and p["backward_error"] <= backward_limit for p in probes
    ) and stability <= tolerance
    omitted = ("gradients", "mixed_gradient", "smooth_gradient", "face_gradients")
    evidence = {key: value for key, value in probes[0].items() if key not in omitted}
    return dict(evidence, gradient_stability=stability, diagnostic_threshold=tolerance,
                supports_kink_stop=bool(supported))


def _solve_rational(matrix, rhs):
    """Solve a square rational system exactly, or raise when it is singular."""
    k = len(rhs)
    rows = [list(matrix[i]) + [rhs[i]] for i in range(k)]
    for column in range(k):
        pivot = next((i for i in range(column, k) if rows[i][column] != 0), None)
        if pivot is None:
            raise ValueError("The active residual faces are linearly dependent.")
        rows[column], rows[pivot] = rows[pivot], rows[column]
        for i in range(k):
            if i != column and rows[i][column] != 0:
                ratio = rows[i][column] / rows[column][column]
                rows[i] = [a - ratio*b for a, b in zip(rows[i], rows[column])]
    return [rows[i][k] / rows[i][i] for i in range(k)]


def residual_face_distances(model, theta):
    """Scaled distances of every residual with a successor shock from its face.

    The mean is linear, so the residuals need no likelihood request.
    """
    theta = np.asarray(theta, dtype=float)
    residuals = model.data - model.features @ theta[:model.columns]
    with np.errstate(over="ignore", invalid="ignore"):
        normals = model.features[:-1] * model.units[:model.columns]
    if not np.isfinite(normals).all():
        raise ValueError("Residual-face scaling is not representable.")
    sensitivity = np.max(abs(normals), axis=1)
    return np.divide(abs(residuals[:-1]), sensitivity,
                     out=np.full(len(sensitivity), np.inf), where=sensitivity > 0)


def bracket_residual_faces(model, theta, rows, *, radius=1, _base=None):
    """Evaluate smooth neighbors of an exact rational point on several residual faces.

    The linear mean makes each kink e_r(b) = 0 an affine hyperplane. Near a
    point on the faces ``rows`` the objective is, to first order,
    ``F + g0'd + sum_i kappa_i |n_i'd|``. Under linear independence of the
    active normals (checked exactly here), first-order minimality is
    tangential stationarity plus normal growth: every ``kappa_i > 0`` and some
    weights in [-1, 1] make ``g0 + sum_i w_i kappa_i n_i`` stationary
    (Griewank and Walther 2016, Optim. Methods Softw. 31(5)). The anchor solves
    the active equalities exactly in rationals by adjusting pivot coefficients.
    Neighbors put each active residual on a chosen side with the smallest
    tested representable displacement. Nothing here certifies the returned doubles.
    """
    rows = [int(row) for row in rows]
    k = len(rows)
    if not model.columns or radius not in (1, 2) or not 1 <= k <= model.columns:
        raise ValueError("Residual faces need mean parameters, radius 1 or 2 and 1..columns rows.")
    if len(set(rows)) != k or any(not 0 <= row < len(model.data)-1 for row in rows):
        raise ValueError("Residual faces must be distinct rows with successor shocks.")
    theta = np.array(theta, dtype=float, copy=True)
    n = len(model.data)
    columns = model.columns
    value, _, _, residuals, _ = model.evaluate(theta, 1, paths=True) if _base is None else _base
    with np.errstate(over="ignore", invalid="ignore"):
        normals = model.features[rows] * model.units[:columns]
    if not np.isfinite(normals).all():
        raise ValueError("Residual-face scaling is not representable.")
    # Column-pivoted Gram-Schmidt chooses well-conditioned pivot coefficients.
    pivots, basis = [], []
    for _ in range(k):
        remainder = normals.copy()
        for q in basis:
            remainder -= np.outer(q, q @ remainder)
        norms = np.linalg.norm(remainder, axis=0)
        norms[pivots] = -1.0
        column = int(np.argmax(norms))
        if norms[column] <= 0:
            raise ValueError("The active residual faces are linearly dependent.")
        pivots.append(column)
        vector = remainder[:, column]
        basis.append(vector / np.linalg.norm(vector))
    exact = lambda number: Fraction(float(number))
    features = [[exact(v) for v in model.features[row]] for row in rows]
    mean = [exact(v) for v in theta[:columns]]
    fixed = [j for j in range(columns) if j not in pivots]
    rhs = [exact(model.data[row]) - sum(features[i][j]*mean[j] for j in fixed)
           for i, row in enumerate(rows)]
    matrix = [[features[i][j] for j in pivots] for i in range(k)]
    anchor = _solve_rational(matrix, rhs)
    if not all(np.isfinite(float(a)) for a in anchor):
        raise ValueError("The residual-zero anchor is not representable.")

    def exact_residuals(point):
        return [exact(model.data[row]) - sum(f*exact(b) for f, b in zip(features[i], point[:columns]))
                for i, row in enumerate(rows)]

    def neighbor(signs):
        spacing = max(float(np.spacing(abs(float(a)))) for a in anchor)
        scale = max(float(np.max(abs(model.features[rows]))), np.finfo(float).tiny)
        # Form the product exactly: a valid subnormal spacing can underflow when
        # multiplied by a small design coefficient in binary64 first.
        step = radius * exact(spacing) * exact(scale)
        for _ in range(64):
            delta = _solve_rational(matrix, [-s*step for s in signs])
            point = theta.copy()
            for j, a, d in zip(pivots, anchor, delta):
                point[j] = float(a + d)
            if all((r > 0) == (s > 0) and r != 0 for r, s in zip(exact_residuals(point), signs)):
                return point
            step *= 2
        raise ValueError("No representable neighbor separates the residual faces.")

    patterns = [np.ones(k)] + [np.where(np.arange(k) == i, -1.0, 1.0) for i in range(k)]
    neighbors = []
    for signs in patterns:
        point = neighbor(signs)
        if not model.assess(point, None)[0]:
            raise ValueError("A residual-face neighbor is outside the model domain.")
        f, g, _, e, _ = model.evaluate(point, 3, paths=True)
        if not np.array_equal(np.sign(e[rows]), signs):
            raise ValueError("Jet residual signs do not resolve the requested faces.")
        feasible, bound, violation = model.assess(point, g)
        neighbors.append(dict(theta=point, signs=signs, nll=f, gradient=g, residuals=e,
                              feasible=feasible, kkt=bound, violation=violation))
    others = np.setdiff1d(np.arange(n), rows)
    reference = np.signbit(residuals[others])
    face_set_exact = all(np.array_equal(np.signbit(p["residuals"][others]), reference)
                         for p in neighbors)
    scale = model.units / n
    with np.errstate(over="ignore", invalid="ignore"):
        scaled = [p["gradient"] * scale for p in neighbors]
    if not all(np.isfinite(a).all() for a in scaled):
        raise ValueError("Neighbor-gradient scaling is not representable.")
    face_normals = np.zeros((k, model.size))
    face_normals[:, :columns] = -normals
    kappas = np.empty(k)
    tangent_jump = 0.0
    for i in range(k):
        jump = scaled[0] - scaled[i + 1]
        kappas[i] = 0.5 * float(jump @ face_normals[i]) / float(face_normals[i] @ face_normals[i])
        tangent_jump = max(tangent_jump, float(np.max(abs(jump - 2*kappas[i]*face_normals[i]))))
    smooth = scaled[0] - kappas @ face_normals
    normal_minimum = bool(np.all(kappas > 0))
    weights = np.full(k, np.nan)
    mixed_kkt, violation, feasible = np.inf, np.inf, False
    mixed = None
    if normal_minimum:
        from scipy.optimize import lsq_linear

        design = (kappas[:, None] * face_normals).T
        weights = lsq_linear(design, -smooth, bounds=(-1.0, 1.0), method="bvls").x
        mixed = (smooth + (kappas * weights) @ face_normals) / scale
        feasible, mixed_kkt, violation = model.assess(theta, mixed)
    backward = max(float(abs(a - exact(theta[j]))) / model.units[j]
                   for j, a in zip(pivots, anchor))
    radius_used = max(float(abs(exact(p["theta"][j]) - a)) / model.units[j]
                      for p in neighbors for j, a in zip(pivots, anchor))
    return dict(
        classification="multi-face neighbor-gradient evidence; not a stationarity or inference certificate",
        certified=False, rows=rows, pivots=pivots, radius=radius,
        anchor=[str(a) for a in anchor], backward_error=backward,
        neighborhood_radius=radius_used, nll=value,
        maximum_value_discrepancy=max(abs(p["nll"] - value) for p in neighbors),
        face_set_exact=bool(face_set_exact),
        feasible=bool(feasible and all(p["feasible"] for p in neighbors)),
        kappas=kappas.tolist(), weights=weights.tolist(),
        saturated=[bool(abs(w) >= 1.0) for w in weights],
        mixed_kkt=mixed_kkt, violation=violation, normal_minimum=normal_minimum,
        tangent_jump=tangent_jump, neighboring_kkt=[p["kkt"] for p in neighbors],
        smooth_gradient=smooth / scale, face_gradients=(kappas[:, None]*face_normals) / scale,
        gradients=[p["gradient"] for p in neighbors],
        patterns=[p["signs"].tolist() for p in neighbors], mixed_gradient=mixed,
    )
