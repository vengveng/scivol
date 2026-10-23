"""Joint linear-mean jets and native defaults for all supported families."""

from __future__ import annotations

from copy import deepcopy
import time
import numpy as np
from scipy.optimize import OptimizeResult

from . import _native_core
from ._active_face_fit import refine_before_restart, refinement_chart_available
from ._mean_fit import PreparedMean, DENSITIES, _pointer, _workspace
from ._native_fit import (
    _aparch_volatility_domain,
    _aparch_persistence_evidence,
    _describe_model,
    _coordinate_bounds,
    _coordinates_to_theta,
    _density_domain,
    _gjr_volatility_domain,
)
from ._fit_credibility import (
    projected_stationarity_bound,
    observed_covariance,
    diagonal_descent_prediction,
    jet_agreement,
)
from .components.mean import ARX, HARX
from .components.vol import GARCH, GJRGARCH, EGARCH, APARCH
from .roles import Role

FAMILIES = (GARCH, GJRGARCH, EGARCH, APARCH)
EGARCH_DEFAULT_MAXITER = 1800
APARCH_PORTFOLIO_MAXITER = 3200
_JET_MASK_NAMES = ("", "F", "G", "F|G", "H", "F|H", "G|H", "F|G|H")


def _describe_jet_requests(counts, before=None):
    counts = [
        int(a - b) for a, b in zip(counts, (0,) * 8 if before is None else before)
    ]
    return {
        **{
            name: sum(count for mask, count in enumerate(counts) if mask & bit)
            for bit, name in ((1, "F"), (2, "G"), (4, "H"))
        },
        "masks": dict(zip(_JET_MASK_NAMES[1:], counts[1:])),
    }


def supports_joint_mean(spec):
    return (
        isinstance(spec.get_component(Role.MEAN), (ARX, HARX))
        and type(spec.get_component(Role.VOLATILITY)) in FAMILIES
        and type(spec.get_component(Role.DENSITY)) in DENSITIES
    )


def uses_joint_native_by_default(spec):
    return supports_joint_mean(spec) and type(spec.get_component(Role.VOLATILITY)) in (
        GARCH,
        GJRGARCH,
        EGARCH,
        APARCH,
    )


class JointMeanModel(PreparedMean):
    """One sample/design and one full parameter vector for every jet request."""

    def __init__(self, spec, data, x=None, *, fixed=True):
        if not supports_joint_mean(spec):
            raise NotImplementedError(
                "Expected ARX/HARX with a supported volatility and density."
            )
        super().__init__(spec, data, x)
        self.volatility = self.spec.get_component(Role.VOLATILITY)
        self.vol = _describe_model(self.volatility + self.density)
        self.family = FAMILIES.index(type(self.volatility))
        self.p_vol, self.q_vol = self.volatility.p, self.volatility.q
        self.size = self.columns + self.vol.n_params
        self.shape_start = self.columns + self.vol.n_vol
        self.h0 = float(np.mean(self.data**2))
        if not np.isfinite(self.h0) or self.h0 <= 0:
            raise ValueError(
                "The conditioning variance must be positive and representable; rescale the data."
            )
        self.descriptor = (
            self.family,
            self.code,
            self.columns,
            self.p_vol,
            self.q_vol,
            len(self.data),
            bool(fixed),
        )
        self.domain = (
            (
                self.blocks[0].order,
                self.blocks[0].width,
                self.blocks[0].offset,
                _pointer(self.blocks[0].basis),
            )
            if self.blocks
            else (0, 0, 0, 0)
        )
        extra = (1, 1 + self.p_vol, 2 * self.q_vol, 1 + int(self.code in (1, 2)))[
            self.family
        ]
        self.rows = 2 * self.domain[0] + extra
        self.workspaces = {}
        self.request_counts = [0] * 8
        self.lower, self.upper = self.theta_bounds()
        self.units = np.r_[self.mean_units, np.ones(self.vol.n_params)]

    def workspace(self, request, transformed=False, *, solve=False, affine=False, restricted=False):
        if affine and not solve:
            raise ValueError("An affine phase needs optimizer workspace.")
        if restricted and (not solve or self.family < 2):
            raise ValueError("Coordinate subsets require an EGARCH/APARCH optimizer workspace.")
        key = (request, bool(transformed), bool(solve), int(affine))
        if restricted:
            key += ("coordinate-subset",)
        if key not in self.workspaces:
            if restricted:
                size = _native_core._linear_joint_restriction_bytes(
                    self.descriptor, self.domain, transformed, int(affine)
                )
            else:
                sizes = _native_core._linear_joint_sizes(
                    self.descriptor, request, transformed, self.domain, int(affine)
                )
                size = sizes[int(solve)]
            self.workspaces[key] = _workspace(size)
        return self.workspaces[key]

    def theta_bounds(self):
        # Density bounds come from the same native domain as the standalone jet.
        lower, upper = np.full(self.size, np.nan), np.full(self.size, np.nan)
        density_low, density_high = np.empty(3 + self.density.n_params), np.empty(
            3 + self.density.n_params
        )
        _native_core._garch_theta_bounds(
            _pointer(density_low), _pointer(density_high), 1, 1, self.code, False
        )
        lower[self.shape_start :], upper[self.shape_start :] = (
            density_low[3:],
            density_high[3:],
        )
        m, p, q = self.columns, self.p_vol, self.q_vol
        if self.family != 2:
            lower[m] = np.nextafter(0.0, 1.0)
            lower[m + 1 : m + 1 + p] = 0.0
            beta = m + 1 + p * (1 if self.family == 0 else 2)
            lower[beta : beta + q] = 0.0
            if self.family == 3:
                lower[m + 1 + p : beta] = np.nextafter(-1.0, 0.0)
                upper[m + 1 + p : beta] = np.nextafter(1.0, 0.0)
                lower[beta + q] = np.nextafter(0.0, 1.0)
        return lower, upper

    def evaluate(self, theta, request=7, *, transformed=False, paths=False):
        theta = np.ascontiguousarray(theta, dtype=np.float64)
        if theta.shape != (self.size,):
            raise ValueError("Wrong number of joint parameters.")
        work = self.workspace(request, transformed)
        value = np.empty(1) if request & 1 else None
        gradient = np.empty(self.size) if request & 2 else None
        hessian = np.empty((self.size, self.size)) if request & 4 else None
        residuals, variances = (
            (np.empty_like(self.data), np.empty_like(self.data))
            if paths
            else (None, None)
        )
        self.request_counts[request] += 1
        valid = _native_core._linear_joint_prevalidated(
            self.descriptor,
            _pointer(theta),
            _pointer(self.data),
            _pointer(self.features),
            self.h0,
            request,
            transformed,
            _pointer(work),
            work.nbytes,
            _pointer(value),
            _pointer(gradient),
            _pointer(hessian),
            _pointer(residuals),
            _pointer(variances),
        )
        if not valid:
            raise ValueError(
                f"Joint jet request {request} is invalid at these parameters."
            )
        return (
            None if value is None else float(value[0]),
            gradient,
            hessian,
            residuals,
            variances,
        )

    def inverse(self, theta):
        theta = np.ascontiguousarray(theta, dtype=np.float64)
        if theta.shape != (self.size,):
            raise ValueError("Wrong number of joint parameters.")
        work, z = self.workspace(3, True), np.empty_like(theta)
        if not _native_core._linear_joint_inverse(
            self.descriptor, _pointer(theta), _pointer(z), _pointer(work), work.nbytes
        ):
            raise ValueError("The joint start is outside the transformed chart.")
        return z

    def constraints(self, theta, *, derivatives=True):
        theta = np.ascontiguousarray(theta, dtype=np.float64)
        if theta.shape != (self.size,) or not np.isfinite(theta).all():
            raise ValueError("Invalid joint parameter vector.")
        values = np.empty(self.rows)
        work = self.workspace(3, False, solve=True)
        values_only = getattr(_native_core, "_linear_joint_constraint_values", None)
        if not derivatives and values_only is not None:
            valid = values_only(
                self.descriptor,
                self.domain,
                _pointer(theta),
                _pointer(values),
                _pointer(work),
                work.nbytes,
            )
            jacobian = None
        else:
            # Older compiled extensions expose only the combined request.
            jacobian = np.empty((self.rows, self.size), order="F")
            valid = _native_core._linear_joint_constraints(
                self.descriptor,
                self.domain,
                _pointer(theta),
                _pointer(values),
                _pointer(jacobian),
                _pointer(work),
                work.nbytes,
            )
        if not valid:
            raise ValueError("Joint domain evaluation failed.")
        return values, jacobian if derivatives else None

    def assess(self, theta, gradient, *, stationarity_solver=None):
        """Keep domain checks and parameter units independent of the bound backend."""
        try:
            try:
                slack, jacobian = self.constraints(
                    theta, derivatives=gradient is not None
                )
            except ValueError:
                if gradient is None:
                    raise
                slack, jacobian = self.constraints(theta, derivatives=False)
            lower = np.where(np.isnan(self.lower), -np.inf, self.lower)
            upper = np.where(np.isnan(self.upper), np.inf, self.upper)
            strict = np.ones(self.rows, dtype=bool)
            if self.family == 1:
                strict[2 * self.domain[0] : -1] = False
            domain_slack = slack.copy()
            if self.family == 0:
                domain_slack[-1] = 1 - np.sum(
                    theta[self.columns + 1 : self.shape_start]
                )
            elif self.family == 1:
                _, _, persistence = _gjr_volatility_domain(
                    self.vol, theta[self.columns :]
                )
                domain_slack[-1] = 1 - persistence
            feasible = bool(
                np.all(theta >= lower)
                and np.all(theta <= upper)
                and np.all(domain_slack[strict] > 0)
                and np.all(domain_slack[~strict] >= 0)
                and _density_domain(self.code, theta[self.shape_start :])[1]
            )
            violation = max(
                0.0,
                float(np.max(lower - theta)),
                float(np.max(theta - upper)),
                float(np.max(-domain_slack)),
            )
            if not feasible or gradient is None or (jacobian is None and stationarity_solver is not None):
                return feasible, np.inf, violation
        except (ValueError, FloatingPointError):
            return False, np.inf, np.inf
        try:
            with np.errstate(over="raise", invalid="raise", divide="raise"):
                units = self.units.copy()
                if self.family != 2:
                    units[self.columns] = self.h0 ** (
                        theta[self.shape_start - 1] / 2 if self.family == 3 else 1
                    )
                assess_stationarity = (
                    projected_stationarity_bound
                    if stationarity_solver is None
                    else stationarity_solver
                )
                bound = assess_stationarity(
                    gradient * units / len(self.data),
                    (lower - theta) / units,
                    (upper - theta) / units,
                    None if jacobian is None else jacobian * units,
                    slack,
                )
                if not np.isfinite(bound) or bound < 0:
                    bound = np.inf
            return True, bound, violation
        except (ValueError, FloatingPointError, OverflowError):
            return True, np.inf, violation

    def predicted_coordinate_gain(self, theta, gradient, hessian):
        """Check weak directions before accepting first-order termination."""
        try:
            slack, jacobian = self.constraints(theta)
            lower = np.where(np.isnan(self.lower), -np.inf, self.lower)
            upper = np.where(np.isnan(self.upper), np.inf, self.upper)
            return diagonal_descent_prediction(
                gradient,
                hessian.diagonal(),
                lower - theta,
                upper - theta,
                jacobian,
                slack,
            )
        except (ValueError, FloatingPointError):
            return None

    def scipy_inequalities(self):
        """Expose the same model-owned domain to an independent SciPy solver."""
        return dict(
            type="ineq",
            fun=lambda theta: self.constraints(theta)[0],
            jac=lambda theta: self.constraints(theta)[1],
        )

    def initialize(self, initial_params=None):
        if initial_params is not None:
            theta = np.array(initial_params, dtype=np.float64, order="C", copy=True)
        else:
            theta = np.r_[
                self.mean.default_start(self.data),
                self.volatility.default_start(self.data),
                self.density.default_start(self.data),
            ]
        self.evaluate(theta, 1)
        if not self.assess(theta, None)[0]:
            raise ValueError("Initial parameters violate the joint model domain.")
        return theta

    def describe_domain(self):
        """Report this plan's domain and conditioning, not optimizer tolerances."""
        restrictions = (
            "omega > 0; alpha,beta >= 0; sum(alpha)+sum(beta) < 1",
            "omega > 0; alpha,beta >= 0; alpha+gamma >= 0; "
            "sum(alpha)+sum(gamma)*E[z^2 I(z<0)]+sum(beta) < 1",
            "stationary implied beta polynomial; no invertibility padding",
            "omega,delta > 0; alpha,beta >= 0; abs(gamma) < 1; "
            "sum(alpha_i*E[(abs(z)-gamma_i*z)^delta])+sum(beta) < 1",
        )
        volatility = restrictions[self.family]
        if self.family == 3 and self.code in (1, 2):
            volatility += "; delta < nu for a finite power moment"
        return {
            "mean": (
                "stationary implied AR polynomial"
                if self.blocks
                else "no autoregressive restriction"
            ),
            "mean_polynomial_order": self.domain[0],
            "mean_coefficient_count": self.domain[1],
            "volatility": volatility,
            "volatility_orders": [self.p_vol, self.q_vol],
            "density": self.density.signature,
            "density_support": "validated native numerical support, not convergence padding",
            "numerical_bound_units": "theta for the prepared fitting sample; see fit_info.scale",
            "numerical_lower": [
                None if not np.isfinite(v) else float(v) for v in self.lower
            ],
            "numerical_upper": [
                None if not np.isfinite(v) else float(v) for v in self.upper
            ],
            "conditioning": "supplied linear design; variance seeded by mean(y^2) on the prepared sample",
            "initial_variance": self.h0,
            "first_scored": self.first,
            "scored_observations": len(self.data) - self.first,
            "kkt_metric": "original parameters in dimensionless measurement units",
        }


def _limit_scales_to_box(scales, lower, upper):
    """Bound a local metric by finite box widths without changing model bounds."""
    with np.errstate(over="ignore", invalid="ignore"):
        width = upper - lower
    finite = np.isfinite(width) & (width > 0)
    limited = scales.copy()
    # frexp handles widths just below powers of two without log2 rounding up.
    cap = np.ldexp(np.ones(np.count_nonzero(finite)), np.frexp(width[finite])[1] - 1)
    limited[finite] = np.minimum(limited[finite], cap)
    return limited


def _initial_joint_coordinates(model, initial, transformed, original):
    if original is None:
        return model.inverse(initial) if transformed else initial.copy()
    if not transformed:
        raise ValueError("Original joint coordinates require a transformed phase.")
    coordinates = np.array(original, dtype=np.float64, order="C", copy=True)
    if coordinates.shape != (model.size,) or not np.isfinite(coordinates).all():
        raise ValueError("Expected finite original joint coordinates with the model width.")
    mapped = np.r_[coordinates[:model.columns],
        _coordinates_to_theta(model.vol, coordinates[model.columns:])]
    if not np.array_equal(mapped, initial):
        raise ValueError("Original joint coordinates do not map to the supplied theta.")
    return coordinates


def _solve_joint_phase(
    model,
    initial,
    *,
    transformed,
    accuracy,
    maxiter,
    use_curvature=True,
    limit_curvature_scale=False,
    speculate=None,
    affine=None,
    curvature_reset=None,
    invalid_trial_policy="interpolate",
    free_coordinates=None,
    stationarity_tolerance=None,
    initial_coordinates=None,
):
    if stationarity_tolerance is None:
        stationarity_tolerance = max(np.sqrt(accuracy), np.sqrt(np.finfo(float).eps))
    if not np.isfinite(stationarity_tolerance) or stationarity_tolerance <= 0:
        raise ValueError("Expected a finite positive stationarity tolerance.")
    if invalid_trial_policy not in ("interpolate", "minimum-step"):
        raise ValueError("Expected interpolate or minimum-step invalid-trial policy.")
    indices = None
    if free_coordinates is not None:
        indices = np.asarray(free_coordinates)
        if (model.family < 2 or indices.ndim != 1 or not len(indices)
                or indices.dtype.kind not in "iu" or np.any(indices < 0)
                or np.any(indices >= model.size) or np.any(indices[1:] <= indices[:-1])):
            raise ValueError("Expected distinct increasing EGARCH/APARCH optimizer coordinate indices.")
        indices = np.ascontiguousarray(indices, dtype=np.uintp)
    requests_before = tuple(model.request_counts)
    speculate = model.family >= 2 if speculate is None else bool(speculate)
    # The optional reset can discard a useful SQP metric; current APARCH
    # controls preserve quality with fewer requests when it remains opt-in.
    curvature_reset = False if curvature_reset is None else bool(curvature_reset)
    normal, target, targets, affine_rows = None, 0.0, None, 0
    if affine is not None:
        normal = np.ascontiguousarray(affine[0], dtype=np.float64)
        targets = np.ascontiguousarray(np.atleast_1d(affine[1]), dtype=np.float64)
        affine_rows = normal.shape[0] if normal.ndim == 2 else int(normal.ndim == 1)
        if (
            model.family < 2
            or normal.shape not in ((model.columns,), (affine_rows, model.columns))
            or not 1 <= affine_rows <= model.columns
            or targets.shape != (affine_rows,)
            or not np.isfinite(normal).all()
            or not np.all(np.any(np.atleast_2d(normal), axis=1))
            or not np.isfinite(targets).all()
        ):
            raise ValueError(
                "Expected finite nonzero affine EGARCH/APARCH mean constraints."
            )
        target = float(targets[0])
        if affine_rows == 1:
            normal = normal.reshape(model.columns)
    coordinates = _initial_joint_coordinates(model, initial, transformed, initial_coordinates)
    scales = np.ones(model.size)
    curvature_prepared = False
    if use_curvature:
        try:
            hessian = model.evaluate(coordinates, 4, transformed=transformed)[2] / len(
                model.data
            )
            scales = np.exp2(
                np.rint(
                    -0.5
                    * np.log2(np.maximum(abs(hessian.diagonal()), np.finfo(float).eps))
                )
            )
            curvature_prepared = True
        except ValueError:
            pass
    if transformed:
        low, high = _coordinate_bounds(model.vol)
        lower, upper = (
            np.r_[np.full(model.columns, np.nan), low],
            np.r_[np.full(model.columns, np.nan), high],
        )
    else:
        lower, upper = model.lower.copy(), model.upper.copy()
    scale_limits = []
    if curvature_prepared and limit_curvature_scale:
        limited = _limit_scales_to_box(scales, lower, upper)
        scale_limits = np.flatnonzero(limited < scales).tolist()
        scales = limited
    point = coordinates / scales
    model.evaluate(point * scales, 1, transformed=transformed)
    lower, upper = lower / scales, upper / scales
    work = model.workspace(3, transformed, solve=True, affine=affine_rows)
    theta, gradient, value, stats = (
        np.full(model.size, np.nan),
        np.full(model.size, np.nan),
        np.full(1, np.nan),
        np.full(6, -1, np.int64),
    )
    native_requests = np.zeros(3, np.int64)
    reset_count = np.zeros(1, np.int64)
    extra = ((affine_rows, 0 if targets is None else _pointer(targets), True)
        if invalid_trial_policy == "minimum-step"
        else (() if affine_rows <= 1 else (affine_rows, _pointer(targets))))
    if indices is not None:
        restriction = model.workspace(3, transformed, solve=True, affine=affine_rows, restricted=True)
        extra = (affine_rows, 0 if targets is None else _pointer(targets),
            invalid_trial_policy == "minimum-step", _pointer(indices), len(indices),
            _pointer(restriction), restriction.nbytes)
    valid = _native_core._linear_joint_fit(
        model.descriptor,
        model.domain,
        _pointer(point),
        _pointer(model.data),
        _pointer(model.features),
        model.h0,
        _pointer(lower),
        _pointer(upper),
        _pointer(scales),
        _pointer(work),
        work.nbytes,
        _pointer(theta),
        _pointer(gradient),
        _pointer(value),
        _pointer(stats),
        accuracy,
        maxiter,
        transformed,
        speculate,
        0 if normal is None else _pointer(normal),
        target,
        _pointer(native_requests),
        curvature_reset,
        _pointer(reset_count),
        *extra,
    )
    for mask, count in enumerate(native_requests, 1):
        model.request_counts[mask] += int(count)
    # Preserve a finite feasible estimate even when the final derivative request fails.
    candidates = []
    for label, candidate in (("initial", initial), ("returned", theta)):
        try:
            nll = model.evaluate(candidate, 1)[0]
            if model.assess(candidate, None)[0]:
                candidates.append((nll, label, candidate.copy()))
        except ValueError:
            pass
    if not candidates:
        raise RuntimeError("The native solve produced no finite feasible estimate.")
    nll, selected, theta = min(
        candidates, key=lambda candidate: (candidate[0], candidate[1] != "returned")
    )
    retained_gradient = (
        gradient if valid and selected == "returned" and np.isfinite(gradient).all()
        else None
    )
    terminated = bool(valid and stats[0] == 0)
    success = terminated and selected == "returned"
    if terminated and selected == "initial":
        # Native derivatives belong to the returned point, not this retained seed.
        try:
            retained_gradient = model.evaluate(theta, 2)[1]
            if retained_gradient is not None and np.isfinite(retained_gradient).all():
                feasible, residual, _ = model.assess(theta, retained_gradient)
                success = bool(feasible and residual <= stationarity_tolerance)
            else:
                retained_gradient = None
        except ValueError:
            retained_gradient = None
    requests = _describe_jet_requests(model.request_counts, requests_before)
    # The final native request returns a checked theta gradient, including z solves.
    return OptimizeResult(
        x=theta,
        optimizer_coordinates=(point.copy() if transformed and selected == "returned"
            else coordinates.copy() if transformed and initial_coordinates is not None else None),
        fun=nll,
        jac=retained_gradient,
        status=int(stats[0]),
        success=success,
        nit=max(0, int(stats[1])),
        nfev=requests["F"],
        njev=requests["G"],
        nhev=requests["H"],
        jet_requests=requests,
        native_jet_requests=dict(zip(_JET_MASK_NAMES[1:4], map(int, native_requests))),
        kernel_dispatch=int(stats[5]),
        curvature_prepared=curvature_prepared,
        curvature_scale_limits=scale_limits,
        speculative_trial_gradients=speculate,
        constraint_curvature_reset=curvature_reset,
        constraint_curvature_resets=int(reset_count[0]),
        stationarity_tolerance=stationarity_tolerance,
        selected=selected,
        coordinate_subset=(None if indices is None else dict(
            space="z" if transformed else "theta", free=indices.tolist())),
        affine_equality=(
            None
            if normal is None
            else dict(
                normal=normal.tolist(),
                target=target,
                violation=float(abs(normal @ theta[: model.columns] - target)),
            )
            if affine_rows == 1
            else dict(
                normals=normal.tolist(),
                targets=targets.tolist(),
                violation=float(np.max(abs(normal @ theta[: model.columns] - targets))),
            )
        ),
    )


def _solve_affine_phase(
    model, initial, rows, *, multiple, accuracy, maxiter, transformed,
    speculate, curvature_reset, invalid_trial_policy,
):
    """Refine at requested accuracy, then strictly only when evidence requires it."""
    from ._nonsmooth_fit import residual_face_evidence

    affine = (model.features[rows], model.data[rows]) if multiple else (
        model.features[rows[0]], model.data[rows[0]])
    before = tuple(model.request_counts)
    remaining, point, attempts, fitted = maxiter, initial, [], []
    best = evidence = None
    strict_accuracy = min(accuracy, np.finfo(float).eps)
    # Measured GH refinements lose time to the preliminary tier and its restart.
    accuracies = (strict_accuracy,) if model.code == 4 else (accuracy, strict_accuracy)
    for phase_accuracy in dict.fromkeys(accuracies):
        if remaining <= 0:
            break
        budget = remaining
        requests_before = tuple(model.request_counts)
        try:
            child = _solve_joint_phase(model, point, transformed=transformed,
                use_curvature=True, accuracy=phase_accuracy, maxiter=budget, affine=affine,
                speculate=speculate, curvature_reset=curvature_reset,
                invalid_trial_policy=invalid_trial_policy)
        except ValueError as failed:
            if best is None:
                raise
            # Phase setup can fail before entering the native optimizer.
            attempts.append(dict(accuracy=phase_accuracy, iteration_budget=budget,
                iterations=0, charged_iterations=0, nll=None, mode=None, error=str(failed),
                selected=False, evidence_requested=False, supports_kink_stop=False,
                jet_requests=_describe_jet_requests(model.request_counts, requests_before)))
            break
        charged = max(1, child.nit)
        if charged > budget:
            raise RuntimeError("Affine refinement exceeded its iteration budget.")
        remaining -= charged
        fitted.append(child)
        child_evidence, error = None, None
        selected = best is None or child.fun < best.fun
        if selected:
            try:
                child_evidence = residual_face_evidence(model, child.x, rows, accuracy, multiple=multiple)
            except ValueError as failed:
                error = str(failed)
            best, evidence = child, child_evidence
        attempts.append(dict(accuracy=phase_accuracy, iteration_budget=budget,
            iterations=child.nit, charged_iterations=charged, nll=child.fun,
            mode=child.status, error=error,
            selected=selected, evidence_requested=selected,
            jet_requests=_describe_jet_requests(model.request_counts, requests_before),
            supports_kink_stop=bool((child_evidence or {}).get("supports_kink_stop"))))
        if (evidence or {}).get("supports_kink_stop"):
            break
        point = best.x
    if best is None:
        raise ValueError("Affine refinement requires a positive iteration budget.")
    result = OptimizeResult(best)
    requests = _describe_jet_requests(model.request_counts, before)
    names = _JET_MASK_NAMES[1:4]
    result.update(nit=maxiter-remaining, jet_requests=requests,
        nfev=requests["F"], njev=requests["G"], nhev=requests["H"],
        native_jet_requests={key: sum((p.get("native_jet_requests") or {}).get(key, 0)
            for p in fitted) for key in names},
        constraint_curvature_resets=sum(p.get("constraint_curvature_resets", 0) for p in fitted),
        refinement_attempts=attempts, refinement_evidence=evidence)
    return result


def _refine_terminated_face(
    model, best, remaining, phases, accuracy, transformed, *, speculate=None, curvature_reset=None,
    invalid_trial_policy="interpolate",
):
    """Refine an identified shock face without claiming smooth stationarity."""
    from ._nonsmooth_fit import bracket_residual_face

    tolerance = max(np.sqrt(accuracy), np.sqrt(np.finfo(float).eps))
    before = tuple(model.request_counts)
    diagnostic = None
    error = None
    try:
        diagnostic = bracket_residual_face(model, best.x)
    except ValueError as failed:
        error = str(failed)
    requests = _describe_jet_requests(model.request_counts, before)
    phases.append(dict(space="z" if transformed else "theta", mode=None, iterations=0,
        algorithm="residual-face-check", iteration_budget=0, nll=best.fun,
        function_evaluations=requests["F"], gradient_evaluations=requests["G"],
        hessian_evaluations=requests["H"], jet_requests=requests, error=error))
    near = tolerance * max(1.0, np.max(abs(best.x/model.units)))
    if diagnostic is None or not (
        diagnostic["single_face"] and diagnostic["feasible"]
        and diagnostic["normal_minimum"] and diagnostic["backward_error"] <= near
        and diagnostic["tangent_jump"] <= tolerance
    ):
        return best, remaining, False

    row = diagnostic["row"]
    budget = max(1, remaining // 2)
    before = tuple(model.request_counts)
    error = None
    child = None
    evidence = None
    try:
        child = _solve_affine_phase(model, best.x, [row], multiple=False, transformed=transformed,
            accuracy=accuracy, maxiter=budget,
            speculate=speculate, curvature_reset=curvature_reset,
            invalid_trial_policy=invalid_trial_policy)
        remaining -= max(1, child.nit)
        if (child.get("jac") is None or np.shape(child.jac) != np.shape(child.x)
                or not np.isfinite(child.jac).all()):
            child.jac = model.evaluate(child.x, 2)[1]
        feasible, child.stationarity_bound, violation = model.assess(child.x, child.jac)
        if feasible and child.fun <= best.fun:
            child.selected_space = "z" if transformed else "theta"
            child.predicted_coordinate_gain = None
            best = child
            evidence = child.refinement_evidence
            best.nearby_kink_evidence = evidence
    except ValueError as failed:
        error = str(failed)
    requests = _describe_jet_requests(model.request_counts, before)
    phases.append(dict(space="z" if transformed else "theta",
        algorithm="native-slsqp-affine-shock-face", mode=None if child is None else child.status,
        iterations=0 if child is None else child.nit, iteration_budget=budget,
        nll=None if child is None else child.fun, row=row, error=error, evidence=evidence,
        requested_accuracy=accuracy,
        accuracy=None if child is None else child.refinement_attempts[0]["accuracy"],
        refinement_attempts=[] if child is None else child.refinement_attempts,
        stationarity_bound=None if child is None else child.get("stationarity_bound"),
        curvature_scaling=False if child is None else child.curvature_prepared,
        affine_equality=None if child is None else child.affine_equality,
        kernel_dispatch=None if child is None else child.kernel_dispatch,
        native_jet_requests=None if child is None else child.get("native_jet_requests"),
        speculative_trial_gradients=None if child is None else child.get("speculative_trial_gradients"),
        constraint_curvature_resets=0 if child is None else child.get("constraint_curvature_resets", 0),
        invalid_trial_policy=invalid_trial_policy,
        function_evaluations=requests["F"], gradient_evaluations=requests["G"],
        hessian_evaluations=requests["H"], jet_requests=requests))
    stop = bool(evidence and evidence["supports_kink_stop"])
    if stop:
        best.success = False
    return best, remaining, stop


def _near_faces(model, theta, near):
    """Linearly independent residual faces within the refinement distance, nearest first."""
    from ._nonsmooth_fit import residual_face_distances

    distances = residual_face_distances(model, theta)
    rows = []
    for row in np.argsort(distances):
        if distances[row] > near or len(rows) == model.columns:
            break
        candidate = model.features[rows + [int(row)]] * model.units[: model.columns]
        if np.linalg.matrix_rank(candidate) == len(rows) + 1:
            rows.append(int(row))
    return rows


def _refine_terminated_faces(
    model, best, remaining, phases, accuracy, transformed, rows, *, speculate=None,
    curvature_reset=None,
    invalid_trial_policy="interpolate",
):
    """Refine several simultaneous shock faces with an active-set phase sequence.

    Each round solves the restricted problem on the current faces, then checks
    tangential stationarity and normal growth (Griewank and Walther 2016); rows
    entering the refinement distance are added and rows whose kink cannot hold a
    minimum are released, as in Womersley and Fletcher (1986). Every round shares
    the caller's budget and retains only finite feasible estimates without NLL loss.
    """
    from ._nonsmooth_fit import residual_face_evidence

    tolerance = max(np.sqrt(accuracy), np.sqrt(np.finfo(float).eps))
    near = tolerance * max(1.0, np.max(abs(best.x / model.units)))
    stop = False
    for _ in range(model.columns):
        if remaining <= 0 or not rows:
            break
        budget = max(1, remaining // 2)
        before = tuple(model.request_counts)
        error = child = evidence = None
        try:
            child = _solve_affine_phase(model, best.x, rows, multiple=True, transformed=transformed,
                accuracy=accuracy, maxiter=budget,
                speculate=speculate, curvature_reset=curvature_reset,
                invalid_trial_policy=invalid_trial_policy)
            remaining -= max(1, child.nit)
            if (child.get("jac") is None or np.shape(child.jac) != np.shape(child.x)
                    or not np.isfinite(child.jac).all()):
                child.jac = model.evaluate(child.x, 2)[1]
            feasible, child.stationarity_bound, _ = model.assess(child.x, child.jac)
            if feasible and child.fun <= best.fun:
                child.selected_space = "z" if transformed else "theta"
                child.predicted_coordinate_gain = None
                best = child
            evidence = (child.refinement_evidence if best is child else
                        residual_face_evidence(model, best.x, rows, accuracy, multiple=True))
            best.nearby_kink_evidence = evidence
        except ValueError as failed:
            error = str(failed)
        requests = _describe_jet_requests(model.request_counts, before)
        phases.append(dict(space="z" if transformed else "theta",
            algorithm="native-slsqp-affine-shock-faces", mode=None if child is None else child.status,
            iterations=0 if child is None else child.nit, iteration_budget=budget,
            nll=None if child is None else child.fun, rows=list(rows), error=error,
            evidence=evidence, requested_accuracy=accuracy,
            accuracy=None if child is None else child.refinement_attempts[0]["accuracy"],
            refinement_attempts=[] if child is None else child.refinement_attempts,
            stationarity_bound=None if child is None else child.get("stationarity_bound"),
            curvature_scaling=False if child is None else child.curvature_prepared,
            affine_equality=None if child is None else child.affine_equality,
            kernel_dispatch=None if child is None else child.kernel_dispatch,
            native_jet_requests=None if child is None else child.get("native_jet_requests"),
            speculative_trial_gradients=None if child is None else child.get("speculative_trial_gradients"),
            constraint_curvature_resets=0 if child is None else child.get("constraint_curvature_resets", 0),
            invalid_trial_policy=invalid_trial_policy,
            function_evaluations=requests["F"], gradient_evaluations=requests["G"],
            hessian_evaluations=requests["H"], jet_requests=requests))
        if evidence is None:
            break
        if evidence["supports_kink_stop"]:
            stop = True
            break
        # A saturated weight or a concave kink means leaving that face decreases F.
        kappas, weights = np.asarray(evidence["kappas"]), np.asarray(evidence["weights"])
        leaving = np.flatnonzero(kappas <= 0)
        if not leaving.size:
            leaving = np.flatnonzero(np.asarray(evidence["saturated"]))
        if leaving.size:
            if len(rows) == 1:
                break
            worst = leaving[np.argmax(np.where(kappas[leaving] <= 0, np.inf, abs(weights[leaving])))]
            rows = [row for i, row in enumerate(rows) if i != worst]
            continue
        # Release incompatible faces before admitting another nearby equality.
        entering = [row for row in _near_faces(model, best.x, near) if row not in rows]
        if entering and len(rows) < model.columns:
            candidate = rows + entering[:1]
            if np.linalg.matrix_rank(model.features[candidate]) == len(candidate):
                rows = candidate
                continue
        break
    if stop:
        best.success = False
    return best, remaining, stop


def _refine_terminated_kinks(
    model, best, remaining, phases, accuracy, transformed, *, speculate=None, curvature_reset=None,
    invalid_trial_policy="interpolate",
):
    """Use the single-face path for one nearby face and the active set otherwise."""
    tolerance = max(np.sqrt(accuracy), np.sqrt(np.finfo(float).eps))
    near = tolerance * max(1.0, np.max(abs(best.x / model.units)))
    try:
        rows = _near_faces(model, best.x, near)
    except ValueError:
        rows = []
    if len(rows) <= 1:
        best, remaining, stop = _refine_terminated_face(model, best, remaining, phases,
            accuracy, transformed, speculate=speculate, curvature_reset=curvature_reset,
            invalid_trial_policy=invalid_trial_policy)
        if stop or remaining <= 0 or not phases or phases[-1].get("algorithm") != (
            "native-slsqp-affine-shock-face"
        ):
            return best, remaining, stop
        try:
            rows = _near_faces(model, best.x, near)
        except ValueError:
            return best, remaining, stop
        if not rows:
            return best, remaining, stop
    return _refine_terminated_faces(model, best, remaining, phases, accuracy, transformed,
        rows, speculate=speculate, curvature_reset=curvature_reset,
        invalid_trial_policy=invalid_trial_policy)


def solve_joint(
    model,
    initial,
    *,
    transformed,
    accuracy,
    maxiter,
    allow_chart_refinement=False,
    initial_curvature=False,
    speculate=None,
    curvature_reset=None,
    terminal_reserve=0,
    full_first_phase=None,
    invalid_trial_policy="interpolate",
    initial_coordinates=None,
):
    """Refine a joint estimate within one shared iteration budget.

    EGARCH keeps the first SQP phase's state until termination or exhaustion of
    the available budget. Later phases use only unspent iterations; an explicit
    terminal reserve is never consumed here. APARCH may use unspent work for a
    final active-bound refinement when theta-space refinement is allowed.
    """
    if not 0 <= terminal_reserve < maxiter or int(terminal_reserve) != terminal_reserve:
        raise ValueError("The terminal reserve must leave a positive SQP budget.")
    if full_first_phase is None:
        full_first_phase = model.family == 2
    theta_refinement_allowed = not transformed or allow_chart_refinement
    solve_requests_before = tuple(model.request_counts)
    phases = []
    best = None
    remaining = maxiter
    point = initial
    coordinates = initial_coordinates
    tolerance = max(np.sqrt(accuracy), np.sqrt(np.finfo(float).eps))
    phase = 0
    while phase < 4:
        available = remaining - terminal_reserve
        if available <= 0:
            break
        phase_budget = min(
            available,
            remaining if phase == 3 or (phase == 0 and full_first_phase)
            else max(1, remaining // 2),
        )
        phase_accuracy = accuracy if phase == 0 else min(accuracy, np.finfo(float).eps)
        # Validated for certified multi-lag quadrature refinements only. Recovery
        # and closed-form paths retain their existing metric and basin search.
        limit_curvature_scale = bool(
            model.family == 3 and model.p_vol > 1 and model.code in (2, 4, 5, 6, 7)
            and best is not None and best.success and best.stationarity_bound <= tolerance
        )
        requests_before = tuple(model.request_counts)
        try:
            candidate = _solve_joint_phase(
                model,
                point,
                transformed=transformed,
                accuracy=phase_accuracy,
                maxiter=phase_budget,
                use_curvature=initial_curvature or phase > 0,
                limit_curvature_scale=limit_curvature_scale,
                speculate=speculate,
                curvature_reset=curvature_reset,
                invalid_trial_policy=invalid_trial_policy,
                stationarity_tolerance=tolerance,
                initial_coordinates=coordinates,
            )
        except ValueError as error:
            requests = _describe_jet_requests(model.request_counts, requests_before)
            phases.append(
                dict(
                    space="z" if transformed else "theta",
                    mode=None,
                    iterations=0,
                    iteration_budget=phase_budget,
                    nll=None,
                    stationarity_bound=None,
                    curvature_scaling=False,
                    accuracy=phase_accuracy,
                    stationarity_tolerance=tolerance,
                    function_evaluations=requests["F"],
                    gradient_evaluations=requests["G"],
                    hessian_evaluations=requests["H"],
                    jet_requests=requests,
                    error=str(error),
                    invalid_trial_policy=invalid_trial_policy,
                )
            )
            if transformed and allow_chart_refinement:
                # An unavailable chart consumes no optimizer phase or iteration budget.
                transformed = False
                coordinates = None
                continue
            if best is not None:
                break
            raise
        remaining -= max(1, candidate.nit)
        gradient = candidate.get("jac")
        if gradient is None:
            try:
                gradient = model.evaluate(candidate.x, 2)[1]
            except ValueError:
                gradient = None
        _, residual, _ = model.assess(candidate.x, gradient)
        candidate.jac = None if gradient is None else gradient.copy()
        candidate.stationarity_bound = residual
        predicted_gain = None
        if candidate.success and residual <= tolerance and remaining > 0:
            try:
                candidate.curvature_hessian = model.evaluate(candidate.x, 4)[2]
                predicted_gain = model.predicted_coordinate_gain(
                    candidate.x, gradient, candidate.curvature_hessian
                )
            except ValueError:
                pass
        requests = _describe_jet_requests(model.request_counts, requests_before)
        phases.append(
            dict(
                space="z" if transformed else "theta",
                mode=candidate.status,
                iterations=candidate.nit,
                iteration_budget=phase_budget,
                nll=candidate.fun,
                stationarity_bound=residual,
                predicted_coordinate_gain=predicted_gain,
                curvature_scaling=candidate.curvature_prepared,
                curvature_scale_limits=candidate.get("curvature_scale_limits", []),
                accuracy=phase_accuracy,
                stationarity_tolerance=tolerance,
                function_evaluations=requests["F"],
                gradient_evaluations=requests["G"],
                hessian_evaluations=requests["H"],
                jet_requests=requests,
                native_jet_requests=candidate.get("native_jet_requests"),
                speculative_trial_gradients=candidate.get(
                    "speculative_trial_gradients"
                ),
                kernel_dispatch=candidate.kernel_dispatch,
                constraint_curvature_reset=candidate.get(
                    "constraint_curvature_reset", False
                ),
                constraint_curvature_resets=candidate.get(
                    "constraint_curvature_resets", 0
                ),
                invalid_trial_policy=invalid_trial_policy,
            )
        )
        previous_value = np.inf if best is None else best.fun
        if (
            best is None
            or candidate.fun < best.fun
            or (candidate.fun == best.fun and residual < best.stationarity_bound)
            or _same_estimate_with_success(candidate, best)
        ):
            best = candidate
            best.stationarity_bound = residual
            best.predicted_coordinate_gain = predicted_gain
            best.selected_space = "z" if transformed else "theta"
        if (
            phase == 0 and model.family == 2 and model.columns and model.p_vol
            and best.success and best.stationarity_bound > tolerance
            and remaining > terminal_reserve
        ):
            best, available, face_stop = _refine_terminated_kinks(
                model, best, remaining-terminal_reserve, phases, accuracy, transformed,
                speculate=speculate, curvature_reset=curvature_reset,
                invalid_trial_policy=invalid_trial_policy,
            )
            remaining = available + terminal_reserve
            if face_stop:
                break
        if (
            phase == 0 and model.family == 3
            and theta_refinement_allowed and remaining > terminal_reserve
            and ((model.code in (1, 3) and not best.success) or not allow_chart_refinement
                 or not refinement_chart_available(model, best.x))
        ):
            best, phases, available = refine_before_restart(
                model, best, phases, remaining-terminal_reserve, accuracy)
            remaining = available + terminal_reserve
        needs_precision = (
            best.predicted_coordinate_gain is not None
            and best.predicted_coordinate_gain > 0.5 * accuracy * len(model.data)
        )
        if (
            best.success
            and best.stationarity_bound <= tolerance
            and not needs_precision
        ):
            break
        if phase and best.fun >= previous_value:
            best.refinement_stalled = True
            break
        point = best.x
        if allow_chart_refinement:
            transformed = not transformed
        coordinates = best.get("optimizer_coordinates") if transformed else None
        phase += 1
    if best is None:
        raise RuntimeError("No valid joint optimizer phase completed.")
    best.phases = tuple(phases)
    best.nit = sum(p["iterations"] for p in phases)
    best.nfev = sum(p["function_evaluations"] for p in phases)
    best.njev = sum(p["gradient_evaluations"] for p in phases)
    best.nhev = sum(p["hessian_evaluations"] for p in phases)
    if model.family == 3 and theta_refinement_allowed:
        best = _refine_joint_active_bounds(model, best, accuracy=accuracy,
            maxiter=maxiter-terminal_reserve, speculate=speculate,
            invalid_trial_policy=invalid_trial_policy)
        last = phases[-1]
        best.chart_handoff_failure = (
            dict(space=last["space"], mode=last["mode"])
            if allow_chart_refinement and last["mode"] != 0
            and last["space"] != best.selected_space else None
        )
    best.jet_requests = _describe_jet_requests(model.request_counts, solve_requests_before)
    return best


def solve_joint_two_routes(
    model,
    initial,
    *,
    accuracy,
    maxiter,
    transformed=False,
    allow_chart_refinement=True,
    controller=None,
    speculate=False,
    initial_coordinates=None,
):
    """Two complete joint routes from one theta start.

    The alternate chart has a reserved budget and runs when the first route lacks
    credibility and has no validated nearby-kink stop. That stop does not certify
    the returned doubles. Both routes retain the existing likelihood and domains.
    """
    if not np.isfinite(maxiter) or maxiter < 1 or int(maxiter) != maxiter:
        raise ValueError("Expected a positive integer total iteration budget.")
    maxiter = int(maxiter)
    controller = solve_joint if controller is None else controller
    original = np.array(initial, copy=True)
    if initial_coordinates is not None:
        initial_coordinates = _initial_joint_coordinates(model, original, True, initial_coordinates)
    allow_alternate = allow_chart_refinement and maxiter > 1
    primary_budget = maxiter // 2 if allow_alternate else maxiter
    primary = controller(
        model, original.copy(), accuracy=accuracy, maxiter=primary_budget,
        transformed=transformed, allow_chart_refinement=False, speculate=speculate,
        **({"initial_coordinates": initial_coordinates}
           if transformed and initial_coordinates is not None else {}),
    )
    tolerance = max(np.sqrt(accuracy), np.sqrt(np.finfo(float).eps))
    predicted_gain = primary.get("predicted_coordinate_gain")
    # The inner route owns curvature refinement. A stalled refinement does not
    # invalidate its separate stationarity certificate.
    credible = (
        primary.success and primary.stationarity_bound <= tolerance
        and (predicted_gain is None or predicted_gain <= 0.5 * accuracy * len(model.data)
             or primary.get("refinement_stalled", False))
    )
    kink_stop = bool((primary.get("nearby_kink_evidence") or {}).get("supports_kink_stop"))
    routes = [("primary", primary)]
    failed_phase = None
    if allow_alternate and not credible and not kink_stop:
        before = tuple(model.request_counts)
        try:
            alternate = controller(
                model, original.copy(), accuracy=accuracy,
                maxiter=maxiter-primary_budget, transformed=not transformed,
                allow_chart_refinement=False, initial_curvature=True, speculate=speculate,
                **({"initial_coordinates": initial_coordinates}
                   if not transformed and initial_coordinates is not None else {}),
            )
            routes.append(("original-start-coordinate-rescue", alternate))
        except ValueError as error:
            # A route that cannot begin still owns its preparation requests.
            requests = _describe_jet_requests(model.request_counts, before)
            failed_phase = dict(
                space="theta" if transformed else "z", mode=None,
                purpose="original-start-coordinate-rescue", iterations=0,
                iteration_budget=maxiter-primary_budget, nll=None, error=str(error),
                jet_requests=requests, function_evaluations=requests["F"],
                gradient_evaluations=requests["G"], hessian_evaluations=requests["H"],
            )
    selected = min((fit for _, fit in routes), key=lambda fit: (fit.fun, fit.stationarity_bound))
    result = OptimizeResult(selected)
    result.phases = tuple(
        dict(phase, purpose=purpose)
        for purpose, fit in routes for phase in fit.phases
    )
    if failed_phase is not None:
        result.phases += (failed_phase,)
    for field, counter in (
        ("nit", "iterations"), ("nfev", "function_evaluations"),
        ("njev", "gradient_evaluations"), ("nhev", "hessian_evaluations"),
    ):
        result[field] = sum(fit[field] for _, fit in routes)
        if failed_phase is not None:
            result[field] += failed_phase[counter]
    result.route_budgets = [primary_budget]
    if len(routes) > 1 or failed_phase is not None:
        result.route_budgets.append(maxiter-primary_budget)
    result.total_iteration_limit = maxiter
    assert result.nit <= maxiter
    return result


def solve_joint_with_retry(model, initial, *, accuracy, maxiter, transformed=False,
                           allow_chart_refinement=True, primary_maxiter=1000,
                           initial_coordinates=None):
    """Preserve a complete primary route, then conditionally retry its start.

    A first phase without stationarity evidence retries the original theta start
    in the alternate chart, with only unspent iterations and curvature scaling
    when the original-start Hessian is available. An unavailable chart falls back
    through the existing phase controller. Explicit coordinate choices disable
    the retry. This controller backs explicit native joint APARCH fits.

    """
    for budget in (maxiter, primary_maxiter):
        if (isinstance(budget, bool) or not np.isfinite(budget)
                or budget < 1 or int(budget) != budget):
            raise ValueError("Expected a positive integer iteration budget.")
    if not np.isfinite(accuracy) or accuracy <= 0:
        raise ValueError("Optimizer accuracy must be positive and finite.")
    maxiter = int(maxiter)
    original = np.array(initial, copy=True)
    requests_before = tuple(model.request_counts)
    primary_budget = min(int(primary_maxiter), maxiter) if allow_chart_refinement else maxiter
    fitted = solve_joint(model, original.copy(), accuracy=accuracy, maxiter=primary_budget,
        transformed=transformed, allow_chart_refinement=allow_chart_refinement,
        initial_coordinates=initial_coordinates)
    fitted.route_budgets = [primary_budget]
    fitted.total_iteration_limit = maxiter
    tolerance = max(np.sqrt(accuracy), np.sqrt(np.finfo(float).eps))
    first = fitted.phases[0]
    first_credible = (first["mode"] == 0
                      and first["stationarity_bound"] <= tolerance)
    kink_stop = bool((fitted.get("nearby_kink_evidence") or {}).get("supports_kink_stop"))
    remaining = maxiter - fitted.nit
    if allow_chart_refinement and remaining > 0 and not first_credible and not kink_stop:
        speculate = first["mode"] == 0
        retry_space = not transformed
        before = tuple(model.request_counts)
        child, error = None, None
        try:
            child = solve_joint(model, original.copy(), accuracy=accuracy, maxiter=remaining,
                transformed=retry_space, allow_chart_refinement=True,
                initial_curvature=True, speculate=speculate,
                initial_coordinates=initial_coordinates if retry_space else None)
        except ValueError as failed:
            error = str(failed)
        requests = _describe_jet_requests(model.request_counts, before)
        phase = dict(space="z" if retry_space else "theta", algorithm="native-slsqp-same-start",
            purpose="same-start-metric-retry",
            iterations=0 if child is None else child.nit, iteration_budget=remaining,
            nll=None if child is None else child.fun, mode=None if child is None else child.status,
            stationarity_bound=None if child is None else child.stationarity_bound,
            accuracy=accuracy, curvature_scaling=True, error=error,
            refinement_phases=() if child is None else child.phases,
            jet_requests=requests, function_evaluations=requests["F"],
            gradient_evaluations=requests["G"], hessian_evaluations=requests["H"])
        fitted = _retain_joint_refinement(fitted, child, phase)
        fitted.route_budgets.append(remaining)
    if not fitted.success and fitted.stationarity_bound <= tolerance and fitted.nit < maxiter:
        fitted = _continue_joint_estimate(model, fitted, accuracy=accuracy, maxiter=maxiter,
            transformed=fitted.selected_space == "z", speculate=True)
    if allow_chart_refinement:
        fitted = _recover_failed_chart_handoff(model, fitted, accuracy=accuracy, maxiter=maxiter)
    fitted.optimizer_controller = "joint-same-start-retry"
    fitted.jet_requests = _describe_jet_requests(model.request_counts, requests_before)
    fitted.total_iteration_limit = maxiter
    assert fitted.nit <= maxiter
    return fitted


def _certified_portfolio_agreement(model, primary, routes, accuracy):
    """Stop an optional route when two feasible certified routes agree locally."""
    if not routes or routes[-1]["name"] != "scaled":
        return None
    scaled = routes[-1]
    tolerance = max(np.sqrt(accuracy), np.sqrt(np.finfo(float).eps))
    for record in (primary, scaled):
        if (not record["success"] or not record["optimizer_terminated"]
                or record["nll"] is None or not np.isfinite(record["nll"])
                or record["stationarity_bound"] is None
                or not np.isfinite(record["stationarity_bound"])
                or record["stationarity_bound"] > tolerance):
            return None
    agreement = accuracy * len(model.data)
    if abs(primary["nll"] - scaled["nll"]) > agreement:
        return None
    if not all(model.assess(np.asarray(row["theta"]), None)[0] for row in (primary, scaled)):
        return None
    return dict(reason="two-certified-routes-agree", likelihood_tolerance=agreement,
        primary_nll=primary["nll"], scaled_nll=scaled["nll"],
        skipped_route="unit", stationarity_tolerance=tolerance)


def solve_joint_with_metric_portfolio(
    model, initial, *, accuracy, maxiter, primary_iteration_limit=800,
    alternate_iteration_limit=800, continuation_iteration_limit=800,
    transformed=False, allow_chart_refinement=True,
    initial_coordinates=None,
):
    """Reserve scaled and unit routes; stop when certified routes agree locally.

    Unspent primary or scaled work is not transferred to another route. This
    controller neither changes the best evaluated point nor borrows certification.
    """
    for budget in (maxiter, primary_iteration_limit, alternate_iteration_limit, continuation_iteration_limit):
        if (isinstance(budget, bool) or not np.isfinite(budget)
                or budget < 1 or int(budget) != budget):
            raise ValueError("Expected positive integer total and route iteration limits.")
    if not np.isfinite(accuracy) or accuracy <= 0:
        raise ValueError("Optimizer accuracy must be positive and finite.")
    maxiter, primary_iteration_limit, alternate_iteration_limit, continuation_iteration_limit = map(
        int, (maxiter, primary_iteration_limit, alternate_iteration_limit, continuation_iteration_limit))
    original = np.array(initial, dtype=np.float64, order="C", copy=True)
    original_coordinates = None if initial_coordinates is None else np.array(
        initial_coordinates, dtype=np.float64, order="C", copy=True)
    primary_budget = min(maxiter, primary_iteration_limit)
    requests_before = tuple(model.request_counts)
    arguments = ({"initial_coordinates": original_coordinates.copy()}
        if transformed and original_coordinates is not None else {})
    fitted = solve_joint_with_retry(model, original.copy(), accuracy=accuracy,
        maxiter=primary_budget, transformed=transformed,
        allow_chart_refinement=allow_chart_refinement, **arguments)
    primary_record = dict(theta=fitted.x.copy(), nll=fitted.fun, success=bool(fitted.success),
        stationarity_bound=fitted.stationarity_bound, status=fitted.status, iterations=fitted.nit,
        coordinates=(None if fitted.get("optimizer_coordinates") is None
            else fitted.optimizer_coordinates.copy()), selected_space=fitted.selected_space,
        optimizer_terminated=bool(fitted.get("termination_success", fitted.status == 0)),
        kkt_tolerance=max(np.sqrt(accuracy), np.sqrt(np.finfo(float).eps)),
        inference_status="not-assessed",
        iteration_budget=primary_budget, phases=tuple(fitted.phases),
        jet_requests=_describe_jet_requests(model.request_counts, requests_before),
        route_budgets=fitted.get("route_budgets"))
    allocated = primary_budget
    route_budgets, routes = [primary_budget], []
    for name, use_curvature in (("scaled", True), ("unit", False)):
        if not allow_chart_refinement or allocated == maxiter:
            break
        if name == "unit":
            agreement = _certified_portfolio_agreement(model, primary_record, routes, accuracy)
            if agreement is not None:
                fitted.portfolio_agreement = agreement
                break
        budget = min(alternate_iteration_limit, maxiter-allocated)
        allocated += budget
        route_budgets.append(budget)
        before = tuple(model.request_counts)
        child, error = None, None
        try:
            arguments = ({"initial_coordinates": original_coordinates.copy()}
                if not transformed and original_coordinates is not None else {})
            child = solve_joint(model, original.copy(), accuracy=accuracy, maxiter=budget,
                transformed=not transformed, allow_chart_refinement=False,
                initial_curvature=use_curvature, speculate=False, **arguments)
        except ValueError as failed:
            error = str(failed)
        requests = _describe_jet_requests(model.request_counts, before)
        phase = dict(space="theta" if transformed else "z",
            algorithm="native-slsqp-metric-portfolio", purpose="same-start-"+name+"-metric",
            iterations=0 if child is None else child.nit, iteration_budget=budget,
            nll=None if child is None else child.fun, mode=None if child is None else child.status,
            stationarity_bound=None if child is None else child.stationarity_bound,
            error=error, accuracy=accuracy, curvature_scaling=use_curvature,
            speculative_trial_gradients=False, refinement_phases=() if child is None else child.phases,
            jet_requests=requests, function_evaluations=requests["F"],
            gradient_evaluations=requests["G"], hessian_evaluations=requests["H"])
        routes.append(dict(name=name, theta=None if child is None else child.x.copy(),
            coordinates=None if child is None else child.get("optimizer_coordinates"),
            nll=phase["nll"], success=False if child is None else bool(child.success),
            optimizer_terminated=False if child is None else bool(
                child.get("termination_success", child.status == 0)),
            stationarity_bound=phase["stationarity_bound"], iterations=phase["iterations"],
            iteration_budget=budget, error=error, jet_requests=requests))
        fitted = _retain_joint_refinement(fitted, child, phase)
    tolerance = max(np.sqrt(accuracy), np.sqrt(np.finfo(float).eps))
    credible = fitted.success and fitted.stationarity_bound <= tolerance
    if allow_chart_refinement and allocated < maxiter and not credible:
        budget = min(continuation_iteration_limit, maxiter-allocated)
        allocated += budget
        route_budgets.append(budget)
        transformed_continuation = fitted.selected_space == "z"
        before = tuple(model.request_counts)
        child, error = None, None
        try:
            child = solve_joint(model, fitted.x.copy(), accuracy=accuracy, maxiter=budget,
                transformed=transformed_continuation, allow_chart_refinement=False,
                initial_curvature=False, full_first_phase=True, speculate=False,
                curvature_reset=False, invalid_trial_policy="minimum-step",
                initial_coordinates=fitted.get("optimizer_coordinates") if transformed_continuation else None)
        except ValueError as failed:
            error = str(failed)
        requests = _describe_jet_requests(model.request_counts, before)
        phase = dict(space="z" if transformed_continuation else "theta",
            algorithm="native-slsqp-portfolio-continuation", purpose="retained-estimate-continuation",
            iterations=0 if child is None else child.nit, iteration_budget=budget,
            nll=None if child is None else child.fun, mode=None if child is None else child.status,
            stationarity_bound=None if child is None else child.stationarity_bound,
            error=error, accuracy=accuracy, curvature_scaling=False,
            full_first_phase=True, speculative_trial_gradients=False, invalid_trial_policy="minimum-step",
            refinement_phases=() if child is None else child.phases, jet_requests=requests,
            function_evaluations=requests["F"], gradient_evaluations=requests["G"],
            hessian_evaluations=requests["H"])
        routes.append(dict(name="continuation", theta=None if child is None else child.x.copy(),
            coordinates=None if child is None else child.get("optimizer_coordinates"),
            nll=phase["nll"], success=False if child is None else bool(child.success),
            stationarity_bound=phase["stationarity_bound"], iterations=phase["iterations"],
            iteration_budget=budget, error=error, jet_requests=requests))
        fitted = _retain_joint_refinement(fitted, child, phase)
    fitted.primary_route = primary_record
    fitted.portfolio_routes = tuple(routes)
    fitted.route_budgets = route_budgets
    fitted.total_iteration_limit = maxiter
    fitted.optimizer_controller = "joint-primary-preserved-metric-portfolio"
    fitted.jet_requests = _describe_jet_requests(model.request_counts, requests_before)
    assert fitted.nit <= sum(route_budgets) <= maxiter
    return fitted


def _recover_failed_chart_handoff(model, fitted, *, accuracy, maxiter):
    """Recover in the selected chart only after comparing complete fitting routes."""
    failure = fitted.get("chart_handoff_failure")
    remaining = maxiter-fitted.nit
    tolerance = max(np.sqrt(accuracy), np.sqrt(np.finfo(float).eps))
    credible = fitted.success and fitted.stationarity_bound <= tolerance
    kink_stop = bool((fitted.get("nearby_kink_evidence") or {}).get("supports_kink_stop"))
    if not failure or remaining <= 0 or credible or kink_stop:
        return fitted
    before = tuple(model.request_counts)
    child, attempted, error = None, None, None
    space = fitted.selected_space
    phase_accuracy = min(accuracy, np.finfo(float).eps)
    try:
        attempted = _solve_joint_phase(model, fitted.x.copy(), transformed=space == "z",
            accuracy=phase_accuracy, maxiter=remaining, use_curvature=True,
            stationarity_tolerance=tolerance,
            initial_coordinates=fitted.get("optimizer_coordinates") if space == "z" else None)
        gradient = attempted.jac
        if gradient is None:
            try:
                gradient = model.evaluate(attempted.x, 2)[1]
            except ValueError:
                pass
        feasible, attempted.stationarity_bound, _ = model.assess(attempted.x, gradient)
        attempted.jac = None if gradient is None else gradient.copy()
        if feasible:
            child = attempted
            child.selected_space, child.predicted_coordinate_gain = space, None
    except ValueError as failed:
        error = str(failed)
    requests = _describe_jet_requests(model.request_counts, before)
    phase = dict(space=space, algorithm="native-slsqp-chart-recovery",
        purpose="failed-handoff-recovery", failed_space=failure["space"],
        failed_mode=failure["mode"], iterations=0 if attempted is None else attempted.nit,
        iteration_budget=remaining, nll=None if attempted is None else attempted.fun,
        mode=None if attempted is None else attempted.status,
        stationarity_bound=None if attempted is None else attempted.get("stationarity_bound"),
        error=error, accuracy=phase_accuracy, stationarity_tolerance=tolerance,
        curvature_scaling=False if attempted is None else attempted.curvature_prepared,
        native_jet_requests=None if attempted is None else attempted.get("native_jet_requests"),
        kernel_dispatch=None if attempted is None else attempted.kernel_dispatch,
        speculative_trial_gradients=None if attempted is None else attempted.get("speculative_trial_gradients"),
        invalid_trial_policy="interpolate", jet_requests=requests,
        function_evaluations=requests["F"], gradient_evaluations=requests["G"],
        hessian_evaluations=requests["H"])
    result = _retain_joint_refinement(fitted, child, phase)
    result.chart_handoff_failure = None
    return result


def _refine_joint_active_bounds(model, fitted, *, accuracy, maxiter,
                               speculate=None, invalid_trial_policy="interpolate"):
    """Propose a bound-face step, then assess it in the full original domain.

    This internal theta-space phase uses only unspent iterations. Fixing a bound
    in its subproblem never changes the model or supplies a stationarity certificate.
    """
    remaining = maxiter-fitted.nit
    tolerance = max(np.sqrt(accuracy), np.sqrt(np.finfo(float).eps))
    if (remaining <= 0 or model.family < 2
            or (fitted.success and fitted.stationarity_bound <= tolerance)):
        return fitted
    gradient = fitted.get("jac")
    if gradient is None or not np.isfinite(gradient).all():
        return fitted
    held = ((fitted.x == model.lower) & (gradient > 0)) | (
        (fitted.x == model.upper) & (gradient < 0))
    if not np.any(held) or np.all(held):
        return fitted
    before = tuple(model.request_counts)
    budget = min(100, remaining)
    child, attempted, error = None, None, None
    try:
        attempted = _solve_joint_phase(model, fitted.x, transformed=False,
            accuracy=accuracy, maxiter=budget, use_curvature=False,
            free_coordinates=np.flatnonzero(~held), curvature_reset=False,
            speculate=speculate, invalid_trial_policy=invalid_trial_policy)
        child = attempted
        feasible, child.stationarity_bound, _ = model.assess(child.x, child.jac)
        child.success = bool(child.success and feasible and child.stationarity_bound <= tolerance)
        if not feasible:
            child = None
        else:
            child.selected_space = "theta"
            child.predicted_coordinate_gain = None
    except ValueError as failed:
        child = None
        error = str(failed)
    requests = _describe_jet_requests(model.request_counts, before)
    phase = dict(space="theta", algorithm="native-slsqp-active-bounds",
        purpose="active-bound-refinement", fixed_coordinates=np.flatnonzero(held).tolist(),
        iterations=0 if attempted is None else attempted.nit, iteration_budget=budget,
        nll=None if attempted is None else attempted.fun, mode=None if attempted is None else attempted.status,
        stationarity_bound=None if attempted is None else attempted.get("stationarity_bound"),
        accuracy=accuracy, curvature_scaling=False, error=error, jet_requests=requests,
        native_jet_requests=None if attempted is None else attempted.get("native_jet_requests"),
        kernel_dispatch=None if attempted is None else attempted.get("kernel_dispatch"),
        speculative_trial_gradients=None if attempted is None else attempted.get("speculative_trial_gradients"),
        invalid_trial_policy=invalid_trial_policy,
        function_evaluations=requests["F"], gradient_evaluations=requests["G"],
        hessian_evaluations=requests["H"])
    return _retain_joint_refinement(fitted, child, phase)


def _same_estimate_with_success(candidate, previous):
    return bool(candidate.success and not previous.success
        and candidate.fun == previous.fun
        and candidate.stationarity_bound == previous.stationarity_bound
        and np.array_equal(candidate.x, previous.x))


def _retain_joint_refinement(fitted, child, phase):
    selected = (fitted if child is None else
                min((fitted, child), key=lambda fit: (fit.fun, fit.stationarity_bound)))
    if child is not None and _same_estimate_with_success(child, fitted):
        selected = child
    result = OptimizeResult(selected)
    result.phases = (*fitted.phases, phase)
    for field, counter in (("nit", "iterations"), ("nfev", "function_evaluations"),
                           ("njev", "gradient_evaluations"), ("nhev", "hessian_evaluations")):
        result[field] = fitted[field] + phase[counter]
    if "route_budgets" in fitted:
        result.route_budgets = list(fitted.route_budgets)
    if "total_iteration_limit" in fitted:
        result.total_iteration_limit = fitted.total_iteration_limit
    return result


def _continue_joint_estimate(model, fitted, *, accuracy, maxiter, transformed, speculate=False):
    remaining = maxiter - fitted.nit
    tolerance = max(np.sqrt(accuracy), np.sqrt(np.finfo(float).eps))
    credible = fitted.success and fitted.stationarity_bound <= tolerance
    kink_stop = bool((fitted.get("nearby_kink_evidence") or {}).get("supports_kink_stop"))
    if remaining <= 0 or credible or kink_stop:
        return fitted
    before = tuple(model.request_counts)
    child, error_message = None, None
    try:
        child = solve_joint(model, fitted.x.copy(), transformed=transformed,
            accuracy=accuracy, maxiter=remaining, allow_chart_refinement=False,
            initial_curvature=False, speculate=speculate, curvature_reset=False,
            invalid_trial_policy="minimum-step",
            initial_coordinates=fitted.get("optimizer_coordinates") if transformed else None)
    except ValueError as error:
        error_message = str(error)
    requests = _describe_jet_requests(model.request_counts, before)
    phase = dict(space="z" if transformed else "theta", algorithm="native-slsqp-continuation",
        purpose="retained-estimate-continuation", iterations=0 if child is None else child.nit,
        invalid_trial_policy="minimum-step", iteration_budget=remaining,
        nll=None if child is None else child.fun, mode=None if child is None else child.status,
        error=error_message, accuracy=accuracy,
        stationarity_bound=None if child is None else child.stationarity_bound,
        curvature_scaling=False, refinement_phases=() if child is None else child.phases,
        jet_requests=requests, function_evaluations=requests["F"],
        gradient_evaluations=requests["G"], hessian_evaluations=requests["H"])
    return _retain_joint_refinement(fitted, child, phase)


def _refine_neighboring_regions(model, fitted, *, accuracy, maxiter, transformed, speculate=False):
    from ._nonsmooth_fit import neighboring_residual_proposals

    remaining = maxiter - fitted.nit
    tolerance = max(np.sqrt(accuracy), np.sqrt(np.finfo(float).eps))
    if (remaining <= 0 or not fitted.success or fitted.stationarity_bound > tolerance
            or model.family != 2 or not model.columns or not model.p_vol):
        return fitted
    before = tuple(model.request_counts)
    try:
        probes = neighboring_residual_proposals(model, fitted.x,
                                                hessian=fitted.get("curvature_hessian"))
    except ValueError as error:
        probes = dict(status="unavailable curvature", error=str(error), proposals=[])
    trial = min(probes["proposals"], key=lambda point: point.get("nll", np.inf), default={})
    budget = min(100, remaining)
    child, error_message = None, None
    if trial.get("nll", np.inf) < fitted.fun - accuracy*len(model.data):
        try:
            child = solve_joint(model, np.array(trial["theta"]), transformed=transformed,
                accuracy=accuracy, maxiter=budget, allow_chart_refinement=False,
                initial_curvature=True, speculate=speculate, curvature_reset=False)
        except ValueError as error:
            error_message = str(error)
    requests = _describe_jet_requests(model.request_counts, before)
    phase = dict(space="z" if transformed else "theta", algorithm="native-slsqp-neighboring-residual-region",
        purpose="local-region-check", iterations=0 if child is None else child.nit,
        iteration_budget=budget, probes=probes, accuracy=accuracy, error=error_message,
        mode=None if child is None else child.status, nll=None if child is None else child.fun,
        stationarity_bound=None if child is None else child.stationarity_bound,
        refinement_phases=[] if child is None else child.phases, curvature_scaling=child is not None,
        invalid_trial_policy="interpolate", jet_requests=requests,
        function_evaluations=requests["F"], gradient_evaluations=requests["G"],
        hessian_evaluations=requests["H"])
    return _retain_joint_refinement(fitted, child, phase)


def solve_joint_egarch(model, initial, *, accuracy, maxiter, transformed=False,
                       allow_chart_refinement=True):
    """Retain the two initial routes, then refine within the caller's total cap.

    The initial routes receive at most 600 iterations together. Continuation
    changes the invalid-trial policy, not the likelihood, domain or initial guess.
    Explicit coordinate choices apply to every subsequent phase too.
    """
    if isinstance(maxiter, bool) or not np.isfinite(maxiter) or maxiter < 1 or int(maxiter) != maxiter:
        raise ValueError("Expected a positive integer total iteration budget.")
    maxiter = int(maxiter)
    # Shared GH density work repays trial gradients on this measured layout.
    # Wider layouts and explicit z-first solves retain value-only trials.
    speculate = (not transformed and model.code >= 4 and model.columns == 4
                 and model.p_vol == model.q_vol == 1)
    fitted = solve_joint_two_routes(model, initial, accuracy=accuracy,
        maxiter=min(600, maxiter), transformed=transformed,
        allow_chart_refinement=allow_chart_refinement, speculate=speculate)
    refinement_space = transformed if not allow_chart_refinement else False
    fitted = _continue_joint_estimate(model, fitted, accuracy=accuracy, maxiter=maxiter,
                                     transformed=refinement_space, speculate=speculate)
    fitted = _refine_neighboring_regions(model, fitted, accuracy=accuracy, maxiter=maxiter,
                                        transformed=refinement_space, speculate=speculate)
    fitted.optimizer_controller = "joint-continuation"
    fitted.total_iteration_limit = maxiter
    assert fitted.nit <= maxiter
    return fitted


def fit_joint_mean_spec(
    spec,
    data,
    *,
    x=None,
    solver="native",
    log_mode=None,
    initial_params=None,
    maxiter=None,
    native_accuracy=1e-12,
    verbose=False,
    **kwargs,
):
    if kwargs:
        raise TypeError(f"Unsupported joint fit options: {', '.join(sorted(kwargs))}")
    if solver != "native":
        raise NotImplementedError("This entry is the independent native joint solver.")
    if maxiter is None:
        volatility = spec.get_component(Role.VOLATILITY)
        if isinstance(volatility, APARCH) and log_mode is None:
            maxiter = APARCH_PORTFOLIO_MAXITER
        else:
            maxiter = EGARCH_DEFAULT_MAXITER if isinstance(volatility, EGARCH) else 1000
    if (
        maxiter < 1
        or int(maxiter) != maxiter
        or not np.isfinite(native_accuracy)
        or native_accuracy <= 0
    ):
        raise ValueError("Optimizer accuracy and iteration budget must be positive.")
    started = time.perf_counter()
    model = JointMeanModel(spec, data, x)
    initial = model.initialize(initial_params)
    transformed = False if log_mode is None else bool(log_mode)
    if model.family == 3:
        controller = solve_joint_with_metric_portfolio if log_mode is None else solve_joint_with_retry
    else:
        controller = solve_joint_egarch if model.family == 2 else solve_joint
    result = controller(
        model,
        initial,
        transformed=transformed,
        accuracy=native_accuracy,
        maxiter=int(maxiter),
        allow_chart_refinement=log_mode is None,
    )
    return assemble_joint_fit(
        model, result, transformed, native_accuracy, maxiter, started
    )


def _completed_joint_phases(phases):
    """Report executed phases, not the aggregate wrappers around child routes."""
    for phase in phases:
        if children := phase.get("refinement_phases"):
            yield from _completed_joint_phases(children)
        elif phase["mode"] is not None:
            yield phase


def assemble_joint_fit(
    model, result, transformed, accuracy, maxiter, started, *, backend="native-slsqp"
):
    from .result import EstimationResult
    from ._evaluation import FilteredState, _egarch_density_metadata

    requests_before = tuple(model.request_counts)
    retained_hessian = result.get("curvature_hessian")
    for request in ((3, 1) if retained_hessian is not None else (7, 3, 1)):
        try:
            value, gradient, hessian, residuals, variance = model.evaluate(
                result.x, request, paths=True
            )
            if retained_hessian is not None:
                hessian = retained_hessian
            break
        except ValueError:
            if request == 1:
                raise RuntimeError("The returned joint estimate cannot be filtered.")
    feasible, residual, violation = model.assess(result.x, gradient)
    if not feasible:
        raise RuntimeError("The returned joint estimate is outside the model domain.")
    agreement = jet_agreement(
        result.fun,
        result.get("jac"),
        value,
        gradient,
        model.units,
        len(model.data),
        accuracy,
    )
    # Different request paths must not erase failed evidence at this same point.
    previous_residual = result.get("stationarity_bound", residual)
    residual = (
        max(residual, previous_residual)
        if np.isfinite(residual) and np.isfinite(previous_residual)
        else np.inf
    )
    if agreement["status"] == "inconsistent":
        residual = np.inf
    result.stationarity_bound = residual
    tolerance = max(np.sqrt(accuracy), np.sqrt(np.finfo(float).eps))
    terminated = bool(result.get("termination_success", result.status == 0))
    nonsmooth_evidence = result.get("nearby_kink_evidence")
    kink_stop = bool(nonsmooth_evidence and nonsmooth_evidence.get("supports_kink_stop"))
    stationary = bool(feasible and residual <= tolerance and not kink_stop)
    result.success = bool(result.success and terminated and stationary)
    result.fun, result.jac = value, gradient
    if result.get("curvature_prediction_pending", False):
        result.predicted_coordinate_gain = (
            model.predicted_coordinate_gain(result.x, gradient, hessian)
            if gradient is not None and hessian is not None
            else None
        )
        result.curvature_prediction_pending = False
    result.message = (
        "Converged with original-parameter stationarity certification."
        if result.success
        else "Finite feasible estimate; convergence is not certified. Parameters are unchanged."
    )
    covariance = observed_covariance(hessian, stationary=result.success)
    postfit_requests = _describe_jet_requests(model.request_counts, requests_before)
    all_requests = _describe_jet_requests(model.request_counts)
    result.nfev, result.njev, result.nhev = (
        all_requests["F"],
        all_requests["G"],
        all_requests["H"],
    )
    model.mean.unpack(result.x[: model.columns])
    model.volatility.unpack(result.x[model.columns : model.shape_start])
    model.density.unpack(result.x[model.shape_start :])
    if model.family == 3:
        persistence, slack = _aparch_persistence_evidence(model.vol, result.x[model.columns :])
        model.volatility.fitted_params["persistence"] = persistence
        model.volatility.fitted_params["stationarity_slack"] = slack
    warnings = [] if result.success else [result.message]
    if kink_stop:
        warnings.append(
            "Neighboring gradients support a nearby residual kink, not a stationarity "
            "certificate at the fitted parameters. Smooth inference remains unavailable."
        )
    if agreement["status"] == "inconsistent":
        warnings.append(
            "Jet requests disagree at the fitted parameters; numerical derivative "
            "evidence is unresolved. Stationarity and standard errors are not certified."
        )
    predicted_gain = result.get("predicted_coordinate_gain")
    if predicted_gain is not None and predicted_gain > 0.5 * accuracy * len(model.data):
        warnings.append(
            "Local curvature predicts further likelihood improvement; refinement "
            "exhausted its budget or stalled. The best feasible estimate is unchanged."
        )
    completed_phases = tuple(_completed_joint_phases(result.phases))
    primary_route = result.get("primary_route")
    if primary_route is not None:
        primary_route = {key: value.tolist() if isinstance(value, np.ndarray) else deepcopy(value)
            for key, value in primary_route.items()}
    if covariance is None:
        warnings.append(
            "Standard errors are unavailable: convergence or positive, numerically resolved observed information is not certified."
        )
    fitted = EstimationResult(
        model.spec,
        result,
        model.data,
        sigma2=variance,
        hessian=hessian,
        cov_matrix=covariance,
        time_elapsed=time.perf_counter() - started,
        fit_info={
            "solver": (
                f"{backend}-hybrid"
                if len({p["space"] for p in completed_phases}) > 1
                else backend
            ),
            "engine": "native-joint-linear-jet",
            "scalar_precision": (
                "double-double primal state; binary64 derivatives"
                if model.family == 2 and model.code != 0
                else "binary64"
            ),
            "log_mode": result.selected_space == "z",
            "optimization_space": f"selected {result.selected_space}-space; mean parameters remain theta",
            "optimizer_mode": result.status,
            "optimizer_status_kind": result.get("status_kind", "slsqp"),
            "optimizer_controller": result.get("optimizer_controller", (
                "joint-two-route" if "route_budgets" in result else "joint-phase-refinement"
            )),
            "route_iteration_budgets": result.get("route_budgets"),
            "primary_route": primary_route,
            "portfolio_agreement": deepcopy(result.get("portfolio_agreement")),
            "optimizer_terminated": terminated,
            "stationarity_status": "certified" if stationary else "uncertified",
            "inference_status": (
                "available" if covariance is not None else "unavailable"
            ),
            "jet_consistency": agreement,
            "nonsmooth_evidence": nonsmooth_evidence,
            "optimizer_phases": tuple(
                p.get("algorithm", f"{backend}-{p['space']}") for p in completed_phases
            ),
            "kernel_dispatch": result.get("kernel_dispatch"),
            "optimizer_phase_modes": tuple(p["mode"] for p in completed_phases),
            "kkt_residual": residual,
            "kkt_tolerance": tolerance,
            "maximum_constraint_violation": violation,
            "function_evaluations": result.nfev,
            "gradient_evaluations": result.njev,
            "hessian_evaluations": result.nhev,
            "jet_requests": all_requests,
            "postfit_jet_requests": postfit_requests,
            "request_count_scope": "prepared model lifetime, including failed jet requests; excludes constraints and map-only calls",
            "maxiter": maxiter,
            "likelihood_n_obs": len(model.data),
            "parameter_domain": model.describe_domain(),
            "warnings": tuple(warnings),
            "phase_diagnostics": result.phases,
        },
    )
    fitted._resid = residuals
    fitted._filtered_state_cache = FilteredState(
        model.data,
        model.data - residuals,
        residuals,
        variance,
        -value,
        None if gradient is None else -gradient,
        None if hessian is None else -hessian,
        model.density.signature,
        **_egarch_density_metadata(model.density, result.x[model.shape_start :]),
    )
    return fitted


def filter_joint_mean_spec(
    spec, data, theta, *, x=None, compute_score=False, compute_hessian=False
):
    from ._evaluation import FilteredState, _egarch_density_metadata

    model = JointMeanModel(spec, data, x)
    theta = np.ascontiguousarray(theta, dtype=np.float64)
    if not model.assess(theta, None)[0]:
        raise ValueError("Parameters violate the declared joint model domain.")
    request = 1 | (2 if compute_score else 0) | (4 if compute_hessian else 0)
    value, gradient, hessian, residuals, variance = model.evaluate(
        theta, request, paths=True
    )
    return FilteredState(
        model.data,
        model.data - residuals,
        residuals,
        variance,
        -value,
        None if gradient is None else -gradient,
        None if hessian is None else -hessian,
        model.density.signature,
        **_egarch_density_metadata(model.density, theta[model.shape_start :]),
    )
