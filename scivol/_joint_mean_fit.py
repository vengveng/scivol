"""Joint linear-mean jets; GARCH/GJR defaults, gated EGARCH/APARCH prototypes."""

from __future__ import annotations

import time
import numpy as np
from scipy.optimize import OptimizeResult

from . import _native_core
from ._mean_fit import PreparedMean, DENSITIES, _pointer, _workspace
from ._native_fit import (
    _describe_model,
    _coordinate_bounds,
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

    def workspace(self, request, transformed=False, *, solve=False, affine=False):
        if affine and not solve:
            raise ValueError("An affine phase needs optimizer workspace.")
        key = (request, bool(transformed), bool(solve), bool(affine))
        if key not in self.workspaces:
            sizes = _native_core._linear_joint_sizes(
                self.descriptor, request, transformed, self.domain, affine
            )
            self.workspaces[key] = _workspace(sizes[int(solve)])
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
            if not feasible or gradient is None or jacobian is None:
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
                    jacobian * units,
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


def _solve_joint_phase(
    model,
    initial,
    *,
    transformed,
    accuracy,
    maxiter,
    use_curvature=True,
    speculate=None,
    affine=None,
    curvature_reset=None,
):
    requests_before = tuple(model.request_counts)
    speculate = model.family >= 2 if speculate is None else bool(speculate)
    curvature_reset = (
        model.family == 3 if curvature_reset is None else bool(curvature_reset)
    )
    normal, target = None, 0.0
    if affine is not None:
        normal = np.ascontiguousarray(affine[0], dtype=np.float64)
        target = float(affine[1])
        if (
            model.family < 2
            or normal.shape != (model.columns,)
            or not np.isfinite(normal).all()
            or not np.any(normal)
            or not np.isfinite(target)
        ):
            raise ValueError(
                "Expected a finite nonzero affine EGARCH/APARCH mean constraint."
            )
    coordinates = model.inverse(initial) if transformed else initial.copy()
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
    point = coordinates / scales
    model.evaluate(point * scales, 1, transformed=transformed)
    if transformed:
        low, high = _coordinate_bounds(model.vol)
        lower, upper = (
            np.r_[np.full(model.columns, np.nan), low],
            np.r_[np.full(model.columns, np.nan), high],
        )
    else:
        lower, upper = model.lower.copy(), model.upper.copy()
    lower, upper = lower / scales, upper / scales
    work = model.workspace(3, transformed, solve=True, affine=normal is not None)
    theta, gradient, value, stats = (
        np.full(model.size, np.nan),
        np.full(model.size, np.nan),
        np.full(1, np.nan),
        np.full(6, -1, np.int64),
    )
    native_requests = np.zeros(3, np.int64)
    reset_count = np.zeros(1, np.int64)
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
    requests = _describe_jet_requests(model.request_counts, requests_before)
    # The final native request returns a checked theta gradient, including z solves.
    # Keep independent EGARCH gradient evidence during replacement validation.
    return OptimizeResult(
        x=theta,
        fun=nll,
        jac=(
            gradient
            if valid
            and model.family != 2
            and selected == "returned"
            and np.isfinite(gradient).all()
            else None
        ),
        status=int(stats[0]),
        success=bool(valid and stats[0] == 0 and selected == "returned"),
        nit=max(0, int(stats[1])),
        nfev=requests["F"],
        njev=requests["G"],
        nhev=requests["H"],
        jet_requests=requests,
        native_jet_requests=dict(zip(_JET_MASK_NAMES[1:4], map(int, native_requests))),
        kernel_dispatch=int(stats[5]),
        curvature_prepared=curvature_prepared,
        speculative_trial_gradients=speculate,
        constraint_curvature_reset=curvature_reset,
        constraint_curvature_resets=int(reset_count[0]),
        selected=selected,
        affine_equality=(
            None
            if normal is None
            else dict(
                normal=normal.tolist(),
                target=target,
                violation=float(abs(normal @ theta[: model.columns] - target)),
            )
        ),
    )


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
):
    """Refine the same joint estimate within one shared iteration budget."""
    if not 0 <= terminal_reserve < maxiter or int(terminal_reserve) != terminal_reserve:
        raise ValueError("The terminal reserve must leave a positive SQP budget.")
    phases = []
    best = None
    remaining = maxiter
    point = initial
    tolerance = max(np.sqrt(accuracy), np.sqrt(np.finfo(float).eps))
    phase = 0
    while phase < 4:
        available = remaining - terminal_reserve
        if available <= 0:
            break
        phase_budget = min(
            available, remaining if phase == 3 else max(1, remaining // 2)
        )
        phase_accuracy = accuracy if phase == 0 else min(accuracy, np.finfo(float).eps)
        requests_before = tuple(model.request_counts)
        try:
            candidate = _solve_joint_phase(
                model,
                point,
                transformed=transformed,
                accuracy=phase_accuracy,
                maxiter=phase_budget,
                use_curvature=initial_curvature or phase > 0,
                speculate=speculate,
                curvature_reset=curvature_reset,
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
                    function_evaluations=requests["F"],
                    gradient_evaluations=requests["G"],
                    hessian_evaluations=requests["H"],
                    jet_requests=requests,
                    error=str(error),
                )
            )
            if transformed and allow_chart_refinement:
                # An unavailable chart consumes no optimizer phase or iteration budget.
                transformed = False
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
                accuracy=phase_accuracy,
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
            )
        )
        previous_value = np.inf if best is None else best.fun
        if (
            best is None
            or candidate.fun < best.fun
            or (candidate.fun == best.fun and residual < best.stationarity_bound)
        ):
            best = candidate
            best.stationarity_bound = residual
            best.predicted_coordinate_gain = predicted_gain
            best.selected_space = "z" if transformed else "theta"
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
            break
        point = best.x
        if allow_chart_refinement:
            transformed = not transformed
        phase += 1
    if best is None:
        raise RuntimeError("No valid joint optimizer phase completed.")
    best.phases = tuple(phases)
    best.nit = sum(p["iterations"] for p in phases)
    best.nfev = sum(p["function_evaluations"] for p in phases)
    best.njev = sum(p["gradient_evaluations"] for p in phases)
    best.nhev = sum(p["hessian_evaluations"] for p in phases)
    return best


def fit_joint_mean_spec(
    spec,
    data,
    *,
    x=None,
    solver="native",
    log_mode=None,
    initial_params=None,
    maxiter=1000,
    native_accuracy=1e-12,
    verbose=False,
    **kwargs,
):
    if kwargs:
        raise TypeError(f"Unsupported joint fit options: {', '.join(sorted(kwargs))}")
    if solver != "native":
        raise NotImplementedError("This entry is the independent native joint solver.")
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
    result = solve_joint(
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


def assemble_joint_fit(
    model, result, transformed, accuracy, maxiter, started, *, backend="native-slsqp"
):
    from .result import EstimationResult
    from ._evaluation import FilteredState, _egarch_density_metadata

    requests_before = tuple(model.request_counts)
    retained_hessian = result.get("curvature_hessian")
    # Keep a full EGARCH evidence request during replacement validation.
    if model.family == 2:
        retained_hessian = None
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
    stationary = bool(feasible and residual <= tolerance)
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
    warnings = [] if result.success else [result.message]
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
    completed_phases = tuple(p for p in result.phases if p["mode"] is not None)
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
                if model.family == 2 and model.code == 7
                else "binary64"
            ),
            "log_mode": result.selected_space == "z",
            "optimization_space": f"selected {result.selected_space}-space; mean parameters remain theta",
            "optimizer_mode": result.status,
            "optimizer_status_kind": result.get("status_kind", "slsqp"),
            "optimizer_terminated": terminated,
            "stationarity_status": "certified" if stationary else "uncertified",
            "inference_status": (
                "available" if covariance is not None else "unavailable"
            ),
            "jet_consistency": agreement,
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
