"""Bounded joint-jet evidence against historical and identical-domain solvers.

No passed timing alone establishes promotion. Reports retain uncertified fits,
cross-evaluated likelihoods, feasibility, stationarity and phase diagnostics.
"""

import argparse
import hashlib
import json
import multiprocessing
import os
from pathlib import Path
import signal
import time
import warnings

import numpy as np
from scipy.optimize import Bounds, minimize

from scivol import ARX, HARX, GJRGARCH, StudentT, _core, _native_core
from benchmark_native_mean import paired_times
from scivol._joint_mean_fit import (
    JointMeanModel,
    FAMILIES,
    DENSITIES,
    fit_joint_mean_spec,
    assemble_joint_fit,
    solve_joint,
)
from scivol._mean_fit import _pointer
from scivol._native_fit import _coordinate_bounds, _coordinates_to_theta
from scivol._kernels import get_routine
from scivol._evaluation import (
    _filter_meanx_garch,
    _filter_meanx_gjr,
    _filter_meanx_egarch,
    _filter_gh,
)


def prepare_native_request(model, theta, mask):
    if mask not in (1, 2, 3, 4):
        raise ValueError("The request benchmark measures F, G, F|G and H.")
    theta = np.ascontiguousarray(theta, dtype=np.float64)
    output = np.empty(
        {1: 1, 2: model.size, 3: 1 + model.size, 4: (model.size, model.size)}[mask]
    )
    work = model.workspace(mask)
    args = (
        model.descriptor,
        _pointer(theta),
        _pointer(model.data),
        _pointer(model.features),
        model.h0,
        mask,
        False,
        _pointer(work),
        work.nbytes,
        _pointer(output) if mask & 1 else 0,
        _pointer(output[1:] if mask == 3 else output) if mask & 2 else 0,
        _pointer(output) if mask == 4 else 0,
        0,
        0,
    )

    def evaluate(_keepalive=(model, theta, work, output)):
        if not _native_core._linear_joint_prevalidated(*args):
            raise ValueError(f"Native request {mask} rejected the comparison point.")

    return evaluate, output


def prepare_legacy_request(model, theta, mask, *, output=None):
    if mask not in (1, 2, 3, 4):
        raise ValueError("The request benchmark measures F, G, F|G and H.")
    theta = np.ascontiguousarray(theta, dtype=np.float64)
    if output is None:
        output = np.empty(
            {1: 1, 2: model.size, 3: 1 + model.size, 4: (model.size, model.size)}[mask]
        )
    if mask == 3:
        value_request, _ = prepare_legacy_request(model, theta, 1, output=output[:1])
        gradient_request, _ = prepare_legacy_request(model, theta, 2, output=output[1:])

        def evaluate_combined():
            value_request()
            gradient_request()

        return evaluate_combined, output
    residuals, variance = np.empty_like(model.data), np.empty_like(model.data)
    family = ("garch", "gjr_garch", "egarch")[model.family]
    suffix = ("normal", "studentt", "skewt", "ged", "gh", "gh", "gh", "gh")[model.code]
    order = "11" if model.p_vol == model.q_vol == 1 and model.code < 4 else "pq"
    operation = {1: "nll", 2: "nll_grad", 4: "hess"}[mask]
    kernel = getattr(_core, f"_linear_mean_{family}_{operation}_{order}_{suffix}")
    args = (
        _pointer(theta),
        _pointer(model.data),
        _pointer(model.features),
        _pointer(residuals),
        _pointer(variance),
    )
    if mask != 1:
        args += (_pointer(output),)
    args += (len(model.data), model.columns)
    if order == "pq":
        args += (model.p_vol, model.q_vol)
    if model.code >= 4:
        args += ({4: np.nan, 5: -0.5, 6: 1.0, 7: np.inf}[model.code],)

    def evaluate(_keepalive=(model, theta, output, residuals, variance)):
        variance[0] = model.h0
        value = kernel(*args)
        if mask & 1:
            output[0] = value

    return evaluate, output


def measure_requests(model, theta, pairs, x):
    """Measure preallocated sparse/combined requests outside preparation and fit timing."""
    records = []
    runtime = None
    if model.family == 3:
        runtime = JointMeanModel(model.spec, model.data, x, fixed=False)
    for mask in (1, 2, 3, 4):
        native, actual = prepare_native_request(model, theta, mask)
        reference, expected = (
            prepare_native_request(runtime, theta, mask)
            if runtime is not None
            else prepare_legacy_request(model, theta, mask)
        )
        native()
        reference()
        error = float(np.max(np.abs(actual - expected)))
        parity = bool(np.allclose(actual, expected, rtol=2e-6, atol=3e-6))
        diagnostics = {}
        left, right, _ = paired_times(
            native,
            reference,
            repeats=pairs,
            target_seconds=0.001,
            diagnostics=diagnostics,
        )
        records.append(
            dict(
                mask=mask,
                point="matched start",
                maximum_absolute_error=error,
                parity=parity,
                native_seconds=left,
                reference_seconds=right,
                speedup=right / left,
                timing_diagnostics=diagnostics,
                reference=(
                    "runtime jet (no legacy APARCH kernel)"
                    if runtime is not None
                    else (
                        "legacy separate F and G kernels"
                        if mask == 3
                        else "legacy C kernel"
                    )
                ),
            )
        )
    return records


def independent_fit(model, initial, maxiter, transformed=False):
    z = model.inverse(initial) if transformed else initial.copy()
    if transformed:
        vol_low, vol_high = _coordinate_bounds(model.vol)
        low = np.r_[np.full(model.columns, np.nan), vol_low]
        high = np.r_[np.full(model.columns, np.nan), vol_high]
    else:
        low, high = model.lower.copy(), model.upper.copy()
    low[np.isnan(low)], high[np.isnan(high)] = -np.inf, np.inf

    work = model.workspace(3, transformed)
    value, gradient = np.empty(1), np.empty(model.size)
    mean_domain_rejections = 0
    invalid_objective_requests = 0
    invalid_constraint_requests = 0

    def objective(z):
        nonlocal mean_domain_rejections, invalid_objective_requests
        model.request_counts[3] += 1
        # Match the native optimizer adapter, not only its underlying recursion.
        try:
            for block in model.blocks:
                block.evaluate(z, derivatives=False, stable=True)
        except ValueError:
            mean_domain_rejections += 1
            invalid_objective_requests += 1
            return np.inf, np.full(model.size, np.nan)
        valid = _native_core._linear_joint_prevalidated(
            model.descriptor,
            _pointer(z),
            _pointer(model.data),
            _pointer(model.features),
            model.h0,
            3,
            transformed,
            _pointer(work),
            work.nbytes,
            _pointer(value),
            _pointer(gradient),
            0,
            0,
            0,
        )
        if not valid:
            invalid_objective_requests += 1
            return np.inf, np.full(model.size, np.nan)
        return value[0] / len(model.data), gradient / len(model.data)

    rows = 2 * model.domain[0] if transformed else model.rows

    def constraints(z, gradient=False):
        nonlocal invalid_constraint_requests
        try:
            if transformed:
                values = np.empty(rows)
                jacobian = np.zeros((rows, model.size))
                offset = 0
                for block in model.blocks:
                    local, derivative = block.evaluate(z, derivatives=gradient)
                    values[offset : offset + len(local)] = local
                    if gradient:
                        jacobian[
                            offset : offset + len(local),
                            block.offset : block.offset + block.width,
                        ] = derivative
                    offset += len(local)
            else:
                values, jacobian = model.constraints(z)
            return jacobian if gradient else values
        except ValueError:
            invalid_constraint_requests += 1
            return (
                np.full((rows, model.size), np.nan) if gradient else np.full(rows, -1.0)
            )

    result = minimize(
        objective,
        z,
        jac=True,
        method="SLSQP",
        bounds=Bounds(low, high),
        constraints=(
            [dict(type="ineq", fun=constraints, jac=lambda z: constraints(z, True))]
            if rows
            else ()
        ),
        options=dict(ftol=1e-12, maxiter=maxiter),
    )
    if transformed:
        result.x = np.r_[
            result.x[: model.columns],
            _coordinates_to_theta(model.vol, result.x[model.columns :]),
        ]
    result.fun = model.evaluate(result.x, 1)[0]
    result.selected_space = "z" if transformed else "theta"
    result.phases = (
        dict(
            space=result.selected_space,
            mode=result.status,
            mean_domain_rejections=mean_domain_rejections,
            invalid_objective_requests=invalid_objective_requests,
            invalid_constraint_requests=invalid_constraint_requests,
        ),
    )
    return result


def independent_complete_fit(spec, y, x, initial, maxiter, transformed=False):
    started = time.perf_counter()
    model = JointMeanModel(spec, y, x)
    initial = model.initialize(initial)
    result = independent_fit(model, initial, maxiter, transformed)
    return assemble_joint_fit(
        model, result, transformed, 1e-12, maxiter, started, backend="scipy-slsqp"
    )


def comparison_sample(seed, har, sample):
    rng = np.random.default_rng(seed)
    if sample == "normal":
        y = rng.normal(size=400)
        return y, rng.normal(size=(400, 4 if har else 2))
    if sample != "persistent":
        raise ValueError("Expected a normal or persistent comparison sample.")
    burn, n, columns = 300, 400, 4 if har else 2
    mean = HARX([1, 3]) if har else ARX(1)
    mean.set_n_exog(columns)
    truth = np.r_[
        [0.1, 0.2, 0.1] if har else [0.1, 0.25],
        np.linspace(-0.1, 0.1, columns),
        [0.06, 0.07, 0.05, 0.84, 8.0],
    ]
    exogenous = rng.normal(size=(n + burn, columns))
    generator = mean + GJRGARCH(1, 1) + StudentT()
    simulated = generator.simulate(n, truth, x=exogenous, burn=burn, seed=seed)
    return np.ascontiguousarray(simulated.data), np.ascontiguousarray(exogenous[burn:])


def assess_comparison_point(model, theta):
    """Keep finite likelihood evidence independent of derivative availability."""
    errors = {}
    try:
        value = model.evaluate(theta, 1)[0]
    except ValueError as error:
        value = np.inf
        errors["value"] = str(error)
    try:
        gradient = model.evaluate(theta, 2)[1]
    except ValueError as error:
        gradient = None
        errors["gradient"] = str(error)
    feasible, stationarity, violation = model.assess(theta, gradient)
    return dict(
        nll=value,
        feasible=bool(feasible),
        kkt_bound=stationarity,
        violation=violation,
        evaluation_errors=errors,
    )


def compare(
    family,
    density,
    har,
    pairs,
    maxiter,
    legacy_log=False,
    native_log=None,
    seed=2785,
    initial_curvature=False,
    reference_log=False,
    reference_engine="legacy",
    native_speculate=None,
    sample="normal",
    native_curvature_reset=None,
):
    if reference_engine not in ("legacy", "canonical"):
        raise ValueError("The reference engine must be 'legacy' or 'canonical'.")
    y, x = comparison_sample(seed, har, sample)
    mean = HARX([1, 3]) if har else ARX(1)
    p, q = (3, 2) if har else (1, 1)
    spec = mean + FAMILIES[family](p, q) + DENSITIES[density]()
    model = JointMeanModel(spec, y, x)
    initial = model.initialize()
    request_costs = measure_requests(model, initial, pairs, x)

    def native():
        if (
            initial_curvature
            or native_speculate is not None
            or native_curvature_reset is not None
        ):
            started = time.perf_counter()
            prepared = JointMeanModel(spec, y, x)
            start = prepared.initialize(initial)
            optimized = solve_joint(
                prepared,
                start,
                transformed=bool(native_log),
                accuracy=1e-12,
                maxiter=maxiter,
                initial_curvature=initial_curvature,
                allow_chart_refinement=native_log is None and not initial_curvature,
                speculate=native_speculate,
                curvature_reset=native_curvature_reset,
            )
            result = assemble_joint_fit(
                prepared, optimized, bool(native_log), 1e-12, maxiter, started
            )
        else:
            result = fit_joint_mean_spec(
                spec,
                y,
                x=x,
                initial_params=initial,
                maxiter=maxiter,
                log_mode=native_log,
            )
        return (
            result.params,
            result.converged,
            result.fit_info.kkt_residual,
            result.fit_info.to_dict(),
        )

    canonical_reference = family == 3 or reference_engine == "canonical"
    if canonical_reference:

        def reference():
            result = independent_complete_fit(
                spec, y, x, initial, maxiter, reference_log
            )
            return (
                result.params,
                result.fit_info.optimizer_terminated,
                result.fit_info.to_dict(),
            )

        label = f"scipy-on-identical-jet, log_mode={reference_log}"
        if family == 3:
            label += "; no legacy joint APARCH fit"
    else:
        routine = get_routine(str(spec))

        def reference():
            result = routine.fit(
                y,
                x=x,
                initial_params=initial,
                maxiter=maxiter,
                solver="slsqp",
                log_mode=legacy_log,
            )
            return result.params, result.converged, result.fit_info.to_dict()

        label = f"actual legacy routine, log_mode={legacy_log}"

    if family == 3:
        old_value = lambda theta: model.evaluate(theta, 1)[0]
    else:
        evaluator = (
            _filter_gh
            if density >= 4
            else (_filter_meanx_garch, _filter_meanx_gjr, _filter_meanx_egarch)[family]
        )
        old_value = lambda theta: -evaluator(
            model.spec, y, theta, x=x, compute_score=False, compute_hessian=False
        ).log_likelihood

    timings = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        native_result = native()
        try:
            legacy_result = reference()
        except (ValueError, RuntimeError) as error:
            theta, success, kkt, native_info = native_result
            return dict(
                family=family,
                density=density,
                har=har,
                reference=label,
                reference_engine="canonical" if canonical_reference else "legacy",
                error=f"Reference failed: {type(error).__name__}: {error}",
                native_nll=model.evaluate(theta, 1)[0],
                native_certified=success,
                native_kkt_bound=kkt,
                native_fit_info=native_info,
                initial=initial.tolist(),
                theta=theta.tolist(),
                requests=request_costs,
            )
        for repeat in range(pairs):
            record = {}
            for name, run in (("native", native), ("reference", reference))[
                :: 1 if repeat % 2 == 0 else -1
            ]:
                started = time.perf_counter()
                run()
                record[name] = time.perf_counter() - started
            timings.append(record)
    theta, success, kkt, native_info = native_result
    old_theta, old_success, reference_info = legacy_result
    native_nll = model.evaluate(theta, 1)[0]
    reference_evidence = assess_comparison_point(model, old_theta)
    reference_nll = reference_evidence["nll"]
    old_values, evaluation_errors = {}, {}
    for name, point in (("native", theta), ("reference", old_theta)):
        try:
            old_values[name] = old_value(point)
        except (ValueError, RuntimeError) as error:
            old_values[name] = None
            evaluation_errors[name] = f"{type(error).__name__}: {error}"
    old_at_native, old_at_reference = old_values["native"], old_values["reference"]
    central = sorted(timings, key=lambda pair: pair["reference"] / pair["native"])[
        len(timings) // 2
    ]
    return dict(
        family=family,
        density=density,
        har=har,
        reference=label,
        reference_engine="canonical" if canonical_reference else "legacy",
        native_nll=native_nll,
        reference_nll=reference_nll,
        legacy_at_native=old_at_native,
        legacy_at_reference=old_at_reference,
        nll_loss=native_nll - reference_nll,
        native_certified=success,
        native_kkt_bound=kkt,
        native_fit_info=native_info,
        reference_terminated=bool(old_success),
        reference_fit_info=reference_info,
        reference_feasible=reference_evidence["feasible"],
        reference_kkt_bound=reference_evidence["kkt_bound"],
        reference_violation=reference_evidence["violation"],
        reference_evaluation_errors=reference_evidence["evaluation_errors"],
        speedup=central["reference"] / central["native"],
        timings=timings,
        initial=initial.tolist(),
        theta=theta.tolist(),
        reference_theta=old_theta.tolist(),
        requests=request_costs,
        cross_evaluation_errors=evaluation_errors,
        comparison_domain=(
            "identical"
            if canonical_reference
            else "legacy restrictions differ; domain matching remains open"
        ),
        comparison_trial_validity=(
            "shared mean stability and jet validity; solver rejection policies differ"
            if canonical_reference
            else "historical adapter validity may differ"
        ),
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pairs", type=int, default=3)
    parser.add_argument("--maxiter", type=int, default=500)
    parser.add_argument("--legacy-log", action="store_true")
    parser.add_argument("--native-log", choices=("auto", "theta", "z"), default="auto")
    parser.add_argument("--seed", type=int, default=2785)
    parser.add_argument("--initial-curvature", action="store_true")
    parser.add_argument("--reference-log", action="store_true")
    parser.add_argument(
        "--reference-engine", choices=("legacy", "canonical"), default="legacy"
    )
    parser.add_argument(
        "--speculate-trials",
        dest="native_speculate",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    parser.add_argument("--family", type=int, nargs="*", default=list(range(4)))
    parser.add_argument(
        "--curvature-reset",
        dest="native_curvature_reset",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    parser.add_argument("--density", type=int, nargs="*", default=[0, 7])
    parser.add_argument("--output", type=Path)
    parser.add_argument("--sample", choices=("normal", "persistent"), default="normal")
    args = parser.parse_args()
    if args.pairs < 3 or args.maxiter < 1:
        parser.error(
            "Use at least three paired measurements and a positive iteration budget."
        )
    if hasattr(signal, "SIGALRM"):
        signal.signal(
            signal.SIGALRM,
            lambda *_: (_ for _ in ()).throw(TimeoutError("120-second hard cap")),
        )
        signal.alarm(120)
    started = time.perf_counter()
    rows = []
    destination = (
        args.output
        or Path("localdev/joint_mean") / f"integration_{time.time_ns()}.json"
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    report = dict(
        rows=rows,
        settings={
            k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()
        },
        complete=False,
        thread_environment={
            name: os.environ.get(name)
            for name in (
                "OPENBLAS_NUM_THREADS",
                "OMP_NUM_THREADS",
                "VECLIB_MAXIMUM_THREADS",
            )
        },
        binary_sha256=hashlib.sha256(
            Path(_native_core.__file__).read_bytes()
        ).hexdigest(),
        legacy_binary_sha256=hashlib.sha256(
            Path(_core.__file__).read_bytes()
        ).hexdigest(),
        source_sha256={
            str(path): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in (
                Path("scivol/_joint_mean_fit.py"),
                Path("scivol/_fit_credibility.py"),
                Path(__file__),
            )
        },
    )

    def checkpoint():
        report["seconds"] = time.perf_counter() - started
        temporary = destination.with_suffix(".tmp")
        temporary.write_text(json.dumps(report, indent=2) + "\n")
        temporary.replace(destination)

    checkpoint()
    for family in args.family:
        for density in args.density:
            for har in (False, True):
                if time.perf_counter() - started > 55:
                    break
                try:
                    row = compare(
                        family,
                        density,
                        har,
                        args.pairs,
                        args.maxiter,
                        args.legacy_log,
                        {"auto": None, "theta": False, "z": True}[args.native_log],
                        args.seed,
                        args.initial_curvature,
                        args.reference_log,
                        args.reference_engine,
                        args.native_speculate,
                        args.sample,
                        args.native_curvature_reset,
                    )
                except (ValueError, RuntimeError) as error:
                    row = dict(
                        family=family,
                        density=density,
                        har=har,
                        error=f"{type(error).__name__}: {error}",
                    )
                rows.append(row)
                checkpoint()
                if "error" in row:
                    print(row, flush=True)
                else:
                    print(
                        f"{family=} {density=} {har=} loss={row['nll_loss']:.6g} speedup={row['speedup']:.3g} certified={row['native_certified']}",
                        flush=True,
                    )
    report["complete"] = len(rows) == len(args.family) * len(args.density) * 2
    checkpoint()
    print(report["seconds"])


def run_with_deadline():
    worker = multiprocessing.get_context("spawn").Process(target=main)
    worker.start()
    worker.join(120)
    if worker.is_alive():
        worker.terminate()
        worker.join(2)
        if worker.is_alive():
            worker.kill()
            worker.join()
        raise SystemExit(
            "Joint benchmark exceeded 120 seconds; completed cases remain checkpointed."
        )
    raise SystemExit(worker.exitcode)


if __name__ == "__main__":
    run_with_deadline()
