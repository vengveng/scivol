"""Small deterministic mean-jet promotion matrix, with a 120-second hard limit."""

from __future__ import annotations

import argparse
import signal
import time
import multiprocessing
import json
from pathlib import Path

import numpy as np

from scivol import ARMA, ARX, HARX, _core
from scivol._mean_fit import DENSITIES, MeanModel, fit_mean_spec, assess_solution


def paired_times(left, right, repeats=7, *, diagnostics=None, target_seconds=0.005):
    initial = []
    for operation in (left, right):
        started = time.perf_counter()
        operation()
        initial.append(time.perf_counter() - started)
    repetitions = max(1, min(100, int(target_seconds / max(initial))))
    pairs = []
    results = {}
    for repeat in range(repeats):
        pair = {}
        order = (("native", left), ("reference", right))
        ordered = order[:: 1 if repeat % 2 else -1]
        for label, operation in ordered:
            started = time.perf_counter()
            cpu_started = time.process_time()
            for _ in range(repetitions):
                results[label] = operation()
            pair[label + "_cpu"] = (time.process_time() - cpu_started) / repetitions
            pair[label] = (time.perf_counter() - started) / repetitions
        pair["first"] = ordered[0][0]
        pairs.append(pair)
    central = sorted(pairs, key=lambda pair: pair["native"] / pair["reference"])[
        len(pairs) // 2
    ]
    if diagnostics is not None:
        diagnostics.update(
            repetitions=repetitions,
            pairs=pairs,
            speedup_range=[
                min(pair["reference"] / pair["native"] for pair in pairs),
                max(pair["reference"] / pair["native"] for pair in pairs),
            ],
        )
    return central["native"], central["reference"], results


def compare_solutions(mean, density, records=None):
    rng = np.random.default_rng(1961)
    data = rng.standard_t(6, size=500) * np.sqrt(2 / 3)
    x = None if isinstance(mean, ARMA) else rng.normal(size=(len(data), 2))
    model = MeanModel(mean + density(), data, x)
    start = model.initialize(None)

    def fit(solver):
        return fit_mean_spec(model.spec, data, x=x, initial_params=start, solver=solver)

    native_time, reference_time, results = paired_times(
        lambda: fit("native"), lambda: fit("slsqp")
    )
    native, reference = results["native"], results["reference"]
    points = [
        np.r_[r.params, r._sigma2[0]] if model.code == 0 else r.params
        for r in (reference, native)
    ]
    # Independently reconstructed model plans evaluate both returned estimates.
    other = MeanModel(model.spec, data, x, fixed=False)

    def established(point):
        from scivol._evaluation import (
            _filter_arma,
            _filter_arma_ged,
            _filter_meanx_normal,
            _filter_meanx_standalone,
            _filter_gh,
        )

        options = dict(compute_score=False, compute_hessian=False)
        if isinstance(mean, ARMA) and model.code in (1, 2):
            return other.evaluate(point, 1)[0]
        if model.code >= 4:
            return -_filter_gh(model.spec, data, point, x=x, **options).log_likelihood
        if isinstance(mean, ARMA):
            if model.code == 0:
                value = -_filter_arma(
                    model.spec, data, point[:-1], **options
                ).log_likelihood
                return value
            return -_filter_arma_ged(model.spec, data, point, **options).log_likelihood
        if model.code == 0:
            return -_filter_meanx_normal(
                model.spec, data, point[:-1], x=x, **options
            ).log_likelihood
        return -_filter_meanx_standalone(
            model.spec, data, point, x=x, **options
        ).log_likelihood

    four = [
        value
        for point in points
        for value in (established(point), model.evaluate(point, 1)[0])
    ]
    np.testing.assert_allclose(four[::2], four[1::2], rtol=1e-11, atol=1e-9)
    feasible = all(model.feasible(point) for point in points)
    kkt = [assess_solution(model, p, model.evaluate(p, 2)[1])[1] for p in points]
    quality = native.converged and reference.converged and feasible and max(kkt) < 1e-6
    quality &= four[3] <= four[1] + 1e-6 and four[2] <= four[0] + 1e-6
    if records is not None:
        records.append(
            {
                "mean": mean.signature,
                "density": density.__name__,
                "native_seconds": native_time,
                "reference_seconds": reference_time,
                "four_nlls": four,
                "kkt": kkt,
                "feasible": feasible,
                "quality": bool(quality),
                "requests": native.fit_info.evaluation_counts,
                "phases": native.fit_info.phase_diagnostics,
                "dispatch": native.fit_info.kernel_dispatch,
            }
        )
    print(
        f"solve {mean.signature:16s} {density.__name__:24s} "
        f"{native_time*1e3:8.3f}/{reference_time*1e3:8.3f}ms "
        f"{reference_time/native_time:5.2f}x "
        f"F/G={native.fit_info.function_evaluations}/{native.fit_info.gradient_evaluations} "
        f"KKT={kkt[0]:.2g}/{kkt[1]:.2g} delta={four[3]-four[1]:.3g} gate={bool(quality)}",
        flush=True,
    )
    return bool(quality), reference_time / native_time


def compare_requests(density, records=None):
    rng = np.random.default_rng(293)
    names = {1: "studentt", 2: "skewt", 3: "ged"}
    marginal = {}
    for n in (128, 2048):
        data = rng.standard_t(6, size=n)
        x = rng.normal(size=(n, 2))
        model = MeanModel(ARX(1) + density(), data, x)
        point = model.initialize(None)
        residual, variance = np.empty(n), np.empty(n)
        grad, hess = np.empty(model.size), np.empty((model.size, model.size))
        base = (
            point.ctypes.data,
            data.ctypes.data,
            model.features.ctypes.data,
            residual.ctypes.data,
            variance.ctypes.data,
        )
        dimensions = (n, model.columns, 0, 0)
        repeats = 4 if model.code >= 4 else 40
        for request, operation in ((1, "nll"), (2, "nll_grad"), (4, "hess")):
            if model.code == 0:
                function = getattr(_core, f"_linear_mean_{operation}_normal")
                base = (
                    point.ctypes.data,
                    data.ctypes.data,
                    model.features.ctypes.data,
                    residual.ctypes.data,
                )
                dimensions = (n, model.columns)
            elif model.code >= 4:
                function = getattr(_core, f"_linear_mean_garch_{operation}_pq_gh")
                fixed_lambda = {4: np.nan, 5: -0.5, 6: 1.0, 7: np.inf}[model.code]
                dimensions = (n, model.columns, 0, 0, fixed_lambda)
            else:
                function = getattr(
                    _core, f"_linear_mean_garch_{operation}_pq_{names[model.code]}"
                )
            output = (
                () if request == 1 else ((grad if request == 2 else hess).ctypes.data,)
            )

            def native():
                for _ in range(repeats):
                    model.evaluate(point, request)

            def reference():
                for _ in range(repeats):
                    function(*base, *output, *dimensions)

            a, b, _ = paired_times(native, reference)
            marginal[n, request] = (a / repeats, b / repeats)
    ratios = []
    for request in (1, 2, 4):
        a, b = np.subtract(marginal[2048, request], marginal[128, request])
        ratios.append(b / a)
        if records is not None:
            records.append(
                {
                    "density": density.__name__,
                    "request": request,
                    "native_seconds": float(a),
                    "reference_seconds": float(b),
                }
            )
        print(
            f"request {density.__name__:24s} {request} "
            f"{a*1e6:.2f}/{b*1e6:.2f}us marginal {b/a:.2f}x",
            flush=True,
        )
    return min(ratios)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--requests-only", action="store_true")
    parser.add_argument(
        "--output", type=Path, help="Write the complete numerical evidence as JSON."
    )
    args = parser.parse_args()
    if hasattr(signal, "SIGALRM"):

        def expired(*_):
            raise TimeoutError("Mean benchmark exceeded its 120-second hard limit.")

        signal.signal(signal.SIGALRM, expired)
        signal.alarm(120)
    started = time.perf_counter()
    report = {"requests": [], "fits": []}
    ratios = [compare_requests(density, report["requests"]) for density in DENSITIES]
    gates = []
    if not args.requests_only:
        for mean in (ARMA(1, 1), ARX(1), HARX([1, 5, 22])):
            for density in DENSITIES:
                if time.perf_counter() - started > 60:
                    raise TimeoutError(
                        "Routine mean benchmark exhausted its 60-second target."
                    )
                quality, speed = compare_solutions(mean, density, report["fits"])
                gates.append(quality and speed >= 1.0)
    print(
        f"elapsed={time.perf_counter()-started:.2f}s "
        f"request_gate={min(ratios)>=1} solve_gate={all(gates)}"
    )
    if hasattr(signal, "SIGALRM"):
        signal.alarm(0)
    report.update(
        elapsed=time.perf_counter() - started,
        request_gate=bool(min(ratios) >= 1),
        solve_gate=all(gates),
    )
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n")
    return 0 if min(ratios) >= 1 and all(gates) else 1


def run_with_deadline():
    # A parent process enforces the cap even if native code holds the GIL.
    worker = multiprocessing.get_context("spawn").Process(target=main_worker)
    worker.start()
    worker.join(120)
    if worker.is_alive():
        worker.terminate()
        worker.join(2)
        if worker.is_alive():
            worker.kill()
            worker.join()
        raise SystemExit("Mean benchmark exceeded its 120-second hard limit.")
    raise SystemExit(worker.exitcode)


def main_worker():
    raise SystemExit(main())


if __name__ == "__main__":
    run_with_deadline()
