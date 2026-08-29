"""Deterministic promotion benchmark for native APARCH jets.

The benchmark stays within the routine development budget. It compares all
eight standalone density jets with the established APARCH kernels at identical
points, then compares matched-start native and independent solves for standalone
and genuine joint models.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import time
from typing import Callable

import numpy as np
from numpy.typing import NDArray

from scivol import (
    APARCH,
    ARMA,
    GED,
    GeneralizedHyperbolic,
    Hyperbolic,
    Normal,
    NormalInverseGaussian,
    SkewT,
    StudentT,
    VarianceGamma,
)
from scivol import _core, _native_core
from scivol._kernels.aparch import get_routine as get_legacy_routine
from scivol._native_fit import (
    _assess_theta_solution,
    _describe_model,
    _evaluate_theta_request,
    _initial_theta,
)


@dataclass(frozen=True, slots=True)
class DensityCase:
    name: str
    native_code: int
    legacy_code: int
    shape: tuple[float, ...]
    fixed_lambda: float = np.nan


DENSITIES = (
    DensityCase("Normal", 0, 0, ()),
    DensityCase("StudentT", 1, 1, (7.0,)),
    DensityCase("SkewT", 2, 2, (7.0, 0.12)),
    DensityCase("GED", 3, 3, (1.7,)),
    DensityCase("GeneralizedHyperbolic", 4, 4, (0.1, 2.0, -0.3)),
    DensityCase("NormalInverseGaussian", 5, 4, (0.1, 2.0), -0.5),
    DensityCase("Hyperbolic", 6, 4, (0.1, 2.0), 1.0),
    DensityCase("VarianceGamma", 7, 4, (0.1, 3.0), np.inf),
)


def _aligned_workspace(size: int) -> tuple[NDArray[np.uint8], int]:
    owner = np.empty(size + 63, dtype=np.uint8)
    return owner, (owner.ctypes.data + 63) & ~63


def _median_seconds(operation: Callable[[], None], repeats: int) -> float:
    samples = []
    operation()
    for _ in range(5):
        started = time.perf_counter()
        for _ in range(repeats):
            operation()
        samples.append((time.perf_counter() - started) / repeats)
    return float(np.median(samples))


def _request_benchmark(
    data: NDArray[np.float64],
) -> list[tuple[str, str, float, float, float, float]]:
    results = []
    initial_variance = float(np.mean(data * data))
    for case in DENSITIES:
        theta = np.ascontiguousarray(
            np.r_[[0.08, 0.10, 0.15, 0.82, 1.70], case.shape],
            dtype=np.float64,
        )
        dimension = theta.size
        value = np.empty(1, dtype=np.float64)
        gradient = np.empty(dimension, dtype=np.float64)
        hessian = np.empty((dimension, dimension), dtype=np.float64)
        variance = np.empty_like(data)
        variance[0] = initial_variance
        workspace_bytes = int(_native_core._aparch_workspace_bytes(
            1, 1, case.native_code, 0, 7
        ))
        workspace_owner, workspace = _aligned_workspace(workspace_bytes)

        def native(request: int) -> None:
            status, valid = _native_core._aparch_jet(
                theta.ctypes.data,
                dimension,
                data.ctypes.data,
                data.size,
                initial_variance,
                1,
                1,
                case.native_code,
                0,
                request,
                workspace,
                workspace_bytes,
                value.ctypes.data,
                gradient.ctypes.data,
                hessian.ctypes.data,
            )
            if status != 0 or valid != request:
                raise RuntimeError(f"native {case.name} request {request} failed")

        def established(request: int) -> None:
            if request == 1:
                value[0] = _core._aparch_nll_pq(
                    theta.ctypes.data, data.ctypes.data, variance.ctypes.data,
                    data.size, 1, 1, case.legacy_code, case.fixed_lambda,
                )
            elif request == 2:
                _core._aparch_nll_grad_pq(
                    theta.ctypes.data, data.ctypes.data, variance.ctypes.data,
                    gradient.ctypes.data, data.size, 1, 1,
                    case.legacy_code, case.fixed_lambda,
                )
            else:
                _core._aparch_nll_hess_pq(
                    theta.ctypes.data, data.ctypes.data, variance.ctypes.data,
                    hessian.ctypes.data, data.size, 1, 1,
                    case.legacy_code, case.fixed_lambda,
                )

        native(7)
        native_value = float(value[0])
        native_gradient = gradient.copy()
        native_hessian = hessian.copy()
        established(1)
        value_error = abs(native_value - float(value[0]))
        established(2)
        gradient_error = float(np.max(np.abs(native_gradient - gradient)))
        established(4)
        hessian_error = float(np.max(np.abs(native_hessian - hessian)))

        for label, request, repeats in (("F", 1, 40), ("G", 2, 40), ("H", 4, 6)):
            native_seconds = _median_seconds(lambda: native(request), repeats)
            established_seconds = _median_seconds(
                lambda: established(request), repeats
            )
            results.append((
                case.name,
                label,
                native_seconds,
                established_seconds,
                max(value_error, gradient_error, hessian_error),
                native_seconds / established_seconds,
            ))
        del workspace_owner
    return results


def _simulate_aparch(*, joint: bool) -> NDArray[np.float64]:
    rng = np.random.default_rng(91 if joint else 83)
    observations = 1_200
    nu = 7.0
    innovations = rng.standard_t(nu, size=observations) * np.sqrt((nu - 2.0) / nu)
    residuals = np.empty(observations, dtype=np.float64)
    powers = np.empty(observations, dtype=np.float64)
    data = np.empty(observations, dtype=np.float64)
    powers[0] = 1.0
    residuals[0] = innovations[0]
    data[0] = residuals[0]
    for index in range(1, observations):
        powers[index] = (
            0.08
            + 0.10 * (abs(residuals[index - 1]) - 0.15 * residuals[index - 1]) ** 1.6
            + 0.80 * powers[index - 1]
        )
        residuals[index] = powers[index] ** (1.0 / 1.6) * innovations[index]
        data[index] = residuals[index]
        if joint:
            data[index] += 0.03 + 0.18 * data[index - 1] - 0.08 * residuals[index - 1]
    return np.ascontiguousarray(data)


def _fit_benchmark() -> list[dict[str, float | str | bool]]:
    results = []
    cases = (
        ("APARCH+Normal", APARCH(1, 1) + Normal(), _simulate_aparch(joint=False)),
        (
            "ARMA+APARCH+StudentT",
            ARMA(1, 1) + APARCH(1, 1) + StudentT(),
            _simulate_aparch(joint=True),
        ),
    )
    for label, spec, data in cases:
        model = _describe_model(spec)
        start, _ = _initial_theta(spec, model, data, None)
        reference_name = "canonical-scipy" if model.joint else "established"

        def fit_native():
            return spec.fit(
                data, solver="native", initial_params=start, maxiter=500
            )

        def fit_reference():
            if model.joint:
                return spec.fit(
                    data, solver="slsqp", initial_params=start, maxiter=500
                )
            return get_legacy_routine(str(spec)).fit(
                data,
                solver="slsqp",
                log_mode=True,
                initial_params=start,
                maxiter=500,
            )

        fit_native()
        fit_reference()
        native_runs = []
        reference_runs = []
        for repeat in range(7):
            operations = (
                (("native", fit_native), ("reference", fit_reference))
                if repeat % 2 == 0
                else (("reference", fit_reference), ("native", fit_native))
            )
            for name, operation in operations:
                started = time.perf_counter()
                fitted = operation()
                elapsed = time.perf_counter() - started
                target = native_runs if name == "native" else reference_runs
                target.append((elapsed, fitted))
        native_seconds, native = sorted(native_runs, key=lambda item: item[0])[3]
        reference_seconds, reference = sorted(
            reference_runs, key=lambda item: item[0]
        )[3]

        native_at_native, native_gradient, _ = _evaluate_theta_request(
            model, data, native.params, request=3
        )
        native_at_reference, reference_gradient, _ = _evaluate_theta_request(
            model, data, reference.params, request=3
        )
        native_assessment = _assess_theta_solution(
            model, native.params, native_gradient, data, 1e-10,
            boundary_reached=False,
        )
        reference_assessment = _assess_theta_solution(
            model, reference.params, reference_gradient, data, 1e-10,
            boundary_reached=False,
        )
        results.append({
            "case": label,
            "reference": reference_name,
            "native_seconds": native_seconds,
            "reference_seconds": reference_seconds,
            "ratio": native_seconds / reference_seconds,
            "native_at_native": native_at_native,
            "native_at_reference": native_at_reference,
            "native_converged": native.converged,
            "reference_converged": bool(reference.converged),
            "native_feasible": native_assessment.model_feasible,
            "reference_feasible": reference_assessment.model_feasible,
            "native_kkt": native_assessment.kkt_residual,
            "reference_kkt": reference_assessment.kkt_residual,
        })
    return results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    data = np.ascontiguousarray(np.random.default_rng(4).normal(size=800))
    request_results = _request_benchmark(data)
    fit_results = _fit_benchmark()

    print("request density                    mask native_us established_us ratio max_error")
    for density, request, native, established, error, ratio in request_results:
        print(
            f"request {density:26s} {request:>4s} {1e6*native:9.2f} "
            f"{1e6*established:14.2f} {ratio:5.3f} {error:.3e}"
        )
    for result in fit_results:
        print(
            "fit {case:24s} ref={reference:15s} ratio={ratio:.3f} "
            "nll=({native_at_native:.9f},{native_at_reference:.9f}) "
            "kkt=({native_kkt:.3e},{reference_kkt:.3e})".format(**result)
        )

    if args.check:
        if any(result[4] > 1e-7 or result[5] > 1.0 for result in request_results):
            raise SystemExit("APARCH request parity or performance gate failed")
        for result in fit_results:
            if not result["native_converged"] or not result["native_feasible"]:
                raise SystemExit(f"native fit credibility failed: {result['case']}")
            if result["native_at_native"] > result["native_at_reference"] + 1e-7:
                raise SystemExit(f"native solution quality failed: {result['case']}")
            if result["ratio"] > 1.0:
                raise SystemExit(f"native complete-fit performance failed: {result['case']}")


if __name__ == "__main__":
    main()
