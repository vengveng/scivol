"""Fixed-input DCC density-ratio fixtures, independent of native sample/layout code."""

from __future__ import annotations

from copy import deepcopy

import numpy as np


LAWS = ("normal", "studentt")
REQUESTS = tuple(range(1, 8))
LAYOUTS = ((1, 1, 2), (1, 2, 3), (2, 1, 4), (2, 2, 2),
           (3, 2, 3), (2, 3, 5), (4, 3, 3), (1, 1, 7))
SHAPES = (np.nextafter(2.0, np.inf), 2.0 + 2.0 ** -30, 8.0,
          np.nextafter(128.0, 0.0), 128.0, np.nextafter(128.0, np.inf), 1e7, 1e15)


def observations(rows, series):
    t = np.arange(rows, dtype=float)[:, None]
    j = np.arange(series, dtype=float)[None, :]
    return np.ascontiguousarray(0.73 * np.sin((t + 1) * (j + 1) * 0.47)
                               + 0.31 * np.cos(t * 0.29 - j * 0.61))


def symmetric_target(series, *, phase=0.0):
    index = np.arange(series, dtype=float)
    loadings = np.column_stack([0.23 * np.cos(index + phase), 0.17 * np.sin(index * 0.7 + phase)])
    return np.ascontiguousarray(np.diag(0.8 + 0.13 * index) + loadings @ loadings.T)


def case(law="normal", p=1, q=1, series=2, *, fixed=False, residuals=None,
         target=None, seed=None, parameters=None, residual_history=None,
         correlation_history=None, recursion_begin=0, first=None, stop=None,
         hold_back=0, case_id=None):
    e = observations(8, series) if residuals is None else np.asarray(residuals, dtype=np.float64)
    target = symmetric_target(series) if target is None else np.asarray(target, dtype=np.float64)
    seed = symmetric_target(series, phase=1.3) if seed is None else np.asarray(seed, dtype=np.float64)
    if parameters is None:
        parameters = [0.11 / p] * p + [0.57 / q] * q + ([7.4] if law == "studentt" else [])
    stop = len(e) if stop is None else stop
    first = hold_back + recursion_begin if first is None else first
    names = tuple([f"a[{i + 1}]" for i in range(p)] + [f"b[{i + 1}]" for i in range(q)]
                  + (["nu"] if law == "studentt" else []))
    return dict(
        case_id=case_id or f"dcc-{p}{q}-N{series}-{law}-{'fixed-flag' if fixed else 'runtime'}",
        law=law, p_corr=p, q_corr=q, series=series, fixed=bool(fixed),
        residuals=np.ascontiguousarray(e).copy(), target=np.ascontiguousarray(target).copy(),
        seed=np.ascontiguousarray(seed).copy(),
        residual_history=np.ascontiguousarray(np.empty((0, series)) if residual_history is None else residual_history,
                                              dtype=np.float64).copy(),
        correlation_history=np.ascontiguousarray(np.empty((0, series, series)) if correlation_history is None
                                                 else correlation_history, dtype=np.float64).copy(),
        parameters=np.asarray(parameters, dtype=np.float64).copy(), theta_names=names,
        hold_back=hold_back, stop=stop, recursion_begin=recursion_begin,
        first_scored=first - hold_back, scored_positions=tuple(range(first, stop)),
        scope="fixed_input_density_ratio", normalization="additive_nll",
        standardized_residuals="external_fixed", target_policy="external_fixed",
        seed_policy="external_fixed", history_order="nearest_first_at_recursion_begin",
        expected_valid_mask=7, constant_mode="jax",
    )


def ordinary_cases():
    return tuple(case(law, p, q, series, fixed=fixed) for p, q, series in LAYOUTS
                 for law in LAWS for fixed in (False, True))


def history_case(law="normal", p=3, q=2, series=3, *, fixed=False):
    shocks = observations(p, series)[::-1].copy()
    histories = np.stack([symmetric_target(series, phase=0.4 + lag) * (1 + 0.1 * lag)
                          for lag in range(q)])
    return case(law, p, q, series, fixed=fixed, residual_history=shocks,
                correlation_history=histories, recursion_begin=1, first=5, stop=7,
                hold_back=1, case_id=f"dcc-history-{p}{q}-N{series}-{law}-{fixed}")


def identity_case(law="normal", series=2, *, nu=8.0, zero=False):
    e = np.zeros((3, series)) if zero else observations(3, series)
    return case(law, series=series, residuals=e, target=np.eye(series), seed=np.eye(series),
                parameters=[0.0, 0.0] + ([nu] if law == "studentt" else []),
                case_id=f"dcc-identity-N{series}-{law}-nu{nu}-zero{zero}")


def shape_case(nu, p=1, q=1, series=2):
    result = case("studentt", p, q, series, residuals=observations(3, series),
                  case_id=f"dcc-shape-{p}{q}-N{series}-nu{nu!r}")
    result["parameters"][-1] = nu
    result["constant_mode"] = "mp_germ"
    return result


def scored_suffix(template, first, stop=None):
    result = deepcopy(template)
    stop = result["stop"] if stop is None else stop
    result.update(first_scored=first - result["hold_back"], stop=stop,
                  scored_positions=tuple(range(first, stop)),
                  case_id=result["case_id"] + f"-score{first}:{stop}")
    return result


def long_case(law):
    result = case(law, p=2, q=2, series=2, residuals=observations(192, 2), first=8,
                  parameters=[1e-4, 5e-5, 0.9989, 0.0008] + ([8.0] if law == "studentt" else []),
                  case_id=f"dcc-long-near-persistence-{law}")
    return result


def exact_sum_cases():
    half_below = np.nextafter(0.5, 0.0)
    gap = 2.0 ** -54
    points = (
        ("below-double-rounding", [0.5, half_below, 0.0], True),
        ("below-extended-rounding", [0.5, half_below, np.nextafter(gap, 0.0)], True),
        ("exact-boundary", [0.5, half_below, gap], False),
        ("above-boundary", [0.5, half_below, np.nextafter(gap, np.inf)], False),
        ("large-first", [0.5, 0.25, np.nextafter(0.25, 0.0)], True),
    )
    return tuple(case("normal", p=2, q=1, residuals=np.zeros((3, 2)),
                      target=np.eye(2), seed=np.zeros((2, 2)), parameters=point,
                      case_id=f"dcc-sum-{label}-{'admit' if valid else 'reject'}")
                 for label, point, valid in points)


HALF_BELOW = float.fromhex("0x1.fffffffffffffp-2")
GAP_BELOW = float.fromhex("0x1.fffffffffffffp-55")
SIMPLEX_ANCHOR_POINTS = (
    ("ordinary-tie", (0.1, 0.4), "0x1p-1"),
    ("tiny-exact", (0.5, HALF_BELOW, GAP_BELOW), "0x1p-107"),
    ("tiny-rounded-up", (0.5, HALF_BELOW, GAP_BELOW, 2.0 ** -161), "0x1p-107"),
    ("tiny-rounded-down", (0.5, HALF_BELOW, GAP_BELOW, 3.0 * 2.0 ** -161), "0x1.ffffffffffffep-108"),
    ("boundary", (0.5, HALF_BELOW, 2.0 ** -54), "0x0p+0"),
)


def simplex_anchor_case(law="normal", kind="exact", *, fixed=False):
    coefficients = [0.5, HALF_BELOW, GAP_BELOW]
    if kind == "rounded-up":
        coefficients.append(2.0 ** -161)
    elif kind == "rounded-down":
        coefficients.append(3.0 * 2.0 ** -161)
    elif kind != "exact":
        raise ValueError(kind)
    result = case(law, p=1, q=len(coefficients) - 1, fixed=fixed,
                  target=np.array([[1.0, 0.25], [0.25, 1.5]]) * (2.0 ** 107),
                  seed=np.array([[1.0, -0.2], [-0.2, 0.8]]),
                  residuals=[[0.3, -0.5], [0.7, 0.2]],
                  parameters=coefficients + ([8.0] if law == "studentt" else []),
                  case_id=f"dcc-simplex-anchor-{kind}-{law}-fixed{fixed}")
    result["simplex_anchor_kind"] = kind
    if law == "studentt":
        result["constant_mode"] = "mp_germ"
    return result


def extreme_student_hessian_case(*, fixed=False):
    return case("studentt", fixed=fixed, target=[[1.0, 0.5], [0.5, 1.0]],
                seed=[[1.0, -0.1], [-0.1, 1.0]], residual_history=[[1.0, 1.0]],
                residuals=[[1e154, 0.0]], parameters=[0.1, 0.4, 8.0],
                case_id=f"dcc-student-extreme-hessian-fixed{fixed}")


def expected_dispatch(declared):
    return int(declared["fixed"] and declared["p_corr"] == declared["q_corr"] == 1
               and declared["series"] == 2)
