"""Independent symbolic/MP check of the native test's emitted point jets.

Run only in an authorized verification slot:
    python unit_map_range_mp.py /absolute/path/to/test_unit_map_range

SymPy differentiates the complete scalar composition. Neither this oracle nor
the native test imports scivol or JAX. Missing dependencies or records fail.
"""

import json
import math
import subprocess
import sys

import mpmath as mp
import sympy as sp


def exact(value):
    numerator, denominator = float(value).as_integer_ratio()
    return sp.Rational(numerator, denominator)


def scalar_objective(record):
    n = len(record["theta"])
    x = sp.symbols(f"x0:{n}")
    r = sp.Symbol("r", positive=True)
    center = sp.symbols(f"a0:{n}")
    mapped = list(x)
    intercept = record["intercept"]
    if intercept < n:
        mapped[intercept] /= r
    for j in range(record["exogenous_count"]):
        axis = record["exogenous"] + j
        mapped[axis] *= exact(record["divisors"][j]) / r
    omega = record["omega"]
    if record["scale"] == 1:
        mapped[omega] /= r**2
    elif record["scale"] == 2:
        mapped[omega] *= r ** (-x[record["delta"]])
    elif record["scale"] == 3:
        beta = sum(x[record["beta"] + j] for j in range(record["beta_count"]))
        mapped[omega] -= 2 * sp.log(r) * (1 - beta)
    anchor = {r: exact(record["r"])}
    anchor.update(zip(x, map(exact, record["theta"])))
    anchor.update(zip(center, map(exact, record["theta"])))
    displacement = [v - v.xreplace(dict(zip(x, center))) for v in mapped]
    objective = sum(exact(g) * v for g, v in zip(record["inner_g"], displacement))
    objective += sum(
        exact(record["inner_h"][i * n + j]) * displacement[i] * displacement[j] / 2
        for i in range(n)
        for j in range(n)
    )
    return x, anchor, objective


def verify(record):
    # The asymmetric synthetic matrix tests linear algebra, not a scalar Hessian.
    n = len(record["theta"])
    h = record["inner_h"]
    if any(h[i * n + j] != h[j * n + i] for i in range(n) for j in range(n)):
        return False
    x, anchor, objective = scalar_objective(record)
    expected_g = [sp.diff(objective, axis) for axis in x]
    expected_h = [sp.diff(objective, a, b) for a in x for b in x]
    with mp.workdps(200):
        for actual, expression in zip(record["g"] + record["h"], expected_g + expected_h):
            assert math.isfinite(actual)
            expected = mp.mpf(str(expression.evalf(200, subs=anchor)))
            rounded = float(expected)
            if rounded == 0 or abs(rounded) < sys.float_info.min:
                assert actual == rounded, (actual, expected)
            else:
                assert abs(mp.mpf(actual) - expected) <= abs(expected) * mp.mpf("2e-13"), (actual, expected)
    return True


def main():
    assert len(sys.argv) == 2, "Pass the prebuilt native range-test executable."
    completed = subprocess.run([sys.argv[1], "--emit"], check=True, text=True, capture_output=True, timeout=30)
    records = [json.loads(line) for line in completed.stdout.splitlines() if line.startswith("{")]
    assert len(records) == 11, f"Expected 11 point records, got {len(records)}"
    assert sum(verify(record) for record in records) == 10
    print("unit range symbolic/MP: 10 complete scalar compositions at 200 digits")


if __name__ == "__main__":
    main()
