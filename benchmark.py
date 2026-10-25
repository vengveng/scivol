"""Fit-time benchmark: one simulated series, a panel of models, median wall time per fit.

    python benchmark.py            # n = 5000, 5 repetitions (well under a minute)
    python benchmark.py 2000 3

Prints, per model: median milliseconds, log-likelihood, iterations, and the
fit's status (success and inference). Compare at equal solution quality: a
faster fit with a lower log-likelihood is not faster.
"""

from __future__ import annotations

import sys
import time

import numpy as np

from scivol import (APARCH, ARMA, ARX, EGARCH, GARCH, GED, GJRGARCH, HARX, NormalInverseGaussian, Normal, SkewT,
                    StudentT)


def simulate(n: int, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """AR(1)-GJR(1,1) returns with skewed, fat-tailed innovations, and one regressor."""
    rng = np.random.default_rng(seed)
    raw = rng.standard_t(7, n) + 0.4 * (rng.exponential(1.0, n) - 1.0)
    z = (raw - raw.mean()) / raw.std()
    x = rng.standard_normal((n, 1))
    y, h, e = np.zeros(n), 1.0, 0.0
    for t in range(n):
        h = 0.05 + (0.04 + 0.10 * (e < 0)) * e * e + 0.85 * h
        e = np.sqrt(h) * z[t]
        y[t] = 0.05 + 0.2 * (y[t - 1] if t else 0.0) + 0.1 * x[t, 0] + e
    return 0.01 * y, x


PANEL = [
    ("GARCH(1,1)+Normal", lambda: GARCH(1, 1) + Normal(), False),
    ("GARCH(1,1)+StudentT", lambda: GARCH(1, 1) + StudentT(), False),
    ("GJR(1,1)+SkewT", lambda: GJRGARCH(1, 1) + SkewT(), False),
    ("EGARCH(1,1)+GED", lambda: EGARCH(1, 1) + GED(), False),
    ("APARCH(1,1)+StudentT", lambda: APARCH(1, 1) + StudentT(), False),
    ("GARCH(1,1)+NIG", lambda: GARCH(1, 1) + NormalInverseGaussian(), False),
    ("ARMA(1,1)+GARCH(1,1)+Normal", lambda: ARMA(1, 1) + GARCH(1, 1) + Normal(), False),
    ("ARX(1)+GJR(1,1)+StudentT", lambda: ARX(1) + GJRGARCH(1, 1) + StudentT(), True),
    ("HARX(1,5,22)+GARCH(1,1)+Normal", lambda: HARX([1, 5, 22]) + GARCH(1, 1) + Normal(), True),
]


def main(n: int = 5000, repetitions: int = 5) -> None:
    y, x = simulate(n)
    (GARCH(1, 1) + Normal()).fit(y)  # warm-up: imports and first-call costs
    print(f"n = {n}, median of {repetitions} fits")
    print(f"{'model':34s} {'ms':>8s} {'log-lik':>12s} {'iter':>5s}  status")
    for name, make, exog in PANEL:
        times, result = [], None
        for _ in range(repetitions):
            started = time.perf_counter()
            result = make().fit(y, x=x if exog else None)
            times.append(time.perf_counter() - started)
        status = f"{'certified' if result.converged else 'uncertified'}, {result.fit_info.inference_status}"
        print(f"{name:34s} {1e3 * np.median(times):8.1f} {result.log_likelihood:12.3f} {result.n_iter:5d}  {status}")


if __name__ == "__main__":
    main(*(int(a) for a in sys.argv[1:3]))
