"""Regenerate bessel_reference.inc with an independent mpmath oracle.

Run from any directory with Python and mpmath >= 1.4.1. The inputs are binary64
values, promoted exactly, so references describe the C++ inputs themselves.
Order derivatives use mp.diff; argument derivatives use adjacent K orders and
the differential equation. Overflow is retained as infinity in the fixtures.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
import random
import re

import mpmath as mp


DESTINATION = Path(__file__).with_suffix(".inc")
ORDERS = [0, 1e-8, -1e-8, 0.25, 0.5, -0.5, 1, 1.5, 2, 2.5, 3,
          7.3, 20, 60, 100, -37.5]
ARGUMENTS = [1e-300, 1e-100, 1e-10, 1e-3, 0.1, 0.5, 1, 1.9,
             2, 2.1, 5, 20, 100, 700, 1e4, 1e8]


def reference(nu: float, z: float, product: bool) -> list[mp.mpf]:
    a, x = mp.mpf(nu), mp.mpf(z)
    if x == 0:
        value = mp.loggamma(a)+(a-1)*mp.log(2)
        dz = -mp.mpf(1) if a == mp.mpf("0.5") else (mp.mpf(0) if a > mp.mpf("0.5") else -mp.inf)
        dzz = (mp.mpf(0) if a == mp.mpf("0.5") else
               (-1/(2*(a-1)) if a > 1 else (-mp.inf if a > mp.mpf("0.5") else mp.inf)))
        return [value, dz, dzz, mp.digamma(a)+mp.log(2), mp.polygamma(1, a),
                mp.mpf(0) if a > mp.mpf("0.5") else -mp.inf]

    def log_k(v: mp.mpf) -> mp.mpf:
        return mp.log(mp.besselk(v, x))

    value = log_k(a)
    dnu = mp.diff(log_k, a)
    dnunu = mp.diff(log_k, a, 2)
    if product:
        slope = lambda v: -mp.besselk(v-1, x)/mp.besselk(v, x)
        dz = slope(a)
        dzz = 1+(2*a-1)*(dz/x)-dz*dz
        value += a*mp.log(x)
        dnu += mp.log(x)
    else:
        # The explicitly even ratio avoids mp.diff amplifying cancellation
        # in v/x-K_(v+1)/K_v at zero order and extremely small arguments.
        slope = lambda v: -(mp.besselk(v-1, x)+mp.besselk(v+1, x))/(2*mp.besselk(v, x))
        dz = slope(a)
        dzz = 1+(a/x)**2-dz/x-dz*dz
    return [value, dz, dzz, dnu, dnunu, mp.diff(slope, a)]


def extended_reference(nu: float, z: float, function: int) -> list[mp.mpf]:
    # Curvature and the gap need guard digits for both huge and tiny z.
    with mp.workdps(max(90, 90+2*abs(int(math.log10(z))) if z else 90)):
        a, x = mp.mpf(nu), mp.mpf(z)
        if x == 0:
            values = reference(nu, z, True)
            values[1] += 1
            return values
        if function != 4:
            product = function == 3
            values = reference(nu, z, product)
            values[0] += x
            values[1] += 1
            return values

        def ratio(v: mp.mpf) -> mp.mpf:
            return mp.besselk(v+1, x)/mp.besselk(v, x)

        def gap(v: mp.mpf) -> mp.mpf:
            return ratio(v+1)-ratio(v)

        def slope(v: mp.mpf) -> mp.mpf:
            r, s = ratio(v), ratio(v+1)
            rp = r*r-(2*v+1)*r/x-1
            sp = s*s-(2*v+3)*s/x-1
            return (sp-rp)/(s-r)

        r, t = ratio(a), ratio(a+1)
        rp = r*r-(2*a+1)*r/x-1
        tp = t*t-(2*a+3)*t/x-1
        rpp = (2*r-(2*a+1)/x)*rp+(2*a+1)*r/(x*x)
        tpp = (2*t-(2*a+3)/x)*tp+(2*a+3)*t/(x*x)
        dz = (tp-rp)/(t-r)
        loggap = lambda v: mp.log(gap(v))
        return [loggap(a), dz, (tpp-rpp)/(t-r)-dz*dz,
                mp.diff(loggap, a), mp.diff(loggap, a, 2), mp.diff(slope, a)]


def boundary_points() -> set[tuple[float, float]]:
    points = {(nu, z) for nu in [-2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 60.0,
                                math.nextafter(128, 0), 128.0, math.nextafter(128, math.inf),
                                math.nextafter(-130, -math.inf), -130.0]
              for z in [math.nextafter(0.5, 0), math.nextafter(0.5, 1),
                        math.nextafter(64, 0), math.nextafter(64, math.inf)]}
    points |= {(nu, z) for nu in [math.nextafter(-1.5, -math.inf), -1.5,
                                 math.nextafter(-1.5, math.inf),
                                 math.nextafter(0.5, 0), math.nextafter(0.5, 1)]
               for z in [math.nextafter(0, 1), 1e-310, 1e-100, 1e-10]}
    return points


def write_extensions() -> None:
    # Preserve every existing witness, and extend E to negative and zero orders.
    points = {(float(nu), float(z)) for nu in ORDERS for z in ARGUMENTS}
    old_text = DESTINATION.read_text().split("// Scaled and ratio-gap references")[0]
    for nu, z in re.findall(r"\{(?:true|false), ([^,]+), ([^,]+), ", old_text):
        if float(z) > 0:
            points.add((float(nu), float(z)))
    for nu in [-60, -20, -2, -1.5, -1-1e-8, -1, -1+1e-8,
               -0.5-1e-8, -0.5, -0.5+1e-8, -1e-8, 0, 1e-8, 0.5, 1, 20, 60]:
        for z in [1e-10, 1e-3, 0.1, 2, 2.000001, 10, 63.999999, 64,
                  100, 1e3, 1e5, 1e6]:
            points.add((float(nu), float(z)))
    points |= boundary_points()
    rng = random.Random(168492)
    for _ in range(80):
        points.add((rng.uniform(-60, 60), 10**rng.uniform(-10, 6)))
    for nu in [-1000, -200, -130, -129, 128, math.nextafter(128, math.inf), 129, 200, 1000]:
        for z in [1e-10, 0.5, 2, 10, 100, 1e4, 1e6]:
            points.add((float(nu), float(z)))
    lines = [old_text.rstrip(), "", "// Scaled and ratio-gap references, mpmath >= 90 decimal digits.",
             "// function: 2 = S, 3 = E, 4 = G", "constexpr Reference extended_references[] = {"]
    ordered = sorted(points, key=lambda p: (p[1], p[0]))
    total = 3*len(ordered)
    for i, (nu, z) in enumerate(ordered):
        for function in [2, 3, 4]:
            values = extended_reference(nu, z, function)
            lines.append("    {"+str(function)+", "+literal(nu)+", "+literal(z)+", {"
                         +", ".join(map(literal, values))+"}},")
        if (i+1) % 25 == 0:
            print(f"{3*(i+1)}/{total} extended reference points", flush=True)
    for nu in [0.25, 0.5, math.nextafter(0.5, 1), 0.75, 1, math.nextafter(1, 2), 1.5, 2, 60]:
        values = extended_reference(nu, 0, 3)
        lines.append("    {3, "+literal(nu)+", 0.0, {"+", ".join(map(literal, values))+"}},")
    lines.append("};")
    DESTINATION.write_text("\n".join(lines)+"\n")
    print(f"Wrote {total+9} extended reference rows", flush=True)


def literal(x: mp.mpf | float) -> str:
    f = float(x)
    if math.isnan(f):
        return "nan"
    if math.isinf(f):
        return "inf" if f > 0 else "-inf"
    return repr(f)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--extensions-only", action="store_true")
    options = parser.parse_args()
    mp.mp.dps = 90
    if options.extensions_only:
        write_extensions()
        return
    points = {(float(nu), float(z), False) for nu in ORDERS for z in ARGUMENTS}
    points |= {(float(nu), float(z), True) for nu in ORDERS if nu > 0 for z in ARGUMENTS+[0]}
    for nu in [math.nextafter(0.5, 0), math.nextafter(0.5, 1),
               math.nextafter(1, 0), math.nextafter(1, 2), 1-1e-8, 1+1e-8,
               math.nextafter(1.5, 0), math.nextafter(1.5, 2), 1.5-1e-8, 1.5+1e-8]:
        for z in [0, 1e-300, 1e-100, 1e-10, 1e-3, 0.1, 2]:
            points.add((nu, z, True))
            if z:
                points.add((nu, z, False))
    rng = random.Random(748291)
    for _ in range(60):
        nu = rng.uniform(-100, 100)
        z = 10**rng.uniform(-300, 8)
        points.add((nu, z, False))
        points.add((abs(nu), z, True))
    for _ in range(60):
        nu, z = rng.uniform(-5, 5), 10**rng.uniform(-1.3, 1.7)
        points.add((nu, z, False))
        points.add((abs(nu), z, True))
    # Independent witnesses for Debye, underflow, and both dispatch boundaries.
    for nu in [math.nextafter(128, 0), math.nextafter(128, math.inf), 200, 1000]:
        for z in [1e-300, 0.5, 2, 10, 100, 1e4, 1e8]:
            points.add((nu, z, False))
            points.add((nu, z, True))
    for nu in [0, 0.5, 1, 2.5, 7.3]:
        for z in [math.nextafter(2, 0), math.nextafter(2, math.inf),
                  math.nextafter(64, 0), 64, math.nextafter(64, math.inf)]:
            points.add((nu, z, False))
            if nu:
                points.add((nu, z, True))
    for nu in [0, 1e-8, 0.25, 0.5, math.nextafter(0.5, 1), 0.75, 1, 1.5, 2, 7.3, 100]:
        for z in [math.nextafter(0, 1), 1e-310, 1e300]:
            points.add((nu, z, False))
            if nu:
                points.add((nu, z, True))
    lines = ["// Generated by bessel_reference.py, mpmath 1.4.1, 90 decimal digits.",
             "// product, nu, z, {value, z, zz, nu, nunu, nuz}",
             "constexpr Reference references[] = {"]
    ordered = sorted(points, key=lambda p: (p[2], p[1], p[0]))
    for i, (nu, z, product) in enumerate(ordered):
        try:
            values = reference(nu, z, product)
        except Exception:
            print(f"Oracle failed at nu={nu!r}, z={z!r}, product={product}", flush=True)
            raise
        lines.append("    {"+str(product).lower()+", "+literal(nu)+", "+literal(z)+", {"
                     +", ".join(map(literal, values))+"}},")
        if (i+1) % 100 == 0:
            print(f"{i+1}/{len(ordered)} reference points", flush=True)
    lines.append("};")
    DESTINATION.write_text("\n".join(lines)+"\n")
    print(f"Wrote {len(ordered)} reference rows to {DESTINATION}")
    write_extensions()


if __name__ == "__main__":
    main()
