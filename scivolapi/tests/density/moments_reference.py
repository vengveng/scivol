"""Generate moments_reference.inc from independent standardized density formulas.

Python + mpmath only. Integrals use mp.quad at 55 decimal digits. mp.diff uses
a 1e-12 step for the elementary densities and 1e-17 for GH members, leaving
31 and 21 decimal digits after second differences. Both steps are small
relative to every fixture's distance from its domain boundary. No production
density code is imported. Regeneration takes minutes; the C++ test is fast.
"""

from __future__ import annotations

import argparse
from functools import lru_cache
from pathlib import Path
import time

import mpmath as mp


DESTINATION = Path(__file__).with_suffix(".inc")
POINTS = [
    (0, ()), (1, (1/8.0,)), (2, (1/8.0, 0.3)), (3, (1.5,)),
    (4, (0.2, 1.3, 0.8)), (5, (0.2, 1.3)), (6, (0.2, 1.3)), (7, (0.2, 2.3)),
    (1, (1/2.00001,)), (2, (1/3.2, 0.999)), (3, (0.3,)),
    (4, (0.95, 0.1, -0.5)), (5, (0.95, 0.02)), (6, (0.95, 0.02)), (7, (0.9, 1.50001)),
    (2, (1/2.00001, 0.999)), (3, (0.01,)), (6, (0.9999, 0.001)),
]
EXTRA_POWERS = [
    (0, (), 2, -0.95, 0.25),
    (1, (1/8.0,), 2, -0.95, 0.25),
    (2, (1/8.0, 0.3), 2, -0.95, 0.25),
    (3, (0.3,), 2, -0.95, 0.25),
    (1, (1/3.01,), 2, 0.3, 3.0),
]


def density(kind, shapes):
    if kind == 0:
        return lambda x: mp.exp(-x*x/2) / mp.sqrt(2*mp.pi), []
    if kind in (1, 2):
        nu = 1/shapes[0]
        c = mp.gamma((nu+1)/2) / (mp.sqrt(mp.pi*(nu-2))*mp.gamma(nu/2))
        if kind == 1:
            return lambda x: c * (1+x*x/(nu-2))**(-(nu+1)/2), []
        lam = shapes[1]
        a = 4*lam*c*(nu-2)/(nu-1)
        b = mp.sqrt(1+3*lam*lam-a*a)

        def pdf(x):
            u = (b*x+a) / (1+lam if b*x+a >= 0 else 1-lam)
            return b*c*(1+u*u/(nu-2))**(-(nu+1)/2)
        return pdf, [-a/b]
    if kind == 3:
        nu = shapes[0]
        scale = mp.sqrt(mp.gamma(1/nu)/mp.gamma(3/nu))
        c = nu / (2*scale*mp.gamma(1/nu))
        return lambda x: c*mp.exp(-(abs(x)/scale)**nu), [mp.mpf(0)]

    rho = shapes[0]
    if kind == 7:
        lam = shapes[1]
        theta = mp.sqrt(2)*rho/mp.sqrt(1-rho*rho)
        scale = mp.sqrt(lam*(1+theta*theta))
        alpha = mp.sqrt(theta*theta+2)*scale
        beta = theta*scale
        mu = -theta*lam/scale
        nu = lam-mp.mpf("0.5")
        c = 2*scale / (mp.sqrt(2*mp.pi)*mp.gamma(lam)*(theta*theta+2)**nu)

        def pdf(x):
            r = abs(x-mu)
            if not r:
                return c*mp.exp(beta*(x-mu))*mp.gamma(nu)*2**(nu-1)
            return c*mp.exp(beta*(x-mu))*(alpha*r)**nu*mp.besselk(nu, alpha*r)
        return pdf, [mu]

    zeta = shapes[1]
    lam = shapes[2] if kind == 4 else mp.mpf("-0.5") if kind == 5 else mp.mpf(1)
    ratio = mp.besselk(lam+1, zeta) / mp.besselk(lam, zeta)
    next_ratio = mp.besselk(lam+2, zeta) / mp.besselk(lam+1, zeta)
    kappa = ratio/zeta
    alpha2 = zeta*zeta*kappa*(1+rho*rho*zeta*(next_ratio-ratio)/(1-rho*rho))/(1-rho*rho)
    alpha = mp.sqrt(alpha2)
    beta = alpha*rho
    delta2 = zeta*zeta/(alpha2*(1-rho*rho))
    mu = -beta*delta2*kappa
    c = (alpha2*(1-rho*rho)/zeta)**lam / (mp.sqrt(2*mp.pi)*mp.besselk(lam,zeta))

    def pdf(x):
        r = mp.sqrt(delta2+(x-mu)**2)
        return c*mp.exp(beta*(x-mu))*(r/alpha)**(lam-mp.mpf("0.5"))*mp.besselk(lam-mp.mpf("0.5"),alpha*r)
    return pdf, [mu+mp.sqrt(delta2)*offset for offset in (-16,-4,-1,0,1,4,16)]


@lru_cache(maxsize=512)
def moment(kind, which, shapes, gamma, delta):
    with mp.workdps(55):
        pdf, breaks = density(kind, shapes)
        exponent = mp.mpf(2) if which == 0 else mp.mpf(1) if which == 1 else delta
        def integrand(x):
            if which == 0:
                return x*x*pdf(x) if x < 0 else mp.mpf(0)
            base = abs(x) if which == 1 else abs(x)-gamma*x
            return base**exponent*pdf(x)

        # Log-radius integration stretches algebraic tails to exponential tails.
        # This also resolves the nu -> 2+ mass without a finite tail cutoff.
        if kind in (1, 2):
            rate = 1/shapes[0]-exponent
            total = mp.mpf(0)
            for sign in (-1, 1):
                if which == 0 and sign == 1:
                    continue
                logs = [mp.log(abs(b))*rate for b in breaks if sign*b > 0]
                logs += [rate*(mp.log(1/shapes[0]-2)/2+offset) for offset in (-10,-2,0,2,10)]
                total += mp.quad(lambda u: integrand(sign*mp.exp(u/rate))*mp.exp(u/rate)/rate,
                                 sorted(set([-mp.inf, *logs, mp.mpf(0), mp.inf])))
            return total
        if kind == 3:
            nu = shapes[0]
            scale = mp.sqrt(mp.gamma(1/nu)/mp.gamma(3/nu))
            a = (exponent+1)/nu
            total = mp.mpf(0)
            for sign in (-1,1):
                if which == 0 and sign == 1:
                    continue
                def transformed(r):
                    x = scale*(a*r)**(1/nu)
                    return integrand(sign*x)*x/(nu*r)
                total += mp.quad(transformed, [0,mp.mpf("0.25"),mp.mpf("0.5"),1,2,4,mp.inf])
            return total
        bounds = sorted(set([-mp.inf, *breaks, mp.mpf(0), mp.inf]))
        return mp.quad(integrand, bounds)


def reference(kind, which, shapes, gamma, delta):
    shapes = tuple(mp.mpf(x) for x in shapes)
    params = shapes + ((mp.mpf(gamma), mp.mpf(delta)) if which == 2 else ())
    def value(*p):
        return moment(kind, which, tuple(p[:len(shapes)]), p[-2] if which == 2 else mp.mpf(0),
                      p[-1] if which == 2 else mp.mpf(1))
    answer = [value(*params)]
    step = mp.mpf("1e-12" if kind < 4 else "1e-17")
    for i in range(len(params)):
        degrees = tuple(int(j == i) for j in range(len(params)))
        answer.append(mp.diff(value, params, degrees, h=step))
    for i in range(len(params)):
        for j in range(i, len(params)):
            degrees = tuple(int(k == i)+int(k == j) for k in range(len(params)))
            answer.append(mp.diff(value, params, degrees, h=step))
    return answer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--kinds", type=int, nargs="*", default=list(range(8)))
    parser.add_argument("--output", type=Path, default=DESTINATION)
    parser.add_argument("--basic", action="store_true")
    parser.add_argument("--extra-only", action="store_true")
    args = parser.parse_args()
    mp.mp.dps = 65
    points = POINTS[:8] if args.basic else POINTS
    requests = [(kind, shapes, which, 0.2, 1.3) for kind, shapes in points for which in range(3)]
    requests = EXTRA_POWERS if args.extra_only else requests + EXTRA_POWERS
    with args.output.open("w") as output:
        output.write("// Generated by moments_reference.py; binary64 inputs promoted exactly.\n")
        output.write("constexpr Reference references[] = {\n")
        for kind, shapes, which, gamma, delta in requests:
            if kind not in args.kinds:
                continue
            start = time.monotonic()
            values = reference(kind, which, shapes, gamma, delta)
            fields = ", ".join(format(float(v), ".17g") for v in values)
            shape_text = ", ".join(format(v, ".17g") for v in shapes)
            output.write(f"    {{{kind}, {which}, {{{shape_text}}}, {gamma:.17g}, {delta:.17g}, {{{fields}}}}},\n")
            output.flush()
            print(f"kind={kind} shapes={shapes} moment={which} {time.monotonic()-start:.1f}s", flush=True)
        output.write("};\n")


if __name__ == "__main__":
    main()
