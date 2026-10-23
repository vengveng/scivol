# Independent core conformance

These tests evaluate the new standalone core through the chief-owned
`native_api.evaluate(case, request)` bridge. The oracles and fixtures do not import
`scivol`, old test modules, native model helpers or fitted outputs. The conformance
`conftest.py` blocks old-package imports. Required dependencies are NumPy, SciPy,
mpmath, JAX with binary64 enabled, and pytest. Missing dependencies fail; required
cases do not use `importorskip` or optional dependency skips.

The suite is a transferred **prototype/incomplete** qualification set. Passing a
shard does not establish Phase 1 completion. See `fixtures/transfer_ledger.json`
for dispositions and debt, and `fixtures/source_receipt.json` for frozen sources.

## Run the suite

Build the independent test probe with the configured production compiler and linker.
Set `SCIVOLAPI_TEST_BRIDGE` to its absolute shared-library path. There is no legacy
fallback. Run in a fresh interpreter, without an imported old package:

```sh
SCIVOLAPI_TEST_BRIDGE=/absolute/path/to/jet_probe.dylib \
  python -B -m pytest -p no:cacheprovider -q tests/conformance
```

Coordinate memory and competing jobs before running JAX shards. Collection does
not initialize JAX. No slow-test marker removes mandatory cases. A scalar-only
oracle check, which does not import JAX or need the bridge, is:

```sh
python -B -m pytest -p no:cacheprovider -q \
  tests/conformance/test_garch_mathematics.py -k oracle_density_normalization
```

This is a correctness suite, not a performance benchmark. It supplies no timing,
frame, allocation, symbol-ownership or optimizer-promotion evidence.

## Fixture and helper contract

The pytest `native` fixture returns a callable `native(case, request)`. Its required
inputs are `model`, `density`, `p`, `q`, `y`, `h0`, `parameters`, `transformed`,
`first`, `stop`, and `hold_back`. Density identifiers are `normal`, `studentt`,
`skewt`, `ged`, `gh`, `nig`, `hyperbolic`, `variance_gamma`. The worker's additional
fields declare `theta_names`, expected `scored_rows`, `original_z`, chart name,
scope, normalization, units, seed provenance and expected validity.

The bridge returns `f`, `g`, `h`, `theta`, `valid_mask`, `status`, `reasons`,
`sentinel`, `dispatch`, and `scored_rows`. Status 0 means success. F/G/H bits are
1/2/4; `g` is length d and `h` is a dense d-by-d array. Unrequested outputs retain
the native output-buffer sentinel bitwise. Dispatch 0 means runtime and 1 fixed.

Theta order is omega, alpha[1:p], gamma[1:p] for GJR only, beta[1:q], then shape.
Shape names are `nu` for Student-t/GED, `nu,lambda` for Hansen Skew-t,
`rho,zeta,lambda` for GH, `rho,zeta` for NIG/Hyperbolic, and `rho,lambda` for VG.
`parameters` holds theta normally and original z when `transformed=True`.

## Statistical meaning

The observations are raw signed residuals of a zero-mean model. Every fixture has
an explicit positive binary64 seed. Physical holdback removes the original prefix;
the supplied seed is h[0] at the first retained row. That row is scored by default.
For retained t>=1 the equations are:

```text
GARCH: h[t] = omega + sum alpha[i]*e[t-i]^2 + sum beta[j]*h[t-j]
GJR:   h[t] = omega + sum (alpha[i]+gamma[i]*I(e[t-i]<0))*e[t-i]^2
                    + sum beta[j]*h[t-j]
```

Only available retained lags enter the sums. Missing earlier lags contribute zero.
`first`/`stop` select original positions, independently of physical holdback.
Earlier retained rows advance state without scoring. Expected rows, seed, theta
and sample length come from the fixture, not implementation metadata.

F is the complete additive NLL over those rows, including the Normal constant and
all shape normalizers. No objective averaging, variance floor, residual padding,
parameter clipping or derived sample seed appears in the references. The GJR shock
coefficient is combined before multiplying its squared residual, preserving genuine
cancellation. The exact power-of-two subnormal fixture avoids input-square rounding
as a competing explanation for conditioning errors.

## Oracle ownership

`oracles/garch.py` owns the scoped scalar JAX equations and AD. Prepared density
constants are outside observation assembly. JAX differentiates the declared
recursion, sufficient state and complete objective, including all shape cross
blocks. Local residual/variance derivatives are independently available; the current
public bridge validates local shape blocks through n=1, not standalone residual or
variance derivatives as native outputs.

`oracles/values.py` supplies separate SciPy density values and mpmath exact-input
scalar objectives. Student-t and GED use SciPy distributions; GH uses SciPy's own
location/scale moments. Density mass, mean and second moment are integrated on true
infinite support. MP converts each binary64 input to an exact real before products.
MP is scalar value evidence, not a replacement for primary AD derivative checks.

The GH JAX reference uses a GIG mixture with chi=psi=zeta. Its Bessel-K integral
is independently discretized on [0,16], with refinement and MP value checks.
NIG and Hyperbolic fix lambda at -0.5 and 1. VG uses the gamma-mixture limit.
Its exact centre value and second-order germ apply only to lambda>1.5; no nearby
smoothing replaces the centre. Local GED references are for smooth residual points.
The fixed-zero complete objective remains smooth in its estimated parameters and
uses the exact zero power, without padding. Joint-mean kink semantics are not claimed.

The mathematical density parameterization is checked against the
[SciPy GH documentation](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.genhyperbolic.html).
The integral reference is [DLMF 10.32.9](https://dlmf.nist.gov/10.32.E9).

## Original-z chart

`oracles/charts.py` owns an independent statement of the transferred interior chart.
Omega is `tiny + softplus(z)`. With m coefficient contributions, a softmax including
one zero slack logit is multiplied by `1 - 8*(m+1)*eps`. GARCH uses those
contributions directly. GJR positive/negative shock contributions divide by
`1-M` and `M`, where `M=E[z^2 I(z<0)]`; gamma is the negative coefficient minus
alpha. Its moment integrates conditional Normal partial second moments under the
GIG/gamma mixing law, independently of native density-space integration.

Named chart support constants are in `charts.py`: Student base `2+8*eps`, GED
base `2^-330`, rho limit 0.995, zeta bounds [0.1,25], GH lambda bounds [-6,6],
VG lambda bounds [1.5000000000000018,50]. These describe the chosen floating-point
chart, not a newly imposed statistical variance floor or universal density domain.
Shape tests separately include valid parameters below historical padding.

The returned actual mapped doubles anchor independent theta F/G/H. Original z
supplies independent Jacobian and map curvature. The pullback includes both
`J.T @ H @ J` and the theta-score contraction with map Hessians. Neither inverse
mapping rounded theta nor native G/H provides an expectation. Separate value-map
checks and endpoint-tail checks prevent that anchoring exception from hiding a bad
map. GJR shape dependence creates explicit volatility/shape cross curvature.

## Qualification boundaries

Exact density reductions are mathematical identities, not promises of identical
addition order. Their tests use a binary64 roundoff budget, `2*gamma_k/(1-gamma_k)`,
where `u=eps/2`, `gamma_k=k*u/(1-k*u)` and `k=8*n_eff+64`. F scales with independent
absolute log-variance, density-base and density-tail terms; G/H scale with the
largest absolute row-contribution norms summed over the declared scored sample.
Both evaluators supply derivative magnitudes for the budget only. They never
supply a mathematical oracle jet. The minimum scale is one. This bounded arithmetic
allowance is separate from, and much smaller than, the unchanged AD/MP tolerances.
The established native reduction test already used F/G/H tolerances of
2e-13/2e-12/2e-11; it never required bit equality. No production kernel changes
are needed to force a common summation order.

The mathematical matrix covers both families, eight densities, positive orders
11/12/21/22/32/23/43 and all seven masks. Fixed dispatch at each common order is
a separate required assertion. A missing GH/VG fixed instantiation remains a gap;
an AD pass cannot certify that specialization or justify weakening the expectation.

The public helper cannot force two implementations of the same order. Same-point
fixed/runtime parity, arbitrary tuple compile-time instantiation, binary metadata,
stack and allocation gates require native tests. The current core declares positive
orders; zero-axis equations are inventoried separately, not silently claimed as
shipped. Unsupported accuracy requests and native admission contracts belong to
the chief-owned native/build tests, not a fabricated oracle error tolerance.

Full joint-mean transfer, OPG/inference/fit behavior, complete old stress grids,
high-precision GH/VG derivative stress at retained cancellation points, VG lower
shape near-centre derivative stress and legacy-chart parity remain explicit debt.
The 1024-observation stress here covers light-density recursive sensitivities;
GH-family support-edge MP tests certify values only. No full empirical panel or
complete-fit claim follows from this suite.
