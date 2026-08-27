# scivol Library Guide for AI Agents

**Last Updated:** 2026-08-27
**Purpose:** Essential architectural rules, patterns, and constraints for developing the scivol time series volatility modeling library.

**Reference implementations:** `localdev/arma_garch_estimator.py` contains verified Python+Numba implementations with analytical gradients/Hessians for ARMA-GARCH models (Normal, Student-t, Skew-t). `localdev/gjr_garch_estimator.py` contains the GJR-GARCH reference implementation. Use as ground truth when porting to C.

**Derivative validation:** Use AD-based development oracles as the primary validation path. Prefer `scivol._devtools.ad_oracle` and the shipped AD-backed tests over finite differences. Keep finite differences only as a coarse backup smoke test when an AD oracle is not yet available.

**Benchmarking:** Routine development benchmarks should finish within 60 seconds and must stop at 120 seconds. Use the smallest representative panel that answers the question. Reserve full CRSP runs for explicit whole-engine or tensor-scale replacements and release gates, never ordinary code changes. `benchmark_optimizers.py` remains the solver-policy benchmark.

**Local development:** All development scripts, experiments, and analysis code should go in `localdev/`. This folder is git-ignored. Directories prefixed with `localdev_*` are also ignored (e.g., `localdev_data/`, `localdev_results/`).

**Bug-fix ledger:** Keep the git-ignored `bugfix.md` evergreen. Whenever a confirmed bug is fixed, append its user impact, root cause, fix, and regression coverage. Do not remove older entries.

---

## Table of Contents

1. [User-Facing Architecture](#1-user-facing-architecture)
2. [C Extension Interface Rules](#2-c-extension-interface-rules)
3. [Development Standards](#3-development-standards)
4. [Internal Implementation](#4-internal-implementation)
5. [Type Safety Requirements](#5-type-safety-requirements)
6. [Build System](#6-build-system)
7. [Quick Reference](#7-quick-reference)

---

## 1. User-Facing Architecture

### ✅ Established Pattern (Keep This)

The user interface is based on **component composition** with a universal `.fit()` method. This pattern is intuitive and proven.

#### Component Composition

```python
# Components represent model parts
from scivol import GARCH, GJRGARCH, ARMA, Normal, StudentT

# Simple models
spec = GARCH(1, 1)                           # Auto-adds Normal() density
spec = GARCH(1, 1) + StudentT()              # Explicit density

# Asymmetric volatility (GJR-GARCH)
spec = GJRGARCH(1, 1) + Normal()             # GJR-GARCH with Normal
spec = GJRGARCH(1, 1) + StudentT()           # GJR-GARCH with Student-t

# Composite models
spec = ARMA(1, 1) + GARCH(1, 1)              # Mean + Volatility
spec = ARMA(1, 1) + GARCH(1, 1) + StudentT() # Full specification

# Universal interface
result = spec.fit(data)

# QMLE with robust (sandwich) standard errors
result = spec.fit(data, method='qmle')
print(result.std_errors_robust)
```

#### QMLE (Quasi-Maximum Likelihood Estimation)

QMLE provides robust standard errors that are valid even when the distributional assumption is wrong. It is invoked via the `method` parameter on `spec.fit()`:

```python
from scivol import GARCH, GJRGARCH, Normal, StudentT, SkewT

# Normal: robust SEs for all GARCH parameters
spec = GARCH(1, 1) + Normal()
result = spec.fit(data, method='qmle')
print(result.std_errors)        # MLE standard errors
print(result.std_errors_robust) # Sandwich (robust) standard errors

# Student-t: two-step procedure
# Step 1: Fit GARCH with Normal LL → robust SEs for GARCH params
# Step 2: Fix GARCH, fit nu → MLE SE for nu
spec = GARCH(1, 1) + StudentT()
result = spec.fit(data, method='qmle')
# result.std_errors_robust[0:3] = robust SEs for [omega, alpha, beta]
# result.std_errors_robust[3]   = MLE SE for nu

# Skew-t: same two-step procedure
spec = GARCH(1, 1) + SkewT()
result = spec.fit(data, method='qmle')
# result.std_errors_robust[0:3] = robust SEs for [omega, alpha, beta]
# result.std_errors_robust[3:5] = MLE SEs for [nu, lambda]

# GJR-GARCH also supported with QMLE
spec = GJRGARCH(1, 1) + Normal()
result = spec.fit(data, method='qmle')

# Joint ARMA-GARCH also supported
spec = ARMA(2, 1) + GARCH(1, 2) + Normal()
result = spec.fit(data, method='qmle')
```

The sandwich covariance is: `V_robust = H^{-1} @ OPG @ H^{-1}` where H is the Hessian and OPG is the outer product of gradients.

#### Optimization Path Transparency

Default `log_mode` is allowed to differ across model families and densities when benchmarks justify it, but the effective path must never be silent in user-visible outputs. Fits should expose the effective solver/path via `result.fit_info` so users can see whether a result came from theta-space or z-space optimization.

#### Component System

**Roles** (`scivol/roles.py`):
- `MEAN`: Mean equation components (e.g., ARMA)
- `VOLATILITY`: Volatility components (e.g., GARCH, GJRGARCH)
- `DENSITY`: Conditional distributions (e.g., Normal, StudentT, SkewT)
- `CORRELATION`: Multivariate correlation components (e.g., DCC)

**Canonical ordering**: MEAN → VOLATILITY → DENSITY

**Auto-injection**: If no density is specified, `Normal()` is added automatically.

#### Composition Operators

Multiple syntaxes for the same operation:

```python
spec = garch + arma          # __add__
spec = garch < arma          # __lt__
spec = garch << arma         # __lshift__
spec = arma >> garch         # __rlshift__
```

All create a `CompositeSpec` that composes components by role.

#### Component Contract

Every component must implement:

```python
class MyComponent(Component):
    role = Role.VOLATILITY  # or MEAN, DENSITY
    
    @property
    def signature(self) -> str:
        """Unique identifier, e.g., 'GARCH(1,1)'"""
        
    @property
    def n_params(self) -> int:
        """Number of parameters in this component"""
    
    def default_start(self) -> np.ndarray:
        """Initial parameter values for optimization"""
    
    def bounds(self) -> List[Tuple[float, float]]:
        """Parameter bounds for constrained optimization"""
    
    def pack(self, params_dict: Dict[str, Any]) -> np.ndarray:
        """Convert dict to flat array"""
    
    def unpack(self, flat_params: np.ndarray) -> Dict[str, Any]:
        """Convert flat array to dict, store in self.fitted_params"""
```

---

## 2. C Extension Interface Rules

### 🔴 Critical Constraints (Must Always Follow)

The C extension interface uses **zero-copy pointer passing** for maximum performance. Violations will cause segfaults or silent data corruption.

#### Memory Management Pattern

**Golden Rule**: Python manages memory, C only computes.

```python
# ✅ CORRECT: Pre-allocate in Python, pass pointer to C
resid2 = np.ascontiguousarray(resid**2, dtype=np.float64)
sigma2 = np.empty(len(resid), dtype=np.float64)  # Pre-allocated
resid2_ptr = resid2.ctypes.data
sigma2_ptr = sigma2.ctypes.data

# Call C function (modifies sigma2 in-place)
_core._garch_variance_11(theta_ptr, resid2_ptr, sigma2_ptr, n)

# ❌ WRONG: Returning new array from C (not how this works)
sigma2 = _core._garch_variance_11(...)  # Will fail
```

#### Array Requirements

**All arrays passed to C must be:**

1. **C-contiguous**: Use `np.ascontiguousarray(arr, dtype=np.float64)`
2. **Float64**: Always `dtype=np.float64`
3. **Pre-allocated**: Output arrays must exist before C call

**Helper function pattern**:

```python
def _as_cptr(arr: NDArray[np.float64]) -> int:
    """Convert NumPy array to C pointer (as integer address)."""
    return np.ascontiguousarray(arr, dtype=np.float64).ctypes.data
```

#### C Function Signatures

**Naming convention**: `{model}_{operation}_{pq|11}_{distribution}`

Examples:
- `_garch_variance_11` - GARCH(1,1) variance (specialized)
- `_garch_variance_pq` - GARCH(p,q) variance (generic)
- `_garch_ll_11_normal` - GARCH(1,1) log-likelihood with Normal
- `_garch_ll_grad_pq_studentt` - GARCH(p,q) gradient with Student-t
- `_gjr_garch_variance_11` - GJR-GARCH(1,1) variance (takes raw residuals)
- `_gjr_garch_ll_11_normal` - GJR-GARCH(1,1) NLL with Normal
- `_gjr_garch_ll_grad_11_studentt` - GJR-GARCH(1,1) gradient with Student-t

**Note:** GJR-GARCH functions take **raw residuals** (not squared) because the indicator I(ε<0) needs the sign.

**Parameter ordering**:
1. Parameter array pointer (`theta_ptr`)
2. Input data pointers (`resid2_ptr`, `eps2_ptr`)
3. Working buffer pointers (`sigma2_ptr`)
4. Output pointers (`grad_ptr`, `hess_ptr`)
5. Scalar sizes (`n`, `p`, `q`)

**Return types**:
- `None` for in-place modification functions
- `float` for scalar log-likelihood functions

#### Type Stubs

All C functions must have type stubs in `scivol/_core.pyi`:

```python
# Type alias for pointer-as-int
_IntPtr = int
_Size = int

def _garch_variance_11(
    theta_ptr: _IntPtr,    # [omega, alpha, beta]
    resid2_ptr: _IntPtr,   # Squared residuals
    sigma2_ptr: _IntPtr,   # Output: conditional variances (modified in-place)
    n: _Size,              # Number of observations
) -> None:
    """GARCH(1,1) variance recursion (optimized)"""
    ...
```

#### Performance Optimizations in C

**Compiler attributes** (in C source):
- `__attribute__((hot))` - Likely to be called frequently
- `__attribute__((flatten))` - Inline all calls
- `__restrict` - Pointer aliasing hints
- `VLK_FORCE_INLINE` - Force inline small helpers

**Legacy specialized vs generic kernels**:
- Provide specialized `_11` functions for GARCH(1,1) (most common)
- Provide generic `_pq` functions for arbitrary orders
- Python code tries specialized first, falls back to generic

### Native Tensor-Jet Completion Contract

This is a release gate, not a design suggestion. A jet is **incomplete** unless it
satisfies every rule below. Here, "general order" or `_pq` means every declared
order axis of the model, such as `(p, q)` for GARCH and
`(p_ar, q_ma, p_vol, q_vol)` for a joint mean-volatility model.

1. **One general-order source of mathematical truth**
   - Implement and validate a runtime general-order jet before calling any fixed
     order complete. A finite set such as 11, 12, 21, and 22 is not `_pq` support.
   - The same model policies must be instantiable with compile-time orders. Fixed
     kernels are generated specializations of the general mathematics, not separate
     handwritten recursions or likelihoods.
   - Fixed/runtime parity is evidence, not permission to duplicate equations.
     Their recursion and observation assembly must have one shared policy owner;
     traversal and storage adapters may differ around that owner.
   - Precompile only measured common orders. Dispatch once at the outer ABI to a
     fixed specialization when available and otherwise use the runtime jet.
   - A runtime-only prototype is also incomplete until the same policy composition
     can compile a fixed order. "Supports specialization" means build-time
     instantiation for any valid order tuple, not a hard-coded 11 implementation.

2. **One canonical jet shape**
   - Every tensor entry separates parameter layout, sample/conditioning plan,
     parameter domain, prepared point state, mean-volatility recursion, local
     density derivatives, theta-space assembly, and any coordinate-map pullback.
   - The local density contract consumes the model's local sufficient state, such
     as residual and variance. Compile-time dependency masks may remove inactive
     sensitivities for zero-mean or otherwise simpler models; they may not create a
     second copy of the density mathematics.
   - Density and recursion policies may vary; the request and assembly contract may
     not. All seven nonempty `F`/`G`/`H` request masks must work without changing
     the mathematical result.
   - Fixed and runtime paths may differ in traversal, indexing, and storage only.
     They must share the objective, constants, scored observations, conditioning,
     domain, parameter order, and derivatives.
   - A declared joint model must be jointly estimated in the mathematics, not only
     composed in its name or API. Mean parameters affect residuals; those residuals
     affect the variance recursion and density; and exact derivatives include every
     cross block among mean, volatility, and density-shape parameters. A staged or
     fixed-residual fit must be labeled as such and may not satisfy a joint surface.

3. **Exact additive likelihood**
   - `F` is the complete additive negative log-likelihood on the declared scored
     sample. It includes every parameter-independent and parameter-dependent
     normalization constant. Constants may never be omitted merely because they do
     not change the minimizer.
   - The jet does not divide by `n_eff`. Exact positive scaling, normally
     `1 / n_eff`, belongs in a typed optimizer adapter and must be reversed in
     likelihood reporting, inference, and matched-objective comparisons.
   - The sample plan owns initial histories, the first scored index, `n_eff`, and
     hold-back rules. Every request, order path, oracle, and production comparison
     uses that same plan.

4. **Request-minimal preparation and reuse**
   - Prepare expensive point-dependent primitives once per parameter point and
     reuse them across observations and all consumers at that point. Examples
     include `lgamma`, digamma/trigamma, Bessel ratios, GH moments, powers, and
     transform state.
   - Compute only the constants and derivatives required by the request. A pure
     `G` or `H` request need not form an unused value constant, but it must compute
     every derivative of that constant required by the requested output.
   - A combined request specifies outputs, not traversal count. A typed objective
     may map `F|G` or another mask to fused or separate specialized passes. Choose
     that request plan from objective and complete-solve measurements, preserve
     identical outputs, and keep the unchosen plan only as measured development
     debt rather than assuming one traversal is faster.
   - Preparation must have an explicit derivative order. Value and first-order
     requests may not construct, copy, or downgrade a second-order point cache.
   - Parameter-independent sample constants are formed once per solve or added once
     per request as an `n_eff` multiple. No distribution constant or other
     point-invariant primitive is recomputed in the observation loop.

5. **Exact sparse derivatives**
   - The theta jet provides exact analytical `F`, `G`, and `H`. An `H` path may
     internally require first-order state, but it must not compute or write an
     unrequested public output.
   - Every shipped transformed path differentiates the complete function
     `L(theta(z))`. Its Hessian includes transform curvature, not only
     `J' H_theta J`.
   - Non-differentiable points have explicit request-specific validity semantics.
     Never return a convenient derivative for a nearby smoothed, clipped, or floored
     model.

6. **Explicit mathematical and numerical domains**
   - Keep the theoretical parameter domain, floating-point representability domain,
     and differentiability domain distinct. A likelihood jet may reject a request
     outside the relevant domain; it may not silently narrow the statistical model.
   - Variance floors, parameter clipping, hidden persistence margins, and density
     padding are forbidden unless they are part of the declared model itself.
     Prefer stable algebra or a generic optimizer rejection over modified math.
   - One model-owned domain contract supplies constraints, coordinate maps,
     initialization, feasibility, KKT assessment, and result reporting.
   - Every validated floating-point support bound has one named model-owned
     definition used by cache preparation, coordinate maps, bounds, and validation.

7. **Planned memory and dispatch**
   - Runtime iterations perform no heap allocation. Dynamically sized scratch uses
     checked caller-owned workspaces; size them with checked arithmetic and reject
     invalid orders or short arenas before entering the recursion.
   - Compile-time fixed scratch may use bounded stack storage only when the binary
     audit records its frame and enforces a documented per-kernel budget. Larger or
     data-dependent scratch belongs in a caller-owned arena.
   - A pure `F` request streams the minimum state. Add derivative rings for `G` and
     `H` only as needed, and materialize sample-length paths only for an explicit
     evidence request.
   - Do not branch on fixed model orders inside the observation loop. Inspect fixed
     specialization assembly to verify that runtime order checks and ring indexing
     disappear where expected. Audit binary text size and stack frames as well as
     timing.

8. **Completion evidence**
   - Validate local density derivatives, recursive state derivatives, the complete
     theta jet, and every shipped transformed jet against independent AD oracles.
   - Verify that every sparse request equals the corresponding projection of a full
     `F|G|H` request, and that runtime and fixed paths agree at shared points.
   - Encode exact distribution reductions as direct cross-density invariants. For
     example, Hansen Skew-t at zero skew must equal the variance-standardized
     Student-t value and all derivatives with respect to common parameters.
   - Test arbitrary orders beyond the precompiled set, zero orders where the model
     declares them valid, constants, conditioning boundaries, theoretical-domain
     boundaries, and request-specific failures.
   - Match the established production likelihood at identical points and benchmark
     objective requests separately from complete solves. Python-boundary savings do
     not excuse a slower objective kernel.
   - Until all clauses pass, label the surface **prototype/incomplete** with its
     blocking clauses. Do not mark it complete in `MODELS.md`, a manifest, a README,
     or a promotion ledger.

9. **Replacement likelihood optimality**
   - Mathematical jet completion and optimizer replacement readiness are separate
     claims. Exact derivatives do not prove that a solver reaches an equally good
     estimate.
   - On deterministic representative samples, fit the established and replacement
     paths from matched starts and evaluate both estimates with both evaluators.
     Record all four values. Differences in additive constants or scaling may be
     normalized explicitly; an unexplained ranking reversal or material loss blocks
     replacement promotion.
   - Require external feasibility and KKT credibility at the returned theta point.
     A historical solver success code is evidence, not a substitute for this gate.
   - When no established implementation exists, compare the native optimizer with
     a trusted independent optimizer on the identical jet and parameterization.
   - Use a tiered cadence: pointwise and tiny seeded tests after kernel changes; a
     small deterministic cross-fit matrix after optimizer, map, or domain changes;
     and full empirical panels only at promotion, default-policy, or release
     milestones. Do not rerun a full panel for ordinary line-level edits.

Additional optimizer-interface rules:

- Validate immutable observations and conditioning seeds once when a native solve
  begins. Each jet request still validates its parameters and requested outputs;
  standalone public jet calls validate all inputs.
- Checked standalone entries and prevalidated optimizer entries wrap the same
  mathematical kernel. Do not encode validation state as a template parameter when
  that would instantiate duplicate recursions.
- Coordinate maps expose value-only and value-plus-Jacobian intentions. Line-search
  `F` trials must not construct a dense transform Jacobian.
- Constraint evaluators should return finite violated constraints for infeasible
  optimizer trials whenever the mathematics permits it. A line-search domain
  failure is handled by a generic finite rejection policy in the optimizer; never
  alter the likelihood or insert a hidden model restriction to make the trial fit.
- Assess transformed solutions in the original parameter domain. Reconstruct active
  theta-constraint and bound multipliers from the returned point, theta gradient,
  and constraint Jacobian; do not substitute zero multipliers or reuse multipliers
  from an unrelated coordinate-space problem.
- Keep independent theta and coordinate solvers independently callable. A policy
  that conditionally polishes one with the other is a third, explicitly named path
  whose result records every formulation it evaluated and which one it selected.

**Build flags** (`setup.py`):
```python
# Legacy C extension only
extra-compile-args = [
    "-O3",              # Maximum optimization
    "-ffast-math",      # Fast floating point
    "-march=native",    # CPU-specific optimization
    "-ffp-contract=fast",
    "-funroll-loops"
]
```

Compile the native C++ optimizer extension with C++20 and IEEE-safe
floating-point semantics. Do not add `-ffast-math`: the optimizer uses NaN
bound sentinels and non-finite checks as part of its control contract. Portable
wheels also omit `-march=native`.

#### Shared Math Functions

Generic math functions go in `scivol/_csrc/math_and_helpers.h`:

```c
// Already provided:
VLK_FORCE_INLINE double digamma_approx(double x);   // ψ(x)
VLK_FORCE_INLINE double trigamma_approx(double x);  // ψ'(x)
VLK_FORCE_INLINE double lgamma_approx(double x);    // log Γ(x)

// Constants:
#define LOG_2PI   1.8378770664093453
#define H_FLOOR   1e-12   // Minimum variance floor
#define NU_MIN    2.001   // Minimum degrees of freedom
#define LAM_MAX   0.999   // Maximum skewness magnitude
```

These are legacy kernel constants, not a license to define new model domains.
New jets and optimizers must use the model-owned domain contract below. A floor
inside a recursion is valid only when it is part of the declared numerical model;
it must never be inserted merely to make an optimizer converge.

#### 🔴 C Performance Rule: Precompute Constants

**Never compute distribution constants inside the observation loop.**

```c
// ❌ WRONG: lgamma computed 6000+ times
for (size_t t = 1; t < n; t++) {
    double cnst = lgamma_approx(0.5*(nu+1)) - lgamma_approx(0.5*nu);  // Wasteful!
    sum_nll += -cnst + 0.5*log(h) + ...;
}

// ✅ CORRECT: Compute once before loop
double cnst = lgamma_approx(0.5*(nu+1)) - lgamma_approx(0.5*nu) - 0.5*log(nu*M_PI);
for (size_t t = 1; t < n; t++) {
    sum_nll += studentt_nll_var(e, h, nu);  // Only h-varying part
}
return sum_nll - n_eff * cnst;  // Additive jet total
```

This applies to Student-t (`lgamma` terms), Skew-t (`a`, `b` constants), GH
(`Bessel` terms), and every future density. If an optimizer needs the sample mean,
its typed scaling adapter applies `1 / n_eff`; the mathematical jet does not.

---

## 3. Development Standards

### Initialization Conventions (ARMA-GARCH)

**Standard initialization for time series models:**

```python
e_0 = 0.0           # Initial residual (conditioned on)
h_0 = mean(y²)      # Initial variance (or mean(eps²) for GARCH-only)
LL starts at t=1    # First obs with proper y_{t-1} available
n_eff = n - 1       # Effective sample size for scaling
```

**Why this matters:**
- At t=0, we don't have y_{-1}, so we can't compute a proper AR term
- Setting e_0=0 and starting LL at t=1 avoids biasing φ, θ estimates
- Consistent h_0 across models enables apples-to-apples LL comparisons

**In C code:**
```c
resid[0] = 0.0;      // e_0 = 0 (conditioning)
sigma2[0] = h0;      // h_0 passed from Python
for (size_t t = 1; t < n; t++) { ... }  // Start at t=1
return sum_nll / (double)(n - 1);       // Scale by n_eff
```

### Analytical Gradients via Sensitivity Recursions

For ARMA-GARCH, compute ∂e_t/∂θ and ∂h_t/∂θ recursively:

```
∂e_t/∂c     = -1 - θ·∂e_{t-1}/∂c
∂e_t/∂φ     = -y_{t-1} - θ·∂e_{t-1}/∂φ
∂e_t/∂θ_ma  = -e_{t-1} - θ·∂e_{t-1}/∂θ_ma

∂h_t/∂ω     = 1 + α·∂(e²)_{t-1}/∂ω + β·∂h_{t-1}/∂ω
∂h_t/∂α     = e²_{t-1} + α·∂(e²)_{t-1}/∂α + β·∂h_{t-1}/∂α
∂h_t/∂β     = h_{t-1} + α·∂(e²)_{t-1}/∂β + β·∂h_{t-1}/∂β

where ∂(e²)/∂θ = 2·e·∂e/∂θ
```

Per-observation gradient contribution:
```
∂ℓ_t/∂θ = (e_t/h_t)·∂e_t/∂θ + 0.5·(1/h_t - e²_t/h²_t)·∂h_t/∂θ
```

For Hessians, also track ∂²e_t/∂θ∂θ' and ∂²h_t/∂θ∂θ'.

### Derivative Validation (Required)

**All analytical derivatives must be validated against an independent AD oracle.**

Primary pattern:

```python
# 1. Compute analytical gradient/Hessian from the production kernel
grad_analytical = compute_gradient(params)
hess_analytical = compute_hessian(params)

# 2. Compute AD-oracle derivatives from an independent reference objective
_, grad_reference, hess_reference = ad_oracle_value_grad_hess(params, data)

# 3. Assert close (adjust tolerance as needed)
np.testing.assert_allclose(grad_analytical, grad_reference, rtol=1e-6, atol=1e-8)
np.testing.assert_allclose(hess_analytical, hess_reference, rtol=1e-5, atol=1e-6)
```

Preferred development oracles:
- Scalar recursive models (`GARCH`, `GJR-GARCH`): AD oracle from `scivol._devtools.ad_oracle`
- Matrix-heavy models (`DCC`): JAX-based AD reference objective
- Log-space / transformed objectives: validate the full optimized object `L(theta(z))`, not only the inner constrained objective

Finite differences are still acceptable as a weak smoke test or temporary fallback, but they are no longer the source of truth.

**Test both**:
- Correctness: Does it match the AD oracle?
- Boundary cases: Does it handle constraints and transforms properly?

### Model Mathematics and Optimizer Policy

The mathematical kernel defines the model. An optimizer may change coordinates,
scale an equivalent objective, choose an evaluation order, or change its search
strategy. It must not change the likelihood or silently narrow the model's
theoretical parameter domain to improve convergence.

For covariance-stationary GARCH, the default domain is:

```text
omega > 0
alpha_i >= 0
beta_j >= 0
sum(alpha) + sum(beta) < 1
```

Apply the same principle to every model family using its correct positivity,
stationarity, invertibility, and density restrictions. Stronger restrictions are
explicit user choices, not hidden defaults.

When stationarity depends on density or shape parameters, the domain contract must
evaluate the exact distribution-dependent moment and its derivatives. A transformed
chart should parameterize persistence contributions such as `alpha_i * moment_i`,
not raw coefficients whose feasible range silently changes when the density changes.

- Keep theoretical restrictions in one model-owned parameter-domain contract.
  Theta constraints, transformed coordinates, initialization, feasibility checks,
  KKT checks, and result reporting must consume that contract.
- Derive unavoidable numerical interiors from floating-point representation and
  transform algebra. Do not use hand-tuned parameter floors or persistence margins
  as convergence aids. Report the effective numerical domain in `result.fit_info`.
- Prefer exact internal data scaling with exact parameter, likelihood, derivative,
  and forecast back-transforms when parameter magnitudes depend on measurement
  units. Scaling must not change the fitted statistical model. Benchmark transformed
  coordinates after scaling because a nonlinear chart can change the optimizer path
  even when the theta-space objective is equivalent.
- Clip or reject every trial against hard parameter bounds before evaluating a jet.
  In particular, the optimizer must never ask a likelihood to accept `omega <= 0`
  or a Student-t density to accept `nu <= 2`. Do not pad the public domain to cover
  for out-of-domain solver requests.
- Diagnostics may warn about boundary estimates, near integration, weak
  identification, or unreliable long-run moments. They must not repair parameters
  or alter the objective.
- `result.success` requires finite requested quantities, theoretical-domain
  feasibility, solver termination, and an original-parameter scaled KKT check.
  Keep the raw solver status separately for diagnosis.
- Report the KKT residual and threshold separately. Unless the API exposes a
  dedicated threshold, use
  `max(sqrt(solver_accuracy), sqrt(machine_epsilon))`; this ties first-order
  credibility to the requested solve accuracy without a model-specific magic
  tolerance.

Every native jet and optimizer path must pass three promotion gates:

1. **Mathematical parity**: value, gradient, and Hessian match an independent AD
   oracle and any established production kernel.
2. **Domain and credibility parity**: all public solvers target the same declared
   theoretical domain and pass matched-point feasibility and KKT checks.
3. **Performance superiority**: sparse objective requests are at least as fast as
   the production path they replace, and matched-domain solve benchmarks improve
   convergence or throughput without weakening the first two gates.

Complete-fit promotion also requires a matched solution-quality comparison. Rescore
all candidates with one common theta-space likelihood and KKT assessor, compare
objective values only among credible fits, and reject a new default that is worse
than production beyond the requested numerical tolerance. Test matched starts to
separate optimizer quality from basin coverage. Alternative starts, coordinate
charts, and exact positive objective scalings are optimizer policies; keep their
budgets and effective path visible. A high-reliability portfolio may be slower than
the default, but it must select only from credible candidates and must not replace
independent public fitting paths silently.

### Independent Public Solution Paths

When scivol exposes multiple **public** fitting paths for the same statistical model
(for example constrained `theta` mode vs unconstrained `z` mode, or alternative
solvers that users may compare on unstable problems), those paths must remain
**genuinely independent** in production behavior.

This is a library philosophy rule, not just a testing preference:

- Users may rely on multiple paths as a credibility check when estimation is
  numerically unstable.
- Therefore, parity between public paths is evidence only if each path actually
  optimized its own stated formulation.
- Do **not** force agreement by routing distinct public modes through the same
  hidden optimizer/objective and then reporting the result as if it came from
  each separate path.
- Do **not** fit multiple public modes internally, pick the best outcome, and
  present that as the result of each mode. That destroys the user's ability to
  cross-check instability.
- Shared infrastructure is still encouraged where it preserves semantics:
  identical data preparation, transform algebra, matched-point diagnostics,
  chain-rule checks, reusable buffers, and localdev/debug tooling are all fine.
- If a stabilization idea is desirable across modes (for example multistart),
  implement it **symmetrically inside each mode's own formulation**, or expose it
  explicitly as a separate API feature. Do not sneak it in by collapsing modes.
- When investigating discrepancies, prefer invariant checks at matched points:
  same constrained `theta`, same mapped starts, same value/gradient/Hessian after
  transformation. Do not replace that investigation with black-box output forcing.
- Keep kernel-debugging and tracing machinery internal-only unless a user-facing
  debugging API is explicitly designed and documented.

### Model Reference (Keep Evergreen)

`scivol/_devtools/surface_manifest.py` is the machine-readable source of truth for
the public tensor. Generated blocks in **`MODELS.md`** present that contract to
humans, and `tests/test_surface_manifest_contract.py` prevents the two from drifting.

Before the first native jet is promoted, extend the manifest with machine-checked
jet-completion fields for general-order support, fixed-order generation, objective
convention, sparse requests, transformed derivatives, domain semantics, and evidence
level. CI must reject a `complete` native-jet status when any required field is
missing or weaker than the completion contract above. Do not maintain a second
handwritten jet-coverage table in `MODELS.md`.

**Keep the contract current** when adding or modifying models:
- Add or change surfaces in the manifest first
- Update implementation and validation evidence at the same time
- Refresh the generated `MODELS.md` blocks in the same change
- Keep explicit verification exceptions until their machine-checked gaps close

### Master DGP Test File (Keep Evergreen)

**`tests/test_dgp_estimation.py`** is the canonical test file for scivol estimation accuracy.

**Keep this file updated** when adding new models or distributions:
1. Add a `simulate_*` DGP function for the new model
2. Add true parameters to `TRUE_PARAMS` dict
3. Add a test class with `test_convergence` and `test_parameter_recovery`

**Test approach:**
- Generate 5000 observations from known true parameters
- Estimate the model using scivol
- Verify optimization converges
- Check parameter recovery within tolerance

**Run with:** `pytest tests/test_dgp_estimation.py -v`

### Testing Patterns

**Use pytest with fixtures**:

```python
import pytest
from scivol import GARCH, Normal

@pytest.fixture
def sample_data():
    np.random.seed(42)
    return np.random.randn(1000) * 0.01

def test_garch_11_estimation(sample_data):
    spec = GARCH(1, 1) + Normal()
    result = spec.fit(sample_data)
    assert result.success
    assert 0 < result.params[0] < 1  # omega
```

**Parametrized tests** for validation:

```python
@pytest.mark.parametrize("bad_data", [
    np.empty((2, 4)),           # 2-D array
    np.array([np.nan, 0.0]),    # Contains NaN
    np.array([np.inf, 1.0]),    # Contains Inf
])
def test_validation_raises(bad_data):
    with pytest.raises(ValueError):
        spec.fit(bad_data)
```

### Slow Tests Warning

**The following test files are slow (4+ minutes) because they fit many models.**
They are marked with `@pytest.mark.slow` and skipped by default.

| Test File | Reason | When to Run |
|-----------|--------|-------------|
| `tests/test_parallel_auto.py` | Fits 4-8 models per test with auto-selection | Only when modifying `_parallel.py`, `_autoselect.py`, or `_mixins.py` |
| `tests/test_multi_series.py` | Fits multiple series in parallel | Only when modifying multi-series or parallel fitting logic |

**Fast tests** (run always): `test_pandas_integration.py`, `test_spec_core.py`, `test_result_core.py`, etc.

Run tests:
```bash
# Run fast tests only (default - skips @pytest.mark.slow)
pytest tests/

# Run ALL tests including slow ones
pytest tests/ --run-slow

# Run only slow tests
pytest tests/ --run-slow -m slow
```

---

## 4. Internal Implementation

### Current State (Under Evolution)

The internal kernel/routine organization is being refined. Current architecture uses a registry-based dispatch system, but simpler alternatives are being explored.

#### Current: Registry-Based Kernels

**Location**: `scivol/_kernels/`

**Pattern**:
1. UID format: `"Component1(p,q)+Component2(r,s)+Density"`
2. Registry maps UID → kernel module
3. Each module provides `get_routine(uid) -> Routine`
4. `Routine` object encapsulates fit logic

**Files involved**:
- `scivol/_kernels/__init__.py` - Registry and dispatcher
- `scivol/_kernels/routine.py` - `Routine` dataclass
- `scivol/_kernels/garch_normal.py` - GARCH+Normal implementations
- `scivol/_kernels/garch_studentt.py` - GARCH+StudentT implementations

### What's Stable

**C extensions are the single source of truth for computations.**

Never duplicate computational logic in Python. If you need a new calculation:
1. Implement in C (`scivol/_csrc/`)
2. Add declaration to `scivol/_csrc/scivol_core.h`
3. Wrap in `scivol/_core.c` with METH_FASTCALL
4. Add type stub to `scivol/_core.pyi`
5. Call from Python via pointer passing

---

## 5. Type Safety Requirements

### Mandatory Patterns

**Import pattern**:

```python
from __future__ import annotations  # Allow forward references
from typing import TYPE_CHECKING, Optional, Union, Dict, List, Tuple
import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    # Import only for type checking (avoid circular imports)
    from .result import EstimationResult
```

**Array types**:

```python
# Always annotate NumPy arrays with dtype
resid: NDArray[np.float64] = np.array([...], dtype=np.float64)
sigma2: NDArray[np.float64] = np.empty(n, dtype=np.float64)
```

**Optional types**:

```python
# Use Optional for nullable returns
def get_component(self, role: Role) -> Optional[Component]:
    return self._component_map.get(role)
```

**Protocol classes** for duck typing:

```python
from typing import Protocol, runtime_checkable

@runtime_checkable
class OptimizeResultLike(Protocol):
    x: np.ndarray
    fun: float
    success: bool
    nit: int
    message: str
```

### Naming Conventions

**Variables**:
- Greek letters as words: `nu` (ν), `lam` (λ), `omega` (ω), `alpha` (α), `beta` (β)
- Short names in hot paths: `p`, `q`, `n`, `eps`, `resid2`, `sigma2`
- Descriptive elsewhere: `optimization_result`, `cov_matrix`, `std_errors`

**Classes**:
- PascalCase: `Component`, `CompositeSpec`, `EstimationResult`, `GARCHParams`
- Components match model names: `GARCH`, `ARMA`, `Normal`, `StudentT`

**Functions**:
- snake_case: `fit_garch`, `compute_hessian`, `default_start`
- C functions: `garch_variance_11`, `garch_ll_pq_normal`

**Files/Modules**:
- lowercase with underscores: `benchmark_optimizers.py`, `localdev/numerical_hessians.py`
- Private modules: underscore prefix: `_mixins.py`, `_kernels/`, `_csrc/`
- Development files: place in `localdev/` folder (git-ignored)

---

## 6. Build System

### Development Workflow

**Makefile targets**:

```bash
make dev    # Editable install with in-place C compilation
make b      # Build wheel and force-reinstall  
make t      # Run test suite
make f      # Full build without installation
```

**Manual build**:

```bash
# Development mode (recommended)
pip install -e . --no-build-isolation

# Rebuild after C changes
rm -rf build/ && make dev
```

### Package Structure

**Build configuration** (`pyproject.toml`):

```toml
[build-system]
requires = ["setuptools>=69", "wheel", "numpy>=1.22"]
build-backend = "setuptools.build_meta"

[[tool.setuptools.ext-modules]]
name = "scivol._core"
sources = [
 "scivol/_core.c",
 "scivol/_csrc/variance_garch.c",
 "scivol/_csrc/variance_gjr_garch.c",
 "scivol/_csrc/likelihood_garch.c",
 "scivol/_csrc/likelihood_gjr_garch.c",
 "scivol/_csrc/likelihood_normal.c",
 "scivol/_csrc/likelihood_studentt.c",
 "scivol/_csrc/errors_garch.c",
 "scivol/_csrc/errors_gjr_garch.c",
]
include-dirs = ["scivol/_csrc"]
```

**Required files**:
- `MANIFEST.in`: Includes C headers in source distribution
- `pyproject.toml`: Build configuration
- `scivol/__init__.py`: Package entry point
- `scivol/_core.pyi`: Type stubs for C extension

### Verify Build

```python
# Should load without errors
import scivol._core
print(scivol._core.__file__)  # Shows .so location

# Test a C function
import numpy as np
theta = np.array([1e-6, 0.05, 0.9], dtype=np.float64)
resid2 = np.random.randn(100)**2
sigma2 = np.empty(100, dtype=np.float64)

from scivol import _core
_core._garch_variance_11(
    theta.ctypes.data,
    resid2.ctypes.data, 
    sigma2.ctypes.data,
    100
)
print(sigma2[:5])  # Should contain positive variances
```

---

## 7. Quick Reference

### Common Patterns

**Create and fit a model**:

```python
from scivol import GARCH, GJRGARCH, StudentT
import numpy as np

# Generate sample data
returns = np.random.randn(1000) * 0.01

# Symmetric GARCH
spec = GARCH(1, 1) + StudentT()
result = spec.fit(returns)

# Asymmetric GJR-GARCH (leverage effect)
spec = GJRGARCH(1, 1) + StudentT()
result = spec.fit(returns)

# Access results
print(f"Log-likelihood: {result.log_likelihood}")
print(f"Parameters: {result.params}")
print(f"AIC: {result.aic}, BIC: {result.bic}")
```

**Add a new component**:

1. Create `scivol/components/my_component.py`
2. Inherit from `Component`, set `role`
3. Implement all abstract methods
4. Add to `scivol/components/__init__.py`
5. Add C implementations if needed
6. Write tests in `tests/`

**Add a new C function**:

1. Implement in `scivol/_csrc/my_function.c`
2. Declare in `scivol/_csrc/scivol_core.h`
3. Wrap in `scivol/_core.c` with proper pointer handling
4. Add type stub to `scivol/_core.pyi`
5. Validate with an AD oracle

### Directory Structure

```
scivol_cursor/                   # Repository root
├── scivol/                      # Core package (ships)
│   ├── _core.c                  # C extension wrapper (Python-C interface)
│   ├── _core.pyi                # Type stubs for C functions
│   ├── _csrc/                   # C implementation
│   │   ├── scivol_core.h        # Public C API declarations
│   │   ├── math_and_helpers.h   # Shared math (lgamma, digamma, constants)
│   │   ├── variance_garch.c     # GARCH variance recursion
│   │   ├── variance_gjr_garch.c # GJR-GARCH variance recursion (raw residuals)
│   │   ├── likelihood_*.c       # Distribution log-likelihoods
│   │   ├── likelihood_gjr_garch.c # GJR-GARCH NLL + grad + Hessian
│   │   ├── arma_garch.c         # ARMA-GARCH NLL + derivatives (Normal/t/Skew-t)
│   │   ├── errors_garch.c       # OPG and Hessian computation
│   │   ├── errors_gjr_garch.c   # GJR-GARCH OPG and Hessian
│   │   └── log_wrappers.c       # Fused log-space NLL+gradient wrappers
│   ├── _kernels/                # Optimization kernels (internal)
│   ├── components/              # User-facing components
│   ├── spec/                    # Composition logic
│   ├── _qmle.py                 # QMLE fitting with robust SEs
│   ├── _validation.py           # Shared validation helpers
│   ├── estimators/              # Estimation methods (reserved)
│   ├── result.py                # EstimationResult class
│   ├── roles.py                 # Role enum (MEAN, VOLATILITY, DENSITY)
│   └── _mixins.py               # FitsMixin helper
├── tests/                       # Test suite (ships)
├── benchmark_optimizers.py      # Optimizer benchmarking (ships, keep evergreen)
├── AGENTS.md                    # Developer guide (ships)
├── README.md                    # User documentation (ships)
├── pyproject.toml               # Build configuration (ships)
├── Makefile                     # Build automation (ships)
├── MANIFEST.in                  # Build manifest (ships)
│
└── localdev/                    # Development scripts (git-ignored)
    ├── arma_garch_estimator.py  # Reference: ARMA-GARCH with analytical grad/Hess
    ├── likelihoods.py           # Reference: Log-likelihood functions
    ├── numerical_hessians.py    # Legacy FD tooling (backup only)
    ├── utilities.py             # Reference: Statistical tests
    ├── analysis.py              # Analysis scripts
    └── ...                      # Other experiments and dev tools

# Also git-ignored: localdev_data/, localdev_results/, localdev_backup/, etc.
```

### Import Patterns

```python
# Public API
from scivol import GARCH, GJRGARCH, ARMA, Normal, StudentT, SkewT, CompositeSpec, Role

# Internal (for development)
from scivol._core import _garch_variance_11, _gjr_garch_variance_11, _garch_ll_pq_normal
from scivol._kernels import get_routine
from scivol.components.base import Component
from scivol.result import EstimationResult
```

---

## Next Steps for Development

### Completed

- ✅ ARMA-GARCH Python reference with analytical gradients/Hessians
- ✅ Normal, Student-t, Skew-t distributions
- ✅ C implementations for ARMA(1,1)-GARCH(1,1) NLL + gradient (Normal)
- ✅ C implementations for ARMA(p,q)-GARCH(P,Q) NLL (all distributions)
- ✅ Sensitivity recursion framework for derivatives
- ✅ Shared math functions in `math_and_helpers.h`
- ✅ GJR-GARCH Python prototype with Numba (`localdev/gjr_garch_estimator.py`)
- ✅ GJR-GARCH C implementation (variance, NLL, gradient, Hessian for Normal and Student-t)
- ✅ GJR-GARCH log-space transforms in C (`transforms_logspace.c`)
- ✅ GJRGARCH component class with distribution-dependent stationarity
- ✅ GJR-GARCH kernel modules (Normal, Student-t, Skew-t)
- ✅ GJR-GARCH DGP estimation tests and gradient validation tests
- ✅ GJR-GARCH benchmark configuration
- ✅ GJR-GARCH(p,q) C gradient and Hessian (Normal and Student-t)
- ✅ Fused log-space C wrappers (`log_wrappers.c`) for GARCH/GJR-GARCH + Normal/Student-t
- ✅ ARMA-GARCH C pack/jacobian in `transforms_logspace.c` (6 pack + 6 jacobian = 12 functions)
- ✅ Fused log-space C wrappers for ARMA-GARCH (3 NLL + 3 gradient functions)
- ✅ DCC(p,q) Student-t copula C likelihood, gradient, Hessian, and scores, with dedicated DCC(1,1) entry points
- ✅ Exact observed C Hessians for ARMA-GARCH + Normal/Student-t with promoted analytical log-Hessians
- ✅ Exact observed C gradients/Hessians for ARMA-GARCH + Skew-t with promoted analytical log-Hessians
- ✅ Exact observed C gradients/Hessians for GARCH + Skew-t with promoted analytical log-Hessians
- ✅ Exact observed C gradients/Hessians for GJR-GARCH + Skew-t with promoted analytical log-Hessians
- ✅ Fused log-space C NLL/gradient wrappers for GARCH/GJR-GARCH + Skew-t
- ✅ Fused log-space C NLL/gradient wrappers for ARMA-GARCH + Skew-t
- ✅ ARMA-GARCH kernel files updated to use fused functions
- ✅ APARCH(p,q) with exact C gradients/Hessians and all shipped densities
- ✅ Warm-started automatic selection, rolling OOS evaluation, and model confidence sets
- ✅ AutoMean screening and three-axis mean/volatility/density search
- ✅ Native C++ function-jet optimizer foundation with caller-owned workspaces,
  typed objectives, strict KKT reporting, and generic credibility refinement
- ✅ Native GARCH(p,q) and genuine joint ARMA(p,q)-GARCH(P,Q) matrices for all
  eight shipped univariate densities

### Immediate Goals

1. Extend the native tensor one complete volatility family at a time
2. Add native QMLE and auto-selection only after their inference and failure
   contracts are explicit
3. Use a bounded representative panel for each default-family promotion; reserve
   the full CRSP panel for whole-engine, tensor-scale, and release gates

### Native Engine Promotion Status

The native C++ function-jet engine is the default MLE path for fixed standalone
GARCH with Normal, Student-t, Skew-t, or GED density. It is available explicitly
through `solver="native"` for every shipped univariate density and genuine joint
ARMA-GARCH specifications at arbitrary declared orders. `solver="slsqp"` and
`solver="legacy"` retain the independent established path, while
`log_mode=False` selects its theta-space formulation when no solver is supplied.

The native path stays quiet during fitting. It records historical Kraft mode,
strict KKT residual and tolerance, maximum violation, refinement restarts, request
plan, and matched production-objective error in `result.fit_info`. A failed strict
check returns a normal result with a persistent warning. Joint ARMA-GARCH,
GH-family GARCH, QMLE, and auto-selection remain on the established default
engine until their native performance and inference contracts are promoted.

Files under `scivol/_native/` marked `SCIVOL-NATIVE: CORE` are production sources.
The SciPy SLSQP translation under `third_party/` is pinned upstream code.
`localdev/native_optimizer_experiment/` retains only the unpromoted
ARMA-APARCH-GH stress model and its minimum C++ test dependencies. Do not copy
that directory wholesale during promotion.

### Fused Log-Space Wrappers

**Location**: `scivol/_csrc/log_wrappers.c`

Fused C functions that perform the full unconstrained optimization pipeline in a single C call:
1. `pack(z -> theta)` - unconstrained to constrained parameters
2. Compute NLL or gradient in theta-space
3. `jacobian(theta -> J)` - Jacobian of the transform
4. `transform(J^T @ grad)` - chain rule to z-space gradient

**Naming**: `_log_{model}_ll[_grad]_pq_{distribution}`

**Available functions**:
- `_log_garch_ll_pq_normal` / `_log_garch_ll_grad_pq_normal` (takes resid2)
- `_log_garch_ll_pq_studentt` / `_log_garch_ll_grad_pq_studentt` (takes resid2)
- `_log_gjr_garch_ll_pq_normal` / `_log_gjr_garch_ll_grad_pq_normal` (takes raw resid)
- `_log_gjr_garch_ll_pq_studentt` / `_log_gjr_garch_ll_grad_pq_studentt` (takes raw resid)
- `_log_arma_nll_pq_normal` / `_log_arma_nll_grad_pq_normal` (takes y, resid, e0)
- `_log_arma_garch_nll_pq_normal` / `_log_arma_garch_nll_grad_pq_normal` (takes y, resid, sigma2, e0, h0)
- `_log_arma_garch_nll_pq_studentt` / `_log_arma_garch_nll_grad_pq_studentt` (takes y, resid, sigma2, e0, h0)
- `_log_arma_garch_nll_pq_skewt` / `_log_arma_garch_nll_grad_pq_skewt` (takes y, resid, sigma2, e0, h0)

Each dispatches to specialized `_11` kernels when applicable (all orders == 1).
ARMA-GARCH Normal, Student-t, and Skew-t theta-space derivative functions now work for all supported orders; the fused wrappers dispatch to the specialized `_11` kernels when all orders equal 1.

**Python kernel pattern** (after fused wrappers):
```python
def obj_log(z):
    return _core._log_garch_ll_pq_normal(_as_cptr(z), resid2_c, sigma2_c, n, p, q) * p_scaler

def jac_log(z):
    _core._log_garch_ll_grad_pq_normal(_as_cptr(z), resid2_c, sigma2_c, _grad_z_c, n, p, q)
    return _grad_z_buf.copy() * p_scaler
```

### GJR-GARCH Stationarity Note

The stationarity constraint for GJR-GARCH is distribution-dependent:
- **Symmetric distributions** (Normal, Student-t): `α + 0.5·γ + β < 1`
- **Asymmetric distributions** (Skew-t): `α + γ·P(z_t < 0) + β < 1` where `P(z_t < 0)` depends on the skewness parameter

The `GJRGARCH` component's `persistence()` method accepts a `p_neg` parameter (default 0.5) for this.

### Future Models

New model families will follow the component pattern:

```python
# Example: a future long-memory volatility component
from scivol import FIGARCH, StudentT

spec = FIGARCH(1, 1) + StudentT()
result = spec.fit(returns)
```

Each model family will need:
- Component class (e.g., `FIGARCH(Component)`)
- C implementations for performance-critical paths
- Kernel routines for optimization
- Comprehensive tests with derivative validation

### New Model Checklist

Before merging a new shipped model family or distribution, make sure all of the
following are true:

1. **Scope is explicit**
   - Supported orders are defined
   - Supported densities are defined
   - Constrained/log-mode support is defined
   - The theoretical parameter domain is written down before coding
   - One domain contract supplies theta constraints, transforms, feasibility, KKT checks, and reporting

2. **User-facing API fits the composition pattern**
   - The model is represented as a component with the correct `Role`
   - `.fit()` works through the standard composition pipeline
   - The public API does not introduce one-off fitting interfaces unless truly necessary

3. **Prototype exists in `localdev/` first**
   - Start with a Python/Numba reference implementation
   - Use it to debug recursion, transforms, and parameterization
   - Keep exploratory scripts out of shipped package code

4. **Derivatives are validated independently**
   - Value, gradient, and Hessian are checked against an AD oracle
   - Finite differences may be used only as a coarse backup smoke test
   - Do not ship a derivative path that has not been independently validated

5. **A native jet passes the completion contract**
   - A runtime general-order path exists for every declared order axis
   - The same policies can generate compile-time fixed-order specializations
   - The jet returns the exact additive likelihood, including all constants
   - All sparse `F`/`G`/`H` requests and every shipped coordinate path are validated
   - Any missing clause keeps the surface labeled prototype/incomplete

6. **Native code remains the shipped source of truth**
   - Performance-critical computations are implemented in the production C/C++ core
   - Python pre-allocates buffers and passes contiguous `float64` pointers
   - Legacy C symbols are declared in `scivol/_csrc/scivol_core.h`, wrapped in `scivol/_core.c`, and stubbed in `scivol/_core.pyi`
   - Function-jet and optimizer entries use the single pointer-level `scivol._native_core` binding and are stubbed in `scivol/_native_core.pyi`

7. **Both optimization formulations are treated deliberately**
   - Theta-space and log-space behavior are both considered
   - Public solution paths remain genuinely independent where they are meant to cross-check stability
   - If one path is approximate or unsupported, document that explicitly

8. **Coverage files are updated immediately**
   - Update `MODELS.md`
   - Update `tests/test_dgp_estimation.py`
   - Update `benchmark_optimizers.py`
   - Update `README.md` if user-facing support changed

9. **Defaults are earned, not assumed**
   - Benchmark solver/log-mode choices on the same parameter domain and real data
   - Run DGP recovery tests
   - Require mathematical parity, domain/KKT credibility, and objective-level speed

10. **Add one model family at a time**
   - Keep commits and validation isolated by model family/distribution
   - Preserve a clear known-good baseline before starting the next model
   - Do not batch unrelated model additions into one verification step

---

## Notes for AI Agents

### When in Doubt

1. **User API**: Keep component composition + `.fit()` pattern
2. **C functions**: Always single source of truth for computation
3. **Memory**: Pre-allocate in Python, pass pointers to C
4. **Arrays**: Always `np.ascontiguousarray(arr, dtype=np.float64)`
5. **Derivatives**: Validate with an independent AD oracle
6. **Types**: Annotate everything with proper type hints

### Common Pitfalls

❌ **Returning arrays from C** - C modifies in-place, doesn't return  
❌ **Forgetting contiguous** - Non-contiguous arrays cause corruption  
❌ **Wrong dtype** - Must be `float64`, not `float32` or Python float  
❌ **No pre-allocation** - Output arrays must exist before C call  
❌ **Duplicating C logic in Python** - Always call C for computation  
❌ **Constants in loops** - Precompute lgamma, distribution constants outside loops

### Development Pattern: Numba First, Then C

For new models:
1. **Implement in Python with `@numba.njit`** - Fast iteration, easy debugging
2. **Validate derivatives** against an independent AD oracle
3. **Port to C** using the Numba code as reference
4. **Verify C matches Numba** to <1e-12 precision

This pattern was used for `localdev/arma_garch_estimator.py` → `arma_garch.c`.

### Benchmark Testing (Keep Evergreen)

**IMPORTANT: When adding a new model/routine, you MUST add it to:**
1. **`benchmark_optimizers.py`** - Tests optimizer configurations with real data
2. **`tests/test_dgp_estimation.py`** - Tests parameter recovery with DGP data
3. **`MODELS.md`** - Documents analytical/numerical kernel coverage for every model

`benchmark_optimizers.py` tests all optimizer configurations:
- Model types: GARCH(1,1), ARMA(1,1)-GARCH(1,1)
- All distributions (Normal, Student-t, Skew-t)
- All solvers (nelder-mead, slsqp, trust)
- Log-mode vs constrained mode
- Multiple real asset classes (stock, bond, commodity, etc.)
- **Gradient verification**: Analytical vs numerical gradients for all C implementations

Run after any changes to optimization code to validate defaults:
```bash
python benchmark_optimizers.py
```

Results saved to `localdev_benchmark_results/` with recommended defaults.

`tests/test_dgp_estimation.py` tests parameter recovery:
- Generates synthetic data from known DGPs (5000 observations)
- Verifies estimation recovers true parameters (within tolerance)
- All model/distribution combinations must have tests

**Current Recommended Defaults** (as of 2026-05-08):

**GARCH(1,1) Models:**

| Distribution | Solver | Log Mode | Avg Time | Conv Rate |
|--------------|--------|----------|----------|-----------|
| Normal       | slsqp  | True     | 0.002s   | 100%      |
| Student-t    | slsqp  | True     | 0.010s   | 100%      |
| Skew-t       | slsqp  | True     | 0.016s   | 100%      |

**GJR-GARCH(1,1) Models:**

| Distribution | Solver | Log Mode | Avg Time | Conv Rate |
|--------------|--------|----------|----------|-----------|
| Normal       | slsqp  | True     | 0.008s   | 100%      |
| Student-t    | slsqp  | False    | 0.010s   | 100%      |
| Skew-t       | slsqp  | False    | 0.028s   | 100%      |

**ARMA(1,1)-GARCH(1,1) Models:**

| Distribution | Solver | Log Mode | Avg Time | Conv Rate |
|--------------|--------|----------|----------|-----------|
| Normal       | slsqp  | False    | 0.007s   | 100%      |
| Student-t    | slsqp  | False    | 0.012s   | 100%      |
| Skew-t       | slsqp  | False    | 0.029s   | 100%      |

**ARMA(1,1) Models:**

| Distribution | Solver | Log Mode | Avg Time | Conv Rate |
|--------------|--------|----------|----------|-----------|
| Normal       | slsqp  | False    | 0.001s   | 100%      |

Key findings:
- `slsqp` is fastest AND most reliable across all distributions
- `trust` solver has poor convergence (0-67%), avoid as default
- `nelder-mead` is reliable but 5-10x slower than `slsqp`
- Standalone `GARCH` currently benchmarks best in log-mode across Normal, Student-t, and Skew-t
- `GJR-GARCH` is density-dependent: Normal prefers log-mode, Student-t and Skew-t prefer theta-mode
- `ARMA` and `ARMA-GARCH` currently benchmark best in theta-mode
- Effective defaults should remain visible via `result.fit_info`, not silent

### Questions to Ask

Before implementing:
- Is there a C function for this computation? (Use it)
- Does this need a new component or can it reuse existing?
- How will this compose with other components?
- What are the parameter bounds and initial values?
- How will I validate derivatives?

---

**End of Guide**

For questions or clarifications about this guide, check:
- Implementation examples in `scivol/components/` and `scivol/_kernels/`
- Reference implementations in `localdev/arma_garch_estimator.py`
- Test patterns in `tests/`
- C interface in `scivol/_core.c` and `scivol/_core.pyi`

**New development work** (scripts, experiments, analysis) should go in `localdev/`.
