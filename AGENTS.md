# volkit Library Guide for AI Agents

**Last Updated:** 2026-01-30  
**Purpose:** Essential architectural rules, patterns, and constraints for developing the volkit time series volatility modeling library.

**Reference implementations:** `localdev/arma_garch_estimator.py` contains verified Python+Numba implementations with analytical gradients/Hessians for ARMA-GARCH models (Normal, Student-t, Skew-t). Use as ground truth when porting to C.

**Benchmarking:** `benchmark_optimizers.py` tests all optimizer configurations on real data. Run periodically to validate/update default settings.

**Local development:** All development scripts, experiments, and analysis code should go in `localdev/`. This folder is git-ignored. Directories prefixed with `localdev_*` are also ignored (e.g., `localdev_data/`, `localdev_results/`).

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
from volkit import GARCH, ARMA, Normal, StudentT

# Simple models
spec = GARCH(1, 1)                           # Auto-adds Normal() density
spec = GARCH(1, 1) + StudentT()              # Explicit density

# Composite models
spec = ARMA(1, 1) + GARCH(1, 1)              # Mean + Volatility
spec = ARMA(1, 1) + GARCH(1, 1) + StudentT() # Full specification

# Universal interface
result = spec.fit(data)

# QMLE with robust (sandwich) standard errors
from volkit.estimators import QMLE
result = QMLE().fit(spec, data)
print(result.std_errors_robust)
```

#### QMLE (Quasi-Maximum Likelihood Estimation)

QMLE provides robust standard errors that are valid even when the distributional assumption is wrong:

```python
from volkit import GARCH, Normal, StudentT, SkewT
from volkit.estimators import QMLE

# Normal: robust SEs for all GARCH parameters
spec = GARCH(1, 1) + Normal()
result = QMLE().fit(spec, data)
print(result.std_errors)        # MLE standard errors
print(result.std_errors_robust) # Sandwich (robust) standard errors

# Student-t: two-step procedure
# Step 1: Fit GARCH with Normal LL → robust SEs for GARCH params
# Step 2: Fix GARCH, fit nu → MLE SE for nu
spec = GARCH(1, 1) + StudentT()
result = QMLE().fit(spec, data)
# result.std_errors_robust[0:3] = robust SEs for [omega, alpha, beta]
# result.std_errors_robust[3]   = MLE SE for nu

# Skew-t: same two-step procedure
spec = GARCH(1, 1) + SkewT()
result = QMLE().fit(spec, data)
# result.std_errors_robust[0:3] = robust SEs for [omega, alpha, beta]
# result.std_errors_robust[3:5] = MLE SEs for [nu, lambda]
```

The sandwich covariance is: `V_robust = H^{-1} @ OPG @ H^{-1}` where H is the Hessian and OPG is the outer product of gradients.

#### Component System

**Roles** (`volkit/roles.py`):
- `MEAN`: Mean equation components (e.g., ARMA)
- `VOLATILITY`: Volatility components (e.g., GARCH, GJR-GARCH)
- `DENSITY`: Conditional distributions (e.g., Normal, StudentT, SkewT)
- `CORRELATION`: Multivariate correlation (future)

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

All C functions must have type stubs in `volkit/_core.pyi`:

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

**Specialized vs Generic**:
- Provide specialized `_11` functions for GARCH(1,1) (most common)
- Provide generic `_pq` functions for arbitrary orders
- Python code tries specialized first, falls back to generic

**Build flags** (`pyproject.toml`):
```python
extra-compile-args = [
    "-O3",              # Maximum optimization
    "-ffast-math",      # Fast floating point
    "-march=native",    # CPU-specific optimization
    "-ffp-contract=fast",
    "-funroll-loops"
]
```

#### Shared Math Functions

Generic math functions go in `volkit/_csrc/math_and_helpers.h`:

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
return (sum_nll - n_eff * cnst) / n_eff;
```

This applies to Student-t (`lgamma` terms) and Skew-t (`a`, `b` constants).

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

**All analytical derivatives must be validated against finite differences.**

Pattern from `localdev/numerical_hessians.py`:

```python
# 1. Compute analytical gradient/Hessian
grad_analytical = compute_gradient(params)
hess_analytical = compute_hessian(params)

# 2. Compute numerical approximation
grad_numerical = finite_difference_gradient(objective, params, eps=1e-5)
hess_numerical = finite_difference_hessian(objective, params, eps=1e-5)

# 3. Assert close (adjust tolerance as needed)
np.testing.assert_allclose(grad_analytical, grad_numerical, rtol=1e-5, atol=1e-8)
np.testing.assert_allclose(hess_analytical, hess_numerical, rtol=1e-4, atol=1e-6)
```

**Test both**:
- Correctness: Does it match finite differences?
- Boundary cases: Does it handle constraints properly?

### Master DGP Test File (Keep Evergreen)

**`tests/test_dgp_estimation.py`** is the canonical test file for volkit estimation accuracy.

**Keep this file updated** when adding new models or distributions:
1. Add a `simulate_*` DGP function for the new model
2. Add true parameters to `TRUE_PARAMS` dict
3. Add a test class with `test_convergence` and `test_parameter_recovery`

**Test approach:**
- Generate 5000 observations from known true parameters
- Estimate the model using volkit
- Verify optimization converges
- Check parameter recovery within tolerance

**Run with:** `pytest tests/test_dgp_estimation.py -v`

### Testing Patterns

**Use pytest with fixtures**:

```python
import pytest
from volkit import GARCH, Normal

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

**Location**: `volkit/_kernels/`

**Pattern**:
1. UID format: `"Component1(p,q)+Component2(r,s)+Density"`
2. Registry maps UID → kernel module
3. Each module provides `get_routine(uid) -> Routine`
4. `Routine` object encapsulates fit logic

**Files involved**:
- `volkit/_kernels/__init__.py` - Registry and dispatcher
- `volkit/_kernels/routine.py` - `Routine` dataclass
- `volkit/_kernels/garch_normal.py` - GARCH+Normal implementations
- `volkit/_kernels/garch_studentt.py` - GARCH+StudentT implementations

### What's Stable

**C extensions are the single source of truth for computations.**

Never duplicate computational logic in Python. If you need a new calculation:
1. Implement in C (`volkit/_csrc/`)
2. Add declaration to `volkit/_csrc/volkit_core.h`
3. Wrap in `volkit/_core.c` with METH_FASTCALL
4. Add type stub to `volkit/_core.pyi`
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
name = "volkit._core"
sources = [
    "volkit/_core.c",
    "volkit/_csrc/variance_garch.c",
    "volkit/_csrc/likelihood_garch.c",
    "volkit/_csrc/likelihood_normal.c",
    "volkit/_csrc/likelihood_studentt.c",
    "volkit/_csrc/errors_garch.c",
]
include-dirs = ["volkit/_csrc"]
```

**Required files**:
- `MANIFEST.in`: Includes C headers in source distribution
- `pyproject.toml`: Build configuration
- `volkit/__init__.py`: Package entry point
- `volkit/_core.pyi`: Type stubs for C extension

### Verify Build

```python
# Should load without errors
import volkit._core
print(volkit._core.__file__)  # Shows .so location

# Test a C function
import numpy as np
theta = np.array([1e-6, 0.05, 0.9], dtype=np.float64)
resid2 = np.random.randn(100)**2
sigma2 = np.empty(100, dtype=np.float64)

from volkit import _core
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
from volkit import GARCH, StudentT
import numpy as np

# Generate sample data
returns = np.random.randn(1000) * 0.01

# Specify and fit
spec = GARCH(1, 1) + StudentT()
result = spec.fit(returns)

# Access results
print(f"Log-likelihood: {result.loglikelihood}")
print(f"Parameters: {result.params}")
print(f"AIC: {result.aic}, BIC: {result.bic}")
```

**Add a new component**:

1. Create `volkit/components/my_component.py`
2. Inherit from `Component`, set `role`
3. Implement all abstract methods
4. Add to `volkit/components/__init__.py`
5. Add C implementations if needed
6. Write tests in `tests/`

**Add a new C function**:

1. Implement in `volkit/_csrc/my_function.c`
2. Declare in `volkit/_csrc/volkit_core.h`
3. Wrap in `volkit/_core.c` with proper pointer handling
4. Add type stub to `volkit/_core.pyi`
5. Validate with finite differences

### Directory Structure

```
volkit_cursor/                   # Repository root
├── volkit/                      # Core package (ships)
│   ├── _core.c                  # C extension wrapper (Python-C interface)
│   ├── _core.pyi                # Type stubs for C functions
│   ├── _csrc/                   # C implementation
│   │   ├── volkit_core.h        # Public C API declarations
│   │   ├── math_and_helpers.h   # Shared math (lgamma, digamma, constants)
│   │   ├── variance_garch.c     # GARCH variance recursion
│   │   ├── likelihood_*.c       # Distribution log-likelihoods
│   │   ├── arma_garch.c         # ARMA-GARCH NLL + gradients (Normal/t/Skew-t)
│   │   └── errors_garch.c       # OPG and Hessian computation
│   ├── _kernels/                # Optimization kernels (internal)
│   ├── components/              # User-facing components
│   ├── spec/                    # Composition logic
│   ├── estimators/              # Estimation methods
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
    ├── numerical_hessians.py    # Reference: Finite difference validation
    ├── utilities.py             # Reference: Statistical tests
    ├── analysis.py              # Analysis scripts
    └── ...                      # Other experiments and dev tools

# Also git-ignored: localdev_data/, localdev_results/, localdev_backup/, etc.
```

### Import Patterns

```python
# Public API
from volkit import GARCH, ARMA, Normal, StudentT, MLE, CompositeSpec, Role

# Internal (for development)
from volkit._core import _garch_variance_11, _garch_ll_pq_normal
from volkit._kernels import get_routine
from volkit.components.base import Component
from volkit.result import EstimationResult
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

### Immediate Goals

1. **Complete C gradient/Hessian** for Student-t and Skew-t distributions
2. **Integrate into volkit component system** - wire up C functions to `volkit/` API
3. **Add missing capabilities** to `volkit/`:
   - ✅ `SkewT` component wrapping C implementation
   - ✅ Robust standard errors (QMLE implementation)
   - ✅ Rich result objects (sigma2, standardized residuals, timing)
   - Log-space (unconstrained) optimization option (partially implemented)

### Future Models

New model families will follow the component pattern:

```python
# Example: GJR-GARCH (asymmetric GARCH)
from volkit import GJRGARCH, StudentT

spec = GJRGARCH(1, 1) + StudentT()
result = spec.fit(returns)
```

Each model family will need:
- Component class (e.g., `GJRGARCH(Component)`)
- C implementations for performance-critical paths
- Kernel routines for optimization
- Comprehensive tests with derivative validation

---

## Notes for AI Agents

### When in Doubt

1. **User API**: Keep component composition + `.fit()` pattern
2. **C functions**: Always single source of truth for computation
3. **Memory**: Pre-allocate in Python, pass pointers to C
4. **Arrays**: Always `np.ascontiguousarray(arr, dtype=np.float64)`
5. **Derivatives**: Validate with finite differences
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
2. **Validate derivatives** against finite differences
3. **Port to C** using the Numba code as reference
4. **Verify C matches Numba** to <1e-12 precision

This pattern was used for `localdev/arma_garch_estimator.py` → `arma_garch.c`.

### Benchmark Testing (Keep Evergreen)

**IMPORTANT: When adding a new model/routine, you MUST add it to:**
1. **`benchmark_optimizers.py`** - Tests optimizer configurations with real data
2. **`tests/test_dgp_estimation.py`** - Tests parameter recovery with DGP data

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

**Current Recommended Defaults** (as of 2026-01-30):

**GARCH(1,1) Models:**

| Distribution | Solver | Log Mode | Avg Time | Conv Rate |
|--------------|--------|----------|----------|-----------|
| Normal       | slsqp  | True     | 0.003s   | 100%      |
| Student-t    | slsqp  | True     | 0.008s   | 100%      |
| Skew-t       | slsqp  | False    | 0.033s   | 100%      |

**ARMA(1,1)-GARCH(1,1) Models:**

| Distribution | Solver | Log Mode | Avg Time | Conv Rate |
|--------------|--------|----------|----------|-----------|
| Normal       | slsqp  | False    | 0.005s   | 100%      |
| Student-t    | slsqp  | False    | 0.055s   | 100%      |
| Skew-t       | slsqp  | False    | 0.027s   | 100%      |

Key findings:
- `slsqp` is fastest AND most reliable across all distributions
- `trust` solver has poor convergence (0-67%), avoid as default
- `nelder-mead` is reliable but 5-10x slower than `slsqp`
- Log-mode (unconstrained) works well for GARCH Normal/Student-t
- Constrained mode better for Skew-t and all ARMA-GARCH models

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
- Implementation examples in `volkit/components/` and `volkit/_kernels/`
- Reference implementations in `localdev/arma_garch_estimator.py`
- Test patterns in `tests/`
- C interface in `volkit/_core.c` and `volkit/_core.pyi`

**New development work** (scripts, experiments, analysis) should go in `localdev/`.
