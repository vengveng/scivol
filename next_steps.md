# Next Steps

## Validation Framework

- Keep production derivatives in analytical C code for now.
- Replace finite-difference gradient validation with a stronger oracle:
  either a small dual-number reference engine or an AD oracle such as `jax`
  for smooth reference implementations.
- Replace finite-difference Hessian validation with a stronger oracle:
  hyper-duals for scalar recursive models, and AD or directional second-order
  checks for matrix-heavy models such as `DCC`.
- Keep finite differences only as a coarse smoke test, not as the main source
  of truth for derivative validation.

## Scope of Validation

- Validate the same objective that the optimizer actually sees.
- For constrained-mode models, validate `L(theta)`.
- For log-mode models, validate the transformed objective `L(theta(z))`, not
  only the inner likelihood in `theta` space.
- Include the parameter transforms in the derivative graph:
  `exp`, softmax-like stationarity maps, `tanh`, softplus, and any Jacobian or
  chain-rule logic used in the fused log-space routines.

## Recommended Oracle Split

- Use a tiny in-repo hyper-dual validator for scalar recursive models:
  `GARCH`, `GJR-GARCH`, and likely selected `ARMA-GARCH` reference paths.
- Use an AD oracle for matrix-heavy models such as `DCC`, where recursion plus
  matrix normalization and Cholesky steps make a handwritten hyper-dual engine
  less attractive.
- Prefer directional Hessian checks for larger models when a full dense Hessian
  oracle becomes expensive.

## Testing Structure

- Layer 1: validate primitive transforms in isolation.
- Layer 2: validate one-step recursion updates and their derivatives.
- Layer 3: validate the full objective on random interior parameter draws and
  synthetic data.
- Compare both absolute and relative errors, especially near small gradients or
  saturated transforms.
- Avoid validation points near kinks introduced by floors, clipping, or
  emergency penalties.

## Immediate Follow-Ups

- Replace current FD-based `z`-space Hessian checks with a stronger oracle.
- Build a reusable validator module for `GARCH(1,1)` and `GJR-GARCH(1,1)`
  first, then expand to additional families.
- Extend the oracle-based checks to Student-t and Skew-t models after the
  primitive special-function rules are validated carefully.
- Investigate whether AD inside the C runtime is feasible enough for some model
  families to reduce derivative-kernel development cost without unacceptable
  runtime regressions.
