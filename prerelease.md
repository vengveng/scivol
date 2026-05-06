Yes. The kernel-start asymmetries are not what I’d worry about right before an alpha. The higher-priority publication items from a quick repo pass are these:

## Must Fix
- **Add a license file.** I didn’t find any `LICENSE*` at the repo root. That’s a real publication blocker.
- **Fill out `pyproject.toml` metadata.** Right now it has only:
  - `name`
  - `version`
  - a very thin description: `"GARCH helpers in C"`
  
  You should add at least:
  - `readme = "README.md"`
  - `license`
  - `authors`
  - `classifiers`
  - `keywords`
  - `project.urls`

  Without `readme`, your package page will be much poorer than it should be.

- **Do not publish portable wheels with `-march=native`.** In `pyproject.toml`, your extension uses:

```toml
extra-compile-args = ["-O3", "-ffast-math", "-march=native", "-ffp-contract=fast", "-funroll-loops"]
```

  `-march=native` is fine for local benchmarking, but it makes binaries tuned to the build machine. If you publish wheels built that way, they may not work correctly on other CPUs. For alpha publication:
  - either publish **sdist only**
  - or remove `-march=native` for release builds

## Should Fix
- **README transform docs are stale.** The README still says log-mode uses `exp(z)` for `omega` and `2 + exp(z)` for `nu`, but the actual code now uses `softplus`.

```218:228:README.md
**Log-mode** (`log_mode=True`) transforms constrained parameters into unconstrained space before optimization:

| Parameter | Constraint | Transform |
|-----------|------------|-----------|
| ω | ω > 0 | exp(z) |
...
| ν | ν > 2 | 2 + exp(z) |
```

  Actual implementation:

```5:7:scivol/_kernels/transforms.py
- ω = softplus(z_ω)               ensures ω > 0
- ν = 2 + softplus(z_ν)           ensures ν > 2 (Student-t)
```

  I would fix that before publishing.

- **Decide how public `DCC` is for alpha.** It is exported in `scivol/__init__.py` and documented in `README.md`. If you consider it experimental, label it clearly or pull it from the alpha-facing docs/export surface.

- **Decide whether `_devtools` should ship.** It is currently included in the package list in `pyproject.toml`. That may be fine, but if you want a cleaner alpha surface, consider whether internal tooling belongs in the published package.

- **Explicitly ship typing artifacts if you care about types.** You have `scivol/_core.pyi`, but I don’t see explicit package-data configuration for it, nor a `py.typed`. If typing is part of the alpha story, make that deliberate.

## Nice To Clean Up
- `README.md` currently shows source/editable install commands, not the polished install story users will see after publication.
- `benchmark_optimizers.py` references ignored local data paths. If that file will be visible as a shipped/public benchmark script, it should either be self-contained or moved under `localdev/`.

## Release Dry Run
Before you publish, I would absolutely do this sequence:

1. `python -m build`
2. `twine check dist/*`
3. create a clean venv
4. install from the built artifact, not editable source
5. smoke test:
   - `import scivol`
   - fit a simple `GARCH(1,1)+Normal()`
   - fit one `GJRGARCH(1,1)+StudentT()`
6. if you publish wheels, test on at least one other machine/CPU family or remove `-march=native`

If you want, I can next make the concrete alpha-publication fixes for:
- `pyproject.toml` metadata
- `README.md` transform/docs cleanup
- adding a `LICENSE`
- a release dry-run checklist file.