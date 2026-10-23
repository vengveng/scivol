# Standalone mathematical core

`scivolapi` is the C++20 mathematical library under development for the scivol
transition. It does not import Python, link the installed scivol extension, fit
models, or alter Python routing. Implementation availability is not mathematical
qualification; the Phase 1 scope and remaining evidence are recorded in
`../contracts/`.

## Build and use

From the repository root:

```sh
cmake -S core -B localdev/scivolapi-build -DCMAKE_BUILD_TYPE=Release
cmake --build localdev/scivolapi-build --parallel 2
ctest --test-dir localdev/scivolapi-build --output-on-failure
cmake --install localdev/scivolapi-build --prefix /your/chosen/prefix
```

An external CMake project uses `find_package(scivolapi CONFIG REQUIRED)` and links
`scivolapi::scivolapi`. Include `<scivolapi/scivolapi.hpp>`. The executable example
is `../examples/native/jet.cpp`; the installed-package consumer is
`../tests/native/installed/`.

## Evaluation contract

Compile a `ModelSpec`, prepare its immutable sample, resolve an objective, reserve
the requested workspace, bind a theta or original-chart point, then evaluate.
Model compilation selects existing native programs; it does not invoke a compiler.
Owning preparation copies inputs. Borrowed preparation requires an immutable lease
whose data remain unchanged for the prepared problem's lifetime.

`F` is the complete additive negative log-likelihood on the declared scored rows.
`G` and `H` use the resolved derivative layout. Every nonempty F/G/H mask is a
request for those outputs, not a promise of one traversal. Check `valid_mask`
before reading requested buffers; successful value evaluation does not certify
derivatives. Unrequested output buffers are not modified. Rebinding a workspace
invalidates its previous borrowed point view.

`ScoreSelection` controls accumulation, not the training information cutoff.
Earlier retained rows still advance the recursion. A default seed may use the
whole retained training sample, and its provenance records those rows. Explicit
histories, unit maps, profiles, or other operations without a registered program
fail explicitly instead of being ignored.

Independent development tests call the new API through `../tests/bridges/`.
The mandatory oracle suite forbids imports from the old Python package. Historical
comparisons and benchmark tools remain separate from production mathematics.
