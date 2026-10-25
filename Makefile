b:
	rm -rf build/ dist/
	python -m build -w
	python -m pip install --force-reinstall dist/scivol-*.whl

# Editable install: Python loads from the tree, extensions from the persistent
# build/ directory, so rerunning `make dev` rebuilds only what changed.
dev:
	CMAKE_BUILD_PARALLEL_LEVEL=4 pip install -e .[dev] --no-build-isolation

t:
	pytest -q tests/test_spec_core.py tests/test_component_view.py tests/test_result_core.py tests/test_fit_mixins.py
	pytest -q tests/test_dgp_estimation.py

# Derivative validation against independent JAX references (requires jax)
deriv:
	pytest -q tests/test_scivolapi_garch_normal.py tests/test_scivolapi_densities.py tests/test_scivolapi_volatility.py tests/test_scivolapi_mean.py tests/test_scivolapi_dcc.py

# Native library tests (C++)
native:
	cmake -S scivolapi -B build/scivolapi-native -G Ninja -DCMAKE_BUILD_TYPE=Release
	cmake --build build/scivolapi-native -j4
	ctest --test-dir build/scivolapi-native

# Full DGP estimation tests
dgp:
	pytest -v tests/test_dgp_estimation.py

bench:
	python benchmark.py

f:
	rm -rf build/ dist/ 
	python -m build