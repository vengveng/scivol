b:
	rm -rf build/ dist/
	python -m build -w
	python -m pip install --force-reinstall dist/scivol-*.whl

dev:
	rm -rf build/ dist/
	pip install -e .[dev] --no-build-isolation --config-settings="--build-option=build_ext --build-option=--inplace"

t:
	pytest -q tests/test_spec_core.py
	pytest -q tests/test_component_view.py 
	pytest -q tests/test_result_core.py
	pytest -q tests/test_fit_mixins.py
	pytest -q tests/test_mle_core.py
	pytest -q tests/test_dgp_estimation.py

# Development-time derivative validation suite (requires `make dev`)
deriv:
	pytest -q tests/test_dcc.py
	pytest -q tests/test_gradients_gjr_garch.py
	pytest -q tests/test_ad_oracle_models.py

# Full DGP estimation tests (slower, more thorough)
dgp:
	pytest -v tests/test_dgp_estimation.py
	

f:
	rm -rf build/ dist/ 
	python -m build