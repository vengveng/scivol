b:
	rm -rf build/ dist/
	python -m build -w
	python -m pip install --force-reinstall dist/volkit-*.whl

dev:
	rm -rf build/ dist/
	pip install -e . --no-build-isolation --config-settings="--build-option=build_ext --build-option=--inplace"

t:
	pytest -q tests/test_spec_core.py
	pytest -q tests/test_component_view.py 
	pytest -q tests/test_result_core.py
	pytest -q tests/test_fit_mixins.py
	pytest -q tests/test_mle_core.py
	

f:
	rm -rf build/ dist/
	python -m build