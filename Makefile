b:
	rm -rf build/ dist/
	python -m build
	python -m pip install --force-reinstall dist/volkit-*.whl

t:
	pytest -q tests/test_spec_core.py
	pytest -q tests/test_component_view.py 
	pytest -q tests/test_result_core.py

f:
	rm -rf build/ dist/
	python -m build