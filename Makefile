b:
	rm -rf build/ dist/
	python -m build
	python -m pip install --force-reinstall dist/volkit-*.whl