
clean_dist:
	rm -rf dist/*

# This step assumes there is already a virtualenv active.
create_dist: clean_dist
	python -m pip install build
	python -m build --sdist .

upload_package: create_dist
	twine upload dist/*
