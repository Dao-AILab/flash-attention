clean_dist:
	rm -rf dist/*

create_dist: clean_dist
	python -m build

upload_package: create_dist
	twine upload dist/*
