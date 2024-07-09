
clean_build:
	rm -rf build/*

clean_dist:
	rm -rf dist/*

setup_install: clean_build
	python setup.py install

create_dist: clean_dist
	python setup.py sdist

upload_package: create_dist
	twine upload dist/*
