# Makefile to support common developer commands

all :: style docs coverage

clean ::
	make -C docs clean

coverage:
	coverage run --concurrency=thread --parallel-mode -m pytest -q ./hklpy2
	coverage combine
	coverage report --precision 3 -m
	coverage html

docs ::
	make -C docs html

doc :: docs

geo_tables:
	python ./docs/make_geometries_doc.py

isort:
	isort --sl ./hklpy2

pre:
	pre-commit run --all-files
	ruff check .

realclean :: clean
	/bin/rm -rf ./docs/build

style :: isort pre

test:
	pytest -q --lf ./hklpy2
