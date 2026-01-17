.RECIPEPREFIX := >
PYTHON ?= python

.PHONY: help setup lint format test train explain consistency

help:
> @echo Targets: setup lint format test train explain consistency

setup:
> $(PYTHON) -m pip install -U pip
> $(PYTHON) -m pip install -e .[dev]

lint:
> ruff check src tests

format:
> ruff check --fix src tests
> black src tests

test:
> pytest -q

train:
> $(PYTHON) -m aet.cli --config configs/base.yaml --stage train

explain:
> $(PYTHON) -m aet.cli --config configs/base.yaml --stage explain

consistency:
> $(PYTHON) -m aet.cli --config configs/base.yaml --stage consistency
