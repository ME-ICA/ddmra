.PHONY: all help lint format unittest integrationtest

all_tests: lint unittest integrationtest

help:
	@echo "Please use 'make <target>' where <target> is one of:"
	@echo "  lint			to run ruff linting on all Python files"
	@echo "  format			to format all Python files with ruff"
	@echo "  unittest		to run unit tests on ddmra"
	@echo "  integrationtest	to run integration tests"

lint:
	@ruff check src/ddmra
	@ruff format --check src/ddmra

format:
	@ruff check --fix src/ddmra
	@ruff format src/ddmra

unittest:
	@pytest -m "not integration" --cov-append --cov-report xml --cov-report term-missing --cov=ddmra

integrationtest:
	@pytest -m "integration" --cov-append --cov-report xml --cov-report term-missing --cov=ddmra
