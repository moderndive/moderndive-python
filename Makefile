.PHONY: help install test cov lint build build-data clean

help:
	@echo "install     Create the dev environment (uv sync --extra dev)"
	@echo "test        Run the test suite (enforces 100% coverage)"
	@echo "cov         Run tests and write an HTML coverage report"
	@echo "lint        Run ruff"
	@echo "build       Build the wheel and sdist"
	@echo "build-data  Rebuild the bundled Parquet datasets from CSVs (see tools/)"
	@echo "clean       Remove build/cache artifacts"

install:
	uv sync --extra dev

test:
	uv run pytest

cov:
	uv run pytest --cov-report=html
	@echo "open htmlcov/index.html"

lint:
	uv run ruff check moderndive tests

build:
	uv build

build-data:
	uv run python tools/build_data.py $(CSV_DIR)

clean:
	rm -rf dist build htmlcov .pytest_cache .coverage
	find . -type d -name __pycache__ -exec rm -rf {} +
