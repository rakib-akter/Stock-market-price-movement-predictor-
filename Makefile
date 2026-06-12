# Convenience targets. On Windows, run the commands directly or use `make` from
# Git Bash / WSL.

.PHONY: help install dev test lint format api dashboard pipeline clean

help:
	@echo "install    Install runtime dependencies"
	@echo "dev        Install dev dependencies (tests, lint, notebooks)"
	@echo "test       Run the test suite"
	@echo "lint       Run ruff + mypy"
	@echo "format     Auto-format with black + ruff --fix"
	@echo "api        Start the FastAPI server (reload)"
	@echo "dashboard  Start the Streamlit dashboard"
	@echo "pipeline   Run end-to-end pipeline (TICKER=AAPL)"
	@echo "clean      Remove caches and build artifacts"

install:
	pip install -r requirements.txt

dev:
	pip install -r requirements-dev.txt

test:
	pytest

lint:
	ruff check backend tests scripts
	mypy backend

format:
	black backend tests scripts frontend
	ruff check --fix backend tests scripts

api:
	uvicorn backend.app.api.main:app --reload

dashboard:
	streamlit run frontend/streamlit_app.py

TICKER ?= AAPL
pipeline:
	python -m scripts.run_pipeline --ticker $(TICKER)

clean:
	rm -rf .pytest_cache .ruff_cache .mypy_cache **/__pycache__ htmlcov .coverage
