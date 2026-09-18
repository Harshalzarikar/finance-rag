PY ?= python

.PHONY: help install install-dev lint format types test check dev ingest ingest-reset verify eval up down logs ps clean

help: ## Show available targets
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-16s\033[0m %s\n", $$1, $$2}'

install: ## Install runtime dependencies
	$(PY) -m pip install -r requirements.txt

install-dev: ## Install runtime and development dependencies
	$(PY) -m pip install -r requirements-dev.txt

lint: ## Lint with ruff
	$(PY) -m ruff check .

format: ## Format with ruff
	$(PY) -m ruff format .

types: ## Type-check with mypy
	$(PY) -m mypy

test: ## Run the test suite
	$(PY) -m pytest -q

check: lint types test ## Run every quality gate

dev: ## Run the API with autoreload on :8000
	$(PY) -m uvicorn src.api.main:app --reload --port 8000

ingest: ## Ingest every PDF in RAW_PDFS_DIR
	$(PY) scripts/ingest.py

ingest-reset: ## Wipe the indexes and re-ingest from scratch
	$(PY) scripts/ingest.py --reset

verify: ## Compare the manifest against the vector store and keyword index
	$(PY) scripts/ingest.py --verify

eval: ## Score retrieval against tests/eval/golden_queries.json
	$(PY) tests/eval/run_eval.py

up: ## Start the production stack
	docker compose up -d --build

down: ## Stop the production stack
	docker compose down

logs: ## Tail API logs
	docker compose logs -f api

ps: ## Show stack status
	docker compose ps

clean: ## Remove tool caches
	rm -rf .pytest_cache .mypy_cache .ruff_cache
	find . -name __pycache__ -type d -not -path './venv/*' -prune -exec rm -rf {} +
