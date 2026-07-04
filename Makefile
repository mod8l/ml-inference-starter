.PHONY: install dev test lint docker-build generate-data compose-up load-test clean

VENV ?= .venv
PYTHON := $(VENV)/bin/python
PIP := $(VENV)/bin/pip

$(VENV):
	python3 -m venv $(VENV)
	$(PIP) install --upgrade pip

install: $(VENV)
	$(PIP) install -e ".[dev]"

dev: $(VENV)
	MODEL_PATH=models/dummy_classifier.pt $(VENV)/bin/uvicorn inference.server:app --app-dir src --reload --host 0.0.0.0 --port 8000

test: $(VENV)
	PYTHONPATH=src $(PYTHON) -m pytest tests -v

lint: $(VENV)
	$(VENV)/bin/ruff check src tests
	$(VENV)/bin/ruff format --check src tests

docker-build:
	docker build -t ml-inference-starter:latest .

generate-data: $(VENV)
	PYTHONPATH=src $(PYTHON) data/dummy/generate.py

compose-up:
	docker compose up --build -d

load-test: $(VENV)
	$(VENV)/bin/locust -f locustfile.py --host http://localhost:8000 --users 10 --spawn-rate 5 --run-time 30s --headless

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	rm -rf .ruff_cache .pytest_cache build dist *.egg-info 2>/dev/null || true
