# AeroSentry AI - Makefile
.PHONY: help install install-edge install-cloud setup-db test lint format docker-edge docker-cloud clean

PYTHON := python3
PIP := pip3

help:
	@echo "AeroSentry AI - Aviation Security System"
	@echo ""
	@echo "Usage:"
	@echo "  make install        - Install all dependencies"
	@echo "  make install-edge   - Install edge node dependencies"
	@echo "  make install-cloud  - Install cloud service dependencies"
	@echo "  make setup-db       - Initialize TimescaleDB schema"
	@echo "  make test           - Run all tests"
	@echo "  make test-unit      - Run unit tests"
	@echo "  make test-int       - Run integration tests"
	@echo "  make lint           - Run linters"
	@echo "  make format         - Format code"
	@echo "  make docker-edge    - Build and run edge Docker containers"
	@echo "  make docker-cloud   - Build and run cloud Docker containers"
	@echo "  make clean          - Clean temporary files"
	@echo "  make evaluate       - Run anomaly detection evaluation"

# Installation targets
install:
	$(PIP) install -r requirements.txt

install-edge:
	$(PIP) install -r requirements-edge.txt

install-cloud:
	$(PIP) install -r requirements-cloud.txt

# Database setup
setup-db:
	$(PYTHON) scripts/setup_db.py

# Testing targets
test:
	pytest tests/ -v --tb=short

test-unit:
	pytest tests/ -v --tb=short -m "not integration"

test-int:
	pytest tests/integration/ -v --tb=short

# Code quality
lint:
	ruff check edge/ cloud/ shared/ disaster/
	mypy edge/ cloud/ shared/ disaster/ --ignore-missing-imports

format:
	ruff format edge/ cloud/ shared/ disaster/
	isort edge/ cloud/ shared/ disaster/

# Docker targets
docker-edge:
	cd docker && docker-compose -f docker-compose.edge.yml up --build

docker-edge-detach:
	cd docker && docker-compose -f docker-compose.edge.yml up --build -d

docker-cloud:
	cd docker && docker-compose -f docker-compose.cloud.yml up --build

docker-cloud-detach:
	cd docker && docker-compose -f docker-compose.cloud.yml up --build -d

docker-stop:
	cd docker && docker-compose -f docker-compose.edge.yml down
	cd docker && docker-compose -f docker-compose.cloud.yml down

# Kubernetes deployment
k8s-deploy:
	kubectl apply -k deploy/kubernetes/

k8s-delete:
	kubectl delete -k deploy/kubernetes/

# Evaluation
evaluate:
	$(PYTHON) scripts/run_evaluation.py --n-normal 100 --n-attacks 40

evaluate-full:
	$(PYTHON) scripts/run_evaluation.py --n-normal 500 --n-attacks 200 --output evaluation_report.md

# Development
dev-edge:
	$(PYTHON) -m edge.main

dev-cloud:
	$(PYTHON) -m cloud.main

# Clean
clean:
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	find . -type f -name "*.pyo" -delete 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".mypy_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	rm -rf build/ dist/
