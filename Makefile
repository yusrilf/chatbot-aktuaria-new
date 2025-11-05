# Makefile for Actuarial Chatbot - Test Suite and Development Commands
# This file provides convenient commands for testing, development, and deployment

# Variables
PYTHON := python3
PIP := pip3
PYTEST := pytest
PROJECT_NAME := actuarial-chatbot
TEST_RUNNER := $(PYTHON) run_tests.py
REPORTS_DIR := test_reports
COVERAGE_DIR := $(REPORTS_DIR)/coverage_html

# Colors for output
RED := \033[0;31m
GREEN := \033[0;32m
YELLOW := \033[1;33m
BLUE := \033[0;34m
NC := \033[0m # No Color

# Default target
.DEFAULT_GOAL := help

# Help target
.PHONY: help
help: ## Show this help message
	@echo "$(BLUE)Actuarial Chatbot - Development Commands$(NC)"
	@echo ""
	@echo "$(YELLOW)Available commands:$(NC)"
	@awk 'BEGIN {FS = ":.*?## "} /^[a-zA-Z_-]+:.*?## / {printf "  $(GREEN)%-20s$(NC) %s\n", $$1, $$2}' $(MAKEFILE_LIST)
	@echo ""
	@echo "$(YELLOW)Test Categories:$(NC)"
	@echo "  $(GREEN)unit$(NC)                Run unit tests only"
	@echo "  $(GREEN)integration$(NC)         Run integration tests only"
	@echo "  $(GREEN)e2e$(NC)                 Run end-to-end tests only"
	@echo "  $(GREEN)performance$(NC)         Run performance tests only"
	@echo "  $(GREEN)security$(NC)            Run security tests only"
	@echo "  $(GREEN)all$(NC)                 Run all test categories"

# Environment Setup
.PHONY: install
install: ## Install all dependencies
	@echo "$(BLUE)Installing dependencies...$(NC)"
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt
	$(PIP) install -r requirements-test.txt
	@echo "$(GREEN)Dependencies installed successfully!$(NC)"

.PHONY: install-dev
install-dev: install ## Install development dependencies
	@echo "$(BLUE)Installing development dependencies...$(NC)"
	$(PIP) install black isort flake8 mypy pylint bandit safety pre-commit
	@echo "$(GREEN)Development dependencies installed successfully!$(NC)"

.PHONY: setup
setup: install-dev ## Complete development environment setup
	@echo "$(BLUE)Setting up development environment...$(NC)"
	@mkdir -p $(REPORTS_DIR)
	@mkdir -p logs
	@mkdir -p uploads
	@mkdir -p screenshots
	pre-commit install
	@echo "$(GREEN)Development environment setup complete!$(NC)"

# Code Quality
.PHONY: format
format: ## Format code with black and isort
	@echo "$(BLUE)Formatting code...$(NC)"
	black app/ tests/ --line-length 100
	isort app/ tests/ --profile black
	@echo "$(GREEN)Code formatting complete!$(NC)"

.PHONY: lint
lint: ## Run all linting tools
	@echo "$(BLUE)Running linting tools...$(NC)"
	@echo "$(YELLOW)Running flake8...$(NC)"
	flake8 app/ tests/ --max-line-length=100 --extend-ignore=E203,W503
	@echo "$(YELLOW)Running mypy...$(NC)"
	mypy app/ --ignore-missing-imports --no-strict-optional
	@echo "$(YELLOW)Running pylint...$(NC)"
	pylint app/ --disable=C0114,C0115,C0116,R0903,R0913 --max-line-length=100
	@echo "$(GREEN)Linting complete!$(NC)"

.PHONY: security-check
security-check: ## Run security checks
	@echo "$(BLUE)Running security checks...$(NC)"
	@echo "$(YELLOW)Running bandit...$(NC)"
	bandit -r app/ --severity-level medium
	@echo "$(YELLOW)Running safety...$(NC)"
	safety check
	@echo "$(GREEN)Security checks complete!$(NC)"

.PHONY: check
check: format lint security-check ## Run all code quality checks
	@echo "$(GREEN)All code quality checks passed!$(NC)"

# Testing Commands
.PHONY: test
test: ## Run all tests
	@echo "$(BLUE)Running comprehensive test suite...$(NC)"
	$(TEST_RUNNER) --category all
	@echo "$(GREEN)All tests completed!$(NC)"

.PHONY: test-unit
test-unit: ## Run unit tests only
	@echo "$(BLUE)Running unit tests...$(NC)"
	$(TEST_RUNNER) --category unit
	@echo "$(GREEN)Unit tests completed!$(NC)"

.PHONY: test-integration
test-integration: ## Run integration tests only
	@echo "$(BLUE)Running integration tests...$(NC)"
	$(TEST_RUNNER) --category integration
	@echo "$(GREEN)Integration tests completed!$(NC)"

.PHONY: test-e2e
test-e2e: ## Run end-to-end tests only
	@echo "$(BLUE)Running end-to-end tests...$(NC)"
	$(TEST_RUNNER) --category e2e
	@echo "$(GREEN)End-to-end tests completed!$(NC)"

.PHONY: test-performance
test-performance: ## Run performance tests only
	@echo "$(BLUE)Running performance tests...$(NC)"
	$(TEST_RUNNER) --category performance
	@echo "$(GREEN)Performance tests completed!$(NC)"

.PHONY: test-security
test-security: ## Run security tests only
	@echo "$(BLUE)Running security tests...$(NC)"
	$(TEST_RUNNER) --category security
	@echo "$(GREEN)Security tests completed!$(NC)"

.PHONY: test-fast
test-fast: ## Run fast tests only (skip slow tests)
	@echo "$(BLUE)Running fast tests...$(NC)"
	$(TEST_RUNNER) --category all --skip-slow
	@echo "$(GREEN)Fast tests completed!$(NC)"

.PHONY: test-parallel
test-parallel: ## Run tests in parallel
	@echo "$(BLUE)Running tests in parallel...$(NC)"
	$(TEST_RUNNER) --category unit
	$(TEST_RUNNER) --category integration
	@echo "$(GREEN)Parallel tests completed!$(NC)"

.PHONY: test-coverage
test-coverage: ## Run tests with coverage report
	@echo "$(BLUE)Running tests with coverage...$(NC)"
	$(PYTEST) tests/unit/ --cov=app --cov-report=html --cov-report=term-missing --cov-report=xml
	@echo "$(GREEN)Coverage report generated at $(COVERAGE_DIR)/index.html$(NC)"

# Performance Testing
.PHONY: test-perf-small
test-perf-small: ## Run performance tests with small load
	@echo "$(BLUE)Running performance tests (small load)...$(NC)"
	$(TEST_RUNNER) --category performance --load-level small

.PHONY: test-perf-medium
test-perf-medium: ## Run performance tests with medium load
	@echo "$(BLUE)Running performance tests (medium load)...$(NC)"
	$(TEST_RUNNER) --category performance --load-level medium

.PHONY: test-perf-large
test-perf-large: ## Run performance tests with large load
	@echo "$(BLUE)Running performance tests (large load)...$(NC)"
	$(TEST_RUNNER) --category performance --load-level large

# Specific Test Scenarios
.PHONY: test-smoke
test-smoke: ## Run smoke tests (basic functionality)
	@echo "$(BLUE)Running smoke tests...$(NC)"
	$(PYTEST) -m "not slow and not performance" tests/ -v
	@echo "$(GREEN)Smoke tests completed!$(NC)"

.PHONY: test-regression
test-regression: ## Run regression tests
	@echo "$(BLUE)Running regression tests...$(NC)"
	$(TEST_RUNNER) --category all
	@echo "$(GREEN)Regression tests completed!$(NC)"

.PHONY: test-api
test-api: ## Run API-specific tests
	@echo "$(BLUE)Running API tests...$(NC)"
	$(PYTEST) tests/integration/test_*_api_*.py -v
	@echo "$(GREEN)API tests completed!$(NC)"

# Development Server
.PHONY: run
run: ## Run the development server
	@echo "$(BLUE)Starting development server...$(NC)"
	FLASK_ENV=development $(PYTHON) -m app.main

.PHONY: run-prod
run-prod: ## Run the production server
	@echo "$(BLUE)Starting production server...$(NC)"
	FLASK_ENV=production $(PYTHON) -m app.main

.PHONY: run-debug
run-debug: ## Run the server in debug mode
	@echo "$(BLUE)Starting debug server...$(NC)"
	FLASK_ENV=development FLASK_DEBUG=1 $(PYTHON) -m app.main

# Database Operations
.PHONY: db-init
db-init: ## Initialize database
	@echo "$(BLUE)Initializing database...$(NC)"
	$(PYTHON) -c "from app.models import db; db.create_all()"
	@echo "$(GREEN)Database initialized!$(NC)"

.PHONY: db-reset
db-reset: ## Reset database
	@echo "$(BLUE)Resetting database...$(NC)"
	$(PYTHON) -c "from app.models import db; db.drop_all(); db.create_all()"
	@echo "$(GREEN)Database reset complete!$(NC)"

# Cleanup Commands
.PHONY: clean
clean: ## Clean up generated files
	@echo "$(BLUE)Cleaning up...$(NC)"
	rm -rf __pycache__/
	rm -rf .pytest_cache/
	rm -rf .coverage
	rm -rf htmlcov/
	rm -rf $(REPORTS_DIR)/
	rm -rf .mypy_cache/
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	@echo "$(GREEN)Cleanup complete!$(NC)"

.PHONY: clean-logs
clean-logs: ## Clean up log files
	@echo "$(BLUE)Cleaning logs...$(NC)"
	rm -rf logs/*.log
	rm -f test_runner.log
	@echo "$(GREEN)Logs cleaned!$(NC)"

.PHONY: clean-uploads
clean-uploads: ## Clean up uploaded files
	@echo "$(BLUE)Cleaning uploads...$(NC)"
	rm -rf uploads/*
	@echo "$(GREEN)Uploads cleaned!$(NC)"

.PHONY: clean-all
clean-all: clean clean-logs clean-uploads ## Clean everything
	@echo "$(GREEN)Complete cleanup finished!$(NC)"

# Documentation
.PHONY: docs
docs: ## Generate documentation
	@echo "$(BLUE)Generating documentation...$(NC)"
	@mkdir -p docs
	$(PYTHON) -c "
	import os
	import subprocess
	if os.path.exists('app'):
		subprocess.run(['pydoc', '-w', 'app'])
		print('Documentation generated!')
	else:
		print('No app directory found')
	"
	@echo "$(GREEN)Documentation generated!$(NC)"

# Reporting
.PHONY: report
report: ## Generate comprehensive test report
	@echo "$(BLUE)Generating test reports...$(NC)"
	$(TEST_RUNNER) --category all
	@echo "$(GREEN)Reports available in $(REPORTS_DIR)/$(NC)"
	@echo "$(YELLOW)Open $(REPORTS_DIR)/test_report.md for summary$(NC)"

.PHONY: coverage-report
coverage-report: test-coverage ## Generate and open coverage report
	@echo "$(BLUE)Opening coverage report...$(NC)"
	@if command -v open >/dev/null 2>&1; then \
		open $(COVERAGE_DIR)/index.html; \
	elif command -v xdg-open >/dev/null 2>&1; then \
		xdg-open $(COVERAGE_DIR)/index.html; \
	else \
		echo "$(YELLOW)Coverage report available at $(COVERAGE_DIR)/index.html$(NC)"; \
	fi

# CI/CD Simulation
.PHONY: ci
ci: check test ## Simulate CI pipeline
	@echo "$(BLUE)Running CI pipeline simulation...$(NC)"
	@echo "$(GREEN)CI pipeline completed successfully!$(NC)"

.PHONY: pre-commit
pre-commit: format lint test-fast ## Run pre-commit checks
	@echo "$(BLUE)Running pre-commit checks...$(NC)"
	@echo "$(GREEN)Pre-commit checks passed!$(NC)"

# Docker Commands (if using Docker)
.PHONY: docker-build
docker-build: ## Build Docker image
	@echo "$(BLUE)Building Docker image...$(NC)"
	docker build -t $(PROJECT_NAME) .
	@echo "$(GREEN)Docker image built!$(NC)"

.PHONY: docker-test
docker-test: ## Run tests in Docker container
	@echo "$(BLUE)Running tests in Docker...$(NC)"
	docker run --rm -v $(PWD):/app $(PROJECT_NAME) make test
	@echo "$(GREEN)Docker tests completed!$(NC)"

# Monitoring and Profiling
.PHONY: profile
profile: ## Profile the application
	@echo "$(BLUE)Profiling application...$(NC)"
	$(PYTHON) -m cProfile -o profile_output.prof -m app.main &
	sleep 5
	kill %%
	$(PYTHON) -c "import pstats; p = pstats.Stats('profile_output.prof'); p.sort_stats('cumulative'); p.print_stats(20)"
	@echo "$(GREEN)Profiling complete!$(NC)"

.PHONY: benchmark
benchmark: ## Run performance benchmarks
	@echo "$(BLUE)Running benchmarks...$(NC)"
	$(TEST_RUNNER) --category performance --load-level medium
	@echo "$(GREEN)Benchmarks completed!$(NC)"

# Utility Commands
.PHONY: deps-check
deps-check: ## Check for dependency updates
	@echo "$(BLUE)Checking for dependency updates...$(NC)"
	$(PIP) list --outdated
	@echo "$(GREEN)Dependency check complete!$(NC)"

.PHONY: deps-update
deps-update: ## Update dependencies
	@echo "$(BLUE)Updating dependencies...$(NC)"
	$(PIP) install --upgrade -r requirements.txt
	$(PIP) install --upgrade -r requirements-test.txt
	@echo "$(GREEN)Dependencies updated!$(NC)"

.PHONY: version
version: ## Show version information
	@echo "$(BLUE)Version Information:$(NC)"
	@echo "Python: $$($(PYTHON) --version)"
	@echo "Pip: $$($(PIP) --version)"
	@echo "Pytest: $$($(PYTEST) --version)"
	@echo "Project: $(PROJECT_NAME)"

# Status Commands
.PHONY: status
status: ## Show project status
	@echo "$(BLUE)Project Status:$(NC)"
	@echo "$(YELLOW)Test Reports:$(NC)"
	@ls -la $(REPORTS_DIR)/ 2>/dev/null || echo "  No reports found"
	@echo "$(YELLOW)Recent Logs:$(NC)"
	@ls -la logs/ 2>/dev/null || echo "  No logs found"
	@echo "$(YELLOW)Git Status:$(NC)"
	@git status --porcelain 2>/dev/null || echo "  Not a git repository"

# Quick Commands
.PHONY: quick-test
quick-test: ## Quick test run (unit tests only)
	@echo "$(BLUE)Running quick tests...$(NC)"
	$(PYTEST) tests/unit/ -x -v
	@echo "$(GREEN)Quick tests completed!$(NC)"

.PHONY: watch-tests
watch-tests: ## Watch for file changes and run tests
	@echo "$(BLUE)Watching for changes...$(NC)"
	@echo "$(YELLOW)Press Ctrl+C to stop$(NC)"
	@while true; do \
		inotifywait -r -e modify app/ tests/ 2>/dev/null && \
		echo "$(YELLOW)Files changed, running tests...$(NC)" && \
		make quick-test; \
	done

# Special targets
.PHONY: all
all: setup check test report ## Run complete setup, checks, tests, and reporting
	@echo "$(GREEN)Complete workflow finished!$(NC)"

# Make sure intermediate files are not deleted
.PRECIOUS: $(REPORTS_DIR)/%.html $(REPORTS_DIR)/%.xml

# Declare all targets as phony to avoid conflicts with files
.PHONY: install install-dev setup format lint security-check check test test-unit test-integration test-e2e test-performance test-security test-fast test-parallel test-coverage test-perf-small test-perf-medium test-perf-large test-smoke test-regression test-api run run-prod run-debug db-init db-reset clean clean-logs clean-uploads clean-all docs report coverage-report ci pre-commit docker-build docker-test profile benchmark deps-check deps-update version status quick-test watch-tests all