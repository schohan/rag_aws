.PHONY: install dev test lint format clean server chat setup deploy-infra synth-infra diff-infra destroy-infra destroy-bootstrap teardown-all clean-cdk docker-build docker-run help

# Development
install:
	pip install -e .

dev:
	pip install -e ".[dev]"

# Testing
test:
	pytest tests/ -v

test-cov:
	pytest tests/ -v --cov=src/rag_agent --cov-report=html --cov-report=term

# Code Quality
lint:
	ruff check src/ tests/
	mypy src/rag_agent

format:
	black src/ tests/
	ruff check --fix src/ tests/

# Application
server:
	rag-agent server --reload

chat:
	rag-agent chat

setup:
	rag-agent setup

# Infrastructure
deploy-infra:
	cd infrastructure && cdk deploy --all

synth-infra:
	cd infrastructure && cdk synth

diff-infra:
	cd infrastructure && cdk diff

# Teardown - Destroy deployed stacks
destroy-infra:
	cd infrastructure && cdk destroy --all --force

# Teardown - Destroy bootstrap stack (removes CDK toolkit from AWS account)
destroy-bootstrap:
	@echo "⚠️  This will remove the CDK bootstrap stack from your AWS account."
	@echo "    You will need to run 'cdk bootstrap' again before deploying."
	@read -p "Are you sure? [y/N] " confirm && [ "$$confirm" = "y" ] || exit 1
	aws cloudformation delete-stack --stack-name CDKToolkit
	@echo "Waiting for CDKToolkit stack deletion..."
	aws cloudformation wait stack-delete-complete --stack-name CDKToolkit || true
	@echo "✅ CDK bootstrap stack deleted."

# Teardown - Full cleanup (stacks + bootstrap + local artifacts)
teardown-all:
	@echo "🧹 Full teardown: destroying all infrastructure..."
	cd infrastructure && cdk destroy --all --force || true
	@echo ""
	@echo "⚠️  Do you also want to remove the CDK bootstrap stack?"
	@read -p "Remove CDKToolkit stack? [y/N] " confirm && \
		if [ "$$confirm" = "y" ]; then \
			aws cloudformation delete-stack --stack-name CDKToolkit && \
			echo "Waiting for CDKToolkit stack deletion..." && \
			aws cloudformation wait stack-delete-complete --stack-name CDKToolkit || true && \
			echo "✅ CDK bootstrap stack deleted."; \
		fi
	rm -rf infrastructure/cdk.out
	@echo "✅ Teardown complete."

# Clean CDK output artifacts
clean-cdk:
	rm -rf infrastructure/cdk.out

# Cleanup
clean:
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info
	rm -rf .pytest_cache
	rm -rf .mypy_cache
	rm -rf .ruff_cache
	rm -rf htmlcov/
	rm -rf .coverage
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete

# Docker (optional)
docker-build:
	docker build -t rag-agent .

docker-run:
	docker run -p 8000:8000 --env-file .env rag-agent

# Help
help:
	@echo "Available targets:"
	@echo ""
	@echo "Development:"
	@echo "  install          - Install the package"
	@echo "  dev              - Install with dev dependencies"
	@echo "  test             - Run tests"
	@echo "  test-cov         - Run tests with coverage"
	@echo "  lint             - Run linters"
	@echo "  format           - Format code"
	@echo ""
	@echo "Application:"
	@echo "  server           - Start development server"
	@echo "  chat             - Start interactive chat"
	@echo "  setup            - Set up AWS infrastructure"
	@echo ""
	@echo "Infrastructure:"
	@echo "  deploy-infra     - Deploy CDK infrastructure"
	@echo "  synth-infra      - Synthesize CloudFormation templates"
	@echo "  diff-infra       - Show infrastructure changes"
	@echo "  destroy-infra    - Destroy deployed CDK stacks"
	@echo "  destroy-bootstrap- Remove CDK bootstrap stack from AWS"
	@echo "  teardown-all     - Full cleanup (stacks + optional bootstrap)"
	@echo "  clean-cdk        - Remove local CDK output artifacts"
	@echo ""
	@echo "Cleanup:"
	@echo "  clean            - Clean build artifacts"

