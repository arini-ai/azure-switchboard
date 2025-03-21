# CLAUDE.md - Azure Switchboard Development Guide

## Project Overview

- **Name**: Azure Switchboard
- **Description**: Batteries-included, coordination-free client loadbalancing for Azure OpenAI
- **Version**: 2025.3.24 (uses CalVer versioning)
- **License**: MIT
- **Repository**: [azure-switchboard](https://github.com/arini-ai/azure-switchboard)

## Build & Test Commands

- Install dependencies: `just install` (uses uv)
- Update dependencies: `just update`
- Run tests: `just test` or `uv run pytest -s -v` (supports xdist with `-n 4` by default)
- Run single test: `uv run pytest tests/test_file.py::test_function_name -v`
- Lint: `just lint` or `uv run ruff check . --fix`
- Format: `uv run ruff format .`
- Demo: `just demo` or `uv run --env-file .env tools/api_demo.py`
- Benchmark: `just bench` or `uv run --env-file .env tools/bench.py -v -r 1000 -d 10 -e 500`
- OpenTelemetry demo: `just otel`
- Bump version: `just bump-version`
- Pre-commit hooks: `just pre-commit`
- Clean: `just clean`

## Code Style Guidelines

- **Imports**: Standard modules first, third-party libraries next, local modules last
- **Type Annotations**: Use typing module extensively with generics, Annotated, overload
- **Formatting**: Clean 4-space indentation, docstrings in triple quotes
- **Naming**: snake_case for variables/functions, PascalCase for classes
- **Error Handling**: Try/except with specific exceptions, use cooldown mechanism for API errors
- **Classes**: Use pydantic BaseModel for configuration classes
- **Async**: Project uses async/await patterns extensively
- **Testing**: Comprehensive unit tests with fixtures and mocks for API calls
- **Principles**: Prefer simple implementations over complex ones, avoid premature optimization
- **Pre-commit Hooks**: ruff for linting/formatting, actionlint for GitHub Actions

## Dependencies

### Runtime

- openai>=1.62.0
- opentelemetry-api>=1.30.0
- tenacity>=9.0.0
- wrapt>=1.17.2

### Development

- uv for package management
- just for task automation
- pytest with asyncio, coverage, xdist for testing
- ruff for linting and formatting
- pre-commit for git hooks
- bumpver for version management
- OpenTelemetry instrumentation for observability

## Project Structure

- `src/azure_switchboard/`: Core implementation
  - `switchboard.py`: Main client implementation with load balancing logic
  - `deployment.py`: Deployment configuration and API client management
- `tests/`: Comprehensive test suite
- `tools/`: Demo and benchmark utilities

## Key Features

- API-compatible drop-in replacement for OpenAI's ChatCompletion API
- Coordination-free load balancing with "power of two random choices" algorithm
- TPM/RPM rate limit tracking per model/deployment
- Session affinity for efficient prompt caching
- Automatic failover with customizable retry policies
- OpenTelemetry integration for monitoring
- Lightweight implementation (<400 LOC) with minimal dependencies
