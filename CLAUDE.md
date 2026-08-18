# CLAUDE.md - Azure Switchboard Development Guide

## Project Overview

- **Name**: Azure Switchboard
- **Description**: Batteries-included, coordination-free client loadbalancing for Azure OpenAI
- **Versioning**: uses CalVer versioning
- **License**: MIT
- **Repository**: [azure-switchboard](https://github.com/arini-ai/azure-switchboard)

## Build & Test Commands

- Install dependencies: `just install` (uses uv)
- Update dependencies: `just update`
- Run tests: `just test` or `uv run pytest -s -v` (supports xdist with `-n 4` by default)
- Run single test: `uv run pytest tests/test_file.py::test_function_name -v`
- Lint: `just lint` or `uv run ruff check . --fix`
- Format: `uv run ruff format .`
- Typecheck: `just typecheck` or `uv run pyright` (scoped to `src/`; runs in CI)
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

- openai>=2.0.0
- loguru>=0.7.3
- opentelemetry-api>=1.30.0
- tenacity>=9.0.0
- wrapt>=1.17.2
- anthropic>=0.122.0

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
  - `foundry.py`: `Foundry`, an Azure resource: endpoint, credential, clients, deployments
  - `model.py`: `ModelBase`, the provider-agnostic deployment: quota, utilization, cooldown
  - `openai_deployment.py`: `OpenAIDeployment`, a deployment speaking Chat Completions
  - `anthropic_deployment.py`: `AnthropicDeployment`, a deployment speaking the Messages API
- `tests/`: Comprehensive test suite
- `tools/`: Demo and benchmark utilities

## Key Features

- API-compatible drop-in for both SDKs, each at its own call path: `sb.chat.completions.*` and `sb.messages.*`
- A Foundry resource hosts deployments of either API; deployments are pooled per API, so the calling surface resolves a model name
- First-party OpenAI/Anthropic fallback once a pool has nothing healthy left (`openai_fallback` / `anthropic_fallback`)
- Coordination-free load balancing with "power of two random choices" algorithm
- TPM/RPM rate limit tracking per deployment
- Cooldowns scoped by error: 429 cools one deployment, connection errors cool the whole resource, timeouts cool nothing
- Session affinity to a resource, for efficient prompt caching
- Automatic failover with customizable retry policies
- OpenTelemetry integration for monitoring
- Lightweight implementation (~1k LOC) with minimal dependencies
