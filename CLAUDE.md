# CLAUDE.md - Azure Switchboard Development Guide

## Build & Test Commands
- Bootstrap: `just bootstrap` (installs dependencies with uv)
- Run tests: `just test` or `uv run pytest -s -v`
- Run single test: `uv run pytest tests/test_file.py::test_function_name -v`
- Lint: `uv run ruff check .`
- Format: `uv run ruff format .`
- Demo: `uv run --env-file .env tools/api_demo.py`

## Code Style Guidelines
- **Imports**: Standard modules first, third-party libraries next, local modules last
- **Type Annotations**: Use typing module extensively with generics, Annotated, overload
- **Formatting**: Clean 4-space indentation, docstrings in triple quotes
- **Naming**: snake_case for variables/functions, PascalCase for classes
- **Error Handling**: Try/except with specific exceptions, use cooldown mechanism for API errors
- **Classes**: Use pydantic BaseModel for configuration classes
- **Async**: Project uses async/await patterns extensively
- **Testing**: Comprehensive unit tests with fixtures and mocks for API calls
- **Principles**: Prefer simple implementations over complex ones, avoid premature optimization.
  Use preparatory refactoring if necessary to make a problem simpler, if that is the appropriate approach.
