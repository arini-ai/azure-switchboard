set fallback

install:
  uv sync --frozen

test *args='--cov=azure_switchboard --cov-report=term-missing:skip-covered':
  uv run pytest {{args}}

lint *args='':
  uv run ruff check . {{args}}

bump-version *args='':
  uv run bumpver update {{args}}

clean:
  find . -name '*.pyc' -delete
  rm -rf .pytest_cache .ruff_cache dist

run what:
  uv run --env-file .env tools/{{ what }}

demo:
  @grep -q "AZURE_OPENAI_ENDPOINT" .env || echo "please set AZURE_OPENAI_ENDPOINT in .env"
  @grep -q "AZURE_OPENAI_API_KEY" .env || echo "please set AZURE_OPENAI_API_KEY in .env"
  just run api_demo.py
