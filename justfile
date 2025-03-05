set fallback

bootstrap:
  uv sync --frozen
  grep -q "AZURE_OPENAI_ENDPOINT" .env || echo "please set AZURE_OPENAI_ENDPOINT in .env"
  grep -q "AZURE_OPENAI_API_KEY" .env || echo "please set AZURE_OPENAI_API_KEY in .env"

install-ci:
  uv sync --frozen --dev

demo:
  uv run --env-file .env tools/api_demo.py

test *args='':
  uv run pytest -s -v {{args}}

lint:
  uv run ruff check .

bump-version *args='':
  uv run bumpver update {{args}}

build *args='':
  uv build {{args}}

publish *args='':
  uv publish {{args}}
