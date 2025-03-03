set fallback

default:
    @just --list

bootstrap:
  uv sync --frozen
  grep -q "AZURE_OPENAI_ENDPOINT" .env || echo "please set AZURE_OPENAI_ENDPOINT in .env"
  grep -q "AZURE_OPENAI_API_KEY" .env || echo "please set AZURE_OPENAI_API_KEY in .env"


test:
  uv run pytest -s -v test/test_switchboard_v3.py

run:
  uv run --env-file .env api_demo.py
