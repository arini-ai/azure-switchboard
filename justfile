set fallback

default:
    @just --list

bootstrap:
  uv sync --frozen
  grep -q "AZURE_OPENAI_ENDPOINT" .env || echo "please set AZURE_OPENAI_ENDPOINT in .env"
  grep -q "AZURE_OPENAI_API_KEY" .env || echo "please set AZURE_OPENAI_API_KEY in .env"


test:
  uv run pytest

run:
  uv run --env-file .env src/switchboard/switchboard.py
