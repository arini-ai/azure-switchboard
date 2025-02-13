set fallback

default:
    @just --list

bootstrap:
  uv sync --frozen

test:
  uv run pytest

run:
  uv run --env-file .env src/switchboard/switchboard.py
