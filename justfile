set fallback

install:
  uv sync --frozen
  uv run pre-commit install

test *args='-n 4':
  uv run pytest {{args}}
alias tests := test

lint *args='--fix':
  uv run ruff check . {{args}}

bump-version *args='':
  uv run bumpver update {{args}}

clean:
  find . -name '*.pyc' -delete
  rm -rf .pytest_cache .ruff_cache dist

run what:
  uv run --env-file .env tools/{{ what }}

bench *args='-v -r 1000 -d 10 -e 500':
  uv run --env-file .env tools/bench.py {{args}}

demo:
  @grep -q "AZURE_OPENAI_ENDPOINT" .env || echo "please set AZURE_OPENAI_ENDPOINT in .env"
  @grep -q "AZURE_OPENAI_API_KEY" .env || echo "please set AZURE_OPENAI_API_KEY in .env"
  just run api_demo.py

otel:
  OTEL_PYTHON_LOGGING_AUTO_INSTRUMENTATION_ENABLED=true \
  OTEL_SERVICE_NAME=switchboard_readme \
  OTEL_TRACES_EXPORTER=console,otlp \
  OTEL_METRICS_EXPORTER=console,otlp \
  OTEL_EXPORTER_OTLP_ENDPOINT="http://localhost:4318" \
  OTEL_EXPORTER_OTLP_PROTOCOL="http/protobuf" \
  uv run --env-file .env opentelemetry-instrument python tools/readme_example.py
