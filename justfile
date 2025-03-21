set fallback

install:
  uv sync --frozen
  # uv run pre-commit install
  trunk actions enable trunk-fmt-pre-commit
  trunk actions enable trunk-fmt-pre-push

update:
  uv sync --upgrade
  uv run pre-commit autoupdate

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

run *what:
  uv run --env-file .env {{ what }}

bench *args='-v -r 1000 -d 10 -e 500':
  just run tools/bench.py {{args}}

demo:
  @grep -q "AZURE_OPENAI_ENDPOINT" .env || echo "please set AZURE_OPENAI_ENDPOINT in .env"
  @grep -q "AZURE_OPENAI_API_KEY" .env || echo "please set AZURE_OPENAI_API_KEY in .env"
  just run tools/api_demo.py

otel-collector:
  docker run --rm -p 4317:4317 -p 4318:4318 \
    -v $(pwd)/otel-collector-config.yaml:/etc/otel-collector-config.yaml \
    otel/opentelemetry-collector:latest \
    --config=/etc/otel-collector-config.yaml

otel-viewer:
  # brew tap CtrlSpice/homebrew-otel-desktop-viewer
  # brew install otel-desktop-viewer
  otel-desktop-viewer --browser 8001

otel-run *cmd='tools/bench.py -r 5 -d 3':
  #!/usr/bin/env zsh
  # export OTEL_EXPORTER_OTLP_ENDPOINT="http://localhost:4318"
  export OTEL_PYTHON_LOGGING_AUTO_INSTRUMENTATION_ENABLED=true
  # export OTEL_SERVICE_NAME=switchboard
  # export OTEL_TRACES_EXPORTER=console,otlp
  # export OTEL_METRICS_EXPORTER=console,otlp
  export OTEL_EXPORTER_OTLP_PROTOCOL="http/protobuf"
  just run opentelemetry-instrument python {{ cmd }}

trunk-check:
  trunk check --all --fix

trunk-fmt:
  trunk fmt
