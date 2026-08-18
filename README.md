# Azure Switchboard

Batteries-included, coordination-free client loadbalancing for Azure AI Foundry — OpenAI and Anthropic models alike.

```bash
uv add azure-switchboard
```

[![PyPI - Version](https://img.shields.io/pypi/v/azure-switchboard)](https://pypi.org/project/azure-switchboard/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![CI](https://github.com/arini-ai/azure-switchboard/actions/workflows/ci.yaml/badge.svg?branch=master)](https://github.com/arini-ai/azure-switchboard/actions/workflows/ci.yaml)

## Overview

`azure-switchboard` is a Python 3 library that implements a client-only, coordination-free loadbalancer for OpenAI/Anthropic models hosted on Azure Foundry. It can be used as a drop-in replacement for the openai or anthropic SDKs and lets you spread inference traffic across model deployments in multiple Azure Foundry resources. Coordination-freedom is achieved via the [power of two random choices](https://www.eecs.harvard.edu/~michaelm/postscripts/handbook2001.pdf) algorithm. See below for performance benchmarks.

List the Azure resources you have and the models deployed on them:

```python
from azure_switchboard import AnthropicDeployment, Foundry, OpenAIDeployment, Switchboard

sb = Switchboard(resources=[
    Foundry(
        name="east",
        api_key=...,
        models=[
            OpenAIDeployment("gpt-4o-mini", tpm=30000, rpm=300),
            AnthropicDeployment("claude-sonnet-5", tpm=30000, rpm=300),
        ],
    ),
    Foundry(
        name="west",
        api_key=...,
        models=[OpenAIDeployment("gpt-4o-mini", tpm=30000, rpm=300)],
    ),
])

async with sb:
    completion = await sb.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": "Hello!"}],
    )
    message = await sb.messages.create(
        model="claude-sonnet-5",
        max_tokens=1024,
        messages=[{"role": "user", "content": "Hello!"}],
    )
```

The endpoint is inferrable from the Foundry resource name or can be overriden explicitly on a per-deployment basis. Usage, ratelimiting, and cooldown status is tracked per-deployment (tpm/rpm) and per-resource (network-level errors) and overloaded models or resources can be configured to fall back to first-party providers if the appropriate API keys are available in the environment.

## Features

- **Multi-Provider**: supports both the OpenAI Chat Completions API and the Anthropic Messages API. Non-OpenAI models that support the OpenAI API spec can be used via the OpenAIDeployment class.
- **Coordination-Free**: The default Two Random Choices algorithm does not require coordination between client instances to achieve excellent load distribution characteristics. See benchmarks for additional details.
- **Utilization-Aware**: TPM/RPM utilization is tracked per deployment for use during selection.
- **Batteries Included**:
  - **Session Affinity**: Provide a `session_id` to route requests in the same session to the same resource, so a session spanning several models keeps one prompt cache warm.
  - **Automatic Failover**: Retries are controlled by a tenacity `AsyncRetrying` policy (`failover_policy`).
  - **First-Party Fallback**: Set `openai_fallback=True` / `anthropic_fallback=True` to back the pool with the vendors' own APIs once nothing healthy is left.
  - **Pluggable Selection**: Custom selection algorithms can be provided by passing a callable to the `selector` parameter on the Switchboard constructor.
  - **OpenTelemetry Integration**: Built-in metrics for request routing and healthy deployment counts.
- **Lightweight**: Small codebase with minimal dependencies: `openai`, `anthropic`, `loguru`, `tenacity`, `wrapt`, and `opentelemetry-api`.

## Runnable Example

See [tools/readme_example.py](https://github.com/arini-ai/azure-switchboard/blob/master/tools/readme_example.py).

## Benchmarks

1000 requests spread over 10 deployments of `gpt-5.4-mini`:

```bash
just bench
uv run tools/bench.py -v -r 1000 -d 10 -e 500
Distributing 1000 requests across 10 deployments
Max inflight requests: 1000

{
    'bench_0': {'gpt-5.4-mini': UtilStats(util=0.337, tpm='8388/30000', rpm='100/300')},
    'bench_1': {'gpt-5.4-mini': UtilStats(util=0.331, tpm='8251/30000', rpm='99/300')},
    ...
    'bench_9': {'gpt-5.4-mini': UtilStats(util=0.342, tpm='8435/30000', rpm='100/300')}
}

Utilization Distribution:
0.000 - 0.100 |   0
0.100 - 0.200 |   0
0.200 - 0.300 |   0
0.300 - 0.400 |  10 ..............................
0.400 - 0.500 |   0
0.500 - 0.600 |   0
0.600 - 0.700 |   0
0.700 - 0.800 |   0
0.800 - 0.900 |   0
0.900 - 1.000 |   0
Avg utilization: 0.338 (0.325 - 0.345)
Std deviation: 0.005

Distribution overhead: 242.59ms
Average response latency: 5066.64ms
Total latency: 10041.57ms
Requests per second: 4122.20
Overhead per request: 0.24ms
```

Distribution overhead scales ~linearly with the number of deployments.

## Development

This project uses [uv](https://github.com/astral-sh/uv) for package management,
and [just](https://github.com/casey/just) for task automation. See the [justfile](https://github.com/arini-ai/azure-switchboard/blob/master/justfile) for available commands.

```bash
git clone https://github.com/arini-ai/azure-switchboard
cd azure-switchboard

just install
```

### Running tests

```bash
just test        # unit tests; every upstream is mocked
just typecheck   # pyright over src/, as CI runs it
```

### Release

This library uses CalVer for versioning. On push to master, if tests pass, a package is automatically built, released, and uploaded to PyPI.

Locally, the package can be built with uv:

```bash
uv build
```

## Contributing

1. Fork/clone repo
2. Make changes
3. Run tests with `just test`
4. Lint with `just lint`
5. Commit and make a PR

## License

MIT
