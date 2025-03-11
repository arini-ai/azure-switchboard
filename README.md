# Azure Switchboard

Batteries-included, coordination-free client loadbalancing for Azure OpenAI.

```bash
pip install azure-switchboard
```

[![PyPI version](https://badge.fury.io/py/azure-switchboard.svg)](https://badge.fury.io/py/azure-switchboard)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Overview

`azure-switchboard` is a asyncio-only Python 3 library that provides an intelligent client loadbalancer for Azure OpenAI. You instantiate the Switchboard client with a set of Azure deployments, and the client distributes your chat completion requests across the provided deployments using the [power of two random choices](https://www.eecs.harvard.edu/~michaelm/postscripts/handbook2001.pdf) method based on utilization. In this sense, it serves as a lightweight service mesh between your application and Azure OpenAI. The basic idea is inspired by [ServiceRouter](https://www.usenix.org/system/files/osdi23-saokar.pdf).

## Features

- **API Compatibility**: `Switchboard.create` is a fully-typed, transparent drop-in replacement for `OpenAI.chat.completions.create`
- **Coordination-Free**: Pick-2 algorithm does not require coordination between client instances to achieve excellent load distribution characteristics
- **Utilization-Aware**: Load is distributed by TPM/RPM utilization per model per deployment
- **Batteries Included**:
    - **Session Affinity**: Provide a `session_id` to route requests in the same session to the same deployment, optimizing for prompt caching
    - **Automatic Failover**: Client retries automatically up to `retries` times on deployment failure, with fallback to OpenAI configurable through the `fallback` parameter
- **Lightweight**: Only three runtime dependencies: `openai`, `tenacity`, `wrapt`
- **100% Test Coverage**: Implementation is fully unit tested.

## Basic Usage

See `tools/readme_example.py` for a runnable example.

```python
import asyncio
import os
from contextlib import asynccontextmanager

from azure_switchboard import DeploymentConfig, Model, Switchboard

# use demo parameters from environment if available
api_base = os.getenv("AZURE_OPENAI_ENDPOINT", "https://test.openai.azure.com/")
api_key = os.getenv("AZURE_OPENAI_API_KEY", "your-api-key")
openai_api_key = os.getenv("OPENAI_API_KEY", "your-openai-api-key")

# define deployments
deployments = []
for name in ("east", "west", "south"):
    deployments.append(
        DeploymentConfig(
            name=name,
            api_base=api_base,  # can reuse since the implementation doesn't know
            api_key=api_key,
            models=[Model(name="gpt-4o-mini", tpm=10000, rpm=60)],
        )
    )

fallback = DeploymentConfig(
    name="openai",
    api_base="",  # gets populated by AsyncOpenAI automatically
    api_key=openai_api_key,
    models=[Model(name="gpt-4o-mini", tpm=10000, rpm=60)],
)


@asynccontextmanager
async def get_switchboard():
    """Use a pattern analogous to FastAPI dependency
    injection for automatic cleanup.
    """

    # Create Switchboard and start background tasks
    switchboard = Switchboard(deployments=deployments, fallback=fallback)
    switchboard.start()

    try:
        yield switchboard
    finally:
        await switchboard.stop()


async def basic_functionality(switchboard: Switchboard):
    # Make a completion request (non-streaming)
    response = await switchboard.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": "Hello, world!"}],
    )

    print("completion:", response.choices[0].message.content)

    # Make a streaming completion request
    stream = await switchboard.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": "Hello, world!"}],
        stream=True,
    )

    print("streaming: ", end="")
    async for chunk in stream:
        if chunk.choices and chunk.choices[0].delta.content:
            print(chunk.choices[0].delta.content, end="", flush=True)

    print()


async def session_affinity(switchboard: Switchboard):
    session_id = "anything"

    # First message will select a random healthy
    # deployment and associate it with the session_id
    _ = await switchboard.create(
        session_id=session_id,
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": "Who won the World Series in 2020?"}],
    )

    # Follow-up requests with the same session_id will route to the same deployment
    _ = await switchboard.create(
        session_id=session_id,
        model="gpt-4o-mini",
        messages=[
            {"role": "user", "content": "Who won the World Series in 2020?"},
            {
                "role": "assistant",
                "content": "The Los Angeles Dodgers won the World Series in 2020.",
            },
            {"role": "user", "content": "Who did they beat?"},
        ],
    )

    # If the deployment becomes unhealthy,
    # requests will fall back to a healthy one

    # Simulate a failure by marking down the deployment
    original_client = switchboard.select_deployment(
        model="gpt-4o-mini", session_id=session_id
    )
    print("original client:", original_client)
    original_client.models["gpt-4o-mini"].cooldown()

    # A new deployment will be selected for this session_id
    _ = await switchboard.create(
        session_id=session_id,
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": "Who won the World Series in 2021?"}],
    )

    new_client = switchboard.select_deployment(
        model="gpt-4o-mini", session_id=session_id
    )
    print("new client:", new_client)
    assert new_client != original_client


async def main():
    async with get_switchboard() as sb:
        print("Basic functionality:")
        await basic_functionality(sb)

        print("Session affinity (should warn):")
        await session_affinity(sb)


if __name__ == "__main__":
    asyncio.run(main())
```

## Configuration Options

### switchboard.Model Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `name` | Configured model name, e.g. "gpt-4o" or "gpt-4o-mini" | Required |
| `tpm` | Tokens per minute rate limit | 0 (unlimited) |
| `rpm` | Requests per minute rate limit | 0 (unlimited) |
| `default_cooldown` | Default cooldown period in seconds | 10.0 |

### switchboard.DeploymentConfig Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `name` | Unique identifier for the deployment | Required |
| `api_base` | Azure OpenAI endpoint URL | Required |
| `api_key` | Azure OpenAI API key | Required |
| `api_version` | Azure OpenAI API version | "2024-10-21" |
| `timeout` | Default timeout in seconds | 600.0 |
| `models` | List of Models configured for this deployment | Required |

### switchboard.Switchboard Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `deployments` | List of DeploymentConfig objects | Required |
| `ratelimit_window` | Seconds before resetting usage counters | 60 |
| `fallback` | Fallback DeploymentConfig | None |
| `retries` | Number of retries on deployment failure | 2 |
| `deployment_factory` | Factory for creating Deployments | azure_client_factory |

## Development

This project uses [uv](https://github.com/astral-sh/uv) for package management,
and [just](https://github.com/casey/just) for task automation. See the [justfile](https://github.com/abizer/switchboard/blob/master/justfile)
for available commands.

```bash
git clone https://github.com/arini-ai/azure-switchboard
cd azure-switchboard

just install
```

### Running tests

```bash
just test
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
4. Lint with `just lint --fix`
5. Commit and make a PR

# TODO

* client inherit from azure client?
* opentelemetry integration
* lru list for usage tracking / better ratelimit handling

## License

MIT
