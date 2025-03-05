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

- **API Compatibility**: `Switchboard.create` is a transparent proxy for `OpenAI.chat.completions.create` and can be used as a drop-in replacement
- **Coordination-Free**: Pick-2 algorithm does not require coordination between client instances to achieve good load distribution characteristics
- **Utilization-Aware**: Tracks TPM/RPM usage per deployment for utilization-based loadbalancing
- **Batteries Included**:
    - **Session Affinity**: Provide a `session_id` to route requests in the same session to the same deployment, optimizing for prompt caching
    - **Automatic Failover**: Internally monitors deployment health and manages retries to fallback deployments automatically
- **Lightweight**: Only three runtime dependencies: `openai`, `tenacity`, `wrapt`

## Basic Usage

See `tools/readme_example.py` for a runnable example.

```python
import asyncio
from contextlib import asynccontextmanager
from azure_switchboard import Switchboard, Deployment

from uuid import uuid4

# Define deployments
deployments = [
    Deployment(
        name="d1",
        api_base="https://your-deployment1.openai.azure.com/",
        api_key="your-api-key",
        # optionally specify ratelimits
        rpm_ratelimit=60,
        tpm_ratelimit=100000,
    ),
    Deployment(
        name="d2",
        api_base="https://your-deployment2.openai.azure.com/",
        api_key="your-api-key2",
        rpm_ratelimit=60,
        tpm_ratelimit=100000,
    ),
    Deployment(
        name="d3",
        api_base="https://your-deployment3.openai.azure.com/",
        api_key="your-api-key3",
        rpm_ratelimit=60,
        tpm_ratelimit=100000,
    ),
]

@asynccontextmanager
async def init_switchboard():
    """Wrap client initialization in a context manager for automatic cleanup.

    Analogous to FastAPI dependency injection.
    """

    try:
        # Create Switchboard with deployments
        switchboard = Switchboard(deployments)

        # Start background tasks
        # (healthchecks, ratelimiting)
        switchboard.start()

        yield switchboard
    finally:
        await switchboard.stop()

async def basic_functionality(switchboard: Switchboard):
    # Make a completion request (non-streaming)
    response = await switchboard.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": "Hello, world!"}]
        )

    print(response.choices[0].message.content)

    # Make a streaming completion request
    stream = await switchboard.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": "Hello, world!"}],
        stream=True
    )

    async for chunk in stream:
        if chunk.choices and chunk.choices[0].delta.content:
            print(chunk.choices[0].delta.content, end="", flush=True)

    print()

async def session_affinity(switchboard: Switchboard):
    session_id = str(uuid4())

    # First message will select a random healthy
    # deployment and associate it with the session_id
    response = await switchboard.create(
        session_id=session_id,
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": "Who won the World Series in 2020?"}]
    )

    # Follow-up requests with the same session_id will route to the same deployment
    response = await switchboard.create(
        session_id=session_id,
        model="gpt-4o-mini",
        messages=[
            {"role": "user", "content": "Who won the World Series in 2020?"},
            {"role": "assistant", "content": "The Los Angeles Dodgers won the World Series in 2020."},
            {"role": "user", "content": "Who did they beat?"}
        ]
    )

    # If the deployment becomes unhealthy,
    # requests will fall back to a healthy one

    # Simulate a failure by marking down the deployment
    original_client = switchboard.select_deployment(session_id)
    original_client.cooldown()

    # A new deployment will be selected for this session_id
    response = await switchboard.create(
        session_id=session_id,
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": "Who won the World Series in 2021?"}]
    )

    new_client = switchboard.select_deployment(session_id)
    assert new_client != original_client

async def main():
    async with init_switchboard() as sb:
        print("Basic functionality:")
        await basic_functionality(sb)

        print("Session affinity:")
        await session_affinity(sb)

if __name__ == "__main__":
    asyncio.run(main())
```

## Configuration Options

### switchboard.Deployment Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `name` | Unique identifier for the deployment | Required |
| `api_base` | Azure OpenAI endpoint URL | Required |
| `api_key` | Azure OpenAI API key | Required |
| `api_version` | Azure OpenAI API version | "2024-10-21" |
| `timeout` | Default timeout in seconds | 600.0 |
| `tpm_ratelimit` | Tokens per minute rate limit | 0 (unlimited) |
| `rpm_ratelimit` | Requests per minute rate limit | 0 (unlimited) |
| `healthcheck_interval` | Seconds between health checks | 30 |
| `cooldown_period` | Seconds to wait after an error before retrying | 60 |

### switchboard.Switchboard Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `deployments` | List of Deployment objects | Required |
| `client_factory` | Function to create clients from deployments | default_client_factory |
| `healthcheck_interval` | Seconds between health checks | 10 |
| `ratelimit_window` | Seconds before resetting usage counters | 60 |

## License

MIT

