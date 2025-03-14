# Azure Switchboard

Batteries-included, coordination-free client loadbalancing for Azure OpenAI.

```bash
pip install azure-switchboard
```

[![PyPI - Version](https://img.shields.io/pypi/v/azure-switchboard)](https://pypi.org/project/azure-switchboard/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Tests](https://github.com/arini-ai/azure-switchboard/actions/workflows/test.yml/badge.svg?branch=master)](https://github.com/arini-ai/azure-switchboard/actions/workflows/test.yml)

## Overview

`azure-switchboard` is a Python 3 asyncio library that provides an intelligent, API-compatible client loadbalancer for Azure OpenAI. You instantiate a Switchboard client with a set of deployments, and the client distributes your chat completion requests across the available deployments using the [power of two random choices](https://www.eecs.harvard.edu/~michaelm/postscripts/handbook2001.pdf) method. In this sense, it functions as a lightweight service mesh between your application and Azure OpenAI. The basic idea is inspired by [ServiceRouter](https://www.usenix.org/system/files/osdi23-saokar.pdf).

## Features

- **API Compatibility**: `Switchboard.create` is a transparently-typed drop-in proxy for `OpenAI.chat.completions.create`.
- **Coordination-Free**: The default Two Random Choices algorithm does not require coordination between client instances to achieve excellent load distribution characteristics.
- **Utilization-Aware**: TPM/RPM ratelimit utilization is tracked per model per deployment for use during selection.
- **Batteries Included**:
    - **Session Affinity**: Provide a `session_id` to route requests in the same session to the same deployment, optimizing for prompt caching
    - **Automatic Failover**: Client automatically retries 3 times on request failure, with optional fallback to OpenAI by providing an `OpenAIDeployment` in  `deployments`. The retry policy can also be customized by passing a tenacity
    `AsyncRetrying` instance to `failover_policy`.
    - **Pluggable Selection**: Custom selection algorithms can be
    provided by passing a callable to the `selector` parameter on the Switchboard constructor.

- **Lightweight**: sub-400 LOC implementation with only three runtime dependencies: `openai`, `tenacity`, `wrapt`. <1ms overhead per request.
- **100% Test Coverage**: There are twice as many lines in the tests as in the implementation.

## Runnable Example

```python
#!/usr/bin/env python3
#
# To run this, use:
#   uv run readme_example.py
#
# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "azure-switchboard",
# ]
# ///

import asyncio
import os

from azure_switchboard import AzureDeployment, Model, OpenAIDeployment, Switchboard

azure_openai_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
azure_openai_api_key = os.getenv("AZURE_OPENAI_API_KEY")
openai_api_key = os.getenv("OPENAI_API_KEY", None)

deployments = []
if azure_openai_endpoint and azure_openai_api_key:
    # create 3 deployments. reusing the endpoint
    # is fine for the purposes of this demo
    for name in ("east", "west", "south"):
        deployments.append(
            AzureDeployment(
                name=name,
                endpoint=azure_openai_endpoint,
                api_key=azure_openai_api_key,
                models=[Model(name="gpt-4o-mini")],
            )
        )

if openai_api_key:
    # we can use openai as a fallback deployment
    # it will pick up the api key from the environment
    deployments.append(OpenAIDeployment())


async def main():
    async with Switchboard(deployments=deployments) as sb:
        print("Basic functionality:")
        await basic_functionality(sb)

        print("Session affinity (should warn):")
        await session_affinity(sb)


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
    r = await switchboard.create(
        session_id=session_id,
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": "Who won the World Series in 2020?"}],
    )

    d1 = switchboard.select_deployment(model="gpt-4o-mini", session_id=session_id)
    print("deployment 1:", d1)
    print("response 1:", r.choices[0].message.content)

    # Follow-up requests with the same session_id will route to the same deployment
    r2 = await switchboard.create(
        session_id=session_id,
        model="gpt-4o-mini",
        messages=[
            {"role": "user", "content": "Who won the World Series in 2020?"},
            {"role": "assistant", "content": r.choices[0].message.content},
            {"role": "user", "content": "Who did they beat?"},
        ],
    )

    print("response 2:", r2.choices[0].message.content)

    # Simulate a failure by marking down the deployment
    d1.models["gpt-4o-mini"].cooldown()

    # A new deployment will be selected for this session_id
    r3 = await switchboard.create(
        session_id=session_id,
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": "Who won the World Series in 2021?"}],
    )

    d2 = switchboard.select_deployment(model="gpt-4o-mini", session_id=session_id)
    print("deployment 2:", d2)
    print("response 3:", r3.choices[0].message.content)
    assert d2 != d1


if __name__ == "__main__":
    asyncio.run(main())
```

## Performance

```bash
(azure-switchboard) .venv > uv run tools/bench.py -r 1000 -d 5
Distributing 1000 requests across 5 deployments
{
    'bench_0': {'gpt-4o-mini': {'util': 0.337, 'tpm': '20200/100000', 'rpm': '201/600'}},
    'bench_1': {'gpt-4o-mini': {'util': 0.341, 'tpm': '20412/100000', 'rpm': '201/600'}},
    'bench_2': {'gpt-4o-mini': {'util': 0.333, 'tpm': '20017/100000', 'rpm': '199/600'}},
    'bench_3': {'gpt-4o-mini': {'util': 0.336, 'tpm': '20462/100000', 'rpm': '200/600'}},
    'bench_4': {'gpt-4o-mini': {'util': 0.335, 'tpm': '20088/100000', 'rpm': '199/600'}}
}
Distribution overhead: 886.08ms
Average response latency: 3727.36ms
Total latency: 12423.68ms
Requests per second: 1128.57
Overhead per request: 0.89ms
```

## Configuration Reference

### switchboard.Model Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `name` | Configured model name, e.g. "gpt-4o" or "gpt-4o-mini" | Required |
| `tpm` | Configured TPM rate limit | 0 (unlimited) |
| `rpm` | Configured RPM rate limit | 0 (unlimited) |
| `default_cooldown` | Default cooldown period in seconds | 10.0 |

### switchboard.AzureDeployment Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `name` | Unique identifier for the deployment | Required |
| `endpoint` | Azure OpenAI endpoint URL | Required |
| `api_key` | Azure OpenAI API key | Required |
| `api_version` | Azure OpenAI API version | "2024-10-21" |
| `timeout` | Default timeout in seconds | 600.0 |
| `models` | List of Models configured for this deployment | Required |

### switchboard.Switchboard Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `deployments` | List of Deployment config objects | Required |
| `selector` | Selection algorithm | `two_random_choices` |
| `failover_policy` | Policy for handling failed requests | `AsyncRetrying(stop=stop_after_attempt(2))` |


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

* deployment inherits from azure client?
* opentelemetry integration
* lru list for usage tracking / better ratelimit handling
* add sync support?

## License

MIT
