# Azure Switchboard

Batteries-included, coordination-free client loadbalancing for Azure OpenAI.

```bash
uv add azure-switchboard
```

[![PyPI - Version](https://img.shields.io/pypi/v/azure-switchboard)](https://pypi.org/project/azure-switchboard/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![CI](https://github.com/arini-ai/azure-switchboard/actions/workflows/ci.yaml/badge.svg?branch=master)](https://github.com/arini-ai/azure-switchboard/actions/workflows/ci.yaml)

## Overview

`azure-switchboard` is a Python 3 asyncio library that provides an intelligent, API-compatible client loadbalancer for Azure OpenAI. You instantiate a Switchboard client with a set of deployments, and the client distributes your chat completion requests across the available deployments using the [power of two random choices](https://www.eecs.harvard.edu/~michaelm/postscripts/handbook2001.pdf) method. In this sense, it functions as a lightweight service mesh between your application and Azure OpenAI. The basic idea is inspired by [ServiceRouter](https://www.usenix.org/system/files/osdi23-saokar.pdf).

## Features

- **API Compatibility**: `Switchboard.create` is a transparently-typed drop-in proxy for `OpenAI.chat.completions.create`.
- **Coordination-Free**: The default Two Random Choices algorithm does not require coordination between client instances to achieve excellent load distribution characteristics.
- **Utilization-Aware**: TPM/RPM ratelimit utilization is tracked per model per deployment for use during selection.
- **Batteries Included**:
  - **Session Affinity**: Provide a `session_id` to route requests in the same session to the same deployment, optimizing for prompt caching
  - **Automatic Failover**: Client automatically retries on request failure. The retry policy can be customized by passing a tenacity `AsyncRetrying` instance to `failover_policy`.
  - **Pluggable Selection**: Custom selection algorithms can be provided by passing a callable to the `selector` parameter on the Switchboard constructor.
  - **OpenTelemetry Integration**: Comprehensive metrics and instrumentation for monitoring deployment health and utilization.

- **Lightweight**: sub-400 LOC implementation with minimal dependencies: `openai`, `tenacity`, `wrapt`, and `opentelemetry-api`. <1ms overhead per request.
- **100% Test Coverage**: Comprehensive test suite with pytest.

## Runnable Example

```python
#!/usr/bin/env python3
#
# To run this, use:
#   uv run --env-file .env tools/readme_example.py
#
# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "azure-switchboard",
# ]
# ///

import asyncio
import os

from azure_switchboard import Deployment, Model, Switchboard

azure_openai_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
azure_openai_api_key = os.getenv("AZURE_OPENAI_API_KEY")

deployments = []
if azure_openai_endpoint and azure_openai_api_key:
    # create 3 deployments. reusing the endpoint
    # is fine for the purposes of this demo
    for name in ("east", "west", "south"):
        deployments.append(
            Deployment(
                name=name,
                base_url=f"{azure_openai_endpoint}/openai/v1/",
                api_key=azure_openai_api_key,
                models=[Model(name="gpt-4o-mini")],
            )
        )


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
    d1.models["gpt-4o-mini"].mark_down()

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

## Benchmarks

```bash
just bench
uv run --env-file .env tools/bench.py -v -r 1000 -d 10 -e 500
Distributing 1000 requests across 10 deployments
Max inflight requests: 1000

Request 500/1000 completed
Utilization Distribution:
0.000 - 0.200 |   0
0.200 - 0.400 |  10 ..............................
0.400 - 0.600 |   0
0.600 - 0.800 |   0
0.800 - 1.000 |   0
Avg utilization: 0.339 (0.332 - 0.349)
Std deviation: 0.006

{
    'bench_0': {'gpt-4o-mini': {'util': 0.361, 'tpm': '10556/30000', 'rpm': '100/300'}},
    'bench_1': {'gpt-4o-mini': {'util': 0.339, 'tpm': '9819/30000', 'rpm': '100/300'}},
    'bench_2': {'gpt-4o-mini': {'util': 0.333, 'tpm': '9405/30000', 'rpm': '97/300'}},
    'bench_3': {'gpt-4o-mini': {'util': 0.349, 'tpm': '10188/30000', 'rpm': '100/300'}},
    'bench_4': {'gpt-4o-mini': {'util': 0.346, 'tpm': '10210/30000', 'rpm': '99/300'}},
    'bench_5': {'gpt-4o-mini': {'util': 0.341, 'tpm': '10024/30000', 'rpm': '99/300'}},
    'bench_6': {'gpt-4o-mini': {'util': 0.343, 'tpm': '10194/30000', 'rpm': '100/300'}},
    'bench_7': {'gpt-4o-mini': {'util': 0.352, 'tpm': '10362/30000', 'rpm': '102/300'}},
    'bench_8': {'gpt-4o-mini': {'util': 0.35, 'tpm': '10362/30000', 'rpm': '102/300'}},
    'bench_9': {'gpt-4o-mini': {'util': 0.365, 'tpm': '10840/30000', 'rpm': '101/300'}}
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
Avg utilization: 0.348 (0.333 - 0.365)
Std deviation: 0.009

Distribution overhead: 926.14ms
Average response latency: 5593.77ms
Total latency: 17565.37ms
Requests per second: 1079.75
Overhead per request: 0.93ms
```

Distribution overhead scales ~linearly with the number of deployments.

## Configuration Reference

### switchboard.Model Parameters

| Parameter          | Description                                           | Default       |
| ------------------ | ----------------------------------------------------- | ------------- |
| `name`             | Configured model name, e.g. "gpt-4o" or "gpt-4o-mini" | Required      |
| `tpm`              | Configured TPM rate limit                             | 0 (unlimited) |
| `rpm`              | Configured RPM rate limit                             | 0 (unlimited) |
| `default_cooldown` | Default cooldown period in seconds                    | 10.0          |

### switchboard.Deployment Parameters

| Parameter  | Description                                                                 | Default  |
| ---------- | --------------------------------------------------------------------------- | -------- |
| `name`     | Unique identifier for the deployment                                        | Required |
| `base_url` | Base URL for the API (e.g., `https://myazure.openai.azure.com/openai/v1/`)  | None     |
| `api_key`  | API key for the deployment                                                  | None     |
| `timeout`  | Default timeout in seconds                                                  | 600.0    |
| `models`   | List of Models configured for this deployment                               | Default  |

### switchboard.Switchboard Parameters

| Parameter          | Description                         | Default                                     |
| ------------------ | ----------------------------------- | ------------------------------------------- |
| `deployments`      | List of Deployment config objects   | Required                                    |
| `selector`         | Selection algorithm                 | `two_random_choices`                        |
| `failover_policy`  | Policy for handling failed requests | `AsyncRetrying(stop=stop_after_attempt(2))` |
| `ratelimit_window` | Ratelimit window in seconds         | 60.0                                        |
| `max_sessions`     | Maximum number of sessions          | 1024                                        |

## Development

This project uses [uv](https://github.com/astral-sh/uv) for package management,
and [just](https://github.com/casey/just) for task automation. See the [justfile](https://github.com/arini-ai/azure-switchboard/blob/master/justfile)
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

### OpenTelemetry Integration

The library provides instrumentation for monitoring deployment health and performance metrics:

```bash
(azure-switchboard) .venv > just otel-run
uv run --env-file .env opentelemetry-instrument python tools/bench.py -r 5 -d 3
Distributing 5 requests across 3 deployments
Max inflight requests: 1000

Distribution overhead: 10.53ms
Average response latency: 2164.03ms
Total latency: 3869.06ms
Requests per second: 475.03
Overhead per request: 2.11ms
{
    "resource_metrics": [
        {
            "resource": {
                "attributes": {
                    "telemetry.sdk.language": "python",
                    "telemetry.sdk.name": "opentelemetry",
                    "telemetry.sdk.version": "1.31.0",
                    "service.name": "switchboard",
                    "telemetry.auto.version": "0.52b0"
                },
                "schema_url": ""
            },
            "scope_metrics": [
                {
                    "scope": {
                        "name": "azure_switchboard.deployment",
                        "version": "",
                        "schema_url": "",
                        "attributes": null
                    },
                    "metrics": [
                        {
                            "name": "model_utilization",
                            "description": "Current utilization of a model deployment (0-1)",
                            "unit": "percent",
                            "data": {
                                "data_points": [
                                    {
                                        "attributes": {
                                            "model": "gpt-4o-mini"
                                        },
                                        "start_time_unix_nano": null,
                                        "time_unix_nano": 1742461487509982000,
                                        "value": 0.008,
                                        "exemplars": []
...
```

## Contributing

1. Fork/clone repo
2. Make changes
3. Run tests with `just test`
4. Lint with `just lint`
5. Commit and make a PR

## License

MIT
