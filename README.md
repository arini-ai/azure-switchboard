# Azure Switchboard

Batteries-included, coordination-free client loadbalancing for Azure AI Foundry — OpenAI and Anthropic models alike.

```bash
uv add azure-switchboard
```

[![PyPI - Version](https://img.shields.io/pypi/v/azure-switchboard)](https://pypi.org/project/azure-switchboard/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![CI](https://github.com/arini-ai/azure-switchboard/actions/workflows/ci.yaml/badge.svg?branch=master)](https://github.com/arini-ai/azure-switchboard/actions/workflows/ci.yaml)

## Overview

`azure-switchboard` is a Python 3 asyncio library that spreads Chat Completions and Anthropic Messages traffic across the deployments you already have, with no coordination between client instances. You ask for a model; it picks a healthy deployment of that model using [power of two random choices](https://www.eecs.harvard.edu/~michaelm/postscripts/handbook2001.pdf), tracks what each one is carrying, and routes around the ones that start refusing.

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

You get the endpoint for free from the resource name, and a resource can host models of both kinds on one credential — so the thing you configure matches the thing you pay for.

## Features

- **API Compatibility**: each surface mirrors its SDK's own call path, so you port by swapping the client and changing nothing else. Return types are exact — no unions to narrow.

  | SDK call                         | Switchboard call             | Returns                     |
  | -------------------------------- | ---------------------------- | --------------------------- |
  | `openai.chat.completions.create` | `sb.chat.completions.create` | `ChatCompletion`            |
  | `openai.chat.completions.parse`  | `sb.chat.completions.parse`  | `ParsedChatCompletion[T]`   |
  | `openai.chat.completions.stream` | `sb.chat.completions.stream` | `AsyncChatCompletionStream` |
  | `anthropic.messages.create`      | `sb.messages.create`         | `Message`                   |
  | `anthropic.messages.parse`       | `sb.messages.parse`          | `ParsedMessage[T]`          |
  | `anthropic.messages.stream`      | `sb.messages.stream`         | `AsyncMessageStream`        |

- **Multi-Provider**: GPT and Claude deployments live side by side in one Switchboard, sharing utilization tracking, affinity and failover. Nothing about your OpenAI setup changes when you add a Claude deployment to it.
- **Coordination-Free**: The default Two Random Choices algorithm does not require coordination between client instances to achieve excellent load distribution characteristics.
- **Utilization-Aware**: TPM/RPM utilization is tracked per deployment for use during selection.
- **Batteries Included**:
  - **Session Affinity**: Provide a `session_id` to route requests in the same session to the same resource, so a session spanning several models keeps one prompt cache warm.
  - **Automatic Failover**: Retries are controlled by a tenacity `AsyncRetrying` policy (`failover_policy`).
  - **First-Party Fallback**: Set `openai_fallback=True` / `anthropic_fallback=True` to back the pool with the vendors' own APIs once nothing healthy is left.
  - **Pluggable Selection**: Custom selection algorithms can be provided by passing a callable to the `selector` parameter on the Switchboard constructor.
  - **OpenTelemetry Integration**: Built-in metrics for request routing and healthy deployment counts.
- **Lightweight**: Small codebase with minimal dependencies: `openai`, `anthropic`, `loguru`, `tenacity`, `wrapt`, and `opentelemetry-api`.

## Migrating from 2026.8.0

The configuration API was replaced wholesale. The old shape made you register an Azure
resource twice to serve both a GPT and a Claude model from it, which split its quota
accounting in two; now one resource holds both.

```python
# before
Switchboard(deployments=[
    OpenAIConfig(
        name="east",
        base_url="https://east.openai.azure.com/openai/v1/",
        api_key=...,
        models=[Model(name="gpt-4o-mini", tpm=30000, rpm=300)],
    ),
    AnthropicConfig(
        name="east-anthropic",          # the same resource, registered again
        resource="east",
        api_key=...,
        models=[Model(name="claude-sonnet-5", tpm=30000, rpm=300)],
    ),
])

# after
Switchboard(resources=[
    Foundry(name="east", api_key=..., models=[
        OpenAIDeployment("gpt-4o-mini", tpm=30000, rpm=300),
        AnthropicDeployment("claude-sonnet-5", tpm=30000, rpm=300),
    ]),
])
```

| 2026.8.0                                      | Now                                                                    |
| --------------------------------------------- | ---------------------------------------------------------------------- |
| `OpenAIConfig(...)` / `AnthropicConfig(...)`  | `Foundry(...)`                                                         |
| `Model("gpt-4o-mini", ...)`                   | `OpenAIDeployment(...)` or `AnthropicDeployment(...)`                  |
| `Switchboard(deployments=[...])`              | `Switchboard(resources=[...])`                                         |
| `base_url="https://east...openai/v1/"`        | derived from `Foundry(name=...)`; `endpoint=` per deployment overrides |
| `AnthropicConfig(resource="east")`            | `Foundry(name="east")`                                                 |
| `OpenAIConfig(base_url=None)` for first-party | `Switchboard(openai_fallback=True)`                                    |
| `sb.select_deployment(model=...)`             | removed — read `sb.sessions[session_id]` for the pinned resource       |
| `selector(model, deployments)`                | `selector(deployments)`                                                |

Three behavioural changes come with it:

- **An exhausted pool raises.** Previously the last resort was a deployment already
  cooling down; now it is `SwitchboardError` unless a first-party fallback is
  configured.
- **Connection errors cool the whole resource**, not just the deployment that saw
  one — see [Cooldown Scope](#cooldown-scope). Rate limits still cool one deployment.
- **Session affinity pins to the resource**, so a session using several models keeps
  one prompt cache warm.

Telemetry follows the rename: the log context key and the `requests` counter
attribute are `resource`, previously `deployment`.

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

from azure_switchboard import Foundry, OpenAIDeployment, Switchboard

azure_openai_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
azure_openai_api_key = os.getenv("AZURE_OPENAI_API_KEY")
openai_api_key = os.getenv("OPENAI_API_KEY")

resources = []
if azure_openai_endpoint and azure_openai_api_key:
    # create 3 resources. reusing the endpoint
    # is fine for the purposes of this demo
    for name in ("east", "west", "south"):
        resources.append(
            Foundry(
                name=name,
                api_key=azure_openai_api_key,
                models=[
                    OpenAIDeployment(
                        name="gpt-4o-mini",
                        endpoint=f"{azure_openai_endpoint}/openai/v1/",
                    )
                ],
            )
        )

if not resources and not openai_api_key:
    raise RuntimeError(
        "Set AZURE_OPENAI_ENDPOINT/AZURE_OPENAI_API_KEY or OPENAI_API_KEY to run this example."
    )


async def main():
    # OPENAI_API_KEY, if set, backs the pool as a last resort
    async with Switchboard(
        resources=resources, openai_fallback=bool(openai_api_key)
    ) as sb:
        print("Basic functionality:")
        await basic_functionality(sb)

        print("Session affinity (should warn):")
        await session_affinity(sb)


async def basic_functionality(switchboard: Switchboard):
    # Make a completion request (non-streaming)
    response = await switchboard.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": "Hello, world!"}],
    )

    print("completion:", response.choices[0].message.content)

    # Make a streaming completion request
    stream = await switchboard.chat.completions.create(
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

    # First message will select a random healthy deployment
    # and pin the session_id to the foundry hosting it
    r = await switchboard.chat.completions.create(
        session_id=session_id,
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": "Who won the World Series in 2020?"}],
    )

    # the session is now pinned to whichever foundry served it
    f1 = switchboard.sessions[session_id]
    print("foundry 1:", f1.name)
    print("response 1:", r.choices[0].message.content)

    # Follow-up requests with the same session_id will route to the same foundry
    r2 = await switchboard.chat.completions.create(
        session_id=session_id,
        model="gpt-4o-mini",
        messages=[
            {"role": "user", "content": "Who won the World Series in 2020?"},
            {"role": "assistant", "content": r.choices[0].message.content},
            {"role": "user", "content": "Who did they beat?"},
        ],
    )

    print("response 2:", r2.choices[0].message.content)

    # Simulate a failure by marking down the deployment that served us
    f1.models["gpt-4o-mini"].mark_down()

    # A new foundry will be selected for this session_id
    r3 = await switchboard.chat.completions.create(
        session_id=session_id,
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": "Who won the World Series in 2021?"}],
    )

    f2 = switchboard.sessions[session_id]
    print("foundry 2:", f2.name)
    print("response 3:", r3.choices[0].message.content)
    assert f2 is not f1


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

### switchboard.Foundry Parameters

| Parameter          | Description                                                                       | Default  |
| ------------------ | --------------------------------------------------------------------------------- | -------- |
| `name`             | Azure resource name, resolving to `https://<name>.services.ai.azure.com/`         | Required |
| `api_key`          | API key for the resource, shared by every deployment on it                        | None     |
| `timeout`          | Per-request timeout in seconds. Raise it for batch jobs that need longer budgets. | 30.0     |
| `models`           | Deployments hosted on this resource                                               | `()`     |
| `default_cooldown` | Cooldown (seconds) applied to the whole resource when it is unreachable           | 10.0     |

### switchboard.OpenAIDeployment / switchboard.AnthropicDeployment Parameters

| Parameter          | Description                                                          | Default       |
| ------------------ | -------------------------------------------------------------------- | ------------- |
| `name`             | Model name, sent as `model` and used as the routing key              | Required      |
| `tpm`              | Tokens-per-minute budget used for utilization tracking and routing   | 0 (unlimited) |
| `rpm`              | Requests-per-minute budget used for utilization tracking and routing | 0 (unlimited) |
| `endpoint`         | Full base URL, overriding the one derived from the resource name     | Derived       |
| `default_cooldown` | Cooldown duration (seconds) after this deployment is marked down     | 10.0          |

`OpenAIDeployment` is served at `sb.chat.completions` and reached at `<resource>/openai/v1/`; `AnthropicDeployment` at `sb.messages` and `<resource>/anthropic/`. Set `endpoint` to point a deployment somewhere else — a legacy `<resource>.openai.azure.com` host, or a non-Azure one.

For a host with no Azure endpoint to derive, use `Resource` in place of `Foundry` and give each deployment an explicit `endpoint`:

```python
Switchboard(resources=[
    Resource(name="gateway", api_key=..., models=[
        OpenAIDeployment("gpt-4o-mini", endpoint="https://gateway.internal/v1/"),
    ]),
])
```

### switchboard.Switchboard Parameters

| Parameter            | Description                                                                                        | Default              |
| -------------------- | -------------------------------------------------------------------------------------------------- | -------------------- |
| `resources`          | `Foundry` or plain `Resource` objects to balance across. May be empty if a fallback is configured. | Required             |
| `selector`           | Deployment selection function, `(eligible_deployments) -> deployment`                              | `two_random_choices` |
| `failover_policy`    | tenacity `AsyncRetrying` policy, copied per call so requests do not share retry state              | 2 attempts           |
| `ratelimit_window`   | How often usage counters reset (seconds). Set `0` to disable periodic reset.                       | 60.0                 |
| `max_sessions`       | LRU capacity for session affinity pins                                                             | 1024                 |
| `openai_fallback`    | Fall back to the OpenAI API, keyed from `OPENAI_API_KEY`                                           | False                |
| `anthropic_fallback` | Fall back to the Anthropic API, keyed from `ANTHROPIC_API_KEY`                                     | False                |

A selector only ever sees deployments of the model that was asked for, so it picks between them on utilization alone. To see which resource a session is pinned to, read `sb.sessions[session_id]`.

`max_tokens` is required by the Messages API and is passed through unchanged — switchboard does not supply a default.

### Streaming

Both SDKs offer two streaming entry points, and both work here.

`create(stream=True)` gives you the raw events:

```python
stream = await sb.messages.create(model=..., max_tokens=1024, messages=[...], stream=True)
async for event in stream:
    ...
```

`stream()` gives you the SDK's accumulating helper, so you get the events _and_ a
finished `Message` or `ChatCompletion` with content, tool-use blocks and usage already
assembled — no reassembling it yourself:

```python
async with sb.messages.stream(model=..., max_tokens=1024, messages=[...]) as s:
    async for event in s:
        ...
    final = await s.get_final_message()
```

Either way, utilization is tracked and a deployment that fails to open is skipped in
favour of another. A stream that dies part-way through cannot be retried.

### First-Party Fallback

`openai_fallback=True` / `anthropic_fallback=True` keeps you serving when your Azure
capacity is exhausted: requests spill over to the vendor's own API, using the
credential from `OPENAI_API_KEY` / `ANTHROPIC_API_KEY`. It never competes with a
healthy deployment — it is only reached when nothing else is left — and it covers
models you have not deployed on Azure at all, so you can call something new without
provisioning it first. Configure a fallback and you can skip `resources` entirely.

A fallback that cannot authenticate fails when you construct the `Switchboard`, not
during the outage it exists to cover.

### Cooldown Scope

An error's cooldown is scoped to what that error actually implicates. Both providers are handled identically, against their respective SDK exception classes.

| Error                | Scope           | Rationale                                                                                                                                                                 |
| -------------------- | --------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `RateLimitError`     | that deployment | Azure allocates TPM/RPM per deployment. One model exhausting its quota says nothing about another's on the same resource, so cooling the resource would discard capacity. |
| `APIConnectionError` | whole resource  | The host is unreachable, so every deployment on it is unreachable.                                                                                                        |
| `APITimeoutError`    | nothing         | Timeouts during an upstream-wide slowdown are _not_ correlated with the chosen deployment. Marking one down wastes capacity without providing a fix.                      |

A cooling resource reports full utilization for every deployment on it, which also releases any session pinned to it back to normal selection.

If your workload has a longer latency budget (e.g. batch structured-output jobs), set `timeout` on the relevant `Foundry` rather than relying on the default.

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
just test        # unit tests; every upstream is mocked
just typecheck   # pyright over src/, as CI runs it
```

`just smoke` drives real inference against a live Foundry resource, to check the
wiring the mocks cannot. It needs `AZURE_FOUNDRY` and `AZURE_API_KEY`, and picks up
`OPENAI_API_KEY` / `ANTHROPIC_API_KEY` to exercise first-party fallback. The
committed `.envrc` sources `.envrc.local`, which is gitignored — put your keys there.

### Release

This library uses CalVer for versioning. On push to master, if tests pass, a package is automatically built, released, and uploaded to PyPI.

Locally, the package can be built with uv:

```bash
uv build
```

### OpenTelemetry Integration

`azure-switchboard` uses OpenTelemetry metrics via the meter `azure_switchboard.switchboard`.

Metrics emitted on the request path include:

- `healthy_deployments_count` (gauge)
- `requests` (counter, with deployment + model attributes)

To run with local OTEL instrumentation:

```bash
just otel-run
```

## Contributing

1. Fork/clone repo
2. Make changes
3. Run tests with `just test`
4. Lint with `just lint`
5. Commit and make a PR

## License

MIT
