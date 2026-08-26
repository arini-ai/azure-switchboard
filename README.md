# Azure Switchboard

`azure-switchboard` is a lightweight, coordination-free client load balancer for inference traffic against models hosted on Azure Foundry. It can be used as a drop-in replacement for the `openai` or `anthropic` SDKs and uses the [power of two random choices](https://www.eecs.harvard.edu/~michaelm/postscripts/handbook2001.pdf) algorithm to achieve stable, uniform load distribution between clients and across deployments without requiring clients to synchronize to track utilization. This is especially useful when deployments are not uniform in terms of quotas/ratelimits or model availability, for example, if your Azure account draws from multiple subscriptions, or when some models are only available/deployed in a subset of regions and not others.

```bash
uv add azure-switchboard
```

[![PyPI - Version](https://img.shields.io/pypi/v/azure-switchboard)](https://pypi.org/project/azure-switchboard/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![CI](https://github.com/arini-ai/azure-switchboard/actions/workflows/ci.yaml/badge.svg?branch=master)](https://github.com/arini-ai/azure-switchboard/actions/workflows/ci.yaml)

## Example

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

The endpoint is inferrable from the Foundry resource name or can be overridden explicitly on a per-deployment basis.

See [tools/readme_example.py](https://github.com/arini-ai/azure-switchboard/blob/master/tools/readme_example.py) for a runnable example.

## Features

- **Multi-Provider**: supports both the OpenAI Chat Completions API and the Anthropic Messages API. Non-OpenAI models that support the OpenAI API spec can be used via the OpenAIDeployment class.
- **Coordination-Free**: The default Two Random Choices algorithm does not require coordination between client instances to achieve excellent load distribution characteristics. See benchmarks for additional details.
- **Automatic Failover**: Failed requests are automatically retried on alternate deployments for the same model per a configurable retry policy, or, if enabled, to the first-party API if no healthy Foundry deployments are available.
- **Session Affinity**: A `session_id` can be provided to route requests in the same session to the same resource, to optimize for prompt caching.
- **Pluggable Selection**: Custom selection algorithms can be provided by passing a callable to the `selector` parameter on the Switchboard constructor.
- **Telemetry**: OpenTelemetry captures request spans and metrics for tokens, latency, failovers, cooldowns, and live utilization tagged by resource.

## Retry Behavior

Switchboard tracks utilization by deployment (tpm/rpm usage on a given model in a given Foundry) and health by resource (non-429 errors from the Foundry endpoint). If an individual model on a given Foundry resource is overloaded or ratelimited, requests for that model will be routed to deployments of the same model on other available Foundries. If the resource as a whole is unhealthy, it gets marked down entirely and excluded from selection until its cooldown period has expired.

Switchboard does not replace the SDK-default retry policy. For example, the `openai` SDK defaults to retrying a failing request 3 times before giving up. Switchboard's failover occurs after the SDK's internal retries are exhausted, according to its own failover policy, which defaults to 2 attempts. This means that out of the box a total of 6 attempts will be made to complete a request. Suppose Switchboard is configured with 3 deployments for an OpenAI model and a custom failover policy of 3 attempts. If two of those three deployments turn out to be unhealthy (and we get unlucky with the selection ordering), at least 7 requests will be attempted: 3 requests from the OpenAI SDK to the first deployment that was selected, another 3 requests from the OpenAI SDK to the second deployment that was selected after Switchboard's first failover attempt, and at least one request to the third deployment that was selected after Switchboard's second failover attempt. If the third deployment also fails, `SwitchboardError` is raised.

The retry policies of both the internal SDK and Switchboard can be configured to trade off between resilience and latency.

## Benchmarks

1000 requests spread over 10 deployments of `gpt-5.4-mini`:

```bash
just bench
uv run tools/bench.py -v -r 1000 -d 10 -e 500
Distributing 1000 requests across 10 deployments
Max inflight requests: 1000

{
    'bench_0': {'gpt-5.4-mini': UtilStats(util=0.337, tpm=Quota(used=8388, limit=30000), rpm=Quota(used=100, limit=300))},
    'bench_1': {'gpt-5.4-mini': UtilStats(util=0.331, tpm=Quota(used=8251, limit=30000), rpm=Quota(used=99, limit=300))},
    ...
    'bench_9': {'gpt-5.4-mini': UtilStats(util=0.342, tpm=Quota(used=8435, limit=30000), rpm=Quota(used=100, limit=300))}
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

## Contributing

1. Fork/clone repo
2. Make changes
3. Run tests with `just test`
4. Lint with `just lint`
5. Commit and make a PR

## License

MIT
