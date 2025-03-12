#!/usr/bin/env python3
#
# To run this, use:
#   uv run api_demo.py
#
# // script
# requires-python = ">=3.10"
# dependencies = [
#     "azure-switchboard",
#     "rich",
# ]
# ///

import argparse
import asyncio
import os
import time

from rich import print as rprint

from azure_switchboard import AzureDeployment, Model, Switchboard


async def bench(n_requests: int = 100, n_deployments: int = 3) -> None:
    deployments = [
        AzureDeployment(
            name=f"bench_{n}",
            endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
            api_key=os.environ["AZURE_OPENAI_API_KEY"],
            models=[Model(name="gpt-4o-mini", tpm=100000, rpm=600)],
        )
        for n in range(n_deployments)
    ]

    async with Switchboard(deployments) as switchboard:
        rprint(f"Distributing {n_requests} requests across {n_deployments} deployments")

        async def _request(i: int):
            start = time.perf_counter()
            await switchboard.create(
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "user",
                        "content": f"Can you tell me a fact about the number {i}?",
                    },
                ],
            )
            end = time.perf_counter()
            # rprint(f"{i}: {(end - start) * 1000:.2f}ms")
            return start, end

        start = time.perf_counter()
        results = await asyncio.gather(*[_request(i) for i in range(n_requests)])
        total_latency = (time.perf_counter() - start) * 1000

        first_start, last_start = results[0][0], results[-1][0]
        distribution_latency = (last_start - first_start) * 1000
        avg_response_latency = sum(
            (end - start) * 1000 for start, end in results
        ) / len(results)

        rprint(switchboard.get_usage())
        rprint(f"Distribution overhead: {distribution_latency:.2f}ms")
        rprint(f"Average response latency: {avg_response_latency:.2f}ms")
        rprint(f"Total latency: {total_latency:.2f}ms")
        rprint(f"Requests per second: {(n_requests / distribution_latency) * 1000:.2f}")
        rprint(f"Overhead per request: {(distribution_latency) / n_requests:.2f}ms")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark Azure OpenAI deployments.")
    parser.add_argument(
        "-r", "--requests", type=int, default=100, help="Number of requests to send."
    )
    parser.add_argument(
        "-d", "--deployments", type=int, default=3, help="Number of deployments to use."
    )
    args = parser.parse_args()

    asyncio.run(bench(n_requests=args.requests, n_deployments=args.deployments))
