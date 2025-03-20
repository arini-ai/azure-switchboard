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

from rich import print

from azure_switchboard import AzureDeployment, Model, Switchboard


async def bench(args: argparse.Namespace) -> None:
    deployments = [
        AzureDeployment(
            name=f"bench_{n}",
            endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
            api_key=os.environ["AZURE_OPENAI_API_KEY"],
            models=[Model(name="gpt-4o-mini", tpm=1000000, rpm=6000)],
        )
        for n in range(args.deployments)
    ]

    async with Switchboard(deployments) as switchboard:
        print("Requests:", args.requests)
        print("Deployments:", args.deployments)
        print("Max inflight:", args.inflight)

        _inflight = asyncio.Semaphore(args.inflight)

        async def _request(i: int):
            async with _inflight:
                start = time.perf_counter()
                response = await switchboard.create(
                    model="gpt-4o-mini",
                    messages=[
                        {
                            "role": "user",
                            "content": f"Can you tell me a fact about the number {i}?",
                        },
                    ],
                )
                end = time.perf_counter()

            if i % 100 == 0 and args.verbose:
                print(response.choices[0].message.content)
                print(switchboard.get_usage())

            return start, end

        try:
            start = time.perf_counter()
            results = await asyncio.gather(*[_request(i) for i in range(args.requests)])
            total_latency = (time.perf_counter() - start) * 1000

            first_start, last_start = results[0][0], results[-1][0]
            distribution_latency = (last_start - first_start) * 1000
            avg_response_latency = sum(
                (end - start) * 1000 for start, end in results
            ) / len(results)
        except Exception as e:
            print(e)
            print(switchboard.get_usage())
            return

        print(switchboard.get_usage())
        print(f"Distribution overhead: {distribution_latency:.2f}ms")
        print(f"Average response latency: {avg_response_latency:.2f}ms")
        print(f"Total latency: {total_latency:.2f}ms")
        print(f"RPS: {(args.requests / distribution_latency) * 1000:.2f}")
        print(f"Overhead per request: {(distribution_latency) / args.requests:.2f}ms")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark Azure OpenAI deployments.")
    parser.add_argument(
        "-r", "--requests", type=int, default=100, help="Number of requests to send."
    )
    parser.add_argument(
        "-d", "--deployments", type=int, default=3, help="Number of deployments to use."
    )
    parser.add_argument(
        "-i",
        "--inflight",
        type=int,
        default=1000,
        help="Maximum number of inflight requests.",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Print verbose output.",
    )
    args = parser.parse_args()

    asyncio.run(bench(args))
