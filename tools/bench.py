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
import math
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
            models=[Model(name="gpt-4o-mini", tpm=30000, rpm=300)],
        )
        for n in range(args.deployments)
    ]

    async with Switchboard(deployments) as switchboard:
        print(
            f"Distributing {args.requests} requests across {args.deployments} deployments"
        )
        print(f"Max inflight requests: {args.inflight}")
        print()

        inflight_requests = asyncio.Semaphore(args.inflight)

        async def _request(i: int):
            async with inflight_requests:
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

            if args.verbose and (i > 0 and i % args.every == 0):
                print(f"Request {i}/{args.requests} completed")
                print_usage_histogram(switchboard.get_usage(), bins=5, absolute=True)
                print()

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

        if args.verbose:
            usage = switchboard.get_usage()
            print(usage)
            print()
            print_usage_histogram(usage, absolute=True)
            print()

        print(f"Distribution overhead: {distribution_latency:.2f}ms")
        print(f"Average response latency: {avg_response_latency:.2f}ms")
        print(f"Total latency: {total_latency:.2f}ms")
        print(f"Requests per second: {args.requests / distribution_latency * 1000:.2f}")
        print(f"Overhead per request: {distribution_latency / args.requests:.2f}ms")


def print_usage_histogram(data: dict, width=30, bins=10, absolute=False):
    """
    Generate a histogram showing the distribution of utilization values.

    Args:
        data: dict of {deployment: {model: {metrics}}} returned by switchboard.get_usage()
        width: Width of the histogram bars in characters
        bins: Number of bins to divide the range into
        absolute: If True, use fixed range of 0.0-1.0; if False, use min-max range
    """
    # Extract the utilization values from the usage data
    all_utils = []
    for model in data.values():
        for metrics in model.values():
            all_utils.append(metrics.get("util"))

    if not all_utils:
        print("No utilization data available")
        return

    min_util = 0.0 if absolute else min(all_utils)
    max_util = 1.0 if absolute else max(all_utils)

    if min_util == max_util and not absolute:
        # Handle case where min and max are the same (all values identical)
        # create a small range around the single value
        min_util = max(0.0, min_util - 0.05)
        max_util = min(1.0, max_util + 0.05)

    # Create bins based on the range
    bin_counts = [0] * bins
    range_size = max_util - min_util

    # Place values in appropriate bins
    for util in all_utils:
        if range_size > 0:
            # Calculate which bin this value belongs in
            bin_index = min(int((util - min_util) / range_size * bins), bins - 1)
            bin_counts[bin_index] += 1

    # Find the maximum count for scaling
    max_count = max(bin_counts) if bin_counts else 0

    print("Utilization Distribution:")

    # Print each bin
    for i in range(bins):
        start = min_util + (i / bins) * range_size
        end = min_util + ((i + 1) / bins) * range_size
        count = bin_counts[i]

        # Calculate the bar length based on the count
        bar_length = round(width * count / max_count) if max_count > 0 else 0
        bar = "." * bar_length  # "█" * bar_length

        print(f"{start:.3f} - {end:.3f} | {count:>3} {bar}")

    # Calculate and print statistics
    avg_util = sum(all_utils) / len(all_utils)
    min_util = min(all_utils)
    max_util = max(all_utils)

    print(f"Avg utilization: {avg_util:.3f} ({min_util:.3f} - {max_util:.3f})")
    if len(all_utils) > 1:
        variance = sum((u - avg_util) ** 2 for u in all_utils) / len(all_utils)
        stddev = math.sqrt(variance)
        print(f"Std deviation: {stddev:.3f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark Switchboard performance")
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
    parser.add_argument(
        "-e",
        "--every",
        type=int,
        default=100,
        help="Print every Nth response.",
    )
    parser.add_argument(
        "-n",
        "--no-reset",
        action="store_true",
        help="Do not reset the usage counter.",
    )
    parser.add_argument(
        "--live",
        action="store_true",
        help="Use real endpoint, otherwise mock.",
    )
    args = parser.parse_args()

    asyncio.run(bench(args))
