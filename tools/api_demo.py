#!/usr/bin/env python3
#
# To run this, use:
#   uv run readme-example.py
#
# ///
# requires-python = ">=3.10"
# dependencies = [
#     "azure-switchboard",
#     "rich",
# ]
# ///

import asyncio
import logging
import os
import time

from rich import print as rprint
from rich.logging import RichHandler

from azure_switchboard import Deployment, Switchboard

logger = logging.getLogger(__name__)

API_BASE = os.getenv("AZURE_OPENAI_ENDPOINT")
API_KEY = os.getenv("AZURE_OPENAI_API_KEY")

assert API_BASE, "AZURE_OPENAI_ENDPOINT must be set"
assert API_KEY, "AZURE_OPENAI_API_KEY must be set"

d1 = Deployment(
    name="demo_1",
    api_base=API_BASE,
    api_key=API_KEY,
    rpm_ratelimit=6,
    tpm_ratelimit=1000,
)

d2 = Deployment(
    name="demo_2",
    api_base=API_BASE,
    api_key=API_KEY,
    rpm_ratelimit=6,
    tpm_ratelimit=1000,
)

d3 = Deployment(
    name="demo_3",
    api_base=API_BASE,
    api_key=API_KEY,
    rpm_ratelimit=6,
    tpm_ratelimit=1000,
)

BASIC_ARGS = {
    "model": "gpt-4o-mini",
    "messages": [
        {"role": "user", "content": "Hello, who are you?"},
    ],
}


async def main() -> None:
    switchboard = Switchboard([d1, d2, d3])
    switchboard.start()

    rprint("# Basic completion")
    await basic_completion(switchboard)

    rprint("# Basic stream")
    await basic_stream(switchboard)

    rprint("# Distribute 10 completions, 1:1:1 ratio")
    await distribute_N(switchboard, 10)
    rprint(switchboard.get_usage())

    rprint("# Distribute 100 completions, 1:2:3 ratio")
    d2.tpm_ratelimit *= 2
    d2.rpm_ratelimit *= 2

    d3.tpm_ratelimit *= 3
    d3.rpm_ratelimit *= 3
    await distribute_N(switchboard, 100)
    rprint(switchboard.get_usage())

    rprint("# Session affinity")
    await session_affinity(switchboard)
    rprint(switchboard.get_usage())

    await switchboard.stop()


async def basic_completion(switchboard: Switchboard) -> None:
    rprint(f"args: {BASIC_ARGS}")
    print("response: ", end="")
    if response := await switchboard.create(**BASIC_ARGS):
        print(response.choices and response.choices[0].message.content)

    # rprint(switchboard)


async def basic_stream(switchboard: Switchboard) -> None:
    rprint(f"args: {BASIC_ARGS}")
    stream = await switchboard.create(stream=True, **BASIC_ARGS)

    print("response: ", end="")
    async for chunk in stream:
        if chunk.choices and chunk.choices[0].delta.content:
            print(chunk.choices[0].delta.content, end="", flush=True)
    print()

    # rprint(switchboard)


async def distribute_N(switchboard: Switchboard, N: int) -> None:
    switchboard.reset_usage()

    completions = []
    for i in range(N):
        completions.append(
            switchboard.create(
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "user",
                        "content": f"Can you tell me a fact about the number {i}?",
                    },
                ],
            )
        )

    responses = await asyncio.gather(*completions)
    rprint("Responses:", len(responses))


async def session_affinity(switchboard: Switchboard) -> None:
    switchboard.reset_usage()

    session_id = f"test_{time.time()}"
    await switchboard.create(session_id=session_id, **BASIC_ARGS)
    await switchboard.create(session_id=session_id, **BASIC_ARGS)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
        datefmt="[%X]",
        handlers=[RichHandler(rich_tracebacks=True)],
    )
    asyncio.run(main())
