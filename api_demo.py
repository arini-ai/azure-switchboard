#! /usr/bin/env python3

import asyncio
import logging
import os

from rich import print as rprint
from rich.logging import RichHandler

from switchboard import Deployment, Switchboard

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

    await basic_completion(switchboard)
    await basic_stream(switchboard)
    await distribute_N(switchboard, 10)
    # await distribute_N(switchboard, 100)

    await switchboard.close()


async def basic_completion(switchboard: Switchboard) -> None:
    rprint("# Basic completion")
    rprint(f"args: {BASIC_ARGS}")
    completion = switchboard.completion(**BASIC_ARGS)

    print("response: ", end="")
    if response := await completion:
        print(response.choices[0].message.content)

    rprint(switchboard)


async def basic_stream(switchboard: Switchboard) -> None:
    rprint("# Basic stream")
    rprint(f"args: {BASIC_ARGS}")
    stream = switchboard.stream(**BASIC_ARGS)

    print("response: ", end="")
    if response := await stream:
        async for chunk in response:
            if chunk.choices and chunk.choices[0].delta.content:
                print(chunk.choices[0].delta.content, end="")
        print()

    rprint(switchboard)


async def distribute_N(switchboard: Switchboard, N: int) -> None:
    switchboard.reset_usage()

    completions = []
    for i in range(N):
        completions.append(
            switchboard.completion(
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "user",
                        "content": f"Can you tell me a fact about the number {i}?",
                    },
                ],
            )
        )

    rprint(f"# Distribute {N} completions")
    responses = await asyncio.gather(*completions)
    rprint("Responses:", len(responses))
    rprint(switchboard)


if __name__ == "__main__":
    # logging.basicConfig(
    #     level=logging.INFO,
    #     format="%(message)s",
    #     datefmt="[%X]",
    #     handlers=[RichHandler(rich_tracebacks=True)],
    # )
    asyncio.run(main())
