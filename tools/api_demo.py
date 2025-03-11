#!/usr/bin/env python3
#
# To run this, use:
#   uv run api_demo.py
#
# /// script
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

from azure_switchboard import DeploymentConfig, Model, Switchboard

logger = logging.getLogger(__name__)

if not (api_base := os.getenv("AZURE_OPENAI_ENDPOINT")):
    api_base = "https://your-deployment1.openai.azure.com/"
if not (api_key := os.getenv("AZURE_OPENAI_API_KEY")):
    api_key = "your-api-key"

d1 = DeploymentConfig(
    name="demo_1",
    api_base=api_base,
    api_key=api_key,
    models=[Model(name="gpt-4o-mini", tpm=1000, rpm=6)],
)

d2 = DeploymentConfig(
    name="demo_2",
    api_base=api_base,
    api_key=api_key,
    models=[Model(name="gpt-4o-mini", tpm=1000, rpm=6)],
)

d3 = DeploymentConfig(
    name="demo_3",
    api_base=api_base,
    api_key=api_key,
    models=[Model(name="gpt-4o-mini", tpm=1000, rpm=6)],
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

    def _update_rl(name: str, multiple: int) -> None:
        switchboard.deployments[name].models["gpt-4o-mini"].tpm *= multiple
        switchboard.deployments[name].models["gpt-4o-mini"].rpm *= multiple

    _update_rl("demo_1", 10)
    _update_rl("demo_2", 20)
    _update_rl("demo_3", 30)

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
