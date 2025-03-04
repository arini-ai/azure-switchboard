#! /usr/bin/env python3

import asyncio
import logging
import os

from rich import print as rprint
from rich.logging import RichHandler

from switchboard.client import Deployment
from switchboard.switchboard import Switchboard

logger = logging.getLogger(__name__)

API_BASE = os.getenv("AZURE_OPENAI_ENDPOINT")
API_KEY = os.getenv("AZURE_OPENAI_API_KEY")

assert API_BASE, "AZURE_OPENAI_ENDPOINT must be set"
assert API_KEY, "AZURE_OPENAI_API_KEY must be set"

deployment = Deployment(
    name="demo_1",
    api_base=API_BASE,
    api_key=API_KEY,
    rpm_ratelimit=10,
    tpm_ratelimit=60,
)


async def main():
    switchboard = Switchboard([deployment])

    completion = switchboard.completion(
        model="gpt-4o-mini",
        messages=[
            {"role": "user", "content": "Hello, who are you?"},
        ],
    )

    if response := await completion:
        print(response.choices[0].message.content)

    rprint(switchboard.usage())

    stream = switchboard.stream(
        model="gpt-4o-mini",
        messages=[
            {"role": "user", "content": "Hello, who are you?"},
        ],
    )

    if response := await stream:
        async for chunk in response:
            # rprint(chunk)
            if chunk.choices and chunk.choices[0].delta.content:
                print(chunk.choices[0].delta.content, end="")
        print()

    rprint(switchboard.usage())

    rprint(switchboard)

    await switchboard.close()


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
        datefmt="[%X]",
        handlers=[RichHandler(rich_tracebacks=True)],
    )
    asyncio.run(main())
