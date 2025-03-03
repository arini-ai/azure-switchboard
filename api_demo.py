#! /usr/bin/env python3

import asyncio
import logging
import os

from rich.logging import RichHandler

from switchboard import Deployment, Switchboard

logger = logging.getLogger(__name__)

API_BASE = os.getenv("AZURE_OPENAI_ENDPOINT")
API_KEY = os.getenv("AZURE_OPENAI_API_KEY")

assert API_BASE, "AZURE_OPENAI_ENDPOINT must be set"
assert API_KEY, "AZURE_OPENAI_API_KEY must be set"

deployment = Deployment(
    name="demo_1",
    api_base=API_BASE,
    api_key=API_KEY,
    max_rpm=10,
    max_tpm=60,
)


async def main():
    switchboard = Switchboard([deployment])

    response = await switchboard.completion(
        model="gpt-4o-mini",
        messages=[
            {"role": "user", "content": "Hello, how are you?"},
        ],
    )

    logger.info(response)

    stream = await switchboard.completion(
        model="gpt-4o-mini",
        messages=[
            {"role": "user", "content": "Hello, how are you?"},
        ],
        stream=True,
    )

    async for chunk in stream:
        logger.info(chunk)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
        datefmt="[%X]",
        handlers=[RichHandler(rich_tracebacks=True)],
    )
    asyncio.run(main())
