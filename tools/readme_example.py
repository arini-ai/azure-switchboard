#!/usr/bin/env python3
#
# To run this, use:
#   uv run readme_example.py
#
# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "azure-switchboard",
# ]
# ///

import asyncio
import os
from contextlib import asynccontextmanager

from azure_switchboard import DeploymentConfig, Model, Switchboard

# use demo parameters from environment if available
api_base = os.getenv("AZURE_OPENAI_ENDPOINT", "https://test.openai.azure.com/")
api_key = os.getenv("AZURE_OPENAI_API_KEY", "your-api-key")
openai_api_key = os.getenv("OPENAI_API_KEY", "your-openai-api-key")

# define deployments
deployments = []
for name in ("east", "west", "south"):
    deployments.append(
        DeploymentConfig(
            name=name,
            api_base=api_base,  # can reuse since the implementation doesn't know
            api_key=api_key,
            models=[Model(name="gpt-4o-mini", tpm=10000, rpm=60)],
        )
    )

fallback = DeploymentConfig(
    name="openai",
    api_base="",  # gets populated by AsyncOpenAI automatically
    api_key=openai_api_key,
    models=[Model(name="gpt-4o-mini", tpm=10000, rpm=60)],
)


@asynccontextmanager
async def get_switchboard():
    """Use a pattern analogous to FastAPI dependency
    injection for automatic cleanup.
    """

    # Create Switchboard and start background tasks
    switchboard = Switchboard(deployments=deployments, fallback=fallback)
    switchboard.start()

    try:
        yield switchboard
    finally:
        await switchboard.stop()


async def basic_functionality(switchboard: Switchboard):
    # Make a completion request (non-streaming)
    response = await switchboard.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": "Hello, world!"}],
    )

    print("completion:", response.choices[0].message.content)

    # Make a streaming completion request
    stream = await switchboard.create(
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

    # First message will select a random healthy
    # deployment and associate it with the session_id
    _ = await switchboard.create(
        session_id=session_id,
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": "Who won the World Series in 2020?"}],
    )

    # Follow-up requests with the same session_id will route to the same deployment
    _ = await switchboard.create(
        session_id=session_id,
        model="gpt-4o-mini",
        messages=[
            {"role": "user", "content": "Who won the World Series in 2020?"},
            {
                "role": "assistant",
                "content": "The Los Angeles Dodgers won the World Series in 2020.",
            },
            {"role": "user", "content": "Who did they beat?"},
        ],
    )

    # If the deployment becomes unhealthy,
    # requests will fall back to a healthy one

    # Simulate a failure by marking down the deployment
    original_client = switchboard.select_deployment(
        model="gpt-4o-mini", session_id=session_id
    )
    print("original client:", original_client)
    original_client.models["gpt-4o-mini"].cooldown()

    # A new deployment will be selected for this session_id
    _ = await switchboard.create(
        session_id=session_id,
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": "Who won the World Series in 2021?"}],
    )

    new_client = switchboard.select_deployment(
        model="gpt-4o-mini", session_id=session_id
    )
    print("new client:", new_client)
    assert new_client != original_client


async def main():
    async with get_switchboard() as sb:
        print("Basic functionality:")
        await basic_functionality(sb)

        print("Session affinity (should warn):")
        await session_affinity(sb)


if __name__ == "__main__":
    asyncio.run(main())
