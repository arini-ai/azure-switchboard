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

from azure_switchboard import AzureDeployment, Model, OpenAIDeployment, Switchboard

azure_openai_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
azure_openai_api_key = os.getenv("AZURE_OPENAI_API_KEY")
openai_api_key = os.getenv("OPENAI_API_KEY", None)

deployments = []
if azure_openai_endpoint and azure_openai_api_key:
    # create 3 deployments. reusing the endpoint
    # is fine for the purposes of this demo
    for name in ("east", "west", "south"):
        deployments.append(
            AzureDeployment(
                name=name,
                endpoint=azure_openai_endpoint,
                api_key=azure_openai_api_key,
                models=[Model(name="gpt-4o-mini")],
            )
        )

if openai_api_key:
    # we can use openai as a fallback deployment
    # it will pick up the api key from the environment
    deployments.append(OpenAIDeployment())


async def main():
    async with Switchboard(deployments=deployments) as sb:
        print("Basic functionality:")
        await basic_functionality(sb)

        print("Session affinity (should warn):")
        await session_affinity(sb)


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
    r = await switchboard.create(
        session_id=session_id,
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": "Who won the World Series in 2020?"}],
    )

    d1 = switchboard.select_deployment(model="gpt-4o-mini", session_id=session_id)
    print("deployment 1:", d1)
    print("response 1:", r.choices[0].message.content)

    # Follow-up requests with the same session_id will route to the same deployment
    r2 = await switchboard.create(
        session_id=session_id,
        model="gpt-4o-mini",
        messages=[
            {"role": "user", "content": "Who won the World Series in 2020?"},
            {"role": "assistant", "content": r.choices[0].message.content},
            {"role": "user", "content": "Who did they beat?"},
        ],
    )

    print("response 2:", r2.choices[0].message.content)

    # Simulate a failure by marking down the deployment
    d1.models["gpt-4o-mini"].cooldown()

    # A new deployment will be selected for this session_id
    r3 = await switchboard.create(
        session_id=session_id,
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": "Who won the World Series in 2021?"}],
    )

    d2 = switchboard.select_deployment(model="gpt-4o-mini", session_id=session_id)
    print("deployment 2:", d2)
    print("response 3:", r3.choices[0].message.content)
    assert d2 != d1


if __name__ == "__main__":
    asyncio.run(main())
