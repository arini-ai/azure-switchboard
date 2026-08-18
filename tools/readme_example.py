#!/usr/bin/env python3
#
# To run this, use:
#   uv run --env-file .env tools/readme_example.py
#
# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "azure-switchboard",
# ]
# ///

import asyncio
import os

from azure_switchboard import Foundry, OpenAIDeployment, Switchboard

azure_openai_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
azure_openai_api_key = os.getenv("AZURE_OPENAI_API_KEY")
openai_api_key = os.getenv("OPENAI_API_KEY")

foundries = []
if azure_openai_endpoint and azure_openai_api_key:
    # create 3 foundries. reusing the endpoint
    # is fine for the purposes of this demo
    for name in ("east", "west", "south"):
        foundries.append(
            Foundry(
                name=name,
                api_key=azure_openai_api_key,
                models=[
                    OpenAIDeployment(
                        name="gpt-4o-mini",
                        endpoint=f"{azure_openai_endpoint}/openai/v1/",
                    )
                ],
            )
        )

if not foundries and not openai_api_key:
    raise RuntimeError(
        "Set AZURE_OPENAI_ENDPOINT/AZURE_OPENAI_API_KEY or OPENAI_API_KEY to run this example."
    )


async def main():
    # OPENAI_API_KEY, if set, backs the pool as a last resort
    async with Switchboard(
        foundries=foundries, openai_fallback=bool(openai_api_key)
    ) as sb:
        print("Basic functionality:")
        await basic_functionality(sb)

        print("Session affinity (should warn):")
        await session_affinity(sb)


async def basic_functionality(switchboard: Switchboard):
    # Make a completion request (non-streaming)
    response = await switchboard.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": "Hello, world!"}],
    )

    print("completion:", response.choices[0].message.content)

    # Make a streaming completion request
    stream = await switchboard.chat.completions.create(
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

    # First message will select a random healthy deployment
    # and pin the session_id to the foundry hosting it
    r = await switchboard.chat.completions.create(
        session_id=session_id,
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": "Who won the World Series in 2020?"}],
    )

    # the session is now pinned to whichever foundry served it
    f1 = switchboard.sessions[session_id]
    print("foundry 1:", f1.name)
    print("response 1:", r.choices[0].message.content)

    # Follow-up requests with the same session_id will route to the same foundry
    r2 = await switchboard.chat.completions.create(
        session_id=session_id,
        model="gpt-4o-mini",
        messages=[
            {"role": "user", "content": "Who won the World Series in 2020?"},
            {"role": "assistant", "content": r.choices[0].message.content},
            {"role": "user", "content": "Who did they beat?"},
        ],
    )

    print("response 2:", r2.choices[0].message.content)

    # Simulate a failure by marking down the deployment that served us
    f1.models["gpt-4o-mini"].mark_down()

    # A new foundry will be selected for this session_id
    r3 = await switchboard.chat.completions.create(
        session_id=session_id,
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": "Who won the World Series in 2021?"}],
    )

    f2 = switchboard.sessions[session_id]
    print("foundry 2:", f2.name)
    print("response 3:", r3.choices[0].message.content)
    assert f2 is not f1


if __name__ == "__main__":
    asyncio.run(main())
