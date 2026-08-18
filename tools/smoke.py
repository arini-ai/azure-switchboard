#!/usr/bin/env python3
#
# Live wiring check against a real Foundry resource. Nothing is mocked.
#
# To run this, use:
#   just smoke
#
# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "azure-switchboard",
# ]
# ///

import asyncio
import os
import sys

from pydantic import BaseModel

from azure_switchboard import (
    AnthropicDeployment,
    Foundry,
    OpenAIDeployment,
    Switchboard,
    SwitchboardError,
)

CHAT_MODEL = os.getenv("SMOKE_CHAT_MODEL", "gpt-5.4-mini")
MESSAGES_MODEL = os.getenv("SMOKE_MESSAGES_MODEL", "claude-sonnet-5")

PROMPT = [{"role": "user", "content": "Reply with the single word: pong"}]

passed: list[str] = []
failed: list[tuple[str, str]] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    print(f"  {'PASS' if ok else 'FAIL'}  {name}{f' — {detail}' if detail else ''}")
    (passed.append(name) if ok else failed.append((name, detail)))


class Weather(BaseModel):
    city: str
    celsius: float


def build() -> Switchboard:
    """One Foundry hosting both APIs — the thing the old model could not express."""
    resource = os.environ["AZURE_FOUNDRY"]
    return Switchboard(
        foundries=[
            Foundry(
                name=resource,
                api_key=os.environ["AZURE_API_KEY"],
                models=[
                    OpenAIDeployment(CHAT_MODEL, tpm=100_000, rpm=600),
                    AnthropicDeployment(MESSAGES_MODEL, tpm=100_000, rpm=600),
                ],
            )
        ],
        openai_fallback=bool(os.getenv("OPENAI_API_KEY")),
        anthropic_fallback=bool(os.getenv("ANTHROPIC_API_KEY")),
        ratelimit_window=0,
    )


async def chat_completions(sb: Switchboard) -> None:
    print("\nchat.completions (Azure)")

    r = await sb.chat.completions.create(model=CHAT_MODEL, messages=PROMPT)
    text = r.choices[0].message.content or ""
    check("create", "pong" in text.lower(), text.strip()[:40])

    stream = await sb.chat.completions.create(
        model=CHAT_MODEL, messages=PROMPT, stream=True
    )
    streamed = ""
    async for chunk in stream:
        # a chunk may carry no choices at all (usage-only) or a null delta
        delta = chunk.choices[0].delta if chunk.choices else None
        if delta and delta.content:
            streamed += delta.content
    check("create(stream=True)", "pong" in streamed.lower(), streamed.strip()[:40])

    parsed = await sb.chat.completions.parse(
        model=CHAT_MODEL,
        messages=[{"role": "user", "content": "It is 18.5C in Paris."}],
        response_format=Weather,
    )
    got = parsed.choices[0].message.parsed
    check("parse", isinstance(got, Weather) and got.city == "Paris", str(got))


async def messages(sb: Switchboard) -> None:
    print("\nmessages (Azure)")

    r = await sb.messages.create(model=MESSAGES_MODEL, max_tokens=64, messages=PROMPT)
    text = "".join(b.text for b in r.content if b.type == "text")
    check("create", "pong" in text.lower(), text.strip()[:40])

    stream = await sb.messages.create(
        model=MESSAGES_MODEL, max_tokens=64, messages=PROMPT, stream=True
    )
    streamed = ""
    async for event in stream:
        if event.type == "content_block_delta" and event.delta.type == "text_delta":
            streamed += event.delta.text
    check("create(stream=True)", "pong" in streamed.lower(), streamed.strip()[:40])


async def stream_helpers(sb: Switchboard) -> None:
    """The .stream() surfaces, driven exactly as the SDKs' own are."""
    print("\nstream() helpers")

    async with sb.messages.stream(
        model=MESSAGES_MODEL, max_tokens=64, messages=PROMPT
    ) as s:
        events = 0
        async for _ in s:
            events += 1
        final = await s.get_final_message()
    text = "".join(b.text for b in final.content if b.type == "text")
    check(
        "messages.stream + get_final_message", "pong" in text.lower(), text.strip()[:40]
    )
    check("messages.stream yielded events", events > 0, f"{events} events")

    # the case a per-event tap would miss: never iterate, just await the result
    async with sb.messages.stream(
        model=MESSAGES_MODEL, max_tokens=64, messages=PROMPT
    ) as s:
        before = sb.foundries[os.environ["AZURE_FOUNDRY"]].models[MESSAGES_MODEL]
        spent_before = before.tpm_usage
        final = await s.get_final_message()
    spent_after = before.tpm_usage
    text = "".join(b.text for b in final.content if b.type == "text")
    check(
        "messages.stream without iterating", "pong" in text.lower(), text.strip()[:40]
    )
    check(
        "usage charged without iterating",
        spent_after > spent_before,
        f"{spent_before} -> {spent_after}",
    )

    async with sb.chat.completions.stream(model=CHAT_MODEL, messages=PROMPT) as s:
        completion = await s.get_final_completion()
    text = completion.choices[0].message.content or ""
    check(
        "chat.stream + get_final_completion", "pong" in text.lower(), text.strip()[:40]
    )


async def usage_tracking(sb: Switchboard) -> None:
    print("\nutilization")
    resource = os.environ["AZURE_FOUNDRY"]
    stats = sb.stats()[resource]
    check(
        "chat deployment spent tokens",
        not stats[CHAT_MODEL].tpm.startswith("0/"),
        stats[CHAT_MODEL].tpm,
    )
    check(
        "messages deployment spent tokens",
        not stats[MESSAGES_MODEL].tpm.startswith("0/"),
        stats[MESSAGES_MODEL].tpm,
    )


async def one_resource_two_clients(sb: Switchboard) -> None:
    print("\none resource, two APIs")
    resource = sb.foundries[os.environ["AZURE_FOUNDRY"]]
    chat = resource.models[CHAT_MODEL]
    msgs = resource.models[MESSAGES_MODEL]
    check("shared credential", chat.resource is msgs.resource)
    check(
        "distinct endpoints",
        chat.url != msgs.url,
        f"{chat.url} vs {msgs.url}",
    )
    check(
        "distinct clients",
        type(chat.client).__name__ != type(msgs.client).__name__,
        f"{type(chat.client).__name__} / {type(msgs.client).__name__}",
    )


async def first_party_fallback(sb: Switchboard) -> None:
    """Cool the Azure resource and confirm the request still lands."""
    print("\nfirst-party fallback")
    resource = sb.foundries[os.environ["AZURE_FOUNDRY"]]
    resource.mark_down(60)

    if os.getenv("OPENAI_API_KEY"):
        chosen = sb.chat.completions._fallback(CHAT_MODEL)
        r = await sb.chat.completions.create(
            model=os.getenv("SMOKE_OPENAI_FALLBACK_MODEL", "gpt-4o-mini"),
            messages=PROMPT,
        )
        text = r.choices[0].message.content or ""
        check(
            "chat falls back to OpenAI",
            "pong" in text.lower(),
            f"via {chosen.resource.name if chosen else '?'}",
        )

    if os.getenv("ANTHROPIC_API_KEY"):
        chosen = sb.messages._fallback(MESSAGES_MODEL)
        r = await sb.messages.create(
            model=os.getenv("SMOKE_ANTHROPIC_FALLBACK_MODEL", "claude-sonnet-4-5"),
            max_tokens=64,
            messages=PROMPT,
        )
        text = "".join(b.text for b in r.content if b.type == "text")
        check(
            "messages falls back to Anthropic",
            "pong" in text.lower(),
            f"via {chosen.resource.name if chosen else '?'}",
        )

    resource.mark_up()


async def no_fallback_raises() -> None:
    print("\nexhausted pool without a fallback")
    resource = Foundry(
        name=os.environ["AZURE_FOUNDRY"],
        api_key=os.environ["AZURE_API_KEY"],
        models=[OpenAIDeployment(CHAT_MODEL)],
    )
    sb = Switchboard(foundries=[resource], ratelimit_window=0)
    resource.models[CHAT_MODEL].mark_down(60)
    try:
        await sb.chat.completions.create(model=CHAT_MODEL, messages=PROMPT)
        check("raises SwitchboardError", False, "call unexpectedly succeeded")
    except SwitchboardError as e:
        check("raises SwitchboardError", "No deployments available" in str(e), str(e))


async def main() -> int:
    missing = [v for v in ("AZURE_FOUNDRY", "AZURE_API_KEY") if not os.getenv(v)]
    if missing:
        print(f"set {', '.join(missing)} to run this")
        return 2

    print(
        f"resource={os.environ['AZURE_FOUNDRY']} chat={CHAT_MODEL} messages={MESSAGES_MODEL}"
    )

    async with build() as sb:
        await one_resource_two_clients(sb)
        await chat_completions(sb)
        await messages(sb)
        await stream_helpers(sb)
        await usage_tracking(sb)
        await first_party_fallback(sb)

    await no_fallback_raises()

    print(f"\n{len(passed)} passed, {len(failed)} failed")
    for name, detail in failed:
        print(f"  FAIL {name}: {detail}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
