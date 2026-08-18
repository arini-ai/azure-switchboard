from typing import Any, Literal, cast
from unittest.mock import AsyncMock

import pytest
import respx
from openai import AsyncStream
from openai.types.chat import ChatCompletion, ChatCompletionChunk, ParsedChatCompletion
from openai.types.chat.chat_completion_chunk import Choice, ChoiceDelta
from openai.types.chat.parsed_chat_completion import (
    ParsedChoice,
    ParsedChatCompletionMessage,
)
from openai.types.completion_usage import (
    CompletionTokensDetails,
    CompletionUsage,
    PromptTokensDetails,
)
from pydantic import BaseModel as PydanticBaseModel

from anthropic.types import (
    Message as AnthropicMessage,
    MessageDeltaUsage,
    RawContentBlockDeltaEvent,
    RawContentBlockStartEvent,
    RawContentBlockStopEvent,
    RawMessageDeltaEvent,
    RawMessageStartEvent,
    RawMessageStopEvent,
    TextBlock,
    TextDelta,
)
from anthropic.types.raw_message_delta_event import Delta

from azure_switchboard import (
    AnthropicDeployment,
    Foundry,
    OpenAIDeployment,
    Switchboard,
)


def select_openai(
    sb: Switchboard, *, model: str, session_id: str | None = None
) -> OpenAIDeployment:
    """Resolve a model the way a chat request would, for selection assertions.

    Switchboard exposes no public selection entry point: the surfaces mirror
    their SDKs, and neither SDK has one. Tests reach for the internals rather
    than the package carrying a method for their benefit.
    """
    surface = sb.chat.completions
    return sb._select(
        surface._pool,
        model=model,
        session_id=session_id,
        fallback=lambda: surface._fallback(model),
    )


def select_anthropic(
    sb: Switchboard, *, model: str, session_id: str | None = None
) -> AnthropicDeployment:
    """Resolve a model the way a messages request would."""
    surface = sb.messages
    return sb._select(
        surface._pool,
        model=model,
        session_id=session_id,
        fallback=lambda: surface._fallback(model),
    )


async def collect_chunks(
    stream: AsyncStream[ChatCompletionChunk],
) -> tuple[list[ChatCompletionChunk], str]:
    """Collect all chunks from a stream and return the chunks and assembled content."""
    received_chunks = []
    content = ""
    async for chunk in stream:
        received_chunks.append(chunk)
        if chunk.choices and chunk.choices[0].delta.content:
            content += chunk.choices[0].delta.content
    return received_chunks, content


def openai_foundry(name: str) -> Foundry:
    """A Foundry resource hosting the Chat Completions models used in tests."""
    return Foundry(
        name=name,
        api_key=name,
        models=[
            OpenAIDeployment(name="gpt-4o-mini", tpm=10000, rpm=60),
            OpenAIDeployment(name="gpt-4o", tpm=10000, rpm=60),
        ],
    )


def chat_completion_mock():
    """Basic mock that replicates openai client chat completion behavior."""

    async def _stream(items: list):
        for item in items:
            yield item

    def side_effect(*args, **kwargs):
        if "stream" in kwargs:
            return _stream(COMPLETION_STREAM_CHUNKS)
        return COMPLETION_RESPONSE

    return AsyncMock(side_effect=side_effect)


@pytest.fixture(autouse=True)
def first_party_credentials(monkeypatch: pytest.MonkeyPatch):
    """Both SDKs read their key from the environment when falling back."""
    monkeypatch.setenv("OPENAI_API_KEY", "test-openai-key")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-anthropic-key")


@pytest.fixture(autouse=True)
def mock_client(request: pytest.FixtureRequest):
    with respx.mock(assert_all_called=False) as respx_mock:
        if request.node.get_closest_marker("mock_models"):
            # Azure deployments use /openai/v1/chat/completions
            respx_mock.route(
                name="azure", method="POST", path="/openai/v1/chat/completions"
            ).respond(json=COMPLETION_RESPONSE_JSON)
            # OpenAI deployments use /v1/chat/completions
            respx_mock.route(
                name="openai", method="POST", path="/v1/chat/completions"
            ).respond(json=COMPLETION_RESPONSE_JSON)

        yield respx_mock


@pytest.fixture
def model():
    """An unbound deployment, for the utilization arithmetic on its own."""
    return OpenAIDeployment(name="gpt-4o-mini", tpm=1000, rpm=6)


@pytest.fixture
def foundry():
    return openai_foundry("test1")


@pytest.fixture
def deployment(foundry: Foundry) -> OpenAIDeployment:
    return cast(OpenAIDeployment, foundry.models["gpt-4o-mini"])


@pytest.fixture
async def switchboard():
    resources = [
        openai_foundry("test1"),
        openai_foundry("test2"),
        openai_foundry("test3"),
    ]
    async with Switchboard(resources=resources, ratelimit_window=0) as sb:
        yield sb


COMPLETION_PARAMS: dict[Literal["model", "messages"], Any] = {
    "model": "gpt-4o-mini",
    "messages": [{"role": "user", "content": "Hello, world!"}],
}

# A deployment is the model, so calls made directly against one carry no model=.
COMPLETION_BODY: dict[str, Any] = {"messages": COMPLETION_PARAMS["messages"]}

COMPLETION_STREAM_CHUNKS = [
    ChatCompletionChunk(
        id="test_chunk_1",
        choices=[
            Choice(
                delta=ChoiceDelta(content="Hello", role="assistant"),
                finish_reason=None,
                index=0,
            )
        ],
        created=1234567890,
        model="gpt-4o-mini",
        object="chat.completion.chunk",
        usage=None,
    ),
    ChatCompletionChunk(
        id="test_chunk_2",
        choices=[
            Choice(
                delta=ChoiceDelta(content=", "),
                finish_reason=None,
                index=0,
            )
        ],
        created=1234567890,
        model="gpt-4o-mini",
        object="chat.completion.chunk",
        usage=None,
    ),
    ChatCompletionChunk(
        id="test_chunk_3",
        choices=[
            Choice(
                delta=ChoiceDelta(content="world!"),
                finish_reason=None,
                index=0,
            )
        ],
        created=1234567890,
        model="gpt-4o-mini",
        object="chat.completion.chunk",
        usage=None,
    ),
    ChatCompletionChunk(
        id="test_chunk_4",
        choices=[
            Choice(
                delta=ChoiceDelta(),
                finish_reason="stop",
                index=0,
            )
        ],
        created=1234567890,
        model="gpt-4o-mini",
        object="chat.completion.chunk",
        usage=CompletionUsage(
            completion_tokens=5,
            prompt_tokens=15,
            total_tokens=20,
            completion_tokens_details=CompletionTokensDetails(reasoning_tokens=5),
            prompt_tokens_details=PromptTokensDetails(cached_tokens=15),
        ),
    ),
]


COMPLETION_RESPONSE_JSON = {
    "choices": [
        {
            "finish_reason": "stop",
            "index": 0,
            "logprobs": None,
            "message": {
                "content": "Hello! How can I assist you today?",
                "refusal": None,
                "role": "assistant",
            },
        }
    ],
    "created": 1741124380,
    "id": "chatcmpl-test",
    "model": "gpt-4o-mini",
    "object": "chat.completion",
    "service_tier": "default",
    "system_fingerprint": "fp_06737a9306",
    "usage": {
        "completion_tokens": 10,
        "completion_tokens_details": {
            "accepted_prediction_tokens": 0,
            "audio_tokens": 0,
            "reasoning_tokens": 5,
            "rejected_prediction_tokens": 0,
        },
        "prompt_tokens": 10,
        "prompt_tokens_details": {"audio_tokens": 0, "cached_tokens": 8},
        "total_tokens": 20,
    },
}

COMPLETION_RESPONSE = ChatCompletion.model_validate(COMPLETION_RESPONSE_JSON)


class WeatherResult(PydanticBaseModel):
    """Test Pydantic model for structured outputs."""

    city: str
    temperature: float
    unit: str


PARSED_COMPLETION_PARAMS = {
    "model": "gpt-4o-mini",
    "messages": [{"role": "user", "content": "What is the weather in Paris?"}],
    "response_format": WeatherResult,
}

PARSED_COMPLETION_BODY = {
    "messages": PARSED_COMPLETION_PARAMS["messages"],
    "response_format": WeatherResult,
}

PARSED_RESPONSE = ParsedChatCompletion[WeatherResult](
    id="chatcmpl-parsed-test",
    choices=[
        ParsedChoice[WeatherResult](
            finish_reason="stop",
            index=0,
            message=ParsedChatCompletionMessage[WeatherResult](
                content='{"city": "Paris", "temperature": 18.5, "unit": "celsius"}',
                role="assistant",
                parsed=WeatherResult(city="Paris", temperature=18.5, unit="celsius"),
                refusal=None,
            ),
        )
    ],
    created=1741124380,
    model="gpt-4o-mini",
    object="chat.completion",
    usage=CompletionUsage(
        completion_tokens=15,
        prompt_tokens=12,
        total_tokens=27,
        completion_tokens_details=CompletionTokensDetails(reasoning_tokens=0),
        prompt_tokens_details=PromptTokensDetails(cached_tokens=0),
    ),
)


# --- Anthropic Messages API fixtures ------------------------------------


def anthropic_foundry(name: str = "foundry") -> Foundry:
    """A Foundry resource hosting the Messages API models used in tests."""
    return Foundry(
        name=name,
        api_key=name,
        models=[
            AnthropicDeployment(name="claude-sonnet-5", tpm=10000, rpm=60),
            AnthropicDeployment(name="claude-haiku-4-5", tpm=10000, rpm=60),
        ],
    )


def message_mock():
    """Basic mock that replicates anthropic client messages behavior."""

    async def _stream(items: list):
        for item in items:
            yield item

    def side_effect(*args, **kwargs):
        if kwargs.get("stream"):
            return _stream(MESSAGE_STREAM_EVENTS)
        return MESSAGE_RESPONSE

    return AsyncMock(side_effect=side_effect)


@pytest.fixture
def anthropic_resource():
    return anthropic_foundry("test1")


@pytest.fixture
def anthropic_deployment(anthropic_resource: Foundry) -> AnthropicDeployment:
    return cast(AnthropicDeployment, anthropic_resource.models["claude-sonnet-5"])


@pytest.fixture
async def anthropic_switchboard():
    resources = [
        anthropic_foundry("test1"),
        anthropic_foundry("test2"),
        anthropic_foundry("test3"),
    ]
    async with Switchboard(resources=resources, ratelimit_window=0) as sb:
        yield sb


MESSAGE_PARAMS: dict[str, Any] = {
    "model": "claude-sonnet-5",
    "max_tokens": 1024,
    "messages": [{"role": "user", "content": "Hello, world!"}],
}

MESSAGE_BODY: dict[str, Any] = {
    "max_tokens": MESSAGE_PARAMS["max_tokens"],
    "messages": MESSAGE_PARAMS["messages"],
}

MESSAGE_RESPONSE_JSON = {
    "id": "msg_test",
    "type": "message",
    "role": "assistant",
    "model": "claude-sonnet-5",
    "content": [{"type": "text", "text": "Hello! How can I assist you today?"}],
    "stop_reason": "end_turn",
    "stop_sequence": None,
    "usage": {
        "input_tokens": 12,
        "output_tokens": 8,
        "cache_read_input_tokens": 4,
        "cache_creation_input_tokens": 0,
    },
}

MESSAGE_RESPONSE = AnthropicMessage.model_validate(MESSAGE_RESPONSE_JSON)

# Usage arrives in two places: input on message_start, cumulative output on
# each message_delta. Totals here: 12 input + 9 output = 21 tokens.
MESSAGE_STREAM_EVENTS = [
    RawMessageStartEvent(
        type="message_start",
        message=AnthropicMessage.model_validate(
            {
                **MESSAGE_RESPONSE_JSON,
                "content": [],
                "stop_reason": None,
                "usage": {
                    "input_tokens": 12,
                    "output_tokens": 0,
                    "cache_read_input_tokens": 4,
                    "cache_creation_input_tokens": 0,
                },
            }
        ),
    ),
    RawContentBlockStartEvent(
        type="content_block_start",
        index=0,
        content_block=TextBlock(type="text", text=""),
    ),
    RawContentBlockDeltaEvent(
        type="content_block_delta",
        index=0,
        delta=TextDelta(type="text_delta", text="Hello"),
    ),
    RawContentBlockDeltaEvent(
        type="content_block_delta",
        index=0,
        delta=TextDelta(type="text_delta", text=", world!"),
    ),
    RawContentBlockStopEvent(type="content_block_stop", index=0),
    RawMessageDeltaEvent(
        type="message_delta",
        delta=Delta(stop_reason="end_turn", stop_sequence=None),
        usage=MessageDeltaUsage(output_tokens=5),
    ),
    RawMessageDeltaEvent(
        type="message_delta",
        delta=Delta(stop_reason="end_turn", stop_sequence=None),
        usage=MessageDeltaUsage(output_tokens=9),
    ),
    RawMessageStopEvent(type="message_stop"),
]


async def collect_events(stream) -> tuple[list, str]:
    """Collect all events from a message stream and assemble the text."""
    received = []
    content = ""
    async for event in stream:
        received.append(event)
        if event.type == "content_block_delta" and event.delta.type == "text_delta":
            content += event.delta.text
    return received, content
