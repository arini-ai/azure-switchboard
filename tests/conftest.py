from typing import Any, Literal
from unittest.mock import AsyncMock

import pytest
import respx
from openai import AsyncStream
from openai.types.chat import ChatCompletion, ChatCompletionChunk
from openai.types.chat.chat_completion_chunk import Choice, ChoiceDelta
from openai.types.completion_usage import (
    CompletionTokensDetails,
    CompletionUsage,
    PromptTokensDetails,
)

from azure_switchboard import (
    AzureDeployment,
    Deployment,
    Model,
    OpenAIDeployment,
    Switchboard,
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


def openai_config() -> OpenAIDeployment:
    return OpenAIDeployment(
        api_key="test",
        models=[Model(name="gpt-4o-mini"), Model(name="gpt-4o")],
    )


def azure_config(name: str, model_names: list[str] = ["gpt-4o-mini", "gpt-4o"]) -> AzureDeployment:
    return AzureDeployment(
        name=name,
        endpoint=f"https://{name}.openai.azure.com/",
        api_key=name,
        models=[Model(name=model_name, tpm=10000, rpm=60) for model_name in model_names],
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
def mock_client(request: pytest.FixtureRequest):
    with respx.mock() as respx_mock:
        if provided_models := request.node.get_closest_marker("mock_models"):
            # Add routes for each model
            for model in provided_models.args:
                if model == "openai":
                    path = "/v1/chat/completions"
                else:
                    path = f"/openai/deployments/{model}/chat/completions"

                respx_mock.route(name=model, method="POST", path=path).respond(
                    json=COMPLETION_RESPONSE_JSON
                )

        yield respx_mock


@pytest.fixture
def model():
    return Model(name="gpt-4o-mini", tpm=1000, rpm=6)


@pytest.fixture
def deployment():
    return Deployment(azure_config("test1"))


@pytest.fixture
async def switchboard():
    deployments = [
        azure_config("test1"),
        azure_config("test2"),
        azure_config("test3"),
    ]
    async with Switchboard(deployments=deployments, ratelimit_window=0) as sb:
        yield sb


COMPLETION_PARAMS: dict[Literal["model", "messages"], Any] = {
    "model": "gpt-4o-mini",
    "messages": [{"role": "user", "content": "Hello, world!"}],
}

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
