from unittest.mock import AsyncMock

from openai import AsyncStream
from openai.types.chat import ChatCompletionChunk

from azure_switchboard import AzureDeployment, Model, OpenAIDeployment

from .fixtures import MOCK_COMPLETION, MOCK_STREAM_CHUNKS


def chat_completion_mock():
    """Basic mock that replicates openai client chat completion behavior."""

    async def _stream(items: list):
        for item in items:
            yield item

    def side_effect(*args, **kwargs):
        if "stream" in kwargs:
            return _stream(MOCK_STREAM_CHUNKS)
        return MOCK_COMPLETION

    return AsyncMock(side_effect=side_effect)


def azure_config(name: str) -> AzureDeployment:
    return AzureDeployment(
        name=name,
        endpoint=f"https://{name}.openai.azure.com/",
        api_key=name,
        models=[
            Model(name="gpt-4o-mini", tpm=10000, rpm=60),
            Model(name="gpt-4o", tpm=10000, rpm=60),
        ],
    )


def openai_config() -> OpenAIDeployment:
    return OpenAIDeployment(
        api_key="test",
        models=[Model(name="gpt-4o-mini"), Model(name="gpt-4o")],
    )


class BaseTestCase:
    """Base class for test cases with common utilities."""

    basic_args = {
        "model": "gpt-4o-mini",
        "messages": [{"role": "user", "content": "Hello, world!"}],
    }

    @staticmethod
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
