from unittest.mock import AsyncMock

from fixtures import MOCK_COMPLETION, MOCK_STREAM_CHUNKS
from openai import AsyncStream
from openai.types.chat import ChatCompletionChunk


def create_mock_openai_client() -> AsyncMock:
    """Create a basic mock client that returns MOCK_COMPLETION."""
    mock = AsyncMock()
    mock.models.list = AsyncMock()

    async def _stream(items: list):
        for item in items:
            yield item

    def side_effect(*args, **kwargs):
        if "stream" in kwargs:
            return _stream(MOCK_STREAM_CHUNKS)
        return MOCK_COMPLETION

    mock.chat.completions.create = AsyncMock(side_effect=side_effect)
    return mock


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
