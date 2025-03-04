from contextlib import asynccontextmanager
from typing import Any, AsyncGenerator, Generator
from unittest.mock import AsyncMock

import pytest
from fixtures import (
    BASIC_CHAT_COMPLETION_ARGS,
    MOCK_COMPLETION,
    MOCK_STREAM_CHUNKS,
    TEST_DEPLOYMENT_1,
)
from openai.types.chat import ChatCompletionChunk

from switchboard import Client


@pytest.fixture
def mock_client() -> Client:
    client_mock = AsyncMock()
    return Client(TEST_DEPLOYMENT_1, client=client_mock)


async def test_client_healthcheck(mock_client: Client):
    # test basic healthcheck
    mock_client.client.models.list = AsyncMock()
    await mock_client.check_health()
    assert mock_client.client.models.list.call_count == 1
    assert mock_client.healthy

    # test basic healthcheck failure
    mock_client.client.models.list.side_effect = Exception("test")
    # don't raise the exception, just mark the client as unhealthy
    await mock_client.check_health()
    assert not mock_client.healthy
    assert mock_client.client.models.list.call_count == 2

    # test healthcheck allows recovery
    mock_client.client.models.list.side_effect = None
    await mock_client.check_health()
    assert mock_client.client.models.list.call_count == 3
    assert mock_client.healthy


async def test_client_completion(mock_client: Client):
    # Create a mock response with usage information

    # test basic chat completion
    chat_completion_mock = AsyncMock(return_value=MOCK_COMPLETION)
    mock_client.client.chat.completions.create = chat_completion_mock
    response = await mock_client.completion(**BASIC_CHAT_COMPLETION_ARGS)
    assert chat_completion_mock.call_count == 1
    assert response == MOCK_COMPLETION

    # Check that token usage was updated
    assert mock_client.ratelimit_tokens == 30
    assert mock_client.ratelimit_requests == 1

    # test that we handle exceptions properly
    chat_completion_mock.side_effect = Exception("test")
    with pytest.raises(Exception, match="test"):
        await mock_client.completion(**BASIC_CHAT_COMPLETION_ARGS)
    assert chat_completion_mock.call_count == 2

    assert mock_client.ratelimit_tokens == 30
    assert mock_client.ratelimit_requests == 2


async def _collect_chunks(stream) -> tuple[list[ChatCompletionChunk], str]:
    received_chunks = []
    content = ""
    async for chunk in stream:
        received_chunks.append(chunk)
        if chunk.choices and chunk.choices[0].delta.content:
            content += chunk.choices[0].delta.content
    return received_chunks, content


async def test_client_stream(mock_client: Client):
    # Create a mock async generator for streaming

    async def mock_stream() -> AsyncGenerator[ChatCompletionChunk, None]:
        for chunk in MOCK_STREAM_CHUNKS:
            yield chunk

    # Set up the mock to return our async generator
    chat_completion_mock = AsyncMock(return_value=mock_stream())
    mock_client.client.chat.completions.create = chat_completion_mock

    # Test basic streaming - stream() returns an async generator, not an awaitable
    stream = mock_client.stream(**BASIC_CHAT_COMPLETION_ARGS)
    assert stream is not None

    # Collect all chunks to verify content
    received_chunks, content = await _collect_chunks(stream)

    # Verify the stream options were set correctly
    chat_completion_mock.assert_called_once()
    call_kwargs = chat_completion_mock.call_args.kwargs
    assert call_kwargs.get("stream") is True
    assert call_kwargs.get("stream_options", {}).get("include_usage") is True

    # Verify we received all chunks
    assert len(received_chunks) == len(MOCK_STREAM_CHUNKS)

    # Verify the content was assembled correctly
    assert content == "Hello, world!"

    # Verify the token usage was tracked correctly
    assert mock_client.ratelimit_tokens == 20  # From the last chunk's usage
    assert mock_client.ratelimit_requests == 1

    # Test exception handling
    chat_completion_mock.side_effect = Exception("test")
    with pytest.raises(Exception, match="test"):
        stream = mock_client.stream(**BASIC_CHAT_COMPLETION_ARGS)
        async for chunk in stream:
            pass
    assert chat_completion_mock.call_count == 2
    # Request count should increment even on failure
    assert mock_client.ratelimit_requests == 2


async def test_client_counters(mock_client: Client):
    # Reset counters
    mock_client.reset_counters()

    # Verify counters were reset
    assert mock_client.ratelimit_tokens == 0
    assert mock_client.ratelimit_requests == 0

    # Set some initial usage values
    mock_client.ratelimit_tokens = 100
    mock_client.ratelimit_requests = 5

    # Reset counters
    counters = mock_client.get_counters()
    assert counters["tokens"] == 100
    assert counters["requests"] == 5

    mock_client.reset_counters()

    # Verify counters were reset
    assert mock_client.ratelimit_tokens == 0
    assert mock_client.ratelimit_requests == 0

    # Verify last_reset was updated
    assert mock_client.last_reset > 0


async def test_client_util(mock_client: Client):
    mock_client.reset_counters()

    # Get initial utilization (nonzero bc random splay)
    initial_util = mock_client.util
    assert 0 < initial_util < 0.02

    # Test with some token usage
    mock_client.ratelimit_tokens = 500  # 50% of TPM limit (1000)
    util_with_tokens = mock_client.util
    assert 0.5 <= util_with_tokens < 0.52  # 50% plus small random factor

    # Test with some request usage
    mock_client.ratelimit_tokens = 0
    mock_client.ratelimit_requests = 3  # 50% of RPM limit (6)
    util_with_requests = mock_client.util
    assert 0.5 <= util_with_requests < 0.52  # 50% plus small random factor

    # Test with both token and request usage (should take max)
    mock_client.ratelimit_tokens = 600  # 60% of TPM
    mock_client.ratelimit_requests = 3  # 50% of RPM
    util_with_both = mock_client.util
    assert 0.6 <= util_with_both < 0.62  # 60% (max) plus small random factor

    # Test with unhealthy client (should be infinity)
    mock_client._last_request_status = False
    assert mock_client.util == float("inf")
