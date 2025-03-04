from unittest.mock import AsyncMock

import pytest
from openai.types.chat import ChatCompletion, ChatCompletionChunk
from openai.types.chat.chat_completion import Choice
from openai.types.chat.chat_completion_chunk import Choice as ChunkChoice
from openai.types.chat.chat_completion_chunk import ChoiceDelta
from openai.types.chat.chat_completion_message import ChatCompletionMessage
from openai.types.completion_usage import CompletionUsage

from switchboard import Client, Deployment

TEST_DEPLOYMENT_1 = Deployment(
    name="test1",
    api_base="https://test1.openai.azure.com/",
    api_key="test1",
    tpm_ratelimit=100,
    rpm_ratelimit=100,
)

TEST_DEPLOYMENT_2 = Deployment(
    name="test2",
    api_base="https://test2.openai.azure.com/",
    api_key="test2",
    tpm_ratelimit=200,
    rpm_ratelimit=200,
)
TEST_DEPLOYMENT_3 = Deployment(
    name="test3",
    api_base="https://test3.openai.azure.com/",
    api_key="test3",
    tpm_ratelimit=300,
    rpm_ratelimit=300,
)

MOCK_COMPLETION = ChatCompletion(
    id="test",
    choices=[
        Choice(
            finish_reason="stop",
            index=0,
            message=ChatCompletionMessage(
                content="Hello, world!",
                role="assistant",
            ),
        )
    ],
    created=1234567890,
    model="gpt-4o-mini",
    object="chat.completion",
    usage=CompletionUsage(
        completion_tokens=10,
        prompt_tokens=20,
        total_tokens=30,
    ),
)

# Create mock streaming chunks for testing
MOCK_STREAM_CHUNKS = [
    ChatCompletionChunk(
        id="test_chunk_1",
        choices=[
            ChunkChoice(
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
            ChunkChoice(
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
            ChunkChoice(
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
            ChunkChoice(
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
        ),
    ),
]


@pytest.fixture
def mock_client():
    client_mock = AsyncMock()
    return Client(TEST_DEPLOYMENT_1, client=client_mock)


async def test_client_basic(mock_client):
    # test basic liveness check
    mock_client.client.models.list = AsyncMock()
    await mock_client.check_liveness()
    assert mock_client.client.models.list.call_count == 1
    assert mock_client.healthy

    # test basic liveness check failure
    mock_client.client.models.list.side_effect = Exception("test")
    # don't raise the exception, just mark the client as unhealthy
    await mock_client.check_liveness()
    assert not mock_client.healthy
    assert mock_client.client.models.list.call_count == 2

    # assert liveness check allows recovery
    mock_client.client.models.list.side_effect = None
    await mock_client.check_liveness()
    assert mock_client.client.models.list.call_count == 3
    assert mock_client.healthy


BASIC_CHAT_COMPLETION_ARGS = {
    "model": "gpt-4o-mini",
    "messages": [{"role": "user", "content": "Hello, world!"}],
}


async def test_client_basic_completion(mock_client):
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


async def _collect_chunks(stream):
    received_chunks = []
    content = ""
    async for chunk in stream:
        received_chunks.append(chunk)
        if chunk.choices and chunk.choices[0].delta.content:
            content += chunk.choices[0].delta.content
    return received_chunks, content


async def test_client_basic_stream(mock_client):
    # Create a mock async generator for streaming
    async def mock_stream():
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


async def test_client_reset_counters(mock_client):
    # Set some initial usage values
    mock_client.ratelimit_tokens = 100
    mock_client.ratelimit_requests = 5

    # Reset counters
    mock_client.reset_counters()

    # Verify counters were reset
    assert mock_client.ratelimit_tokens == 0
    assert mock_client.ratelimit_requests == 0

    # Verify last_reset was updated
    assert mock_client.last_reset > 0


async def test_client_utilization(mock_client):
    mock_client.reset_counters()

    # Get initial utilization (nonzero bc random splay)
    initial_util = mock_client.util
    assert 0 < initial_util < 0.02

    # Test with some token usage
    mock_client.ratelimit_tokens = 50  # 50% of TPM limit (100)
    util_with_tokens = mock_client.util
    assert 0.5 <= util_with_tokens < 0.52  # 50% plus small random factor

    # Test with some request usage
    mock_client.ratelimit_tokens = 0
    mock_client.ratelimit_requests = 50  # 50% of RPM limit (100)
    util_with_requests = mock_client.util
    assert 0.5 <= util_with_requests < 0.52  # 50% plus small random factor

    # Test with both token and request usage (should take max)
    mock_client.ratelimit_tokens = 30  # 30% of TPM
    mock_client.ratelimit_requests = 70  # 70% of RPM
    util_with_both = mock_client.util
    assert 0.7 <= util_with_both < 0.72  # 70% (max) plus small random factor

    # Test with unhealthy client (should be infinity)
    mock_client._last_request_status = False
    assert mock_client.util == float("inf")
