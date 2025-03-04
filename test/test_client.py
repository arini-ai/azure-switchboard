from unittest.mock import AsyncMock

import pytest
from openai.types.chat import ChatCompletion, ChatCompletionChunk
from openai.types.chat.chat_completion import Choice
from openai.types.chat.chat_completion_chunk import Choice as ChunkChoice
from openai.types.chat.chat_completion_chunk import ChoiceDelta
from openai.types.chat.chat_completion_message import ChatCompletionMessage
from openai.types.completion_usage import CompletionUsage
from test_utils import BASIC_CHAT_COMPLETION_ARGS, TEST_DEPLOYMENT_1

from switchboard import Client

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
    assert mock_client.healthy

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


async def test_client_basic_stream(mock_client):
    # Create a mock async generator for streaming
    async def mock_stream():
        for chunk in MOCK_STREAM_CHUNKS:
            yield chunk

    # Set up the mock to return our async generator
    chat_completion_mock = AsyncMock(return_value=mock_stream())
    mock_client.client.chat.completions.create = chat_completion_mock

    # Reset counters for clean test
    await mock_client.reset_counters()

    # Test basic streaming - stream() returns an async generator, not an awaitable
    stream = mock_client.stream(**BASIC_CHAT_COMPLETION_ARGS)
    assert stream is not None

    # Collect all chunks to verify content
    received_chunks = []
    content = ""
    async for chunk in stream:
        received_chunks.append(chunk)
        if chunk.choices and chunk.choices[0].delta.content:
            content += chunk.choices[0].delta.content

    # Verify we received all chunks
    assert len(received_chunks) == len(MOCK_STREAM_CHUNKS)

    # Verify the content was assembled correctly
    assert content == "Hello, world!"

    # Verify the token usage was tracked correctly
    assert mock_client.ratelimit_tokens == 20  # From the last chunk's usage
    assert mock_client.ratelimit_requests == 1

    # Verify the stream options were set correctly
    chat_completion_mock.assert_called_once()
    call_kwargs = chat_completion_mock.call_args.kwargs
    assert call_kwargs.get("stream") is True
    assert call_kwargs.get("stream_options", {}).get("include_usage") is True

    # Test exception handling
    chat_completion_mock.side_effect = Exception("test")
    with pytest.raises(Exception, match="test"):
        stream = mock_client.stream(**BASIC_CHAT_COMPLETION_ARGS)
        async for chunk in stream:
            pass
    assert chat_completion_mock.call_count == 2
    assert (
        mock_client.ratelimit_requests == 2
    )  # Request count should increment even on failure


async def test_client_reset_counters(mock_client):
    # Set some initial usage values
    mock_client.ratelimit_tokens = 100
    mock_client.ratelimit_requests = 5

    # Reset counters
    await mock_client.reset_counters()

    # Verify counters were reset
    assert mock_client.ratelimit_tokens == 0
    assert mock_client.ratelimit_requests == 0

    # Verify last_reset was updated
    assert mock_client.last_reset > 0


async def test_client_utilization(mock_client):
    # Test with healthy client and no usage
    mock_client.healthy = True
    mock_client.ratelimit_tokens = 0
    mock_client.ratelimit_requests = 0

    # Get initial utilization (should be close to 0 plus small random factor)
    initial_util = mock_client.utilization
    assert 0 <= initial_util < 0.02  # Small random factor is 0-0.01

    # Test with some token usage
    mock_client.ratelimit_tokens = 50  # 50% of TPM limit (100)
    util_with_tokens = mock_client.utilization
    assert 0.5 <= util_with_tokens < 0.52  # 50% plus small random factor

    # Test with some request usage
    mock_client.ratelimit_tokens = 0
    mock_client.ratelimit_requests = 50  # 50% of RPM limit (100)
    util_with_requests = mock_client.utilization
    assert 0.5 <= util_with_requests < 0.52  # 50% plus small random factor

    # Test with both token and request usage (should take max)
    mock_client.ratelimit_tokens = 30  # 30% of TPM
    mock_client.ratelimit_requests = 70  # 70% of RPM
    util_with_both = mock_client.utilization
    assert 0.7 <= util_with_both < 0.72  # 70% (max) plus small random factor

    # Test with unhealthy client (should be infinity)
    mock_client.healthy = False
    assert mock_client.utilization == float("inf")


def test_client_str_representation(mock_client):
    # Set some values to test the string representation
    mock_client.name = "test_client"
    mock_client.healthy = True
    mock_client.ratelimit_tokens = 100
    mock_client.ratelimit_requests = 50

    # Get the string representation
    client_str = str(mock_client)

    # Verify all important information is included
    assert "test_client" in client_str
    assert "healthy=True" in client_str
    assert "tokens=100" in client_str
    assert "requests=50" in client_str
    assert "utilization=" in client_str  # Exact value will vary due to random factor

    # Test with unhealthy client
    mock_client.healthy = False
    client_str = str(mock_client)
    assert "healthy=False" in client_str
