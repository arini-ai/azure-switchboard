from unittest.mock import AsyncMock

import pytest
from openai.types.chat import ChatCompletion
from openai.types.chat.chat_completion import Choice
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
    chat_completion_mock = AsyncMock()
    mock_client.client.chat.completions.create = chat_completion_mock

    # test basic streaming
    stream = mock_client.stream(**BASIC_CHAT_COMPLETION_ARGS)
    assert stream is not None

    # test that we can iterate over the stream
    async for chunk in stream:
        assert chunk is not None

    assert chat_completion_mock.call_count == 1

    # test that we handle exceptions properly
    chat_completion_mock.side_effect = Exception("test")
    with pytest.raises(Exception, match="test"):
        stream = mock_client.stream(**BASIC_CHAT_COMPLETION_ARGS)
        async for chunk in stream:
            assert chunk is not None
    assert chat_completion_mock.call_count == 2
