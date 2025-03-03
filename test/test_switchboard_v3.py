import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from openai import AsyncAzureOpenAI

from switchboard.switchboard_v3 import AzureDeployment, Client, Switchboard

TEST_DEPLOYMENT_1 = AzureDeployment(
    name="test1",
    api_base="https://test1.openai.azure.com/",
    api_key="test1",
    max_tpm=100,
    max_rpm=100,
)

TEST_DEPLOYMENT_2 = AzureDeployment(
    name="test2",
    api_base="https://test2.openai.azure.com/",
    api_key="test2",
    max_tpm=200,
    max_rpm=200,
)

TEST_DEPLOYMENT_3 = AzureDeployment(
    name="test3",
    api_base="https://test3.openai.azure.com/",
    api_key="test3",
    max_tpm=300,
    max_rpm=300,
)

TEST_DEPLOYMENTS = [TEST_DEPLOYMENT_1, TEST_DEPLOYMENT_2, TEST_DEPLOYMENT_3]


@pytest.fixture
def mock_client():
    return AsyncMock()


BASIC_CHAT_COMPLETION_ARGS = {
    "model": "gpt-4o-mini",
    "messages": [{"role": "user", "content": "Hello, world!"}],
}


@pytest.mark.asyncio
async def test_client_basic():
    mock_client = AsyncMock()
    client = Client(TEST_DEPLOYMENT_1, client=mock_client)
    assert client.healthy

    # test basic liveness check
    mock_client.models.list = AsyncMock()
    await client.check_liveness()
    assert mock_client.models.list.call_count == 1
    assert client.healthy

    # test basic liveness check failure
    mock_client.models.list.side_effect = Exception("test")
    # don't raise the exception, just mark the client as unhealthy
    await client.check_liveness()
    assert not client.healthy
    assert mock_client.models.list.call_count == 2

    # assert liveness check allows recovery
    mock_client.models.list.side_effect = None
    await client.check_liveness()
    assert mock_client.models.list.call_count == 3
    assert client.healthy

    # test basic chat completion
    chat_completion_mock = AsyncMock()
    mock_client.chat.completions.create = chat_completion_mock
    _ = await client.chat_completion(**BASIC_CHAT_COMPLETION_ARGS)
    assert chat_completion_mock.call_count == 1

    # test that passing arbitrary kwargs works
    _ = await client.chat_completion(**BASIC_CHAT_COMPLETION_ARGS, stream=True)
    chat_completion_mock.assert_called_with(
        **BASIC_CHAT_COMPLETION_ARGS, stream=True, timeout=TEST_DEPLOYMENT_1.timeout
    )

    # test that we mark unhealthy if the client raises an exception
    chat_completion_mock.side_effect = Exception("test")
    with pytest.raises(Exception, match="test"):
        _ = await client.chat_completion(**BASIC_CHAT_COMPLETION_ARGS)
    assert not client.healthy
    assert chat_completion_mock.call_count == 3
