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
async def test_client_basic(mock_client):
    client = Client(TEST_DEPLOYMENT_1, mock_client)
    assert client.healthy

    # test basic liveness check
    mock_client.models.list = AsyncMock()
    await client.check_liveness()
    mock_client.models.list.assert_called_once()

    # test basic chat completion
    mock_client.chat.completions.create = AsyncMock()
    _ = await client.chat_completion(**BASIC_CHAT_COMPLETION_ARGS)
    mock_client.chat.completions.create.assert_called()

    # test that passing arbitrary kwargs works
    _ = await client.chat_completion(**BASIC_CHAT_COMPLETION_ARGS, stream=True)
    mock_client.chat.completions.create.assert_called_with(
        **BASIC_CHAT_COMPLETION_ARGS, stream=True, timeout=TEST_DEPLOYMENT_1.timeout
    )

    # test that we mark unhealthy if the client raises an exception
    mock_client.chat.completions.create.side_effect = Exception("test")
    with pytest.raises(Exception, match="test"):
        _ = await client.chat_completion(**BASIC_CHAT_COMPLETION_ARGS)
    assert not client.healthy
