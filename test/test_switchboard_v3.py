import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from openai import AsyncAzureOpenAI

from switchboard.switchboard_v3 import Client, Deployment, Switchboard

TEST_DEPLOYMENT_1 = Deployment(
    name="test1",
    api_base="https://test1.openai.azure.com/",
    api_key="test1",
    max_tpm=100,
    max_rpm=100,
)

TEST_DEPLOYMENT_2 = Deployment(
    name="test2",
    api_base="https://test2.openai.azure.com/",
    api_key="test2",
    max_tpm=200,
    max_rpm=200,
)

TEST_DEPLOYMENT_3 = Deployment(
    name="test3",
    api_base="https://test3.openai.azure.com/",
    api_key="test3",
    max_tpm=300,
    max_rpm=300,
)

TEST_DEPLOYMENTS = [TEST_DEPLOYMENT_1, TEST_DEPLOYMENT_2, TEST_DEPLOYMENT_3]


@pytest.fixture
def mock_client():
    return Client(TEST_DEPLOYMENT_1, client=AsyncMock())


BASIC_CHAT_COMPLETION_ARGS = {
    "model": "gpt-4o-mini",
    "messages": [{"role": "user", "content": "Hello, world!"}],
}


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


async def test_client_chat_completion(mock_client):
    # test basic chat completion
    chat_completion_mock = AsyncMock()
    mock_client.client.chat.completions.create = chat_completion_mock
    _ = await mock_client.chat_completion(**BASIC_CHAT_COMPLETION_ARGS)
    assert chat_completion_mock.call_count == 1

    # test that passing arbitrary kwargs works
    _ = await mock_client.chat_completion(**BASIC_CHAT_COMPLETION_ARGS, stream=True)
    chat_completion_mock.assert_called_with(
        **BASIC_CHAT_COMPLETION_ARGS, stream=True, timeout=TEST_DEPLOYMENT_1.timeout
    )

    # test that we mark unhealthy if the client raises an exception
    chat_completion_mock.side_effect = Exception("test")
    with pytest.raises(Exception, match="test"):
        _ = await mock_client.chat_completion(**BASIC_CHAT_COMPLETION_ARGS)
    assert chat_completion_mock.call_count == 3


@pytest.fixture
def mock_switchboard():
    mock_client = AsyncMock()
    return Switchboard(
        TEST_DEPLOYMENTS,
        client_factory=lambda x: Client(x, mock_client),
        healthcheck_interval=0,  # disable healthchecks
    )


async def test_switchboard_basic(mock_switchboard: Switchboard):
    # test that we select a deployment
    client = mock_switchboard.select_deployment()
    assert client.name in (t.name for t in TEST_DEPLOYMENTS)

    # test that we can select a specific deployment
    client = mock_switchboard.select_deployment(session_id="test2")
    assert client.name == "test2"

    # test basic chat completion
    await mock_switchboard.completion(**BASIC_CHAT_COMPLETION_ARGS)
