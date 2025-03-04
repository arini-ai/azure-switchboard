from unittest.mock import AsyncMock

import pytest
from test_client import (
    TEST_DEPLOYMENT_1,
    TEST_DEPLOYMENT_2,
    TEST_DEPLOYMENT_3,
)

from switchboard import Client, Switchboard

BASIC_CHAT_COMPLETION_ARGS = {
    "model": "gpt-4o-mini",
    "messages": [{"role": "user", "content": "Hello, world!"}],
}


@pytest.fixture
def mock_switchboard():
    mock_client = AsyncMock()
    return Switchboard(
        [TEST_DEPLOYMENT_1, TEST_DEPLOYMENT_2, TEST_DEPLOYMENT_3],
        client_factory=lambda x: Client(x, mock_client),
        healthcheck_interval=0,  # disable healthchecks
        ratelimit_window=0,  # disable usage resets
    )


async def test_switchboard_basic_selection(mock_switchboard: Switchboard):
    # test that we select a deployment
    client = mock_switchboard.select_deployment()
    assert client.name in mock_switchboard.deployments

    # test that we can select a specific deployment
    client = mock_switchboard.select_deployment(session_id="test2")
    assert client.name == "test2"


async def test_switchboard_completion(mock_switchboard: Switchboard):
    mock_client = mock_switchboard.deployments["test1"]

    # test basic chat completion
    chat_completion_mock = AsyncMock()
    mock_client.client.chat.completions.create = chat_completion_mock
    await mock_switchboard.completion(**BASIC_CHAT_COMPLETION_ARGS)

    chat_completion_mock.assert_called_once()
    assert mock_client.ratelimit_tokens == 30
    assert mock_client.ratelimit_requests == 1
    assert mock_client.healthy
