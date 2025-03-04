from unittest.mock import AsyncMock

import pytest
from test_utils import BASIC_CHAT_COMPLETION_ARGS, TEST_DEPLOYMENTS

from switchboard import Client, Switchboard


@pytest.fixture
def mock_switchboard():
    mock_client = AsyncMock()
    return Switchboard(
        TEST_DEPLOYMENTS,
        client_factory=lambda x: Client(x, mock_client),
        healthcheck_interval=0,  # disable healthchecks
        usage_reset_interval=0,  # disable usage resets
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
