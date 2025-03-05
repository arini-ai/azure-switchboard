import asyncio
from unittest.mock import AsyncMock

import pytest
from fixtures import (
    BASIC_CHAT_COMPLETION_ARGS,
    MOCK_COMPLETION,
    MOCK_COMPLETION_PARSED,
    MOCK_COMPLETION_RAW,
    MOCK_STREAM_CHUNKS,
    TEST_DEPLOYMENT_1,
)

from switchboard import Client, Switchboard


@pytest.fixture
def mock_client():
    client_mock = AsyncMock()
    return Client(TEST_DEPLOYMENT_1, client=client_mock)


@pytest.fixture
def mock_switchboard(mock_client):
    from switchboard.client import Deployment

    # Create three test deployments with the same mock client
    deployments = [
        Deployment(
            name=f"test{i}",
            api_base=f"https://test{i}.openai.azure.com/",
            api_key=f"test{i}",
            tpm_ratelimit=1000,
            rpm_ratelimit=6,
        )
        for i in range(1, 4)
    ]

    # Use the mock client for all deployments
    switchboard = Switchboard(deployments)
    for deployment in switchboard.deployments.values():
        deployment.client = mock_client.client

    return switchboard


async def test_simple_cascading_failures(mock_switchboard: Switchboard, mock_client):
    """Test proper handling when multiple deployments fail in sequence."""
    # Get access to the deployments
    deployments = list(mock_switchboard.deployments.values())
    assert len(deployments) == 3, "Need exactly 3 deployments for this test"

    # Create separate mocks for each deployment to better track calls
    for i, deployment in enumerate(deployments):
        deployment.client.chat.completions.create = AsyncMock(
            return_value=MOCK_COMPLETION
        )
        deployment.reset_cooldown()
        deployment.reset_counters()

    # First make sure all deployments work
    response = await mock_switchboard.create(**BASIC_CHAT_COMPLETION_ARGS)
    assert response == MOCK_COMPLETION

    # Mark the first deployment as unhealthy
    deployments[0].cooldown(seconds=999)
    assert not deployments[0].healthy

    # Should still work (uses a different deployment)
    response = await mock_switchboard.create(**BASIC_CHAT_COMPLETION_ARGS)
    assert response == MOCK_COMPLETION

    # Mark the second deployment as unhealthy
    deployments[1].cooldown(seconds=999)
    assert not deployments[1].healthy

    # Should still work (only one healthy deployment left)
    response = await mock_switchboard.create(**BASIC_CHAT_COMPLETION_ARGS)
    assert response == MOCK_COMPLETION

    # Make the last deployment unhealthy
    deployments[2].cooldown(seconds=999)
    assert not deployments[2].healthy

    # Now all are unhealthy - should raise an exception
    with pytest.raises(Exception):
        await mock_switchboard.create(**BASIC_CHAT_COMPLETION_ARGS)

    # Restore the first deployment
    deployments[0].reset_cooldown()
    assert deployments[0].healthy

    # Should work again
    response = await mock_switchboard.create(**BASIC_CHAT_COMPLETION_ARGS)
    assert response == MOCK_COMPLETION
