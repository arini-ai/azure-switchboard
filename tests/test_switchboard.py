import asyncio

import pytest
from fixtures import (
    BASIC_CHAT_COMPLETION_ARGS,
    MOCK_COMPLETION,
    TEST_DEPLOYMENT_1,
    TEST_DEPLOYMENT_2,
    TEST_DEPLOYMENT_3,
)
from utils import BaseTestCase, create_mock_openai_client

from azure_switchboard import Client, Switchboard


@pytest.fixture
def mock_switchboard() -> Switchboard:
    """Create a Switchboard with a shared underlying client."""
    mock_openai = create_mock_openai_client()
    return Switchboard(
        [TEST_DEPLOYMENT_1, TEST_DEPLOYMENT_2, TEST_DEPLOYMENT_3],
        client_factory=lambda x: Client(x, mock_openai),
        healthcheck_interval=0,  # disable healthchecks
        ratelimit_window=0,  # disable usage resets
    )


@pytest.fixture
def mock_multi_switchboard() -> Switchboard:
    """Create a Switchboard with separate mock clients for each deployment."""

    switchboard = Switchboard(
        [TEST_DEPLOYMENT_1, TEST_DEPLOYMENT_2, TEST_DEPLOYMENT_3],
        client_factory=lambda x: Client(x, create_mock_openai_client()),
        healthcheck_interval=0,
        ratelimit_window=0,
    )

    return switchboard


class TestSwitchboard(BaseTestCase):
    """Basic switchboard functionality tests."""

    def _get_client(self, switchboard) -> Client:
        return next(iter(switchboard.deployments.values()))

    async def test_completion(self, mock_switchboard):
        """Test basic chat completion through switchboard."""
        response = await mock_switchboard.create(**BASIC_CHAT_COMPLETION_ARGS)
        self._get_client(
            mock_switchboard
        ).client.chat.completions.create.assert_called_once()
        assert response == MOCK_COMPLETION

    async def test_streaming(self, mock_switchboard):
        """Test streaming through switchboard."""

        stream = await mock_switchboard.create(
            stream=True, **BASIC_CHAT_COMPLETION_ARGS
        )
        _, content = await self.collect_chunks(stream)

        self._get_client(
            mock_switchboard
        ).client.chat.completions.create.assert_called_once()
        assert content == "Hello, world!"

    async def test_selection(self, mock_switchboard):
        """Test basic deployment selection."""
        client = mock_switchboard.select_deployment()
        assert client.config.name in mock_switchboard.deployments

    async def test_failover(self, mock_switchboard):
        """Test proper handling when multiple deployments fail in sequence."""
        # Get list of deployments
        deployments = list(mock_switchboard.deployments.values())
        assert len(deployments) == 3, "Need exactly 3 deployments for this test"

        # Initial request should work
        response = await mock_switchboard.create(**BASIC_CHAT_COMPLETION_ARGS)
        assert response == MOCK_COMPLETION

        # Mark first deployment as unhealthy
        deployments[0].cooldown()
        response = await mock_switchboard.create(**BASIC_CHAT_COMPLETION_ARGS)
        assert response == MOCK_COMPLETION

        # Mark second deployment as unhealthy
        deployments[1].cooldown()
        response = await mock_switchboard.create(**BASIC_CHAT_COMPLETION_ARGS)
        assert response == MOCK_COMPLETION

        # Mark last deployment as unhealthy
        deployments[2].cooldown()
        with pytest.raises(Exception):
            await mock_switchboard.create(**BASIC_CHAT_COMPLETION_ARGS)

        # Restore first deployment
        deployments[0].reset_cooldown()
        response = await mock_switchboard.create(**BASIC_CHAT_COMPLETION_ARGS)
        assert response == MOCK_COMPLETION

    async def test_session_stickiness(self, mock_switchboard):
        """Test session stickiness and failover."""
        # Test consistent deployment selection for session
        client_1 = mock_switchboard.select_deployment(session_id="test")
        client_2 = mock_switchboard.select_deployment(session_id="test")
        assert client_1.config.name == client_2.config.name

        # Test failover when selected deployment is unhealthy
        mock_switchboard.deployments[client_1.config.name].cooldown()
        client_3 = mock_switchboard.select_deployment(session_id="test")
        assert client_3.config.name != client_1.config.name

        # Test session maintains failover assignment
        client_4 = mock_switchboard.select_deployment(session_id="test")
        assert client_4.config.name == client_3.config.name

    async def test_session_stickiness_failover(self, mock_switchboard):
        """Test session affinity when preferred deployment becomes unavailable."""
        session_id = "test-session-123"

        # Initial request establishes session affinity
        response1 = await mock_switchboard.create(
            session_id=session_id, **BASIC_CHAT_COMPLETION_ARGS
        )
        assert response1 == MOCK_COMPLETION

        # Get assigned deployment
        assigned_client = mock_switchboard._sessions[session_id]
        original_deployment = assigned_client.config.name

        # Verify session stickiness
        response2 = await mock_switchboard.create(
            session_id=session_id, **BASIC_CHAT_COMPLETION_ARGS
        )
        assert response2 == MOCK_COMPLETION
        assert mock_switchboard._sessions[session_id].config.name == original_deployment

        # Make assigned deployment unhealthy
        mock_switchboard.deployments[original_deployment].cooldown()

        # Verify failover
        response3 = await mock_switchboard.create(
            session_id=session_id, **BASIC_CHAT_COMPLETION_ARGS
        )
        assert response3 == MOCK_COMPLETION
        assert mock_switchboard._sessions[session_id].config.name != original_deployment

        # Verify session maintains new assignment
        fallback_deployment = mock_switchboard._sessions[session_id].config.name
        response4 = await mock_switchboard.create(
            session_id=session_id, **BASIC_CHAT_COMPLETION_ARGS
        )
        assert response4 == MOCK_COMPLETION
        assert mock_switchboard._sessions[session_id].config.name == fallback_deployment

    def _within_bounds(self, val, min, max, tolerance=0.05):
        """Check if a value is within bounds, accounting for tolerance."""
        return min <= val <= max or min * (1 - tolerance) <= val <= max * (
            1 + tolerance
        )

    async def test_load_distribution(self, mock_switchboard):
        """Test that load is distributed across healthy deployments."""
        # Make 100 requests
        for _ in range(100):
            await mock_switchboard.create(**BASIC_CHAT_COMPLETION_ARGS)

        # Verify all deployments were used
        for client in mock_switchboard.deployments.values():
            assert self._within_bounds(
                val=client.ratelimit_requests,
                min=25,
                max=40,
            )

    async def test_load_distribution_health_awareness(self, mock_switchboard):
        """Test load distribution when some deployments are unhealthy."""
        # Mark one deployment as unhealthy
        mock_switchboard.deployments["test2"].cooldown()

        # Make 100 requests
        for _ in range(100):
            await mock_switchboard.create(**BASIC_CHAT_COMPLETION_ARGS)

        # Verify distribution
        assert self._within_bounds(
            val=mock_switchboard.deployments["test1"].ratelimit_requests,
            min=40,
            max=60,
        )
        assert self._within_bounds(
            val=mock_switchboard.deployments["test2"].ratelimit_requests,
            min=0,
            max=0,
        )
        assert self._within_bounds(
            val=mock_switchboard.deployments["test3"].ratelimit_requests,
            min=40,
            max=60,
        )

    async def test_load_distribution_utilization_awareness(self, mock_switchboard):
        """Selection should prefer to load deployments with lower utilization."""

        # Make 100 requests to preload the deployments, should be evenly distributed
        for _ in range(100):
            await mock_switchboard.create(**BASIC_CHAT_COMPLETION_ARGS)

        # reset utilization of one deployment
        client = mock_switchboard.select_deployment()
        client.reset_counters()

        # make another 100 requests
        for _ in range(100):
            await mock_switchboard.create(**BASIC_CHAT_COMPLETION_ARGS)

        # verify the load distribution is still roughly even
        # (ie, we preferred to send requests to the underutilized deployment)
        for client in mock_switchboard.deployments.values():
            assert self._within_bounds(
                val=client.ratelimit_requests,
                min=60,
                max=70,
                tolerance=0.1,
            )

    async def test_load_distribution_session_stickiness(self, mock_switchboard):
        """Test that session stickiness works correctly with load distribution."""
        session_ids = ["session1", "session2", "session3", "session4", "session5"]

        # Make 50 requests total (10 per session ID)
        requests = []
        for _ in range(10):
            for session_id in session_ids:
                requests.append(
                    mock_switchboard.create(
                        session_id=session_id, **BASIC_CHAT_COMPLETION_ARGS
                    )
                )

        await asyncio.gather(*requests)

        # Check distribution (should be uneven due to session stickiness)
        request_counts = sorted(
            [
                client.ratelimit_requests
                for client in mock_switchboard.deployments.values()
            ]
        )
        assert sum(request_counts) == 50
        assert request_counts == [10, 20, 20], (
            "5 sessions into 3 deployments should create 1:2:2 distribution"
        )
