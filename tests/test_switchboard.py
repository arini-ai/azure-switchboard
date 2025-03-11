import asyncio

import pytest
from fixtures import (
    MOCK_COMPLETION,
    TEST_DEPLOYMENT_1,
    TEST_DEPLOYMENT_2,
    TEST_DEPLOYMENT_3,
)
from utils import BaseTestCase, create_mock_openai_client

from azure_switchboard import Deployment, Switchboard, SwitchboardError


@pytest.fixture
def mock_switchboard():
    """Create a Switchboard with a shared underlying client."""
    try:
        mock_openai = create_mock_openai_client()
        switchboard = Switchboard(
            [TEST_DEPLOYMENT_1, TEST_DEPLOYMENT_2, TEST_DEPLOYMENT_3],
            client_factory=lambda x: Deployment(x, mock_openai),
            ratelimit_window=0,  # disable usage resets
        )
        yield switchboard
    finally:
        switchboard.reset_usage()
        for client in switchboard.deployments.values():
            for model in client.models.values():
                model.reset_cooldown()


class TestSwitchboard(BaseTestCase):
    """Basic switchboard functionality tests."""

    def _get_deployment(self, switchboard: Switchboard) -> Deployment:
        return next(iter(switchboard.deployments.values()))

    async def test_repr(self, mock_switchboard: Switchboard):
        sb_repr = repr(mock_switchboard)
        assert sb_repr.startswith("Switchboard")

    async def test_completion(self, mock_switchboard: Switchboard):
        """Test basic chat completion through switchboard."""

        response = await mock_switchboard.create(**self.basic_args)
        deployment = self._get_deployment(mock_switchboard)
        deployment.client.chat.completions.create.assert_called_once()
        assert response == MOCK_COMPLETION

    async def test_streaming(self, mock_switchboard: Switchboard):
        """Test streaming through switchboard."""

        stream = await mock_switchboard.create(stream=True, **self.basic_args)
        _, content = await self.collect_chunks(stream)

        deployment = self._get_deployment(mock_switchboard)
        deployment.client.chat.completions.create.assert_called_once()
        assert content == "Hello, world!"

    async def test_selection(self, mock_switchboard: Switchboard):
        """Test basic selection invariants"""
        client = mock_switchboard.select_deployment(model="gpt-4o-mini")
        assert client.config.name in mock_switchboard.deployments

        deployments = list(mock_switchboard.deployments.values())
        assert len(deployments) == 3, "Need exactly 3 deployments for this test"

        # Initial request should work
        response = await mock_switchboard.create(**self.basic_args)
        assert response == MOCK_COMPLETION

        # Mark first deployment as unhealthy
        deployments[0].models["gpt-4o-mini"].cooldown()
        response = await mock_switchboard.create(**self.basic_args)
        assert response == MOCK_COMPLETION

        # Mark second deployment as unhealthy
        deployments[1].models["gpt-4o-mini"].cooldown()
        response = await mock_switchboard.create(**self.basic_args)
        assert response == MOCK_COMPLETION

        # Mark last deployment as unhealthy
        deployments[2].models["gpt-4o-mini"].cooldown()
        with pytest.raises(SwitchboardError):
            await mock_switchboard.create(**self.basic_args)

        # Restore first deployment
        deployments[0].models["gpt-4o-mini"].reset_cooldown()
        response = await mock_switchboard.create(**self.basic_args)
        assert response == MOCK_COMPLETION

    async def test_session_stickiness(self, mock_switchboard: Switchboard) -> None:
        """Test session stickiness and failover."""

        # Test consistent deployment selection for session
        client_1 = mock_switchboard.select_deployment(
            session_id="test", model="gpt-4o-mini"
        )
        client_2 = mock_switchboard.select_deployment(
            session_id="test", model="gpt-4o-mini"
        )
        assert client_1.config.name == client_2.config.name

        # Test failover when selected deployment is unhealthy
        client_1.models["gpt-4o-mini"].cooldown()
        client_3 = mock_switchboard.select_deployment(
            session_id="test", model="gpt-4o-mini"
        )
        assert client_3.config.name != client_1.config.name

        # Test session maintains failover assignment
        client_4 = mock_switchboard.select_deployment(
            session_id="test", model="gpt-4o-mini"
        )
        assert client_4.config.name == client_3.config.name

    async def test_session_stickiness_failover(self, mock_switchboard: Switchboard):
        """Test session affinity when preferred deployment becomes unavailable."""

        session_id = "test"

        # Initial request establishes session affinity
        response1 = await mock_switchboard.create(
            session_id=session_id, **self.basic_args
        )
        assert response1 == MOCK_COMPLETION

        # Get assigned deployment
        assigned_deployment = mock_switchboard._sessions[session_id]
        original_deployment = assigned_deployment

        # Verify session stickiness
        response2 = await mock_switchboard.create(
            session_id=session_id, **self.basic_args
        )
        assert response2 == MOCK_COMPLETION
        assert mock_switchboard._sessions[session_id] == original_deployment

        # Make assigned deployment unhealthy
        model = original_deployment.models["gpt-4o-mini"]
        model.cooldown()

        # Verify failover
        response3 = await mock_switchboard.create(
            session_id=session_id, **self.basic_args
        )
        assert response3 == MOCK_COMPLETION
        assert mock_switchboard._sessions[session_id] != original_deployment

        # Verify session maintains new assignment
        fallback_deployment = mock_switchboard._sessions[session_id]
        response4 = await mock_switchboard.create(
            session_id=session_id, **self.basic_args
        )
        assert response4 == MOCK_COMPLETION
        assert mock_switchboard._sessions[session_id] == fallback_deployment

    def _within_bounds(self, val, min, max, tolerance=0.05):
        """Check if a value is within bounds, accounting for tolerance."""
        return min <= val <= max or min * (1 - tolerance) <= val <= max * (
            1 + tolerance
        )

    async def test_load_distribution(self, mock_switchboard: Switchboard):
        """Test that load is distributed across healthy deployments."""

        # Make 100 requests
        for _ in range(100):
            await mock_switchboard.create(**self.basic_args)

        # Verify all deployments were used
        for client in mock_switchboard.deployments.values():
            assert self._within_bounds(
                val=client.models["gpt-4o-mini"]._rpm_usage,
                min=25,
                max=40,
            )

    async def test_load_distribution_health_awareness(
        self, mock_switchboard: Switchboard
    ):
        """Test load distribution when some deployments are unhealthy."""

        # Mark one deployment as unhealthy
        mock_switchboard.deployments["test2"].models["gpt-4o-mini"].cooldown()

        # Make 100 requests
        for _ in range(100):
            await mock_switchboard.create(**self.basic_args)

        # Verify distribution
        assert self._within_bounds(
            val=mock_switchboard.deployments["test1"].models["gpt-4o-mini"]._rpm_usage,
            min=40,
            max=60,
        )
        assert self._within_bounds(
            val=mock_switchboard.deployments["test2"].models["gpt-4o-mini"]._rpm_usage,
            min=0,
            max=0,
        )
        assert self._within_bounds(
            val=mock_switchboard.deployments["test3"].models["gpt-4o-mini"]._rpm_usage,
            min=40,
            max=60,
        )

    async def test_load_distribution_utilization_awareness(
        self, mock_switchboard: Switchboard
    ):
        """Selection should prefer to load deployments with lower utilization."""

        # Make 100 requests to preload the deployments, should be evenly distributed
        for _ in range(100):
            await mock_switchboard.create(**self.basic_args)

        # reset utilization of one deployment
        client = mock_switchboard.select_deployment(model="gpt-4o-mini")
        client.reset_usage()

        # make another 100 requests
        for _ in range(100):
            await mock_switchboard.create(**self.basic_args)

        # verify the load distribution is still roughly even
        # (ie, we preferred to send requests to the underutilized deployment)
        for client in mock_switchboard.deployments.values():
            assert self._within_bounds(
                val=client.models["gpt-4o-mini"]._rpm_usage,
                min=60,
                max=70,
                tolerance=0.1,
            )

    async def test_load_distribution_session_stickiness(
        self, mock_switchboard: Switchboard
    ):
        """Test that session stickiness works correctly with load distribution."""

        session_ids = ["session1", "session2", "session3", "session4", "session5"]

        # Make 50 requests total (10 per session ID)
        requests = []
        for _ in range(10):
            for session_id in session_ids:
                requests.append(
                    mock_switchboard.create(session_id=session_id, **self.basic_args)
                )

        await asyncio.gather(*requests)

        # Check distribution (should be uneven due to session stickiness)
        request_counts = sorted(
            [
                client.models["gpt-4o-mini"]._rpm_usage
                for client in mock_switchboard.deployments.values()
            ]
        )
        assert sum(request_counts) == 50
        assert request_counts == [10, 20, 20], (
            "5 sessions into 3 deployments should create 1:2:2 distribution"
        )

    async def test_ratelimit_reset(self, mock_switchboard: Switchboard):
        """Test that the ratelimit is reset correctly."""

        # disabled upstream so restart it for the purpose
        # of this test
        mock_switchboard.ratelimit_window = 1
        mock_switchboard.start()

        # make some requests to add usage
        for _ in range(10):
            await mock_switchboard.create(**self.basic_args)

        usage = mock_switchboard.get_usage()

        # wait for the ratelimit to reset
        await asyncio.sleep(2)

        usage2 = mock_switchboard.get_usage()

        assert usage2 != usage

        await mock_switchboard.stop()
        mock_switchboard.ratelimit_window = 0
