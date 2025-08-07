import asyncio
from unittest.mock import patch

import pytest
import respx


from azure_switchboard import OpenAIDeployment, Switchboard, SwitchboardError

from .conftest import (
    COMPLETION_PARAMS,
    COMPLETION_RESPONSE,
    azure_config,
    chat_completion_mock,
    collect_chunks,
    openai_config,
)


class TestSwitchboard:
    """Basic switchboard functionality tests."""

    @pytest.mark.mock_models("gpt-4o-mini")
    async def test_completion(
        self, switchboard: Switchboard, mock_client: respx.MockRouter
    ):
        """Test chat completion through switchboard."""

        assert "Switchboard" in repr(switchboard)

        response = await switchboard.create(**COMPLETION_PARAMS)
        assert mock_client["gpt-4o-mini"].call_count == 1
        assert response == COMPLETION_RESPONSE

        assert any(
            filter(
                lambda d: d.get("gpt-4o-mini").rpm == "1/60",  # pyright: ignore[reportAttributeAccessIssue]
                switchboard.stats().values(),
            )
        )

    async def test_streaming(self, switchboard: Switchboard):
        """Test streaming through switchboard."""

        with patch("azure_switchboard.deployment.Deployment.create") as mock:
            mock.side_effect = chat_completion_mock()
            stream = await switchboard.create(stream=True, **COMPLETION_PARAMS)
            _, content = await collect_chunks(stream)

            assert mock.call_count == 1
            assert content == "Hello, world!"

    @pytest.mark.mock_models("gpt-4o-mini")
    async def test_selection(
        self, switchboard: Switchboard, mock_client: respx.MockRouter
    ):
        """Test basic selection invariants"""
        client = switchboard.select_deployment(model="gpt-4o-mini")
        assert client.config.name in switchboard.deployments

        deployments = list(switchboard.deployments.values())
        assert len(deployments) == 3, "Need exactly 3 deployments for this test"

        # Initial request should work
        response = await switchboard.create(**COMPLETION_PARAMS)
        assert response == COMPLETION_RESPONSE
        assert mock_client["gpt-4o-mini"].call_count == 1
        host_0 = mock_client["gpt-4o-mini"].calls.last.request.url.host

        # Mark first deployment as unhealthy
        deployments[0].models["gpt-4o-mini"].mark_down()
        response = await switchboard.create(**COMPLETION_PARAMS)
        assert response == COMPLETION_RESPONSE
        assert mock_client["gpt-4o-mini"].call_count == 2
        host_1 = mock_client["gpt-4o-mini"].calls.last.request.url.host

        # Mark second deployment as unhealthy
        deployments[1].models["gpt-4o-mini"].mark_down()
        response = await switchboard.create(**COMPLETION_PARAMS)
        assert response == COMPLETION_RESPONSE
        assert mock_client["gpt-4o-mini"].call_count == 3
        host_2 = mock_client["gpt-4o-mini"].calls.last.request.url.host

        # Mark last deployment as unhealthy
        deployments[2].models["gpt-4o-mini"].mark_down()
        with pytest.raises(
            SwitchboardError, match="No eligible deployments available for gpt-4o-mini"
        ):
            await switchboard.create(**COMPLETION_PARAMS)
        assert mock_client["gpt-4o-mini"].call_count == 3

        # Restore first deployment
        deployments[0].models["gpt-4o-mini"].mark_up()
        response = await switchboard.create(**COMPLETION_PARAMS)
        assert response == COMPLETION_RESPONSE
        assert mock_client["gpt-4o-mini"].call_count == 4
        host_3 = mock_client["gpt-4o-mini"].calls.last.request.url.host

        assert len(set([host_0, host_1, host_2, host_3])) > 1

    async def test_session_stickiness(self, switchboard: Switchboard) -> None:
        """Test session stickiness and failover."""

        # Test consistent deployment selection for session
        client_1 = switchboard.select_deployment(session_id="test", model="gpt-4o-mini")
        client_2 = switchboard.select_deployment(session_id="test", model="gpt-4o-mini")
        assert client_1.config.name == client_2.config.name

        # Test failover when selected deployment is unhealthy
        client_1.models["gpt-4o-mini"].mark_down()
        client_3 = switchboard.select_deployment(session_id="test", model="gpt-4o-mini")
        assert client_3.config.name != client_1.config.name

        # Test session maintains failover assignment
        client_4 = switchboard.select_deployment(session_id="test", model="gpt-4o-mini")
        assert client_4.config.name == client_3.config.name

    @pytest.mark.mock_models("gpt-4o-mini")
    async def test_session_stickiness_failover(
        self, switchboard: Switchboard, mock_client: respx.MockRouter
    ):
        """Test session affinity when preferred deployment becomes unavailable."""

        session_id = "test"

        # Initial request establishes session affinity
        response1 = await switchboard.create(session_id=session_id, **COMPLETION_PARAMS)
        assert response1 == COMPLETION_RESPONSE
        assert mock_client["gpt-4o-mini"].call_count == 1
        # Get assigned deployment
        assigned_deployment = switchboard.sessions[session_id]
        original_deployment = assigned_deployment

        # Verify session stickiness
        response2 = await switchboard.create(session_id=session_id, **COMPLETION_PARAMS)
        assert response2 == COMPLETION_RESPONSE
        assert switchboard.sessions[session_id] == original_deployment

        # Make assigned deployment unhealthy
        model = original_deployment.models["gpt-4o-mini"]
        model.mark_down()

        # Verify failover
        response3 = await switchboard.create(session_id=session_id, **COMPLETION_PARAMS)
        assert response3 == COMPLETION_RESPONSE
        assert switchboard.sessions[session_id] != original_deployment

        # Verify session maintains new assignment
        fallback_deployment = switchboard.sessions[session_id]
        response4 = await switchboard.create(session_id=session_id, **COMPLETION_PARAMS)
        assert response4 == COMPLETION_RESPONSE
        assert switchboard.sessions[session_id] == fallback_deployment

    @pytest.mark.mock_models("gpt-4o-mini", "openai")
    async def test_fallback_to_openai(self, mock_client: respx.MockRouter):
        """Test that the switchboard can fallback to OpenAI."""

        switchboard = Switchboard(deployments=[azure_config("test1"), openai_config()])

        assert switchboard.fallback is not None
        assert isinstance(switchboard.fallback.config, OpenAIDeployment)

        # basic test to verify the fallback works
        response = await switchboard.fallback.create(**COMPLETION_PARAMS)
        assert response == COMPLETION_RESPONSE
        assert mock_client["openai"].call_count == 1

        # default: use the healthy azure deployment
        response = await switchboard.create(**COMPLETION_PARAMS)
        assert response == COMPLETION_RESPONSE
        assert mock_client["gpt-4o-mini"].call_count == 1

        # make deployment unhealthy so it falls back to openai
        switchboard.deployments["test1"].models["gpt-4o-mini"].mark_down()
        await switchboard.create(**COMPLETION_PARAMS)
        assert mock_client["openai"].call_count == 2

        # make openai fallback unhealthy, verify it throws
        mock_client["openai"].side_effect = Exception("test")
        with pytest.raises(SwitchboardError, match="No eligible deployments available for gpt-4o-mini"):
            await switchboard.create(**COMPLETION_PARAMS)
        # 3 total additional calls were made, because openai retries twice internally
        assert mock_client["openai"].call_count == 5

        # bring the deployment back, verify we use it
        switchboard.deployments["test1"].models["gpt-4o-mini"].mark_up()
        await switchboard.create(**COMPLETION_PARAMS)
        assert mock_client["gpt-4o-mini"].call_count == 2

        # make everything unhealthy, verify it throws
        switchboard.deployments["test1"].models["gpt-4o-mini"].mark_down()
        with pytest.raises(SwitchboardError, match="No eligible deployments available for gpt-4o-mini"):
            await switchboard.create(**COMPLETION_PARAMS)
        # With the new implementation, no additional calls are made when all deployments are unhealthy
        assert mock_client["openai"].call_count == 5

        # reset fallback, verify it works
        mock_client["openai"].side_effect = None
        switchboard.fallback.models["gpt-4o-mini"].mark_up()
        await switchboard.create(**COMPLETION_PARAMS)
        assert mock_client["openai"].call_count == 6

        # reset the deployment and verify it gets used again
        switchboard.deployments["test1"].models["gpt-4o-mini"].mark_up()
        await switchboard.create(**COMPLETION_PARAMS)
        assert mock_client["gpt-4o-mini"].call_count == 3

    def _within_bounds(self, val, min, max, tolerance=0.05):
        """Check if a value is within bounds, accounting for tolerance."""
        return min <= val <= max or min * (1 - tolerance) <= val <= max * (
            1 + tolerance
        )

    @pytest.mark.mock_models("gpt-4o-mini")
    async def test_load_distribution(self, switchboard: Switchboard):
        """Test that load is distributed across healthy deployments."""

        # Make 100 requests
        await asyncio.gather(
            *[switchboard.create(**COMPLETION_PARAMS) for _ in range(100)]
        )

        # Verify all deployments were used
        for deployment in switchboard.deployments.values():
            assert self._within_bounds(
                val=deployment.models["gpt-4o-mini"].rpm_usage,
                min=25,
                max=40,
            )

    @pytest.mark.mock_models("gpt-4o-mini")
    async def test_load_distribution_health_awareness(self, switchboard: Switchboard):
        """Test load distribution when some deployments are unhealthy."""

        # Mark one deployment as unhealthy
        switchboard.deployments["test2"].models["gpt-4o-mini"].mark_down()

        # Make 100 requests
        for _ in range(100):
            await switchboard.create(**COMPLETION_PARAMS)

        # Verify distribution
        assert self._within_bounds(
            val=switchboard.deployments["test1"].models["gpt-4o-mini"].rpm_usage,
            min=40,
            max=60,
        )
        assert self._within_bounds(
            val=switchboard.deployments["test2"].models["gpt-4o-mini"].rpm_usage,
            min=0,
            max=0,
        )
        assert self._within_bounds(
            val=switchboard.deployments["test3"].models["gpt-4o-mini"].rpm_usage,
            min=40,
            max=60,
        )

    @pytest.mark.mock_models("gpt-4o-mini")
    async def test_load_distribution_utilization_awareness(
        self, switchboard: Switchboard
    ):
        """Selection should prefer to load deployments with lower utilization."""

        # Make 100 requests to preload the deployments, should be evenly distributed
        for _ in range(100):
            await switchboard.create(**COMPLETION_PARAMS)

        # reset utilization of one deployment
        client = switchboard.select_deployment(model="gpt-4o-mini")
        client.reset_usage()

        # make another 100 requests
        for _ in range(100):
            await switchboard.create(**COMPLETION_PARAMS)

        # verify the load distribution is still roughly even
        # (ie, we preferred to send requests to the underutilized deployment)
        for client in switchboard.deployments.values():
            assert self._within_bounds(
                val=client.models["gpt-4o-mini"].rpm_usage,
                min=60,
                max=70,
                tolerance=0.1,
            )

    @pytest.mark.mock_models("gpt-4o-mini")
    async def test_load_distribution_session_stickiness(self, switchboard: Switchboard):
        """Test that session stickiness works correctly with load distribution."""

        session_ids = ["1", "2", "3", "4", "5"]

        # Make 100 requests total (10 per session ID)
        requests = []
        for _ in range(20):
            for session_id in session_ids:
                requests.append(
                    switchboard.create(session_id=session_id, **COMPLETION_PARAMS)
                )

        await asyncio.gather(*requests)

        # Check distribution (should be uneven due to session stickiness)
        request_counts = sorted(
            [
                client.models["gpt-4o-mini"].rpm_usage
                for client in switchboard.deployments.values()
            ]
        )
        assert sum(request_counts) == 100
        _1_2_2 = [20, 40, 40]
        _0_4_6 = [0, 40, 60]
        assert request_counts == _1_2_2 or request_counts == _0_4_6, (
            "5 sessions into 3 deployments should create 1:2:2 or occasionally 0:4:6 distribution"
        )

    @pytest.mark.mock_models("gpt-4o-mini")
    async def test_ratelimit_reset(self):
        """Test that the ratelimit is reset correctly."""

        # create an switchboard with nonzero ratelimit_window to verify reset behavior
        async with Switchboard(
            deployments=[azure_config("test1")], ratelimit_window=0.5
        ) as switchboard:
            assert switchboard.ratelimit_reset_task

            # make some requests to add usage
            await asyncio.gather(
                *[switchboard.create(**COMPLETION_PARAMS) for _ in range(10)]
            )

            for d in switchboard.deployments.values():
                m = d.models["gpt-4o-mini"]
                assert m.tpm_usage > 0
                assert m.rpm_usage > 0

            # wait for the ratelimit to reset
            await asyncio.sleep(1)

            for d in switchboard.deployments.values():
                m = d.models["gpt-4o-mini"]
                assert m.tpm_usage == 0
                assert m.rpm_usage == 0

    async def test_no_deployments(self):
        """Test that the switchboard raises an error if no deployments are provided."""

        with pytest.raises(SwitchboardError, match="No deployments provided"):
            Switchboard(deployments=[])

    async def test_invalid_model(self, switchboard: Switchboard):
        """Test that an invalid model is not eligible on a deployment."""

        with pytest.raises(
            SwitchboardError,
            match="No eligible deployments available for invalid-model",
        ):
            await switchboard.create(model="invalid-model", messages=[])

    @pytest.mark.mock_models("openai")
    async def test_only_openai_deployment(self, mock_client: respx.MockRouter):
        """Test the edge case where only OpenAI deployment is configured."""
        # Create switchboard with only OpenAI deployment
        switchboard = Switchboard(deployments=[openai_config()])

        assert switchboard.fallback is not None
        assert isinstance(switchboard.fallback.config, OpenAIDeployment)
        assert len(switchboard.deployments) == 0

        # Verify that OpenAI deployment is selected
        deployment = switchboard.select_deployment(model="gpt-4o-mini")
        assert deployment == switchboard.fallback

        # Verify with session_id to cover line 171
        deployment_with_session = switchboard.select_deployment(model="gpt-4o-mini", session_id="test")
        assert deployment_with_session == switchboard.fallback
        assert switchboard.sessions["test"] == switchboard.fallback

        # Verify that requests work correctly
        response = await switchboard.create(**COMPLETION_PARAMS)
        assert response == COMPLETION_RESPONSE
        assert mock_client["openai"].call_count == 1

    async def test_no_fallback_no_deployments(self):
        """Test error when no deployments available and no fallback configured."""
        # Create a switchboard with an Azure deployment but no fallback
        switchboard = Switchboard(deployments=[azure_config("test1")])

        # Mark the deployment as unhealthy
        switchboard.deployments["test1"].models["gpt-4o-mini"].mark_down()

        # This should raise an error since there's no fallback
        with pytest.raises(
            SwitchboardError,
            match="No eligible deployments available for gpt-4o-mini",
        ):
            await switchboard.create(**COMPLETION_PARAMS)

    async def test_retry_exhausted_edge_case(self):
        """Test the edge case where retry policy returns without raising RetryError."""
        from unittest.mock import AsyncMock, MagicMock, patch

        # Create a mock retry policy that yields once then exits without raising
        class MockAttempt:
            def __enter__(self):
                return self
            def __exit__(self, exc_type, exc_val, exc_tb):
                # Suppress the exception so retry continues
                return True

        async def mock_retry_policy():
            yield MockAttempt()
            # Exit without raising StopAsyncIteration or RetryError
            return

        mock_policy = AsyncMock()
        mock_policy.__aiter__ = lambda self: mock_retry_policy()

        switchboard = Switchboard(
            deployments=[azure_config("test1")],
            failover_policy=mock_policy
        )

        # This should hit the "should never be reached" line
        with pytest.raises(
            SwitchboardError,
            match="All retry attempts exhausted for model gpt-4o-mini",
        ):
            await switchboard.create(**COMPLETION_PARAMS)
