import asyncio
from unittest.mock import patch

import pytest
import respx

from azure_switchboard import OpenAIDeployment, Switchboard, SwitchboardError

from .fixtures import MOCK_COMPLETION, MOCK_COMPLETION_JSON
from .utils import BaseTestCase, azure_config, chat_completion_mock, openai_config


@pytest.fixture(scope="function")
def switchboard():
    deployments = [
        azure_config("test1"),
        azure_config("test2"),
        azure_config("test3"),
    ]
    return Switchboard(deployments=deployments)


@pytest.fixture(scope="function", autouse=True)
def mock_client(request: pytest.FixtureRequest):
    with respx.mock() as respx_mock:
        # By default, we don't assert all routes were called
        respx_mock._assert_all_called = False

        # Check if we have specific models to mock
        mock_models_marker = request.node.get_closest_marker("mock_models")
        models_to_mock = []

        if mock_models_marker:
            # Get models from marker
            models_to_mock = mock_models_marker.args

            # If models were specified, enable assert_all_called
            if models_to_mock:
                respx_mock._assert_all_called = True

        # Add routes for each model
        for model in models_to_mock:
            if model == "openai":
                respx_mock.route(
                    name="openai",
                    method="POST",
                    host="api.openai.com",
                    path="/v1/chat/completions",
                ).respond(status_code=200, json=MOCK_COMPLETION_JSON)
            else:
                respx_mock.route(
                    name=model,
                    method="POST",
                    path=f"/openai/deployments/{model}/chat/completions",
                ).respond(status_code=200, json=MOCK_COMPLETION_JSON)

        yield respx_mock


class TestSwitchboard(BaseTestCase):
    """Basic switchboard functionality tests."""

    @pytest.mark.mock_models("gpt-4o-mini")
    async def test_completion(
        self, switchboard: Switchboard, mock_client: respx.MockRouter
    ):
        """Test chat completion through switchboard."""

        assert "Switchboard" in repr(switchboard)

        response = await switchboard.create(**self.basic_args)
        assert mock_client["gpt-4o-mini"].call_count == 1
        assert response == MOCK_COMPLETION

        assert any(
            filter(
                lambda d: d["gpt-4o-mini"].get("rpm") == "1/60",
                switchboard.get_usage().values(),
            )
        )

    async def test_streaming(self, switchboard: Switchboard):
        """Test streaming through switchboard."""

        with patch("azure_switchboard.deployment.Deployment.create") as mock:
            mock.side_effect = chat_completion_mock()
            stream = await switchboard.create(stream=True, **self.basic_args)
            _, content = await self.collect_chunks(stream)

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
        response = await switchboard.create(**self.basic_args)
        assert response == MOCK_COMPLETION
        assert mock_client["gpt-4o-mini"].call_count == 1
        host_0 = mock_client["gpt-4o-mini"].calls.last.request.url.host

        # Mark first deployment as unhealthy
        deployments[0].models["gpt-4o-mini"].cooldown()
        response = await switchboard.create(**self.basic_args)
        assert response == MOCK_COMPLETION
        assert mock_client["gpt-4o-mini"].call_count == 2
        host_1 = mock_client["gpt-4o-mini"].calls.last.request.url.host

        # Mark second deployment as unhealthy
        deployments[1].models["gpt-4o-mini"].cooldown()
        response = await switchboard.create(**self.basic_args)
        assert response == MOCK_COMPLETION
        assert mock_client["gpt-4o-mini"].call_count == 3
        host_2 = mock_client["gpt-4o-mini"].calls.last.request.url.host

        # Mark last deployment as unhealthy
        deployments[2].models["gpt-4o-mini"].cooldown()
        with pytest.raises(SwitchboardError, match="All attempts failed"):
            await switchboard.create(**self.basic_args)
        assert mock_client["gpt-4o-mini"].call_count == 3

        # Restore first deployment
        deployments[0].models["gpt-4o-mini"].reset_cooldown()
        response = await switchboard.create(**self.basic_args)
        assert response == MOCK_COMPLETION
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
        client_1.models["gpt-4o-mini"].cooldown()
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
        response1 = await switchboard.create(session_id=session_id, **self.basic_args)
        assert response1 == MOCK_COMPLETION
        assert mock_client["gpt-4o-mini"].call_count == 1
        # Get assigned deployment
        assigned_deployment = switchboard._sessions[session_id]
        original_deployment = assigned_deployment

        # Verify session stickiness
        response2 = await switchboard.create(session_id=session_id, **self.basic_args)
        assert response2 == MOCK_COMPLETION
        assert switchboard._sessions[session_id] == original_deployment

        # Make assigned deployment unhealthy
        model = original_deployment.models["gpt-4o-mini"]
        model.cooldown()

        # Verify failover
        response3 = await switchboard.create(session_id=session_id, **self.basic_args)
        assert response3 == MOCK_COMPLETION
        assert switchboard._sessions[session_id] != original_deployment

        # Verify session maintains new assignment
        fallback_deployment = switchboard._sessions[session_id]
        response4 = await switchboard.create(session_id=session_id, **self.basic_args)
        assert response4 == MOCK_COMPLETION
        assert switchboard._sessions[session_id] == fallback_deployment

    @pytest.mark.mock_models("gpt-4o-mini", "openai")
    async def test_fallback_to_openai(self, mock_client: respx.MockRouter):
        """Test that the switchboard can fallback to OpenAI."""

        switchboard = Switchboard(deployments=[azure_config("test1"), openai_config()])

        assert switchboard.fallback is not None
        assert isinstance(switchboard.fallback.config, OpenAIDeployment)

        # basic test to verify the fallback works
        response = await switchboard.fallback.create(**self.basic_args)
        assert response == MOCK_COMPLETION
        assert mock_client["openai"].call_count == 1

        # default: use the healthy azure deployment
        response = await switchboard.create(**self.basic_args)
        assert response == MOCK_COMPLETION
        assert mock_client["gpt-4o-mini"].call_count == 1

        # make deployment unhealthy so it falls back to openai
        switchboard.deployments["test1"].models["gpt-4o-mini"].cooldown()
        await switchboard.create(**self.basic_args)
        assert mock_client["openai"].call_count == 2

        # make openai fallback unhealthy, verify it throws
        mock_client["openai"].side_effect = Exception("test")
        with pytest.raises(SwitchboardError, match="Fallback to OpenAI failed"):
            await switchboard.create(**self.basic_args)
        # 3 total additional calls were made, because openai retries twice internally
        assert mock_client["openai"].call_count == 5

        # bring the deployment back, verify we use it
        switchboard.deployments["test1"].models["gpt-4o-mini"].reset_cooldown()
        await switchboard.create(**self.basic_args)
        assert mock_client["gpt-4o-mini"].call_count == 2

        # make everything unhealthy, verify it throws
        switchboard.deployments["test1"].models["gpt-4o-mini"].cooldown()
        with pytest.raises(SwitchboardError, match="Fallback to OpenAI failed"):
            await switchboard.create(**self.basic_args)
        assert mock_client["openai"].call_count == 8

        # reset fallback, verify it works
        mock_client["openai"].side_effect = None
        await switchboard.create(**self.basic_args)
        assert mock_client["openai"].call_count == 9

        # reset the deployment and verify it gets used again
        switchboard.deployments["test1"].models["gpt-4o-mini"].reset_cooldown()
        await switchboard.create(**self.basic_args)
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
            *[switchboard.create(**self.basic_args) for _ in range(100)]
        )

        # Verify all deployments were used
        for deployment in switchboard.deployments.values():
            assert self._within_bounds(
                val=deployment.models["gpt-4o-mini"]._rpm_usage,
                min=25,
                max=40,
            )

    @pytest.mark.mock_models("gpt-4o-mini")
    async def test_load_distribution_health_awareness(self, switchboard: Switchboard):
        """Test load distribution when some deployments are unhealthy."""

        # Mark one deployment as unhealthy
        switchboard.deployments["test2"].models["gpt-4o-mini"].cooldown()

        # Make 100 requests
        for _ in range(100):
            await switchboard.create(**self.basic_args)

        # Verify distribution
        assert self._within_bounds(
            val=switchboard.deployments["test1"].models["gpt-4o-mini"]._rpm_usage,
            min=40,
            max=60,
        )
        assert self._within_bounds(
            val=switchboard.deployments["test2"].models["gpt-4o-mini"]._rpm_usage,
            min=0,
            max=0,
        )
        assert self._within_bounds(
            val=switchboard.deployments["test3"].models["gpt-4o-mini"]._rpm_usage,
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
            await switchboard.create(**self.basic_args)

        # reset utilization of one deployment
        client = switchboard.select_deployment(model="gpt-4o-mini")
        client.reset_usage()

        # make another 100 requests
        for _ in range(100):
            await switchboard.create(**self.basic_args)

        # verify the load distribution is still roughly even
        # (ie, we preferred to send requests to the underutilized deployment)
        for client in switchboard.deployments.values():
            assert self._within_bounds(
                val=client.models["gpt-4o-mini"]._rpm_usage,
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
                    switchboard.create(session_id=session_id, **self.basic_args)
                )

        await asyncio.gather(*requests)

        # Check distribution (should be uneven due to session stickiness)
        request_counts = sorted(
            [
                client.models["gpt-4o-mini"]._rpm_usage
                for client in switchboard.deployments.values()
            ]
        )
        assert sum(request_counts) == 100
        assert request_counts == [20, 40, 40], (
            "5 sessions into 3 deployments should create 1:2:2 distribution"
        )

    @pytest.mark.mock_models("gpt-4o-mini")
    async def test_ratelimit_reset(self, switchboard: Switchboard):
        """Test that the ratelimit is reset correctly."""

        # create an simple switchboard to verify autostart
        async with Switchboard(deployments=[azure_config("test1")]) as sb:
            assert sb.ratelimit_reset_task

        # speed things up a bit
        switchboard._ratelimit_window = 0.5
        switchboard.start()

        assert switchboard.ratelimit_reset_task is not None

        # make some requests to add usage
        for _ in range(10):
            await switchboard.create(**self.basic_args)

        for d in switchboard.deployments.values():
            m = d.models["gpt-4o-mini"]
            assert m._tpm_usage > 0
            assert m._rpm_usage > 0

        # wait for the ratelimit to reset
        await asyncio.sleep(1)

        for d in switchboard.deployments.values():
            m = d.models["gpt-4o-mini"]
            assert m._tpm_usage == 0
            assert m._rpm_usage == 0

        await switchboard.stop()

    async def test_no_deployments(self):
        """Test that the switchboard raises an error if no deployments are provided."""

        with pytest.raises(SwitchboardError, match="No deployments provided"):
            Switchboard(deployments=[])
