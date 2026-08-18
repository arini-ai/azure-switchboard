import asyncio
from unittest.mock import patch

import pytest
import respx
from anthropic import APIConnectionError
from httpx import Request
from pydantic import BaseModel

from azure_switchboard import Model, OpenAIConfig, Switchboard, SwitchboardError
from azure_switchboard.anthropic_api import AnthropicDeployment
from azure_switchboard.openai_api import OpenAIDeployment

from .conftest import (
    COMPLETION_PARAMS,
    COMPLETION_RESPONSE,
    MESSAGE_PARAMS,
    anthropic_config,
    chat_completion_mock,
    collect_chunks,
    message_mock,
    openai_config,
)


class Weather(BaseModel):
    city: str


class TestSwitchboard:
    """Basic switchboard functionality tests."""

    @pytest.mark.mock_models("gpt-4o-mini")
    async def test_completion(
        self, switchboard: Switchboard, mock_client: respx.MockRouter
    ):
        """Test chat completion through switchboard."""

        assert "Switchboard" in repr(switchboard)

        response = await switchboard.chat.completions.create(**COMPLETION_PARAMS)
        assert mock_client["azure"].call_count == 1
        assert response == COMPLETION_RESPONSE

        assert any(
            filter(
                lambda d: d.get("gpt-4o-mini").rpm == "1/60",  # pyright: ignore[reportAttributeAccessIssue]
                switchboard.stats().values(),
            )
        )

    async def test_streaming(self, switchboard: Switchboard):
        """Test streaming through switchboard."""

        with patch("azure_switchboard.openai_api.OpenAIDeployment.create") as mock:
            mock.side_effect = chat_completion_mock()
            stream = await switchboard.chat.completions.create(
                stream=True, **COMPLETION_PARAMS
            )
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
        response = await switchboard.chat.completions.create(**COMPLETION_PARAMS)
        assert response == COMPLETION_RESPONSE
        assert mock_client["azure"].call_count == 1
        host_0 = mock_client["azure"].calls.last.request.url.host

        # Mark first deployment as unhealthy
        deployments[0].models["gpt-4o-mini"].mark_down()
        response = await switchboard.chat.completions.create(**COMPLETION_PARAMS)
        assert response == COMPLETION_RESPONSE
        assert mock_client["azure"].call_count == 2
        host_1 = mock_client["azure"].calls.last.request.url.host

        # Mark second deployment as unhealthy
        deployments[1].models["gpt-4o-mini"].mark_down()
        response = await switchboard.chat.completions.create(**COMPLETION_PARAMS)
        assert response == COMPLETION_RESPONSE
        assert mock_client["azure"].call_count == 3
        host_2 = mock_client["azure"].calls.last.request.url.host

        # Mark last deployment as unhealthy — all three are now down.
        # Switchboard falls back to an unhealthy deployment rather than failing.
        deployments[2].models["gpt-4o-mini"].mark_down()
        response = await switchboard.chat.completions.create(**COMPLETION_PARAMS)
        assert response == COMPLETION_RESPONSE
        assert mock_client["azure"].call_count == 4

        # Restore first deployment
        deployments[0].models["gpt-4o-mini"].mark_up()
        response = await switchboard.chat.completions.create(**COMPLETION_PARAMS)
        assert response == COMPLETION_RESPONSE
        assert mock_client["azure"].call_count == 5
        host_4 = mock_client["azure"].calls.last.request.url.host

        assert len(set([host_0, host_1, host_2, host_4])) > 1

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

    async def test_session_cached_deployment_missing_model_does_not_keyerror(self):
        """Session fallback should handle model-missing deployments without KeyError."""
        switchboard = Switchboard(
            deployments=[
                OpenAIConfig(
                    name="mini-only",
                    base_url="https://mini-only.openai.azure.com/openai/v1/",
                    api_key="mini-only",
                    models=[Model(name="gpt-4o-mini", tpm=1000, rpm=6)],
                ),
                OpenAIConfig(
                    name="full-only",
                    base_url="https://full-only.openai.azure.com/openai/v1/",
                    api_key="full-only",
                    models=[Model(name="gpt-4o", tpm=1000, rpm=6)],
                ),
            ],
            ratelimit_window=0,
        )

        # Session points to a deployment that does not have gpt-4o.
        switchboard.sessions["test"] = switchboard.deployments["mini-only"]

        selection = switchboard.select_deployment(session_id="test", model="gpt-4o")
        assert selection.name == "full-only"
        assert switchboard.sessions["test"].name == "full-only"

    @pytest.mark.mock_models("gpt-4o-mini")
    async def test_session_stickiness_failover(
        self, switchboard: Switchboard, mock_client: respx.MockRouter
    ):
        """Test session affinity when preferred deployment becomes unavailable."""

        session_id = "test"

        # Initial request establishes session affinity
        response1 = await switchboard.chat.completions.create(
            session_id=session_id, **COMPLETION_PARAMS
        )
        assert response1 == COMPLETION_RESPONSE
        assert mock_client["azure"].call_count == 1
        # Get assigned deployment
        assigned_deployment = switchboard.sessions[session_id]
        original_deployment = assigned_deployment

        # Verify session stickiness
        response2 = await switchboard.chat.completions.create(
            session_id=session_id, **COMPLETION_PARAMS
        )
        assert response2 == COMPLETION_RESPONSE
        assert switchboard.sessions[session_id] == original_deployment

        # Make assigned deployment unhealthy
        model = original_deployment.models["gpt-4o-mini"]
        model.mark_down()

        # Verify failover
        response3 = await switchboard.chat.completions.create(
            session_id=session_id, **COMPLETION_PARAMS
        )
        assert response3 == COMPLETION_RESPONSE
        assert switchboard.sessions[session_id] != original_deployment

        # Verify session maintains new assignment
        fallback_deployment = switchboard.sessions[session_id]
        response4 = await switchboard.chat.completions.create(
            session_id=session_id, **COMPLETION_PARAMS
        )
        assert response4 == COMPLETION_RESPONSE
        assert switchboard.sessions[session_id] == fallback_deployment

    @pytest.mark.mock_models("gpt-4o-mini", "openai")
    async def test_multiple_deployment_types(self, mock_client: respx.MockRouter):
        """Test that both Azure and OpenAI deployments work together."""
        switchboard = Switchboard(
            deployments=[openai_config("test1"), openai_config("openai", azure=False)]
        )

        assert len(switchboard.deployments) == 2
        assert "test1" in switchboard.deployments
        assert "openai" in switchboard.deployments

        # make azure deployment unhealthy, should route to openai
        switchboard.deployments["test1"].models["gpt-4o-mini"].mark_down()
        response = await switchboard.chat.completions.create(**COMPLETION_PARAMS)
        assert response == COMPLETION_RESPONSE
        assert mock_client["openai"].call_count == 1

        # bring azure back, should route to either
        switchboard.deployments["test1"].models["gpt-4o-mini"].mark_up()
        response = await switchboard.chat.completions.create(**COMPLETION_PARAMS)
        assert response == COMPLETION_RESPONSE

        # make both unhealthy — switchboard falls back to an unhealthy deployment
        switchboard.deployments["test1"].models["gpt-4o-mini"].mark_down()
        switchboard.deployments["openai"].models["gpt-4o-mini"].mark_down()
        response = await switchboard.chat.completions.create(**COMPLETION_PARAMS)
        assert response == COMPLETION_RESPONSE

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
            *[
                switchboard.chat.completions.create(**COMPLETION_PARAMS)
                for _ in range(100)
            ]
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
            await switchboard.chat.completions.create(**COMPLETION_PARAMS)

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
            await switchboard.chat.completions.create(**COMPLETION_PARAMS)

        # reset utilization of one deployment
        client = switchboard.select_deployment(model="gpt-4o-mini")
        client.reset_usage()

        # make another 100 requests
        for _ in range(100):
            await switchboard.chat.completions.create(**COMPLETION_PARAMS)

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
                    switchboard.chat.completions.create(
                        session_id=session_id, **COMPLETION_PARAMS
                    )
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

    async def test_ratelimit_reset(self):
        """The periodic task zeroes usage once the window elapses.

        Usage is seeded synchronously rather than via requests: with an await
        between spending and asserting, a slow runner can let the reset fire
        first and the assertion races. Accumulation from real requests is
        covered in test_openai_api.py / test_anthropic_api.py.
        """

        # nonzero ratelimit_window so the reset task actually runs
        async with Switchboard(
            deployments=[openai_config("test1")], ratelimit_window=0.5
        ) as switchboard:
            assert switchboard.ratelimit_reset_task

            models = [d.models["gpt-4o-mini"] for d in switchboard.deployments.values()]
            for m in models:
                m.spend_tokens(100)
                m.spend_request()
                # no await between spending and asserting, so this cannot race
                assert m.tpm_usage > 0
                assert m.rpm_usage > 0

            # wait for the ratelimit to reset
            await asyncio.sleep(1)

            for m in models:
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
            match="No deployments available for invalid-model",
        ):
            await switchboard.chat.completions.create(
                model="invalid-model", messages=[]
            )

    @pytest.mark.mock_models("openai")
    async def test_single_deployment(self, mock_client: respx.MockRouter):
        """Test the edge case where only a single deployment is configured."""
        switchboard = Switchboard(deployments=[openai_config("openai", azure=False)])
        assert len(switchboard.deployments) == 1

        # Verify that the deployment is selected
        deployment = switchboard.select_deployment(model="gpt-4o-mini")
        assert deployment == switchboard.deployments["openai"]

        # Verify with session_id
        deployment_with_session = switchboard.select_deployment(
            model="gpt-4o-mini", session_id="test"
        )
        assert deployment_with_session == switchboard.deployments["openai"]
        assert switchboard.sessions["test"] == switchboard.deployments["openai"]

        # Verify that requests work correctly
        response = await switchboard.chat.completions.create(**COMPLETION_PARAMS)
        assert response == COMPLETION_RESPONSE
        assert mock_client["openai"].call_count == 1

    @pytest.mark.mock_models("gpt-4o-mini")
    async def test_all_deployments_unhealthy_falls_back(
        self, mock_client: respx.MockRouter
    ):
        """When all deployments are unhealthy, switchboard falls back rather than failing."""
        switchboard = Switchboard(deployments=[openai_config("test1")])

        # Mark the deployment as unhealthy
        switchboard.deployments["test1"].models["gpt-4o-mini"].mark_down()

        # Should fall back to the unhealthy deployment and succeed
        response = await switchboard.chat.completions.create(**COMPLETION_PARAMS)
        assert response == COMPLETION_RESPONSE
        assert mock_client["azure"].call_count == 1

    async def test_duplicate_deployment_names(self):
        """Test that duplicate deployment names raise an error."""
        with pytest.raises(SwitchboardError, match="Duplicate deployment name: test1"):
            Switchboard(deployments=[openai_config("test1"), openai_config("test1")])

    async def test_handle_cancelled_error(self):
        """Test that Switchboard.create properly propagates asyncio.CancelledError."""
        switchboard = Switchboard(deployments=[openai_config("test1")])

        # Patch the underlying deployment.create to raise CancelledError
        with patch.object(
            switchboard.deployments["test1"],
            "create",
            side_effect=asyncio.CancelledError,
        ):
            # CancelledError should propagate to allow proper task cancellation
            with pytest.raises(asyncio.CancelledError):
                await switchboard.chat.completions.create(**COMPLETION_PARAMS)

        # Verify that the deployment is still selected as expected after cancellation
        deployment = switchboard.select_deployment(model="gpt-4o-mini")
        assert deployment == switchboard.deployments["test1"]

        deployment_with_session = switchboard.select_deployment(
            model="gpt-4o-mini", session_id="test"
        )
        assert deployment_with_session == switchboard.deployments["test1"]
        assert switchboard.sessions["test"] == switchboard.deployments["test1"]


@pytest.fixture
async def mixed():
    async with Switchboard(
        deployments=[
            openai_config("oai1"),
            openai_config("oai2"),
            anthropic_config("ant1"),
            anthropic_config("ant2"),
        ],
        ratelimit_window=0,
    ) as sb:
        yield sb


class TestMixedPoolSelection:
    """A single Switchboard holding both OpenAI and Anthropic deployments."""

    def test_builds_the_right_runtime_per_config(self, mixed: Switchboard):
        assert isinstance(mixed.deployments["oai1"], OpenAIDeployment)
        assert isinstance(mixed.deployments["ant1"], AnthropicDeployment)

    def test_model_name_routes_to_its_provider(self, mixed: Switchboard):
        """Model names are unique to a provider, so the name alone routes."""
        for _ in range(20):
            assert isinstance(
                mixed.select_deployment(model="gpt-4o-mini"), OpenAIDeployment
            )
            assert isinstance(
                mixed.select_deployment(model="claude-sonnet-5"), AnthropicDeployment
            )

    def test_stats_span_both_providers(self, mixed: Switchboard):
        stats = mixed.stats()
        assert "gpt-4o-mini" in stats["oai1"]
        assert "claude-sonnet-5" in stats["ant1"]

    def test_reset_usage_spans_both_providers(self, mixed: Switchboard):
        mixed.deployments["oai1"].model("gpt-4o-mini").spend_tokens(100)
        mixed.deployments["ant1"].model("claude-sonnet-5").spend_tokens(100)
        mixed.reset_usage()
        assert mixed.deployments["oai1"].model("gpt-4o-mini").tpm_usage == 0
        assert mixed.deployments["ant1"].model("claude-sonnet-5").tpm_usage == 0


class TestMixedPoolSessionAffinity:
    def test_session_is_sticky_within_an_api(self, mixed: Switchboard):
        first = mixed.select_deployment(model="gpt-4o-mini", session_id="s1")
        for _ in range(10):
            assert (
                mixed.select_deployment(model="gpt-4o-mini", session_id="s1") is first
            )

    def test_shared_session_id_does_not_cross_providers(self, mixed: Switchboard):
        """A cached session deployment is only reused when it serves the
        requested model, so one session_id spanning both APIs is safe."""
        chat = mixed.select_deployment(model="gpt-4o-mini", session_id="shared")
        msgs = mixed.select_deployment(model="claude-sonnet-5", session_id="shared")
        assert isinstance(chat, OpenAIDeployment)
        assert isinstance(msgs, AnthropicDeployment)


class TestMixedPoolDispatch:
    async def test_both_surfaces_route_correctly(self, mixed: Switchboard):
        with patch(
            "azure_switchboard.openai_api.OpenAIDeployment.create",
            side_effect=chat_completion_mock(),
        ) as chat_mock:
            await mixed.chat.completions.create(**COMPLETION_PARAMS)
        with patch(
            "azure_switchboard.anthropic_api.AnthropicDeployment.messages",
            side_effect=message_mock(),
        ) as msg_mock:
            await mixed.messages.create(**MESSAGE_PARAMS)

        chat_mock.assert_called_once()
        msg_mock.assert_called_once()

    async def test_parse_messages_routes_to_the_messages_api(self, mixed: Switchboard):
        with patch(
            "azure_switchboard.anthropic_api.AnthropicDeployment.parse",
            side_effect=message_mock(),
        ) as mock:
            await mixed.messages.parse(
                model="claude-sonnet-5",
                output_format=Weather,
                max_tokens=1024,
                messages=[{"role": "user", "content": "Weather in Paris?"}],
            )
        mock.assert_called_once()
        assert mock.call_args.kwargs["output_format"] is Weather

    async def test_messages_fails_over_to_a_healthy_deployment(
        self, mixed: Switchboard
    ):
        """The failover policy is copied per call, so a marked-down deployment
        is skipped on retry rather than failing the request."""
        calls = []

        async def flaky(self, *, model, **kwargs):
            calls.append(self.name)
            if len(calls) == 1:
                self.model(model).mark_down()
                raise APIConnectionError(request=Request("POST", "https://x/"))
            return "ok"

        with patch(
            "azure_switchboard.anthropic_api.AnthropicDeployment.messages", new=flaky
        ):
            assert await mixed.messages.create(**MESSAGE_PARAMS) == "ok"

        assert len(calls) == 2
        assert calls[0] != calls[1]

    async def test_unknown_model_raises_without_retrying(self, mixed: Switchboard):
        """SwitchboardError is excluded from the retry predicate."""
        with pytest.raises(SwitchboardError, match="No deployments available"):
            await mixed.messages.create(
                model="claude-opus-5", max_tokens=1, messages=[]
            )


class TestMixedPoolConstruction:
    def test_model_registered_on_both_providers_rejected(self):
        """Routing is by model name alone, so a name shared across providers
        would make the serving surface ambiguous. Reject it up front."""
        with pytest.raises(SwitchboardError, match="must belong to exactly one"):
            Switchboard(
                deployments=[
                    OpenAIConfig(
                        name="compat",
                        base_url="https://r.services.ai.azure.com/openai/v1/",
                        api_key="k",
                        models=[Model(name="claude-sonnet-5")],
                    ),
                    anthropic_config("native"),
                ]
            )

    def test_same_model_on_many_deployments_of_one_provider_is_fine(self):
        """Sharing a model across deployments is the whole point of the pool."""
        sb = Switchboard(
            deployments=[openai_config("a"), openai_config("b"), openai_config("c")]
        )
        assert len(sb.deployments) == 3

    def test_duplicate_names_across_providers_rejected(self):
        with pytest.raises(SwitchboardError, match="Duplicate deployment name"):
            Switchboard(
                deployments=[
                    OpenAIConfig(
                        name="dup", api_key="k", models=[Model(name="gpt-4o")]
                    ),
                    anthropic_config("dup"),
                ]
            )
