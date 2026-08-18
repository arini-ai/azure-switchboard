import asyncio
from unittest.mock import patch

import pytest
import respx
from anthropic import APIConnectionError
from httpx import Request
from pydantic import BaseModel

from azure_switchboard import Foundry, OpenAIModel, Switchboard, SwitchboardError
from azure_switchboard.anthropic_model import AnthropicModel
from azure_switchboard.model import ModelBase

from .conftest import (
    COMPLETION_PARAMS,
    COMPLETION_RESPONSE,
    MESSAGE_PARAMS,
    anthropic_foundry,
    chat_completion_mock,
    collect_chunks,
    message_mock,
    openai_foundry,
    select_anthropic,
    select_openai,
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

        with patch("azure_switchboard.openai_model.OpenAIModel.create") as mock:
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
        client = select_openai(switchboard, model="gpt-4o-mini")
        assert client.foundry.name in switchboard.foundries

        deployments = list(switchboard.foundries.values())
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

        # Mark last deployment as unhealthy — all three are now down, and with
        # no fallback configured there is nothing left to select.
        deployments[2].models["gpt-4o-mini"].mark_down()
        with pytest.raises(SwitchboardError, match="No deployments available"):
            await switchboard.chat.completions.create(**COMPLETION_PARAMS)
        assert mock_client["azure"].call_count == 3

        # Restore first deployment
        deployments[0].models["gpt-4o-mini"].mark_up()
        response = await switchboard.chat.completions.create(**COMPLETION_PARAMS)
        assert response == COMPLETION_RESPONSE
        assert mock_client["azure"].call_count == 4
        host_4 = mock_client["azure"].calls.last.request.url.host

        assert len(set([host_0, host_1, host_2, host_4])) > 1

    async def test_session_stickiness(self, switchboard: Switchboard) -> None:
        """Test session stickiness and failover."""

        # Test consistent deployment selection for session
        client_1 = select_openai(switchboard, session_id="test", model="gpt-4o-mini")
        client_2 = select_openai(switchboard, session_id="test", model="gpt-4o-mini")
        assert client_1.foundry.name == client_2.foundry.name

        # Test failover when selected deployment is unhealthy
        client_1.mark_down()
        client_3 = select_openai(switchboard, session_id="test", model="gpt-4o-mini")
        assert client_3.foundry.name != client_1.foundry.name

        # Test session maintains failover assignment
        client_4 = select_openai(switchboard, session_id="test", model="gpt-4o-mini")
        assert client_4.foundry.name == client_3.foundry.name

    async def test_session_cached_deployment_missing_model_does_not_keyerror(self):
        """Session fallback should handle model-missing deployments without KeyError."""
        switchboard = Switchboard(
            foundries=[
                Foundry(
                    name="mini-only",
                    api_key="mini-only",
                    models=[OpenAIModel(name="gpt-4o-mini", tpm=1000, rpm=6)],
                ),
                Foundry(
                    name="full-only",
                    api_key="full-only",
                    models=[OpenAIModel(name="gpt-4o", tpm=1000, rpm=6)],
                ),
            ],
            ratelimit_window=0,
        )

        # Session is pinned to a foundry that does not host gpt-4o.
        switchboard.sessions["test"] = switchboard.foundries["mini-only"]

        selection = select_openai(switchboard, session_id="test", model="gpt-4o")
        assert selection.foundry.name == "full-only"
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
    async def test_first_party_fallback(self, mock_client: respx.MockRouter):
        """The vendor's own API is what selection reaches for once the pool
        has nothing healthy left."""
        switchboard = Switchboard(
            foundries=[openai_foundry("test1")], openai_fallback=True
        )

        # healthy pool: the fallback is not touched
        await switchboard.chat.completions.create(**COMPLETION_PARAMS)
        assert mock_client["azure"].call_count == 1
        assert mock_client["openai"].call_count == 0

        switchboard.foundries["test1"].models["gpt-4o-mini"].mark_down()
        response = await switchboard.chat.completions.create(**COMPLETION_PARAMS)
        assert response == COMPLETION_RESPONSE
        assert mock_client["openai"].call_count == 1

    async def test_first_party_fallback_serves_unconfigured_models(self):
        """A model on no foundry at all still routes, when fallback is on."""
        switchboard = Switchboard(
            foundries=[openai_foundry("test1")], openai_fallback=True
        )
        selection = select_openai(switchboard, model="gpt-5.4")
        assert selection.foundry.name == "first-party-openai"

    async def test_first_party_fallback_can_itself_be_marked_down(self):
        """A rate-limited first-party API should not be hammered either."""
        switchboard = Switchboard(
            foundries=[openai_foundry("test1")], openai_fallback=True
        )
        switchboard.foundries["test1"].models["gpt-4o-mini"].mark_down()

        fallback = select_openai(switchboard, model="gpt-4o-mini")
        fallback.mark_down()

        with pytest.raises(SwitchboardError, match="No deployments available"):
            select_openai(switchboard, model="gpt-4o-mini")

    async def test_no_fallback_configured_raises(self):
        switchboard = Switchboard(foundries=[openai_foundry("test1")])
        switchboard.foundries["test1"].models["gpt-4o-mini"].mark_down()

        with pytest.raises(SwitchboardError, match="No deployments available"):
            select_openai(switchboard, model="gpt-4o-mini")

    async def test_anthropic_first_party_fallback(self):
        switchboard = Switchboard(
            foundries=[anthropic_foundry("ant1")], anthropic_fallback=True
        )
        switchboard.foundries["ant1"].models["claude-sonnet-5"].mark_down()

        selection = select_anthropic(switchboard, model="claude-sonnet-5")
        assert isinstance(selection, AnthropicModel)
        assert selection.foundry.name == "first-party-anthropic"

    async def test_each_fallback_is_scoped_to_its_own_api(self):
        """The two vendors are separate hosts, so one going down must not take
        the other with it."""
        switchboard = Switchboard(
            foundries=[openai_foundry("oai"), anthropic_foundry("ant")],
            openai_fallback=True,
            anthropic_fallback=True,
        )
        for foundry in switchboard.foundries.values():
            for deployment in foundry.models.values():
                deployment.mark_down()

        openai_fb = select_openai(switchboard, model="gpt-4o-mini")
        anthropic_fb = select_anthropic(switchboard, model="claude-sonnet-5")
        assert openai_fb.foundry is not anthropic_fb.foundry

        openai_fb.foundry.mark_down()
        assert not openai_fb.is_healthy()
        assert anthropic_fb.is_healthy()

    async def test_fallback_deployments_appear_in_stats(self):
        switchboard = Switchboard(
            foundries=[openai_foundry("test1")], openai_fallback=True
        )
        assert "first-party-openai" in switchboard.stats()

    async def test_foundry_must_hold_typed_deployments(self):
        resource = Foundry(name="odd", api_key="k")
        resource.models["mystery"] = ModelBase(name="mystery")
        with pytest.raises(SwitchboardError, match="not an OpenAIModel"):
            Switchboard(foundries=[resource])

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
        for deployment in switchboard.foundries.values():
            assert self._within_bounds(
                val=deployment.models["gpt-4o-mini"].rpm_usage,
                min=25,
                max=40,
            )

    @pytest.mark.mock_models("gpt-4o-mini")
    async def test_load_distribution_health_awareness(self, switchboard: Switchboard):
        """Test load distribution when some deployments are unhealthy."""

        # Mark one deployment as unhealthy
        switchboard.foundries["test2"].models["gpt-4o-mini"].mark_down()

        # Make 100 requests
        for _ in range(100):
            await switchboard.chat.completions.create(**COMPLETION_PARAMS)

        # Verify distribution
        assert self._within_bounds(
            val=switchboard.foundries["test1"].models["gpt-4o-mini"].rpm_usage,
            min=40,
            max=60,
        )
        assert self._within_bounds(
            val=switchboard.foundries["test2"].models["gpt-4o-mini"].rpm_usage,
            min=0,
            max=0,
        )
        assert self._within_bounds(
            val=switchboard.foundries["test3"].models["gpt-4o-mini"].rpm_usage,
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
        client = select_openai(switchboard, model="gpt-4o-mini")
        client.reset_usage()

        # make another 100 requests
        for _ in range(100):
            await switchboard.chat.completions.create(**COMPLETION_PARAMS)

        # verify the load distribution is still roughly even
        # (ie, we preferred to send requests to the underutilized deployment)
        for client in switchboard.foundries.values():
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
                for client in switchboard.foundries.values()
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

        Usage is seeded synchronously: an await between spending and
        asserting lets the reset fire first and the assertion races.
        """

        # nonzero ratelimit_window so the reset task actually runs
        async with Switchboard(
            foundries=[openai_foundry("test1")], ratelimit_window=0.5
        ) as switchboard:
            assert switchboard.ratelimit_reset_task

            models = [d.models["gpt-4o-mini"] for d in switchboard.foundries.values()]
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

        with pytest.raises(SwitchboardError, match="No foundries provided"):
            Switchboard(foundries=[])

    async def test_invalid_model(self, switchboard: Switchboard):
        """Test that an invalid model is not eligible on a deployment."""

        with pytest.raises(
            SwitchboardError,
            match="No deployments available for invalid-model",
        ):
            await switchboard.chat.completions.create(
                model="invalid-model", messages=[]
            )

    @pytest.mark.mock_models("gpt-4o-mini")
    async def test_single_deployment(self, mock_client: respx.MockRouter):
        """Test the edge case where only a single foundry is configured."""
        switchboard = Switchboard(foundries=[openai_foundry("solo")])
        assert len(switchboard.foundries) == 1

        only = switchboard.foundries["solo"].models["gpt-4o-mini"]
        assert select_openai(switchboard, model="gpt-4o-mini") is only

        # Verify with session_id
        assert (
            select_openai(switchboard, model="gpt-4o-mini", session_id="test") is only
        )
        assert switchboard.sessions["test"] is switchboard.foundries["solo"]

        # Verify that requests work correctly
        response = await switchboard.chat.completions.create(**COMPLETION_PARAMS)
        assert response == COMPLETION_RESPONSE
        assert mock_client["azure"].call_count == 1

    @pytest.mark.mock_models("gpt-4o-mini")
    async def test_all_deployments_unhealthy_raises(
        self, mock_client: respx.MockRouter
    ):
        """Without a fallback there is no honest answer but to fail.

        Pre-Foundry this handed back a cooling deployment, because failing was
        worse. openai_fallback=True is the real alternative now.
        """
        switchboard = Switchboard(foundries=[openai_foundry("test1")])
        switchboard.foundries["test1"].models["gpt-4o-mini"].mark_down()

        with pytest.raises(SwitchboardError, match="No deployments available"):
            await switchboard.chat.completions.create(**COMPLETION_PARAMS)
        assert mock_client["azure"].call_count == 0

    async def test_duplicate_foundry_names(self):
        """Test that duplicate foundry names raise an error."""
        with pytest.raises(SwitchboardError, match="Duplicate foundry name: test1"):
            Switchboard(foundries=[openai_foundry("test1"), openai_foundry("test1")])

    async def test_handle_cancelled_error(self):
        """Test that Switchboard.create properly propagates asyncio.CancelledError."""
        switchboard = Switchboard(foundries=[openai_foundry("test1")])

        # Patch the underlying deployment.create to raise CancelledError
        with patch.object(
            switchboard.foundries["test1"].models["gpt-4o-mini"],
            "create",
            side_effect=asyncio.CancelledError,
        ):
            # CancelledError should propagate to allow proper task cancellation
            with pytest.raises(asyncio.CancelledError):
                await switchboard.chat.completions.create(**COMPLETION_PARAMS)

        # Verify that the deployment is still selected as expected after cancellation
        deployment = select_openai(switchboard, model="gpt-4o-mini")
        assert deployment.foundry is switchboard.foundries["test1"]

        deployment_with_session = select_openai(
            switchboard, model="gpt-4o-mini", session_id="test"
        )
        assert deployment_with_session.foundry is switchboard.foundries["test1"]
        assert switchboard.sessions["test"] is switchboard.foundries["test1"]


@pytest.fixture
async def mixed():
    async with Switchboard(
        foundries=[
            openai_foundry("oai1"),
            openai_foundry("oai2"),
            anthropic_foundry("ant1"),
            anthropic_foundry("ant2"),
        ],
        ratelimit_window=0,
    ) as sb:
        yield sb


class TestMixedPoolSelection:
    """A single Switchboard holding both OpenAI and Anthropic deployments."""

    def test_pools_by_api_not_by_foundry(self, mixed: Switchboard):
        assert set(mixed.chat.completions._pool) == {"gpt-4o-mini", "gpt-4o"}
        assert set(mixed.messages._pool) == {"claude-sonnet-5", "claude-haiku-4-5"}
        assert all(
            isinstance(d, OpenAIModel)
            for d in mixed.chat.completions._pool["gpt-4o-mini"]
        )
        assert all(
            isinstance(d, AnthropicModel)
            for d in mixed.messages._pool["claude-sonnet-5"]
        )

    def test_model_name_routes_to_its_provider(self, mixed: Switchboard):
        """Model names are unique to a provider, so the name alone routes."""
        for _ in range(20):
            assert isinstance(
                select_openai(mixed, model="gpt-4o-mini"),
                OpenAIModel,
            )
            assert isinstance(
                select_anthropic(mixed, model="claude-sonnet-5"),
                AnthropicModel,
            )

    def test_stats_span_both_providers(self, mixed: Switchboard):
        stats = mixed.stats()
        assert "gpt-4o-mini" in stats["oai1"]
        assert "claude-sonnet-5" in stats["ant1"]

    def test_reset_usage_spans_both_providers(self, mixed: Switchboard):
        oai = mixed.foundries["oai1"].models["gpt-4o-mini"]
        ant = mixed.foundries["ant1"].models["claude-sonnet-5"]
        oai.spend_tokens(100)
        ant.spend_tokens(100)
        mixed.reset_usage()
        assert oai.tpm_usage == 0
        assert ant.tpm_usage == 0


class TestMixedPoolSessionAffinity:
    def test_session_is_sticky_within_an_api(self, mixed: Switchboard):
        first = select_openai(mixed, model="gpt-4o-mini", session_id="s1")
        for _ in range(10):
            assert select_openai(mixed, model="gpt-4o-mini", session_id="s1") is first

    def test_shared_session_id_does_not_cross_providers(self, mixed: Switchboard):
        """A cached session deployment is only reused when it serves the
        requested model, so one session_id spanning both APIs is safe."""
        chat = select_openai(mixed, model="gpt-4o-mini", session_id="shared")
        msgs = select_anthropic(mixed, model="claude-sonnet-5", session_id="shared")
        assert isinstance(chat, OpenAIModel)
        assert isinstance(msgs, AnthropicModel)


class TestMixedPoolDispatch:
    async def test_both_surfaces_route_correctly(self, mixed: Switchboard):
        with patch(
            "azure_switchboard.openai_model.OpenAIModel.create",
            side_effect=chat_completion_mock(),
        ) as chat_mock:
            await mixed.chat.completions.create(**COMPLETION_PARAMS)
        with patch(
            "azure_switchboard.anthropic_model.AnthropicModel.create",
            side_effect=message_mock(),
        ) as msg_mock:
            await mixed.messages.create(**MESSAGE_PARAMS)

        chat_mock.assert_called_once()
        msg_mock.assert_called_once()

    async def test_parse_messages_routes_to_the_messages_api(self, mixed: Switchboard):
        with patch(
            "azure_switchboard.anthropic_model.AnthropicModel.parse",
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

        async def flaky(self, **kwargs):
            calls.append(self.foundry.name)
            if len(calls) == 1:
                self.mark_down()
                raise APIConnectionError(request=Request("POST", "https://x/"))
            return "ok"

        with patch(
            "azure_switchboard.anthropic_model.AnthropicModel.create", new=flaky
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
    def test_model_name_may_belong_to_both_providers(self):
        """The calling surface picks the pool, so a shared name is unambiguous.

        Pre-Foundry this had to be rejected at construction: one flat pool keyed
        by model name could not say which API a name spoke.
        """
        sb = Switchboard(
            foundries=[
                Foundry(
                    name="compat",
                    api_key="k",
                    models=[OpenAIModel(name="claude-sonnet-5")],
                ),
                anthropic_foundry("native"),
            ]
        )

        assert sb.chat.completions._pool["claude-sonnet-5"][0].foundry.name == "compat"
        assert sb.messages._pool["claude-sonnet-5"][0].foundry.name == "native"

    def test_same_model_on_many_foundries_is_fine(self):
        """Sharing a model across resources is the whole point of the pool."""
        sb = Switchboard(
            foundries=[openai_foundry("a"), openai_foundry("b"), openai_foundry("c")]
        )
        assert len(sb.foundries) == 3
        assert len(sb.chat.completions._pool["gpt-4o-mini"]) == 3

    def test_duplicate_foundry_names_rejected(self):
        with pytest.raises(SwitchboardError, match="Duplicate foundry name"):
            Switchboard(
                foundries=[
                    Foundry(
                        name="dup", api_key="k", models=[OpenAIModel(name="gpt-4o")]
                    ),
                    anthropic_foundry("dup"),
                ]
            )

    def test_duplicate_model_on_one_foundry_rejected(self):
        with pytest.raises(SwitchboardError, match="duplicate model"):
            Foundry(
                name="r",
                api_key="k",
                models=[OpenAIModel(name="gpt-4o"), OpenAIModel(name="gpt-4o")],
            )

    def test_one_foundry_serves_both_apis(self):
        """The case the pre-Foundry model could not express at all.

        Both deployments share one resource, one credential, and one endpoint
        host, but each gets the client for the API it speaks.
        """
        resource = Foundry(
            name="east",
            api_key="k",
            models=[
                OpenAIModel(name="gpt-4o-mini"),
                AnthropicModel(name="claude-sonnet-5"),
            ],
        )
        sb = Switchboard(foundries=[resource])

        assert sb.chat.completions._pool["gpt-4o-mini"][0].foundry is resource
        assert sb.messages._pool["claude-sonnet-5"][0].foundry is resource
        assert (
            resource.base_url(resource.models["gpt-4o-mini"])
            == "https://east.services.ai.azure.com/openai/v1/"
        )
        assert (
            resource.base_url(resource.models["claude-sonnet-5"])
            == "https://east.services.ai.azure.com/anthropic/"
        )

    def test_endpoint_override_wins(self):
        """Legacy <resource>.openai.azure.com hosts stay reachable."""
        resource = Foundry(
            name="legacy",
            api_key="k",
            models=[
                OpenAIModel(
                    name="gpt-4o",
                    endpoint="https://legacy.openai.azure.com/openai/v1/",
                )
            ],
        )
        assert (
            resource.base_url(resource.models["gpt-4o"])
            == "https://legacy.openai.azure.com/openai/v1/"
        )

    def test_one_client_per_api_per_foundry(self):
        """Two deployments of the same API share a connection pool."""
        resource = Foundry(
            name="east",
            api_key="k",
            models=[OpenAIModel(name="gpt-4o-mini"), OpenAIModel(name="gpt-4o")],
        )
        a, b = resource.models["gpt-4o-mini"], resource.models["gpt-4o"]
        assert a.client is b.client

    def test_client_is_built_only_for_the_apis_in_use(self):
        resource = Foundry(
            name="east", api_key="k", models=[OpenAIModel(name="gpt-4o-mini")]
        )
        _ = resource.models["gpt-4o-mini"].client
        assert list(resource._clients) == [
            ("openai", "https://east.services.ai.azure.com/openai/v1/")
        ]
