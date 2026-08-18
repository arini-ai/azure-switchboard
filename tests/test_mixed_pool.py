"""Tests for a single Switchboard holding both chat and messages deployments."""

from unittest.mock import patch

import pytest

from azure_switchboard import Model, OpenAIConfig, Switchboard, SwitchboardError
from azure_switchboard.anthropic_api import AnthropicDeployment
from azure_switchboard.openai_api import OpenAIDeployment

from .conftest import (
    COMPLETION_PARAMS,
    MESSAGE_PARAMS,
    anthropic_config,
    azure_config,
    chat_completion_mock,
    message_mock,
)


@pytest.fixture
async def mixed():
    async with Switchboard(
        deployments=[
            azure_config("oai1"),
            azure_config("oai2"),
            anthropic_config("ant1"),
            anthropic_config("ant2"),
        ],
        ratelimit_window=0,
    ) as sb:
        yield sb


class TestSelection:
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


class TestSessionAffinity:
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


class TestDispatch:
    async def test_both_surfaces_route_correctly(self, mixed: Switchboard):
        with patch(
            "azure_switchboard.openai_api.OpenAIDeployment.create",
            side_effect=chat_completion_mock(),
        ) as chat_mock:
            await mixed.create(**COMPLETION_PARAMS)
        with patch(
            "azure_switchboard.anthropic_api.AnthropicDeployment.messages",
            side_effect=message_mock(),
        ) as msg_mock:
            await mixed.messages(**MESSAGE_PARAMS)

        chat_mock.assert_called_once()
        msg_mock.assert_called_once()

    async def test_parse_messages_routes_to_the_messages_api(self, mixed: Switchboard):
        from pydantic import BaseModel

        class Weather(BaseModel):
            city: str

        with patch(
            "azure_switchboard.anthropic_api.AnthropicDeployment.parse",
            side_effect=message_mock(),
        ) as mock:
            await mixed.parse_messages(
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
        from anthropic import APIConnectionError
        from httpx import Request

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
            assert await mixed.messages(**MESSAGE_PARAMS) == "ok"

        assert len(calls) == 2
        assert calls[0] != calls[1]

    async def test_unknown_model_raises_without_retrying(self, mixed: Switchboard):
        """SwitchboardError is excluded from the retry predicate."""
        with pytest.raises(SwitchboardError, match="No deployments available"):
            await mixed.messages(model="claude-opus-5", max_tokens=1, messages=[])


class TestConstruction:
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
            deployments=[azure_config("a"), azure_config("b"), azure_config("c")]
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
