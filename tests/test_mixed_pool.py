"""Tests for a single Switchboard holding both chat and messages deployments."""

from unittest.mock import patch

import pytest

from azure_switchboard import Model, OpenAIConfig, Switchboard, SwitchboardError

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
        assert mixed.deployments["oai1"].api == "chat"
        assert mixed.deployments["ant1"].api == "messages"

    def test_api_filter_scopes_selection(self, mixed: Switchboard):
        for _ in range(20):
            assert mixed.select_deployment(model="gpt-4o-mini").api == "chat"
            assert (
                mixed.select_deployment(model="claude-sonnet-5", api="messages").api
                == "messages"
            )

    def test_wrong_api_for_model_is_rejected(self, mixed: Switchboard):
        """A chat model must not be reachable over the messages API, and
        vice versa, even though both live in the same pool."""
        with pytest.raises(SwitchboardError, match="No deployments available"):
            mixed.select_deployment(model="gpt-4o-mini", api="messages")
        with pytest.raises(SwitchboardError, match="No deployments available"):
            mixed.select_deployment(model="claude-sonnet-5", api="chat")

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
        """Reusing a session_id across APIs must not hand back a deployment
        that doesn't speak the requested protocol."""
        chat = mixed.select_deployment(model="gpt-4o-mini", session_id="shared")
        msgs = mixed.select_deployment(
            model="claude-sonnet-5", api="messages", session_id="shared"
        )
        assert chat.api == "chat"
        assert msgs.api == "messages"
        assert chat is not msgs


class TestDispatch:
    async def test_both_surfaces_route_correctly(self, mixed: Switchboard):
        with patch(
            "azure_switchboard.chat.OpenAIDeployment.create",
            side_effect=chat_completion_mock(),
        ) as chat_mock:
            await mixed.create(**COMPLETION_PARAMS)
        with patch(
            "azure_switchboard.messages.AnthropicDeployment.messages",
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
            "azure_switchboard.messages.AnthropicDeployment.parse",
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
            "azure_switchboard.messages.AnthropicDeployment.messages", new=flaky
        ):
            assert await mixed.messages(**MESSAGE_PARAMS) == "ok"

        assert len(calls) == 2
        assert calls[0] != calls[1]

    async def test_unknown_model_raises_without_retrying(self, mixed: Switchboard):
        """SwitchboardError is excluded from the retry predicate."""
        with pytest.raises(SwitchboardError, match="No deployments available"):
            await mixed.messages(model="claude-opus-5", max_tokens=1, messages=[])


class TestConstruction:
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
