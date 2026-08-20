"""Everything specific to the Messages deployment.

Mirrors tests/test_openai_deployment.py class for class: the two are symmetric
implementations, so what is asserted about one is asserted about the other.
Quota arithmetic and cooldown mechanics are shared by every deployment and live
in test_deployment.py instead.
"""

from unittest.mock import AsyncMock, patch

import pytest
from anthropic import APIConnectionError, APITimeoutError, RateLimitError
from httpx import Request, Response
from pydantic import BaseModel

from azure_switchboard import AnthropicDeployment, Foundry
from azure_switchboard.anthropic_deployment import _content_len
from azure_switchboard.resource import Resource

from .conftest import (
    MESSAGE_BODY,
    MESSAGE_RESPONSE,
    MESSAGE_STREAM_EVENTS,
    anthropic_foundry,
    assert_cooldown_scope,
    collect_events,
    message_mock,
)


class Weather(BaseModel):
    city: str


def _request() -> Request:
    return Request("POST", "https://test.services.ai.azure.com/anthropic/v1/messages")


def _rate_limit() -> RateLimitError:
    return RateLimitError(
        "rate limited",
        response=Response(429, request=_request()),
        body=None,
    )


COOLDOWN_CASES = [
    (_rate_limit(), "model"),
    (APIConnectionError(request=_request()), "resource"),
    (APITimeoutError(request=_request()), None),
]
COOLDOWN_IDS = ["rate_limit", "connection", "timeout"]


class TestAnthropicEndpoint:
    def test_resource_name_builds_the_anthropic_url(self):
        deployment = anthropic_foundry("my-res").models["claude-sonnet-5"]
        assert str(deployment.client.base_url).startswith(
            "https://my-res.services.ai.azure.com/anthropic"
        )

    def test_endpoint_override_wins(self):
        resource = Foundry(
            name="d",
            api_key="k",
            models=[
                AnthropicDeployment(
                    name="claude-sonnet-5", endpoint="https://custom.example/anthropic/"
                )
            ],
        )
        client = resource.models["claude-sonnet-5"].client
        assert "custom.example" in str(client.base_url)

    def test_resource_client_is_the_azure_variant(self):
        """AsyncAnthropicFoundry sends Azure's api-key header, so it is not
        interchangeable with the first-party client."""
        deployment = anthropic_foundry("my-res").models["claude-sonnet-5"]
        assert type(deployment.client).__name__ == "AsyncAnthropicFoundry"

    def test_first_party_deployment_derives_no_url(self, monkeypatch):
        """A Resource with no base derives no URL, so the SDK falls back to its
        vendor default and the plain client is the right one."""
        monkeypatch.setenv("ANTHROPIC_API_KEY", "k")
        resource = Resource(
            "first-party-anthropic", models=[AnthropicDeployment("claude-sonnet-5")]
        )
        deployment = resource.models["claude-sonnet-5"]
        assert deployment.url is None
        assert type(deployment.client).__name__ == "AsyncAnthropic"


class TestAnthropicDeployment:
    async def test_create_charges_the_deployment(
        self, anthropic_deployment: AnthropicDeployment
    ):
        with patch.object(
            anthropic_deployment.client.messages,
            "create",
            new=AsyncMock(return_value=MESSAGE_RESPONSE),
        ):
            await anthropic_deployment.create(**MESSAGE_BODY)

        # input + output, since the Messages API reports no total
        usage = anthropic_deployment.stats()
        assert usage.tpm.used == 20
        assert usage.rpm.used == 1

    async def test_streaming_accumulates_usage(
        self, anthropic_deployment: AnthropicDeployment
    ):
        """Input lands on message_start; output_tokens is cumulative per delta."""
        with patch.object(
            anthropic_deployment.client.messages, "create", side_effect=message_mock()
        ):
            stream = await anthropic_deployment.create(stream=True, **MESSAGE_BODY)
            received, content = await collect_events(stream)

        assert len(received) == len(MESSAGE_STREAM_EVENTS)
        assert content == "Hello, world!"

        # 12 input + 9 output, with the cumulative delta counted once
        usage = anthropic_deployment.stats()
        assert usage.tpm.used == 21
        assert usage.rpm.used == 1

    async def test_parse_charges_the_deployment(
        self, anthropic_deployment: AnthropicDeployment
    ):
        """Structured output is charged the same as any other response."""
        with patch.object(
            anthropic_deployment.client.messages,
            "parse",
            new=AsyncMock(return_value=MESSAGE_RESPONSE),
        ):
            await anthropic_deployment.parse(
                output_format=Weather,
                max_tokens=16,
                messages=[{"role": "user", "content": "Weather in Paris?"}],
            )

        assert anthropic_deployment.stats().tpm.used == 20
        assert anthropic_deployment.stats().rpm.used == 1

    async def test_response_without_usage_only_spends_the_estimate(
        self, anthropic_deployment: AnthropicDeployment
    ):
        """Some responses carry no usage block; the preflight estimate stands."""
        no_usage = MESSAGE_RESPONSE.model_copy(update={"usage": None})
        with patch.object(
            anthropic_deployment.client.messages,
            "create",
            new=AsyncMock(return_value=no_usage),
        ):
            await anthropic_deployment.create(**MESSAGE_BODY)

        # "Hello, world!" is 13 chars -> 3 tokens
        assert anthropic_deployment.tpm_usage == 3


class TestAnthropicErrorHandling:
    """Mirrors the chat path: rate limits cool a deployment, connection errors
    cool its whole resource, timeouts cool nothing."""

    @pytest.mark.parametrize("error,scope", COOLDOWN_CASES, ids=COOLDOWN_IDS)
    async def test_cooldown_policy(
        self, anthropic_deployment: AnthropicDeployment, error, scope
    ):
        with patch.object(
            anthropic_deployment.client.messages, "create", side_effect=error
        ):
            with pytest.raises(type(error)):
                await anthropic_deployment.create(**MESSAGE_BODY)
        assert_cooldown_scope(anthropic_deployment, scope)

    @pytest.mark.parametrize("error,scope", COOLDOWN_CASES, ids=COOLDOWN_IDS)
    async def test_cooldown_policy_on_stream(
        self, anthropic_deployment: AnthropicDeployment, error, scope
    ):
        async def _raising_stream(*args, **kwargs):
            raise error
            yield  # pragma: no cover

        with patch.object(
            anthropic_deployment.client.messages,
            "create",
            new=AsyncMock(side_effect=lambda *a, **k: _raising_stream()),
        ):
            stream = await anthropic_deployment.create(stream=True, **MESSAGE_BODY)
            with pytest.raises(type(error)):
                await collect_events(stream)
        assert_cooldown_scope(anthropic_deployment, scope)

    @pytest.mark.parametrize("error,scope", COOLDOWN_CASES, ids=COOLDOWN_IDS)
    async def test_cooldown_policy_on_parse(
        self, anthropic_deployment: AnthropicDeployment, error, scope
    ):
        with patch.object(
            anthropic_deployment.client.messages, "parse", side_effect=error
        ):
            with pytest.raises(type(error)):
                await anthropic_deployment.parse(
                    output_format=Weather,
                    max_tokens=16,
                    messages=[{"role": "user", "content": "hi"}],
                )
        assert_cooldown_scope(anthropic_deployment, scope)

    async def test_an_error_the_sdk_does_not_name_cools_nothing(
        self, anthropic_deployment: AnthropicDeployment
    ):
        """A failure in our own accounting is not the upstream's fault."""
        with patch.object(
            anthropic_deployment.client.messages, "create", side_effect=message_mock()
        ):
            stream = await anthropic_deployment.create(stream=True, **MESSAGE_BODY)
            with patch.object(
                anthropic_deployment,
                "spend_tokens",
                side_effect=Exception("accounting"),
            ):
                with pytest.raises(Exception, match="accounting"):
                    await collect_events(stream)

        assert_cooldown_scope(anthropic_deployment, None)


class TestAnthropicTokenEstimate:
    """Charged before the request goes out, so concurrent selections see it.

    Unlike the chat path, content may be a list of blocks and the system prompt
    lives outside messages.
    """

    def test_content_len(self):
        assert _content_len("hello") == 5
        assert _content_len([{"type": "text", "text": "hello"}]) == 5
        assert _content_len([{"type": "image", "source": {}}]) == 0
        assert _content_len(None) == 0

    def test_estimate_counts_blocks_and_system(
        self, anthropic_deployment: AnthropicDeployment
    ):
        estimate = anthropic_deployment._estimate_token_usage(
            {
                "system": "s" * 40,
                "messages": [
                    {"role": "user", "content": "u" * 40},
                    {
                        "role": "assistant",
                        "content": [{"type": "text", "text": "a" * 40}],
                    },
                ],
            }
        )
        assert estimate == 30  # 120 chars // 4

    async def test_a_failed_request_still_costs_the_estimate(
        self, anthropic_deployment: AnthropicDeployment
    ):
        """It is spent before the call goes out, so a failure leaves it charged
        rather than refunded."""
        with patch.object(
            anthropic_deployment.client.messages,
            "create",
            side_effect=APITimeoutError(request=_request()),
        ):
            with pytest.raises(APITimeoutError):
                await anthropic_deployment.create(**MESSAGE_BODY)

        # "Hello, world!" is 13 chars -> 3 tokens
        assert anthropic_deployment.tpm_usage == 3
        assert anthropic_deployment.rpm_usage == 1


class TestAnthropicStreamHelper:
    """anthropic.messages.stream() accumulates a terminal Message, so usage is
    charged from what it accumulated rather than tapped per event."""

    def test_reconcile_charges_the_snapshot(
        self, anthropic_deployment: AnthropicDeployment
    ):
        class _Stream:
            current_message_snapshot = MESSAGE_RESPONSE

        anthropic_deployment.reconcile_stream(_Stream(), offset=3)  # type: ignore[arg-type]
        # 12 input + 8 output, less the preflight estimate already spent
        assert anthropic_deployment.tpm_usage == 17

    def test_reconcile_is_a_noop_when_nothing_accumulated(
        self, anthropic_deployment: AnthropicDeployment
    ):
        """The SDK asserts on the snapshot before anything is consumed."""

        class _Stream:
            @property
            def current_message_snapshot(self):
                raise AssertionError("nothing consumed")

        anthropic_deployment.reconcile_stream(_Stream(), offset=3)  # type: ignore[arg-type]
        assert anthropic_deployment.tpm_usage == 0
