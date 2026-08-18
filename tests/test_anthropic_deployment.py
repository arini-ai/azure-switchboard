from unittest.mock import AsyncMock, patch

import pytest
from anthropic import APIConnectionError, APITimeoutError, RateLimitError
from httpx import Request, Response
from pydantic import BaseModel

from azure_switchboard import AnthropicDeployment, Foundry
from azure_switchboard.foundry import FirstParty
from azure_switchboard.anthropic_deployment import _content_len

from .conftest import (
    MESSAGE_BODY,
    MESSAGE_RESPONSE,
    MESSAGE_STREAM_EVENTS,
    anthropic_foundry,
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


def _assert_cooldown_scope(deployment: AnthropicDeployment, scope: str | None) -> None:
    """A 429 is one deployment's quota; a connection error is the whole host."""
    assert deployment.is_cooling() is (scope == "model")
    assert deployment.foundry.is_cooling() is (scope == "foundry")
    # either scope takes this deployment out of selection
    assert deployment.is_healthy() is (scope is None)


class TestAnthropicEndpoint:
    def test_foundry_name_builds_the_anthropic_url(self):
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

    def test_foundry_client_is_the_azure_variant(self):
        """AsyncAnthropicFoundry sends Azure's api-key header, so it is not
        interchangeable with the first-party client."""
        deployment = anthropic_foundry("my-res").models["claude-sonnet-5"]
        assert type(deployment.client).__name__ == "AsyncAnthropicFoundry"

    def test_first_party_client_is_the_plain_variant(self, monkeypatch):
        monkeypatch.setenv("ANTHROPIC_API_KEY", "k")
        resource = FirstParty(
            "anthropic", models=[AnthropicDeployment("claude-sonnet-5")]
        )
        deployment = resource.models["claude-sonnet-5"]
        assert deployment.url is None
        assert type(deployment.client).__name__ == "AsyncAnthropic"


class TestAnthropicDeployment:
    async def test_init(
        self, anthropic_deployment: AnthropicDeployment, anthropic_resource
    ):
        assert anthropic_deployment.name == "claude-sonnet-5"
        assert anthropic_deployment.foundry is anthropic_resource
        assert anthropic_deployment.client is not None

    async def test_messages(self, anthropic_deployment: AnthropicDeployment):
        with patch.object(
            anthropic_deployment.client.messages, "create", side_effect=message_mock()
        ) as mock:
            response = await anthropic_deployment.create(**MESSAGE_BODY)
            mock.assert_called_once()
            assert response == MESSAGE_RESPONSE

        # input_tokens + output_tokens, since there is no total_tokens
        usage = anthropic_deployment.stats()
        assert usage.tpm.startswith("20/")
        assert usage.rpm.startswith("1/")

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
        assert usage.tpm.startswith("21/")
        assert usage.rpm.startswith("1/")

    async def test_parse(self, anthropic_deployment: AnthropicDeployment):
        with patch.object(
            anthropic_deployment.client.messages, "parse", side_effect=message_mock()
        ) as mock:
            await anthropic_deployment.parse(
                output_format=Weather,
                max_tokens=1024,
                messages=[{"role": "user", "content": "Weather in Paris?"}],
            )
            assert mock.call_args.kwargs["output_format"] is Weather

        usage = anthropic_deployment.stats()
        assert usage.tpm.startswith("20/")


class TestAnthropicErrorHandling:
    """Mirrors the chat path: rate limits and connection errors cool a model
    down, timeouts do not."""

    @pytest.mark.parametrize(
        "error,scope",
        [
            (_rate_limit(), "model"),
            (APIConnectionError(request=_request()), "foundry"),
            (APITimeoutError(request=_request()), None),
        ],
        ids=["rate_limit", "connection", "timeout"],
    )
    async def test_cooldown_policy(
        self, anthropic_deployment: AnthropicDeployment, error, scope
    ):
        with patch.object(
            anthropic_deployment.client.messages, "create", side_effect=error
        ):
            with pytest.raises(type(error)):
                await anthropic_deployment.create(**MESSAGE_BODY)
        _assert_cooldown_scope(anthropic_deployment, scope)

    @pytest.mark.parametrize(
        "error,scope",
        [
            (_rate_limit(), "model"),
            (APIConnectionError(request=_request()), "foundry"),
            (APITimeoutError(request=_request()), None),
        ],
        ids=["rate_limit", "connection", "timeout"],
    )
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
        _assert_cooldown_scope(anthropic_deployment, scope)


class TestTokenEstimate:
    """Content may be a plain string or a list of blocks, and the system
    prompt lives outside messages."""

    def test_content_len(self):
        assert _content_len("hello") == 5
        assert _content_len([{"type": "text", "text": "hello"}]) == 5
        assert _content_len([{"type": "image", "source": {}}]) == 0
        assert _content_len(None) == 0

    def test_estimate_counts_blocks_and_system(self):
        d = anthropic_foundry("d").models["claude-sonnet-5"]
        estimate = d._estimate_token_usage(
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


class TestParseErrorHandling:
    @pytest.mark.parametrize(
        "error,scope",
        [
            (_rate_limit(), "model"),
            (APIConnectionError(request=_request()), "foundry"),
            (APITimeoutError(request=_request()), None),
        ],
        ids=["rate_limit", "connection", "timeout"],
    )
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
        _assert_cooldown_scope(anthropic_deployment, scope)

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
