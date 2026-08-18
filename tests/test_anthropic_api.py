from unittest.mock import AsyncMock, patch

import pytest
from anthropic import APIConnectionError, APITimeoutError, RateLimitError
from httpx import Request, Response

from azure_switchboard import AnthropicConfig, Model, SwitchboardError
from azure_switchboard.anthropic_api import AnthropicDeployment, _content_len

from .conftest import (
    MESSAGE_PARAMS,
    MESSAGE_RESPONSE,
    MESSAGE_STREAM_EVENTS,
    collect_events,
    message_mock,
)


def _request() -> Request:
    return Request("POST", "https://test.services.ai.azure.com/anthropic/v1/messages")


def _rate_limit() -> RateLimitError:
    return RateLimitError(
        "rate limited",
        response=Response(429, request=_request()),
        body=None,
    )


class TestAnthropicConfig:
    def test_resource_builds_foundry_url(self):
        client = AnthropicConfig(name="d", resource="my-res", api_key="k").get_client()
        assert str(client.base_url).startswith(
            "https://my-res.services.ai.azure.com/anthropic"
        )

    def test_base_url_overrides_resource(self):
        client = AnthropicConfig(
            name="d", base_url="https://custom.example/anthropic/", api_key="k"
        ).get_client()
        assert "custom.example" in str(client.base_url)

    def test_requires_an_endpoint(self):
        with pytest.raises(SwitchboardError, match="resource or base_url"):
            AnthropicConfig(name="d", api_key="k").get_client()


class TestAnthropicDeployment:
    async def test_init(self, anthropic_deployment: AnthropicDeployment):
        assert anthropic_deployment.api == "messages"
        assert anthropic_deployment.client is not None
        assert anthropic_deployment.model("claude-sonnet-5") is not None

    async def test_unknown_model_rejected(
        self, anthropic_deployment: AnthropicDeployment
    ):
        with pytest.raises(SwitchboardError, match="not configured"):
            await anthropic_deployment.messages(
                model="gpt-4o", max_tokens=1, messages=[]
            )

    async def test_messages(self, anthropic_deployment: AnthropicDeployment):
        with patch.object(
            anthropic_deployment.client.messages, "create", side_effect=message_mock()
        ) as mock:
            response = await anthropic_deployment.messages(**MESSAGE_PARAMS)
            mock.assert_called_once()
            assert response == MESSAGE_RESPONSE

        # input_tokens + output_tokens, since there is no total_tokens
        usage = anthropic_deployment.model("claude-sonnet-5").stats()
        assert usage.tpm.startswith("20/")
        assert usage.rpm.startswith("1/")

    async def test_streaming_accumulates_usage(
        self, anthropic_deployment: AnthropicDeployment
    ):
        """Input lands on message_start; output_tokens is cumulative per delta."""
        with patch.object(
            anthropic_deployment.client.messages, "create", side_effect=message_mock()
        ):
            stream = await anthropic_deployment.messages(stream=True, **MESSAGE_PARAMS)
            received, content = await collect_events(stream)

        assert len(received) == len(MESSAGE_STREAM_EVENTS)
        assert content == "Hello, world!"

        # 12 input + 9 output, with the cumulative delta counted once
        usage = anthropic_deployment.model("claude-sonnet-5").stats()
        assert usage.tpm.startswith("21/")
        assert usage.rpm.startswith("1/")

    async def test_parse(self, anthropic_deployment: AnthropicDeployment):
        from pydantic import BaseModel

        class Weather(BaseModel):
            city: str

        with patch.object(
            anthropic_deployment.client.messages, "parse", side_effect=message_mock()
        ) as mock:
            await anthropic_deployment.parse(
                model="claude-sonnet-5",
                output_format=Weather,
                max_tokens=1024,
                messages=[{"role": "user", "content": "Weather in Paris?"}],
            )
            assert mock.call_args.kwargs["output_format"] is Weather

        usage = anthropic_deployment.model("claude-sonnet-5").stats()
        assert usage.tpm.startswith("20/")


class TestAnthropicErrorHandling:
    """Mirrors the chat path: rate limits and connection errors cool a model
    down, timeouts do not."""

    @pytest.mark.parametrize(
        "error,should_mark_down",
        [
            (_rate_limit(), True),
            (APIConnectionError(request=_request()), True),
            (APITimeoutError(request=_request()), False),
        ],
        ids=["rate_limit", "connection", "timeout"],
    )
    async def test_cooldown_policy(
        self, anthropic_deployment: AnthropicDeployment, error, should_mark_down
    ):
        model = anthropic_deployment.model("claude-sonnet-5")
        with patch.object(
            anthropic_deployment.client.messages, "create", side_effect=error
        ):
            with pytest.raises(type(error)):
                await anthropic_deployment.messages(**MESSAGE_PARAMS)
        assert model.is_cooling() is should_mark_down

    @pytest.mark.parametrize(
        "error,should_mark_down",
        [
            (_rate_limit(), True),
            (APIConnectionError(request=_request()), True),
            (APITimeoutError(request=_request()), False),
        ],
        ids=["rate_limit", "connection", "timeout"],
    )
    async def test_cooldown_policy_on_stream(
        self, anthropic_deployment: AnthropicDeployment, error, should_mark_down
    ):
        model = anthropic_deployment.model("claude-sonnet-5")

        async def _raising_stream(*args, **kwargs):
            raise error
            yield  # pragma: no cover

        with patch.object(
            anthropic_deployment.client.messages,
            "create",
            new=AsyncMock(side_effect=lambda *a, **k: _raising_stream()),
        ):
            stream = await anthropic_deployment.messages(stream=True, **MESSAGE_PARAMS)
            with pytest.raises(type(error)):
                await collect_events(stream)
        assert model.is_cooling() is should_mark_down


class TestTokenEstimate:
    """Content may be a plain string or a list of blocks, and the system
    prompt lives outside messages."""

    def test_content_len(self):
        assert _content_len("hello") == 5
        assert _content_len([{"type": "text", "text": "hello"}]) == 5
        assert _content_len([{"type": "image", "source": {}}]) == 0
        assert _content_len(None) == 0

    def test_estimate_counts_blocks_and_system(self):
        d = AnthropicDeployment(
            AnthropicConfig(
                name="d",
                resource="r",
                api_key="k",
                models=[Model(name="claude-sonnet-5")],
            )
        )
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
        "error,should_mark_down",
        [
            (_rate_limit(), True),
            (APIConnectionError(request=_request()), True),
            (APITimeoutError(request=_request()), False),
        ],
        ids=["rate_limit", "connection", "timeout"],
    )
    async def test_cooldown_policy_on_parse(
        self, anthropic_deployment: AnthropicDeployment, error, should_mark_down
    ):
        from pydantic import BaseModel

        class Weather(BaseModel):
            city: str

        model = anthropic_deployment.model("claude-sonnet-5")
        with patch.object(
            anthropic_deployment.client.messages, "parse", side_effect=error
        ):
            with pytest.raises(type(error)):
                await anthropic_deployment.parse(
                    model="claude-sonnet-5",
                    output_format=Weather,
                    max_tokens=16,
                    messages=[{"role": "user", "content": "hi"}],
                )
        assert model.is_cooling() is should_mark_down

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
            await anthropic_deployment.messages(**MESSAGE_PARAMS)

        # "Hello, world!" is 13 chars -> 3 tokens
        assert anthropic_deployment.model("claude-sonnet-5").tpm_usage == 3
