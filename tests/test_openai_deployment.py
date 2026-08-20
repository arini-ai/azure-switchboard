"""Everything specific to the Chat Completions deployment.

Mirrors tests/test_anthropic_deployment.py class for class: the two are
symmetric implementations, so what is asserted about one is asserted about the
other. Quota arithmetic and cooldown mechanics are shared by every deployment
and live in test_deployment.py instead.
"""

from unittest.mock import AsyncMock, patch

import pytest
from httpx import Request, Response
from openai import (
    APIConnectionError,
    APITimeoutError,
    AsyncOpenAI,
    RateLimitError,
)

from azure_switchboard import Foundry, OpenAIDeployment
from azure_switchboard.resource import Resource

from .conftest import (
    COMPLETION_BODY,
    COMPLETION_RESPONSE,
    COMPLETION_STREAM_CHUNKS,
    PARSED_COMPLETION_BODY,
    PARSED_RESPONSE,
    WeatherResult,
    assert_cooldown_scope,
    chat_completion_mock,
    collect_chunks,
    openai_foundry,
)


def _request() -> Request:
    return Request("POST", "https://test.services.ai.azure.com/openai/v1/completions")


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


class TestOpenAIEndpoint:
    def test_resource_name_builds_the_chat_url(self):
        deployment = openai_foundry("my-res").models["gpt-4o-mini"]
        assert str(deployment.client.base_url).startswith(
            "https://my-res.services.ai.azure.com/openai/v1"
        )

    def test_endpoint_override_wins(self):
        resource = Foundry(
            name="d",
            api_key="k",
            models=[
                OpenAIDeployment(
                    name="gpt-4o-mini", endpoint="https://custom.example/openai/v1/"
                )
            ],
        )
        client = resource.models["gpt-4o-mini"].client
        assert "custom.example" in str(client.base_url)

    def test_first_party_deployment_derives_no_url(self, monkeypatch):
        """A Resource with no base derives no URL, so the SDK falls back to its
        vendor default."""
        monkeypatch.setenv("OPENAI_API_KEY", "k")
        resource = Resource("first-party-openai", models=[OpenAIDeployment("gpt-4o")])
        deployment = resource.models["gpt-4o"]
        assert deployment.url is None
        assert type(deployment.client) is AsyncOpenAI


class TestOpenAIDeployment:
    @pytest.mark.mock_models("gpt-4o-mini")
    async def test_create_charges_the_deployment(self, deployment: OpenAIDeployment):
        await deployment.create(**COMPLETION_BODY)

        usage = deployment.stats()
        assert usage.tpm.used == COMPLETION_RESPONSE.usage.total_tokens  # pyright: ignore[reportOptionalMemberAccess]
        assert usage.rpm.used == 1

    async def test_streaming_accumulates_usage(self, deployment: OpenAIDeployment):
        """Only the final chunk carries usage, so it is tapped as chunks pass."""
        with patch.object(
            deployment.client.chat.completions,
            "create",
            side_effect=chat_completion_mock(),
        ):
            stream = await deployment.create(stream=True, **COMPLETION_BODY)
            received, content = await collect_chunks(stream)

        assert len(received) == len(COMPLETION_STREAM_CHUNKS)
        assert content == "Hello, world!"

        usage = deployment.stats()
        assert usage.tpm.used == 20
        assert usage.rpm.used == 1

    async def test_parse_charges_the_deployment(self, deployment: OpenAIDeployment):
        """Structured output is charged the same as any other response."""
        with patch.object(
            deployment.client.chat.completions,
            "parse",
            new=AsyncMock(return_value=PARSED_RESPONSE),
        ):
            parsed = await deployment.parse(**PARSED_COMPLETION_BODY)

        assert parsed.choices[0].message.parsed == WeatherResult(
            city="Paris", temperature=18.5, unit="celsius"
        )
        assert deployment.stats().tpm.used == PARSED_RESPONSE.usage.total_tokens  # pyright: ignore[reportOptionalMemberAccess]
        assert deployment.stats().rpm.used == 1

    async def test_response_without_usage_only_spends_the_estimate(
        self, deployment: OpenAIDeployment
    ):
        """Some responses carry no usage block; the preflight estimate stands."""
        no_usage = COMPLETION_RESPONSE.model_copy(update={"usage": None})
        with patch.object(
            deployment.client.chat.completions,
            "create",
            new=AsyncMock(return_value=no_usage),
        ):
            await deployment.create(**COMPLETION_BODY)

        # "Hello, world!" is 13 chars -> 3 tokens
        assert deployment.tpm_usage == 3


class TestOpenAIErrorHandling:
    """Mirrors the messages path: rate limits cool a deployment, connection
    errors cool its whole resource, timeouts cool nothing."""

    @pytest.mark.parametrize("error,scope", COOLDOWN_CASES, ids=COOLDOWN_IDS)
    async def test_cooldown_policy(self, deployment: OpenAIDeployment, error, scope):
        with patch.object(
            deployment.client.chat.completions, "create", side_effect=error
        ):
            with pytest.raises(type(error)):
                await deployment.create(**COMPLETION_BODY)
        assert_cooldown_scope(deployment, scope)

    @pytest.mark.parametrize("error,scope", COOLDOWN_CASES, ids=COOLDOWN_IDS)
    async def test_cooldown_policy_on_stream(
        self, deployment: OpenAIDeployment, error, scope
    ):
        async def _raising_stream(*args, **kwargs):
            raise error
            yield  # pragma: no cover

        with patch.object(
            deployment.client.chat.completions,
            "create",
            new=AsyncMock(side_effect=lambda *a, **k: _raising_stream()),
        ):
            stream = await deployment.create(stream=True, **COMPLETION_BODY)
            with pytest.raises(type(error)):
                await collect_chunks(stream)
        assert_cooldown_scope(deployment, scope)

    @pytest.mark.parametrize("error,scope", COOLDOWN_CASES, ids=COOLDOWN_IDS)
    async def test_cooldown_policy_on_parse(
        self, deployment: OpenAIDeployment, error, scope
    ):
        with patch.object(
            deployment.client.chat.completions, "parse", side_effect=error
        ):
            with pytest.raises(type(error)):
                await deployment.parse(
                    response_format=WeatherResult,
                    messages=[{"role": "user", "content": "hi"}],
                )
        assert_cooldown_scope(deployment, scope)

    async def test_an_error_the_sdk_does_not_name_cools_nothing(
        self, deployment: OpenAIDeployment
    ):
        """A failure in our own accounting is not the upstream's fault."""
        with patch.object(
            deployment.client.chat.completions,
            "create",
            side_effect=chat_completion_mock(),
        ):
            stream = await deployment.create(stream=True, **COMPLETION_BODY)
            with patch.object(
                deployment, "spend_tokens", side_effect=Exception("accounting")
            ):
                with pytest.raises(Exception, match="accounting"):
                    await collect_chunks(stream)

        assert_cooldown_scope(deployment, None)


class TestOpenAITokenEstimate:
    """Charged before the request goes out, so concurrent selections see it."""

    def test_estimate_counts_message_content(self, deployment: OpenAIDeployment):
        estimate = deployment._estimate_token_usage(
            {
                "messages": [
                    {"role": "user", "content": "u" * 40},
                    {"role": "assistant", "content": "a" * 40},
                ]
            }
        )
        assert estimate == 20  # 80 chars // 4

    async def test_a_failed_request_still_costs_the_estimate(
        self, deployment: OpenAIDeployment
    ):
        """It is spent before the call goes out, so a failure leaves it charged
        rather than refunded."""
        with patch.object(
            deployment.client.chat.completions,
            "create",
            side_effect=APITimeoutError(request=_request()),
        ):
            with pytest.raises(APITimeoutError):
                await deployment.create(**COMPLETION_BODY)

        # "Hello, world!" is 13 chars -> 3 tokens
        assert deployment.tpm_usage == 3
        assert deployment.rpm_usage == 1


class TestOpenAIStreamHelper:
    """openai.chat.completions.stream() accumulates a terminal completion, so
    usage is charged from what it accumulated rather than tapped per chunk."""

    def test_reconcile_charges_the_snapshot(self, deployment: OpenAIDeployment):
        class _Stream:
            current_completion_snapshot = COMPLETION_RESPONSE

        deployment.reconcile_stream(_Stream(), offset=5)  # type: ignore[arg-type]
        # 20 total, less the preflight estimate already spent
        assert deployment.tpm_usage == 15

    def test_reconcile_is_a_noop_without_usage(self, deployment: OpenAIDeployment):
        """Usage is absent unless the request asked for it."""

        class _Stream:
            current_completion_snapshot = COMPLETION_RESPONSE.model_copy(
                update={"usage": None}
            )

        deployment.reconcile_stream(_Stream(), offset=5)  # type: ignore[arg-type]
        assert deployment.tpm_usage == 0
