from unittest.mock import AsyncMock, patch

import pytest
import respx
from httpx import Request, Response
from openai import APIConnectionError, APITimeoutError, RateLimitError

from azure_switchboard import Foundry, OpenAIDeployment

from .conftest import (
    COMPLETION_BODY,
    COMPLETION_PARAMS,
    COMPLETION_RESPONSE,
    COMPLETION_STREAM_CHUNKS,
    chat_completion_mock,
    collect_chunks,
)


class TestOpenAIDeployment:
    """Chat Completions deployment tests."""

    @pytest.mark.mock_models("gpt-4o-mini")
    async def test_completion_charges_the_deployment(
        self, mock_client: respx.MockRouter, deployment: OpenAIDeployment
    ):
        """A response is charged against the deployment that served it, and a
        failed one still costs the preflight estimate."""

        deployment.client.max_retries = 0

        await deployment.create(**COMPLETION_BODY)
        usage = deployment.stats()
        assert usage.tpm.used == COMPLETION_RESPONSE.usage.total_tokens  # pyright: ignore[reportOptionalMemberAccess]
        assert usage.rpm.used == 1

        mock_client.routes["azure"].side_effect = Exception("test")
        with pytest.raises(APIConnectionError):
            await deployment.create(**COMPLETION_BODY)

        # nothing came back, so only the preflight estimate landed
        usage = deployment.stats()
        assert usage.tpm.used == 23
        assert usage.rpm.used == 2

    async def test_streaming(self, deployment: OpenAIDeployment):
        """Test streaming functionality.

        It's annoying to try to mock HTTP streaming responses so we cheat
        a little bit with an AsyncMock.
        """

        deployment.client.max_retries = 0

        with patch.object(
            deployment.client.chat.completions,
            "create",
            side_effect=chat_completion_mock(),
        ) as mock:
            stream = await deployment.create(stream=True, **COMPLETION_BODY)
            mock.assert_called_once()

            # verify basic behavior
            received_chunks, content = await collect_chunks(stream)
            assert len(received_chunks) == len(COMPLETION_STREAM_CHUNKS)
            assert content == "Hello, world!"

            # Verify token usage tracking
            usage = deployment.stats()
            assert usage.tpm.used == 20
            assert usage.rpm.used == 1

        # verify connection error handling marks down
        connection_error = APIConnectionError(
            request=Request(
                "POST", "https://test.openai.azure.com/openai/v1/chat/completions"
            )
        )
        with patch.object(
            deployment.client.chat.completions,
            "create",
            side_effect=connection_error,
        ) as mock:
            with pytest.raises(APIConnectionError):
                stream = await deployment.create(stream=True, **COMPLETION_BODY)
                async for _ in stream:
                    pass
            mock.assert_called_once()

            usage = deployment.stats()
            assert usage.tpm.used == 23
            assert usage.rpm.used == 2

        # A connection error cools the resource, not just this deployment
        assert deployment.resource.is_cooling()
        assert not deployment.is_cooling()
        deployment.resource.mark_up()

        # Test midstream exception handling — generic exceptions do NOT mark down
        with patch.object(
            deployment.client.chat.completions,
            "create",
            side_effect=chat_completion_mock(),
        ) as mock:
            stream = await deployment.create(stream=True, **COMPLETION_BODY)

            with patch.object(
                stream._self_model,  # type: ignore[reportAttributeAccessIssue]
                "spend_tokens",
                side_effect=Exception("asyncstream error"),
            ):
                with pytest.raises(Exception, match="asyncstream error"):
                    await collect_chunks(stream)
                assert mock.call_count == 1
                assert deployment.is_healthy()

        # Test midstream connection error marks down
        connection_error = APIConnectionError(
            request=Request(
                "POST", "https://test.openai.azure.com/openai/v1/chat/completions"
            )
        )

        async def connection_error_stream():
            yield COMPLETION_STREAM_CHUNKS[0]
            raise connection_error

        with patch.object(
            deployment.client.chat.completions,
            "create",
            new=AsyncMock(return_value=connection_error_stream()),
        ) as mock:
            stream = await deployment.create(stream=True, **COMPLETION_BODY)
            with pytest.raises(APIConnectionError):
                await collect_chunks(stream)
            assert mock.call_count == 1
            assert not deployment.is_healthy()

        deployment.mark_up()

        # Test midstream rate limit handling
        rate_limit_error = RateLimitError(
            "rate limited",
            response=Response(
                status_code=429,
                request=Request(
                    "POST", "https://test.openai.azure.com/openai/v1/chat/completions"
                ),
            ),
            body={"error": {"message": "rate limited"}},
        )

        async def rate_limited_stream():
            yield COMPLETION_STREAM_CHUNKS[0]
            raise rate_limit_error

        with patch.object(
            deployment.client.chat.completions,
            "create",
            new=AsyncMock(return_value=rate_limited_stream()),
        ) as mock:
            stream = await deployment.create(stream=True, **COMPLETION_BODY)
            with pytest.raises(RateLimitError):
                await collect_chunks(stream)
            assert mock.call_count == 1
            assert not deployment.is_healthy()

    async def test_mark_down(self, deployment: OpenAIDeployment):
        """Test model-level cooldown functionality."""

        model = deployment

        model.mark_down()
        assert not model.is_healthy()

        model.mark_up()
        assert model.is_healthy()

    async def test_usage(self, deployment: OpenAIDeployment, foundry: Foundry):
        """Test deployment-level counters"""

        # Reset and verify initial state
        for model in foundry.models.values():
            assert "tpm=0/" in str(model)

        # Test deployment-level usage
        model = deployment
        assert model.stats().tpm == (0, model.tpm_limit)
        assert model.stats().rpm == (0, model.rpm_limit)

        # Set and verify values
        model.spend_tokens(100)
        model.spend_request(5)
        assert model.stats().tpm == (100, model.tpm_limit)
        assert model.stats().rpm == (5, model.rpm_limit)

        # Reset and verify again
        foundry.reset_usage()
        assert model.stats().tpm == (0, model.tpm_limit)
        assert model.stats().rpm == (0, model.rpm_limit)

    async def test_utilization(self, deployment: OpenAIDeployment):
        """Test utilization calculation."""

        model = deployment

        # Check initial utilization (nonzero due to random splay)
        initial_util = model.util
        assert 0 <= initial_util < 0.02

        # Test token-based utilization
        model.spend_tokens(5000)  # 50% of TPM limit
        util_with_tokens = model.util
        assert 0.5 <= util_with_tokens < 0.52

        # Test request-based utilization
        model.reset_usage()
        model.spend_request(30)  # 50% of RPM limit
        util_with_requests = model.util
        assert 0.5 <= util_with_requests < 0.52

        # Test combined utilization (should take max of the two)
        model.reset_usage()
        model.spend_tokens(6000)  # 60% of TPM
        model.spend_request(30)  # 50% of RPM
        util_with_both = model.util
        assert 0.6 <= util_with_both < 0.62

        # Test unhealthy client
        model.mark_down()
        assert model.util == 1

    @pytest.mark.mock_models("gpt-4o-mini", "gpt-4o")
    async def test_multiple_models(
        self, mock_client: respx.MockRouter, foundry: Foundry
    ):
        """Deployments on one resource keep separate usage counters."""

        gpt4o = foundry.models["gpt-4o"]
        gpt4o_mini = foundry.models["gpt-4o-mini"]
        deployment = gpt4o_mini

        _ = await gpt4o.create(messages=COMPLETION_PARAMS["messages"])
        assert mock_client.routes["azure"].call_count == 1
        assert gpt4o.rpm_usage == 1
        assert gpt4o.tpm_usage > 0
        assert gpt4o_mini.tpm_usage == 0
        assert gpt4o_mini.rpm_usage == 0

        _ = await deployment.create(**COMPLETION_BODY)
        assert mock_client.routes["azure"].call_count == 2

        assert gpt4o.rpm_usage == 1
        assert gpt4o.tpm_usage > 0
        assert gpt4o_mini.tpm_usage > 0
        assert gpt4o_mini.rpm_usage == 1

    async def test_timeout_does_not_mark_down(self, deployment: OpenAIDeployment):
        """APITimeoutError should not mark the deployment down."""

        deployment.client.max_retries = 0
        timeout_error = APITimeoutError(
            request=Request(
                "POST", "https://test.openai.azure.com/openai/v1/chat/completions"
            )
        )

        with patch.object(
            deployment.client.chat.completions,
            "create",
            side_effect=timeout_error,
        ):
            with pytest.raises(APITimeoutError):
                await deployment.create(**COMPLETION_BODY)
            assert deployment.is_healthy()

        # Repeated timeouts should not pin utilization to 1
        for _ in range(5):
            with patch.object(
                deployment.client.chat.completions,
                "create",
                side_effect=timeout_error,
            ):
                with pytest.raises(APITimeoutError):
                    await deployment.create(**COMPLETION_BODY)

        assert deployment.is_healthy()
        assert deployment.util < 1

    async def test_timeout_does_not_mark_down_stream(
        self, deployment: OpenAIDeployment
    ):
        """Mid-stream APITimeoutError should not mark the deployment down."""

        deployment.client.max_retries = 0
        timeout_error = APITimeoutError(
            request=Request(
                "POST", "https://test.openai.azure.com/openai/v1/chat/completions"
            )
        )

        async def timeout_stream():
            yield COMPLETION_STREAM_CHUNKS[0]
            raise timeout_error

        with patch.object(
            deployment.client.chat.completions,
            "create",
            new=AsyncMock(return_value=timeout_stream()),
        ):
            stream = await deployment.create(stream=True, **COMPLETION_BODY)
            with pytest.raises(APITimeoutError):
                await collect_chunks(stream)
            assert deployment.is_healthy()

    async def test_rate_limit_marks_down(self, deployment: OpenAIDeployment):
        """Rate limit errors should mark the model down and re-raise."""

        deployment.client.max_retries = 0
        rate_limit_error = RateLimitError(
            "rate limited",
            response=Response(
                status_code=429,
                request=Request(
                    "POST", "https://test.openai.azure.com/openai/v1/chat/completions"
                ),
            ),
            body={"error": {"message": "rate limited"}},
        )

        with patch.object(
            deployment.client.chat.completions,
            "create",
            side_effect=rate_limit_error,
        ) as mock:
            with pytest.raises(RateLimitError):
                await deployment.create(**COMPLETION_BODY)
            assert mock.call_count == 1
            assert not deployment.is_healthy()


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
