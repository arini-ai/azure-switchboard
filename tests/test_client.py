import asyncio
from unittest.mock import patch

import openai
import pytest
import respx
from fixtures import (
    MOCK_COMPLETION,
    MOCK_STREAM_CHUNKS,
    TEST_DEPLOYMENT_1,
)
from httpx import Response, TimeoutException
from openai.types.chat import ChatCompletion
from utils import BaseTestCase, create_mock_openai_client

from azure_switchboard import Client, ModelState
from azure_switchboard.client import SwitchboardClientError, default_client_factory


@pytest.fixture
def mock_client():
    """Create a Client instance with a basic mock."""
    try:
        openai_mock = create_mock_openai_client()
        client = Client(TEST_DEPLOYMENT_1, client=openai_mock)
        yield client
    finally:
        client.reset_usage()
        for model in client.config.models.values():
            model.reset_cooldown()


class TestClient(BaseTestCase):
    """Client functionality tests."""

    def _get_model(self, client: Client, model: str) -> ModelState:
        return client.config.models[model]

    async def test_completion(self, mock_client: Client):
        """Test basic chat completion functionality."""

        response = await mock_client.create(**self.basic_args)
        assert mock_client.client.chat.completions.create.call_count == 1
        assert response == MOCK_COMPLETION

        # Check token usage tracking
        model = self._get_model(mock_client, "gpt-4o-mini")
        usage = model.get_usage()
        assert usage["tpm"] == "11/10000"
        assert usage["rpm"] == "1/60"

        # Test exception handling
        mock_client.client.chat.completions.create.side_effect = Exception("test")
        with pytest.raises(Exception, match="test"):
            await mock_client.create(**self.basic_args)
        assert mock_client.client.chat.completions.create.call_count == 2

        # account for preflight estimate
        usage = model.get_usage()
        assert usage["tpm"] == "14/10000"
        assert usage["rpm"] == "2/60"

    async def test_streaming(self, mock_client: Client):
        """Test streaming functionality."""

        stream = await mock_client.create(stream=True, **self.basic_args)
        assert stream is not None

        # Verify stream options
        create_mock = mock_client.client.chat.completions.create
        assert create_mock.call_args.kwargs.get("stream") is True
        assert (
            create_mock.call_args.kwargs.get("stream_options", {}).get("include_usage")
            is True
        )

        # Collect chunks and verify content
        received_chunks, content = await self.collect_chunks(stream)

        # Verify chunk handling
        assert len(received_chunks) == len(MOCK_STREAM_CHUNKS)
        assert content == "Hello, world!"

        # Verify token usage tracking
        model = self._get_model(mock_client, "gpt-4o-mini")
        usage = model.get_usage()
        assert usage["tpm"] == "20/10000"
        assert usage["rpm"] == "1/60"

        # Test exception handling
        mock_client.client.chat.completions.create.side_effect = Exception("test")
        with pytest.raises(Exception, match="test"):
            stream = await mock_client.create(stream=True, **self.basic_args)
            async for _ in stream:
                pass
        assert mock_client.client.chat.completions.create.call_count == 2
        usage = model.get_usage()
        assert usage["tpm"] == "23/10000"
        assert usage["rpm"] == "2/60"

        # Test midstream exception handling
        # shim through spend tokens to raise the exception
        mock_client.client.chat.completions.create.side_effect = None
        stream = await mock_client.create(stream=True, **self.basic_args)
        with patch.object(stream.__wrapped__, "__aiter__") as mock:  # type: ignore
            mock.side_effect = Exception("spend_tokens error")
            with pytest.raises(Exception, match="spend_tokens error"):
                assert mock_client.client.chat.completions.create.call_count == 3
                await self.collect_chunks(stream)
            assert mock.call_count == 1
            assert not model.is_healthy()

        model.reset_cooldown()
        assert model.is_healthy()

    async def test_cooldown(self, mock_client: Client):
        """Test model-level cooldown functionality."""
        model = self._get_model(mock_client, "gpt-4o-mini")

        model.cooldown()
        assert not model.is_healthy()

        model.reset_cooldown()
        assert model.is_healthy()

    async def test_valid_model(self, mock_client: Client):
        """Test that an invalid model raises an error."""
        with pytest.raises(SwitchboardClientError, match="gpt-fake not configured"):
            await mock_client.create(model="gpt-fake", messages=[])

    async def test_usage(self, mock_client: Client):
        """Test client-level counters"""
        # Reset and verify initial state
        for model in mock_client.config.models.values():
            model_str = str(model)
            assert "tpm=0/10000" in model_str
            assert "rpm=0/60" in model_str

        # Test client-level usage
        usage = mock_client.get_usage()
        assert usage["gpt-4o-mini"]["tpm"] == "0/10000"
        assert usage["gpt-4o-mini"]["rpm"] == "0/60"
        client_str = str(mock_client)
        assert "models=" in client_str

        # Set and verify values
        model = self._get_model(mock_client, "gpt-4o-mini")
        model.spend_tokens(100)
        model.spend_request(5)
        usage = model.get_usage()
        assert usage["tpm"] == "100/10000"
        assert usage["rpm"] == "5/60"

        usage = mock_client.get_usage()
        assert usage["gpt-4o-mini"]["tpm"] == "100/10000"
        assert usage["gpt-4o-mini"]["rpm"] == "5/60"

        # Reset and verify again
        mock_client.reset_usage()
        usage = mock_client.get_usage()
        assert usage["gpt-4o-mini"]["tpm"] == "0/10000"
        assert usage["gpt-4o-mini"]["rpm"] == "0/60"
        assert model._last_reset > 0

    async def test_utilization(self, mock_client: Client):
        """Test utilization calculation."""

        model = self._get_model(mock_client, "gpt-4o-mini")

        # Check initial utilization (nonzero due to random splay)
        initial_util = mock_client.util("gpt-4o-mini")
        assert 0 <= initial_util < 0.02

        # Test token-based utilization
        model.spend_tokens(5000)  # 50% of TPM limit
        util_with_tokens = model.util()
        assert 0.5 <= util_with_tokens < 0.52

        # Test request-based utilization
        model.reset_usage()
        model.spend_request(30)  # 50% of RPM limit
        util_with_requests = model.util()
        assert 0.5 <= util_with_requests < 0.52

        # Test combined utilization (should take max of the two)
        model.reset_usage()
        model.spend_tokens(6000)  # 60% of TPM
        model.spend_request(30)  # 50% of RPM
        util_with_both = model.util()
        assert 0.6 <= util_with_both < 0.62

        # Test unhealthy client
        model.cooldown()
        assert model.util() == 1

    async def test_concurrency(self, mock_client: Client):
        """Test handling of multiple concurrent requests."""
        mock_client.reset_usage()
        # Create and run concurrent requests
        num_requests = 10
        tasks = [mock_client.create(**self.basic_args) for _ in range(num_requests)]
        responses = await asyncio.gather(*tasks)

        # Verify results
        model = self._get_model(mock_client, "gpt-4o-mini")
        assert len(responses) == num_requests
        assert all(r == MOCK_COMPLETION for r in responses)
        assert mock_client.client.chat.completions.create.call_count == num_requests
        usage = model.get_usage()
        assert usage["tpm"] == f"{11 * num_requests}/10000"
        assert usage["rpm"] == f"{num_requests}/60"

    @pytest.fixture
    def d1_mock(self):
        with respx.mock(base_url="https://test1.openai.azure.com") as respx_mock:
            respx_mock.post(
                "/openai/deployments/gpt-4o-mini/chat/completions",
                name="completion",
            )
            yield respx_mock

    @pytest.fixture
    def test_client(self):
        """Create a real Client instance using the default factory, but use
        respx to mock out the underlying httpx client so we can verify
        the retry logic.
        """
        return default_client_factory(TEST_DEPLOYMENT_1)

    async def test_timeout_retry(self, d1_mock, test_client):
        """Test timeout retry behavior."""
        # Test successful retry after timeouts
        expected_response = Response(status_code=200, json=MOCK_COMPLETION_RAW)
        d1_mock.routes["completion"].side_effect = [
            TimeoutException("Timeout 1"),
            TimeoutException("Timeout 2"),
            expected_response,
        ]
        response = await test_client.create(**self.basic_args)
        assert response == ChatCompletion.model_validate(MOCK_COMPLETION_RAW)
        assert d1_mock.routes["completion"].call_count == 3

        # Test failure after max retries
        d1_mock.routes["completion"].reset()
        d1_mock.routes["completion"].side_effect = [
            TimeoutException("Timeout 1"),
            TimeoutException("Timeout 2"),
            TimeoutException("Timeout 3"),
        ]

        with pytest.raises(openai.APITimeoutError):
            await test_client.create(**self.basic_args)
        assert d1_mock.routes["completion"].call_count == 3
        assert not test_client.is_healthy("gpt-4o-mini")


MOCK_COMPLETION_RAW = {
    "choices": [
        {
            "finish_reason": "stop",
            "index": 0,
            "logprobs": None,
            "message": {
                "content": "Hello! How can I assist you today?",
                "refusal": None,
                "role": "assistant",
            },
        }
    ],
    "created": 1741124380,
    "id": "chatcmpl-test",
    "model": "gpt-4o-mini",
    "object": "chat.completion",
    "service_tier": "default",
    "system_fingerprint": "fp_06737a9306",
    "usage": {
        "completion_tokens": 10,
        "completion_tokens_details": {
            "accepted_prediction_tokens": 0,
            "audio_tokens": 0,
            "reasoning_tokens": 0,
            "rejected_prediction_tokens": 0,
        },
        "prompt_tokens": 8,
        "prompt_tokens_details": {"audio_tokens": 0, "cached_tokens": 0},
        "total_tokens": 18,
    },
}
