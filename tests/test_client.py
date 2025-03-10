import asyncio

import openai
import pytest
import respx
from fixtures import (
    BASIC_CHAT_COMPLETION_ARGS,
    MOCK_COMPLETION,
    MOCK_COMPLETION_PARSED,
    MOCK_COMPLETION_RAW,
    MOCK_STREAM_CHUNKS,
    TEST_DEPLOYMENT_1,
)
from httpx import Response, TimeoutException
from utils import BaseTestCase, create_mock_openai_client

from azure_switchboard import Client, ModelConfig
from azure_switchboard.client import default_client_factory


@pytest.fixture
def mock_client() -> Client:
    """Create a Client instance with a basic mock."""
    openai_mock = create_mock_openai_client()
    return Client(TEST_DEPLOYMENT_1, client=openai_mock)


class TestClient(BaseTestCase):
    """Basic client functionality tests."""

    # async def test_healthcheck(self, mock_client):
    #     """Test basic healthcheck functionality."""
    #     # Test basic healthcheck
    #     await mock_client.check_health()
    #     assert mock_client.client.models.list.call_count == 1
    #     assert mock_client.healthy

    #     # Test healthcheck failure
    #     mock_client.client.models.list.side_effect = Exception("test")
    #     await mock_client.check_health()
    #     assert not mock_client.healthy
    #     assert mock_client.client.models.list.call_count == 2

    #     # Test cooldown reset allows recovery
    #     mock_client.reset_cooldown()
    #     assert mock_client.healthy

    def _get_model(self, client: Client, model: str) -> ModelConfig:
        return client.config.models[model]

    async def test_completion(self, mock_client):
        """Test basic chat completion functionality."""

        mock_client.reset_usage()
        model = self._get_model(mock_client, "gpt-4o-mini")
        response = await mock_client.create(**self.basic_args)
        assert mock_client.client.chat.completions.create.call_count == 1
        assert response == MOCK_COMPLETION

        # Check token usage tracking
        assert model.tpm_usage == 11
        assert model.rpm_usage == 1

        # Test exception handling
        mock_client.client.chat.completions.create.side_effect = Exception("test")
        with pytest.raises(Exception, match="test"):
            await mock_client.create(**self.basic_args)
        assert mock_client.client.chat.completions.create.call_count == 2

        # account for preflight estimate
        assert model.tpm_usage == 14
        assert model.rpm_usage == 2

    async def test_streaming(self, mock_client):
        """Test streaming functionality."""

        mock_client.reset_usage()
        model = self._get_model(mock_client, "gpt-4o-mini")
        stream = await mock_client.create(stream=True, **self.basic_args)
        assert stream is not None

        # Collect chunks and verify content
        received_chunks, content = await self.collect_chunks(stream)

        # Verify stream options
        assert (
            mock_client.client.chat.completions.create.call_args.kwargs.get("stream")
            is True
        )
        assert (
            mock_client.client.chat.completions.create.call_args.kwargs.get(
                "stream_options", {}
            ).get("include_usage")
            is True
        )

        # Verify chunk handling
        assert len(received_chunks) == len(MOCK_STREAM_CHUNKS)
        assert content == "Hello, world!"

        # Verify token usage tracking
        assert model.tpm_usage == 20
        assert model.rpm_usage == 1

        # Test exception handling
        mock_client.client.chat.completions.create.side_effect = Exception("test")
        with pytest.raises(Exception, match="test"):
            stream = await mock_client.create(stream=True, **BASIC_CHAT_COMPLETION_ARGS)
            async for _ in stream:
                pass
        assert mock_client.client.chat.completions.create.call_count == 2
        assert model.rpm_usage == 2

    async def test_usage(self, mock_client):
        """Test counter management."""
        # Reset and verify initial state
        mock_client.reset_usage()
        for model in mock_client.config.models.values():
            assert model.tpm_usage == 0
            assert model.rpm_usage == 0

        # Set and verify values
        model = self._get_model(mock_client, "gpt-4o-mini")
        model.tpm_usage = 100
        model.rpm_usage = 5
        counters = model.get_usage()
        assert counters["tpm"] == "100/10000"
        assert counters["rpm"] == "5/60"

        # Reset and verify again
        mock_client.reset_usage()
        assert model.tpm_usage == 0
        assert model.rpm_usage == 0
        assert model._last_reset > 0

    async def test_utilization(self, mock_client):
        """Test utilization calculation."""
        mock_client.reset_usage()
        mock_client.config.models["gpt-4o-mini"].reset_cooldown()

        # Check initial utilization (nonzero due to random splay)
        initial_util = mock_client.util("gpt-4o-mini")
        assert 0 <= initial_util < 0.02

        # Test token-based utilization
        model = self._get_model(mock_client, "gpt-4o-mini")
        model.tpm_usage = 5000  # 50% of TPM limit
        util_with_tokens = model.util()
        assert 0.5 <= util_with_tokens < 0.52

        # Test request-based utilization
        model.rpm_usage = 30  # 50% of RPM limit
        util_with_requests = model.util()
        assert 0.5 <= util_with_requests < 0.52

        # Test combined utilization (should take max of the two)
        model.tpm_usage = 6000  # 60% of TPM
        model.rpm_usage = 30  # 50% of RPM
        util_with_both = model.util()
        assert 0.6 <= util_with_both < 0.62

        # Test unhealthy client
        model.cooldown()
        assert model.util() == 1

    async def test_concurrency(self, mock_client):
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
        assert model.tpm_usage == 11 * num_requests
        assert model.rpm_usage == num_requests

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
        response = await test_client.create(**BASIC_CHAT_COMPLETION_ARGS)
        assert response == MOCK_COMPLETION_PARSED
        assert d1_mock.routes["completion"].call_count == 3

        # Test failure after max retries
        d1_mock.routes["completion"].reset()
        d1_mock.routes["completion"].side_effect = [
            TimeoutException("Timeout 1"),
            TimeoutException("Timeout 2"),
            TimeoutException("Timeout 3"),
        ]

        with pytest.raises(openai.APITimeoutError):
            await test_client.create(**BASIC_CHAT_COMPLETION_ARGS)
        assert d1_mock.routes["completion"].call_count == 3
        assert not test_client.is_healthy("gpt-4o-mini")
