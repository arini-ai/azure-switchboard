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

from azure_switchboard import Client
from azure_switchboard.client import default_client_factory


@pytest.fixture
def mock_client() -> Client:
    """Create a Client instance with a basic mock."""
    openai_mock = create_mock_openai_client()
    return Client(TEST_DEPLOYMENT_1, client=openai_mock)


class TestClient(BaseTestCase):
    """Basic client functionality tests."""

    async def test_healthcheck(self, mock_client):
        """Test basic healthcheck functionality."""
        # Test basic healthcheck
        await mock_client.check_health()
        assert mock_client.client.models.list.call_count == 1
        assert mock_client.healthy

        # Test healthcheck failure
        mock_client.client.models.list.side_effect = Exception("test")
        await mock_client.check_health()
        assert not mock_client.healthy
        assert mock_client.client.models.list.call_count == 2

        # Test cooldown reset allows recovery
        mock_client.reset_cooldown()
        assert mock_client.healthy

    async def test_completion(self, mock_client):
        """Test basic chat completion functionality."""

        # Test basic completion
        response = await mock_client.create(**BASIC_CHAT_COMPLETION_ARGS)
        assert mock_client.client.chat.completions.create.call_count == 1
        assert response == MOCK_COMPLETION

        # Check token usage tracking
        assert mock_client.ratelimit_tokens == 11
        assert mock_client.ratelimit_requests == 1

        # Test exception handling
        mock_client.client.chat.completions.create.side_effect = Exception("test")
        with pytest.raises(Exception, match="test"):
            await mock_client.create(**BASIC_CHAT_COMPLETION_ARGS)
        assert mock_client.client.chat.completions.create.call_count == 2

        assert mock_client.ratelimit_tokens == 14  # account for preflight estimate
        assert mock_client.ratelimit_requests == 2

    async def test_streaming(self, mock_client):
        """Test streaming functionality."""

        stream = await mock_client.create(stream=True, **BASIC_CHAT_COMPLETION_ARGS)
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
        assert mock_client.ratelimit_tokens == 20
        assert mock_client.ratelimit_requests == 1

        # Test exception handling
        mock_client.client.chat.completions.create.side_effect = Exception("test")
        with pytest.raises(Exception, match="test"):
            stream = await mock_client.create(stream=True, **BASIC_CHAT_COMPLETION_ARGS)
            async for _ in stream:
                pass
        assert mock_client.client.chat.completions.create.call_count == 2
        assert mock_client.ratelimit_requests == 2

    async def test_counters(self, mock_client):
        """Test counter management."""
        # Reset and verify initial state
        mock_client.reset_counters()
        assert mock_client.ratelimit_tokens == 0
        assert mock_client.ratelimit_requests == 0

        # Set and verify values
        mock_client.ratelimit_tokens = 100
        mock_client.ratelimit_requests = 5
        counters = mock_client.get_counters()
        assert counters["tokens"] == 100
        assert counters["requests"] == 5

        # Reset and verify again
        mock_client.reset_counters()
        assert mock_client.ratelimit_tokens == 0
        assert mock_client.ratelimit_requests == 0
        assert mock_client.last_reset > 0

    async def test_utilization(self, mock_client):
        """Test utilization calculation."""
        mock_client.reset_counters()

        # Check initial utilization (nonzero due to random splay)
        initial_util = mock_client.util
        assert 0 <= initial_util < 0.02

        # Test token-based utilization
        mock_client.ratelimit_tokens = 5000  # 50% of TPM limit
        util_with_tokens = mock_client.util
        assert 0.5 <= util_with_tokens < 0.52

        # Test request-based utilization
        mock_client.ratelimit_tokens = 0
        mock_client.ratelimit_requests = 30  # 50% of RPM limit
        util_with_requests = mock_client.util
        assert 0.5 <= util_with_requests < 0.52

        # Test combined utilization (should take max)
        mock_client.ratelimit_tokens = 6000  # 60% of TPM
        mock_client.ratelimit_requests = 30  # 50% of RPM
        util_with_both = mock_client.util
        assert 0.6 <= util_with_both < 0.62

        # Test unhealthy client
        mock_client.cooldown()
        assert mock_client.util == 1

    async def test_concurrency(self, mock_client):
        """Test handling of multiple concurrent requests."""

        # Create and run concurrent requests
        num_requests = 10
        tasks = [
            mock_client.create(**BASIC_CHAT_COMPLETION_ARGS)
            for _ in range(num_requests)
        ]
        responses = await asyncio.gather(*tasks)

        # Verify results
        assert len(responses) == num_requests
        assert all(r == MOCK_COMPLETION for r in responses)
        assert mock_client.client.chat.completions.create.call_count == num_requests
        assert mock_client.ratelimit_tokens == 11 * num_requests
        assert mock_client.ratelimit_requests == num_requests

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
        assert not test_client.healthy
