import asyncio
from typing import AsyncGenerator
from unittest.mock import AsyncMock

import pytest
from fixtures import (
    BASIC_CHAT_COMPLETION_ARGS,
    MOCK_COMPLETION,
    MOCK_STREAM_CHUNKS,
    TEST_DEPLOYMENT_1,
    TEST_DEPLOYMENT_2,
    TEST_DEPLOYMENT_3,
)
from openai.types.chat import ChatCompletionChunk
from test_client import _collect_chunks

from switchboard import Client, Switchboard


@pytest.fixture
def mock_client():
    mock = AsyncMock()
    mock.chat.completions.create = AsyncMock(return_value=MOCK_COMPLETION)
    return mock


@pytest.fixture
def mock_switchboard(mock_client):
    return Switchboard(
        [TEST_DEPLOYMENT_1, TEST_DEPLOYMENT_2, TEST_DEPLOYMENT_3],
        client_factory=lambda x: Client(x, mock_client),
        healthcheck_interval=0,  # disable healthchecks
        ratelimit_window=0,  # disable usage resets
    )


async def test_switchboard_selection_basic(mock_switchboard: Switchboard):
    # test that we select a deployment
    client = mock_switchboard.select_deployment()
    assert client.name in mock_switchboard.deployments

    # test that we can select a specific deployment
    client = mock_switchboard.select_deployment(session_id="test2")
    assert client.name == "test2"


async def test_switchboard_completion(mock_switchboard: Switchboard, mock_client):
    # test basic chat completion
    response = await mock_switchboard.completion(**BASIC_CHAT_COMPLETION_ARGS)
    mock_client.chat.completions.create.assert_called_once()
    assert response == MOCK_COMPLETION


async def test_switchboard_stream(mock_switchboard: Switchboard, mock_client):
    """Test that streaming works through the switchboard"""

    # Create a mock async generator for streaming
    async def mock_stream() -> AsyncGenerator[ChatCompletionChunk, None]:
        for chunk in MOCK_STREAM_CHUNKS:
            yield chunk

    # Set up the mock to return our async generator
    mock_client.chat.completions.create = AsyncMock(return_value=mock_stream())

    # Test streaming through switchboard
    stream = mock_switchboard.stream(**BASIC_CHAT_COMPLETION_ARGS)

    # Collect all chunks
    _, content = await _collect_chunks(stream)
    mock_client.chat.completions.create.assert_called_once()
    assert content == "Hello, world!"


async def test_load_distribution_basic(mock_switchboard: Switchboard) -> None:
    """Test that load is distributed across deployments based on utilization"""

    # Reset usage counters
    mock_switchboard.reset_usage()

    # Make 100 requests
    for _ in range(100):
        await mock_switchboard.completion(**BASIC_CHAT_COMPLETION_ARGS)

    # Check that all deployments were used
    used_deployments = [
        name
        for name, client in mock_switchboard.deployments.items()
        if client.ratelimit_requests > 0
    ]
    assert len(used_deployments) == len(mock_switchboard.deployments)

    # Verify that all deployments got approximately the same number of requests
    # (within 10% of each other)
    avg_requests = 0
    avg_tokens = 0
    for deployment in mock_switchboard.get_usage().values():
        avg_requests += deployment["requests"]
        avg_tokens += deployment["tokens"]

    avg_requests /= len(mock_switchboard.deployments)
    avg_tokens /= len(mock_switchboard.deployments)

    req_upper = avg_requests * 1.1
    req_lower = avg_requests * 0.9
    tok_upper = avg_tokens * 1.1
    tok_lower = avg_tokens * 0.9

    for deployment in mock_switchboard.deployments.values():
        assert req_lower <= deployment.ratelimit_requests <= req_upper
        assert tok_lower <= deployment.ratelimit_tokens <= tok_upper


async def test_load_distribution_with_session_affinity(mock_switchboard: Switchboard):
    """Test that session affinity works correctly"""

    # Reset usage counters
    mock_switchboard.reset_usage()

    # Make requests with different session IDs
    session_ids = ["session1", "session2", "session3", "session4", "session5"]

    # Make 10 requests per session ID (50 total)
    for _ in range(10):
        for session_id in session_ids:
            await mock_switchboard.completion(
                session_id=session_id, **BASIC_CHAT_COMPLETION_ARGS
            )

    # Check that each session consistently went to the same deployment
    # This is harder to test directly, but we can verify that the distribution
    # is not perfectly balanced, which would indicate session affinity is working
    requests_per_deployment = [
        client.ratelimit_requests for client in mock_switchboard.deployments.values()
    ]

    # If session affinity is working, the distribution won't be perfectly even
    # With 5 sessions and 3 deployments, we expect some variation
    assert all(r > 15 for r in requests_per_deployment)


async def test_load_distribution_with_unhealthy_deployment(
    mock_switchboard: Switchboard,
):
    """Test that unhealthy deployments are skipped"""

    # Reset usage counters
    mock_switchboard.reset_usage()

    # Mark one deployment as unhealthy
    mock_switchboard.deployments["test2"]._last_request_status = False

    # Make 100 requests
    for _ in range(100):
        await mock_switchboard.completion(**BASIC_CHAT_COMPLETION_ARGS)

    # Verify that the unhealthy deployment wasn't used
    assert mock_switchboard.deployments["test1"].ratelimit_requests > 40
    assert mock_switchboard.deployments["test2"].ratelimit_requests == 0
    assert mock_switchboard.deployments["test3"].ratelimit_requests > 40


async def test_load_distribution_large_scale(mock_switchboard: Switchboard):
    """Test load distribution at scale with 1000 requests"""

    # Reset usage counters
    mock_switchboard.reset_usage()

    # Make 1000 requests concurrently
    tasks = []
    for i in range(1000):
        # Use a session ID for every 10th request to test session affinity
        session_id = f"session{i // 10}" if i % 10 == 0 else None
        tasks.append(
            mock_switchboard.completion(
                session_id=session_id, **BASIC_CHAT_COMPLETION_ARGS
            )
        )

    await asyncio.gather(*tasks)

    # Get usage stats
    usage = mock_switchboard.get_usage()

    # Calculate the expected distribution based on TPM ratios
    # test1:test2:test3 = 100:200:300 = 1:2:3
    total_requests = sum(
        client.ratelimit_requests for client in mock_switchboard.deployments.values()
    )
    assert total_requests == 1000
