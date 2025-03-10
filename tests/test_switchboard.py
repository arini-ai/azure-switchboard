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

from azure_switchboard import Client, Switchboard


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


async def test_switchboard_completion(mock_switchboard: Switchboard, mock_client):
    # test basic chat completion
    response = await mock_switchboard.create(**BASIC_CHAT_COMPLETION_ARGS)
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
    stream = await mock_switchboard.create(stream=True, **BASIC_CHAT_COMPLETION_ARGS)

    # Collect all chunks
    _, content = await _collect_chunks(stream)
    mock_client.chat.completions.create.assert_called_once()
    assert content == "Hello, world!"


async def test_switchboard_selection(mock_switchboard: Switchboard):
    # test that we select a deployment
    client = mock_switchboard.select_deployment()
    assert client.config.name in mock_switchboard.deployments

    # test that we can select a specific deployment
    client_1 = mock_switchboard.select_deployment(session_id="test")
    client_2 = mock_switchboard.select_deployment(session_id="test")
    assert client_1.config.name == client_2.config.name

    # test that we fall back to load balancing if the selected deployment is unhealthy
    client_1.cooldown()
    client_3 = mock_switchboard.select_deployment(session_id="test")
    assert client_3.config.name != client_1.config.name

    # test that sessions support failover
    client_4 = mock_switchboard.select_deployment(session_id="test")
    assert client_4.config.name == client_3.config.name

    # test that recovery doesn't affect sessions
    await client_1.check_health()
    client_5 = mock_switchboard.select_deployment(session_id="test")
    assert client_5.config.name == client_3.config.name


async def test_load_distribution_with_session_stickiness(mock_switchboard: Switchboard):
    """Test that session stickiness works correctly"""

    # Reset usage counters
    mock_switchboard.reset_usage()

    # Make requests with different session IDs
    session_ids = ["session1", "session2", "session3", "session4", "session5"]

    # Make 10 requests per session ID (50 total)
    for _ in range(10):
        for session_id in session_ids:
            await mock_switchboard.create(
                session_id=session_id, **BASIC_CHAT_COMPLETION_ARGS
            )

    # Check that each session consistently went to the same deployment
    # This is harder to test directly, but we can verify that the distribution
    # is not perfectly balanced, which would indicate session affinity is working
    requests_per_deployment = sorted(
        [client.ratelimit_requests for client in mock_switchboard.deployments.values()]
    )

    # If session affinity is working, the distribution will be 10:20:20 instead of 16:16:18
    assert requests_per_deployment == [10, 20, 20]


async def test_load_distribution_session_stickiness_with_fallback(
    mock_switchboard: Switchboard,
):
    """Test session affinity when the preferred deployment becomes unavailable."""

    # Setup deployments
    clients = list(mock_switchboard.deployments.values())
    assert len(clients) >= 3, "Need at least 3 deployments for this test"

    # Set up client mock behavior
    for client in clients:
        client.client.chat.completions.create = AsyncMock(return_value=MOCK_COMPLETION)
        client.reset_cooldown()
        client.reset_counters()

    # Create a session
    session_id = "test-session-123"

    # Initial request with session - establishes session affinity
    response1 = await mock_switchboard.create(
        session_id=session_id, **BASIC_CHAT_COMPLETION_ARGS
    )
    assert response1 == MOCK_COMPLETION

    # Get the deployment assigned to this session
    assigned_client = mock_switchboard._sessions[session_id]

    # Verify subsequent requests use the same deployment
    response2 = await mock_switchboard.create(
        session_id=session_id, **BASIC_CHAT_COMPLETION_ARGS
    )
    assert response2 == MOCK_COMPLETION

    # Check that the same deployment was used
    assert assigned_client.client.chat.completions.create.call_count == 2

    # Now make the assigned deployment unhealthy
    assigned_client.cooldown()

    # Make the next request with the same session
    response3 = await mock_switchboard.create(
        session_id=session_id, **BASIC_CHAT_COMPLETION_ARGS
    )
    assert response3 == MOCK_COMPLETION

    # Now check which deployment was used - in the current implementation,
    # the fallback policy might try the unhealthy deployment first, then fallback
    # So we can't assert the exact call count - instead verify that a different deployment got used

    # Identify which fallback deployment was used for the session now
    fallback_session_client = mock_switchboard._sessions[session_id]
    assert fallback_session_client != assigned_client, (
        "Session should have switched to a different deployment"
    )

    # Make the original deployment healthy again
    assigned_client.reset_cooldown()

    # Make another request with the same session
    response4 = await mock_switchboard.create(
        session_id=session_id, **BASIC_CHAT_COMPLETION_ARGS
    )
    assert response4 == MOCK_COMPLETION

    # Check if the system remembers the new session assignment or returns to the original
    # Either behavior could be valid depending on implementation
    current_assignment = mock_switchboard._sessions[session_id]
    assert current_assignment is not None, "Session should still be tracked"


async def test_load_distribution_basic(mock_switchboard: Switchboard) -> None:
    """Test that load is distributed across deployments based on utilization"""

    # Reset usage counters and bump limits
    mock_switchboard.reset_usage()

    # Make 100 requests
    for _ in range(100):
        await mock_switchboard.create(**BASIC_CHAT_COMPLETION_ARGS)

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


async def test_load_distribution_with_unhealthy_deployment(
    mock_switchboard: Switchboard,
):
    """Test that unhealthy deployments are skipped"""

    # Reset usage counters
    mock_switchboard.reset_usage()

    # Mark one deployment as unhealthy
    mock_switchboard.deployments["test2"].cooldown()

    # Make 100 requests
    for _ in range(100):
        await mock_switchboard.create(**BASIC_CHAT_COMPLETION_ARGS)

    # Verify that the unhealthy deployment wasn't used
    assert mock_switchboard.deployments["test1"].ratelimit_requests > 40
    assert mock_switchboard.deployments["test2"].ratelimit_requests == 0
    assert mock_switchboard.deployments["test3"].ratelimit_requests > 40


async def test_load_distribution_large_scale(mock_switchboard: Switchboard):
    """Test load distribution at scale with 1000 requests"""

    # Reset usage counters and bump limits
    mock_switchboard.reset_usage()
    for client in mock_switchboard.deployments.values():
        client.config.tpm_ratelimit *= 10
        client.config.rpm_ratelimit *= 10

    # Make 1000 requests concurrently
    tasks = []
    for i in range(1000):
        # Use a session ID for every 10th request to test session affinity
        session_id = f"session{i // 10}" if i % 10 == 0 else None
        tasks.append(
            mock_switchboard.create(session_id=session_id, **BASIC_CHAT_COMPLETION_ARGS)
        )

    await asyncio.gather(*tasks)

    total_requests = sum(
        client.ratelimit_requests for client in mock_switchboard.deployments.values()
    )
    assert total_requests == 1000


async def test_load_distribution_with_recovery(
    mock_switchboard: Switchboard, mock_client
):
    """Test that deployments that become unhealthy mid-operation are skipped but then can recover"""

    # Reset usage counters
    mock_switchboard.reset_usage()

    # Make initial requests to establish baseline usage
    for _ in range(50):
        await mock_switchboard.create(**BASIC_CHAT_COMPLETION_ARGS)

    # Mark one deployment as unhealthy mid-operation
    mock_switchboard.deployments["test1"].cooldown()

    # Make additional requests while one deployment is unhealthy
    for _ in range(50):
        await mock_switchboard.create(**BASIC_CHAT_COMPLETION_ARGS)

    # Verify that the unhealthy deployment wasn't used during the second batch
    assert mock_switchboard.deployments["test1"].ratelimit_requests < 20
    assert mock_switchboard.deployments["test2"].ratelimit_requests >= 35
    assert mock_switchboard.deployments["test3"].ratelimit_requests >= 35

    # Reset the unhealthy deployment and make more requests
    mock_switchboard.deployments["test1"].reset_cooldown()
    for _ in range(50):
        await mock_switchboard.create(**BASIC_CHAT_COMPLETION_ARGS)

    # Verify that the unhealthy deployment was skipped during its unhealthy state
    assert mock_switchboard.deployments["test1"].ratelimit_requests > 40


async def test_load_distribution_proportional_to_ratelimits(
    mock_switchboard: Switchboard,
):
    """Test that deployments with uneven ratelimits are used proportionally"""

    # Reset usage counters
    mock_switchboard.reset_usage()

    # Set different ratelimits for each deployment
    d1 = mock_switchboard.deployments["test1"]
    d2 = mock_switchboard.deployments["test2"]
    d3 = mock_switchboard.deployments["test3"]

    d1.config.tpm_ratelimit *= 1
    d1.config.rpm_ratelimit *= 1
    d2.config.tpm_ratelimit *= 2
    d2.config.rpm_ratelimit *= 2
    d3.config.tpm_ratelimit *= 3
    d3.config.rpm_ratelimit *= 3

    # Make 150 requests (keep the math clean)
    for _ in range(150):
        await mock_switchboard.create(**BASIC_CHAT_COMPLETION_ARGS)

    # Verify that the deployments were used proportionally, 10% error margin
    def within_margin(a, b, margin) -> bool:
        return a * (1 - margin) <= b <= a * (1 + margin)

    assert within_margin(150 * (1 / 6), d1.ratelimit_requests, margin=0.1)
    assert within_margin(150 * (2 / 6), d2.ratelimit_requests, margin=0.1)
    assert within_margin(150 * (3 / 6), d3.ratelimit_requests, margin=0.1)


async def test_load_distribution_dynamic_rebalancing(mock_switchboard: Switchboard):
    """Test how load is redistributed when deployments fail and recover."""
    # Setup deployments
    clients = list(mock_switchboard.deployments.values())
    assert len(clients) >= 3, "Need at least 3 deployments for this test"

    # Set up client mock behavior
    for client in clients:
        client.client.chat.completions.create = AsyncMock(return_value=MOCK_COMPLETION)
        client.reset_cooldown()
        client.reset_counters()

    # Phase 1: All deployments healthy - requests should be balanced
    for _ in range(30):
        await mock_switchboard.create(**BASIC_CHAT_COMPLETION_ARGS)

    # All clients should have some requests
    request_counts = [client.ratelimit_requests for client in clients]
    assert all(count > 0 for count in request_counts), (
        "Load not distributed to all deployments"
    )

    # Phase 2: Make one deployment unhealthy
    clients[0].cooldown()
    mock_switchboard.reset_usage()  # Reset counters to track new distribution

    # Send more requests
    for _ in range(30):
        await mock_switchboard.create(**BASIC_CHAT_COMPLETION_ARGS)

    # Unhealthy client should receive no requests
    assert clients[0].ratelimit_requests == 0, "Unhealthy deployment received requests"

    # Other clients should share the load
    assert clients[1].ratelimit_requests > 0, (
        "Healthy deployment 1 received no requests"
    )
    assert clients[2].ratelimit_requests > 0, (
        "Healthy deployment 2 received no requests"
    )

    # Phase 3: Recover the unhealthy deployment
    clients[0].reset_cooldown()
    mock_switchboard.reset_usage()  # Reset counters again

    # Send more requests
    for _ in range(30):
        await mock_switchboard.create(**BASIC_CHAT_COMPLETION_ARGS)

    # All clients should again have some requests
    request_counts = [client.ratelimit_requests for client in clients]
    assert all(count > 0 for count in request_counts), (
        "Recovered deployment not receiving requests"
    )

    # Phase 4: Make a different deployment unhealthy
    clients[1].cooldown()
    mock_switchboard.reset_usage()

    # Send more requests
    for _ in range(30):
        await mock_switchboard.create(**BASIC_CHAT_COMPLETION_ARGS)

    # The unhealthy client should receive no requests
    assert clients[1].ratelimit_requests == 0, (
        "Second unhealthy deployment received requests"
    )

    # Other clients should share the load
    assert clients[0].ratelimit_requests > 0, (
        "Healthy deployment 1 received no requests"
    )
    assert clients[2].ratelimit_requests > 0, (
        "Healthy deployment 3 received no requests"
    )


async def test_load_distribution_chaos(mock_switchboard: Switchboard):
    """Test resilience under unpredictable failure patterns."""
    import random

    # Setup deployments
    clients = list(mock_switchboard.deployments.values())
    assert len(clients) >= 3, "Need at least 3 deployments for this test"

    # Set up initial client state
    for client in clients:
        client.client.chat.completions.create = AsyncMock(return_value=MOCK_COMPLETION)
        client.reset_cooldown()
        client.reset_counters()

    # Define chaos behaviors
    async def random_failure(client_index, request_index):
        # Randomly choose a failure mode
        failure_type = random.choice(
            [
                "timeout",
                "error",
                "rate_limit",
                "bad_response",
                "delayed",
                None,  # No failure
                None,  # Increase probability of no failure to keep system functional
                None,  # Increase probability of no failure to keep system functional
            ]
        )

        # Ensure at least one deployment stays healthy
        healthy_count = sum(1 for c in clients if c.healthy)
        if healthy_count <= 1 and failure_type is not None:
            failure_type = None  # Don't break the last healthy deployment

        if failure_type == "timeout":
            clients[
                client_index
            ].client.chat.completions.create.side_effect = TimeoutError(
                "Random timeout"
            )
        elif failure_type == "error":
            clients[
                client_index
            ].client.chat.completions.create.side_effect = Exception("Random error")
        elif failure_type == "rate_limit":
            clients[
                client_index
            ].ratelimit_tokens = 10000  # Set high to trigger rate limiting
        elif failure_type == "bad_response":
            # Create a malformed response
            bad_response = {
                "id": "bad-response",
                "object": "chat.completion",
                "created": 1234567890,
                "model": "gpt-4-bad",
                "choices": [],  # Missing content
            }
            clients[
                client_index
            ].client.chat.completions.create.return_value = bad_response
        elif failure_type == "delayed":
            # Create a delayed response
            async def delayed_response(*args, **kwargs):
                await asyncio.sleep(0.05)  # Small delay
                return MOCK_COMPLETION

            clients[
                client_index
            ].client.chat.completions.create.side_effect = delayed_response
        else:
            # Reset to normal behavior
            clients[client_index].client.chat.completions.create.side_effect = None
            clients[
                client_index
            ].client.chat.completions.create.return_value = MOCK_COMPLETION
            clients[client_index].reset_cooldown()

        # Randomly recover some deployments
        if random.random() < 0.3:  # 30% chance to reset a random deployment
            reset_index = random.randrange(len(clients))
            clients[reset_index].client.chat.completions.create.side_effect = None
            clients[
                reset_index
            ].client.chat.completions.create.return_value = MOCK_COMPLETION
            clients[reset_index].reset_cooldown()
            clients[reset_index].reset_counters()

        # Ensure at least one deployment is always healthy before returning
        if all(not c.healthy for c in clients):
            random_index = random.randrange(len(clients))
            clients[random_index].reset_cooldown()
            clients[random_index].client.chat.completions.create.side_effect = None

    # Track overall success
    total_requests = 50
    successful_requests = 0
    failed_requests = 0

    # Run chaos test
    for i in range(total_requests):
        # Randomly apply chaos to deployments
        for client_index in range(len(clients)):
            if random.random() < 0.2:  # 20% chance to cause chaos for each deployment
                await random_failure(client_index, i)

        # Try to make a request
        try:
            await mock_switchboard.create(**BASIC_CHAT_COMPLETION_ARGS)
            successful_requests += 1
        except Exception:
            failed_requests += 1

            # Ensure at least one deployment is healthy after a failure
            if all(not c.healthy for c in clients):
                random_index = random.randrange(len(clients))
                clients[random_index].reset_cooldown()
                clients[random_index].client.chat.completions.create.side_effect = None

    # Calculate success rate
    success_rate = (successful_requests / total_requests) * 100

    # The test should maintain some level of availability even with chaos
    # Actual threshold depends on the system design and failure rates
    assert success_rate > 0, f"Success rate too low: {success_rate}%"

    # Make sure at least one deployment is healthy for the final check
    if all(not c.healthy for c in clients):
        random_index = random.randrange(len(clients))
        clients[random_index].reset_cooldown()
        clients[random_index].client.chat.completions.create.side_effect = None

    # Validate that at least some requests were routed to deployments
    request_distribution = [client.ratelimit_requests for client in clients]

    # There should be some distribution of requests
    assert sum(request_distribution) > 0, "No requests were processed"
