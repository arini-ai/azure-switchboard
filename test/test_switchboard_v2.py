import asyncio
import random
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from openai import AsyncAzureOpenAI

# Import the code that will be tested
from switchboard import Deployment, Switchboard, default_client_factory

DEPLOYMENT_1 = Deployment(
    name="deployment1",
    api_base="https://hireai-openai-east.openai.azure.com/",
    api_key="72532109eae542c399b88dd5c017c6ef",
    max_tpm=100,
    max_rpm=5,
)

DEPLOYMENT_2 = Deployment(
    name="deployment2",
    api_base="https://hireai-openai-east.openai.azure.com/",
    api_key="72532109eae542c399b88dd5c017c6ef",
    max_tpm=100,
    max_rpm=5,
)

DEPLOYMENT_3 = Deployment(
    name="deployment3",
    api_base="https://hireai-openai-east.openai.azure.com/",
    api_key="72532109eae542c399b88dd5c017c6ef",
    max_tpm=100,
    max_rpm=5,
)

TEST_DEPLOYMENTS = [DEPLOYMENT_1, DEPLOYMENT_2, DEPLOYMENT_3]


def test_make_switchboard():
    sb = Switchboard(
        deployments=TEST_DEPLOYMENTS,
        client_factory=default_client_factory,
    )
    assert sb is not None


@pytest.fixture
def test_live_switchboard():
    return Switchboard(
        deployments=TEST_DEPLOYMENTS,
        client_factory=default_client_factory,
    )


@pytest.mark.asyncio
async def test_basic_functionality(test_live_switchboard):
    sb = test_live_switchboard
    response = await sb.chat.completions.create(
        model="gpt-4o-mini", messages=[{"role": "user", "content": "Hello"}]
    )
    assert response is not None
    print(response.choices[0].message.content)


@pytest.mark.asyncio
async def test_multiple_calls(test_live_switchboard):
    sb = test_live_switchboard
    response = await sb.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": "Who are the founders of OpenAI?"}],
    )
    assert response is not None
    print(response.choices[0].message.content)

    response = await sb.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": "What's the capital of France?"}],
    )
    assert response is not None
    print(response.choices[0].message.content)

    response = await sb.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": "What's the capital of Burundi?"}],
    )
    assert response is not None
    print(response.choices[0].message.content)


# @pytest.fixture
# def switchboard(deployment_configs):
#     """Create a Switchboard instance with mock clients for testing"""
#     # Create Switchboard with mocked clients and disabled health checks
#     sb = Switchboard(
#         deployments=deployment_configs,
#         client_factory=mock_client_factory,
#         auto_start_health_checks=False,
#     )
#     return sb

# def test_switchboard_creation(deployment_configs):
#     """Test that Switchboard can be created with valid configurations"""
#     sb = Switchboard(
#         deployments=deployment_configs,
#         client_factory=mock_client_factory,
#         auto_start_health_checks=False,
#     )
#     assert sb is not None


# # Test the DeploymentState class
# def test_deployment_state_utilization():
#     """Test that utilization calculations work correctly"""
#     config = AzureDeployment(
#         endpoint="https://test.openai.azure.com/",
#         api_key="key",
#         deployment_name="test",
#         tpm_limit=1000,
#         rpm_limit=100,
#     )

#     client = AsyncMock()
#     state = Deployment(config=config, client=client)

#     # Test initial state
#     assert state.token_utilization == 0.0
#     assert state.request_utilization == 0.0

#     # Test after adding some usage
#     state.tpm_usage = 500
#     state.rpm_usage = 50
#     assert state.token_utilization == 0.5
#     assert state.request_utilization == 0.5

#     # Test counter reset
#     original_time = state.last_reset
#     state.last_reset = time.time() - 61  # More than a minute ago
#     assert state.token_utilization == 0.0
#     assert state.request_utilization == 0.0
#     assert state.tpm_usage == 0
#     assert state.last_reset > original_time


# # Test the Switchboard initialization
# def test_switchboard_init(deployment_configs):
#     """Test that Switchboard initializes correctly"""
#     sb = Switchboard(
#         deployments=deployment_configs,
#         client_factory=mock_client_factory,
#         auto_start_health_checks=False,
#     )

#     assert len(sb.deployments) == 2
#     assert all(isinstance(d.client, AsyncMock) for d in sb.deployments)
#     assert all(d.healthy for d in sb.deployments)


# # Test the deployment selection logic
# def test_deployment_selection(switchboard):
#     """Test the power of two random choices algorithm"""
#     with patch("random.sample") as mock_sample:
#         # Force random.sample to return the first two deployments
#         mock_sample.return_value = [
#             switchboard.deployments[0],
#             switchboard.deployments[1],
#         ]

#         # Test when both deployments have equal load
#         switchboard.deployments[0].tokens_used_1min = 50000  # 50% of tpm_limit
#         switchboard.deployments[0].requests_used_1min = 500  # 50% of rpm_limit
#         switchboard.deployments[1].tokens_used_1min = 25000  # 50% of tpm_limit
#         switchboard.deployments[1].requests_used_1min = 250  # 50% of rpm_limit

#         selected = switchboard._select_deployment()
#         # Since both have 50% utilization, it should pick the first one (tie-breaker)
#         assert selected == switchboard.deployments[0]

#         # Test when second deployment has lower load
#         switchboard.deployments[0].tokens_used_1min = 70000  # 70% of tpm_limit
#         switchboard.deployments[1].tokens_used_1min = 20000  # 40% of tpm_limit

#         selected = switchboard._select_deployment()
#         assert selected == switchboard.deployments[1]

#         # Test when first deployment has lower load
#         switchboard.deployments[0].tokens_used_1min = 30000  # 30% of tpm_limit
#         switchboard.deployments[1].tokens_used_1min = 30000  # 60% of tpm_limit

#         selected = switchboard._select_deployment()
#         assert selected == switchboard.deployments[0]


# # Test the health check logic
# # @pytest.mark.asyncio
# # @pytest.mark.skipped
# async def _test_health_check(switchboard):
#     """Test that health checks update deployment health status correctly"""
#     # Start with all deployments healthy
#     assert all(d.healthy for d in switchboard.deployments)

#     # Make the first deployment unhealthy
#     await switchboard.deployments[0].client.set_unhealthy()

#     # Run a health check manually
#     await switchboard._health_check_loop()

#     # First deployment should now be marked as unhealthy
#     assert not switchboard.deployments[0].healthy
#     assert switchboard.deployments[1].healthy

#     # Restore health and check again
#     await switchboard.deployments[0].client.set_healthy()
#     await switchboard._health_check_loop()

#     # All deployments should be healthy again
#     assert all(d.healthy for d in switchboard.deployments)


# # Test that dynamic dispatch works correctly
# @pytest.mark.asyncio
# async def test_dynamic_dispatch(switchboard):
#     """Test that method calls are properly forwarded to the selected deployment"""
#     # Call the chat.completions.create method
#     response = await switchboard.chat.completions.create(
#         model="gpt-4", messages=[{"role": "user", "content": "Hello"}]
#     )

#     # Verify the response
#     assert response.choices[0].message.content == "This is a mock response"

#     # Check that token usage was tracked
#     selected_deployment = switchboard._select_deployment()
#     assert selected_deployment.tokens_used_1min == 30
#     assert selected_deployment.requests_used_1min == 1


# # Test handling of unhealthy deployments
# # @pytest.mark.asyncio
# # async def test_unhealthy_deployment_handling(switchboard):
# #     """Test that unhealthy deployments are skipped during selection"""
# #     # Make all deployments unhealthy
# #     for deployment in switchboard.deployments:
# #         await deployment.client.set_unhealthy()

# #     # Run a health check
# #     await switchboard._health_check_loop()

# #     # All deployments should be unhealthy
# #     assert all(not d.healthy for d in switchboard.deployments)

# #     # Attempting to select a deployment should raise an error
# #     with pytest.raises(RuntimeError, match="No healthy deployments available"):
# #         switchboard._select_deployment()

# #     # Make one deployment healthy again
# #     await switchboard.deployments[0].client.set_healthy()
# #     await switchboard._health_check_loop()

# #     # Now selection should work and return the only healthy deployment
# #     selected = switchboard._select_deployment()
# #     assert selected == switchboard.deployments[0]


# # Test error handling during method calls
# @pytest.mark.asyncio
# async def test_error_handling(switchboard):
#     """Test that errors during method calls are properly handled"""
#     # Configure the method to raise an exception
#     error_message = "API rate limit exceeded"
#     switchboard.deployments[0].client.chat.completions.create.side_effect = Exception(
#         error_message
#     )

#     # Mark only the first deployment as healthy to ensure it's selected
#     switchboard.deployments[0].healthy = True
#     switchboard.deployments[1].healthy = False

#     # Call should raise the exception
#     with pytest.raises(Exception, match=error_message):
#         await switchboard.chat.completions.create(
#             model="gpt-4", messages=[{"role": "user", "content": "Hello"}]
#         )

#     # The deployment should be marked as unhealthy
#     assert not switchboard.deployments[0].healthy

#     # With both deployments unhealthy, the next call should fail with a different error
#     with pytest.raises(RuntimeError, match="No healthy deployments available"):
#         await switchboard.chat.completions.create(
#             model="gpt-4", messages=[{"role": "user", "content": "Hello"}]
#         )


# # Test load balancing under high load
# @pytest.mark.asyncio
# async def test_load_balancing(switchboard):
#     """Test that load is distributed according to utilization"""
#     # Reset deployments
#     for deployment in switchboard.deployments:
#         deployment.healthy = True
#         deployment.tokens_used_1min = 0
#         deployment.requests_used_1min = 0

#     # Set up to track which deployment is used for each call
#     calls_per_deployment = {0: 0, 1: 0}

#     # Function to make a call and record which deployment was used
#     async def make_call():
#         # Force random.sample to pick deployments in order for predictability
#         # but still call the actual selection logic
#         with patch("random.sample", side_effect=lambda population, k: population[:k]):
#             await switchboard.chat.completions.create(
#                 model="gpt-4", messages=[{"role": "user", "content": "Hello"}]
#             )

#             # The deployment with the lowest utilization should have been used
#             if (
#                 switchboard.deployments[0].token_utilization
#                 <= switchboard.deployments[1].token_utilization
#             ):
#                 calls_per_deployment[0] += 1
#             else:
#                 calls_per_deployment[1] += 1

#     # Make 20 calls
#     for _ in range(20):
#         await make_call()

#     # Since deployments have different capacities, load should not be evenly distributed
#     # The deployment with higher capacity should receive more calls
#     assert calls_per_deployment[0] > calls_per_deployment[1]


# # Integration test simulating a realistic workload
# # @pytest.mark.asyncio
# # async def test_integration_simulation(switchboard):
# #     """Simulate a realistic workload with occasional errors"""
# #     # Reset deployments
# #     for deployment in switchboard.deployments:
# #         deployment.healthy = True
# #         deployment.tokens_used_1min = 0
# #         deployment.requests_used_1min = 0

# #     # Configure the first client to occasionally fail
# #     original_side_effect = switchboard.deployments[
# #         0
# #     ].client.chat.completions.create.side_effect

# #     call_count = 0

# #     async def sometimes_fail(*args, **kwargs):
# #         nonlocal call_count
# #         call_count += 1
# #         if call_count % 5 == 0:  # Every 5th call fails
# #             raise Exception("Occasional failure")
# #         return await original_side_effect(*args, **kwargs)

# #     switchboard.deployments[
# #         0
# #     ].client.chat.completions.create.side_effect = sometimes_fail

# #     # Simulate a workload of 30 calls
# #     successes = 0
# #     failures = 0

# #     for i in range(30):
# #         try:
# #             await switchboard.chat.completions.create(
# #                 model="gpt-4", messages=[{"role": "user", "content": f"Query {i}"}]
# #             )
# #             successes += 1
# #         except Exception:
# #             failures += 1
# #             # Run a health check to mark the failed deployment as unhealthy
# #             await switchboard._health_check_loop()

# #         # Occasional health check to recover unhealthy deployments
# #         if i % 10 == 0:
# #             # Reset the failure count for the test
# #             call_count = 0
# #             # Make all deployments healthy
# #             for deployment in switchboard.deployments:
# #                 deployment.healthy = True

# #     # We should have significantly more successes than failures
# #     assert successes > failures
# #     # If we expect every 5th call to fail and we have 30 calls, we expect about 6 failures
# #     assert (
# #         failures > 0 and failures <= 10
# #     )  # Some margin for the power of two choices algorithm
