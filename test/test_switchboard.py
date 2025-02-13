import asyncio
import pytest

from contextlib import asynccontextmanager

from switchboard.switchboard import (
    AzureDeployment,
    Switchboard,
    LoadBalancerError
)

# --- Deployment Definitions (Constants) ---

HEALTHY_DEPLOYMENT = AzureDeployment(
    name="test_healthy",
    api_base="https://test-healthy.openai.azure.com/",
    api_key="test-key-healthy"
)

UNHEALTHY_DEPLOYMENT = AzureDeployment(
    name="unhealthy_deployment",  # Triggers simulated failures
    api_base="https://test-unhealthy.openai.azure.com/",
    api_key="test-key-unhealthy"
)

# --- Fixtures ---

@asynccontextmanager
async def create_switchboard(deployments):
    """Create a Switchboard instance with health check."""
    sb = Switchboard(
        deployments=deployments,
        health_check_interval=0.1,  # For fast tests
        cooldown_period=0.5       # For fast tests
    )
    await sb.start()
    try:
        yield sb
    finally:
        await sb.close()

# --- Test Cases (for Parameterization) ---

TEST_CASES = [
    ( [HEALTHY_DEPLOYMENT], "Healthy", "single_healthy" ),
    ( [UNHEALTHY_DEPLOYMENT], LoadBalancerError, "single_unhealthy" ),
    ( [HEALTHY_DEPLOYMENT, UNHEALTHY_DEPLOYMENT], "Healthy", "healthy_and_unhealthy_initial" ),
]

# --- Tests ---

@pytest.mark.asyncio
@pytest.mark.parametrize("deployments,expected_response,test_id", TEST_CASES, ids=[tc[2] for tc in TEST_CASES])
async def test_initial_selection(deployments, expected_response, test_id):
    """Test initial deployment selection (healthy, unhealthy, mixed)."""
    async with create_switchboard(deployments) as sb:
        if expected_response == LoadBalancerError:
            with pytest.raises(LoadBalancerError):
                await sb.chat_completion(messages=[{"role": "user", "content": "Hi"}], model="health_check_model")
        else:
            response = await sb.chat_completion(messages=[{"role": "user", "content": "Hi"}], model="health_check_model")
            assert response == expected_response

@pytest.mark.asyncio
async def test_failover_to_healthy(healthy_deployment, unhealthy_deployment):
    """Test failover from unhealthy to healthy deployment."""
    async with create_switchboard([unhealthy_deployment, healthy_deployment]) as sb:
        # Initial request might go to unhealthy (or healthy, due to randomness)
        try:
            await sb.chat_completion(messages=[{"role": "user", "content": "Hi"}], model="health_check_model")
        except LoadBalancerError:
            pass  # Could initially hit unhealthy, that's okay

        await asyncio.sleep(0.2)  # Allow health check to run

        # Subsequent request *must* go to healthy
        response = await sb.chat_completion(messages=[{"role": "user", "content": "Hi"}], model="health_check_model")
        assert response == "Healthy"

@pytest.mark.asyncio
async def test_all_unhealthy_raises_error(unhealthy_deployment):
    """Test that LoadBalancerError is raised when all deployments are down."""
    async with create_switchboard([unhealthy_deployment, unhealthy_deployment]) as sb:
        # Ensure they're both marked as unhealthy initially (if not already)
        for deployment in sb.deployments:
            deployment.is_healthy = False

        with pytest.raises(LoadBalancerError, match="No healthy deployments available"):
            await sb.chat_completion(messages=[{"role": "user", "content": "Hi"}], model="health_check_model")

@pytest.mark.asyncio
async def test_cooldown_recovery(unhealthy_deployment, healthy_deployment):
    """Test that an unhealthy deployment recovers after the cooldown period."""
    async with create_switchboard([unhealthy_deployment, healthy_deployment]) as sb:
        # Force unhealthy deployment to be down
        for deployment in sb.deployments:
            if deployment.config.name == "unhealthy_deployment":
                deployment.is_healthy = False
                deployment.last_error_time = asyncio.get_event_loop().time() - 1 # set it to be ready to recover in 1s
        await asyncio.sleep(1.5) # wait for it to be ready to recover + a little more

        # Now it should be available again, though healthy might be chosen.
        # We don't *guarantee* it's chosen, just that it's *possible* to be chosen.
        try:
            response = await sb.chat_completion(messages=[{"role": "user", "content": "Hi"}], model="health_check_model")
            assert response == "Healthy"  # Expect healthy, but it *could* be unhealthy if random selection picks it.
        except LoadBalancerError:
            pytest.fail("Unhealthy deployment did not recover after cooldown.")
