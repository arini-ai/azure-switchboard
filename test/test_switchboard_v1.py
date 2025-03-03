import asyncio

import pytest

from switchboard.switchboard_v1 import (
    AzureDeployment,
    LoadBalancerError,
    Switchboard,
    get_switchboard,
)


def make_test_deployments(n: int = 3):
    return [
        AzureDeployment(
            name=f"test-{i}",
            # these are typed as pydantic AnyHttpUrl/SecretStr,
            # which validate  but pylance isn't able to
            # infer the type compatibility
            api_base=f"https://test-{i}.openai.azure.com/",
            api_key=f"key-{i}",
        )
        for i in range(1, n + 1)
    ]


# @pytest.mark.skipped
# @pytest.fixture
# async def default_switchboard():
#     deployments = [AzureDeployment.model_validate(d) for d in TEST_DEPLOYMENTS]
#     # speed things up for the tests
#     test_params = {"health_check_interval": 0.1, "cooldown_period": 0.5}
#     async with get_switchboard(deployments, **test_params) as sb:
#         yield sb


@pytest.mark.skipped
async def test_simple_switchboard(switchboard: Switchboard):
    response = await switchboard.chat_completion(
        messages=[{"role": "user", "content": "Hello, world!"}],
    )
