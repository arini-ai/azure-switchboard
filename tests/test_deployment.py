import asyncio
from unittest.mock import patch

import openai
import pytest
import respx
from httpx import Response, TimeoutException

from azure_switchboard import Deployment, DeploymentError

from .fixtures import (
    MOCK_COMPLETION,
    MOCK_COMPLETION_JSON,
    MOCK_STREAM_CHUNKS,
)
from .utils import BaseTestCase, azure_config, chat_completion_mock


@pytest.fixture(scope="function")
def mock_client(request: pytest.FixtureRequest):
    with respx.mock(base_url="https://test1.openai.azure.com") as respx_mock:
        # By default, we don't assert all routes were called
        respx_mock._assert_all_called = False

        if provided_models := request.node.get_closest_marker("mock_models"):
            respx_mock._assert_all_called = True

            # Add routes for each model
            for model in provided_models.args:
                respx_mock.post(
                    f"/openai/deployments/{model}/chat/completions",
                    name=model,
                ).respond(status_code=200, json=MOCK_COMPLETION_JSON)

        yield respx_mock


class TestDeployment(BaseTestCase):
    """Deployment functionality tests."""

    @pytest.mark.mock_models("gpt-4o-mini")
    async def test_completion(self, mock_client: respx.MockRouter):
        """Test basic chat completion functionality."""

        deployment = Deployment(azure_config("test1"))
        deployment.client.max_retries = 0

        response = await deployment.create(**self.basic_args)
        assert mock_client.routes["gpt-4o-mini"].call_count == 1
        assert response == MOCK_COMPLETION

        # Check token usage tracking
        model = deployment.models["gpt-4o-mini"]
        usage = model.get_usage()
        assert usage["tpm"] == "18/10000"
        assert usage["rpm"] == "1/60"

        # Test exception handling
        mock_client.routes["gpt-4o-mini"].side_effect = Exception("test")
        with pytest.raises(openai.APIConnectionError):
            await deployment.create(**self.basic_args)
        assert mock_client.routes["gpt-4o-mini"].call_count == 2

        # account for preflight estimate
        usage = model.get_usage()
        assert usage["tpm"] == "21/10000"
        assert usage["rpm"] == "2/60"

    async def test_streaming(self):
        """Test streaming functionality.

        It's annoying to try to mock HTTP streaming responses so we cheat
        a little bit with an AsyncMock.
        """

        deployment = Deployment(azure_config("test1"))
        deployment.client.max_retries = 0

        with patch.object(
            deployment.client.chat.completions,
            "create",
            side_effect=chat_completion_mock(),
        ) as mock:
            stream = await deployment.create(stream=True, **self.basic_args)
            mock.assert_called_once()

            # verify basic behavior
            received_chunks, content = await self.collect_chunks(stream)
            assert len(received_chunks) == len(MOCK_STREAM_CHUNKS)
            assert content == "Hello, world!"

            # Verify token usage tracking
            usage = deployment.models["gpt-4o-mini"].get_usage()
            assert usage["tpm"] == "20/10000"
            assert usage["rpm"] == "1/60"

        # verify exception handling
        with patch.object(
            deployment.client.chat.completions,
            "create",
            side_effect=Exception("test"),
        ) as mock:
            with pytest.raises(Exception, match="test"):
                stream = await deployment.create(stream=True, **self.basic_args)
                async for _ in stream:
                    pass
            mock.assert_called_once()

            usage = deployment.models["gpt-4o-mini"].get_usage()
            assert usage["tpm"] == "23/10000"
            assert usage["rpm"] == "2/60"

        # Test midstream exception handling
        with patch.object(
            deployment.client.chat.completions,
            "create",
            side_effect=chat_completion_mock(),
        ) as mock:
            stream = await deployment.create(stream=True, **self.basic_args)

            with patch.object(
                stream._self_model_ref,  # type: ignore[reportAttributeAccessIssue]
                "spend_tokens",
                side_effect=Exception("asyncstream error"),
            ):
                with pytest.raises(DeploymentError, match="Error in wrapped stream"):
                    await self.collect_chunks(stream)
                assert mock.call_count == 1
                assert not deployment.models["gpt-4o-mini"].is_healthy()

            deployment.models["gpt-4o-mini"].reset_cooldown()
            assert deployment.models["gpt-4o-mini"].is_healthy()

    async def test_cooldown(self):
        """Test model-level cooldown functionality."""

        deployment = Deployment(azure_config("test1"))
        model = deployment.models["gpt-4o-mini"]

        model.cooldown()
        assert not model.is_healthy()

        model.reset_cooldown()
        assert model.is_healthy()

    async def test_valid_model(self):
        """Test that an invalid model raises an error."""

        deployment = Deployment(azure_config("test1"))
        with pytest.raises(DeploymentError, match="gpt-fake not configured"):
            await deployment.create(model="gpt-fake", messages=[])

    async def test_usage(self):
        """Test client-level counters"""

        deployment = Deployment(azure_config("test1"))
        # Reset and verify initial state
        for model in deployment.models.values():
            model_str = str(model)
            assert "tpm=0/10000" in model_str
            assert "rpm=0/60" in model_str

        # Test client-level usage
        usage = deployment.get_usage()
        assert usage["gpt-4o-mini"]["tpm"] == "0/10000"
        assert usage["gpt-4o-mini"]["rpm"] == "0/60"
        client_str = str(deployment)
        assert "models=" in client_str

        # Set and verify values
        model = deployment.models["gpt-4o-mini"]
        model.spend_tokens(100)
        model.spend_request(5)
        usage = model.get_usage()
        assert usage["tpm"] == "100/10000"
        assert usage["rpm"] == "5/60"

        usage = deployment.get_usage()
        assert usage["gpt-4o-mini"]["tpm"] == "100/10000"
        assert usage["gpt-4o-mini"]["rpm"] == "5/60"

        # Reset and verify again
        deployment.reset_usage()
        usage = deployment.get_usage()
        assert usage["gpt-4o-mini"]["tpm"] == "0/10000"
        assert usage["gpt-4o-mini"]["rpm"] == "0/60"
        assert model._last_reset > 0

    async def test_utilization(self):
        """Test utilization calculation."""

        deployment = Deployment(azure_config("test1"))

        model = deployment.models["gpt-4o-mini"]

        # Check initial utilization (nonzero due to random splay)
        initial_util = deployment.util("gpt-4o-mini")
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
        model.cooldown()
        assert model.util == 1

    @pytest.mark.mock_models("gpt-4o-mini", "gpt-4o")
    async def test_multiple_models(self, mock_client: respx.MockRouter):
        """Test that multiple models are handled correctly."""

        deployment = Deployment(azure_config("test1"))

        gpt4o = deployment.models["gpt-4o"]
        gpt4o_mini = deployment.models["gpt-4o-mini"]

        assert gpt4o is not None
        assert gpt4o_mini is not None

        _ = await deployment.create(
            model="gpt-4o", messages=self.basic_args["messages"]
        )
        assert mock_client.routes["gpt-4o"].call_count == 1
        assert gpt4o._rpm_usage == 1
        assert gpt4o._tpm_usage > 0
        assert gpt4o_mini._tpm_usage == 0
        assert gpt4o_mini._rpm_usage == 0

        _ = await deployment.create(
            model="gpt-4o-mini", messages=self.basic_args["messages"]
        )
        assert mock_client.routes["gpt-4o-mini"].call_count == 1

        assert gpt4o._rpm_usage == 1
        assert gpt4o._tpm_usage > 0
        assert gpt4o_mini._tpm_usage > 0
        assert gpt4o_mini._rpm_usage == 1

    @pytest.mark.mock_models("gpt-4o-mini")
    async def test_concurrency(self, mock_client: respx.MockRouter):
        """Test handling of multiple concurrent requests."""
        deployment = Deployment(azure_config("test1"))

        # Create and run concurrent requests
        num_requests = 10
        tasks = [deployment.create(**self.basic_args) for _ in range(num_requests)]
        responses = await asyncio.gather(*tasks)

        # Verify results
        model = deployment.models["gpt-4o-mini"]
        assert len(responses) == num_requests
        assert all(r == MOCK_COMPLETION for r in responses)
        assert mock_client.routes["gpt-4o-mini"].call_count == num_requests
        usage = model.get_usage()
        assert usage["tpm"] == f"{18 * num_requests}/10000"
        assert usage["rpm"] == f"{num_requests}/60"

    @pytest.mark.mock_models("gpt-4o-mini")
    async def test_timeout_retry(self, mock_client: respx.MockRouter):
        """Test timeout retry behavior."""

        deployment = Deployment(azure_config("test1"))

        # Test successful retry after timeouts
        expected_response = Response(status_code=200, json=MOCK_COMPLETION_JSON)
        mock_client.routes["gpt-4o-mini"].side_effect = [
            TimeoutException("Timeout 1"),
            TimeoutException("Timeout 2"),
            expected_response,
        ]
        response = await deployment.create(**self.basic_args)
        assert response == MOCK_COMPLETION
        assert mock_client.routes["gpt-4o-mini"].call_count == 3

        # Test failure after max retries
        mock_client.routes["gpt-4o-mini"].reset()
        mock_client.routes["gpt-4o-mini"].side_effect = [
            TimeoutException("Timeout 1"),
            TimeoutException("Timeout 2"),
            TimeoutException("Timeout 3"),
        ]

        with pytest.raises(openai.APITimeoutError):
            await deployment.create(**self.basic_args)
        assert mock_client.routes["gpt-4o-mini"].call_count == 3
        assert not deployment.is_healthy("gpt-4o-mini")
