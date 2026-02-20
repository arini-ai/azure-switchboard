from unittest.mock import AsyncMock, patch

import pytest
from httpx import Request, Response
from openai import APIConnectionError, RateLimitError

from azure_switchboard import Switchboard, SwitchboardError
from azure_switchboard.deployment import DeploymentState

from .conftest import (
    PARSED_COMPLETION_PARAMS,
    PARSED_RESPONSE,
    WeatherResult,
)


class TestDeploymentParse:
    """DeploymentState.parse() tests — mirrors TestDeployment completion tests."""

    async def test_parse_returns_parsed_completion(self, deployment: DeploymentState):
        """Test basic parse returns ParsedChatCompletion with correct parsed model."""
        with patch.object(
            deployment.client.beta.chat.completions,
            "parse",
            new=AsyncMock(return_value=PARSED_RESPONSE),
        ) as mock:
            response = await deployment.parse(**PARSED_COMPLETION_PARAMS)

            mock.assert_called_once()
            assert response == PARSED_RESPONSE
            assert response.choices[0].message.parsed is not None
            assert isinstance(response.choices[0].message.parsed, WeatherResult)
            assert response.choices[0].message.parsed.city == "Paris"

    async def test_parse_tracks_usage(self, deployment: DeploymentState):
        """Test that parse() updates TPM/RPM counters like create() does."""
        with patch.object(
            deployment.client.beta.chat.completions,
            "parse",
            new=AsyncMock(return_value=PARSED_RESPONSE),
        ):
            await deployment.parse(**PARSED_COMPLETION_PARAMS)

            model = deployment.model("gpt-4o-mini")
            usage = model.stats()
            assert usage.tpm.startswith(str(PARSED_RESPONSE.usage.total_tokens))
            assert usage.rpm.startswith("1")

    async def test_parse_invalid_model(self, deployment: DeploymentState):
        """Test that an unconfigured model raises SwitchboardError."""
        with pytest.raises(SwitchboardError, match="gpt-fake not configured"):
            await deployment.parse(
                model="gpt-fake",
                messages=[],
                response_format=WeatherResult,
            )

    async def test_parse_rate_limit_marks_down(self, deployment: DeploymentState):
        """Test that RateLimitError marks model down and raises SwitchboardError."""
        rate_limit_error = RateLimitError(
            "rate limited",
            response=Response(
                status_code=429,
                request=Request(
                    "POST",
                    "https://test.openai.azure.com/openai/v1/chat/completions",
                ),
            ),
            body={"error": {"message": "rate limited"}},
        )

        with patch.object(
            deployment.client.beta.chat.completions,
            "parse",
            new=AsyncMock(side_effect=rate_limit_error),
        ):
            with pytest.raises(
                SwitchboardError,
                match="Rate limit exceeded in deployment parse",
            ):
                await deployment.parse(**PARSED_COMPLETION_PARAMS)

            assert not deployment.model("gpt-4o-mini").is_healthy()

    async def test_parse_generic_exception_does_not_mark_down(
        self, deployment: DeploymentState
    ):
        """Test that generic exceptions do NOT mark model down — only network/rate-limit errors do."""
        with patch.object(
            deployment.client.beta.chat.completions,
            "parse",
            new=AsyncMock(side_effect=Exception("upstream error")),
        ):
            with pytest.raises(Exception, match="upstream error"):
                await deployment.parse(**PARSED_COMPLETION_PARAMS)

            assert deployment.model("gpt-4o-mini").is_healthy()

    async def test_parse_connection_error_marks_down(self, deployment: DeploymentState):
        """Test that APIConnectionError marks model down and raises RuntimeError."""
        connection_error = APIConnectionError(
            request=Request(
                "POST",
                "https://test.openai.azure.com/openai/v1/chat/completions",
            )
        )
        with patch.object(
            deployment.client.beta.chat.completions,
            "parse",
            new=AsyncMock(side_effect=connection_error),
        ):
            with pytest.raises(
                RuntimeError, match="Connection error in deployment parse"
            ):
                await deployment.parse(**PARSED_COMPLETION_PARAMS)

            assert not deployment.model("gpt-4o-mini").is_healthy()


class TestSwitchboardParse:
    """Switchboard.parse() tests — mirrors TestSwitchboard completion tests."""

    async def test_parse(self, switchboard: Switchboard):
        """Test parse through switchboard with load balancing."""
        with patch(
            "azure_switchboard.deployment.DeploymentState.parse",
            new=AsyncMock(return_value=PARSED_RESPONSE),
        ) as mock:
            response = await switchboard.parse(**PARSED_COMPLETION_PARAMS)

            mock.assert_called_once()
            assert response == PARSED_RESPONSE
            assert response.choices[0].message.parsed.city == "Paris"

    async def test_parse_session_affinity(self, switchboard: Switchboard):
        """Test that session_id routes to same deployment."""
        with patch(
            "azure_switchboard.deployment.DeploymentState.parse",
            new=AsyncMock(return_value=PARSED_RESPONSE),
        ):
            await switchboard.parse(
                session_id="test-session", **PARSED_COMPLETION_PARAMS
            )
            deployment_1 = switchboard.sessions["test-session"]

            await switchboard.parse(
                session_id="test-session", **PARSED_COMPLETION_PARAMS
            )
            deployment_2 = switchboard.sessions["test-session"]

            assert deployment_1.name == deployment_2.name

    async def test_parse_failover(self, switchboard: Switchboard):
        """Test that parse fails over to another deployment on error."""
        call_count = 0

        async def failing_then_success(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise RuntimeError("first attempt fails")
            return PARSED_RESPONSE

        with patch(
            "azure_switchboard.deployment.DeploymentState.parse",
            new=AsyncMock(side_effect=failing_then_success),
        ):
            response = await switchboard.parse(**PARSED_COMPLETION_PARAMS)
            assert response == PARSED_RESPONSE
            assert call_count == 2

    async def test_parse_invalid_model(self, switchboard: Switchboard):
        """Test that an invalid model raises SwitchboardError."""
        with pytest.raises(
            SwitchboardError,
            match="No deployments available for invalid-model",
        ):
            await switchboard.parse(
                model="invalid-model",
                messages=[],
                response_format=WeatherResult,
            )
