from unittest.mock import AsyncMock, patch

import pytest
from httpx import Request, Response
from openai import RateLimitError

from azure_switchboard import SwitchboardError
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

    async def test_parse_exception_marks_down(self, deployment: DeploymentState):
        """Test that generic exceptions mark model down and raise RuntimeError."""
        with patch.object(
            deployment.client.beta.chat.completions,
            "parse",
            new=AsyncMock(side_effect=Exception("upstream error")),
        ):
            with pytest.raises(RuntimeError, match="Error in deployment parse"):
                await deployment.parse(**PARSED_COMPLETION_PARAMS)

            assert not deployment.model("gpt-4o-mini").is_healthy()
