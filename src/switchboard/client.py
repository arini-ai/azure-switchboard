import logging
import random
import time
from typing import Annotated, AsyncGenerator

from openai import AsyncAzureOpenAI
from openai.types.chat import ChatCompletion, ChatCompletionChunk
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class Deployment(BaseModel):
    """Metadata about the Azure deployment"""

    name: str
    api_base: str
    api_key: str
    api_version: str = "2024-10-21"
    timeout: float = 30.0
    tpm_ratelimit: Annotated[int, Field(description="TPM Ratelimit")] = 0
    rpm_ratelimit: Annotated[int, Field(description="RPM Ratelimit")] = 0
    healthcheck_interval: int = 30


class Client:
    """Runtime state of a deployment"""

    def __init__(
        self, config: Deployment, client: AsyncAzureOpenAI | None = None
    ) -> None:
        self.name = config.name
        self.config = config
        self.client = client or AsyncAzureOpenAI(
            api_key=config.api_key,
            api_version=config.api_version,
            base_url=config.api_base,
            timeout=config.timeout,
        )
        self.healthy = True

        self.ratelimit_tokens = 0
        self.ratelimit_requests = 0
        self.last_reset = time.time()

    def __str__(self):
        ents = {
            "name": self.name,
            "healthy": self.healthy,
            "tokens": self.ratelimit_tokens,
            "requests": self.ratelimit_requests,
            "utilization": self.utilization,
        }
        return f"Client({', '.join([f'{k}={v}' for k, v in ents.items()])})"

    @property
    def utilization(self) -> float:
        """
        Calculate the load weight of this client.
        Lower weight means this client is a better choice for new requests.
        """
        # If not healthy, return infinity (never choose)
        if not self.healthy:
            return float("inf")

        # Calculate token utilization (as a percentage of max)
        token_util = (
            self.ratelimit_tokens / self.config.tpm_ratelimit
            if self.config.tpm_ratelimit > 0
            else 0
        )

        # Azure allocates RPM at a ratio of 6:1000 to TPM
        request_util = (
            self.ratelimit_requests / self.config.rpm_ratelimit
            if self.config.rpm_ratelimit > 0
            else 0
        )

        # Use the higher of the two utilizations as the weight
        # Add a small random factor to prevent oscillation
        return max(token_util, request_util) + random.uniform(0, 0.01)

    async def check_liveness(self):
        try:
            logger.debug(f"{self}: checking liveness")
            await self.client.models.list()
            self.healthy = True
        except Exception:
            logger.exception(f"{self}: liveness check failed")
            self.healthy = False

    async def reset_counters(self):
        """Reset usage counters - should be called periodically"""
        logger.debug(f"{self}: resetting usage counters")
        self.ratelimit_tokens = 0
        self.ratelimit_requests = 0
        self.last_reset = time.time()

    async def completion(self, **kwargs) -> ChatCompletion:
        """
        Send a chat completion request to this client.
        Tracks usage metrics for load balancing.
        """

        kwargs["timeout"] = kwargs.get("timeout", self.config.timeout)
        if kwargs.get("stream", False):
            kwargs["stream_options"] = {"include_usage": True}

        self.ratelimit_requests += 1
        response = await self.client.chat.completions.create(**kwargs)

        if hasattr(response, "usage"):
            self.ratelimit_tokens += response.usage.total_tokens

        return response

    async def stream(self, **kwargs) -> AsyncGenerator[ChatCompletionChunk, None]:
        """
        Send a streaming request to this client.
        Tracks usage metrics for load balancing.
        """
        kwargs.pop("stream", None)  # pop and pass ourselves to aid the typechecker
        kwargs["stream_options"] = kwargs.get("stream_options", {"include_usage": True})
        self.ratelimit_requests += 1
        stream = await self.client.chat.completions.create(stream=True, **kwargs)

        async for chunk in stream:
            if chunk.usage:  # last chunk has usage info
                self.ratelimit_tokens += chunk.usage.total_tokens

            yield chunk


def azure_client_factory(deployment: Deployment) -> AsyncAzureOpenAI:
    return AsyncAzureOpenAI(
        azure_endpoint=deployment.api_base,
        api_key=deployment.api_key,
        api_version=deployment.api_version,
        timeout=deployment.timeout,
    )


def default_client_factory(deployment: Deployment) -> Client:
    return Client(config=deployment, client=azure_client_factory(deployment))
