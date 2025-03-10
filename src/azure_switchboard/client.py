import logging
import random
import time
from typing import Annotated, AsyncIterator, Literal, cast, overload

import wrapt
from openai import AsyncAzureOpenAI, AsyncStream
from openai.types.chat import ChatCompletion, ChatCompletionChunk
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class Deployment(BaseModel):
    """Metadata about the Azure deployment"""

    name: str
    api_base: str
    api_key: str
    api_version: str = "2024-10-21"
    timeout: float = 600.0
    tpm_ratelimit: Annotated[int, Field(description="TPM Ratelimit")] = 0
    rpm_ratelimit: Annotated[int, Field(description="RPM Ratelimit")] = 0
    healthcheck_interval: int = 30
    cooldown_period: int = 60


class Client:
    """Runtime state of a deployment"""

    def __init__(self, config: Deployment, client: AsyncAzureOpenAI) -> None:
        self.config = config
        self.client = client
        self._cooldown_until = 0

        self.ratelimit_tokens = 0
        self.ratelimit_requests = 0
        self.last_reset = 0

    def __repr__(self) -> str:
        return f"Client(name={self.config.name}, util={self.util}, tpm={self.ratelimit_tokens}/{self.config.tpm_ratelimit}, rpm={self.ratelimit_requests}/{self.config.rpm_ratelimit})"

    def cooldown(self, seconds: int = 0) -> None:
        self._cooldown_until = time.time() + (seconds or self.config.cooldown_period)

    def reset_cooldown(self) -> None:
        self._cooldown_until = 0

    @property
    def healthy(self) -> bool:
        # the deployment is healthy if we have available capacity
        # to serve requests
        return self.util < 1

    @property
    def util(self) -> float:
        """
        Calculate the load weight of this client as a value between 0 and 1.
        Lower weight means this client is a better choice for new requests.
        """
        # return full utilization if we're cooling down to avoid selection
        if time.time() < self._cooldown_until:
            return 1

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
        return round(max(token_util, request_util) + random.uniform(0, 0.01), 3)

    async def check_health(self):
        try:
            logger.debug(f"{self}: checking health")
            await self.client.models.list()
            # maybe bring back cooldown reset when have a better HC
            # self.reset_cooldown()
        except Exception:
            logger.exception(f"{self}: health check failed")
            self.cooldown()

    def reset_counters(self):
        """Reset usage counters - should be called periodically"""

        logger.debug(f"{self}: resetting ratelimit counters")
        self.ratelimit_tokens = 0
        self.ratelimit_requests = 0
        self.last_reset = time.time()

    def get_counters(self) -> dict[str, int | float | str]:
        return {
            "util": self.util,
            "tokens": self.ratelimit_tokens,
            "requests": self.ratelimit_requests,
        }

    @overload
    async def create(
        self, *, stream: Literal[True], **kwargs
    ) -> AsyncStream[ChatCompletionChunk]: ...

    @overload
    async def create(self, **kwargs) -> ChatCompletion: ...

    async def create(
        self,
        *,
        stream: bool = False,
        **kwargs,
    ) -> ChatCompletion | AsyncStream[ChatCompletionChunk]:
        """
        Send a chat completion request to this client.
        Tracks usage metrics for load balancing.
        """

        self.ratelimit_requests += 1
        # add input token estimate before we send the request so utilization is
        # kept up to date for concurrent requests.
        _preflight_util_estimate = self._estimate_input_tokens(kwargs["messages"])
        self.ratelimit_tokens += _preflight_util_estimate

        kwargs["timeout"] = kwargs.get("timeout", self.config.timeout)
        try:
            if stream:
                stream_options = kwargs.pop("stream_options", {})
                stream_options["include_usage"] = True

                logging.debug("Creating chat completion stream")
                response = await self.client.chat.completions.create(
                    stream=True,
                    stream_options=stream_options,
                    **kwargs,
                )

                # streaming util gets updated inside the WrappedAsyncStream
                return _WrappedAsyncStream(
                    response, self, usage_adjustment=_preflight_util_estimate
                )

            else:
                logging.debug("Creating chat completion")
                response = cast(
                    ChatCompletion, await self.client.chat.completions.create(**kwargs)
                )
                if response.usage:
                    logging.debug("Chat completion usage: %s", response.usage)
                    self.ratelimit_tokens += (
                        # dont double-count our preflight estimate
                        response.usage.total_tokens - _preflight_util_estimate
                    )

                return response
        except Exception as e:
            self.cooldown()
            raise e

    def _estimate_input_tokens(self, messages: list[dict]) -> int:
        # loose estimate of input token count. openai says roughly 4
        # characters per token, so sum len of messages and divide by 4.
        return sum(len(m.get("content", "")) for m in messages) // 4


class _WrappedAsyncStream(wrapt.ObjectProxy):
    """Wrap an openai.AsyncStream to track usage"""

    def __init__(
        self,
        wrapped: AsyncStream[ChatCompletionChunk],
        runtime: Client,
        usage_adjustment: int = 0,
    ):
        super(_WrappedAsyncStream, self).__init__(wrapped)
        self._self_runtime = runtime
        self._self_adjustment = usage_adjustment

    async def __anext__(self) -> ChatCompletionChunk:
        chunk: ChatCompletionChunk = await self.__wrapped__.__anext__()
        if chunk.usage:
            self._self_runtime.ratelimit_tokens += (
                # dont double-count our preflight estimate
                chunk.usage.total_tokens - self._self_adjustment
            )
        return chunk

    async def __aiter__(self) -> AsyncIterator[ChatCompletionChunk]:
        async for chunk in self.__wrapped__:
            chunk = cast(ChatCompletionChunk, chunk)
            # only the last chunk contains the usage info
            if chunk.usage:
                self._self_runtime.ratelimit_tokens += (
                    # dont double-count our preflight estimate
                    chunk.usage.total_tokens - self._self_adjustment
                )
            yield chunk


def azure_client_factory(deployment: Deployment) -> AsyncAzureOpenAI:
    return AsyncAzureOpenAI(
        azure_endpoint=deployment.api_base,
        api_key=deployment.api_key,
        api_version=deployment.api_version,
    )


def default_client_factory(deployment: Deployment) -> Client:
    return Client(config=deployment, client=azure_client_factory(deployment))
