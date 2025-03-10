import logging
import random
import time
from typing import Annotated, AsyncIterator, Literal, cast, overload

import wrapt
from openai import AsyncAzureOpenAI, AsyncStream
from openai.types.chat import ChatCompletion, ChatCompletionChunk
from pydantic import BaseModel, Field, PrivateAttr

logger = logging.getLogger(__name__)


class SwitchboardClientError(Exception):
    pass


class ModelConfig(BaseModel):
    """Model-specific configuration on a deployment"""

    model: str = Field(description="Model name")
    tpm_ratelimit: Annotated[int, Field(description="TPM Ratelimit")] = 0
    rpm_ratelimit: Annotated[int, Field(description="RPM Ratelimit")] = 0

    tpm_usage: Annotated[int, Field(description="TPM Usage")] = 0
    rpm_usage: Annotated[int, Field(description="RPM Usage")] = 0

    default_cooldown: float = Field(
        default=60.0, repr=False, description="Default cooldown period in seconds"
    )
    _cooldown_until: float = PrivateAttr(default=0)
    _last_reset: float = PrivateAttr(default=0)

    def cooldown(self, seconds: float = 0.0) -> None:
        self._cooldown_until = time.time() + (seconds or self.default_cooldown)

    def reset_cooldown(self) -> None:
        self._cooldown_until = 0

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
            self.tpm_usage / self.tpm_ratelimit if self.tpm_ratelimit > 0 else 0
        )

        # Azure allocates RPM at a ratio of 6:1000 to TPM
        request_util = (
            self.rpm_usage / self.rpm_ratelimit if self.rpm_ratelimit > 0 else 0
        )

        # Use the higher of the two utilizations as the weight
        # Add a small random factor to prevent oscillation
        return round(max(token_util, request_util) + random.uniform(0, 0.01), 3)

    def is_healthy(self) -> bool:
        """
        Check if the model is healthy based on utilization.
        """
        return self.util() < 1

    def reset_usage(self) -> None:
        """Call periodically to reset usage counters"""

        logger.debug(f"{self}: resetting ratelimit counters")
        self.tpm_usage = 0
        self.rpm_usage = 0
        self._last_reset = time.time()

    def get_usage(self) -> dict[str, str | float]:
        return {
            "util": self.util(),
            "tpm": f"{self.tpm_usage}/{self.tpm_ratelimit}",
            "rpm": f"{self.rpm_usage}/{self.rpm_ratelimit}",
        }


class Deployment(BaseModel):
    """Metadata about the Azure deployment"""

    name: str
    api_base: str
    api_key: str
    api_version: str = "2024-10-21"
    timeout: float = 600.0
    healthcheck_interval: int = 30
    cooldown_period: int = 60
    models: dict[str, ModelConfig] = Field(default_factory=dict)


class Client:
    """Runtime state of a deployment"""

    def __init__(self, config: Deployment, client: AsyncAzureOpenAI) -> None:
        self.config = config
        self.client = client

    def __repr__(self) -> str:
        models = {m.model: m.get_usage() for m in self.config.models.values()}
        return f"Client(name={self.config.name}, models={models}])"

    def reset_usage(self) -> None:
        for model in self.config.models.values():
            model.reset_usage()

    def get_usage(self) -> dict[str, dict]:
        return {m.model: m.get_usage() for m in self.config.models.values()}

    def is_healthy(self, model: str) -> bool:
        return self.config.models[model].is_healthy()

    def util(self, model: str) -> float:
        return self.config.models[model].util()

    @overload
    async def create(
        self, *, model: str, stream: Literal[True], **kwargs
    ) -> AsyncStream[ChatCompletionChunk]: ...

    @overload
    async def create(self, *, model: str, **kwargs) -> ChatCompletion: ...

    async def create(
        self,
        *,
        model: str,
        stream: bool = False,
        **kwargs,
    ) -> ChatCompletion | AsyncStream[ChatCompletionChunk]:
        """
        Send a chat completion request to this client.
        Tracks usage metrics for load balancing.
        """

        if model not in self.config.models:
            raise SwitchboardClientError(f"{model} not configured for deployment")

        model_stats = self.config.models[model]

        model_stats.rpm_usage += 1
        # add input token estimate before we send the request so utilization is
        # kept up to date for concurrent requests.
        _preflight_estimate = self._estimate_input_tokens(kwargs["messages"])
        model_stats.tpm_usage += _preflight_estimate

        kwargs["timeout"] = kwargs.get("timeout", self.config.timeout)
        try:
            if stream:
                stream_options = kwargs.pop("stream_options", {})
                stream_options["include_usage"] = True

                logging.debug("Creating streaming completion")
                response = await self.client.chat.completions.create(
                    model=model,
                    stream=True,
                    stream_options=stream_options,
                    **kwargs,
                )

                # streaming util gets updated inside the WrappedAsyncStream
                return _WrappedAsyncStream(
                    response, model_stats, usage_adjustment=_preflight_estimate
                )

            else:
                logging.debug("Creating chat completion")
                response = cast(
                    ChatCompletion,
                    await self.client.chat.completions.create(model=model, **kwargs),
                )
                if response.usage:
                    model_stats.tpm_usage += (
                        # dont double-count our preflight estimate
                        response.usage.total_tokens - _preflight_estimate
                    )

                return response
        except Exception as e:
            model_stats.cooldown()
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
        runtime: ModelConfig,
        usage_adjustment: int = 0,
    ):
        super(_WrappedAsyncStream, self).__init__(wrapped)
        self._self_runtime = runtime
        self._self_adjustment = usage_adjustment

    async def __anext__(self) -> ChatCompletionChunk:
        chunk: ChatCompletionChunk = await self.__wrapped__.__anext__()
        if chunk.usage:
            self._self_runtime.tpm_usage += (
                # dont double-count our preflight estimate
                chunk.usage.total_tokens - self._self_adjustment
            )
        return chunk

    async def __aiter__(self) -> AsyncIterator[ChatCompletionChunk]:
        async for chunk in self.__wrapped__:
            chunk = cast(ChatCompletionChunk, chunk)
            # only the last chunk contains the usage info
            if chunk.usage:
                self._self_runtime.tpm_usage += (
                    # dont double-count our preflight estimate
                    chunk.usage.total_tokens - self._self_adjustment
                )
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
