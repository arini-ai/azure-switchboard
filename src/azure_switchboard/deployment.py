from __future__ import annotations

import asyncio
import logging
import random
import time
from typing import AsyncIterator, Literal, cast, overload

import wrapt
from openai import AsyncAzureOpenAI, AsyncOpenAI, AsyncStream
from openai.types.chat import ChatCompletion, ChatCompletionChunk
from opentelemetry import metrics
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

# Create a meter for recording metrics
meter = metrics.get_meter("azure_switchboard.deployment")
model_util_gauge = meter.create_gauge(
    name="model_utilization",
    description="Current utilization of a model deployment (0-1)",
    unit="percent",
)
token_usage_counter = meter.create_counter(
    name="token_usage",
    description="Number of tokens used by a model deployment",
    unit="tokens",
)
request_usage_counter = meter.create_counter(
    name="request_usage",
    description="Number of requests sent to a model deployment",
    unit="requests",
)


class Model:
    """Runtime state of a model on a deployment"""

    def __init__(
        self,
        name: str,
        tpm: int = 0,
        rpm: int = 0,
        default_cooldown: float = 10.0,
    ):
        self.name = name
        self.tpm = tpm
        self.rpm = rpm
        self.default_cooldown = default_cooldown

        self._tpm_usage: int = 0
        self._rpm_usage: int = 0
        self._cooldown_until: float = 0
        self._last_reset: float = 0

    def cooldown(self, seconds: float = 0.0) -> None:
        self._cooldown_until = time.time() + (seconds or self.default_cooldown)

    def reset_cooldown(self) -> None:
        self._cooldown_until = 0

    def is_healthy(self) -> bool:
        """
        Check if the model is healthy based on utilization.
        """
        return self.util < 1

    def is_cooling(self) -> bool:
        return time.time() < self._cooldown_until

    @property
    def util(self) -> float:
        """
        Calculate the load weight of this client as a value between 0 and 1.
        Lower weight means this client is a better choice for new requests.
        """
        # return full utilization if we're cooling down to avoid selection
        if self.is_cooling():
            return 1

        # Calculate token utilization (as a percentage of max)
        token_util = self._tpm_usage / self.tpm if self.tpm > 0 else 0

        # Azure allocates RPM at a ratio of 6:1000 to TPM
        request_util = self._rpm_usage / self.rpm if self.rpm > 0 else 0

        # Use the higher of the two utilizations as the weight
        # Add a small random factor to prevent oscillation
        utilization = round(max(token_util, request_util) + random.uniform(0, 0.01), 3)

        # Record the utilization metric
        model_util_gauge.set(utilization, {"model": self.name})

        return utilization

    def reset_usage(self) -> None:
        """Call periodically to reset usage counters"""

        logger.debug(f"{self}: resetting ratelimit counters")
        self._tpm_usage = 0
        self._rpm_usage = 0
        self._last_reset = time.time()

    def get_usage(self) -> dict[str, str | float]:
        return {
            "util": self.util,
            "tpm": f"{self._tpm_usage}/{self.tpm}",
            "rpm": f"{self._rpm_usage}/{self.rpm}",
        }

    def spend_request(self, n: int = 1) -> None:
        self._rpm_usage += n
        # Record request usage metric
        request_usage_counter.add(n, {"model": self.name})

    def spend_tokens(self, n: int) -> None:
        self._tpm_usage += n

    def __repr__(self) -> str:
        elems = ", ".join(f"{k}={v}" for k, v in self.get_usage().items())
        return f"{self.__class__.__name__}(name={self.name}, {elems})"


class AzureDeployment(BaseModel, arbitrary_types_allowed=True):
    """Metadata about an Azure deployment"""

    name: str
    endpoint: str
    api_key: str
    api_version: str = "2024-10-21"
    timeout: float = 600.0
    models: list[Model]
    client: AsyncAzureOpenAI | None = None

    def get_client(self) -> AsyncAzureOpenAI:
        return AsyncAzureOpenAI(
            azure_endpoint=self.endpoint,
            api_key=self.api_key,
            api_version=self.api_version,
            timeout=self.timeout,
        )


class OpenAIDeployment(BaseModel, arbitrary_types_allowed=True):
    """Metadata about an OpenAI deployment"""

    name: str = "openai"
    api_key: str | None = None
    base_url: str | None = None
    timeout: float = 600.0
    models: list[Model] = Field(
        default_factory=lambda: [Model(name="gpt-4o-mini"), Model(name="gpt-4o")]
    )
    client: AsyncOpenAI | None = None

    def get_client(self) -> AsyncOpenAI:
        return AsyncOpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
            timeout=self.timeout,
        )


class DeploymentError(Exception):
    pass


class Deployment:
    """Runtime state of a deployment"""

    def __init__(
        self,
        config: AzureDeployment | OpenAIDeployment,
    ) -> None:
        self.config = config
        self.client = config.get_client()
        self.models = {m.name: m for m in config.models}

    def __repr__(self) -> str:
        elems = ", ".join(map(str, self.models.values()))
        return f"Deployment(name={self.config.name}, models=[{elems}])"

    def reset_usage(self) -> None:
        for model in self.models.values():
            model.reset_usage()

    def get_usage(self) -> dict[str, dict]:
        return {name: state.get_usage() for name, state in self.models.items()}

    def is_healthy(self, model: str) -> bool:
        return self.models[model].is_healthy()

    def util(self, model: str) -> float:
        return self.models[model].util

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

        if model not in self.models:
            raise DeploymentError(f"{model} not configured for deployment")

        # add input token estimate before we send the request so utilization is
        # kept up to date for other requests that might be executing concurrently.
        _preflight_estimate = self._estimate_token_usage(kwargs)
        self.models[model].spend_tokens(_preflight_estimate)
        self.models[model].spend_request()

        kwargs["timeout"] = kwargs.get("timeout", self.config.timeout)
        try:
            if stream:
                logging.debug("Creating streaming completion")
                response_stream = await self.client.chat.completions.create(
                    model=model,
                    stream=True,
                    stream_options=kwargs.pop(
                        "stream_options", {"include_usage": True}
                    ),
                    **kwargs,
                )

                # streaming util gets updated inside the WrappedAsyncStream
                return _AsyncStreamWrapper(
                    wrapped=response_stream,
                    model_ref=self.models[model],
                    usage_adjustment=_preflight_estimate,
                )

            else:
                logging.debug("Creating chat completion")
                response = await self.client.chat.completions.create(
                    model=model, **kwargs
                )
                response = cast(ChatCompletion, response)

                if response.usage:
                    self.models[model].spend_tokens(
                        # dont double-count our preflight estimate
                        response.usage.total_tokens - _preflight_estimate
                    )

                return response
        except Exception:
            self.models[model].cooldown()
            raise

    def _estimate_token_usage(self, kwargs: dict) -> int:
        # loose estimate of token cost. were only considering
        # input tokens for now, we can add output estimates as well later.
        # openai says roughly 4 characters per token, so sum len of messages
        # and divide by 4.
        t_input = sum(len(m.get("content", "")) for m in kwargs.get("messages", []))
        # t_output = kwargs.get("max_tokens", 500)
        return t_input // 4


class _AsyncStreamWrapper(wrapt.ObjectProxy):
    """Wrap an openai.AsyncStream to track usage"""

    def __init__(
        self,
        wrapped: AsyncStream[ChatCompletionChunk],
        model_ref: Model,
        usage_adjustment: int = 0,
    ):
        super(_AsyncStreamWrapper, self).__init__(wrapped)
        self._self_model_ref = model_ref
        self._self_adjustment = usage_adjustment

    async def __aiter__(self) -> AsyncIterator[ChatCompletionChunk]:
        try:
            async for chunk in self.__wrapped__:
                chunk = cast(ChatCompletionChunk, chunk)
                # only the last chunk contains the usage info
                if chunk.usage:
                    self._self_model_ref.spend_tokens(
                        # dont double-count our preflight estimate
                        chunk.usage.total_tokens - self._self_adjustment
                    )
                yield chunk
        except asyncio.CancelledError:  # pragma: no cover
            logger.debug("Cancelled mid-stream")
            return
        except Exception as e:
            self._self_model_ref.cooldown()
            raise DeploymentError("Error in wrapped stream") from e
