from __future__ import annotations

import asyncio
import logging
import random
import time
from typing import Annotated, AsyncIterator, Literal, cast, overload

import wrapt
from openai import AsyncAzureOpenAI, AsyncStream
from openai.types.chat import ChatCompletion, ChatCompletionChunk
from pydantic import BaseModel, Field, PrivateAttr, computed_field

logger = logging.getLogger(__name__)


class Model(BaseModel):
    """Runtime state of a model on a deployment"""

    name: str
    tpm: Annotated[int, Field(description="TPM Ratelimit")] = 0
    rpm: Annotated[int, Field(description="RPM Ratelimit")] = 0
    default_cooldown: Annotated[
        float, Field(repr=False, description="Default cooldown period in seconds")
    ] = 10.0

    _tpm_usage: int = PrivateAttr(default=0)
    _rpm_usage: int = PrivateAttr(default=0)
    _cooldown_until: float = PrivateAttr(default=0)
    _last_reset: float = PrivateAttr(default=0)

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

    @computed_field
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
        return round(max(token_util, request_util) + random.uniform(0, 0.01), 3)

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

    def spend_tokens(self, n: int) -> None:
        self._tpm_usage += n

    def __str__(self) -> str:
        return " ".join(f"{k}={v}" for k, v in self.get_usage().items())


class DeploymentConfig(BaseModel):
    """Metadata about the Azure deployment"""

    name: str
    api_base: str
    api_key: str
    api_version: str = "2024-10-21"
    timeout: float = 600.0
    models: list[Model]


class DeploymentError(Exception):
    pass


class Deployment:
    """Runtime state of a deployment"""

    def __init__(self, config: DeploymentConfig, client: AsyncAzureOpenAI) -> None:
        self.config = config
        self.client = client
        self.models = {m.name: m for m in config.models}

    def __repr__(self) -> str:
        return f"Client(name={self.config.name}, models={self.models})"

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
                stream_options = kwargs.pop("stream_options", {})
                stream_options["include_usage"] = True

                logging.debug("Creating streaming completion")
                response_stream = await self.client.chat.completions.create(
                    model=model,
                    stream=True,
                    stream_options=stream_options,
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
        except Exception as e:
            self.models[model].cooldown()
            raise e

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
        except Exception:
            logger.exception("Error in wrapped stream")
            self._self_model_ref.cooldown()
            raise


def azure_client_factory(deployment: DeploymentConfig) -> AsyncAzureOpenAI:
    return AsyncAzureOpenAI(
        azure_endpoint=deployment.api_base,
        api_key=deployment.api_key,
        api_version=deployment.api_version,
        timeout=deployment.timeout,
    )


def default_deployment_factory(deployment: DeploymentConfig) -> Deployment:
    return Deployment(config=deployment, client=azure_client_factory(deployment))
