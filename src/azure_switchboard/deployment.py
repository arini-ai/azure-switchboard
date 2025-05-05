from __future__ import annotations

import asyncio
import logging
from typing import AsyncIterator, Literal, cast, overload

import wrapt
from openai import AsyncAzureOpenAI, AsyncOpenAI, AsyncStream
from openai.types.chat import ChatCompletion, ChatCompletionChunk
from openai.types.completion_usage import CompletionUsage
from opentelemetry import trace
from pydantic import BaseModel, Field

from azure_switchboard.model import UtilStats

from .model import Model

logger = logging.getLogger(__name__)


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
        default_factory=lambda: [
            Model(name="gpt-4o"),
            Model(name="gpt-4o-mini"),
            Model(name="gpt-4.1"),
            Model(name="gpt-4.1-mini"),
            Model(name="gpt-4.1-nano"),
        ]
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
        return f"Deployment<{self.config.name}>([{elems}])"

    def reset_usage(self) -> None:
        for model in self.models.values():
            model.reset_usage()

    def stats(self) -> dict[str, UtilStats]:
        return {name: model.stats() for name, model in self.models.items()}

    def is_healthy(self, model: str) -> bool:
        if model not in self.models:
            return False
        return self.model(model).is_healthy()

    def util(self, model: str) -> float:
        return self.model(model).util

    def model(self, name: str) -> Model:
        return self.models[name]

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

                # streaming util gets updated inside _AsyncStreamWrapper
                return _AsyncStreamWrapper(
                    stream=response_stream,
                    deployment=self,
                    model=self.models[model],
                    offset=_preflight_estimate,
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
                    self._set_span_attributes(response.usage)

                return response
        except Exception as e:
            logger.exception(
                f"marking down {self.config.name}/{model} for chat completion error"
            )
            self.models[model].mark_down()
            raise DeploymentError("Error in deployment chat completion") from e

    def _estimate_token_usage(self, kwargs: dict) -> int:
        # loose estimate of token cost. were only considering
        # input tokens for now, we can add output estimates as well later.
        # openai says roughly 4 characters per token, so sum len of messages
        # and divide by 4.
        t_input = sum(len(m.get("content", "")) for m in kwargs.get("messages", []))
        # t_output = kwargs.get("max_tokens", 500)
        return t_input // 4

    def _set_span_attributes(self, usage: CompletionUsage) -> None:
        span = trace.get_current_span()
        if prompt := usage.prompt_tokens_details:
            if prompt.cached_tokens:
                span.set_attribute("gen_ai.usage.cached_tokens", prompt.cached_tokens)
        if completion := usage.completion_tokens_details:
            if completion.reasoning_tokens:
                span.set_attribute(
                    "gen_ai.usage.reasoning_tokens", completion.reasoning_tokens
                )


class _AsyncStreamWrapper(wrapt.ObjectProxy):
    """Wrap an openai.AsyncStream to track usage"""

    def __init__(
        self,
        stream: AsyncStream[ChatCompletionChunk],
        deployment: Deployment,
        model: Model,
        offset: int = 0,
    ):
        super().__init__(stream)
        self._self_deployment = deployment
        self._self_model = model
        self._self_offset = offset

    async def __aiter__(self) -> AsyncIterator[ChatCompletionChunk]:
        try:
            async for chunk in self.__wrapped__:
                chunk = cast(ChatCompletionChunk, chunk)
                # only the last chunk contains the usage info
                if chunk.usage:
                    self._self_model.spend_tokens(
                        # dont double-count our preflight estimate
                        chunk.usage.total_tokens - self._self_offset
                    )
                    self._self_deployment._set_span_attributes(chunk.usage)

                yield chunk
        except asyncio.CancelledError:  # pragma: no cover
            logger.exception("Cancelled mid-stream")
            return
        except Exception as e:
            logger.exception(
                f"marking down {self._self_deployment.config.name}/{self._self_model.name} for error"
            )
            self._self_model.mark_down()
            raise DeploymentError("Error in wrapped stream") from e
