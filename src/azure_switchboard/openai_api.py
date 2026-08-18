from __future__ import annotations

from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Literal, TypeVar, cast, overload

import wrapt
from loguru import logger
from openai import (
    APIConnectionError,
    APITimeoutError,
    AsyncOpenAI,
    AsyncStream,
    RateLimitError,
)
from openai.types.chat import (
    ChatCompletion,
    ChatCompletionChunk,
    ParsedChatCompletion,
)
from openai.types.completion_usage import CompletionUsage
from pydantic import BaseModel

from .deployment import DeploymentBase
from .model import Model

_T = TypeVar("_T", bound=BaseModel)


@dataclass
class OpenAIConfig:
    """Configuration for an Azure OpenAI or OpenAI deployment."""

    name: str
    base_url: str | None = None
    api_key: str | None = None
    # 30s suits interactive workloads; override for long-running batch jobs
    timeout: float = 30.0
    models: list[Model] = field(default_factory=list)

    def get_client(self) -> AsyncOpenAI:
        return AsyncOpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
            timeout=self.timeout,
        )


class OpenAIDeployment(DeploymentBase):
    """Runtime state of a deployment speaking the Chat Completions API"""

    def __init__(self, config: OpenAIConfig) -> None:
        super().__init__(config.name, config.models)
        self.config = config
        self.client = config.get_client()

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

        self._check_model(model)

        # add input token estimate before we send the request so utilization is
        # kept up to date for other requests that might be executing concurrently.
        _preflight_estimate = self._estimate_token_usage(kwargs)
        self.models[model].spend_tokens(_preflight_estimate)
        self.models[model].spend_request()

        try:
            if stream:
                logger.trace("Creating streaming completion")
                response_stream = await self.client.chat.completions.create(
                    model=model,
                    stream=True,
                    stream_options=kwargs.pop(
                        "stream_options", {"include_usage": True}
                    ),
                    **kwargs,
                )

                return _AsyncStreamWrapper(
                    stream=response_stream,
                    deployment=self,
                    model=self.models[model],
                    offset=_preflight_estimate,
                )

            logger.trace("Creating chat completion")
            response = cast(
                ChatCompletion,
                await self.client.chat.completions.create(
                    model=model,
                    **kwargs,
                ),
            )

            if response.usage:
                # dont double-count our preflight estimate
                self.models[model].spend_tokens(
                    response.usage.total_tokens - _preflight_estimate
                )
                self._set_span_attributes(response.usage)

            return response

        except RateLimitError:
            logger.exception("Marking down model for rate limit")
            self.models[model].mark_down()
            raise
        except APITimeoutError:
            # Timeouts during upstream-wide slowdowns are uncorrelated with
            # which deployment was chosen — marking it down wastes capacity.
            logger.warning("Upstream timeout on completion; not marking down")
            raise
        except APIConnectionError:
            logger.exception("Marking down model for connection error")
            self.models[model].mark_down()
            raise

    async def parse(
        self,
        *,
        model: str,
        response_format: type[_T],
        **kwargs,
    ) -> ParsedChatCompletion[_T]:
        """
        Send a structured output parse request to this client.
        Tracks usage metrics for load balancing.
        """

        self._check_model(model)

        _preflight_estimate = self._estimate_token_usage(kwargs)
        self.models[model].spend_tokens(_preflight_estimate)
        self.models[model].spend_request()

        try:
            logger.trace("Creating parsed completion")
            response = await self.client.chat.completions.parse(
                model=model, response_format=response_format, **kwargs
            )

            if response.usage:
                self.models[model].spend_tokens(
                    response.usage.total_tokens - _preflight_estimate
                )
                self._set_span_attributes(response.usage)

            return response
        except RateLimitError:
            logger.exception("Marking down model for rate limit on parse")
            self.models[model].mark_down()
            raise
        except APITimeoutError:
            logger.warning("Upstream timeout on parse; not marking down")
            raise
        except APIConnectionError:
            logger.exception("Marking down model for connection error on parse")
            self.models[model].mark_down()
            raise

    def _estimate_token_usage(self, kwargs: dict) -> int:
        # ~4 chars per token, input only.
        t_input = sum(len(m.get("content", "")) for m in kwargs.get("messages", []))
        return t_input // 4

    def _set_span_attributes(self, usage: CompletionUsage) -> None:
        prompt = usage.prompt_tokens_details
        completion = usage.completion_tokens_details
        self._record_token_details(
            cached=prompt.cached_tokens if prompt else None,
            reasoning=completion.reasoning_tokens if completion else None,
        )


class _AsyncStreamWrapper(wrapt.ObjectProxy):
    """Wrap an openai.AsyncStream to track usage"""

    def __init__(
        self,
        stream: AsyncStream[ChatCompletionChunk],
        deployment: OpenAIDeployment,
        model: Model,
        offset: int = 0,
    ):
        super().__init__(stream)
        self._self_deployment: OpenAIDeployment = deployment
        self._self_model: Model = model
        self._self_offset: int = offset
        # Chunks are consumed after create() returns, once its contextualize
        # scope is gone, so bind the context for mid-stream error logs.
        self._self_logger = logger.bind(
            deployment=deployment.name,
            model=model.name,
        )

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
        except RateLimitError:
            self._self_logger.exception("Marking down model for rate limit on stream")
            self._self_model.mark_down()
            raise
        except APITimeoutError:
            self._self_logger.warning("Upstream timeout on stream; not marking down")
            raise
        except APIConnectionError:
            self._self_logger.exception(
                "Marking down model for connection error on stream"
            )
            self._self_model.mark_down()
            raise
