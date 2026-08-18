from __future__ import annotations

from collections.abc import AsyncIterator
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
from openai.lib.streaming.chat import AsyncChatCompletionStream
from openai.types.chat import (
    ChatCompletion,
    ChatCompletionChunk,
    ParsedChatCompletion,
)
from openai.types.completion_usage import CompletionUsage
from pydantic import BaseModel

from .deployment import ModelDeployment

_T = TypeVar("_T", bound=BaseModel)


class OpenAIDeployment(ModelDeployment):
    """A model deployment speaking the Chat Completions API."""

    # Assigned by Foundry.add, which shares one client per URL.
    client: AsyncOpenAI

    @property
    def url(self) -> str | None:
        if self.endpoint:
            return self.endpoint
        base = self.resource.base
        return f"{base}/openai/v1/" if base else None

    def new_client(self) -> AsyncOpenAI:
        return AsyncOpenAI(
            api_key=self.resource.api_key,
            base_url=self.url,
            timeout=self.resource.timeout,
        )

    @overload
    async def create(
        self, *, stream: Literal[True], **kwargs
    ) -> AsyncStream[ChatCompletionChunk]: ...

    @overload
    async def create(self, **kwargs) -> ChatCompletion: ...

    async def create(
        self, *, stream: bool = False, **kwargs
    ) -> ChatCompletion | AsyncStream[ChatCompletionChunk]:
        """
        Send a chat completion request to this deployment.
        Tracks usage metrics for load balancing.
        """

        # add input token estimate before we send the request so utilization is
        # kept up to date for other requests that might be executing concurrently.
        _preflight_estimate = self._estimate_token_usage(kwargs)
        self.spend_tokens(_preflight_estimate)
        self.spend_request()

        try:
            if stream:
                logger.trace("Creating streaming completion")
                response_stream = await self.client.chat.completions.create(
                    model=self.name,
                    stream=True,
                    stream_options=kwargs.pop(
                        "stream_options", {"include_usage": True}
                    ),
                    **kwargs,
                )

                return _AsyncStreamWrapper(
                    stream=response_stream,
                    model=self,
                    offset=_preflight_estimate,
                )

            logger.trace("Creating chat completion")
            response = cast(
                ChatCompletion,
                await self.client.chat.completions.create(model=self.name, **kwargs),
            )

            if response.usage:
                # dont double-count our preflight estimate
                self.spend_tokens(response.usage.total_tokens - _preflight_estimate)
                self._set_span_attributes(response.usage)

            return response

        except Exception as e:
            self._handle_error(e, "completion")
            raise

    async def open_stream(self, **kwargs) -> AsyncChatCompletionStream:
        """Enter the SDK's stream helper, which accumulates a terminal completion.

        Unlike create(stream=True) the caller may never iterate — awaiting
        get_final_completion() alone is enough — so usage is not tapped per
        chunk here. reconcile_stream charges it once the stream is done.
        """
        self.spend_tokens(self._estimate_token_usage(kwargs))
        self.spend_request()

        try:
            logger.trace("Opening completion stream")
            return await self.client.chat.completions.stream(
                model=self.name,
                # the snapshot carries no usage unless it is asked for
                stream_options=kwargs.pop("stream_options", {"include_usage": True}),
                **kwargs,
            ).__aenter__()
        except Exception as e:
            self._handle_error(e, "stream")
            raise

    def reconcile_stream(self, stream: AsyncChatCompletionStream, offset: int) -> None:
        """Charge what the stream actually accumulated against the estimate."""
        usage = stream.current_completion_snapshot.usage
        if usage:
            self.spend_tokens(usage.total_tokens - offset)
            self._set_span_attributes(usage)

    async def parse(
        self, *, response_format: type[_T], **kwargs
    ) -> ParsedChatCompletion[_T]:
        """
        Send a structured output parse request to this deployment.
        Tracks usage metrics for load balancing.
        """

        _preflight_estimate = self._estimate_token_usage(kwargs)
        self.spend_tokens(_preflight_estimate)
        self.spend_request()

        try:
            logger.trace("Creating parsed completion")
            response = await self.client.chat.completions.parse(
                model=self.name, response_format=response_format, **kwargs
            )

            if response.usage:
                self.spend_tokens(response.usage.total_tokens - _preflight_estimate)
                self._set_span_attributes(response.usage)

            return response
        except Exception as e:
            self._handle_error(e, "parse")
            raise

    def _handle_error(self, exc: Exception, op: str, log=logger) -> None:
        """Scope a markdown to what the error actually implicates.

        A 429 is one deployment's quota; a connection error is the whole
        resource. Timeouts during upstream-wide slowdowns are uncorrelated with
        which deployment was chosen, so they mark down nothing.
        """
        if isinstance(exc, RateLimitError):
            log.exception(f"Marking down model for rate limit on {op}")
            self.mark_down()
        elif isinstance(exc, APITimeoutError):
            log.warning(f"Upstream timeout on {op}; not marking down")
        elif isinstance(exc, APIConnectionError):
            log.exception(f"Marking down resource for connection error on {op}")
            self.resource.mark_down()

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
        model: OpenAIDeployment,
        offset: int = 0,
    ):
        super().__init__(stream)
        self._self_model: OpenAIDeployment = model
        self._self_offset: int = offset
        # Chunks are consumed after create() returns, once its contextualize
        # scope is gone, so bind the context for mid-stream error logs.
        self._self_logger = logger.bind(
            resource=model.resource.name,
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
                    self._self_model._set_span_attributes(chunk.usage)

                yield chunk
        except Exception as e:
            self._self_model._handle_error(e, "stream", log=self._self_logger)
            raise
